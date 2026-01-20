/*
PROV: GREENFIELD.GOV.AUDIT_INTENT.01
REQ: SYS-ARCH-15, AUD-REQ-10
WHY: Audit an intent for governance completeness (task specs, new requirements tracked, and REQ tags in code).
*/

import fs from "node:fs";
import path from "node:path";

import { repoRootFromHere, relPosix } from "../lib/paths.mjs";

const SKIP_DIR_NAMES = new Set([
  ".git",
  ".next",
  ".cache",
  ".turbo",
  ".vercel",
  ".idea",
  ".vscode",
  "node_modules",
  "dist",
  "build",
  "coverage",
  "tmp",
  "logs",
  ".dart_tool",
  ".gradle",
  "Pods",
]);

function readJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function writeJson(p, obj) {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, JSON.stringify(obj, null, 2) + "\n", "utf8");
}

function iterFiles(root, predicate) {
  const out = [];
  for (const p of fs.readdirSync(root, { withFileTypes: true })) {
    const abs = path.join(root, p.name);
    if (p.isDirectory()) {
      if (SKIP_DIR_NAMES.has(p.name)) continue;
      out.push(...iterFiles(abs, predicate));
    } else if (p.isFile() && predicate(abs)) {
      out.push(abs);
    }
  }
  return out;
}

function scanRepoReqTags(repoRoot) {
  const exts = new Set([".js", ".mjs", ".ts", ".tsx", ".py", ".sh"]);
  const files = iterFiles(repoRoot, (p) => exts.has(path.extname(p)));
  const ids = new Set();

  for (const abs of files) {
    const text = fs.readFileSync(abs, "utf8");
    const match = text.match(/^\s*(?:#|\/\/|\/\*|\*)?\s*REQ:\s*(.+?)\s*$/gm);
    if (!match) continue;
    for (const line of match) {
      const raw = line.replace(/^\s*(?:#|\/\/|\/\*|\*)?\s*REQ:\s*/i, "").trim();
      for (const tok of raw.split(",")) {
        const id = tok.trim();
        if (id) ids.add(id);
      }
    }
  }

  return ids;
}

function loadRequirementsBundle({ repoRoot, requirementsSourceRel }) {
  const abs = path.join(repoRoot, requirementsSourceRel);
  const root = readJson(abs);

  if (root?.type === "requirements_index") {
    const files = Array.isArray(root.files) ? root.files.map(String) : [];
    if (!files.length) throw new Error(`requirements_index_empty:${requirementsSourceRel}`);
    const requirements = [];
    for (const rel of files) {
      const area = readJson(path.join(repoRoot, rel));
      for (const r of area.requirements || []) requirements.push(r);
    }
    return { requirements, areaFilesRel: files };
  }

  if (Array.isArray(root?.requirements)) {
    return { requirements: root.requirements, areaFilesRel: [requirementsSourceRel] };
  }

  throw new Error(`requirements_source_invalid:${requirementsSourceRel}`);
}

function parseArgs(argv) {
  const out = { intentId: "", outPath: "" };
  const args = [...argv];
  while (args.length) {
    const a = args.shift();
    if (a === "--intent-id") out.intentId = String(args.shift() || "");
    else if (a === "--out") out.outPath = String(args.shift() || "");
  }
  return out;
}

function utcNow() {
  return new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
}

function main() {
  const repoRoot = repoRootFromHere(import.meta.url);
  const args = parseArgs(process.argv.slice(2));
  const intentId = String(args.intentId || "").trim();
  if (!intentId) {
    process.stderr.write("Usage: node scripts/audit/audit_intent.mjs --intent-id INT-001 [--out status/audit/INT-001/runs/<id>/audit_report.json]\n");
    process.exit(2);
  }

  const project = readJson(path.join(repoRoot, "spec", "project.json"));
  const governance = project.governance || {};
  const newReqPrefixes = Array.isArray(governance.new_requirement_id_prefixes) ? governance.new_requirement_id_prefixes.map(String) : ["REQ-"];

  const intentPath = path.join(repoRoot, "spec", "intents", `${intentId}.json`);
  if (!fs.existsSync(intentPath)) {
    process.stderr.write(`[audit:error] missing intent: ${relPosix(path.relative(repoRoot, intentPath))}\n`);
    process.exit(2);
  }
  const intent = readJson(intentPath);
  const intentStatus = String(intent.status || "").trim();
  const expectedNewReqImpl = intentStatus === "closed" ? "done" : "todo";

  const requirementsSourceRel = String(project.requirements_source || "");
  const requirementsBundle = loadRequirementsBundle({ repoRoot, requirementsSourceRel });
  const requirementsById = new Map((requirementsBundle.requirements || []).map((r) => [String(r.id || "").trim(), r]));

  const tasksDir = path.join(repoRoot, "spec", "tasks");
  const plannedTasks = Array.isArray(intent.task_ids_planned) ? intent.task_ids_planned.map(String) : [];
  const missingTaskSpecs = [];
  const newReqIds = new Set();

  for (const tid of plannedTasks) {
    const taskPath = path.join(tasksDir, `${tid}.json`);
    if (!fs.existsSync(taskPath)) {
      missingTaskSpecs.push(relPosix(path.relative(repoRoot, taskPath)));
      continue;
    }
    const task = readJson(taskPath);
    const subtasks = Array.isArray(task.subtasks) ? task.subtasks : [];
    for (const st of subtasks) {
      const sid = String(st.subtask_id || "").trim();
      if (!sid) continue;
      if (newReqPrefixes.some((pfx) => sid.startsWith(pfx))) newReqIds.add(sid);
      for (const r of Array.isArray(st.requirements_to_add) ? st.requirements_to_add : []) {
        const rid = String(r.id || "").trim();
        if (!rid) continue;
        if (newReqPrefixes.some((pfx) => rid.startsWith(pfx))) newReqIds.add(rid);
      }
    }
  }

  const missingNewReqs = [];
  const newReqImplMismatches = [];
  for (const rid of [...newReqIds].sort()) {
    const r = requirementsById.get(rid);
    if (!r) {
      missingNewReqs.push(rid);
      continue;
    }
    const impl = String(r.tracking?.implementation || "todo");
    if (impl !== expectedNewReqImpl) {
      newReqImplMismatches.push({ id: rid, expected: expectedNewReqImpl, got: impl });
    }
  }

  const reqIdsInCode = scanRepoReqTags(repoRoot);
  const newReqsWithCodeRefs = [...newReqIds].filter((rid) => reqIdsInCode.has(rid)).sort();
  const newReqsMissingCodeRefs = [...newReqIds].filter((rid) => !reqIdsInCode.has(rid)).sort();

  const errors = [];
  if (missingTaskSpecs.length) errors.push({ code: "missing_task_specs", items: missingTaskSpecs });
  if (missingNewReqs.length) errors.push({ code: "missing_new_requirements", items: missingNewReqs });
  if (newReqImplMismatches.length) errors.push({ code: "new_requirement_tracking_mismatch", items: newReqImplMismatches });

  const report = {
    type: "intent_audit_report",
    schema_version: 1,
    intent_id: intentId,
    intent_status: intentStatus,
    timestamp: utcNow(),
    errors,
    summary: {
      planned_tasks: plannedTasks.length,
      missing_task_specs: missingTaskSpecs.length,
      new_requirements_declared: newReqIds.size,
      new_requirements_with_code_refs: newReqsWithCodeRefs.length,
      new_requirements_missing_code_refs: newReqsMissingCodeRefs.length,
    },
    new_requirements: {
      declared: [...newReqIds].sort(),
      with_code_refs: newReqsWithCodeRefs,
      missing_code_refs: newReqsMissingCodeRefs,
    },
  };

  if (args.outPath) {
    const outAbs = path.isAbsolute(args.outPath) ? args.outPath : path.join(repoRoot, args.outPath);
    writeJson(outAbs, report);
    process.stdout.write(`[audit] wrote ${relPosix(path.relative(repoRoot, outAbs))}\n`);
  } else {
    process.stdout.write(JSON.stringify(report, null, 2) + "\n");
  }

  if (errors.length) process.exit(2);
  process.stdout.write("[audit] ok\n");
}

try {
  main();
} catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
  process.stderr.write(`[audit:error] ${msg}\n`);
  process.exit(2);
}
