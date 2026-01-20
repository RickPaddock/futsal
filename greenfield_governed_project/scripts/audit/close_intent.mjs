/*
PROV: GREENFIELD.GOV.CLOSE_INTENT.01
REQ: SYS-ARCH-15, AUD-REQ-10
WHY: Close an intent by applying status updates to canonical sources after a successful audit.
*/

import fs from "node:fs";
import path from "node:path";

import { repoRootFromHere, relPosix } from "../lib/paths.mjs";

function readJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function writeJson(p, obj) {
  fs.writeFileSync(p, JSON.stringify(obj, null, 2) + "\n", "utf8");
}

function parseArgs(argv) {
  const out = { intentId: "", closedDate: "", apply: false };
  const args = [...argv];
  while (args.length) {
    const a = args.shift();
    if (a === "--intent-id") out.intentId = String(args.shift() || "");
    else if (a === "--closed-date") out.closedDate = String(args.shift() || "");
    else if (a === "--apply") out.apply = true;
  }
  return out;
}

function scanRepoReqTags(repoRoot) {
  const exts = new Set([".js", ".mjs", ".ts", ".tsx", ".py", ".sh"]);
  const skip = new Set([
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

  function iterFiles(root, predicate) {
    const out = [];
    for (const p of fs.readdirSync(root, { withFileTypes: true })) {
      const abs = path.join(root, p.name);
      if (p.isDirectory()) {
        if (skip.has(p.name)) continue;
        out.push(...iterFiles(abs, predicate));
      } else if (p.isFile() && predicate(abs)) {
        out.push(abs);
      }
    }
    return out;
  }

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

function main() {
  const repoRoot = repoRootFromHere(import.meta.url);
  const args = parseArgs(process.argv.slice(2));
  const intentId = String(args.intentId || "").trim();
  if (!intentId) {
    process.stderr.write("Usage: node scripts/audit/close_intent.mjs --intent-id INT-001 --closed-date YYYY-MM-DD [--apply]\n");
    process.exit(2);
  }
  if (!args.closedDate || !/^\d{4}-\d{2}-\d{2}$/.test(args.closedDate)) {
    process.stderr.write("Missing/invalid --closed-date (expected YYYY-MM-DD)\n");
    process.exit(2);
  }

  const project = readJson(path.join(repoRoot, "spec", "project.json"));
  const governance = project.governance || {};
  const newReqPrefixes = Array.isArray(governance.new_requirement_id_prefixes) ? governance.new_requirement_id_prefixes.map(String) : ["REQ-"];
  const requirementsSourceRel = String(project.requirements_source || "");

  const intentPath = path.join(repoRoot, "spec", "intents", `${intentId}.json`);
  if (!fs.existsSync(intentPath)) throw new Error(`missing_intent:${relPosix(path.relative(repoRoot, intentPath))}`);
  const intent = readJson(intentPath);
  const currentStatus = String(intent.status || "").trim();
  if (currentStatus === "draft") {
    process.stderr.write(`[close:error] intent is draft; set to todo before closing: ${intentId}\n`);
    process.exit(2);
  }
  if (currentStatus === "closed" && args.apply) {
    process.stderr.write(`[close:error] intent already closed (refusing to re-apply): ${intentId}\n`);
    process.exit(2);
  }

  const requirementsIndexPath = path.join(repoRoot, requirementsSourceRel);
  const index = readJson(requirementsIndexPath);
  const areaFilesRel =
    index?.type === "requirements_index" && Array.isArray(index.files)
      ? index.files.map(String)
      : [requirementsSourceRel];

  const areaFilesAbs = areaFilesRel.map((rel) => path.join(repoRoot, rel));
  const areas = areaFilesAbs.map((abs) => ({ abs, rel: relPosix(path.relative(repoRoot, abs)), data: readJson(abs) }));
  const reqLocations = new Map();
  for (const area of areas) {
    for (const r of area.data.requirements || []) {
      const id = String(r.id || "").trim();
      if (!id) continue;
      reqLocations.set(id, area);
    }
  }

  const tasksDir = path.join(repoRoot, "spec", "tasks");
  const plannedTasks = Array.isArray(intent.task_ids_planned) ? intent.task_ids_planned.map(String) : [];
  const newReqIds = new Set();
  for (const tid of plannedTasks) {
    const taskPath = path.join(tasksDir, `${tid}.json`);
    if (!fs.existsSync(taskPath)) throw new Error(`missing_task_spec:${relPosix(path.relative(repoRoot, taskPath))}`);
    const task = readJson(taskPath);
    for (const st of Array.isArray(task.subtasks) ? task.subtasks : []) {
      const sid = String(st.subtask_id || "").trim();
      if (newReqPrefixes.some((pfx) => sid.startsWith(pfx))) newReqIds.add(sid);
      for (const r of Array.isArray(st.requirements_to_add) ? st.requirements_to_add : []) {
        const rid = String(r.id || "").trim();
        if (newReqPrefixes.some((pfx) => rid.startsWith(pfx))) newReqIds.add(rid);
      }
    }
  }

  const reqIdsInCode = scanRepoReqTags(repoRoot);
  const missingCodeRefs = [...newReqIds].filter((id) => !reqIdsInCode.has(id));
  if (missingCodeRefs.length) {
    process.stderr.write(`[close:error] new requirements missing code REQ: references: ${missingCodeRefs.join(", ")}\n`);
    process.exit(2);
  }

  if (args.apply) {
    for (const id of [...newReqIds].sort()) {
      const loc = reqLocations.get(id);
      if (!loc) throw new Error(`new_requirement_missing_in_requirements:${id}`);
      const req = (loc.data.requirements || []).find((x) => String(x.id || "").trim() === id);
      if (!req) throw new Error(`new_requirement_missing_in_requirements:${id}`);
      req.tracking = req.tracking || {};
      req.tracking.implementation = "done";
      writeJson(loc.abs, loc.data);
    }

    intent.status = "closed";
    intent.closed_date = args.closedDate;
    writeJson(intentPath, intent);
    process.stdout.write(`[close] updated ${relPosix(path.relative(repoRoot, intentPath))}\n`);
  }

  process.stdout.write("[close] ok\n");
}

try {
  main();
} catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
  process.stderr.write(`[close:error] ${msg}\n`);
  process.exit(2);
}
