/*
PROV: GREENFIELD.GOV.CLOSE_INTENT.01
REQ: SYS-ARCH-15, AUD-REQ-10, GREENFIELD-GOV-015, GREENFIELD-GOV-017, GREENFIELD-GOV-018, GREENFIELD-GOV-021
WHY: Close an intent by applying status updates to canonical sources after a successful audit (including runbook navigation hygiene and explicit path scope).
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
  const out = { intentId: "", runId: "", closedDate: "", requireQualityAudit: false, apply: false };
  const args = [...argv];
  while (args.length) {
    const a = args.shift();
    if (a === "--intent-id") out.intentId = String(args.shift() || "");
    else if (a === "--run-id") out.runId = String(args.shift() || "");
    else if (a === "--closed-date") out.closedDate = String(args.shift() || "");
    else if (a === "--require-quality-audit") out.requireQualityAudit = true;
    else if (a === "--apply") out.apply = true;
  }
  return out;
}

function relFrom(repoRoot, abs) {
  return relPosix(path.relative(repoRoot, abs));
}

function requireFile(repoRoot, abs, label) {
  if (!fs.existsSync(abs) || fs.statSync(abs).isDirectory()) {
    throw new Error(`missing_${label}:${relFrom(repoRoot, abs)}`);
  }
}

function requireEvidenceRunOk(repoRoot, abs) {
  requireFile(repoRoot, abs, "evidence_run_json");
  const obj = readJson(abs);
  const exitCode = Number(obj?.exit_code ?? 1);
  if (exitCode !== 0) throw new Error(`evidence_nonzero_exit:${relFrom(repoRoot, abs)}:${exitCode}`);
  return obj;
}

function requireEvidenceCommandIncludes(repoRoot, abs, needle) {
  const obj = readJson(abs);
  const cmd = String(obj?.command || "");
  if (!cmd.includes(needle)) {
    throw new Error(`evidence_command_mismatch:${relFrom(repoRoot, abs)}:expected_includes:${needle}`);
  }
}

function requireAuditReportOk(repoRoot, abs) {
  requireFile(repoRoot, abs, "audit_report_json");
  const obj = readJson(abs);
  const errors = Array.isArray(obj?.errors) ? obj.errors : null;
  if (!errors) throw new Error(`audit_report_missing_errors_array:${relFrom(repoRoot, abs)}`);
  if (errors.length) throw new Error(`audit_report_has_errors:${relFrom(repoRoot, abs)}:${errors.length}`);
  return obj;
}

function requireIntentQualityAuditOk(repoRoot, abs, { intentId, runId }) {
  requireFile(repoRoot, abs, "quality_audit_json");
  const obj = readJson(abs);
  if (String(obj?.type || "").trim() !== "intent_quality_audit_report") {
    throw new Error(`invalid_quality_audit_type:${relFrom(repoRoot, abs)}`);
  }
  if (Number(obj?.schema_version ?? 0) !== 1) {
    throw new Error(`invalid_quality_audit_schema_version:${relFrom(repoRoot, abs)}`);
  }
  if (String(obj?.intent_id || "").trim() !== intentId) {
    throw new Error(`quality_audit_intent_mismatch:${relFrom(repoRoot, abs)}`);
  }
  if (String(obj?.run_id || "").trim() !== runId) {
    throw new Error(`quality_audit_run_mismatch:${relFrom(repoRoot, abs)}`);
  }
  const improvements = Array.isArray(obj?.improvements) ? obj.improvements : null;
  if (!improvements) throw new Error(`quality_audit_missing_improvements_array:${relFrom(repoRoot, abs)}`);
  if (improvements.length !== 10) {
    throw new Error(`quality_audit_improvements_not_10:${relFrom(repoRoot, abs)}:${improvements.length}`);
  }
  return obj;
}

function requireTaskQualityAuditPass(repoRoot, abs, { intentId, runId, taskId }) {
  requireFile(repoRoot, abs, "task_quality_audit_json");
  const rep = readJson(abs);
  if (String(rep?.type || "").trim() !== "task_quality_audit_report") {
    throw new Error(`invalid_task_quality_audit_type:${relFrom(repoRoot, abs)}`);
  }
  if (Number(rep?.schema_version ?? 0) !== 1) {
    throw new Error(`invalid_task_quality_audit_schema_version:${relFrom(repoRoot, abs)}`);
  }
  if (String(rep?.intent_id || "").trim() !== intentId) {
    throw new Error(`task_quality_audit_intent_mismatch:${relFrom(repoRoot, abs)}`);
  }
  if (String(rep?.run_id || "").trim() !== runId) {
    throw new Error(`task_quality_audit_run_mismatch:${relFrom(repoRoot, abs)}`);
  }
  if (String(rep?.task_id || "").trim() !== taskId) {
    throw new Error(`task_quality_audit_task_mismatch:${relFrom(repoRoot, abs)}`);
  }

  const gate = rep?.gate;
  const gateStatus = String(gate?.status || "").trim();
  const gateBlockers = Array.isArray(gate?.blockers) ? gate.blockers : null;
  if (!gateBlockers) {
    throw new Error(`task_quality_audit_missing_gate_blockers_array:${relFrom(repoRoot, abs)}`);
  }
  if (gateStatus !== "pass") {
    throw new Error(`task_quality_audit_not_pass:${relFrom(repoRoot, abs)}:${gateStatus || "missing"}`);
  }
  if (gateBlockers.length) {
    throw new Error(`task_quality_audit_has_blockers:${relFrom(repoRoot, abs)}:${gateBlockers.length}`);
  }

  const functional = rep?.functional;
  const functionalStatus = String(functional?.status || "").trim();
  if (!functionalStatus) throw new Error(`task_quality_audit_missing_functional_status:${relFrom(repoRoot, abs)}`);
  if (functionalStatus !== "pass") throw new Error(`task_quality_audit_functional_not_pass:${relFrom(repoRoot, abs)}:${functionalStatus}`);

  const nonfunctional = rep?.nonfunctional;
  if (!nonfunctional || typeof nonfunctional !== "object") {
    throw new Error(`task_quality_audit_missing_nonfunctional_section:${relFrom(repoRoot, abs)}`);
  }
  const requiredNfr = ["correctness_safety", "performance", "security", "maintainability"];
  for (const key of requiredNfr) {
    const status = String(nonfunctional?.[key]?.status || "").trim();
    if (!status) throw new Error(`task_quality_audit_missing_nonfunctional_status:${relFrom(repoRoot, abs)}:${key}`);
    if (status !== "pass") throw new Error(`task_quality_audit_nonfunctional_not_pass:${relFrom(repoRoot, abs)}:${key}:${status}`);
  }
  const nonfunctionalOverall = String(nonfunctional?.overall_status || "").trim();
  if (!nonfunctionalOverall) throw new Error(`task_quality_audit_missing_nonfunctional_overall_status:${relFrom(repoRoot, abs)}`);
  if (nonfunctionalOverall !== "pass") {
    throw new Error(`task_quality_audit_nonfunctional_overall_not_pass:${relFrom(repoRoot, abs)}:${nonfunctionalOverall}`);
  }

  return rep;
}

function requireRunbooksDecisionOk(repoRoot, intentId, intent) {
  const runbooks = intent?.runbooks;
  if (!runbooks || typeof runbooks !== "object") {
    throw new Error(`intent_missing_runbooks:${intentId}`);
  }
  const decision = String(runbooks.decision || "").trim();
  if (!["none", "create", "update"].includes(decision)) {
    throw new Error(`intent_invalid_runbooks_decision:${intentId}:${decision || "missing"}`);
  }
  const notes = String(runbooks.notes || "").trim();
  if (!notes) throw new Error(`intent_runbooks_notes_required:${intentId}`);
  const paths = Array.isArray(runbooks.paths_mdt) ? runbooks.paths_mdt.map(String).map((s) => s.trim()).filter(Boolean) : null;
  if (!paths) throw new Error(`intent_runbooks_paths_required:${intentId}`);
  if ((decision === "create" || decision === "update") && paths.length === 0) {
    throw new Error(`intent_runbooks_paths_empty_for_decision:${intentId}:${decision}`);
  }
  for (const rel of paths || []) {
    if (!rel.startsWith("spec/md/docs/runbooks/") || !rel.endsWith(".mdt")) {
      throw new Error(`intent_runbooks_path_invalid_scope:${intentId}:${rel}`);
    }
    const abs = path.join(repoRoot, rel);
    if (!fs.existsSync(abs) || fs.statSync(abs).isDirectory()) {
      throw new Error(`intent_runbooks_template_missing:${intentId}:${rel}`);
    }
  }
}

function requirePathScopeOk(intentId, intent) {
  const status = String(intent?.status || "").trim();
  if (status === "draft") return;
  const allowed = Array.isArray(intent?.paths_allowed) ? intent.paths_allowed.map(String).map((s) => s.trim()).filter(Boolean) : null;
  const excluded = Array.isArray(intent?.paths_excluded) ? intent.paths_excluded.map(String).map((s) => s.trim()).filter(Boolean) : null;
  if (!allowed) throw new Error(`intent_paths_allowed_required:${intentId}`);
  if (!excluded) throw new Error(`intent_paths_excluded_required:${intentId}`);
  if (!allowed.length) throw new Error(`intent_paths_allowed_empty:${intentId}`);
  if (!excluded.length) throw new Error(`intent_paths_excluded_empty:${intentId}`);
  const requiredExcluded = ["docs/", "status/intents/", "status/portal/"];
  const missing = requiredExcluded.filter((pfx) => !excluded.some((x) => x === pfx || x.startsWith(pfx)));
  if (missing.length) throw new Error(`intent_paths_excluded_missing_generated:${intentId}:${missing.join(",")}`);
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
    process.stderr.write(
      "Usage: node scripts/audit/close_intent.mjs --intent-id INT-001 --closed-date YYYY-MM-DD [--run-id YYYYMMDD_HHMMSS] [--require-quality-audit] [--apply]\n",
    );
    process.exit(2);
  }
  if (!args.closedDate || !/^\d{4}-\d{2}-\d{2}$/.test(args.closedDate)) {
    process.stderr.write("Missing/invalid --closed-date (expected YYYY-MM-DD)\n");
    process.exit(2);
  }
  if (args.apply && (!args.runId || args.runId.includes("<") || args.runId.includes(">"))) {
    process.stderr.write("Missing/invalid --run-id (expected YYYYMMDD_HHMMSS)\n");
    process.exit(2);
  }
  if (args.requireQualityAudit && (!args.runId || args.runId.includes("<") || args.runId.includes(">"))) {
    process.stderr.write("Missing/invalid --run-id (required when using --require-quality-audit)\n");
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

  requireRunbooksDecisionOk(repoRoot, intentId, intent);
  requirePathScopeOk(intentId, intent);

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

  if (args.runId) {
    const runRoot = path.join(repoRoot, "status", "audit", intentId, "runs", args.runId);
    const auditRun = path.join(runRoot, "audit", "run.json");
    const auditReport = path.join(runRoot, "audit", "audit_report.json");
    const guardrailsRun = path.join(runRoot, "guardrails", "run.json");
    const generateRun = path.join(runRoot, "generate", "run.json");
    const qualityReport = path.join(runRoot, "quality_audit.json");

    requireEvidenceRunOk(repoRoot, auditRun);
    requireEvidenceCommandIncludes(repoRoot, auditRun, "npm run audit:intent");
    requireAuditReportOk(repoRoot, auditReport);
    requireEvidenceRunOk(repoRoot, guardrailsRun);
    requireEvidenceCommandIncludes(repoRoot, guardrailsRun, "npm run guardrails");
    if (args.requireQualityAudit) {
      // Ensure derived surfaces were refreshed for the audited run.
      requireEvidenceRunOk(repoRoot, generateRun);
      requireEvidenceCommandIncludes(repoRoot, generateRun, "npm run generate");
      requireIntentQualityAuditOk(repoRoot, qualityReport, { intentId, runId: args.runId });
      for (const tid of plannedTasks) {
        const taskId = String(tid || "").trim();
        if (!taskId) continue;
        const taskQualityPath = path.join(runRoot, "tasks", taskId, "quality_audit.json");
        requireTaskQualityAuditPass(repoRoot, taskQualityPath, { intentId, runId: args.runId, taskId });
      }
    }
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
