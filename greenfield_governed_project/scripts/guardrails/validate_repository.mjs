/*
PROV: GREENFIELD.SCAFFOLD.GUARDRAILS.01
REQ: AUD-REQ-10, SYS-ARCH-15, GREENFIELD-GOV-018, GREENFIELD-GOV-019
WHY: Enforce governance invariants (generated markdown only, deterministic outputs, evidence format, runbook navigation hygiene) with inline error recovery guidance.
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

function iterFiles(root, predicate) {
  const out = [];
  for (const p of fs.readdirSync(root, { withFileTypes: true })) {
    const abs = path.join(root, p.name);
    if (p.isDirectory()) {
      if (SKIP_DIR_NAMES.has(p.name)) continue;
      out.push(...iterFiles(abs, predicate));
    }
    else if (p.isFile() && predicate(abs)) out.push(abs);
  }
  return out;
}

function globToRegExp(glob) {
  const escaped = glob.replace(/[.+^${}()|[\]\\]/g, "\\$&");
  const re = escaped
    .replace(/\\\*\\\*\\\//g, "(?:.*\\/)?")
    .replace(/\\\*\\\*/g, ".*")
    .replace(/\\\*/g, "[^/]*");
  return new RegExp(`^${re}$`);
}

function hasGeneratedFrontmatter(text) {
  if (!text.startsWith("---\n")) return false;
  const end = text.indexOf("\n---\n", 4);
  if (end === -1) return false;
  const fm = text.slice(0, end + "\n---\n".length);
  return /\ngenerated:\s*true\s*\n/.test(fm);
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
    return { requirements, sourcesRel: [requirementsSourceRel, ...files] };
  }

  if (Array.isArray(root?.requirements)) {
    return { requirements: root.requirements, sourcesRel: [requirementsSourceRel] };
  }

  throw new Error(`requirements_source_invalid:${requirementsSourceRel}`);
}

function normalizeReqIdList(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value.map(String).map((s) => s.trim()).filter(Boolean);
  return [];
}

function normalizeStringList(value) {
  return normalizeReqIdList(value);
}

function isIsoDate(value) {
  return typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value);
}

function looksLikeFilePath(rel) {
  const base = path.posix.basename(String(rel || "").replace(/\\/g, "/"));
  if (!base) return false;
  if (base.endsWith("/")) return false;
  if (base.includes(".")) return true;
  return ["Makefile", "Dockerfile", "LICENSE", "NOTICE"].includes(base);
}

function validateIntentRunbooks({ repoRoot, intentId, intent }) {
  const status = String(intent?.status || "").trim();
  if (status === "draft") return [];

  const errors = [];
  const runbooks = intent?.runbooks;
  if (!runbooks || typeof runbooks !== "object") {
    errors.push(`intent_missing_runbooks:${intentId} → Fix: Add runbooks section to spec/intents/${intentId}.json: {"decision": "none", "notes": "...", "paths_mdt": []}`);
    return errors;
  }

  const decision = String(runbooks.decision || "").trim();
  const allowed = new Set(["none", "create", "update"]);
  if (!allowed.has(decision)) {
    errors.push(`intent_invalid_runbooks_decision:${intentId}:${decision || "missing"} → Fix: Set runbooks.decision to one of: "none", "create", "update"`);
  }

  const notes = String(runbooks.notes || "").trim();
  if (!notes) errors.push(`intent_runbooks_notes_required:${intentId} → Fix: Add runbooks.notes with rationale for decision`);

  const paths = Array.isArray(runbooks.paths_mdt) ? runbooks.paths_mdt.map(String).map((s) => s.trim()).filter(Boolean) : null;
  if (!paths) {
    errors.push(`intent_runbooks_paths_required:${intentId} → Fix: Add runbooks.paths_mdt array (can be empty for decision="none")`);
    return errors;
  }

  if ((decision === "create" || decision === "update") && paths.length === 0) {
    errors.push(`intent_runbooks_paths_empty_for_decision:${intentId}:${decision} → Fix: Add template paths to runbooks.paths_mdt (e.g., ["spec/md/docs/runbooks/my-topic.mdt"])`);
  }

  for (const rel of paths) {
    if (!looksLikeFilePath(rel)) {
      errors.push(`intent_runbooks_path_not_file:${intentId}:${rel}`);
      continue;
    }
    if (!rel.startsWith("spec/md/docs/runbooks/") || !rel.endsWith(".mdt")) {
      errors.push(`intent_runbooks_path_invalid_scope:${intentId}:${rel}`);
      continue;
    }
    const abs = path.join(repoRoot, rel);
    if (!fs.existsSync(abs) || fs.statSync(abs).isDirectory()) {
      errors.push(`intent_runbooks_template_missing:${intentId}:${rel}`);
      continue;
    }
    const outRel = rel.replace(/^spec\/md\/docs\/runbooks\//, "docs/runbooks/").replace(/\.mdt$/, ".md");
    const outAbs = path.join(repoRoot, outRel);
    if (!fs.existsSync(outAbs) || fs.statSync(outAbs).isDirectory()) {
      errors.push(`intent_runbooks_generated_missing:${intentId}:${outRel}`);
      continue;
    }
  }

  return errors;
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

function validateGovernanceSpec(repoRoot) {
  const project = readJson(path.join(repoRoot, "spec", "project.json"));
  const requirementsSourceRel = String(project.requirements_source || "");
  if (!requirementsSourceRel) throw new Error("project_missing_requirements_source");

  const governance = project.governance || {};
  const newReqPrefixes = new Set(normalizeReqIdList(governance.new_requirement_id_prefixes || ["REQ-"]));
  const sizeLimits = governance.file_size_limits_bytes || {};

  const requirementsBundle = loadRequirementsBundle({ repoRoot, requirementsSourceRel });
  const requirements = requirementsBundle.requirements || [];
  const requirementById = new Map();
  for (const r of requirements) {
    const id = String(r.id || "").trim();
    if (!id) throw new Error("requirements_missing_id");
    if (requirementById.has(id)) throw new Error(`requirements_duplicate_id:${id}`);
    const guardrails = Array.isArray(r.guardrails) ? r.guardrails.map(String).map((s) => s.trim()).filter(Boolean) : [];
    if (!guardrails.length) throw new Error(`requirements_missing_guardrails:${id}`);
    const isNewReq = [...newReqPrefixes].some((pfx) => id.startsWith(pfx));
    if (isNewReq && !guardrails.includes("guardrails:req_tag_enforced_on_done")) {
      throw new Error(`requirements_new_req_missing_regression_guardrail:${id}`);
    }
    requirementById.set(id, r);
  }

  const referencedReqIdsInCode = scanRepoReqTags(repoRoot);
  for (const [id, r] of requirementById.entries()) {
    const status = String(r.status || "").trim();
    if (!["draft", "canonical", "deprecated"].includes(status)) {
      throw new Error(`requirements_invalid_status:${id}:${status}`);
    }

    const tracking = r.tracking || {};
    const impl = String(tracking.implementation || "todo").trim();
    if (!["todo", "done"].includes(impl)) throw new Error(`requirements_invalid_tracking_implementation:${id}:${impl}`);
    if (impl === "done" && !referencedReqIdsInCode.has(id)) {
      throw new Error(`requirements_implementation_done_without_code_ref:${id} → Fix: Add REQ: tags to code that implements this requirement (e.g., // REQ: ${id}) or set tracking.implementation to "todo"`);
    }

    const isNewReq = [...newReqPrefixes].some((pfx) => id.startsWith(pfx));
    if (isNewReq && impl !== "todo" && impl !== "done") {
      throw new Error(`requirements_new_req_invalid_tracking:${id}:${impl}`);
    }
  }

  const intentsDir = path.join(repoRoot, "spec", "intents");
  const intentFiles = fs.readdirSync(intentsDir).filter((n) => n.endsWith(".json")).sort();
  const intentIds = new Set();
  const intents = [];
  const intentById = new Map();
  for (const name of intentFiles) {
    const abs = path.join(intentsDir, name);
    const obj = readJson(abs);
    const intentId = String(obj.intent_id || "").trim();
    if (!intentId) throw new Error(`intent_missing_intent_id:${name}`);
    if (name !== `${intentId}.json`) throw new Error(`intent_filename_mismatch:${name}:${intentId}`);
    if (intentIds.has(intentId)) throw new Error(`intent_duplicate_intent_id:${intentId}`);
    intentIds.add(intentId);
    intents.push(obj);
    intentById.set(intentId, obj);
    const runbookErrors = validateIntentRunbooks({ repoRoot, intentId, intent: obj });
    if (runbookErrors.length) throw new Error(`intent_runbooks_invalid:${runbookErrors[0]}`);
  }

  const tasksDir = path.join(repoRoot, "spec", "tasks");
  const taskFiles = fs.existsSync(tasksDir) ? fs.readdirSync(tasksDir).filter((n) => n.endsWith(".json")).sort() : [];
  const taskById = new Map();
  for (const name of taskFiles) {
    const abs = path.join(tasksDir, name);
    const t = readJson(abs);
    const taskId = String(t.task_id || "").trim();
    if (!taskId) throw new Error(`task_missing_task_id:${name}`);
    if (name !== `${taskId}.json`) throw new Error(`task_filename_mismatch:${name}:${taskId}`);
    if (taskById.has(taskId)) throw new Error(`task_duplicate_task_id:${taskId}`);
    taskById.set(taskId, t);

    const intentId = String(t.intent_id || "").trim();
    if (!intentId) throw new Error(`task_missing_intent_id:${taskId}`);
    if (!intentIds.has(intentId)) throw new Error(`task_unknown_intent_id:${taskId}:${intentId}`);
    const intentStatus = String(intentById.get(intentId)?.status || "").trim();

    const taskStatus = String(t.status || "").trim();
    if (taskStatus === "todo") {
      const scope = normalizeStringList(t.scope);
      if (!scope.length) throw new Error(`task_todo_requires_scope:${taskId}`);
      const acceptance = normalizeStringList(t.acceptance);
      if (!acceptance.length) throw new Error(`task_todo_requires_acceptance:${taskId}`);

      const deliverables = Array.isArray(t.deliverables) ? t.deliverables : [];
      if (!deliverables.length) throw new Error(`task_todo_requires_deliverable:${taskId}`);
      for (const d of deliverables) {
        const did = String(d?.deliverable_id || "").trim();
        const title = String(d?.title || "").trim();
        if (!did) throw new Error(`task_deliverable_missing_id:${taskId}`);
        if (!title) throw new Error(`task_deliverable_missing_title:${taskId}:${did}`);

        const paths = Array.isArray(d?.paths) ? d.paths.map(String).map((s) => s.trim()).filter(Boolean) : [];
        if (!paths.length) throw new Error(`task_deliverable_missing_paths:${taskId}:${did}`);
        if (!paths.every(looksLikeFilePath)) throw new Error(`task_deliverable_requires_file_paths:${taskId}:${did}`);

        const dAcc = normalizeStringList(d?.acceptance);
        if (!dAcc.length) throw new Error(`task_deliverable_missing_acceptance:${taskId}:${did}`);
      }
    }

    const subtasks = Array.isArray(t.subtasks) ? t.subtasks : [];
    if (taskStatus === "todo" && !subtasks.length) throw new Error(`task_todo_requires_subtasks:${taskId}`);
    const seenSubtaskIds = new Set();
    for (const st of subtasks) {
      const subtaskId = String(st.subtask_id || "").trim();
      if (!subtaskId) throw new Error(`subtask_missing_id:${taskId}`);
      if (seenSubtaskIds.has(subtaskId)) throw new Error(`subtask_duplicate_id:${taskId}:${subtaskId}`);
      seenSubtaskIds.add(subtaskId);

      const area = String(st.area || "").trim();
      if (!area) throw new Error(`subtask_missing_area:${taskId}:${subtaskId}`);
      const prov = String(st.provenance_prefix || "").trim();
      if (!prov) throw new Error(`subtask_missing_provenance_prefix:${taskId}:${subtaskId}`);
      const doneWhen = normalizeStringList(st.done_when);
      if (!doneWhen.length) throw new Error(`subtask_missing_done_when:${taskId}:${subtaskId}`);

      const isNewReq = [...newReqPrefixes].some((pfx) => subtaskId.startsWith(pfx));
      if (isNewReq) {
        if (!requirementById.has(subtaskId)) {
          throw new Error(`subtask_new_req_missing_in_requirements:${taskId}:${subtaskId}`);
        }
        const req = requirementById.get(subtaskId);
        const impl = String(req.tracking?.implementation || "todo");
        if (intentStatus === "closed") {
          if (impl !== "done") throw new Error(`subtask_new_req_must_be_done_when_intent_closed:${taskId}:${subtaskId}`);
        } else {
          if (impl !== "todo") throw new Error(`subtask_new_req_must_start_tracking_todo:${taskId}:${subtaskId}`);
        }
      }

      const reqsToAdd = Array.isArray(st.requirements_to_add) ? st.requirements_to_add : [];
      for (const r of reqsToAdd) {
        const rid = String(r.id || "").trim();
        if (!rid) throw new Error(`subtask_requirements_to_add_missing_id:${taskId}:${subtaskId}`);
        const okPrefix = [...newReqPrefixes].some((pfx) => rid.startsWith(pfx));
        if (!okPrefix) throw new Error(`subtask_new_req_invalid_prefix:${taskId}:${rid}`);
        if (!requirementById.has(rid)) {
          throw new Error(`subtask_requirements_to_add_missing_in_requirements:${taskId}:${rid}`);
        }
        const req = requirementById.get(rid);
        const impl = String(req.tracking?.implementation || "todo");
        if (intentStatus === "closed") {
          if (impl !== "done") throw new Error(`subtask_requirements_to_add_must_be_done_when_intent_closed:${taskId}:${rid}`);
        } else {
          if (impl !== "todo") throw new Error(`subtask_requirements_to_add_must_start_tracking_todo:${taskId}:${rid}`);
        }
      }
    }
  }

  for (const intent of intents) {
    const intentId = String(intent.intent_id || "").trim();
    const status = String(intent.status || "").trim();
    if (!["draft", "todo", "closed"].includes(status)) throw new Error(`intent_invalid_status:${intentId}:${status || "missing"}`);

    const createdDate = String(intent.created_date || "").trim();
    if (createdDate && !isIsoDate(createdDate)) throw new Error(`intent_invalid_created_date:${intentId}:${createdDate}`);
    const closedDate = String(intent.closed_date || "").trim();
    if (closedDate && !isIsoDate(closedDate)) throw new Error(`intent_invalid_closed_date:${intentId}:${closedDate}`);

    if (status === "draft") {
      const plannedTasks = normalizeReqIdList(intent.task_ids_planned);
      const wps = Array.isArray(intent.work_packages) ? intent.work_packages : [];
      if (plannedTasks.length) throw new Error(`intent_draft_must_have_no_tasks:${intentId}`);
      if (wps.length) throw new Error(`intent_draft_must_have_no_work_packages:${intentId}`);
      if (closedDate) throw new Error(`intent_draft_must_not_have_closed_date:${intentId}`);
      continue;
    }

    if (!normalizeStringList(intent.summary).length) throw new Error(`intent_missing_summary:${intentId}`);
    if (!normalizeStringList(intent.non_goals).length) throw new Error(`intent_missing_non_goals:${intentId}`);
    if (!normalizeStringList(intent.success_criteria).length) throw new Error(`intent_missing_success_criteria:${intentId}`);

    if (status === "closed") {
      if (!closedDate) throw new Error(`intent_closed_requires_closed_date:${intentId}`);
    } else {
      if (closedDate) throw new Error(`intent_todo_must_not_have_closed_date:${intentId}`);
    }

    const reqInScope = normalizeReqIdList(intent.requirements_in_scope);
    for (const rid of reqInScope) {
      if (!requirementById.has(rid)) throw new Error(`intent_unknown_requirement:${intentId}:${rid}`);
    }

    const plannedTasks = normalizeReqIdList(intent.task_ids_planned);
    if (!plannedTasks.length) throw new Error(`intent_missing_task_ids_planned:${intentId}`);
    for (const tid of plannedTasks) {
      if (!taskById.has(tid)) throw new Error(`intent_task_missing_spec:${intentId}:${tid}`);
      const t = taskById.get(tid);
      if (String(t.intent_id || "").trim() !== intentId) {
        throw new Error(`intent_task_wrong_intent:${intentId}:${tid}:${t.intent_id}`);
      }
    }

    const wps = Array.isArray(intent.work_packages) ? intent.work_packages : [];
    for (const wp of wps) {
      const wpId = String(wp.work_package_id || "").trim();
      if (!wpId.startsWith(`${intentId}-`)) throw new Error(`work_package_id_mismatch:${intentId}:${wpId}`);
      const items = Array.isArray(wp.items) ? wp.items : [];
      for (const item of items) {
        const first = String(item || "").trim().split(/\s+/)[0];
        if (!first) continue;
        if (!plannedTasks.includes(first)) throw new Error(`work_package_item_task_not_planned:${intentId}:${first}`);
      }
    }
  }

  const limitEntries = Object.entries(sizeLimits).map(([glob, bytes]) => ({
    glob,
    re: globToRegExp(String(glob)),
    maxBytes: Number(bytes),
  }));
  if (limitEntries.some((x) => !Number.isFinite(x.maxBytes) || x.maxBytes <= 0)) {
    throw new Error("file_size_limits_invalid");
  }
  const allFiles = iterFiles(repoRoot, () => true);
  for (const abs of allFiles) {
    const rel = relPosix(path.relative(repoRoot, abs));
    const size = fs.statSync(abs).size;
    for (const lim of limitEntries) {
      if (lim.re.test(rel) && size > lim.maxBytes) {
        throw new Error(`file_too_large:${rel}:${size}:${lim.maxBytes}`);
      }
    }
  }
}

function validateAllMarkdownGenerated(repoRoot) {
  const mdFiles = iterFiles(repoRoot, (p) => p.endsWith(".md"));
  const failures = [];
  for (const p of mdFiles) {
    const rel = relPosix(path.relative(repoRoot, p));
    const text = fs.readFileSync(p, "utf8");
    if (!hasGeneratedFrontmatter(text)) {
      failures.push(`${rel}: missing generated frontmatter (generated: true)`);
    }
  }
  if (failures.length) {
    for (const f of failures) process.stderr.write(`[guardrails:error] ${f}\n`);
    throw new Error("markdown_generation_violation");
  }
}

async function runGenerateCheck(repoRoot) {
  const { spawnSync } = await import("node:child_process");
  const res = spawnSync("node", ["scripts/generate_all.mjs", "--check"], { cwd: repoRoot, stdio: "inherit" });
  if (res.status !== 0) throw new Error("generate_check_failed");
}

async function main() {
  const repoRoot = repoRootFromHere(import.meta.url);
  await runGenerateCheck(repoRoot);
  validateGovernanceSpec(repoRoot);
  validateAllMarkdownGenerated(repoRoot);
  process.stdout.write("✓ guardrails ok\n");
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  process.stderr.write(`[guardrails:error] ${msg}\n`);
  process.exit(2);
});
