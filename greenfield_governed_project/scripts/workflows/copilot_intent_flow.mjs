/*
PROV: GREENFIELD.GOV.WORKFLOW.COPILOT_INTENT.01
REQ: SYS-ARCH-15, AUD-REQ-10
WHY: Generate a Copilot prompt file and apply Copilot-produced JSON to create governed intent/task/requirement specs.
*/

import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { spawnSync } from "node:child_process";

import { repoRootFromHere, relPosix } from "../lib/paths.mjs";

function parseArgs(argv) {
  const out = { intentId: "", new: false };
  const args = [...argv];
  while (args.length) {
    const a = args.shift();
    if (a === "--intent-id") out.intentId = String(args.shift() || "");
    else if (a === "--new") out.new = true;
  }
  return out;
}

function readJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function readJsonSafe(p) {
  try {
    if (!fs.existsSync(p)) return null;
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return null;
  }
}

function writeText(p, text) {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, text, "utf8");
}

function writeJson(p, obj) {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, JSON.stringify(obj, null, 2) + "\n", "utf8");
}

function ensureArray(v) {
  return Array.isArray(v) ? v : [];
}

function normalizeIdList(v) {
  return ensureArray(v).map(String).map((s) => s.trim()).filter(Boolean);
}

function nextNumericSuffix(existingIds, prefix, pad) {
  let max = 0;
  for (const id of existingIds) {
    if (!id.startsWith(prefix)) continue;
    const raw = id.slice(prefix.length);
    const n = Number.parseInt(raw, 10);
    if (Number.isFinite(n)) max = Math.max(max, n);
  }
  const next = max + 1;
  return `${prefix}${String(next).padStart(pad, "0")}`;
}

function listIntentIds(repoRoot) {
  const dir = path.join(repoRoot, "spec", "intents");
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir)
    .filter((n) => n.endsWith(".json"))
    .map((n) => n.replace(/\.json$/, ""))
    .sort();
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

function listAreaChoices(areaFilesRel) {
  return ensureArray(areaFilesRel).map((rel) => ({ label: path.basename(rel, ".json"), rel }));
}

function safePosixRelative(p) {
  const rel = String(p || "").replace(/\\/g, "/").trim();
  if (!rel) return "";
  if (rel.startsWith("/") || rel.includes("..")) throw new Error(`invalid_path:${p}`);
  return rel;
}

function run(cmd, args, cwd) {
  process.stdout.write(`[flow] run: ${cmd} ${args.join(" ")}\n`);
  const res = spawnSync(cmd, args, { cwd, stdio: "inherit" });
  if (res.status !== 0) throw new Error(`command_failed:${cmd}`);
}

async function withRl(fn) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  try {
    return await fn(rl);
  } finally {
    rl.close();
  }
}

function question(rl, prompt) {
  return new Promise((resolve) => rl.question(prompt, resolve));
}

async function askNonEmpty(rl, prompt, { defaultValue } = {}) {
  while (true) {
    const raw = await question(rl, defaultValue ? `${prompt} [${defaultValue}]: ` : `${prompt}: `);
    const v = raw.trim() || String(defaultValue || "").trim();
    if (v) return v;
    process.stdout.write("[flow] please enter a value\n");
  }
}

async function askYesNo(rl, prompt, { defaultValue } = {}) {
  const hint = defaultValue === true ? "Y/n" : defaultValue === false ? "y/N" : "y/n";
  while (true) {
    const raw = (await question(rl, `${prompt} (${hint}): `)).trim().toLowerCase();
    if (!raw && typeof defaultValue === "boolean") return defaultValue;
    if (["y", "yes"].includes(raw)) return true;
    if (["n", "no"].includes(raw)) return false;
    process.stdout.write("[flow] please answer y/n\n");
  }
}

function openInEditor(filePath) {
  const tried = [];

  const code = spawnSync("code", ["-r", filePath], { stdio: "ignore" });
  tried.push({ cmd: "code -r", status: code.status });
  if (code.status === 0) return true;

  if (process.platform === "darwin") {
    const openBundle = spawnSync("open", ["-b", "com.microsoft.VSCode", filePath], { stdio: "ignore" });
    tried.push({ cmd: "open -b com.microsoft.VSCode", status: openBundle.status });
    if (openBundle.status === 0) return true;

    const openBundleInsiders = spawnSync("open", ["-b", "com.microsoft.VSCodeInsiders", filePath], { stdio: "ignore" });
    tried.push({ cmd: "open -b com.microsoft.VSCodeInsiders", status: openBundleInsiders.status });
    if (openBundleInsiders.status === 0) return true;

    const openVsCode = spawnSync("open", ["-a", "Visual Studio Code", filePath], { stdio: "ignore" });
    tried.push({ cmd: "open -a Visual Studio Code", status: openVsCode.status });
    if (openVsCode.status === 0) return true;

    const openDefault = spawnSync("open", [filePath], { stdio: "ignore" });
    tried.push({ cmd: "open", status: openDefault.status });
    if (openDefault.status === 0) return true;
  }

  if (process.platform === "linux") {
    const xdg = spawnSync("xdg-open", [filePath], { stdio: "ignore" });
    tried.push({ cmd: "xdg-open", status: xdg.status });
    if (xdg.status === 0) return true;
  }

  if (process.platform === "win32") {
    const start = spawnSync("cmd", ["/c", "start", "", filePath], { stdio: "ignore" });
    tried.push({ cmd: "cmd /c start", status: start.status });
    if (start.status === 0) return true;
  }

  process.stdout.write(`[flow] unable to auto-open file; tried: ${tried.map((t) => `${t.cmd}(${t.status})`).join(", ")}\n`);
  return false;
}

function addRequirementToArea({ repoRoot, areaRel, requirement }) {
  const areaAbs = path.join(repoRoot, areaRel);
  const area = readJson(areaAbs);
  area.requirements = ensureArray(area.requirements);
  const id = String(requirement.id || "").trim();
  if (area.requirements.some((r) => String(r.id || "").trim() === id)) return false;
  area.requirements.push(requirement);
  area.requirements.sort((a, b) => String(a.id || "").localeCompare(String(b.id || "")));
  writeJson(areaAbs, area);
  return true;
}

function validatePlan({ repoRoot, plan, areaChoices, newReqPrefix }) {
  const errors = [];

  const intent = plan?.intent;
  if (!intent || typeof intent !== "object") errors.push("plan.intent missing");
  const tasks = ensureArray(plan?.tasks);
  if (!tasks.length) errors.push("plan.tasks must be non-empty");

  const intentId = String(intent?.intent_id || "").trim();
  if (!/^INT-\d{3}$/.test(intentId)) errors.push("intent.intent_id must match INT-###");
  if (String(intent?.status || "") !== "todo") errors.push("intent.status must be todo");

  const intentPath = path.join(repoRoot, "spec", "intents", `${intentId}.json`);
  if (fs.existsSync(intentPath)) errors.push(`intent already exists: spec/intents/${intentId}.json`);

  const taskIdsPlanned = normalizeIdList(intent?.task_ids_planned);
  if (!taskIdsPlanned.length) errors.push("intent.task_ids_planned must be non-empty");

  const taskIdsInPlan = tasks.map((t) => String(t?.task_id || "").trim()).filter(Boolean);
  for (const tid of taskIdsInPlan) {
    if (!/^TASK-[A-Z0-9-]+-\d{3}$/.test(tid)) errors.push(`invalid task id: ${tid}`);
  }
  for (const tid of taskIdsPlanned) {
    if (!taskIdsInPlan.includes(tid)) errors.push(`intent.task_ids_planned missing task in plan.tasks: ${tid}`);
  }

  const intentRequirements = normalizeIdList(intent?.requirements_in_scope);
  const requirementsSourceRel = String(readJson(path.join(repoRoot, "spec", "project.json")).requirements_source || "");
  const bundle = loadRequirementsBundle({ repoRoot, requirementsSourceRel });
  const existingReqIds = new Set(bundle.requirements.map((r) => String(r.id || "").trim()).filter(Boolean));

  const reqAdds = ensureArray(plan?.requirements_to_add);
  for (const item of reqAdds) {
    const areaFile = String(item?.area_file || "").trim();
    const req = item?.requirement;
    if (!areaFile) errors.push("requirements_to_add[*].area_file missing");
    if (!req || typeof req !== "object") errors.push("requirements_to_add[*].requirement missing");
    const choice = areaChoices.find((c) => c.rel === areaFile);
    if (!choice) errors.push(`requirements_to_add area_file not in index: ${areaFile}`);

    const id = String(req?.id || "").trim();
    if (!id.startsWith(newReqPrefix)) errors.push(`new requirement id must start with ${newReqPrefix}: ${id}`);
    if (existingReqIds.has(id)) errors.push(`requirement already exists: ${id}`);
    if (String(req?.tracking?.implementation || "todo") !== "todo") errors.push(`new requirement tracking.implementation must be todo: ${id}`);
  }

  for (const rid of intentRequirements) {
    if (!existingReqIds.has(rid) && !reqAdds.some((x) => String(x?.requirement?.id || "") === rid)) {
      errors.push(`unknown requirement in intent.requirements_in_scope: ${rid}`);
    }
  }

  return { ok: errors.length === 0, errors, intentId };
}

function applyPlan({ repoRoot, plan, areaChoices }) {
  const intent = plan.intent;
  const intentId = String(intent.intent_id || "").trim();

  for (const item of ensureArray(plan.requirements_to_add)) {
    addRequirementToArea({ repoRoot, areaRel: item.area_file, requirement: item.requirement });
  }

  const tasksDir = path.join(repoRoot, "spec", "tasks");
  for (const t of ensureArray(plan.tasks)) {
    const taskId = String(t.task_id || "").trim();
    const taskPath = path.join(tasksDir, `${taskId}.json`);
    if (fs.existsSync(taskPath)) throw new Error(`task_already_exists:${taskId}`);

    const deliverables = ensureArray(t.deliverables).map((d) => ({
      deliverable_id: String(d.deliverable_id || "").trim() || `DELIV-${taskId}-001`,
      title: String(d.title || "").trim(),
      paths: ensureArray(d.paths).map(safePosixRelative).filter(Boolean),
    }));

    const taskSpec = {
      schema_version: 1,
      task_id: taskId,
      intent_id: intentId,
      title: String(t.title || "").trim(),
      status: String(t.status || "todo").trim(),
      deliverables,
      subtasks: ensureArray(t.subtasks),
    };
    writeJson(taskPath, taskSpec);
  }

  const intentPath = path.join(repoRoot, "spec", "intents", `${intentId}.json`);
  writeJson(intentPath, intent);
}

function promptText({ repoRootRel, intentId, newReqPrefix, requirementsSourceRel, areaChoices }) {
  const areas = areaChoices.map((a) => `- ${a.label}: \`${a.rel}\``).join("\n");
  return [
    "You are GitHub Copilot Chat running inside VS Code.",
    "",
    "You will help me create a governed intent + tasks (JSON only).",
    "",
    "First: ask me clarifying questions (max 10). Ask one question at a time. After I answer, ask the next.",
    "",
    "When done, output ONLY a single JSON object (no markdown fences) that matches this schema:",
    "",
    "{",
    `  \"intent\": { /* write to spec/intents/${intentId}.json */`,
    `    \"intent_id\": \"${intentId}\",`,
    "    \"title\": \"...\",",
    "    \"status\": \"todo\",",
    "    \"created_date\": \"YYYY-MM-DD\",",
    "    \"close_gate\": [\"npm run guardrails\", \"npm run generate:check\", \"npm run audit:intent -- --intent-id " + intentId + "\"],",
    "    \"summary\": [\"...\"],",
    "    \"requirements_in_scope\": [\"...\"],",
    "    \"task_ids_planned\": [\"TASK-AREA-001\"],",
    "    \"work_packages\": [",
    "      {",
    "        \"work_package_id\": \"" + intentId + "-001\",",
    "        \"title\": \"...\",",
    "        \"items\": [\"TASK-AREA-001 ...\"]",
    "      }",
    "    ]",
    "  },",
    "  \"tasks\": [",
    "    {",
    "      \"task_id\": \"TASK-AREA-001\",",
    "      \"title\": \"deliverable title\",",
    "      \"status\": \"todo\",",
    "      \"deliverables\": [",
    "        {",
    "          \"deliverable_id\": \"DELIV-TASK-AREA-001-001\",",
    "          \"title\": \"...\",",
    "          \"paths\": [\"pipeline/\"]",
    "        }",
    "      ],",
    "      \"subtasks\": [",
    "        {",
    `          \"subtask_id\": \"${newReqPrefix}0001\",`,
    "          \"title\": \"new requirement title\",",
    "          \"status\": \"todo\",",
    "          \"area\": \"core\",",
    "          \"requirements_to_add\": [",
    "            {",
    `              \"id\": \"${newReqPrefix}0001\",`,
    "              \"status\": \"draft\",",
    "              \"tracking\": { \"implementation\": \"todo\" },",
    "              \"title\": \"...\",",
    "              \"acceptance\": [\"...\"],",
    "              \"owner\": \"platform\",",
    "              \"tags\": [\"...\"]",
    "            }",
    "          ]",
    "        }",
    "      ]",
    "    }",
    "  ],",
    "  \"requirements_to_add\": [",
    "    {",
    "      \"area_file\": \"spec/requirements/areas/core.json\",",
    "      \"requirement\": {",
    `        \"id\": \"${newReqPrefix}0001\",`,
    "        \"status\": \"draft\",",
    "        \"tracking\": { \"implementation\": \"todo\" },",
    "        \"title\": \"...\",",
    "        \"acceptance\": [\"...\"],",
    "        \"owner\": \"platform\",",
    "        \"tags\": [\"...\"]",
    "      }",
    "    }",
    "  ]",
    "}",
    "",
    "Rules you MUST follow:",
    "- Do not edit or propose edits to any .md files (generated).",
    "- Only output JSON for canonical sources (intent/tasks/requirements).",
    "- New requirements must have ids starting with " + newReqPrefix + " and tracking.implementation=\"todo\".",
    "- Prefer 1–4 tasks; make tasks real deliverables.",
    "- Use only relative paths; no absolute paths; no '..'.",
    "",
    "Repo pointers:",
    `- Requirements entrypoint: \`${requirementsSourceRel}\``,
    "Requirements areas:",
    areas,
    `- Intent statuses: draft | todo | closed (create todo now)`,
    `- Run validation after applying: \`npm run generate\` then \`npm run guardrails\` (from ${repoRootRel})`,
    "",
  ].join("\n");
}

async function main() {
  const repoRoot = repoRootFromHere(import.meta.url);
  const repoRootRel = ".";
  const args = parseArgs(process.argv.slice(2));
  const project = readJson(path.join(repoRoot, "spec", "project.json"));
  const requirementsSourceRel = String(project.requirements_source || "");
  const governance = project.governance || {};
  const newReqPrefixes = Array.isArray(governance.new_requirement_id_prefixes) ? governance.new_requirement_id_prefixes.map(String) : ["REQ-"];
  const newReqPrefix = newReqPrefixes[0] || "REQ-";

  const bundle = loadRequirementsBundle({ repoRoot, requirementsSourceRel });
  const areaChoices = listAreaChoices(bundle.areaFilesRel);
  const intentIds = listIntentIds(repoRoot);
  const defaultIntentId = nextNumericSuffix(intentIds, "INT-", 3);

  const wizardRoot = path.join(repoRoot, "status", "wizard");
  const statePath = path.join(wizardRoot, "ACTIVE.json");

  function getOrCreateState() {
    if (!args.new) {
      const existing = readJsonSafe(statePath);
      if (existing?.intent_id) {
        process.stdout.write(`[flow] continuing active session: ${existing.intent_id}\n`);
        return existing;
      }
    }
    const intentId = (args.intentId || defaultIntentId).trim();
    if (!/^INT-\d{3}$/.test(intentId)) throw new Error(`invalid_intent_id:${intentId}`);
    const intentPath = path.join(repoRoot, "spec", "intents", `${intentId}.json`);
    if (fs.existsSync(intentPath)) throw new Error(`intent_already_exists:${intentId}`);
    const wizardDir = path.join(wizardRoot, intentId);
    const promptPath = path.join(wizardDir, "copilot_prompt.txt");
    const planPath = path.join(wizardDir, "copilot_plan.json");
    const state = {
      schema_version: 1,
      intent_id: intentId,
      prompt_path: relPosix(path.relative(repoRoot, promptPath)),
      plan_path: relPosix(path.relative(repoRoot, planPath)),
    };
    writeJson(statePath, state);
    writeText(promptPath, promptText({ repoRootRel, intentId, newReqPrefix, requirementsSourceRel, areaChoices }));
    if (!fs.existsSync(planPath)) {
      writeJson(planPath, { note: "Paste Copilot final JSON here (replace this object)." });
    }
    process.stdout.write(`[flow] started new session: ${intentId} (next available by default)\n`);
    return state;
  }

  const state = getOrCreateState();
  const promptAbs = path.join(repoRoot, state.prompt_path);
  const planAbs = path.join(repoRoot, state.plan_path);

  const plan = readJsonSafe(planAbs);
  const planLooksReady = !!(plan && typeof plan === "object" && plan.intent && plan.tasks);

  if (!planLooksReady) {
    process.stdout.write(`[flow] Copilot prompt: ${state.prompt_path}\n`);
    process.stdout.write(`[flow] Plan file (paste Copilot FINAL JSON here): ${state.plan_path}\n`);
    openInEditor(promptAbs);
    process.stdout.write("[flow] Paste the prompt into Copilot Chat, answer questions, then paste the FINAL JSON into the plan file.\n");
    process.stdout.write("[flow] Re-run this same task to apply.\n");
    return;
  }

  const validation = validatePlan({ repoRoot, plan, areaChoices, newReqPrefix });
  if (!validation.ok) {
    process.stderr.write("[flow:error] invalid plan:\n");
    for (const e of validation.errors) process.stderr.write(`- ${e}\n`);
    process.exit(2);
  }

  await withRl(async (rl) => {
    process.stdout.write(`[flow] plan ready for ${validation.intentId}\n`);
    const ok = await askYesNo(rl, "Apply plan now? (will write spec/ files)", { defaultValue: true });
    if (!ok) {
      process.stdout.write("[flow] canceled\n");
      return;
    }
    applyPlan({ repoRoot, plan, areaChoices });
    run("npm", ["run", "generate"], repoRoot);
    run("npm", ["run", "guardrails"], repoRoot);
    fs.rmSync(statePath, { force: true });
    process.stdout.write("[flow] ok\n");
  });
}

main().catch((e) => {
  const msg = e instanceof Error ? e.message : String(e);
  process.stderr.write(`[flow:error] ${msg}\n`);
  process.exit(2);
});
