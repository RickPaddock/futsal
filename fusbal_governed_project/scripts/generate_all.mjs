/*
PROV: GREENFIELD.SCAFFOLD.GEN.01
REQ: AUD-REQ-10, SYS-ARCH-15, GREENFIELD-GOV-018, GREENFIELD-GEN-001, GREENFIELD-SCHEMA-001
WHY: Deterministically generate all human-readable .md outputs and portal feeds from spec sources (including runbook navigation cues).
*/

import fs from "node:fs";
import path from "node:path";
import Ajv from "ajv";
import { repoRootFromHere, relPosix } from "./lib/paths.mjs";
import { sha256Bytes, sha256File } from "./lib/sha256.mjs";

const ajv = new Ajv({ allErrors: true });
let DRY_RUN_DIFFS_FOUND = false;

function readJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function validateJson(repoRoot, data, schemaName) {
  const schemaPath = path.join(repoRoot, "spec", "schemas", `${schemaName}.schema.json`);
  if (!fs.existsSync(schemaPath)) return; // Skip validation if schema doesn't exist
  
  const schema = readJson(schemaPath);
  const validate = ajv.compile(schema);
  const valid = validate(data);
  
  if (!valid) {
    const errors = validate.errors.map(e => `${e.instancePath} ${e.message}`).join("; ");
    throw new Error(`schema_validation_failed:${schemaName}:${errors}`);
  }
}

function sha256Sources(repoRoot, sourcesRel) {
  const lines = sourcesRel
    .slice()
    .sort()
    .map((rel) => `${rel} sha256:${sha256File(path.join(repoRoot, rel))}`);
  return sha256Bytes(Buffer.from(lines.join("\n"), "utf8"));
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function renderTemplate(templatePath, vars) {
  const raw = fs.readFileSync(templatePath, "utf8");
  return raw.replace(/\$\{([a-zA-Z0-9_]+)\}/g, (_, k) => String(vars[k] ?? ""));
}

function extractSchemaSnippets(repoRoot) {
  const schemaPath = path.join(repoRoot, "pipeline", "schemas", "contract.v1.schema.json");
  if (!fs.existsSync(schemaPath)) {
    return {
      BREAK_REASONS: "[Schema not found]",
      TEAM_LABELS: "[Schema not found]", 
      POS_STATES: "[Schema not found]",
      ENTITY_TYPES: "[Schema not found]",
      FRAMES: "[Schema not found]",
    };
  }
  
  try {
    const schema = readJson(schemaPath);
    const trackRecord = schema.$defs?.TrackRecordV1?.properties;
    
    const extractEnum = (prop) => {
      if (prop?.enum) {
        return prop.enum.map(v => `\`${v}\``).join(", ");
      }
      return "[Not found]";
    };
    
    return {
      BREAK_REASONS: extractEnum(trackRecord?.break_reason),
      TEAM_LABELS: extractEnum(trackRecord?.team),
      POS_STATES: extractEnum(trackRecord?.pos_state),
      ENTITY_TYPES: extractEnum(trackRecord?.entity_type),
      FRAMES: extractEnum(trackRecord?.frame),
    };
  } catch (e) {
    return {
      BREAK_REASONS: "[Parse error]",
      TEAM_LABELS: "[Parse error]",
      POS_STATES: "[Parse error]", 
      ENTITY_TYPES: "[Parse error]",
      FRAMES: "[Parse error]",
    };
  }
}

function iterFiles(root, predicate) {
  const out = [];
  for (const p of fs.readdirSync(root, { withFileTypes: true })) {
    const abs = path.join(root, p.name);
    if (p.isDirectory()) out.push(...iterFiles(abs, predicate));
    else if (p.isFile() && predicate(abs)) out.push(abs);
  }
  return out;
}

function writeFileChecked(outPath, content, check, dryRun) {
  if (check) {
    const existing = fs.existsSync(outPath) ? fs.readFileSync(outPath, "utf8") : null;
    if (existing !== content) {
      throw new Error(`drift:${relPosix(outPath)}`);
    }
    return;
  }
  if (dryRun) {
    const existing = fs.existsSync(outPath) ? fs.readFileSync(outPath, "utf8") : null;
    if (existing === content) return;
    const action = existing === null ? "CREATE" : "UPDATE";
    process.stdout.write(`[dry-run] ${action}: ${relPosix(outPath)}\n`);
    DRY_RUN_DIFFS_FOUND = true;
    const maxLines = 18;
    const oldLines = (existing || "").split(/\r?\n/);
    const newLines = String(content || "").split(/\r?\n/);
    let first = 0;
    while (first < oldLines.length && first < newLines.length && oldLines[first] === newLines[first]) first++;
    let lastOld = oldLines.length - 1;
    let lastNew = newLines.length - 1;
    while (lastOld >= first && lastNew >= first && oldLines[lastOld] === newLines[lastNew]) {
      lastOld--;
      lastNew--;
    }
    const oldChunk = oldLines.slice(first, Math.min(oldLines.length, first + maxLines));
    const newChunk = newLines.slice(first, Math.min(newLines.length, first + maxLines));
    if (oldChunk.length || newChunk.length) {
      process.stdout.write(`[dry-run] diff (around first change @line ${first + 1})\n`);
      for (const l of oldChunk) process.stdout.write(`- ${l}\n`);
      for (const l of newChunk) process.stdout.write(`+ ${l}\n`);
    }
    return;
  }
  ensureDir(path.dirname(outPath));
  fs.writeFileSync(outPath, content, "utf8");
}

function generatedFrontmatter({ source, sourceHash }) {
  return `---\ngenerated: true\nsource: ${source}\nsource_sha256: sha256:${sourceHash}\n---\n\n`;
}

function generateMdFromMdt({ repoRoot, templateRel, outRel, vars, check, dryRun }) {
  const templatePath = path.join(repoRoot, templateRel);
  const sourceHash = sha256File(templatePath);
  const rendered = renderTemplate(templatePath, vars);
  // Ensure all .md outputs include deterministic generated frontmatter.
  const body = rendered.replace(/^---\n[\s\S]*?\n---\n\n/, "");
  const out = generatedFrontmatter({ source: templateRel, sourceHash }) + body;
  writeFileChecked(path.join(repoRoot, outRel), out, check, dryRun);
}

function generateAllMdTemplates({ repoRoot, templatesRootRel, vars, check, dryRun }) {
  const templatesRootAbs = path.join(repoRoot, templatesRootRel);
  const reservedOut = new Set(["docs/requirements/requirements.md"]);
  const templates = iterFiles(templatesRootAbs, (p) => p.endsWith(".mdt")).sort();
  for (const abs of templates) {
    const templateRel = relPosix(path.relative(repoRoot, abs));
    const relFromTemplatesRoot = relPosix(path.relative(templatesRootAbs, abs));
    const outRel = relFromTemplatesRoot.replace(/\.mdt$/, ".md");
    if (reservedOut.has(outRel)) {
      throw new Error(`template_output_reserved:${outRel}`);
    }
    generateMdFromMdt({ repoRoot, templateRel, outRel, vars, check, dryRun });
  }
}

function loadRequirementsBundle({ repoRoot, requirementsSourceRel }) {
  const abs = path.join(repoRoot, requirementsSourceRel);
  const root = readJson(abs);

  if (root?.type === "requirements_index") {
    const files = Array.isArray(root.files) ? root.files.map(String) : [];
    if (!files.length) throw new Error(`requirements_index_empty:${requirementsSourceRel}`);

    const requirements = [];
    for (const f of files) {
      const rel = path.isAbsolute(f) ? relPosix(path.relative(repoRoot, f)) : f;
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

function generateRequirementsMd({ repoRoot, vars, check, dryRun }) {
  const bundle = loadRequirementsBundle({ repoRoot, requirementsSourceRel: vars.requirements_source });
  const requirements = { requirements: bundle.requirements };
  const lines = [];
  lines.push(`# Requirements (generated)`);
  lines.push("");
  lines.push(`Source: \`${vars.requirements_source}\``);
  lines.push("");
  for (const r of requirements.requirements || []) {
    lines.push(`## ${r.id} — ${r.title}`);
    lines.push("");
    lines.push(`- Status: \`${r.status}\``);
    if (r.tracking?.implementation) lines.push(`- Implementation: \`${r.tracking.implementation}\``);
    if (Array.isArray(r.guardrails) && r.guardrails.length) lines.push(`- Guardrails: ${r.guardrails.map((g) => `\`${g}\``).join(", ")}`);
    if (r.owner) lines.push(`- Owner: \`${r.owner}\``);
    if (Array.isArray(r.tags) && r.tags.length) lines.push(`- Tags: ${r.tags.map((t) => `\`${t}\``).join(", ")}`);
    lines.push("");
    if (Array.isArray(r.acceptance) && r.acceptance.length) {
      lines.push("Acceptance:");
      for (const a of r.acceptance) lines.push(`- ${a}`);
      lines.push("");
    }
  }
  const outRel = "docs/requirements/requirements.md";
  const source = bundle.sourcesRel.length === 1 ? bundle.sourcesRel[0] : bundle.sourcesRel.join(" + ");
  const out = generatedFrontmatter({ source, sourceHash: sha256Sources(repoRoot, bundle.sourcesRel) }) + lines.join("\n") + "\n";
  writeFileChecked(path.join(repoRoot, outRel), out, check, dryRun);
}

function generateIntentFiles({ repoRoot, vars, check, dryRun }) {
  const intentsDir = path.join(repoRoot, "spec", "intents");
  const entries = fs.readdirSync(intentsDir, { withFileTypes: true }).filter((d) => d.isFile() && d.name.endsWith(".json"));
  for (const e of entries) {
    const obj = readJson(path.join(intentsDir, e.name));
    const intentId = String(obj.intent_id || "").trim();
    if (!intentId) continue;

    const statusDir = path.join(repoRoot, "status", "intents", intentId);
    if (!dryRun) ensureDir(statusDir);

	    const scope = {
	      intent_id: intentId,
	      title: obj.title || "",
	      requirements_in_scope: obj.requirements_in_scope || [],
	      task_ids_planned: obj.task_ids_planned || [],
	      paths_allowed: obj.paths_allowed || [],
	      paths_excluded: obj.paths_excluded || [],
	      runbooks: obj.runbooks || null,
	      close_gate: { commands: obj.close_gate || [], evidence_out_root: `status/audit/${intentId}/runs` },
	    };
	    validateJson(repoRoot, scope, "intent_scope");
	    const scopeJson = JSON.stringify(scope, null, 2) + "\n";
    writeFileChecked(path.join(statusDir, "scope.json"), scopeJson, check, dryRun);

    const workPackages = {
      intent_id: intentId,
      work_packages: (obj.work_packages || []).map((wp) => ({
        work_package_id: wp.work_package_id,
        title: wp.title,
        primary_task_ids: (wp.items || []).map((x) => String(x).split(/\s+/)[0]),
      })),
    };
    validateJson(repoRoot, workPackages, "work_packages");
    const workJson = JSON.stringify(workPackages, null, 2) + "\n";
    writeFileChecked(path.join(statusDir, "work_packages.json"), workJson, check, dryRun);

    const mdLines = [];
    mdLines.push("---");
    mdLines.push(`generated: true`);
    mdLines.push(`source: spec/intents/${e.name}`);
    mdLines.push(`source_sha256: sha256:${sha256File(path.join(intentsDir, e.name))}`);
    mdLines.push(`intent_id: ${intentId}`);
    mdLines.push(`title: ${obj.title}`);
    mdLines.push(`status: ${obj.status}`);
    mdLines.push(`created_date: ${obj.created_date}`);
    if (obj.closed_date) mdLines.push(`closed_date: ${obj.closed_date}`);
    mdLines.push("close_gate:");
    for (const c of obj.close_gate || []) mdLines.push(`  - \"${c}\"`);
    mdLines.push("---");
    mdLines.push("");
    mdLines.push(`# Intent: ${intentId}`);
    mdLines.push("");
    for (const s of obj.summary || []) mdLines.push(`- ${s}`);
    mdLines.push("");
    mdLines.push("## Work packages");
    mdLines.push("");
    for (const wp of obj.work_packages || []) {
      mdLines.push(`### ${wp.work_package_id} — ${wp.title}`);
      mdLines.push("");
      for (const item of wp.items || []) mdLines.push(`- ${item}`);
      mdLines.push("");
    }

	    mdLines.push("## Runbooks (LLM navigation)");
	    mdLines.push("");
	    const rb = obj.runbooks || {};
	    mdLines.push(`- Decision: \`${rb.decision || "missing"}\``);
	    if (Array.isArray(rb.paths_mdt) && rb.paths_mdt.length) {
	      mdLines.push(`- Templates: ${rb.paths_mdt.map((p) => `\`${p}\``).join(", ")}`);
	    } else {
	      mdLines.push(`- Templates: (none)`);
	    }
	    mdLines.push(`- Notes: ${rb.notes || ""}`);
	    mdLines.push("");

	    mdLines.push("## Scope (paths)");
	    mdLines.push("");
	    const allowed = Array.isArray(obj.paths_allowed) ? obj.paths_allowed.map(String).map((s) => s.trim()).filter(Boolean) : [];
	    const excluded = Array.isArray(obj.paths_excluded) ? obj.paths_excluded.map(String).map((s) => s.trim()).filter(Boolean) : [];
	    mdLines.push(`- Allowed: ${allowed.length ? allowed.map((p) => `\`${p}\``).join(", ") : "(missing)"}`);
	    mdLines.push(`- Excluded: ${excluded.length ? excluded.map((p) => `\`${p}\``).join(", ") : "(missing)"}`);
	    mdLines.push("");

	    writeFileChecked(path.join(statusDir, "intent.md"), mdLines.join("\n") + "\n", check, dryRun);
	  }
	}

function generateInternalIntentsFeed({ repoRoot, vars, check, dryRun }) {
  const requirementsBundle = loadRequirementsBundle({ repoRoot, requirementsSourceRel: vars.requirements_source });
  const requirementsById = new Map(
    (requirementsBundle.requirements || []).map((r) => [
      String(r.id || "").trim(),
      {
        id: String(r.id || "").trim(),
        title: r.title || "",
        status: r.status || "",
        tracking_implementation: r.tracking?.implementation || "todo",
      },
    ]),
  );

  const tasksDir = path.join(repoRoot, "spec", "tasks");
  const taskFiles = fs.existsSync(tasksDir)
    ? fs.readdirSync(tasksDir, { withFileTypes: true }).filter((d) => d.isFile() && d.name.endsWith(".json")).map((d) => d.name).sort()
    : [];
  const tasksByIntent = new Map();
  for (const name of taskFiles) {
    const t = readJson(path.join(tasksDir, name));
    const intentId = String(t.intent_id || "").trim();
    const taskId = String(t.task_id || "").trim();
    if (!intentId || !taskId) continue;
    if (!tasksByIntent.has(intentId)) tasksByIntent.set(intentId, []);
    tasksByIntent.get(intentId).push({ task_id: taskId, title: t.title || "", status: t.status || "todo" });
  }
  for (const [k, v] of tasksByIntent.entries()) {
    v.sort((a, b) => a.task_id.localeCompare(b.task_id));
    tasksByIntent.set(k, v);
  }

  const intentsDir = path.join(repoRoot, "spec", "intents");
  const intentFiles = fs.readdirSync(intentsDir, { withFileTypes: true }).filter((d) => d.isFile() && d.name.endsWith(".json")).map((d) => d.name).sort();
  const intents = [];
  for (const name of intentFiles) {
    const obj = readJson(path.join(intentsDir, name));
    const intentId = String(obj.intent_id || "").trim();
    if (!intentId) continue;
    const reqs = (obj.requirements_in_scope || []).map(String).map((id) => id.trim()).filter(Boolean).sort();
    intents.push({
      intent_id: intentId,
      title: obj.title || "",
      status: obj.status || "",
      created_date: obj.created_date || "",
      closed_date: obj.closed_date || "",
      requirements_in_scope: reqs.map((id) => requirementsById.get(id) || { id, title: "", status: "", tracking_implementation: "todo" }),
      tasks: tasksByIntent.get(intentId) || [],
    });
  }

  const out = {
    generated: true,
    source: `spec/intents/*.json + spec/tasks/*.json + ${vars.requirements_source}`,
    requirements_source: vars.requirements_source,
    intents: intents.sort((a, b) => a.intent_id.localeCompare(b.intent_id)),
  };
  validateJson(repoRoot, out, "internal_intents");
  const outPath = path.join(repoRoot, "status", "portal", "internal_intents.json");
  writeFileChecked(outPath, JSON.stringify(out, null, 2) + "\n", check, dryRun);
}

function main() {
  const repoRoot = repoRootFromHere(import.meta.url);
  const check = process.argv.includes("--check");
  const dryRun = process.argv.includes("--dry-run");
  const project = readJson(path.join(repoRoot, "spec", "project.json"));
  const schemaSnippets = extractSchemaSnippets(repoRoot);
  const vars = {
    project_name: project.project_name,
    project_id: project.project_id,
    intent_prefix: project.intent_prefix,
    requirements_source: project.requirements_source,
    md_templates_root: project.md_templates_root,
    ...schemaSnippets,
  };

  if (dryRun) {
    process.stdout.write("[generate] DRY-RUN MODE: listing files that would be generated\n");
  }

  generateAllMdTemplates({ repoRoot, templatesRootRel: vars.md_templates_root, vars, check, dryRun });

  generateRequirementsMd({ repoRoot, vars, check, dryRun });
  generateIntentFiles({ repoRoot, vars, check, dryRun });
  generateInternalIntentsFeed({ repoRoot, vars, check, dryRun });
}

try {
  main();
  if (process.argv.includes("--check")) {
    process.stdout.write("[generate] ok\n");
  } else if (!process.argv.includes("--dry-run")) {
    process.stdout.write("[generate] ok\n");
  } else if (process.argv.includes("--dry-run") && DRY_RUN_DIFFS_FOUND) {
    process.stderr.write("[generate] dry-run found diffs\n");
    process.exitCode = 1;
  }
} catch (err) {
  const msg = err instanceof Error ? err.message : String(err);
  process.stderr.write(`[generate:error] ${msg}\n`);
  process.exitCode = 2;
}
