/*
PROV: GREENFIELD.PORTAL.PROMPT_API.01
REQ: SYS-ARCH-15, GREENFIELD-PORTAL-011, GREENFIELD-PORTAL-015, GREENFIELD-PORTAL-020
WHY: Serve rendered prompt templates via a stable API with explicit substitution validation.
*/

import path from "node:path";
import fs from "node:fs";
import {
  repoRootFromPortalCwd,
  safeReadJson,
  readText,
  substituteTemplate,
  findUnsubstitutedPlaceholders,
  isValidIntentId,
  isValidRunId,
  isIsoDate,
} from "../../../lib/portal_read_model.js";

function utcRunId() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}_${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}`;
}

function utcDate() {
  return new Date().toISOString().slice(0, 10);
}

function nextNumericSuffix(existingIds, prefix, pad) {
  let max = 0;
  for (const id of existingIds) {
    if (!String(id || "").startsWith(prefix)) continue;
    const raw = String(id || "").slice(prefix.length);
    const n = Number.parseInt(raw, 10);
    if (Number.isFinite(n)) max = Math.max(max, n);
  }
  const next = max + 1;
  return `${prefix}${String(next).padStart(pad, "0")}`;
}

function loadRequirementsBundle({ repoRoot, requirementsSourceRel }) {
  const abs = path.join(repoRoot, requirementsSourceRel);
  const root = safeReadJson(abs);
  if (!root) return { requirements: [], areaFilesRel: [] };

  if (root?.type === "requirements_index") {
    const files = Array.isArray(root.files) ? root.files.map(String) : [];
    const requirements = [];
    for (const rel of files) {
      const area = safeReadJson(path.join(repoRoot, rel));
      for (const r of area?.requirements || []) requirements.push(r);
    }
    return { requirements, areaFilesRel: files };
  }

  if (Array.isArray(root?.requirements)) {
    return { requirements: root.requirements, areaFilesRel: [requirementsSourceRel] };
  }

  return { requirements: [], areaFilesRel: [] };
}

function listAreaChoices(areaFilesRel) {
  return (Array.isArray(areaFilesRel) ? areaFilesRel : []).map((rel) => ({ label: path.basename(rel, ".json"), rel }));
}

function badRequest(res, error, details) {
  res.status(400).json({ ok: false, error, details: details || null });
}

export default async function handler(req, res) {
  if (req.method !== "GET") {
    res.status(405).json({ ok: false, error: "method_not_allowed" });
    return;
  }

  const kind = String(req.query?.kind || "").trim();
  const intentIdRaw = String(req.query?.intentId || "").trim();
  const runId = String(req.query?.runId || "").trim();
  const closedDate = String(req.query?.closedDate || "").trim();

  const repoRoot = repoRootFromPortalCwd();

  const templateByKind = {
    create: "intent_create_end_to_end.prompt.txt",
    preflight: "intent_preflight_review.prompt.txt",
    implement: "intent_implement_end_to_end.prompt.txt",
    audit: "intent_quality_audit.prompt.txt",
    close: "intent_close_end_to_end.prompt.txt",
  };

  if (!Object.prototype.hasOwnProperty.call(templateByKind, kind)) {
    badRequest(res, "invalid_kind", { allowed: Object.keys(templateByKind) });
    return;
  }

  // Resolve intent id (create can be provided or derived).
  let intentId = intentIdRaw;
  if (kind !== "create") {
    if (!isValidIntentId(intentId)) {
      badRequest(res, "invalid_intent_id", { expected: "INT-###" });
      return;
    }
  } else {
    if (intentId && !isValidIntentId(intentId)) {
      badRequest(res, "invalid_intent_id", { expected: "INT-###" });
      return;
    }
    if (!intentId) {
      const intentsDir = path.join(repoRoot, "spec", "intents");
      const intentIds = fs.existsSync(intentsDir)
        ? fs.readdirSync(intentsDir, { withFileTypes: true }).filter((d) => d.isFile() && d.name.endsWith(".json")).map((d) => d.name.replace(/\.json$/, "")).sort()
        : [];
      intentId = nextNumericSuffix(intentIds, "INT-", 3);
    }
  }

  const finalRunId = runId || utcRunId();
  if ((kind === "implement" || kind === "audit" || kind === "close") && !isValidRunId(finalRunId)) {
    badRequest(res, "invalid_run_id", { expected: "YYYYMMDD_HHMMSS" });
    return;
  }

  const finalClosedDate = closedDate || utcDate();
  if (kind === "close" && !isIsoDate(finalClosedDate)) {
    badRequest(res, "invalid_closed_date", { expected: "YYYY-MM-DD" });
    return;
  }

  const templatePath = path.join(repoRoot, "spec", "prompts", templateByKind[kind]);
  const template = readText(templatePath);
  if (!template) {
    res.status(500).json({ ok: false, error: "template_missing", details: { templatePath } });
    return;
  }

  const project = safeReadJson(path.join(repoRoot, "spec", "project.json")) || {};
  const requirementsSourceRel = String(project.requirements_source || "").trim();
  const governance = project.governance || {};
  const newReqPrefix = Array.isArray(governance.new_requirement_id_prefixes) ? String(governance.new_requirement_id_prefixes[0] || "REQ-") : "REQ-";
  const requirementsBundle = requirementsSourceRel ? loadRequirementsBundle({ repoRoot, requirementsSourceRel }) : { areaFilesRel: [] };
  const areaChoices = listAreaChoices(requirementsBundle.areaFilesRel || []);
  const areasList = areaChoices.map((a) => `- ${a.label}: \`${a.rel}\``).join("\n");

  const intentSpec = intentId ? safeReadJson(path.join(repoRoot, "spec", "intents", `${intentId}.json`)) : null;
  const intentTitle = String(intentSpec?.title || "").trim();

  const vars = {
    INTENT_ID: intentId,
    INTENT_TITLE: intentTitle,
    run_id: finalRunId,
    closed_date: finalClosedDate,
    TODAY: utcDate(),
    new_req_prefix: newReqPrefix,
    requirements_source: requirementsSourceRel,
    requirements_areas: areasList || "(none found)",
  };

  const text = substituteTemplate(template, vars);
  const leftovers = findUnsubstitutedPlaceholders(text);
  if (leftovers.length) {
    res.status(500).json({ ok: false, error: "unsubstituted_placeholders", details: { leftovers } });
    return;
  }

  res.status(200).json({ ok: true, kind, intent_id: intentId || null, run_id: finalRunId, closed_date: kind === "close" ? finalClosedDate : null, text });
}
