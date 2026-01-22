/*
PROV: GREENFIELD.GOV.PORTAL_REFRESH.01
REQ: SYS-ARCH-15, GREENFIELD-PORTAL-004, GREENFIELD-PORTAL-012, GREENFIELD-PORTAL-014, GREENFIELD-PORTAL-015, GREENFIELD-PORTAL-024, GREENFIELD-EVIDENCE-002
WHY: Allow the portal "Refresh" button to run generation so the portal feed stays current (CSRF + host allowlist + evidence capture via shared recorder).
*/

import path from "node:path";
import fs from "node:fs";
import { spawnSync } from "node:child_process";
import { isValidIntentId, isValidRunId, relPosix } from "../../../lib/portal_read_model.js";
import { csrfOk } from "../../../lib/portal_csrf.js";
import { rejectIfNotInternal } from "../../../lib/portal_request_guards.js";
import { utcRunId } from "../../../../../scripts/lib/time.mjs";

function repoRootFromPortalCwd() {
  return path.resolve(process.cwd(), "..", "..");
}

function allowRefresh() {
  const allowExplicit = process.env.PORTAL_ALLOW_GENERATE === "1";
  if (allowExplicit) return true;
  if (process.env.NODE_ENV === "production") return false;
  return true;
}

function readBodyJson(req) {
  const b = req.body;
  if (!b) return {};
  if (typeof b === "object") return b;
  try {
    return JSON.parse(String(b));
  } catch {
    return {};
  }
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ ok: false, error: "method_not_allowed" });
    return;
  }

  if (!allowRefresh()) {
    res.status(403).json({ ok: false, error: "refresh_disabled" });
    return;
  }

  if (rejectIfNotInternal(req, res)) return;

  if (!csrfOk(req)) {
    res.status(403).json({ ok: false, error: "csrf_missing_or_invalid" });
    return;
  }

  const repoRoot = repoRootFromPortalCwd();
  const body = readBodyJson(req);
  const intentId = String(body.intentId || "").trim();
  const runId = utcRunId();
  const stage = "portal_refresh";
  const safeIntent = isValidIntentId(intentId) ? intentId : "PORTAL";
  const evidenceRel = path.join("status", "audit", safeIntent, "runs", runId, stage, "run.json");
  const evidenceAbs = path.join(repoRoot, evidenceRel);
  fs.mkdirSync(path.dirname(evidenceAbs), { recursive: true });

  const cmd = [
    "node",
    "tools/evidence/record_run.mjs",
    "--intent-id",
    safeIntent,
    "--actor",
    "portal",
    "--out",
    evidenceRel,
    "npm",
    "run",
    "generate",
  ];

  const result = spawnSync(cmd[0], cmd.slice(1), { cwd: repoRoot, env: process.env, stdio: ["ignore", "pipe", "pipe"] });
  const stdout = result.stdout ? result.stdout.toString("utf8") : "";
  const stderr = result.stderr ? result.stderr.toString("utf8") : "";

  const rep = fs.existsSync(evidenceAbs) ? JSON.parse(fs.readFileSync(evidenceAbs, "utf8")) : null;
  const exitCode = Number(rep?.exit_code ?? result.status ?? 1);

  if (exitCode !== 0) {
    res.status(500).json({
      ok: false,
      error: "generate_failed",
      run_id: runId,
      evidence_run_json: relPosix(evidenceRel),
      stdout: rep?.stdout || stdout,
      stderr: rep?.stderr || stderr,
    });
    return;
  }

  res.status(200).json({ ok: true, run_id: runId, evidence_run_json: relPosix(evidenceRel), stdout: rep?.stdout || stdout });
}
