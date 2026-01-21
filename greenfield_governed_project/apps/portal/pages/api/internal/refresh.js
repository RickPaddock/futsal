/*
PROV: GREENFIELD.GOV.PORTAL_REFRESH.01
REQ: SYS-ARCH-15, GREENFIELD-PORTAL-004, GREENFIELD-PORTAL-012, GREENFIELD-PORTAL-014, GREENFIELD-PORTAL-015
WHY: Allow the portal "Refresh" button to run generation so the portal feed stays current (same-origin hardened + evidence capture).
*/

import path from "node:path";
import fs from "node:fs";
import { spawn } from "node:child_process";
import { isValidIntentId, isValidRunId, relPosix } from "../../../lib/portal_read_model.js";

function repoRootFromPortalCwd() {
  return path.resolve(process.cwd(), "..", "..");
}

function allowRefresh() {
  if (process.env.PORTAL_ALLOW_GENERATE === "1") return true;
  return process.env.NODE_ENV !== "production";
}

function utcNow() {
  return new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
}

function utcRunId() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}_${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}`;
}

function isSameOrigin(req) {
  const origin = String(req.headers.origin || "");
  if (!origin) return true;
  const host = String(req.headers.host || "");
  if (!host) return false;
  return origin === `http://${host}` || origin === `https://${host}`;
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

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function writeRunEvidence({ repoRoot, intentId, runId, stage, command, args, exitCode }) {
  const safeIntent = isValidIntentId(intentId) ? intentId : "PORTAL";
  const safeRunId = isValidRunId(runId) ? runId : utcRunId();
  const outRel = path.join("status", "audit", safeIntent, "runs", safeRunId, stage, "run.json");
  const outAbs = path.join(repoRoot, outRel);
  ensureDir(path.dirname(outAbs));
  const payload = {
    run_id: safeRunId,
    timestamp_start: null,
    timestamp_end: null,
    command,
    args,
    exit_code: exitCode,
    cwd: ".",
    actor: "portal",
    intent_id: isValidIntentId(intentId) ? intentId : undefined,
    artefacts: [],
    stage,
    run_json_path: relPosix(outRel),
  };
  fs.writeFileSync(outAbs, JSON.stringify(payload, null, 2) + "\n", "utf8");
  return { outRel: relPosix(outRel), outAbs };
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

  if (!isSameOrigin(req)) {
    res.status(403).json({ ok: false, error: "cross_origin_blocked" });
    return;
  }

  const repoRoot = repoRootFromPortalCwd();
  const body = readBodyJson(req);
  const intentId = String(body.intentId || "").trim();
  const runId = utcRunId();
  const stage = "portal_refresh";
  const cmd = "npm run generate";
  const args = ["npm", "run", "generate"];
  const evidence = writeRunEvidence({ repoRoot, intentId, runId, stage, command: cmd, args, exitCode: 1 });
  const evidenceAbs = evidence.outAbs;
  const timestampStart = utcNow();

  const child = spawn("npm", ["run", "generate"], {
    cwd: repoRoot,
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"],
  });

  let stdout = "";
  let stderr = "";
  const MAX = 200_000;
  child.stdout.on("data", (d) => {
    stdout += d.toString("utf8");
    if (stdout.length > MAX) stdout = stdout.slice(stdout.length - MAX);
  });
  child.stderr.on("data", (d) => {
    stderr += d.toString("utf8");
    if (stderr.length > MAX) stderr = stderr.slice(stderr.length - MAX);
  });

  const exitCode = await new Promise((resolve) => {
    child.on("close", (code) => resolve(code ?? 1));
  });

  const timestampEnd = utcNow();
  // Update evidence with timestamps + final exit code.
  try {
    const payload = JSON.parse(fs.readFileSync(evidenceAbs, "utf8"));
    payload.timestamp_start = timestampStart;
    payload.timestamp_end = timestampEnd;
    payload.exit_code = exitCode;
    fs.writeFileSync(evidenceAbs, JSON.stringify(payload, null, 2) + "\n", "utf8");
  } catch {
    // If evidence update fails, keep the initial record.
  }

  if (exitCode !== 0) {
    res.status(500).json({ ok: false, error: "generate_failed", run_id: runId, evidence_run_json: relPosix(evidence.outRel), stdout, stderr });
    return;
  }

  res.status(200).json({ ok: true, run_id: runId, evidence_run_json: relPosix(evidence.outRel), stdout });
}

