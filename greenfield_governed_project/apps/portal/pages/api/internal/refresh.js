/*
PROV: GREENFIELD.GOV.PORTAL_REFRESH.01
REQ: SYS-ARCH-15
WHY: Allow the portal "Refresh" button to run generation so the portal feed stays current.
*/

import path from "node:path";
import { spawn } from "node:child_process";

function repoRootFromPortalCwd() {
  return path.resolve(process.cwd(), "..", "..");
}

function allowRefresh() {
  if (process.env.PORTAL_ALLOW_GENERATE === "1") return true;
  return process.env.NODE_ENV !== "production";
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

  const repoRoot = repoRootFromPortalCwd();

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

  if (exitCode !== 0) {
    res.status(500).json({ ok: false, error: "generate_failed", exit_code: exitCode, stdout, stderr });
    return;
  }

  res.status(200).json({ ok: true, exit_code: exitCode, stdout, stderr });
}

