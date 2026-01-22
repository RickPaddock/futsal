/*
PROV: FUSBAL.SCRIPTS.DEV.HOPPSCOTCH.01
REQ: SYS-ARCH-15
WHY: Provide a local-only, dev-friendly Hoppscotch API client launcher with no Docker dependency.
*/

import { spawnSync } from "node:child_process";
import os from "node:os";
import path from "node:path";
import fs from "node:fs";

function hasCmd(cmd) {
  const r = spawnSync(cmd, ["--version"], { stdio: ["ignore", "pipe", "pipe"] });
  return r.status === 0;
}

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { stdio: "inherit", ...opts });
  return r.status ?? 1;
}

function runCapture(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { stdio: ["ignore", "pipe", "pipe"], ...opts });
  return {
    status: r.status ?? 1,
    stdout: r.stdout ? r.stdout.toString("utf8") : "",
    stderr: r.stderr ? r.stderr.toString("utf8") : "",
  };
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function readJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

const REPO_URL = process.env.HOPPSCOTCH_REPO_URL || "https://github.com/hoppscotch/hoppscotch.git";
const REF = process.env.HOPPSCOTCH_REF || "main";
const PORT = Number.parseInt(process.env.HOPPSCOTCH_PORT || "3001", 10);
const HOST = process.env.HOPPSCOTCH_HOST || "127.0.0.1";
const CACHE_DIR = process.env.HOPPSCOTCH_DIR || path.join(os.homedir(), ".cache", "fusbal", "hoppscotch");
const SKIP_INSTALL = process.env.HOPPSCOTCH_SKIP_INSTALL === "1";

if (!Number.isFinite(PORT) || PORT <= 0 || PORT > 65535) {
  process.stderr.write(`[hoppscotch:dev] invalid port: ${process.env.HOPPSCOTCH_PORT}\n`);
  process.exit(2);
}

if (!hasCmd("git")) {
  process.stderr.write("[hoppscotch:dev] git not found. Install git and retry.\n");
  process.exit(2);
}

if (!hasCmd("corepack")) {
  process.stderr.write("[hoppscotch:dev] corepack not found. Install a Node.js version that includes corepack and retry.\n");
  process.exit(2);
}

process.stdout.write(`[hoppscotch:dev] local install dir: ${CACHE_DIR}\n`);
ensureDir(path.dirname(CACHE_DIR));

if (!fs.existsSync(CACHE_DIR) || !fs.existsSync(path.join(CACHE_DIR, ".git"))) {
  process.stdout.write(`[hoppscotch:dev] cloning ${REPO_URL} (${REF})...\n`);
  const cloneRc = run("git", ["clone", "--depth", "1", "--branch", REF, REPO_URL, CACHE_DIR]);
  if (cloneRc !== 0) process.exit(cloneRc);
} else {
  process.stdout.write(`[hoppscotch:dev] updating ${REF}...\n`);
  const fetch = run("git", ["-C", CACHE_DIR, "fetch", "--depth", "1", "origin", REF]);
  if (fetch !== 0) process.exit(fetch);
  const reset = run("git", ["-C", CACHE_DIR, "reset", "--hard", `origin/${REF}`]);
  if (reset !== 0) process.exit(reset);
}

// Resolve the repo's preferred pnpm version from package.json (packageManager: "pnpm@X.Y.Z").
let pnpmVersion = "10.23.0";
try {
  const pkg = readJson(path.join(CACHE_DIR, "package.json"));
  const pm = String(pkg.packageManager || "");
  const m = pm.match(/^pnpm@(.+)$/);
  if (m) pnpmVersion = m[1];
} catch {
  // keep default
}

run("corepack", ["enable"]);
process.stdout.write(`[hoppscotch:dev] activating pnpm@${pnpmVersion} via corepack...\n`);
{
  const r = run("corepack", ["prepare", `pnpm@${pnpmVersion}`, "--activate"]);
  if (r !== 0) process.exit(r);
}

if (!SKIP_INSTALL) {
  process.stdout.write("[hoppscotch:dev] pnpm install (this may take a while the first time)...\n");
  const hasLockfile = fs.existsSync(path.join(CACHE_DIR, "pnpm-lock.yaml"));
  const installArgs = hasLockfile ? ["install", "--frozen-lockfile"] : ["install", "--no-frozen-lockfile"];
  const r = run("pnpm", installArgs, { cwd: CACHE_DIR });
  if (r !== 0) process.exit(r);
} else {
  process.stdout.write("[hoppscotch:dev] skipping install (HOPPSCOTCH_SKIP_INSTALL=1)\n");
}

process.stdout.write(`[hoppscotch:dev] starting Hoppscotch web on http://${HOST}:${PORT}\n`);
process.stdout.write("[hoppscotch:dev] point it at the portal base URL: http://localhost:3015\n");
process.stdout.write("[hoppscotch:dev] note: portal write endpoints enforce same-origin + CSRF and will fail cross-origin.\n");

const exitCode = run(
  "pnpm",
  ["--filter", "@hoppscotch/selfhost-web", "exec", "vite", "--host", HOST, "--port", String(PORT)],
  { cwd: CACHE_DIR },
);
process.exit(exitCode);
