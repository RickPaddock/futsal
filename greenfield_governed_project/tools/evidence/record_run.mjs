/*
PROV: GREENFIELD.SCAFFOLD.EVIDENCE.01
REQ: SYS-ARCH-15, GREENFIELD-EVIDENCE-001
WHY: Run a command and write a deterministic run.json evidence record with stdout/stderr capture.
*/

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { repoRootFromHere, relPosix } from "../../scripts/lib/paths.mjs";

function sha256File(p) {
  const buf = fs.readFileSync(p);
  return `sha256:${crypto.createHash("sha256").update(buf).digest("hex")}`;
}

function parseArgs(argv) {
  const out = { out: "", intentId: "", actor: "human", artefacts: [], cmd: [] };
  const args = [...argv];
  while (args.length) {
    const a = args.shift();
    if (a === "--out") out.out = String(args.shift() || "");
    else if (a === "--intent-id") out.intentId = String(args.shift() || "");
    else if (a === "--actor") out.actor = String(args.shift() || "");
    else if (a === "--artefact") out.artefacts.push(String(args.shift() || ""));
    else {
      out.cmd = [a, ...args].filter(Boolean);
      break;
    }
  }
  return out;
}

function utcNow() {
  return new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function collectArtefacts(repoRoot, artefacts) {
  const out = [];
  for (const a of artefacts) {
    const abs = path.isAbsolute(a) ? a : path.join(repoRoot, a);
    if (!fs.existsSync(abs) || fs.statSync(abs).isDirectory()) continue;
    out.push({ path: relPosix(path.relative(repoRoot, abs)), sha256: sha256File(abs) });
  }
  return out;
}

const repoRoot = repoRootFromHere(import.meta.url);
const args = parseArgs(process.argv.slice(2));
if (!args.out || !args.out.endsWith("run.json")) {
  process.stderr.write("Usage: node tools/evidence/record_run.mjs --out status/audit/INT-001/runs/<id>/run.json -- <cmd>\n");
  process.exit(2);
}
if (!args.cmd.length) {
  process.stderr.write("Missing command\n");
  process.exit(2);
}

const outPath = path.isAbsolute(args.out) ? args.out : path.join(repoRoot, args.out);
const timestampStart = utcNow();
const res = spawnSync(args.cmd[0], args.cmd.slice(1), { cwd: repoRoot, stdio: ["inherit", "pipe", "pipe"] });
const timestampEnd = utcNow();

const MAX_OUTPUT = 50_000;
let stdout = res.stdout ? res.stdout.toString("utf8") : "";
let stderr = res.stderr ? res.stderr.toString("utf8") : "";
if (stdout.length > MAX_OUTPUT) stdout = stdout.slice(stdout.length - MAX_OUTPUT);
if (stderr.length > MAX_OUTPUT) stderr = stderr.slice(stderr.length - MAX_OUTPUT);

// Print captured output to console so it's still visible
if (stdout) process.stdout.write(stdout);
if (stderr) process.stderr.write(stderr);

ensureDir(path.dirname(outPath));
const payload = {
  run_id: path.basename(path.dirname(outPath)),
  timestamp_start: timestampStart,
  timestamp_end: timestampEnd,
  command: args.cmd.join(" "),
  args: args.cmd,
  exit_code: res.status ?? 1,
  cwd: ".",
  actor: args.actor,
  intent_id: args.intentId || undefined,
  artefacts: collectArtefacts(repoRoot, args.artefacts),
};
if (stdout) payload.stdout = stdout;
if (stderr) payload.stderr = stderr;
fs.writeFileSync(outPath, JSON.stringify(payload, null, 2) + "\n", "utf8");
process.exit(res.status ?? 1);

