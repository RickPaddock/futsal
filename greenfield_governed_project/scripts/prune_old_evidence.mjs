/*
PROV: GREENFIELD.OPS.EVIDENCE_RETENTION.01
REQ: SYS-ARCH-15, GREENFIELD-OPS-003
WHY: Keep audit evidence bounded by pruning older run folders deterministically (optional ops helper).
*/

import fs from "node:fs";
import path from "node:path";

import { repoRootFromHere, relPosix } from "./lib/paths.mjs";

function parseArgs(argv) {
  const out = { keep: 10, apply: false, intentId: "" };
  const args = [...argv];
  while (args.length) {
    const a = args.shift();
    if (a === "--keep") out.keep = Number.parseInt(String(args.shift() || "10"), 10);
    else if (a === "--apply") out.apply = true;
    else if (a === "--intent-id") out.intentId = String(args.shift() || "").trim();
  }
  return out;
}

function isValidRunDirName(name) {
  return /^\d{8}_\d{6}$/.test(String(name || ""));
}

function listRunDirs(runsDirAbs) {
  if (!fs.existsSync(runsDirAbs)) return [];
  return fs
    .readdirSync(runsDirAbs, { withFileTypes: true })
    .filter((d) => d.isDirectory() && isValidRunDirName(d.name))
    .map((d) => d.name)
    .sort();
}

function pruneRuns({ runsDirAbs, keep, apply }) {
  const dirs = listRunDirs(runsDirAbs);
  const toDelete = dirs.length > keep ? dirs.slice(0, dirs.length - keep) : [];
  for (const runId of toDelete) {
    const abs = path.join(runsDirAbs, runId);
    const rel = relPosix(path.relative(REPO_ROOT, abs));
    if (apply) {
      fs.rmSync(abs, { recursive: true, force: true });
      process.stdout.write(`[prune] deleted ${rel}\n`);
    } else {
      process.stdout.write(`[prune] would_delete ${rel}\n`);
    }
  }
  return { total: dirs.length, kept: Math.min(dirs.length, keep), deleted: toDelete.length };
}

const REPO_ROOT = repoRootFromHere(import.meta.url);
const args = parseArgs(process.argv.slice(2));
if (!Number.isFinite(args.keep) || args.keep < 0) {
  process.stderr.write("Invalid --keep (expected non-negative integer)\n");
  process.exit(2);
}

const auditRoot = path.join(REPO_ROOT, "status", "audit");
if (!fs.existsSync(auditRoot)) {
  process.stdout.write("[prune] no status/audit folder; nothing to do\n");
  process.exit(0);
}

const intentDirs = fs.readdirSync(auditRoot, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name).sort();
const targets = args.intentId ? intentDirs.filter((x) => x === args.intentId) : intentDirs;
if (args.intentId && targets.length === 0) {
  process.stderr.write(`Unknown intent_id under status/audit: ${args.intentId}\n`);
  process.exit(2);
}

let totals = { intents: 0, total: 0, kept: 0, deleted: 0 };
for (const intentId of targets) {
  const runsDirAbs = path.join(auditRoot, intentId, "runs");
  if (!fs.existsSync(runsDirAbs)) continue;
  const res = pruneRuns({ runsDirAbs, keep: args.keep, apply: args.apply });
  totals = { intents: totals.intents + 1, total: totals.total + res.total, kept: totals.kept + res.kept, deleted: totals.deleted + res.deleted };
}

process.stdout.write(`[prune] ok intents=${totals.intents} total_runs=${totals.total} kept=${totals.kept} deleted=${totals.deleted} apply=${args.apply}\n`);
