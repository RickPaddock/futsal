/*
PROV: GREENFIELD.GOV.WORKFLOW.COPILOT_OPEN_PROMPT.01
REQ: SYS-ARCH-15
WHY: Generate an intent-scoped Copilot prompt file (prefilled) and open it in VS Code for the user to paste into Copilot Chat.
*/

import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

import { repoRootFromHere, relPosix } from "../lib/paths.mjs";

function readJsonSafe(p) {
  try {
    if (!fs.existsSync(p)) return null;
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return null;
  }
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function writeText(p, text) {
  ensureDir(path.dirname(p));
  fs.writeFileSync(p, text, "utf8");
}

function utcDate() {
  return new Date().toISOString().slice(0, 10);
}

function utcRunId() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}_${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}`;
}

function parseArgs(argv) {
  const out = { kind: "", intentId: "", runId: "", closedDate: "" };
  const args = [...argv];
  while (args.length) {
    const a = args.shift();
    if (a === "--kind") out.kind = String(args.shift() || "");
    else if (a === "--intent-id") out.intentId = String(args.shift() || "");
    else if (a === "--run-id") out.runId = String(args.shift() || "");
    else if (a === "--closed-date") out.closedDate = String(args.shift() || "");
  }
  return out;
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

  process.stdout.write(`[prompt] unable to auto-open; tried: ${tried.map((t) => `${t.cmd}(${t.status})`).join(", ")}\n`);
  return false;
}

function substitute(text, vars) {
  let out = String(text);
  for (const [k, v] of Object.entries(vars)) {
    out = out.replaceAll(`<${k}>`, String(v));
  }
  return out;
}

function getActiveIntentId(repoRoot) {
  const state = readJsonSafe(path.join(repoRoot, "status", "wizard", "ACTIVE.json"));
  const id = String(state?.intent_id || "").trim();
  return id || "";
}

function resolveKind(kind) {
  const k = String(kind || "").trim().toLowerCase();
  if (k === "quality-audit") {
    return { templateRel: "spec/prompts/intent_quality_audit.prompt.txt", outName: "intent_quality_audit.prompt.txt" };
  }
  if (k === "close-intent") {
    return { templateRel: "spec/prompts/intent_close_end_to_end.prompt.txt", outName: "intent_close_end_to_end.prompt.txt" };
  }
  throw new Error("Usage: node scripts/workflows/copilot_open_prompt.mjs --kind quality-audit|close-intent [--intent-id INT-###] [--run-id <id>] [--closed-date YYYY-MM-DD]");
}

function main() {
  const repoRoot = repoRootFromHere(import.meta.url);
  const args = parseArgs(process.argv.slice(2));
  const { templateRel, outName } = resolveKind(args.kind);

  const intentId = (args.intentId || getActiveIntentId(repoRoot)).trim();
  if (!intentId) throw new Error("missing_intent_id (provide --intent-id or create an active wizard session)");
  if (!/^INT-\d{3}$/.test(intentId)) throw new Error(`invalid_intent_id:${intentId}`);

  const runId = (args.runId || utcRunId()).trim();
  const closedDate = (args.closedDate || utcDate()).trim();

  const templateAbs = path.join(repoRoot, templateRel);
  if (!fs.existsSync(templateAbs)) throw new Error(`missing_template:${templateRel}`);
  const raw = fs.readFileSync(templateAbs, "utf8");

  const rendered = substitute(raw, { INTENT_ID: intentId, run_id: runId, closed_date: closedDate });
  const outAbs = path.join(repoRoot, "status", "wizard", intentId, outName);
  writeText(outAbs, rendered);

  const rel = relPosix(path.relative(repoRoot, outAbs));
  process.stdout.write(`[prompt] wrote ${rel}\n`);
  if (!openInEditor(outAbs)) {
    process.stdout.write(`[prompt] open manually: ${rel}\n`);
  }
}

try {
  main();
} catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
  process.stderr.write(`[prompt:error] ${msg}\n`);
  process.exit(2);
}

