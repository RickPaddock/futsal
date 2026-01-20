/*
PROV: GREENFIELD.SCAFFOLD.PORTAL.04
REQ: SYS-ARCH-15, GREENFIELD-PORTAL-002
WHY: List intents and provide copy-ready prompt overlays for create/implement/audit/close flows.
*/

import fs from "node:fs";
import path from "node:path";
import Link from "next/link";
import { useRouter } from "next/router";
import { useState } from "react";

function safeReadJson(p) {
  if (!fs.existsSync(p)) return null;
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function readText(p) {
  if (!fs.existsSync(p)) return "";
  return fs.readFileSync(p, "utf8");
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

function listAuditRuns(repoRoot, intentId) {
  const runsDir = path.join(repoRoot, "status", "audit", intentId, "runs");
  if (!fs.existsSync(runsDir)) return [];
  const runs = [];
  for (const d of fs.readdirSync(runsDir, { withFileTypes: true }).filter((x) => x.isDirectory()).map((x) => x.name)) {
    const runPath = path.join(runsDir, d, "run.json");
    const run = safeReadJson(runPath);
    if (run) runs.push(run);
  }
  runs.sort((a, b) => String(b.timestamp_end || "").localeCompare(String(a.timestamp_end || "")));
  return runs;
}

function isIntentAuditRun(run) {
  const cmd = String(run?.command || "");
  return cmd.includes("audit:intent") || cmd.includes("scripts/audit/audit_intent.mjs");
}

const STATUS_ORDER = {
  todo: 0,
  draft: 1,
  closed: 2,
  unknown: 3,
};

export async function getServerSideProps() {
  const repoRoot = path.resolve(process.cwd(), "..", "..");
  const feedPath = path.join(repoRoot, "status", "portal", "internal_intents.json");
  const feed = safeReadJson(feedPath) || { intents: [] };

  const project = safeReadJson(path.join(repoRoot, "spec", "project.json")) || {};
  const requirementsSourceRel = String(project.requirements_source || "").trim();
  const governance = project.governance || {};
  const newReqPrefix = Array.isArray(governance.new_requirement_id_prefixes) ? String(governance.new_requirement_id_prefixes[0] || "REQ-") : "REQ-";

  const intentsDir = path.join(repoRoot, "spec", "intents");
  const intentIds = fs.existsSync(intentsDir)
    ? fs.readdirSync(intentsDir, { withFileTypes: true }).filter((d) => d.isFile() && d.name.endsWith(".json")).map((d) => d.name.replace(/\.json$/, "")).sort()
    : [];
  const nextIntentId = nextNumericSuffix(intentIds, "INT-", 3);

  const requirementsBundle = requirementsSourceRel ? loadRequirementsBundle({ repoRoot, requirementsSourceRel }) : { areaFilesRel: [] };
  const areaChoices = listAreaChoices(requirementsBundle.areaFilesRel || []);
  const areasList = areaChoices.map((a) => `- ${a.label}: \`${a.rel}\``).join("\n");
  const createTemplate = readText(path.join(repoRoot, "spec", "prompts", "intent_create_end_to_end.prompt.txt"));
  const createPrompt = substitute(createTemplate, {
    INTENT_ID: nextIntentId,
    new_req_prefix: newReqPrefix,
    requirements_source: requirementsSourceRel,
    requirements_areas: areasList || "(none found)",
  });

  const qualityAuditTemplate = readText(path.join(repoRoot, "spec", "prompts", "intent_quality_audit.prompt.txt"));
  const closeIntentTemplate = readText(path.join(repoRoot, "spec", "prompts", "intent_close_end_to_end.prompt.txt"));
  const implementIntentTemplate = readText(path.join(repoRoot, "spec", "prompts", "intent_implement_end_to_end.prompt.txt"));

  const intents = (feed.intents || []).map((i) => {
    const runs = listAuditRuns(repoRoot, i.intent_id);
    const latestAudit = runs.find(isIntentAuditRun) || null;
    const audit = latestAudit
      ? { timestamp_end: latestAudit.timestamp_end || "", exit_code: latestAudit.exit_code ?? 1, command: latestAudit.command || "" }
      : null;
    return { ...i, audit };
  }).sort((a, b) => {
    const as = String(a.status || "unknown").toLowerCase();
    const bs = String(b.status || "unknown").toLowerCase();
    const ao = STATUS_ORDER[as] ?? STATUS_ORDER.unknown;
    const bo = STATUS_ORDER[bs] ?? STATUS_ORDER.unknown;
    if (ao !== bo) return ao - bo;
    return String(a.intent_id).localeCompare(String(b.intent_id));
  });
  return { props: { intents, createPrompt, nextIntentId, qualityAuditTemplate, closeIntentTemplate, implementIntentTemplate } };
}

function utcRunId() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}_${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}`;
}

function utcDate() {
  return new Date().toISOString().slice(0, 10);
}

function substitute(text, vars) {
  let out = String(text || "");
  for (const [k, v] of Object.entries(vars)) out = out.replaceAll(`<${k}>`, String(v));
  return out;
}

async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

function PromptOverlay({ title, text, onClose }) {
  const [copied, setCopied] = useState(false);
  async function onCopy() {
    const ok = await copyToClipboard(text);
    if (!ok) return;
    setCopied(true);
    setTimeout(() => setCopied(false), 800);
  }

  return (
    <div className="overlay" role="dialog" aria-modal="true">
      <div className="modal">
        <div className="modalHeader">
          <div>
            <div className="muted">Prompt (copy/paste into your LLM)</div>
            <h2 style={{ margin: "6px 0 0 0" }}>{title}</h2>
          </div>
          <div className="modalActions">
            <button className="btn" type="button" onClick={onCopy}>{copied ? "Copied" : "Copy"}</button>
            <button className="btn" type="button" onClick={onClose}>Close</button>
          </div>
        </div>
        <textarea className="promptBox" readOnly value={text} />
      </div>
    </div>
  );
}

export default function IntentsIndex({ intents, createPrompt, nextIntentId, qualityAuditTemplate, closeIntentTemplate, implementIntentTemplate }) {
  const router = useRouter();
  const [refreshing, setRefreshing] = useState(false);
  const [overlay, setOverlay] = useState(null);

  async function refreshAndReload() {
    setRefreshing(true);
    try {
      const res = await fetch("/api/internal/refresh", { method: "POST" });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg = payload?.error ? String(payload.error) : `http_${res.status}`;
        throw new Error(msg);
      }
      router.reload();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      alert(`Refresh failed: ${msg}`);
    } finally {
      setRefreshing(false);
    }
  }

  function openCreatePrompt() {
    setOverlay({ title: `Create intent prompt (${nextIntentId})`, text: createPrompt });
  }

  function openAuditPrompt(intentId) {
    const runId = utcRunId();
    const text = substitute(qualityAuditTemplate, { INTENT_ID: intentId, run_id: runId });
    setOverlay({ title: `Audit + quality audit prompt (${intentId})`, text });
  }

  function openImplementPrompt(intentId) {
    const runId = utcRunId();
    const text = substitute(implementIntentTemplate, { INTENT_ID: intentId, run_id: runId });
    setOverlay({ title: `Implement intent prompt (${intentId})`, text });
  }

  function openClosePrompt(intentId) {
    const runId = utcRunId();
    const closedDate = utcDate();
    const text = substitute(closeIntentTemplate, { INTENT_ID: intentId, run_id: runId, closed_date: closedDate });
    setOverlay({ title: `Close intent prompt (${intentId})`, text });
  }

  return (
    <main className="page">
      <div className="toolbar">
        <div>
          <div className="muted">
            <Link href="/internal">Internal</Link> · <Link href="/internal/tasks">Tasks</Link>
          </div>
          <h1 style={{ margin: "6px 0 0 0" }}>Intents</h1>
        </div>
        <div className="toolbarActions">
          <button className="btn" type="button" onClick={openCreatePrompt}>
            Create prompt
          </button>
          <button className="btn" type="button" disabled={refreshing} onClick={refreshAndReload}>
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>
      <div className="list">
        {intents.map((i) => {
          const intentId = i.intent_id;
          const href = `/internal/intents/${encodeURIComponent(intentId)}`;
          const showOps = String(i.status || "").toLowerCase() === "todo";
          return (
            <div key={intentId} className="card">
              <div className="row">
                <div className="rowLeft">
                  <Link href={href} className="cardLink"><strong>{intentId}</strong></Link>
                  <span className="badge">{i.status}</span>
                </div>
                <div className="rowRight">
                  {showOps ? (
                    <>
                      <button className="btn btnSmall" type="button" onClick={() => openImplementPrompt(intentId)}>Implement</button>
                      <button className="btn btnSmall" type="button" onClick={() => openAuditPrompt(intentId)}>Audit</button>
                      <button className="btn btnSmall" type="button" onClick={() => openClosePrompt(intentId)}>Close</button>
                    </>
                  ) : null}
                </div>
              </div>
              <Link href={href} className="cardBody">
                <div className="muted">{i.title}</div>
                <div className="meta">
                  <span>{(i.tasks || []).filter((t) => t.status === "done").length}/{(i.tasks || []).length} tasks done</span>
                  <span>{(i.requirements_in_scope || []).filter((r) => r.tracking_implementation === "done").length}/{(i.requirements_in_scope || []).length} reqs done</span>
                  <span>
                    Latest audit:{" "}
                    {i.audit
                      ? i.audit.exit_code === 0
                        ? "pass"
                        : "fail"
                      : "none"}
                  </span>
                </div>
              </Link>
            </div>
          );
        })}
      </div>

      {overlay ? (
        <PromptOverlay title={overlay.title} text={overlay.text} onClose={() => setOverlay(null)} />
      ) : null}
    </main>
  );
}
