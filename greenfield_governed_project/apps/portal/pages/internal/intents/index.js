/*
PROV: GREENFIELD.SCAFFOLD.PORTAL.04
REQ: SYS-ARCH-15, GREENFIELD-PORTAL-002, GREENFIELD-PORTAL-005, GREENFIELD-PORTAL-011, GREENFIELD-PORTAL-014, GREENFIELD-PORTAL-018, GREENFIELD-PORTAL-020, GREENFIELD-PORTAL-021
WHY: List intents, compute readiness from evidence, and provide copy-ready prompt overlays via API.
*/

import fs from "node:fs";
import path from "node:path";
import Link from "next/link";
import { useRouter } from "next/router";
import { useEffect, useState } from "react";

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

const STATUS_ORDER = {
  todo: 0,
  draft: 1,
  closed: 2,
  unknown: 3,
};

export async function getServerSideProps() {
  const {
    repoRootFromPortalCwd,
    safeReadJson,
    listAuditRunsDeep,
    isIntentAuditRun,
    computeIntentReadiness,
  } = await import("../../../lib/portal_read_model.js");

  const repoRoot = repoRootFromPortalCwd();
  const feedPath = path.join(repoRoot, "status", "portal", "internal_intents.json");
  const feed = safeReadJson(feedPath) || { intents: [] };
  const intentsDir = path.join(repoRoot, "spec", "intents");
  const intentIds = fs.existsSync(intentsDir)
    ? fs.readdirSync(intentsDir, { withFileTypes: true }).filter((d) => d.isFile() && d.name.endsWith(".json")).map((d) => d.name.replace(/\.json$/, "")).sort()
    : [];
  const nextIntentId = nextNumericSuffix(intentIds, "INT-", 3);

  const intents = (feed.intents || []).map((i) => {
    const intentId = String(i.intent_id || "");
    const intentSpec = safeReadJson(path.join(repoRoot, "spec", "intents", `${intentId}.json`)) || null;
    const scope = safeReadJson(path.join(repoRoot, "status", "intents", intentId, "scope.json")) || null;
    const runs = listAuditRunsDeep(repoRoot, intentId);
    const latestAudit = runs.find(isIntentAuditRun) || null;
    const audit = latestAudit
      ? { timestamp_end: latestAudit.timestamp_end || "", exit_code: latestAudit.exit_code ?? 1, command: latestAudit.command || "" }
      : null;
    const readiness = intentSpec
      ? computeIntentReadiness({ repoRoot, intentId, intentSpec, tasks: i.tasks || [], scope: scope || {}, runs })
      : { canImplement: false, canAudit: false, canClose: false, blockers: { missing_task_specs: [], missing_close_gates: [], missing_task_quality_audits: [], failing_task_quality_audits: [] } };
    return { ...i, audit, readiness };
  }).sort((a, b) => {
    const as = String(a.status || "unknown").toLowerCase();
    const bs = String(b.status || "unknown").toLowerCase();
    const ao = STATUS_ORDER[as] ?? STATUS_ORDER.unknown;
    const bo = STATUS_ORDER[bs] ?? STATUS_ORDER.unknown;
    if (ao !== bo) return ao - bo;
    return String(a.intent_id).localeCompare(String(b.intent_id));
  });
  return { props: { intents, nextIntentId } };
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

function PromptOverlay({ title, kind, defaultIntentId, defaultRunId, defaultClosedDate, onClose }) {
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [text, setText] = useState("");
  const [intentId, setIntentId] = useState(defaultIntentId || "");
  const [runId, setRunId] = useState(defaultRunId || "");
  const [closedDate, setClosedDate] = useState(defaultClosedDate || "");

  async function renderPrompt() {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams();
      params.set("kind", kind);
      if (intentId) params.set("intentId", intentId);
      if (runId) params.set("runId", runId);
      if (closedDate) params.set("closedDate", closedDate);
      const res = await fetch(`/api/internal/prompt?${params.toString()}`);
      const payload = await res.json().catch(() => ({}));
      if (!res.ok || !payload?.ok) {
        const msg = payload?.error ? String(payload.error) : `http_${res.status}`;
        const details = payload?.details ? ` (${JSON.stringify(payload.details)})` : "";
        throw new Error(`${msg}${details}`);
      }
      setText(String(payload.text || ""));
      if (payload.intent_id) setIntentId(String(payload.intent_id));
      if (payload.run_id) setRunId(String(payload.run_id));
      if (payload.closed_date) setClosedDate(String(payload.closed_date));
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setText("");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void renderPrompt();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
            <button className="btn" type="button" onClick={renderPrompt} disabled={loading}>{loading ? "Rendering…" : "Render"}</button>
            <button className="btn" type="button" onClick={onCopy}>{copied ? "Copied" : "Copy"}</button>
            <button className="btn" type="button" onClick={onClose}>Close</button>
          </div>
        </div>
        <div className="grid" style={{ margin: "10px 0" }}>
          <div className="panel" style={{ padding: 12 }}>
            <div className="kv"><span>intentId</span><span><input value={intentId} onChange={(e) => setIntentId(e.target.value)} /></span></div>
            {(kind === "preflight" || kind === "implement" || kind === "audit" || kind === "close") ? (
              <div className="kv"><span>runId</span><span><input value={runId} onChange={(e) => setRunId(e.target.value)} /></span></div>
            ) : null}
            {kind === "close" ? (
              <div className="kv"><span>closedDate</span><span><input value={closedDate} onChange={(e) => setClosedDate(e.target.value)} /></span></div>
            ) : null}
            {error ? <div className="muted" style={{ color: "#b91c1c" }}>Error: {error}</div> : null}
          </div>
        </div>
        <textarea className="promptBox" readOnly value={text} placeholder={loading ? "Rendering prompt…" : ""} />
      </div>
    </div>
  );
}

export default function IntentsIndex({ intents, nextIntentId }) {
  const router = useRouter();
  const [refreshing, setRefreshing] = useState(false);
  const [overlay, setOverlay] = useState(null);
  const [page, setPage] = useState(1);
  const itemsPerPage = 20;
  
  const totalPages = Math.ceil(intents.length / itemsPerPage);
  const startIdx = (page - 1) * itemsPerPage;
  const endIdx = startIdx + itemsPerPage;
  const paginatedIntents = intents.slice(startIdx, endIdx);

  useEffect(() => {
    if (!router.isReady) return;
    const wantsCreate = String(router.query?.create || "").trim().toLowerCase();
    if (wantsCreate !== "1" && wantsCreate !== "true") return;
    setOverlay((prev) => prev || { title: `Create intent (${nextIntentId})`, kind: "create", intentId: nextIntentId });
    void router.replace("/internal/intents", undefined, { shallow: true });
  }, [router.isReady, router.query?.create, nextIntentId, router]);

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
    setOverlay({ title: `Create intent (${nextIntentId})`, kind: "create", intentId: nextIntentId });
  }

  function openAuditPrompt(intentId) {
    const runId = utcRunId();
    setOverlay({ title: `Audit + quality audit prompt (${intentId})`, kind: "audit", intentId, runId });
  }

  function openPreflightPrompt(intentId) {
    setOverlay({ title: `Preflight review prompt (${intentId})`, kind: "preflight", intentId, runId: utcRunId() });
  }

  function openImplementPrompt(intentId) {
    const runId = utcRunId();
    setOverlay({ title: `Implement intent prompt (${intentId})`, kind: "implement", intentId, runId });
  }

  function openClosePrompt(intentId) {
    const runId = utcRunId();
    const closedDate = utcDate();
    setOverlay({ title: `Close intent prompt (${intentId})`, kind: "close", intentId, runId, closedDate });
  }

  return (
    <main className="page">
      <div className="toolbar">
        <div>
          <div className="navPills">
            <Link className="btn btnSmall" href="/internal">Internal</Link>
            <Link className="btn btnSmall btnActive" href="/internal/intents">Intents</Link>
            <Link className="btn btnSmall" href="/internal/tasks">Tasks</Link>
          </div>
          <h1 style={{ margin: "6px 0 0 0" }}>Intents</h1>
        </div>
        <div className="toolbarActions">
          <button className="btn" type="button" onClick={openCreatePrompt}>
            Create intent
          </button>
          <button className="btn" type="button" disabled={refreshing} onClick={refreshAndReload}>
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>
      <div className="list">
        {paginatedIntents.map((i) => {
          const intentId = i.intent_id;
          const href = `/internal/intents/${encodeURIComponent(intentId)}`;
          const readiness = i.readiness || {};
          const status = String(i.status || "unknown").toLowerCase();
          const canPreflight = status !== "closed";
          return (
            <div key={intentId} className="card">
              <div className="row">
                <div className="rowLeft">
                  <Link href={href} className="cardLink"><strong>{intentId}</strong></Link>
                  <span className="badge">{i.status}</span>
                </div>
                <div className="rowRight">
                  {canPreflight ? (
                    <button className="btn btnSmall" type="button" onClick={() => openPreflightPrompt(intentId)}>Preflight</button>
                  ) : null}
                  {readiness.canImplement ? (
                    <button className="btn btnSmall" type="button" onClick={() => openImplementPrompt(intentId)}>Implement</button>
                  ) : null}
                  {readiness.canAudit ? (
                    <button className="btn btnSmall" type="button" onClick={() => openAuditPrompt(intentId)}>Audit</button>
                  ) : null}
                  {readiness.canClose ? (
                    <button className="btn btnSmall" type="button" onClick={() => openClosePrompt(intentId)}>Close</button>
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
                  {readiness?.canClose ? <span>Close: ready</span> : readiness?.canAudit ? <span>Close: blocked</span> : null}
                </div>
              </Link>
            </div>
          );
        })}
      </div>

      {totalPages > 1 ? (
        <div className="pagination" style={{ display: "flex", gap: "8px", justifyContent: "center", marginTop: "20px", alignItems: "center" }}>
          <button 
            className="btn btnSmall" 
            disabled={page === 1} 
            onClick={() => setPage(page - 1)}
          >
            Previous
          </button>
          <span style={{ padding: "0 12px" }}>
            Page {page} of {totalPages} ({intents.length} total)
          </span>
          <button 
            className="btn btnSmall" 
            disabled={page === totalPages} 
            onClick={() => setPage(page + 1)}
          >
            Next
          </button>
        </div>
      ) : null}

      {overlay ? (
        <PromptOverlay
          title={overlay.title}
          kind={overlay.kind}
          defaultIntentId={overlay.intentId}
          defaultRunId={overlay.runId}
          defaultClosedDate={overlay.closedDate}
          onClose={() => setOverlay(null)}
        />
      ) : null}
    </main>
  );
}
