/*
PROV: GREENFIELD.SCAFFOLD.PORTAL.05
REQ: SYS-ARCH-15, GREENFIELD-PORTAL-002, GREENFIELD-PORTAL-005, GREENFIELD-PORTAL-006, GREENFIELD-PORTAL-007, GREENFIELD-PORTAL-008, GREENFIELD-PORTAL-009, GREENFIELD-PORTAL-011, GREENFIELD-PORTAL-014, GREENFIELD-PORTAL-015, GREENFIELD-PORTAL-016, GREENFIELD-PORTAL-019, GREENFIELD-PORTAL-020, GREENFIELD-PORTAL-022
WHY: Intent detail view (surfaces specs/audits, evidence-based readiness, and prompt-driven actions).
*/

import fs from "node:fs";
import path from "node:path";
import Link from "next/link";
import { useRouter } from "next/router";
import { useEffect, useState } from "react";

function formatUkDateTime(value) {
  if (!value) return "";
  const d = new Date(String(value));
  if (Number.isNaN(d.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: "Europe/London",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(d);
}

function utcRunId() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}_${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}`;
}

function utcDate() {
  return new Date().toISOString().slice(0, 10);
}

export async function getServerSideProps(ctx) {
  const {
    repoRootFromPortalCwd,
    relPosix,
    safeReadJson,
    listAuditRunsDeep,
    computeIntentReadiness,
    loadPerTaskQualityAudits,
    isValidIntentId,
  } = await import("../../../lib/portal_read_model.js");

  const intentId = String(ctx.params?.intentId || "");
  if (!isValidIntentId(intentId)) return { notFound: true };
  const repoRoot = repoRootFromPortalCwd();

  const feedPath = path.join(repoRoot, "status", "portal", "internal_intents.json");
  const feed = safeReadJson(feedPath) || { intents: [] };
  const feedIntent = (feed.intents || []).find((x) => x.intent_id === intentId) || null;

  const intentSpecPath = path.join(repoRoot, "spec", "intents", `${intentId}.json`);
  const intentSpec = safeReadJson(intentSpecPath);

  const taskIds = Array.isArray(intentSpec?.task_ids_planned) ? intentSpec.task_ids_planned.map(String) : [];
  const tasks = taskIds
    .map((tid) => safeReadJson(path.join(repoRoot, "spec", "tasks", `${tid}.json`)))
    .filter(Boolean);

  const mdPath = path.join(repoRoot, "status", "intents", intentId, "intent.md");
  const md = fs.existsSync(mdPath) ? fs.readFileSync(mdPath, "utf8") : "";
  const scope = safeReadJson(path.join(repoRoot, "status", "intents", intentId, "scope.json"));
  const workPackages = safeReadJson(path.join(repoRoot, "status", "intents", intentId, "work_packages.json"));

  const runs = listAuditRunsDeep(repoRoot, intentId).slice(0, 25);
  const readiness = computeIntentReadiness({ repoRoot, intentId, intentSpec: intentSpec || {}, tasks, scope: scope || {}, runs });
  const plannedTaskIds = Array.isArray(intentSpec?.task_ids_planned) ? intentSpec.task_ids_planned.map(String) : [];
  const runId = readiness?.blockers?.latest_quality_run_id || null;
  const perTask = runId
    ? loadPerTaskQualityAudits({ repoRoot, intentId, runId, taskIds: plannedTaskIds })
    : { audits: [], missing: plannedTaskIds.map((task_id) => ({ task_id, path: "" })) };

  function latestJsonUnderRuns({ candidates }) {
    const runsDir = path.join(repoRoot, "status", "audit", intentId, "runs");
    if (!fs.existsSync(runsDir)) return null;
    let best = null;
    for (const d of fs.readdirSync(runsDir, { withFileTypes: true }).filter((x) => x.isDirectory()).map((x) => x.name)) {
      for (const rel of candidates) {
        const abs = path.join(runsDir, d, rel);
        const rep = safeReadJson(abs);
        if (!rep) continue;
        const ts = String(rep.timestamp || rep.timestamp_end || "");
        if (!best || ts.localeCompare(String(best.timestamp || "")) > 0) {
          best = { report: rep, timestamp: ts, rel: relPosix(path.join("status", "audit", intentId, "runs", d, rel)) };
        }
      }
    }
    return best;
  }

  const auditReportInfo = latestJsonUnderRuns({ candidates: ["audit/audit_report.json", "audit_report.json"] });
  const qualityReportInfo = latestJsonUnderRuns({ candidates: ["quality_audit.json"] });

  const feedTaskIds = Array.isArray(feedIntent?.tasks) ? feedIntent.tasks.map((t) => String(t.task_id || "")).filter(Boolean) : [];
  const missingFromFeed = plannedTaskIds.filter((tid) => !feedTaskIds.includes(tid));

  return { props: { intentId, feedIntent, intentSpec, tasks, scope, workPackages, md, runs, readiness, auditReportInfo, qualityReportInfo, perTask, missingFromFeed } };
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
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 800);
    } catch {
      // ignore
    }
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

export default function IntentDetail({ intentId, feedIntent, intentSpec, tasks, scope, workPackages, md, runs, readiness, auditReportInfo, qualityReportInfo, perTask, missingFromFeed }) {
  const router = useRouter();
  const requirements = feedIntent?.requirements_in_scope || [];
  const tasksDone = tasks.filter((t) => String(t?.status || "").trim() === "done").length;
  const requirementsImplemented = requirements.filter((r) => String(r?.tracking_implementation || "").trim() === "done").length;
  const status = String(feedIntent?.status || intentSpec?.status || "unknown").toLowerCase();
  const [refreshing, setRefreshing] = useState(false);
  const [overlay, setOverlay] = useState(null);
  const [copiedCommand, setCopiedCommand] = useState("");

  async function copyEvidenceCommand(runId, command) {
    try {
      const evidenceCmd = `node tools/evidence/record_run.mjs --intent-id ${intentId} --out status/audit/${intentId}/runs/${runId}/run.json -- ${command}`;
      await navigator.clipboard.writeText(evidenceCmd);
      setCopiedCommand(runId);
      setTimeout(() => setCopiedCommand(""), 1200);
    } catch {
      // ignore
    }
  }

  async function refreshAndReload() {
    setRefreshing(true);
    try {
      const res = await fetch("/api/internal/refresh", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ intentId }),
      });
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

  function openImplementPrompt() {
    setOverlay({ title: `Implement intent prompt (${intentId})`, kind: "implement", intentId, runId: utcRunId() });
  }

  function openPreflightPrompt() {
    setOverlay({ title: `Preflight review prompt (${intentId})`, kind: "preflight", intentId, runId: utcRunId() });
  }

  function openAuditPrompt() {
    setOverlay({ title: `Audit + quality audit prompt (${intentId})`, kind: "audit", intentId, runId: utcRunId() });
  }

  function openClosePrompt() {
    setOverlay({ title: `Close intent prompt (${intentId})`, kind: "close", intentId, runId: utcRunId(), closedDate: utcDate() });
  }

  return (
    <main className="page">
      <div className="toolbar">
        <div>
          <div className="navPills">
            <Link className="btn btnSmall" href="/internal">Internal</Link>
            <Link className="btn btnSmall" href="/internal/intents">Intents</Link>
            <Link className="btn btnSmall" href="/internal/tasks">Tasks</Link>
            <Link className="btn btnSmall btnActive" href={`/internal/intents/${encodeURIComponent(intentId)}`}>Intent: {intentId}</Link>
          </div>
          <h1 style={{ margin: "6px 0 0 0" }}>{intentId}</h1>
          <div className="muted">{feedIntent?.title || intentSpec?.title || ""}</div>
        </div>
        <div className="toolbarActions">
          <Link className="btn" href="/internal/intents?create=1">Create intent</Link>
          {status !== "closed" ? <button className="btn" type="button" onClick={openPreflightPrompt}>Preflight</button> : null}
          {readiness?.canImplement ? <button className="btn" type="button" onClick={openImplementPrompt}>Implement</button> : null}
          {readiness?.canAudit ? <button className="btn" type="button" onClick={openAuditPrompt}>Audit</button> : null}
          {readiness?.canClose ? <button className="btn" type="button" onClick={openClosePrompt}>Close</button> : null}
          <button className="btn" type="button" disabled={refreshing} onClick={refreshAndReload}>
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      <div className="grid">
        <section className="panel">
          <h2>Overview</h2>
          <div className="kv"><span>Status</span><span>{feedIntent?.status || intentSpec?.status || "unknown"}</span></div>
          <div className="kv"><span>Created</span><span>{feedIntent?.created_date || intentSpec?.created_date || ""}</span></div>
          <div className="kv"><span>Closed</span><span>{feedIntent?.closed_date || intentSpec?.closed_date || ""}</span></div>
          <div className="kv"><span>Tasks completed</span><span>{tasksDone} of {tasks.length}</span></div>
          <div className="kv"><span>Requirements implemented</span><span>{requirementsImplemented} of {requirements.length}</span></div>
          <div className="muted">“Requirements implemented” counts requirements where <code>tracking.implementation</code> is <code>done</code>.</div>
        </section>

        <section className="panel">
          <h2>Close Gate</h2>
          {(readiness?.closeGates?.satisfied || []).length ? (
            <>
              <div className="muted">Satisfied</div>
              <ul className="bullets">
                {(readiness.closeGates.satisfied || []).map((c) => (
                  <li key={`ok:${c}`}><code>{c}</code></li>
                ))}
              </ul>
            </>
          ) : null}
          {(readiness?.closeGates?.missing || []).length ? (
            <>
              <div className="muted" style={{ color: "#b91c1c" }}>Missing</div>
              <ul className="bullets">
                {(readiness.closeGates.missing || []).map((c) => (
                  <li key={`miss:${c}`}><code>{c}</code></li>
                ))}
              </ul>
            </>
          ) : <div className="muted">No close gates declared.</div>}
        </section>
      </div>

      {missingFromFeed?.length ? (
        <section className="panel">
          <h2>Feed issues</h2>
          <div className="muted" style={{ color: "#b91c1c" }}>These planned tasks are missing from the generated feed:</div>
          <ul className="bullets">
            {missingFromFeed.map((t) => <li key={t}><code>{t}</code></li>)}
          </ul>
        </section>
      ) : null}

      <section className="panel">
        <h2>Requirements in scope</h2>
        <div className="table">
          <div className="thead">
            <div>ID</div>
            <div>Definition status</div>
            <div>Implementation tracking</div>
            <div>Title</div>
          </div>
          {(requirements || []).map((r) => (
            <div key={r.id} className="trow">
              <div><code>{r.id}</code></div>
              <div>{r.status || ""}</div>
              <div>{r.tracking_implementation || "todo"}</div>
              <div>{r.title || ""}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Tasks</h2>
        <div className="list">
          {tasks.map((t) => (
            <Link key={t.task_id} className="card" href={`/internal/tasks/${encodeURIComponent(t.task_id)}`}>
              <div className="row">
                <strong>{t.task_id}</strong>
                <span className="badge">{t.status}</span>
              </div>
              <div className="muted">{t.title}</div>
              <div className="meta">
                <span>{(t.subtasks || []).filter((s) => s.status === "done").length}/{(t.subtasks || []).length} subtasks done</span>
                <span>{(t.deliverables || []).length} deliverables</span>
              </div>
              {(t.deliverables || []).length ? (
                <ul className="bullets">
                  {(t.deliverables || []).map((d) => (
                    <li key={d.deliverable_id}>
                      <code>{d.deliverable_id}</code> — {d.title}
                    </li>
                  ))}
                </ul>
              ) : null}
              {(t.subtasks || []).length ? (
                <div className="meta">
                  <strong>Subtasks</strong>
                  <span />
                </div>
              ) : null}
              {(t.subtasks || []).length ? (
                <ul className="bullets">
                  {(t.subtasks || []).map((s) => (
                    <li key={s.subtask_id}>
                      <code>{s.subtask_id}</code> <span className="badge">{s.status || "todo"}</span> — {s.title}
                    </li>
                  ))}
                </ul>
              ) : null}
            </Link>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Work packages (generated)</h2>
        <pre className="pre">{JSON.stringify(workPackages || {}, null, 2)}</pre>
        <pre className="pre">{JSON.stringify(scope || {}, null, 2)}</pre>
      </section>

      <section className="panel">
        <h2>Audit</h2>
        {auditReportInfo ? (
          <div className="kv">
            <span>Latest audit_report.json</span>
            <span>
              <a href={`/api/internal/file?rel=${encodeURIComponent(auditReportInfo.rel)}`} target="_blank" rel="noreferrer">raw</a>
            </span>
          </div>
        ) : <div className="muted">No audit_report.json found in status/audit.</div>}

        <h2 style={{ marginTop: 16 }}>Quality audit</h2>
        {qualityReportInfo ? (
          <div className="kv">
            <span>Latest quality_audit.json</span>
            <span>
              <a href={`/api/internal/file?rel=${encodeURIComponent(qualityReportInfo.rel)}`} target="_blank" rel="noreferrer">raw</a>
            </span>
          </div>
        ) : <div className="muted">No quality_audit.json found in status/audit.</div>}

        <h3 style={{ marginTop: 16 }}>Per-task quality audits</h3>
        <div className="table">
          <div className="thead">
            <div>Task</div>
            <div>Run</div>
            <div>Timestamp (UK)</div>
            <div>Gate</div>
            <div>Functional</div>
            <div>Non-functional</div>
            <div>Blockers</div>
            <div>Report</div>
          </div>
          {(perTask?.audits || []).map((a) => {
            const gate = String(a?.report?.gate?.status || "unknown");
            const functional = String(a?.report?.functional?.status || "unknown");
            const nfr = String(a?.report?.nonfunctional?.overall_status || "unknown");
            const blockers = Array.isArray(a?.report?.gate?.blockers) ? a.report.gate.blockers : [];
            const ts = formatUkDateTime(a?.report?.timestamp || "");
            const runLinkRel = `status/audit/${intentId}/runs/${runId}/quality_audit.json`;
            return (
              <div key={`ok:${a.task_id}`} className="trow">
                <div><code>{a.task_id}</code></div>
                <div><a href={`/api/internal/file?rel=${encodeURIComponent(runLinkRel)}`} target="_blank" rel="noreferrer"><code>{runId}</code></a></div>
                <div>{ts}</div>
                <div>{gate}</div>
                <div>{functional}</div>
                <div>{nfr}</div>
                <div>{blockers.length ? blockers.join("; ") : ""}</div>
                <div><a href={`/api/internal/file?rel=${encodeURIComponent(a.path)}`} target="_blank" rel="noreferrer">raw</a></div>
              </div>
            );
          })}
          {(perTask?.missing || []).map((m) => (
            <div key={`missing:${m.task_id}`} className="trow">
              <div><code>{m.task_id}</code></div>
              <div><code>{runId || ""}</code></div>
              <div />
              <div style={{ color: "#b91c1c" }}>missing</div>
              <div />
              <div />
              <div />
              <div><code>{m.path || ""}</code></div>
            </div>
          ))}
        </div>
        {!(perTask?.audits || []).length && !(perTask?.missing || []).length ? <div className="muted">No per-task quality audits found.</div> : null}
        {runs?.length ? (
          <div className="table">
            <div className="thead">
              <div>Run</div>
              <div>Stage</div>
              <div>End</div>
              <div>Exit</div>
              <div>Command</div>
              <div>Action</div>
            </div>
            {runs.map((r) => (
              <div key={`${r.run_id}:${r.stage}:${r.run_json_path || ""}`} className="trow">
                <div><code>{r.run_id}</code></div>
                <div>{r.stage || ""}</div>
                <div>{formatUkDateTime(r.timestamp_end || "")}</div>
                <div>{String(r.exit_code ?? "")}</div>
                <div><code>{r.command || ""}</code></div>
                <div>
                  <button 
                    className="btn" 
                    style={{ fontSize: "11px", padding: "2px 8px" }}
                    onClick={() => copyEvidenceCommand(r.run_id, r.command || "")}
                  >
                    {copiedCommand === r.run_id ? "Copied!" : "Copy evidence cmd"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : null}
      </section>

      <section className="panel">
        <h2>Intent markdown (generated)</h2>
        <pre className="pre">{md}</pre>
      </section>

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
