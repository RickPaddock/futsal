/*
PROV: GREENFIELD.SCAFFOLD.PORTAL.05
REQ: SYS-ARCH-15
WHY: Intent detail view (renders generated intent.md).
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

function latestAuditReport(repoRoot, intentId) {
  const runsDir = path.join(repoRoot, "status", "audit", intentId, "runs");
  if (!fs.existsSync(runsDir)) return null;
  let best = null;
  for (const d of fs.readdirSync(runsDir, { withFileTypes: true }).filter((x) => x.isDirectory()).map((x) => x.name)) {
    const p = path.join(runsDir, d, "audit_report.json");
    const rep = safeReadJson(p);
    if (!rep) continue;
    const ts = String(rep.timestamp || "");
    if (!best || ts.localeCompare(String(best.timestamp || "")) > 0) best = rep;
  }
  return best;
}

function latestQualityAuditReport(repoRoot, intentId) {
  const runsDir = path.join(repoRoot, "status", "audit", intentId, "runs");
  if (!fs.existsSync(runsDir)) return null;
  let best = null;
  for (const d of fs.readdirSync(runsDir, { withFileTypes: true }).filter((x) => x.isDirectory()).map((x) => x.name)) {
    const p = path.join(runsDir, d, "quality_audit.json");
    const rep = safeReadJson(p);
    if (!rep) continue;
    const ts = String(rep.timestamp || "");
    if (!best || ts.localeCompare(String(best.timestamp || "")) > 0) best = rep;
  }
  return best;
}

export async function getServerSideProps(ctx) {
  const intentId = String(ctx.params?.intentId || "");
  const repoRoot = path.resolve(process.cwd(), "..", "..");

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

  const runs = listAuditRuns(repoRoot, intentId).slice(0, 10);
  const auditReport = latestAuditReport(repoRoot, intentId);
  const qualityAuditReport = latestQualityAuditReport(repoRoot, intentId);

  return { props: { intentId, feedIntent, intentSpec, tasks, scope, workPackages, md, runs, auditReport, qualityAuditReport } };
}

export default function IntentDetail({ intentId, feedIntent, intentSpec, tasks, scope, workPackages, md, runs, auditReport, qualityAuditReport }) {
  const router = useRouter();
  const requirements = feedIntent?.requirements_in_scope || [];
  const [refreshing, setRefreshing] = useState(false);

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

  return (
    <main className="page">
      <div className="toolbar">
        <div>
          <div className="muted">
            <Link href="/internal/intents">← Back</Link>
          </div>
          <h1 style={{ margin: "6px 0 0 0" }}>{intentId}</h1>
          <div className="muted">{feedIntent?.title || intentSpec?.title || ""}</div>
        </div>
        <button className="btn" type="button" disabled={refreshing} onClick={refreshAndReload}>
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      <div className="grid">
        <section className="panel">
          <h2>Overview</h2>
          <div className="kv"><span>Status</span><span>{feedIntent?.status || intentSpec?.status || "unknown"}</span></div>
          <div className="kv"><span>Created</span><span>{feedIntent?.created_date || intentSpec?.created_date || ""}</span></div>
          <div className="kv"><span>Closed</span><span>{feedIntent?.closed_date || intentSpec?.closed_date || ""}</span></div>
          <div className="kv"><span>Tasks</span><span>{tasks.filter((t) => t.status === "done").length}/{tasks.length} done</span></div>
          <div className="kv"><span>Reqs</span><span>{requirements.filter((r) => r.tracking_implementation === "done").length}/{requirements.length} done</span></div>
        </section>

        <section className="panel">
          <h2>Close Gate</h2>
          <ul className="bullets">
            {(intentSpec?.close_gate || []).map((c) => (
              <li key={c}><code>{c}</code></li>
            ))}
          </ul>
        </section>
      </div>

      <section className="panel">
        <h2>Requirements in scope</h2>
        <div className="table">
          <div className="thead">
            <div>ID</div>
            <div>Definition</div>
            <div>Implementation</div>
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
        {auditReport ? <pre className="pre">{JSON.stringify(auditReport, null, 2)}</pre> : <div className="muted">No audit_report.json found in status/audit.</div>}
        <h2 style={{ marginTop: 16 }}>Quality audit</h2>
        {qualityAuditReport ? (
          <pre className="pre">{JSON.stringify(qualityAuditReport, null, 2)}</pre>
        ) : (
          <div className="muted">No quality_audit.json found in status/audit.</div>
        )}
        {runs?.length ? (
          <div className="table">
            <div className="thead">
              <div>Run</div>
              <div>End</div>
              <div>Exit</div>
              <div>Command</div>
            </div>
            {runs.map((r) => (
              <div key={r.run_id} className="trow">
                <div><code>{r.run_id}</code></div>
                <div>{r.timestamp_end || ""}</div>
                <div>{String(r.exit_code ?? "")}</div>
                <div><code>{r.command || ""}</code></div>
              </div>
            ))}
          </div>
        ) : null}
      </section>

      <section className="panel">
        <h2>Intent markdown (generated)</h2>
        <pre className="pre">{md}</pre>
      </section>
    </main>
  );
}
