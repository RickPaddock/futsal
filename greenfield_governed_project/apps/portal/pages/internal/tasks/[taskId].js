/*
PROV: GREENFIELD.GOV.PORTAL.TASKS.02
REQ: SYS-ARCH-15, GREENFIELD-PORTAL-001, GREENFIELD-PORTAL-015, GREENFIELD-PORTAL-012
WHY: Task detail view (human-readable summary + canonical JSON for spec/tasks/<TASK_ID>.json).
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

export async function getServerSideProps(ctx) {
  const { isValidTaskId, repoRootFromPortalCwd } = await import("../../../lib/portal_read_model.js");

  const taskId = String(ctx.params?.taskId || "");
  if (!isValidTaskId(taskId)) return { notFound: true };
  const repoRoot = repoRootFromPortalCwd();
  const taskSpecPath = path.join(repoRoot, "spec", "tasks", `${taskId}.json`);
  const task = safeReadJson(taskSpecPath);
  return { props: { taskId, task } };
}

export default function TaskDetail({ taskId, task }) {
  const router = useRouter();
  const [refreshing, setRefreshing] = useState(false);

  async function refreshAndReload() {
    setRefreshing(true);
    try {
      const res = await fetch("/api/internal/refresh", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ intentId: intentId || "" }),
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

  const intentId = String(task?.intent_id || "").trim();
  const status = String(task?.status || "todo").trim() || "todo";
  const scope = Array.isArray(task?.scope) ? task.scope.map(String).filter(Boolean) : [];
  const acceptance = Array.isArray(task?.acceptance) ? task.acceptance.map(String).filter(Boolean) : [];
  const deliverables = Array.isArray(task?.deliverables) ? task.deliverables : [];
  const subtasks = Array.isArray(task?.subtasks) ? task.subtasks : [];
  const subtasksDone = subtasks.filter((s) => String(s?.status || "").trim() === "done").length;

  return (
    <main className="page">
      <div className="toolbar">
        <div>
          <div className="navPills">
            <Link className="btn btnSmall" href="/internal">Internal</Link>
            <Link className="btn btnSmall" href="/internal/intents">Intents</Link>
            <Link className="btn btnSmall" href="/internal/tasks">Tasks</Link>
            {intentId ? (
              <Link className="btn btnSmall" href={`/internal/intents/${encodeURIComponent(intentId)}`}>Intent: {intentId}</Link>
            ) : null}
            <Link className="btn btnSmall btnActive" href={`/internal/tasks/${encodeURIComponent(taskId)}`}>Task: {taskId}</Link>
          </div>
          <h1 style={{ margin: "6px 0 0 0" }}>{taskId}</h1>
          <div className="muted">{String(task?.title || "")}</div>
        </div>
        <div className="toolbarActions">
          <Link className="btn" href="/internal/intents?create=1">Create intent</Link>
          <button className="btn" type="button" disabled={refreshing} onClick={refreshAndReload}>
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      <div className="grid">
        <section className="panel">
          <h2>Overview</h2>
          <div className="kv"><span>Status</span><span>{status}</span></div>
          <div className="kv"><span>Intent</span><span>{intentId ? <Link href={`/internal/intents/${encodeURIComponent(intentId)}`}>{intentId}</Link> : "—"}</span></div>
          <div className="kv"><span>Deliverables</span><span>{deliverables.length}</span></div>
          <div className="kv"><span>Subtasks</span><span>{subtasksDone}/{subtasks.length} done</span></div>
        </section>

        <section className="panel">
          <h2>Links</h2>
          <ul className="bullets">
            <li><code>spec/tasks/{taskId}.json</code></li>
            {intentId ? <li><code>spec/intents/{intentId}.json</code></li> : null}
          </ul>
        </section>
      </div>

      {scope.length ? (
        <section className="panel">
          <h2>Scope</h2>
          <ul className="bullets">
            {scope.map((s) => <li key={s}>{s}</li>)}
          </ul>
        </section>
      ) : null}

      {acceptance.length ? (
        <section className="panel">
          <h2>Acceptance</h2>
          <ul className="bullets">
            {acceptance.map((s) => <li key={s}>{s}</li>)}
          </ul>
        </section>
      ) : null}

      <section className="panel">
        <h2>Deliverables</h2>
        {deliverables.length ? (
          <div className="list">
            {deliverables.map((d) => {
              const did = String(d?.deliverable_id || "").trim();
              const title = String(d?.title || "").trim();
              const paths = Array.isArray(d?.paths) ? d.paths.map(String).filter(Boolean) : [];
              const dAcceptance = Array.isArray(d?.acceptance) ? d.acceptance.map(String).filter(Boolean) : [];
              const evidence = Array.isArray(d?.evidence) ? d.evidence.map(String).filter(Boolean) : [];
              return (
                <div key={did || title} className="card">
                  <div className="row">
                    <strong><code>{did || "DELIV-?"}</code></strong>
                    <span className="badge">deliverable</span>
                  </div>
                  <div className="muted">{title}</div>
                  {paths.length ? (
                    <>
                      <div className="meta"><strong>Paths</strong><span /></div>
                      <ul className="bullets">
                        {paths.map((p) => <li key={p}><code>{p}</code></li>)}
                      </ul>
                    </>
                  ) : null}
                  {dAcceptance.length ? (
                    <>
                      <div className="meta"><strong>Acceptance</strong><span /></div>
                      <ul className="bullets">
                        {dAcceptance.map((a) => <li key={a}>{a}</li>)}
                      </ul>
                    </>
                  ) : null}
                  {evidence.length ? (
                    <>
                      <div className="meta"><strong>Evidence</strong><span /></div>
                      <ul className="bullets">
                        {evidence.map((e) => <li key={e}><code>{e}</code></li>)}
                      </ul>
                    </>
                  ) : null}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="muted">No deliverables found.</div>
        )}
      </section>

      <section className="panel">
        <h2>Subtasks</h2>
        {subtasks.length ? (
          <div className="list">
            {subtasks.map((s) => {
              const sid = String(s?.subtask_id || "").trim();
              const title = String(s?.title || "").trim();
              const stStatus = String(s?.status || "todo").trim() || "todo";
              const area = String(s?.area || "").trim();
              const prov = String(s?.provenance_prefix || "").trim();
              const doneWhen = Array.isArray(s?.done_when) ? s.done_when.map(String).filter(Boolean) : [];
              const evidence = Array.isArray(s?.evidence) ? s.evidence.map(String).filter(Boolean) : [];
              return (
                <div key={sid || title} className="card">
                  <div className="row">
                    <strong><code>{sid || "SUB-?"}</code></strong>
                    <span className="badge">{stStatus}</span>
                  </div>
                  <div className="muted">{title}</div>
                  <div className="meta">
                    <span>Area: <code>{area || "—"}</code></span>
                    <span>Prov: <code>{prov || "—"}</code></span>
                  </div>
                  {doneWhen.length ? (
                    <>
                      <div className="meta"><strong>Done when</strong><span /></div>
                      <ul className="bullets">
                        {doneWhen.map((d) => <li key={d}>{d}</li>)}
                      </ul>
                    </>
                  ) : null}
                  {evidence.length ? (
                    <>
                      <div className="meta"><strong>Evidence</strong><span /></div>
                      <ul className="bullets">
                        {evidence.map((e) => <li key={e}><code>{e}</code></li>)}
                      </ul>
                    </>
                  ) : null}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="muted">No subtasks found.</div>
        )}
      </section>

      <section className="panel">
        <h2>Task spec (canonical JSON)</h2>
        <details>
          <summary className="muted">Show raw JSON</summary>
          <pre className="pre">{JSON.stringify(task || {}, null, 2)}</pre>
        </details>
      </section>
    </main>
  );
}
