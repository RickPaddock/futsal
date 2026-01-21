/*
PROV: GREENFIELD.GOV.PORTAL.TASKS.01
REQ: SYS-ARCH-15, GREENFIELD-PORTAL-001
WHY: List task specs from spec/tasks/*.json.
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

export async function getServerSideProps() {
  const repoRoot = path.resolve(process.cwd(), "..", "..");
  const tasksDir = path.join(repoRoot, "spec", "tasks");
  const taskFiles = fs.existsSync(tasksDir)
    ? fs.readdirSync(tasksDir, { withFileTypes: true }).filter((d) => d.isFile() && d.name.endsWith(".json")).map((d) => d.name).sort()
    : [];

  const tasks = [];
  for (const name of taskFiles) {
    const t = safeReadJson(path.join(tasksDir, name));
    if (!t) continue;
    const taskId = String(t.task_id || "").trim();
    if (!taskId) continue;
    tasks.push({
      task_id: taskId,
      intent_id: String(t.intent_id || "").trim(),
      status: String(t.status || "").trim(),
      title: String(t.title || "").trim(),
      deliverables: Array.isArray(t.deliverables) ? t.deliverables.length : 0,
      subtasks: Array.isArray(t.subtasks) ? t.subtasks.length : 0,
    });
  }

  tasks.sort((a, b) => a.task_id.localeCompare(b.task_id));
  return { props: { tasks } };
}

export default function TasksIndex({ tasks }) {
  const router = useRouter();
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
          <div className="navPills">
            <Link className="btn btnSmall" href="/internal">Internal</Link>
            <Link className="btn btnSmall" href="/internal/intents">Intents</Link>
            <Link className="btn btnSmall btnActive" href="/internal/tasks">Tasks</Link>
          </div>
          <h1 style={{ margin: "6px 0 0 0" }}>Tasks</h1>
        </div>
        <div className="toolbarActions">
          <Link className="btn" href="/internal/intents?create=1">Create intent</Link>
          <button className="btn" type="button" disabled={refreshing} onClick={refreshAndReload}>
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>
      <div className="list">
        {tasks.map((t) => (
          <Link key={t.task_id} className="card" href={`/internal/tasks/${encodeURIComponent(t.task_id)}`}>
            <div className="row">
              <strong>{t.task_id}</strong>
              <span className="badge">{t.status || "todo"}</span>
            </div>
            <div className="muted">{t.title}</div>
            <div className="meta">
              <span>Intent: {t.intent_id || "—"}</span>
              <span>{t.deliverables} deliverables</span>
              <span>{t.subtasks} subtasks</span>
            </div>
          </Link>
        ))}
      </div>
    </main>
  );
}
