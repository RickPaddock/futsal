/*
PROV: GREENFIELD.GOV.PORTAL.TASKS.01
REQ: SYS-ARCH-15
WHY: List task specs from spec/tasks/*.json.
*/

import fs from "node:fs";
import path from "node:path";
import Link from "next/link";
import { useRouter } from "next/router";

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
  return (
    <main className="page">
      <div className="toolbar">
        <h1 style={{ margin: 0 }}>Tasks</h1>
        <button className="btn" type="button" onClick={() => router.reload()}>
          Refresh
        </button>
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

