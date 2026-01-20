/*
PROV: GREENFIELD.GOV.PORTAL.TASKS.02
REQ: SYS-ARCH-15
WHY: Task detail view (renders spec/tasks/<TASK_ID>.json).
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
  const taskId = String(ctx.params?.taskId || "");
  const repoRoot = path.resolve(process.cwd(), "..", "..");
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
      const res = await fetch("/api/internal/refresh", { method: "POST" });
      if (!res.ok) throw new Error(`refresh_failed_http_${res.status}`);
      router.reload();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      alert(`Refresh failed: ${msg}`);
    } finally {
      setRefreshing(false);
    }
  }

  const intentId = String(task?.intent_id || "").trim();

  return (
    <main className="page">
      <div className="toolbar">
        <div>
          <div className="muted">
            <Link href="/internal/tasks">← Back</Link>
            {intentId ? (
              <>
                {" "}
                · <Link href={`/internal/intents/${encodeURIComponent(intentId)}`}>Intent {intentId}</Link>
              </>
            ) : null}
          </div>
          <h1 style={{ margin: "6px 0 0 0" }}>{taskId}</h1>
          <div className="muted">{String(task?.title || "")}</div>
        </div>
        <button className="btn" type="button" disabled={refreshing} onClick={refreshAndReload}>
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      <section className="panel">
        <h2>Task spec (canonical)</h2>
        <pre className="pre">{JSON.stringify(task || {}, null, 2)}</pre>
      </section>
    </main>
  );
}

