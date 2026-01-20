/*
PROV: GREENFIELD.SCAFFOLD.PORTAL.04
REQ: SYS-ARCH-15
WHY: List intents from generated status/portal/internal_intents.json.
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
  return { props: { intents } };
}

export default function IntentsIndex({ intents }) {
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
        <h1 style={{ margin: 0 }}>Intents</h1>
        <button className="btn" type="button" disabled={refreshing} onClick={refreshAndReload}>
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
      </div>
      <div className="list">
        {intents.map((i) => (
          <Link key={i.intent_id} className="card" href={`/internal/intents/${encodeURIComponent(i.intent_id)}`}>
            <div className="row">
              <strong>{i.intent_id}</strong>
              <span className="badge">{i.status}</span>
            </div>
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
        ))}
      </div>
    </main>
  );
}
