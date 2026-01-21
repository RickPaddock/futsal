/*
PROV: GREENFIELD.PORTAL.HOME.01
REQ: SYS-ARCH-15, GREENFIELD-PORTAL-010
WHY: Provide a root dashboard entrypoint (/) for the internal governance portal.
*/

import Link from "next/link";

export default function Home() {
  return (
    <main className="page">
      <h1>Fusbal – Internal Portal</h1>
      <div className="muted">Governed intent lifecycle and evidence surfaces.</div>

      <div className="grid" style={{ marginTop: 16 }}>
        <section className="panel">
          <h2>Governance</h2>
          <ul className="bullets">
            <li><Link href="/internal/intents">Intents</Link></li>
            <li><Link href="/internal/tasks">Tasks</Link></li>
          </ul>
        </section>

        <section className="panel">
          <h2>Links</h2>
          <ul className="bullets">
            <li><Link href="/internal">Internal home</Link></li>
          </ul>
        </section>
      </div>
    </main>
  );
}
