/*
PROV: GREENFIELD.SCAFFOLD.PORTAL.03
REQ: SYS-ARCH-15
WHY: Portal internal landing page.
*/

import Link from "next/link";

export default function InternalHome() {
  return (
    <main className="page">
      <h1>Internal</h1>
      <div className="navPills" style={{ marginTop: 10 }}>
        <Link className="btn btnSmall btnActive" href="/internal">Internal</Link>
        <Link className="btn btnSmall" href="/internal/intents">Intents</Link>
        <Link className="btn btnSmall" href="/internal/tasks">Tasks</Link>
      </div>
      <div style={{ marginTop: 14 }}>
        <Link className="btn" href="/internal/intents?create=1">Create intent</Link>
      </div>
    </main>
  );
}
