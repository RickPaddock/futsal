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
      <ul>
        <li>
          <Link href="/internal/intents">Intents</Link>
        </li>
        <li>
          <Link href="/internal/tasks">Tasks</Link>
        </li>
      </ul>
    </main>
  );
}
