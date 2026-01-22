#!/usr/bin/env node
/*
Validates portal CSRF wiring without invoking generation.

Usage:
  1) Start the portal: npm run dev
  2) Run: npm run check:csrf

Exit codes:
  0 = OK
  1 = Failed
*/

const baseUrl = process.env.PORTAL_BASE_URL || "http://localhost:3015";

function fail(msg) {
  process.stderr.write(`${msg}\n`);
  process.exit(1);
}

function parseCsrfFromSetCookie(setCookieValue) {
  const raw = String(setCookieValue || "");
  const m = raw.match(/(?:^|;\s*)portal_csrf=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : "";
}

async function getSetCookieHeader(res) {
  // Node/undici supports headers.getSetCookie() in newer versions.
  if (typeof res.headers.getSetCookie === "function") {
    const all = res.headers.getSetCookie();
    return Array.isArray(all) ? all.join("; ") : String(all || "");
  }
  return String(res.headers.get("set-cookie") || "");
}

async function main() {
  // Hit an SSR page that ensures the CSRF cookie exists.
  const pageRes = await fetch(`${baseUrl}/internal/tasks`, {
    method: "GET",
    redirect: "follow",
  });

  const setCookie = await getSetCookieHeader(pageRes);
  const token = parseCsrfFromSetCookie(setCookie);
  if (!token) {
    fail(`Could not obtain portal_csrf cookie from ${baseUrl}/internal/tasks (got set-cookie='${setCookie || ""'}')`);
  }

  const apiRes = await fetch(`${baseUrl}/api/internal/csrf_check`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      // Provide both cookie + matching header.
      cookie: `portal_csrf=${encodeURIComponent(token)}`,
      "x-portal-csrf": token,
      // Provide origin to validate same-origin rules.
      origin: baseUrl,
    },
    body: JSON.stringify({}),
  });

  const payload = await apiRes.json().catch(() => ({}));
  if (!apiRes.ok) {
    fail(`CSRF check failed: http_${apiRes.status} ${(payload && payload.error) ? String(payload.error) : ""}`.trim());
  }

  process.stdout.write("CSRF check OK\n");
}

main().catch((e) => fail(e && e.stack ? e.stack : String(e)));
