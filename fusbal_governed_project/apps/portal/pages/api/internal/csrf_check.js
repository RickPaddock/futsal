/*
PROV: GREENFIELD.GOV.PORTAL.CSRF_CHECK.01
REQ: SYS-ARCH-15, GREENFIELD-PORTAL-014
WHY: Regression-friendly endpoint to validate same-origin + host allowlist + CSRF header/cookie wiring without triggering generation.
*/

import { csrfOk } from "../../../lib/portal_csrf.js";
import { rejectIfNotInternal } from "../../../lib/portal_request_guards.js";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ ok: false, error: "method_not_allowed" });
    return;
  }

  if (rejectIfNotInternal(req, res)) return;

  if (!csrfOk(req)) {
    res.status(403).json({ ok: false, error: "csrf_missing_or_invalid" });
    return;
  }

  res.status(200).json({ ok: true });
}
