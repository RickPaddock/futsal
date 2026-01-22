/*
PROV: GREENFIELD.PORTAL.FILE_API.01
REQ: SYS-ARCH-15, GREENFIELD-PORTAL-015, GREENFIELD-PORTAL-016
WHY: Serve selected repo artefacts (status/spec) to the portal UI without exposing arbitrary filesystem paths.
*/

import fs from "node:fs";
import path from "node:path";
import { repoRootFromPortalCwd } from "../../../lib/portal_read_model.js";

function isSafeRelPath(rel) {
  const p = String(rel || "").replace(/\\/g, "/");
  if (!p) return false;
  if (p.startsWith("/")) return false;
  if (p.includes("..")) return false;
  if (!(p.startsWith("status/") || p.startsWith("spec/"))) return false;
  return true;
}

export default async function handler(req, res) {
  if (req.method !== "GET") {
    res.status(405).json({ ok: false, error: "method_not_allowed" });
    return;
  }

  const rel = String(req.query?.rel || "").trim();
  if (!isSafeRelPath(rel)) {
    res.status(400).json({ ok: false, error: "invalid_path" });
    return;
  }

  const repoRoot = repoRootFromPortalCwd();
  const abs = path.join(repoRoot, rel);
  if (!fs.existsSync(abs) || fs.statSync(abs).isDirectory()) {
    res.status(404).json({ ok: false, error: "not_found" });
    return;
  }

  const ext = path.extname(abs).toLowerCase();
  const raw = fs.readFileSync(abs);
  if (ext === ".json") {
    res.setHeader("content-type", "application/json; charset=utf-8");
  } else {
    res.setHeader("content-type", "text/plain; charset=utf-8");
  }
  res.status(200).send(raw);
}
