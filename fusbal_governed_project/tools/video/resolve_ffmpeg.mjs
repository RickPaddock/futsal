/*
PROV: FUSBAL.TOOLS.VIDEO.RESOLVE_FFMPEG.01
REQ: REQ-V1-VIDEO-INGEST-001, SYS-ARCH-15
WHY: Provide a deterministic fallback for locating ffmpeg/ffprobe binaries (system PATH or vendored via npm) for governed video ingest.
*/

import fs from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

function safeResolve(modName) {
  try {
    return require(modName);
  } catch {
    return null;
  }
}

function existsFile(p) {
  try {
    return typeof p === "string" && Boolean(p) && fs.existsSync(p) && fs.statSync(p).isFile();
  } catch {
    return false;
  }
}

function toPath(value) {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && typeof value.path === "string") return value.path;
  return null;
}

const ffmpegPath = toPath(safeResolve("ffmpeg-static"));
const ffprobePath = toPath(safeResolve("ffprobe-static"));

const out = {
  ffmpeg: existsFile(ffmpegPath) ? ffmpegPath : null,
  ffprobe: existsFile(ffprobePath) ? ffprobePath : null,
};

process.stdout.write(`${JSON.stringify(out)}\n`);
