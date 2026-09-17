import type { ReactNode } from "react";
import type { Label } from "../api/types";
import type { ApiError } from "../api/client";
import { LABEL_TEXT } from "../lib/verdict";

const LABEL_STYLES: Record<Label, string> = {
  REAL: "bg-real/15 text-real ring-real/40",
  AI_GENERATED: "bg-fake/15 text-fake ring-fake/40",
  UNCERTAIN: "bg-unsure/15 text-unsure ring-unsure/40",
};

export function LabelBadge({ label }: { label: Label }) {
  return (
    <span className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold ring-1 ${LABEL_STYLES[label]}`}>
      {LABEL_TEXT[label]}
    </span>
  );
}

export function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <section className={`rounded-xl border border-line bg-panel p-5 ${className}`}>{children}</section>;
}

export function Spinner({ label }: { label: string }) {
  return (
    <div role="status" className="flex items-center gap-3 text-sm text-slate-300">
      <span className="h-5 w-5 animate-spin rounded-full border-2 border-accent border-t-transparent" />
      {label}
    </div>
  );
}

const FRIENDLY: Record<string, string> = {
  NO_FACE_DETECTED: "No face found. Try a clearer photo where the face is reasonably large and facing the camera.",
  UNSUPPORTED_MEDIA_TYPE: "That file isn't a JPG, PNG or WebP image.",
  FILE_TOO_LARGE: "That file is larger than 10 MB.",
  INVALID_IMAGE: "The image couldn't be read — it may be corrupted.",
  RATE_LIMITED: "Too many requests. Please wait a moment and try again.",
  MODEL_NOT_LOADED: "The detection model isn't available right now.",
  HISTORY_DISABLED: "Accounts and history are turned off on this deployment.",
  NETWORK_ERROR: "Can't reach the DeepTrace API.",
};

export function ErrorAlert({ error, className = "" }: { error: ApiError | string; className?: string }) {
  const message = typeof error === "string" ? error : (FRIENDLY[error.code] ?? error.message);
  const detail = typeof error === "string" ? null : error.requestId;
  return (
    <div role="alert" className={`rounded-lg border border-fake/40 bg-fake/10 px-4 py-3 text-sm text-rose-200 ${className}`}>
      {message}
      {detail && <span className="mt-1 block font-mono text-[11px] text-rose-300/60">request {detail}</span>}
    </div>
  );
}

export function Disclaimer() {
  return (
    <p className="rounded-lg border border-unsure/30 bg-unsure/10 px-4 py-3 text-sm text-amber-100">
      <strong className="font-semibold">A detection aid, not proof.</strong> The model can be wrong — especially on
      images from newer AI generators, heavily compressed photos and screenshots. Never use a result as the sole
      basis for an accusation.
    </p>
  );
}
