import type { ExplanationMethod, Label } from "../api/types";

export const LABEL_TEXT: Record<Label, string> = {
  REAL: "Likely real",
  AI_GENERATED: "Likely AI-generated",
  UNCERTAIN: "Uncertain",
};

export const LABEL_DESCRIPTION: Record<Label, string> = {
  REAL: "The model found no strong evidence of AI generation. That does not prove the photo is authentic.",
  AI_GENERATED:
    "The model found patterns it associates with AI-generated faces. This is a statistical signal, not proof.",
  UNCERTAIN: "The score is too close to the decision threshold to call either way. Treat this as inconclusive.",
};

export const EXPLANATION_TEXT: Record<ExplanationMethod, string> = {
  gradcam:
    "Grad-CAM: warmer regions pushed the model's score toward “AI-generated”. It shows where the evidence was, not why.",
  attention_rollout:
    "Attention rollout: warmer regions are where the model looked most. It is class-agnostic — it does not say whether those regions looked real or fake.",
};

/** Positions (0–100%) for the probability meter: marker, threshold, and the uncertain band. */
export function meterGeometry(prob: number, threshold: number, delta: number) {
  const clamp = (v: number) => Math.min(100, Math.max(0, v * 100));
  return {
    marker: clamp(prob),
    threshold: clamp(threshold),
    bandStart: clamp(threshold - delta),
    bandEnd: clamp(threshold + delta),
  };
}

export function formatPercent(prob: number): string {
  const pct = prob * 100;
  if (pct > 0 && pct < 1) return "<1%";
  if (pct < 100 && pct > 99) return ">99%";
  return `${pct.toFixed(0)}%`;
}
