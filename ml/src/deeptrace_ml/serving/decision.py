"""The REAL / AI_GENERATED / UNCERTAIN rule — shared by evaluation and the API (no heavy deps)."""

from __future__ import annotations

LABELS = ("REAL", "AI_GENERATED", "UNCERTAIN")


def verdict(prob: float, threshold: float, delta: float) -> str:
    """UNCERTAIN inside [threshold - delta, threshold + delta); otherwise the thresholded label."""
    if delta > 0 and abs(prob - threshold) < delta:
        return "UNCERTAIN"
    return "AI_GENERATED" if prob >= threshold else "REAL"
