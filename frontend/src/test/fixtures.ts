import type { Face, PredictResponse } from "../api/types";

// 1x1 transparent PNG
export const TINY_PNG =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

export function makeFace(overrides: Partial<Face> = {}): Face {
  return {
    face_index: 0,
    bbox: [10, 20, 110, 140],
    detection_score: 0.998,
    label: "AI_GENERATED",
    prob_ai_generated: 0.87,
    heatmap_png_base64: TINY_PNG,
    face_crop_png_base64: TINY_PNG,
    ...overrides,
  };
}

export function makePrediction(faces: Face[] = [makeFace()]): PredictResponse {
  return {
    request_id: "req-1",
    model: { name: "effnet_b0", version: "1.0.0", threshold: 0.62, uncertainty_delta: 0.05, explanation_method: "gradcam" },
    image: { width: 640, height: 480 },
    faces,
    stored: false,
    history_id: null,
    timings_ms: { decode: 5, detect: 80, inference: 20, explain: 4, encode: 6 },
    disclaimer: "A detection aid, not proof.",
  };
}
