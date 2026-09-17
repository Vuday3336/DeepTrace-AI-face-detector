// Mirrors backend/app/schemas.py. Keep in sync with /api/v1/openapi.json.

export type Label = "REAL" | "AI_GENERATED" | "UNCERTAIN";
export type ExplanationMethod = "gradcam" | "attention_rollout";

export interface ModelSummary {
  name: string;
  version: string;
  threshold: number;
  uncertainty_delta: number;
  explanation_method: ExplanationMethod;
}

export interface Face {
  face_index: number;
  bbox: [number, number, number, number];
  detection_score: number;
  label: Label;
  prob_ai_generated: number;
  heatmap_png_base64: string | null;
  face_crop_png_base64: string;
}

export interface PredictResponse {
  request_id: string;
  model: ModelSummary;
  image: { width: number; height: number };
  faces: Face[];
  stored: boolean;
  history_id: string | null;
  timings_ms: Record<string, number>;
  disclaimer: string;
}

export interface Health {
  status: "ok" | "degraded";
  model_loaded: boolean;
  model_name: string | null;
  model_version: string | null;
  model_error: string | null;
  history: "enabled" | "disabled";
  db: "ok" | "error" | "disabled";
}

export interface TestSetMetrics {
  description: string;
  n: number;
  roc_auc: number | null;
  roc_auc_ci95: { low: number; high: number } | null;
  balanced_accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  fpr: number;
  ece: number;
}

export interface RobustnessRow {
  family: string;
  level: number;
  n: number;
  roc_auc: number | null;
  balanced_accuracy: number;
}

export interface ModelInfo {
  model: { name: string; version: string; family: string; timm_name: string; frozen_backbone: boolean; input_size: number };
  decision: { temperature: number; threshold: number; uncertainty_delta: number; target_fpr: number };
  explanation: { method: ExplanationMethod };
  training_data: Record<string, string>;
  metrics:
    | { status: "not_evaluated" }
    | { status: "evaluated"; test_sets: Record<string, TestSetMetrics>; robustness?: Record<string, RobustnessRow[]> };
  known_limitations: string[];
  disclaimer: string;
  preprocessing: Record<string, string | number>;
}

export interface User {
  id: string;
  email: string;
  created_at: string;
}

export interface HistoryItem {
  id: string;
  created_at: string;
  model_name: string;
  model_version: string;
  num_faces: number;
  faces: { face_index: number; bbox: number[]; label: Label; prob_ai_generated: number }[];
  has_image: boolean;
}

export interface HistoryPage {
  items: HistoryItem[];
  total: number;
  limit: number;
  offset: number;
}
