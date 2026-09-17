import type { Face, ModelSummary } from "../api/types";
import { LABEL_DESCRIPTION } from "../lib/verdict";
import { HeatmapViewer } from "./HeatmapViewer";
import { ProbabilityMeter } from "./ProbabilityMeter";
import { LabelBadge, Panel } from "./ui";

export function ResultCard({ face, model, total }: { face: Face; model: ModelSummary; total: number }) {
  return (
    <Panel className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-sm font-medium text-slate-400">
          Face {face.face_index + 1}
          {total > 1 && <span className="text-slate-600"> of {total}</span>}
        </h3>
        <LabelBadge label={face.label} />
      </div>
      <p className="text-sm leading-relaxed text-slate-300">{LABEL_DESCRIPTION[face.label]}</p>
      <ProbabilityMeter prob={face.prob_ai_generated} threshold={model.threshold} delta={model.uncertainty_delta} />
      <HeatmapViewer
        cropBase64={face.face_crop_png_base64}
        heatmapBase64={face.heatmap_png_base64}
        method={model.explanation_method}
      />
      <p className="text-[11px] text-slate-600">
        Face detector confidence {(face.detection_score * 100).toFixed(1)}% · box [{face.bbox.join(", ")}]
      </p>
    </Panel>
  );
}
