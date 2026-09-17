import { useId, useState } from "react";
import type { ExplanationMethod } from "../api/types";
import { EXPLANATION_TEXT } from "../lib/verdict";

interface Props {
  cropBase64: string;
  heatmapBase64: string | null;
  method: ExplanationMethod;
}

const png = (b64: string) => `data:image/png;base64,${b64}`;

/** Side-by-side face crop and heatmap overlay with an opacity slider. */
export function HeatmapViewer({ cropBase64, heatmapBase64, method }: Props) {
  const [opacity, setOpacity] = useState(0.6);
  const sliderId = useId();
  return (
    <div>
      <div className="grid grid-cols-2 gap-3">
        <figure>
          <img src={png(cropBase64)} alt="Face crop the model analysed" className="w-full rounded-lg" />
          <figcaption className="mt-1 text-center text-xs text-slate-500">What the model saw</figcaption>
        </figure>
        <figure>
          <div className="relative">
            <img src={png(cropBase64)} alt="" className="w-full rounded-lg" />
            {heatmapBase64 && (
              <img
                src={png(heatmapBase64)}
                alt="Heatmap overlay"
                className="absolute inset-0 h-full w-full rounded-lg"
                style={{ opacity }}
                data-testid="heatmap-overlay"
              />
            )}
          </div>
          <figcaption className="mt-1 text-center text-xs text-slate-500">
            {heatmapBase64 ? "Regions that influenced the score" : "Heatmap not requested"}
          </figcaption>
        </figure>
      </div>
      {heatmapBase64 && (
        <div className="mt-3">
          <label htmlFor={sliderId} className="mb-1 flex justify-between text-xs text-slate-400">
            <span>Heatmap opacity</span>
            <span className="font-mono">{Math.round(opacity * 100)}%</span>
          </label>
          <input
            id={sliderId}
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="slider"
          />
          <p className="mt-2 text-xs leading-relaxed text-slate-400">{EXPLANATION_TEXT[method]}</p>
        </div>
      )}
    </div>
  );
}
