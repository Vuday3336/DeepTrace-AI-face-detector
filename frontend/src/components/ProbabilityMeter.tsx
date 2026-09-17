import { formatPercent, meterGeometry } from "../lib/verdict";

interface Props {
  prob: number;
  threshold: number;
  delta: number;
}

/** 0–100% bar with the decision threshold and the "uncertain" band drawn on it. */
export function ProbabilityMeter({ prob, threshold, delta }: Props) {
  const g = meterGeometry(prob, threshold, delta);
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-sm">
        <span className="text-slate-400">Probability AI-generated</span>
        <span className="font-mono text-lg font-semibold text-white" data-testid="prob-value">
          {formatPercent(prob)}
        </span>
      </div>
      <div
        className="relative h-3 rounded-full bg-gradient-to-r from-real/40 via-slate-600/40 to-fake/50"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(g.marker)}
        aria-label="Probability the face is AI-generated"
      >
        {delta > 0 && (
          <div
            className="absolute inset-y-0 bg-unsure/30"
            style={{ left: `${g.bandStart}%`, width: `${g.bandEnd - g.bandStart}%` }}
            data-testid="uncertain-band"
            title="Uncertain zone"
          />
        )}
        <div className="absolute -inset-y-1 w-px bg-white/70" style={{ left: `${g.threshold}%` }} title="Decision threshold" />
        <div
          className="absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-ink bg-white shadow"
          style={{ left: `${g.marker}%` }}
          data-testid="prob-marker"
        />
      </div>
      <div className="mt-1 flex justify-between text-[11px] text-slate-500">
        <span>real</span>
        <span>threshold {formatPercent(threshold)}</span>
        <span>AI-generated</span>
      </div>
    </div>
  );
}
