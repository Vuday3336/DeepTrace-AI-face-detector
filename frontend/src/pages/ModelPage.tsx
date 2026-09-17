import { api } from "../api/client";
import type { ModelInfo, RobustnessRow, TestSetMetrics } from "../api/types";
import { ErrorAlert, Panel, Spinner } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { EXPLANATION_TEXT } from "../lib/verdict";

const f3 = (v: number | null | undefined) => (v === null || v === undefined ? "—" : v.toFixed(3));

function MetricsTable({ sets }: { sets: Record<string, TestSetMetrics> }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-left text-sm">
        <thead className="text-xs uppercase tracking-wide text-slate-500">
          <tr>
            {["Test set", "n", "ROC-AUC (95% CI)", "Bal. acc.", "Precision", "Recall", "F1", "FPR", "ECE"].map((h) => (
              <th key={h} className="border-b border-line px-2 py-2 font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Object.entries(sets).map(([name, m]) => (
            <tr key={name} className="border-b border-line/60" title={m.description}>
              <td className="px-2 py-2">
                <div className="font-mono text-xs text-white">{name}</div>
                <div className="text-[11px] text-slate-500">{m.description}</div>
              </td>
              <td className="px-2 py-2 font-mono">{m.n}</td>
              <td className="px-2 py-2 font-mono">
                {f3(m.roc_auc)}
                {m.roc_auc_ci95 && <span className="text-slate-500"> [{f3(m.roc_auc_ci95.low)}, {f3(m.roc_auc_ci95.high)}]</span>}
              </td>
              {[m.balanced_accuracy, m.precision, m.recall, m.f1, m.fpr, m.ece].map((v, i) => (
                <td key={i} className="px-2 py-2 font-mono">{f3(v)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RobustnessTable({ rows }: { rows: RobustnessRow[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[420px] text-left text-sm">
        <thead className="text-xs uppercase tracking-wide text-slate-500">
          <tr>
            {["Perturbation", "Level", "Bal. acc.", "ROC-AUC"].map((h) => (
              <th key={h} className="border-b border-line px-2 py-2 font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.family}-${r.level}`} className="border-b border-line/60">
              <td className="px-2 py-1.5">{r.family === "none" ? "clean" : r.family}</td>
              <td className="px-2 py-1.5 font-mono">{r.family === "none" ? "—" : r.level}</td>
              <td className="px-2 py-1.5 font-mono">{f3(r.balanced_accuracy)}</td>
              <td className="px-2 py-1.5 font-mono">{f3(r.roc_auc)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Dl({ items }: { items: [string, string | number][] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
      {items.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-slate-500">{k}</dt>
          <dd className="font-mono text-slate-200">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function Content({ info }: { info: ModelInfo }) {
  const metrics = info.metrics;
  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-2">
        <Panel className="space-y-4">
          <h2 className="font-semibold text-white">Model</h2>
          <Dl items={[
            ["Name", info.model.name],
            ["Version", info.model.version],
            ["Backbone", info.model.timm_name],
            ["Training", info.model.frozen_backbone ? "frozen backbone + linear probe" : "full fine-tuning"],
            ["Input", `${info.model.input_size}×${info.model.input_size} face crop`],
          ]} />
          <p className="text-xs text-slate-400">{EXPLANATION_TEXT[info.explanation.method]}</p>
        </Panel>
        <Panel className="space-y-4">
          <h2 className="font-semibold text-white">Decision settings</h2>
          <Dl items={[
            ["Threshold", info.decision.threshold.toFixed(4)],
            ["Uncertain band", `± ${info.decision.uncertainty_delta}`],
            ["Target false-positive rate", `${(info.decision.target_fpr * 100).toFixed(0)}% of real photos`],
            ["Temperature", info.decision.temperature.toFixed(3)],
          ]} />
          <p className="text-xs text-slate-400">
            Fitted on a held-out calibration split, never on test data. The threshold limits how often real photos are
            wrongly flagged, because a false accusation is the most harmful error.
          </p>
        </Panel>
      </div>

      <Panel className="space-y-4">
        <h2 className="font-semibold text-white">Evaluation</h2>
        {metrics.status === "evaluated" ? (
          <>
            <MetricsTable sets={metrics.test_sets} />
            {metrics.robustness &&
              Object.entries(metrics.robustness).map(([set, rows]) => (
                <div key={set} className="space-y-2">
                  <h3 className="text-sm text-slate-300">Robustness — {set}</h3>
                  <RobustnessTable rows={rows} />
                </div>
              ))}
          </>
        ) : (
          <p className="text-sm text-amber-200">
            No evaluation results are bundled with this model yet. Metrics appear here only after the evaluation
            scripts have been run — they are never typed in by hand.
          </p>
        )}
      </Panel>

      <div className="grid gap-6 md:grid-cols-2">
        <Panel className="space-y-3">
          <h2 className="font-semibold text-white">Known limitations</h2>
          <ul className="list-disc space-y-2 pl-5 text-sm text-slate-300">
            {info.known_limitations.map((l) => <li key={l}>{l}</li>)}
          </ul>
        </Panel>
        <Panel className="space-y-3">
          <h2 className="font-semibold text-white">Training data & preprocessing</h2>
          <Dl items={[...Object.entries(info.training_data), ...Object.entries(info.preprocessing)]} />
        </Panel>
      </div>
    </div>
  );
}

export function ModelPage() {
  const { data, error, loading } = useApi(() => api.modelInfo(), []);
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-white">About the model</h1>
      {loading && <Spinner label="Loading model card…" />}
      {error && <ErrorAlert error={error} />}
      {data && <Content info={data} />}
    </div>
  );
}
