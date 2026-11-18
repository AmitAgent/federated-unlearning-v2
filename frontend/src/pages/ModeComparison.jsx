import { useEffect, useMemo, useState } from "react";
import { apiFetch } from "../lib/api.js";
import ErrorPanel from "../components/ErrorPanel.jsx";
import ProvenanceBadge from "../components/ProvenanceBadge.jsx";

// Method Comparison -- every unlearning mode next to the retrained
// reference, read from experiments/<id>/mode_comparison.json (written
// by `python -m core.unlearning.compare_modes`). Read-only: it never
// starts a run. Gaps are computed here from the stored accuracies, so
// older files without a gap field still display correctly.

const MODE_ORDER = [
  ["fixed", "Fixed GA + KD", "Gradient ascent for a fixed step count, then distillation repair."],
  ["calibrated", "Calibrated GA + KD", "Ascent stops by the calibrated rule (it may take 0 steps), then distillation repair."],
  ["adaptive", "Adaptive GA + KD", "Ascent and repair settings chosen from client heterogeneity."],
  ["graddiff", "GradDiff (untuned)", "Baseline: ascent on the target client and descent on retained clients in one loss, default settings."],
  ["retain_finetune", "Retain-only fine-tune", "Baseline: no forgetting objective, just keeps training on retained clients."],
];

const pct = (v) => (typeof v === "number" ? `${(v * 100).toFixed(2)}%` : "—");
const gap = (v, ref) => (typeof v === "number" && typeof ref === "number" ? Math.abs(v - ref) : null);
const gapText = (g) => (g === null ? "—" : `${(g * 100).toFixed(2)} pts`);

function buildRows(comparison) {
  const ref = comparison.retrained_reference || {};
  const modes = comparison.modes || {};
  const rows = [
    {
      key: "retrain",
      label: "Retrain (reference)",
      note: "Full retraining without the target client.",
      global: ref.global_accuracy,
      forget: ref.forget_client_accuracy,
      retained: ref.retained_client_accuracy,
      mia: ref.mia_success,
      steps: null,
      seconds: null,
      isReference: true,
    },
  ];
  for (const [key, label, note] of MODE_ORDER) {
    const m = modes[key];
    if (!m) continue;
    rows.push({
      key,
      label,
      note,
      global: m.global_accuracy,
      forget: m.forget_client_accuracy,
      retained: m.retained_client_accuracy,
      mia: m.mia_success,
      steps: m.ga_steps_completed ?? null,
      stepsKind: key === "retain_finetune" ? "fine-tune steps" : "ascent steps",
      seconds: m.duration_seconds ?? null,
      gapForget: gap(m.forget_client_accuracy, ref.forget_client_accuracy),
      gapRetained: gap(m.retained_client_accuracy, ref.retained_client_accuracy),
      gapGlobal: gap(m.global_accuracy, ref.global_accuracy),
    });
  }
  return rows;
}

function closestKey(rows, field) {
  let best = null;
  for (const r of rows) {
    if (r.isReference || r[field] === null || r[field] === undefined) continue;
    if (best === null || r[field] < best.value) best = { key: r.key, value: r[field] };
  }
  return best ? best.key : null;
}

export default function ModeComparison() {
  const [experiments, setExperiments] = useState([]);
  const [experimentId, setExperimentId] = useState("");
  const [comparison, setComparison] = useState(null);
  const [status, setStatus] = useState("loading"); // loading | ready | error
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    apiFetch("/api/experiments")
      .then((body) => {
        if (cancelled) return;
        const list = body.experiments || [];
        setExperiments(list);
        const preferred = list.find((e) => e.experiment_id === "mnist-real") || list[0];
        if (preferred) setExperimentId(preferred.experiment_id);
        else setStatus("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err);
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!experimentId) return undefined;
    let cancelled = false;
    setStatus("loading");
    setError(null);
    setComparison(null);
    apiFetch(`/api/mode-comparison/${encodeURIComponent(experimentId)}`)
      .then((body) => {
        if (cancelled) return;
        setComparison(body.comparison);
        setStatus("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err);
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [experimentId]);

  const rows = useMemo(() => (comparison ? buildRows(comparison) : []), [comparison]);
  const best = useMemo(
    () => ({
      forget: closestKey(rows, "gapForget"),
      retained: closestKey(rows, "gapRetained"),
      global: closestKey(rows, "gapGlobal"),
    }),
    [rows],
  );
  const cell = (rowKey, field) => (best[field] === rowKey ? "bg-emerald-50 font-semibold text-emerald-800" : "text-gray-900");

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-2xl font-semibold text-gray-900">Method Comparison</h2>
        <ProvenanceBadge kind="precomputed" />
      </div>
      <p className="mt-2 max-w-3xl text-sm text-gray-600">
        Every unlearning method scored against the retrained reference on the same trained model and the same
        evaluation code. The gap columns show distance from the reference; the closest method in each gap
        column is highlighted.
      </p>

      <div className="mt-4 flex items-center gap-2 text-sm">
        <label htmlFor="mode-exp" className="font-medium text-gray-700">
          Experiment
        </label>
        <select
          id="mode-exp"
          value={experimentId}
          onChange={(e) => setExperimentId(e.target.value)}
          className="rounded-md border border-gray-300 bg-white px-2 py-1 font-mono text-xs"
        >
          {experiments.map((e) => (
            <option key={e.experiment_id} value={e.experiment_id}>
              {e.experiment_id}
            </option>
          ))}
        </select>
        {comparison && (
          <span className="text-xs text-gray-500">
            dataset {comparison.dataset ?? "—"} · target client {comparison.target_client_id ?? "—"}
          </span>
        )}
      </div>

      {status === "loading" && <p className="mt-6 text-sm text-gray-500">Loading…</p>}
      {status === "error" && (
        <div className="mt-6">
          <ErrorPanel error={error} />
        </div>
      )}
      {status === "ready" && !comparison && experiments.length === 0 && (
        <p className="mt-6 text-sm text-gray-500">No experiments found yet.</p>
      )}

      {status === "ready" && comparison && (
        <>
          <div className="mt-6 overflow-x-auto rounded-lg border border-gray-200 bg-white">
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">
                <tr>
                  <th className="px-4 py-3">Method</th>
                  <th className="px-4 py-3">Forget acc</th>
                  <th className="px-4 py-3">Retained acc</th>
                  <th className="px-4 py-3">Global acc</th>
                  <th className="px-4 py-3">MIA</th>
                  <th className="px-4 py-3">Steps</th>
                  <th className="px-4 py-3">Time</th>
                  <th className="px-4 py-3">Gap: forget</th>
                  <th className="px-4 py-3">Gap: retained</th>
                  <th className="px-4 py-3">Gap: global</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {rows.map((r) => (
                  <tr key={r.key} className={r.isReference ? "bg-indigo-50/40" : ""}>
                    <td className="px-4 py-3">
                      <div className="font-medium text-gray-900">{r.label}</div>
                      <div className="max-w-xs text-xs text-gray-500">{r.note}</div>
                    </td>
                    <td className="px-4 py-3 tabular-nums">{pct(r.forget)}</td>
                    <td className="px-4 py-3 tabular-nums">{pct(r.retained)}</td>
                    <td className="px-4 py-3 tabular-nums">{pct(r.global)}</td>
                    <td className="px-4 py-3 tabular-nums">{typeof r.mia === "number" ? r.mia.toFixed(3) : "—"}</td>
                    <td className="px-4 py-3 tabular-nums">
                      {r.steps === null ? "—" : r.steps}
                      {r.steps !== null && <div className="text-[11px] text-gray-400">{r.stepsKind}</div>}
                    </td>
                    <td className="px-4 py-3 tabular-nums">{r.seconds === null ? "—" : `${r.seconds.toFixed(1)} s`}</td>
                    <td className={`px-4 py-3 tabular-nums ${r.isReference ? "text-gray-400" : cell(r.key, "forget")}`}>
                      {r.isReference ? "—" : gapText(r.gapForget)}
                    </td>
                    <td className={`px-4 py-3 tabular-nums ${r.isReference ? "text-gray-400" : cell(r.key, "retained")}`}>
                      {r.isReference ? "—" : gapText(r.gapRetained)}
                    </td>
                    <td className={`px-4 py-3 tabular-nums ${r.isReference ? "text-gray-400" : cell(r.key, "global")}`}>
                      {r.isReference ? "—" : gapText(r.gapGlobal)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ul className="mt-4 max-w-3xl list-disc space-y-1 pl-5 text-xs text-gray-600">
            <li>One run, one target client, one seed: small differences are not statistically meaningful.</li>
            <li>GradDiff and retain-only fine-tune use default settings, not settings tuned for them.</li>
            <li>A near-chance MIA (about 0.5) cannot separate the methods.</li>
            <li>Neither the baselines nor GA + KD give a formal unlearning guarantee.</li>
          </ul>
        </>
      )}
    </div>
  );
}
