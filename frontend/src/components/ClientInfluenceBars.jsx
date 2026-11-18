// "Client Influence" before/after bars (Dashboard Spec Ch.19). Values
// are always real measured accuracy -- retained-client accuracy from
// Phase 09's Knowledge Distillation (before/after distillation), and
// the target client's accuracy from Phase 08's Gradient Ascent
// (before ascent / after ascent) shown alongside for the "one client's
// influence drops while the others are preserved" contrast the spec
// asks for. No influence score is invented -- these are the actual
// implemented accuracy metrics used as the influence proxy.
function Row({ label, value, highlight }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="flex items-center gap-3">
      <span className={`w-28 shrink-0 text-xs font-medium ${highlight ? "text-rose-700" : "text-gray-600"}`}>
        {label}
        {/* Not color alone -- the target row is also labeled in text. */}
        {highlight && <span className="font-normal text-rose-500"> (target)</span>}
      </span>
      <div className="h-3 flex-1 overflow-hidden rounded-full bg-gray-100">
        <div
          className={`h-full rounded-full ${highlight ? "bg-rose-500" : "bg-indigo-500"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-14 shrink-0 text-right text-xs font-semibold text-gray-900">
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function Column({ title, targetClientId, values, ids }) {
  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500">{title}</h3>
      <div className="mt-3 space-y-2.5">
        {ids.map((id) => (
          <Row key={id} label={`Client ${id}`} value={values[id] ?? 0} highlight={id === targetClientId} />
        ))}
      </div>
    </div>
  );
}

export default function ClientInfluenceBars({ targetClientId, influence, legacy }) {
  // New runs: every client measured on the same checkpoints (original vs
  // final unlearned model). Older stored runs only have the mixed legacy
  // fields, so flag them instead of presenting them as a clean comparison.
  const hasConsistent = Boolean(influence?.original && influence?.final);
  const before = hasConsistent ? influence.original : { ...legacy.retainedBefore, [targetClientId]: legacy.targetBefore };
  const after = hasConsistent ? influence.final : { ...legacy.retainedAfter, [targetClientId]: legacy.targetAfter };
  const ids = Object.keys(before).map(Number).sort((a, b) => a - b);

  return (
    <section aria-labelledby="client-influence-heading" className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
      <h2 id="client-influence-heading" className="text-base font-semibold text-gray-900">
        Client Influence — Before vs After
      </h2>
      <p className="mt-1 text-sm text-gray-500">
        Accuracy on each client&rsquo;s own data: original global model vs the final unlearned model.
        The target client&rsquo;s accuracy should drop; every retained client&rsquo;s should stay high.
      </p>
      {!hasConsistent && (
        <p className="mt-2 text-xs text-amber-700">
          Stored before this update: &ldquo;Before&rdquo; mixes pre-ascent (target) and post-ascent
          (retained) values. Re-run unlearning for a consistent comparison.
        </p>
      )}
      <div className="mt-5 grid gap-6 sm:grid-cols-2">
        <Column title="Before (original model)" targetClientId={targetClientId} values={before} ids={ids} />
        <Column title="After (unlearned model)" targetClientId={targetClientId} values={after} ids={ids} />
      </div>
    </section>
  );
}
