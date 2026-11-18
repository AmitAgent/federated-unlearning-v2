import { Router } from "express";
import { readArtifact, isValidExperimentId, validateExperimentIdParam } from "../artifacts.js";
import { runEngine } from "../runEngine.js";
import { sendCorruptArtifact, sendInvalidRequest, sendNotFound } from "../errors.js";

const router = Router();
router.param("experimentId", validateExperimentIdParam);

// GET /api/mode-comparison/:experimentId -> re-read the stored
// experiments/<id>/mode_comparison.json written by
// `python -m core.unlearning.compare_modes` (fixed / calibrated /
// adaptive GA+KD plus the GradDiff and retain-only baselines, each
// scored against the retrained reference). Read-only: no Python, no
// training, nothing is written.
router.get("/:experimentId", (req, res) => {
  const { experimentId } = req.params;
  const artifact = readArtifact(experimentId, "mode_comparison.json");
  if (artifact.state === "missing") {
    return sendNotFound(
      res,
      `No method comparison found for experiment "${experimentId}". Run core.unlearning.compare_modes first.`,
    );
  }
  if (artifact.state === "corrupt") return sendCorruptArtifact(res, "method comparison", experimentId);
  res.status(200).json({ comparison: artifact.data });
});

const WHICH = ["all", "core", "baselines"];
const OWN_MODES = ["fixed", "calibrated", "adaptive"];

// POST /api/mode-comparison -> run the method comparison (our modes and/or
// the GradDiff, NegGrad+ and retain-only baselines) on the chosen compute
// backend, write experiments/<id>/mode_comparison.json, and return it. This
// is the same work as `python -m core.unlearning.compare_modes`. It can take
// a long time (a transformer run per method); no time limit is applied
// unless ENGINE_TIMEOUT_MS is set.
router.post("/", (req, res) => {
  const { experiment_id, client_id, which, modes, backend } = req.body || {};
  if (!experiment_id || !isValidExperimentId(String(experiment_id))) {
    return sendInvalidRequest(res, "A valid experiment_id is required.");
  }
  if (which !== undefined && !WHICH.includes(which)) {
    return sendInvalidRequest(res, `which must be one of: ${WHICH.join(", ")}.`);
  }
  if (modes !== undefined && !(Array.isArray(modes) && modes.every((m) => OWN_MODES.includes(m)))) {
    return sendInvalidRequest(res, `modes must be a list drawn from: ${OWN_MODES.join(", ")}.`);
  }
  if (client_id !== undefined && client_id !== null && !Number.isInteger(Number(client_id))) {
    return sendInvalidRequest(res, "client_id must be a whole number.");
  }

  const args = ["-m", "core.unlearning.compare_modes", "--experiment-id", String(experiment_id), "--json"];
  if (client_id !== undefined && client_id !== null) args.push("--client-id", String(client_id));
  if (which !== undefined) args.push("--which", which);
  if (modes && modes.length > 0) args.push("--modes", modes.join(","));
  if (backend !== undefined) args.push("--backend", String(backend));

  runEngine({
    args,
    stage: "comparison",
    res,
    experimentId: String(experiment_id),
    onSuccess: (parsed) => res.status(200).json(parsed),
  });
});

export default router;
