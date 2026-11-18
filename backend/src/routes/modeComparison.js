import { Router } from "express";
import { readArtifact, validateExperimentIdParam } from "../artifacts.js";
import { sendCorruptArtifact, sendNotFound } from "../errors.js";

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

export default router;
