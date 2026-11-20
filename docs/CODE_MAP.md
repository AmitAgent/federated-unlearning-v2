# Code map

Where each part of the project lives.

## How a request flows

```
React dashboard (frontend/)  ->  Express API (backend/)  ->  ComputeBackend  ->  Python engine (core/)
                                                   |
                                                   +-> experiments/<id>/*.json, *.pt   (everything is saved here)
```

The frontend contains no ML logic. The backend routes thin requests to a
`ComputeBackend` (local machine, a remote GPU server, Google Colab, or the
read-only demo backend) and reads results back from `experiments/<id>/`.

## Python engine (`core/`)

| Area | Module | What it does |
|---|---|---|
| Datasets | `core/datasets/` | Loaders and `manifest.json` for MNIST, CIFAR-10, CIFAR-100 and AG News |
| Partitioning | `core/partitioning/engine.py`, `storage.py`, `cli.py` | IID and non-IID (Dirichlet) splits across clients, built from real labels; saved per experiment |
| Federated training | `core/federated/model.py` | Two small transformers: a Vision Transformer (`transformer`) for images and `text_transformer` for text. `make_optimizer` picks SGD for the first and AdamW for the second |
| | `client.py`, `fedavg.py`, `server.py` | Local training, sample-weighted FedAvg, the round loop (`fedavg_train`) with optional read-only `round_callback` |
| | `storage.py`, `cli.py` | Saves `fl_history.json` and the global model checkpoint |
| Unlearning (GA + KD) | `core/unlearning/gradient_ascent.py` | Gradient ascent on the target client's data, with clipping and plateau early stopping |
| | `knowledge_distillation.py` | Hinton-style distillation: the pre-ascent model teaches the post-ascent model on all other clients |
| | `unlearning_engine.py`, `storage.py`, `cli.py` | `start_unlearning` / `get_unlearning_status`, chains GA then KD |
| Calibration and adaptation | `calibration.py`, `heterogeneity.py`, `adaptive.py` | Opt-in calibrated stopping and the heterogeneity-adaptive controller (see `docs/RESEARCH_EXTENSION.md`) |
| Other methods | `baselines.py` | GradDiff, NegGrad+ (Kurmanji et al., 2023), retain-only fine-tune |
| | `federaser.py`, `federaser_cli.py` | FedEraser (Liu et al., 2021): replays recorded training rounds without the target client (see `docs/FEDERASER.md`) |
| Method comparison | `compare_modes.py` | Runs the methods above against the retrained reference and writes `mode_comparison.json` |
| Evaluation | `core/evaluation/accuracy.py`, `mia.py`, `evaluation_engine.py` | Global, forget-client and retained-client accuracy, a loss-threshold membership inference attack, runtime |
| Retraining baseline | `retraining.py`, `retraining_cli.py`, `comparison.py` | Full retraining without the target client under the original protocol; original vs unlearned vs retrained comparison |
| Compute backends | `core/compute/` | One `ComputeBackend` interface, four backends (local, college GPU, Colab, demo) and an Auto fallback. See `docs/COMPUTE_BACKENDS.md` |
| Model updates | `core/distribution/` | Release rules for publishing an unlearned model to simulated devices. See `docs/MODEL_UPDATES.md` |
| Acceptance checks | `core/acceptance/` | `python -m core.acceptance.cli --all` checks stored experiments against the machine-checkable parts of the checklist |

## Backend API (`backend/src/routes/`)

| Route | File | Purpose |
|---|---|---|
| `/api/datasets` | `datasets.js` | List datasets |
| `/api/partition` | `partition.js` | Create and read a partition |
| `/api/training` | `training.js` | Run or read federated training |
| `/api/unlearning` | `unlearning.js` | Run or read unlearning |
| `/api/evaluation` | `evaluation.js` | Accuracy, MIA, runtime |
| `/api/retraining` | `retraining.js` | Retraining baseline and comparison |
| `/api/mode-comparison` | `modeComparison.js` | Method Comparison table |
| `/api/experiments` | `experiments.js` | Experiment registry and history |
| `/api/compute` | `compute.js` | Backend status and selection |
| `/api/releases` | `releases.js` | Model update releases |

## Dashboard pages (`frontend/src/pages/`)

| Page | File |
|---|---|
| Overview | `Overview.jsx` |
| Setup | `ExperimentSetup.jsx` |
| Data Distribution | `DataDistribution.jsx` |
| Federated Learning | `FederatedLearning.jsx` |
| Client Selection | `ClientSelection.jsx` |
| Unlearning Process | `Unlearning.jsx` |
| Results | `Evaluation.jsx` |
| Retraining Comparison | `RetrainingComparison.jsx` |
| Method Comparison | `ModeComparison.jsx` |
| Experiment History | `ExperimentHistory.jsx` |
| Model Updates | `ModelUpdates.jsx` |
| Professor Demo | `ProfessorDemo.jsx` |

Shared pieces (client cards, distribution heatmap, FedAvg diagram, round
progress, metric charts, pipeline mini-map) are in `frontend/src/components/`.

## Tests (`tests/`)

Run everything with `python -m pytest tests -q`. Test files are named after what
they cover (`test_partitioning.py`, `test_federated.py`, `test_unlearning.py`,
`test_baselines.py`, `test_federaser.py`, `test_retraining.py`, and so on).
