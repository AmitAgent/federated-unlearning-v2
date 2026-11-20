# core/

The Python ML code for the Federated Unlearning Lab. The Express backend runs these modules as `python -m core.<module>.cli`, and every stage writes its results to `experiments/<id>/`.

| Package | What it does |
|---------|--------------|
| `datasets/` | Dataset interface and loaders for MNIST, CIFAR-10, CIFAR-100 and AG News (`manifest.json` holds their metadata) |
| `partitioning/` | Seeded IID and Dirichlet (Non-IID) splits across clients, built from real labels when available |
| `federated/` | FedAvg: client training, server aggregation loop, models (small Vision Transformer, text transformer) and storage |
| `unlearning/` | Gradient Ascent, Knowledge Distillation, calibrated and adaptive modes, baselines (GradDiff, NegGrad+, retain-only fine-tune), FedEraser, and `compare_modes` |
| `evaluation/` | Global, forget-client and retained-client accuracy, membership inference, the full-retraining baseline and the original / unlearned / retrained comparison |
| `compute/` | Compute backend interface, the four backends (College GPU, Colab, Local, Demo), the Auto fallback chain, and the remote compute server |
| `distribution/` | Model releases and the device update policy (Auto update, Request then force) |
| `acceptance/` | Machine-checkable acceptance checks for stored experiments |
| `provenance.py` | Reproducibility metadata (git commit, settings) attached to results |

Install the dependencies with `pip install -r core/requirements.txt` and run the tests from the repo root with `python -m pytest tests -q`. See the root [`README.md`](../README.md) for the full pipeline and commands.
