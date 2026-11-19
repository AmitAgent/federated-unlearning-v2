# Research extension: heterogeneity-aware adaptive federated unlearning (HAFU)

> **Status: partly implemented, not shown to beat the fixed baseline.**
> The opt-in calibrated stopping and the heterogeneity-adaptive controller
> exist in the code (see "Implementation Status" below). Class-aware
> distillation is not implemented. No result in this repository shows the
> adaptive method beating fixed Gradient Ascent + Knowledge Distillation; the
> measured comparison is in [`results.md`](./results.md).

## Proposed Research Extension: Heterogeneity-Aware Adaptive Federated Unlearning (HAFU)

### Motivation

The current project studies client-level federated unlearning using
Gradient Ascent (GA) and Knowledge Distillation (KD), with full
retraining as a reference baseline. A potential extension is to
investigate whether the forgetting strength and knowledge-recovery
strategy should adapt to client-data heterogeneity rather than use fixed
settings.

We use **HAFU (Heterogeneity-Aware Adaptive Federated Unlearning)** as a
working name for this research proposal. The name does not imply that
the method is novel, implemented, or proven superior. A focused
literature review and experiments are required before making a novelty
claim.

### Research Question

Can an adaptive unlearning controller that accounts for client
heterogeneity improve the trade-off between forgetting the target
client's influence and preserving the utility of retained clients,
compared with fixed-strength GA+KD?

### Proposed Method

The proposed pipeline is:

``` text
Federated Training (e.g., FedAvg)
              |
              v
Select Target / Forget Client
              |
              v
Measure Client Heterogeneity
              |
              v
Adaptive Gradient-Ascent Forgetting
              |
              v
Class-Aware Knowledge Recovery
              |
              v
Evaluate Forgetting, Utility, Privacy Risk, and Cost
              |
              v
Compare Against GA+KD and Full Retraining
```

#### 1. Client-heterogeneity analysis

Estimate how the target client's data distribution differs from the
retained-client distributions. In simulated experiments, this can use
known partition statistics; in a privacy-conscious deployment, the
method should specify what statistics can be computed without exposing
raw client data.

Possible measurements include label-distribution divergence and class
coverage. The exact metric must be selected and justified during
implementation.

#### 2. Adaptive forgetting strength

Instead of fixing the Gradient Ascent strength for every unlearning
request, adapt it according to measured forgetting and retained-client
utility. For example, the controller may increase forgetting strength
when the forgetting target is not met, and reduce it or trigger recovery
when retained-client utility degrades beyond a predefined tolerance.

A basic gradient-ascent update can be written as:

$$\theta' = \theta + \eta \lambda_f \nabla_\theta \mathcal{L}_f(\theta)$$

where $\theta$ denotes model parameters, $\eta$ is the
learning rate, $\lambda_f$ is the forgetting-strength
coefficient, and $\mathcal{L}_f$ is the loss on the target
client's forget data. The adaptive rule for $\lambda_f$ is a
research component to be defined and evaluated; this equation alone is
not a novel contribution.

#### 3. Class-aware knowledge recovery

Investigate whether the distillation objective can weight classes
according to evidence that they remain represented in retained clients.
The aim is to recover useful retained knowledge without indiscriminately
restoring behavior associated with the forgotten client.

A candidate objective is:

$$\mathcal{L}_{KD} = \sum_{c} w_c \, D_{KL}\left(p_T^c \,\|\, p_S^c\right)$$

where $p_T^c$ and $p_S^c$ are teacher and student predictive
distributions for class $c$, and $w_c$ is a class-retention weight
derived from retained-client information. This is a design hypothesis,
not a guarantee of client-level forgetting: when classes overlap across
clients, class-level weights may not distinguish which client
contributed the knowledge.

#### 4. Joint control objective

Explore a controller that balances forgetting, retained-client utility,
and privacy-risk measurements:

$$J = \alpha F + \beta U - \gamma R$$

Here, (F) represents a defined forgetting score, (U) retained-client
utility, and (R) a privacy-risk measure such as membership-inference
attack performance. The exact definitions, optimization direction,
constraints, and coefficients must be specified before experiments.
Attack metrics are empirical evidence and do not constitute a formal
guarantee that information has been erased.

### Evaluation Plan

Compare the proposed method against:

-   The original federated model before unlearning.
-   Full retraining without the target client, as a reference baseline.
-   The existing GA+KD implementation.
-   Adaptive GA without class-aware KD (ablation).
-   GA with class-aware KD but without adaptive control (ablation).
-   The complete proposed method.
-   Relevant published federated-unlearning baselines, where
    implementations and experimental conditions permit a fair
    comparison.

Evaluate under both IID and multiple non-IID client partitions. Vary the
target client and random seed, and report mean and standard deviation.

Recommended metrics:

-   Retained-client test accuracy and macro-F1.
-   Forget-client behavior and forgetting effectiveness, with clearly
    defined metrics.
-   Membership-inference attack performance, where implemented.
-   Similarity to the full-retraining reference.
-   Unlearning runtime and computational overhead.
-   Communication cost, if communication is measured by the
    implementation.
-   Performance under repeated deletion requests, if supported.

Do not infer successful unlearning from low forget-client accuracy
alone. Report the trade-off between forgetting, retained utility,
privacy-risk measurements, and efficiency.

### Research Novelty and Limitations

GA, KD, client-heterogeneity analysis, and adaptive optimization are
established research ideas. Combining them does not by itself establish
novelty. Before presenting HAFU as a novel method, review closely
related work and determine whether the specific controller, objective,
and class-aware recovery mechanism already exist.

A defensible contribution would require:

1.  A precisely specified mechanism that differs from relevant prior
    methods.
2.  A reproducible implementation and controlled experiments.
3.  Ablation studies showing which components provide benefit.
4.  Consistent improvements over appropriate baselines, including GA+KD
    and full retraining as a reference.
5.  A careful discussion of limitations and any privacy or deletion
    guarantees that are---and are not---provided.

This proposal does not claim certified or exact machine unlearning. Full
retraining without the target client's data is a useful reference, but
its guarantee depends on the training procedure and the scope of the
deletion request.

### Implementation Status

- **Implemented so far:** opt-in calibrated stopping for Gradient Ascent (`--calibrated`), and an opt-in heterogeneity-adaptive controller (`--adaptive`, `core/unlearning/adaptive.py`): it measures how distinctive the target client's label distribution is (Jensen-Shannon divergence vs. all other clients, min-max normalised) and sets the calibration threshold, ascent step cap and KD epochs/alpha from that. The mapping is a simple monotone heuristic with no theoretical guarantee. `python -m core.unlearning.compare_modes --experiment-id <id>` measures fixed vs. calibrated vs. adaptive against the retrained reference. Class-aware KD is not implemented. No result showing adaptive beats fixed is claimed until that comparison has been run and recorded.
-   **Existing project:** Federated-learning and unlearning components
    should be described as implemented only when confirmed in the
    current repository.
-   **HAFU proposal:** Not implemented or experimentally validated
    unless subsequent code and results establish otherwise.
-   **Results:** No accuracy, forgetting, privacy, runtime, or
    communication results are claimed in this proposal.
-   **Next step:** Review related literature, define the adaptive
    controller mathematically, implement it as a separate experimental
    method, and compare it against the existing GA+KD baseline using
    reproducible configurations.

### Suggested Repository Organization (after implementation)

Add files only when the corresponding functionality is implemented.
Adapt names to the repository's actual structure.

``` text
configs/
  hafu.yaml

src/
  unlearning/
    adaptive_controller.py
    heterogeneity.py
    class_aware_distillation.py
    hafu.py

scripts/
  run_hafu.py

tests/
  test_adaptive_controller.py
  test_class_aware_distillation.py

artifacts/
  hafu/
```

Do not add placeholder paths to the main README as if they already
exist. Update the main README's commands, architecture, and status only
after the code is present and tested.
