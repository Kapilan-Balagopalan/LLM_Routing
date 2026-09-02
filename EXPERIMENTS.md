# Experiment record

This file preserves research decisions that cannot be recovered reliably from
source code alone. Add one entry for every result that changes the next research
decision.

## Branch map

| Branch | Purpose |
|---|---|
| `main` | Real ARC-Easy routing and supervised skyline baseline |
| `experiment/synthetic-sanity` | Controlled nonlinear `x1*x2` sanity check |
| `experiment/residual-diagnostics` | Real-data model-specification diagnostics |
| `experiment/prompt-embedding` | Incremental semantic prompt-feature test |
| `backup/current-combined` | Recovery snapshot before branch separation |

Initial combined checkpoint: tag `current-combined-v1`, commit `3905bbf`.

## Fixed interpretation

- `Y=1`: weak and strong extracted answers disagree.
- `Y=0`: they agree.
- Action `1`: route to strong and observe `Y`.
- Action `0`: use weak and do not observe `Y`.
- Routing accuracy means agreement with the cached strong-model answer, not ARC
  ground-truth accuracy.

## Established experimental history

### Weak model and features

- The original Pythia-2.8B weak model was replaced by
  `Qwen/Qwen2.5-0.5B-Instruct` after experiments showed that Pythia answer
  quality was too low for useful disagreement routing.
- Tested context ideas included low-dimensional hidden-state PCA plus
  uncertainty, prompt-only embeddings, uncertainty-only contexts, and richer
  candidate features.
- The current cached context retains the two sources that were most useful in
  ablations: weak-model uncertainty and internal hidden states. It contains 14
  uncertainty features plus 64 fixed whitened-PCA features.
- The current PCA is fixed across rounds and may use all collected hidden states
  without labels. This is an explicit transductive, outcome-free choice.

### Current real-data player baseline

- ETC: XGBoost, 100 initial tastes, then freeze.
- CBPSide: regularized linear logistic model and confidence-based routing;
  class-balanced bootstrap of 10 agreements and 10 disagreements, capped at 50
  total tastes.
- IGW: XGBoost, fixed `gamma=32`, `mu=2`, no fixed minimum beyond the same
  class-balanced bootstrap, inverse-propensity weights capped at 10.
- Random routing is matched to ETC traffic.
- Default losses: `l01 = [1.82, 2.22, 2.67, 3.33]`, `l11 = 1`.

### Nonlinear synthetic sanity check

Data are generated independently with two features and

```text
P(Y=1|X) = sigmoid(x1*x2).
```

This deliberately violates a linear-logistic specification. Small HGB closely
approached the known Bayes oracle while linear logistic was near random,
confirming that the supervised and routing pipeline can expose strong nonlinear
structure when it is present. Current synthetic defaults are 3,000 supervised
training samples, 2,000 independent test/online samples, 500 ETC tastes, and 10
loss-derived thresholds spanning approximately `alpha=0.95` to `0.05`.

### Real-data residual diagnostic, 2026-09-02

Configuration:

- 2,351 eligible ARC-Easy examples;
- 78-dimensional cached context;
- five-fold stratified out-of-fold predictions;
- selected capacities: HGB-350 and MLP-8.

| Model | AUC | Log loss | Brier | 10-bin ECE | Significant residual bins |
|---|---:|---:|---:|---:|---:|
| Linear logistic | 0.6646 | 0.6351 | 0.2228 | 0.0279 | 0/10 |
| HGB-350 | 0.6520 | 0.6436 | 0.2264 | 0.0289 | 3/10 |
| MLP-8 | 0.6415 | 0.6640 | 0.2330 | 0.0709 | 6/10 |

Interpretation: logistic was best calibrated and best on the main probability
metrics. HGB showed localized calibration errors and a compressed probability
range. MLP-8 was overconfident, underpredicting disagreement at low scores and
overpredicting it at high scores. Residual-versus-probability plots assess
calibration but do not establish the stronger condition `E[Y-p(X)|X]=0`.

Implemented next diagnostic: nested cross-fit an HGB residual regressor on the
original features and test whether it predicts held-out logistic residuals
better than a zero-residual baseline. For every outer fold, the evaluated rows
are absent from both the logistic and residual-learner fits; inner logistic
cross-fitting constructs the residual targets used for residual-learner
training. A shuffled-training-residual permutation reference tests whether the
observed held-out MSE improvement exceeds chance. Record the resulting metrics
below after the full-cache run.

### 2026-09-02 — prompt-embedding hypothesis

Branch: `experiment/prompt-embedding`

Research question: do outcome-free semantic prompt features explain
weak/strong disagreement beyond uncertainty and weak-model hidden states, and
do nonlinear estimators benefit more than logistic regression?

Design:

- frozen default encoder: `sentence-transformers/all-MiniLM-L6-v2`;
- encoder input: question and labeled choices only;
- no gold answer, weak/strong answer, response text, or disagreement outcome;
- raw embedding saved once in an ID-aligned versioned sidecar;
- fixed transductive outcome-free PCA with 32 components;
- contexts compared on identical folds: current 78D, prompt 32D, combined 110D;
- supervised models unchanged from the baseline comparison;
- incremental diagnostic: current-context logistic base, prompt-only HGB
  residual learner, nested cross-fitting, and 100 shuffled-training-residual
  permutations.

Decision rule: prompt information is incrementally useful only if the held-out
residual MSE improvement is positive and the permutation evidence is credible.
Whether nonlinear models gain more must be judged from their out-of-fold change
relative to logistic across the same context and folds.

## Template for future entries

```text
### YYYY-MM-DD — short experiment name

Branch:
Commit or tag:
Research question:
Cache/dataset:
Exact command:
Seed:
Parameters:
Primary metrics:
Artifact location:
Interpretation:
Decision and next step:
```
