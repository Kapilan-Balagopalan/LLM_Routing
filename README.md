# Offline LLM-routing simulation

This repository replays weak-versus-strong LLM routing policies from collected
benchmark caches. It is CPU-only: the simulation does not load an LLM, contact
Hugging Face, require an `HF_TOKEN`, or need a GPU.

The active work on `experiment/boolq-cbpside-beta1` performs one complete 138D
BoolQ run with exploration scaled to the number of eligible online samples.
CBPSide uses empirical scale 0.25 and cap 0.5, ETC uses
`ceil(n^(2/3))` forced tastes, and IGW uses `gamma=sqrt(n)`. The real routing
target is cached weak/strong disagreement. A separate synthetic-label positive
control is available for implementation sanity checks; it must not be
interpreted as real benchmark routing performance.

For experiment history and conclusions, read [EXPERIMENTS.md](EXPERIMENTS.md).
For module boundaries and data flow, read [ARCHITECTURE.md](ARCHITECTURE.md).

## Routing definitions

- `Y=1`: the extracted weak- and strong-model answers disagree.
- `Y=0`: their extracted answers agree.
- Action `0`: retain the weak answer; the player receives no outcome feedback.
- Action `1`: route to the strong model; the disagreement outcome is revealed.
- A **taste** is an action-1 round whose outcome becomes available for learning.
- Routing accuracy is agreement with the cached strong-model answer. Benchmark
  gold answers are not routing labels, online feedback, or the reported accuracy
  reference.

For `l11=1`, the empirical asymmetric decision loss reported in analysis is

```text
decision loss = routing_rate + l01 * (1 - routing_accuracy)
realized total cost = n * decision loss
```

Raw routing accuracy should not be used alone to compare policies with different
routing rates.

## Data

The active `boolq-routing-cache-full.zip` uses schema v2 and contains 12,697
collected rows, of which 12,648 are eligible. Rows are ineligible when either
model answer could not be parsed. The earlier ARC-Easy cache is retained for
reproducibility.

The BoolQ cache stores a 138-dimensional context:

| Block | Dimensions | Description |
|---|---:|---|
| Uncertainty | 10 | Binary-choice and token-probability summaries |
| Hidden-state PCA | 64 | Whitened PCA of weak-model hidden states |
| Prompt-embedding PCA | 64 | Whitened PCA of context-free prompt embeddings |

The prompt-routing code finds all blocks through
`manifest.json -> context_blocks`; it never hardcodes column numbers. These
block ranges are authoritative. This matters because the BoolQ manifest's free
text `context_definition` still says 14 uncertainty features even though its
block range correctly specifies 10. The PCA is fixed, transductive, and
outcome-free.

## Setup

Clone the repository and select the BoolQ confidence-scale follow-up branch:

```powershell
git clone https://github.com/Kapilan-Balagopalan/LLM_Routing.git
Set-Location LLM_Routing
git switch experiment/boolq-cbpside-beta1
```

Create the environment and install the project:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
```

## Active BoolQ sample-size-scaled experiment

The next experiment selects all complete blocks through
`manifest.context_blocks`: 10 uncertainty features, 64 hidden-state PCA
components, and 64 prompt-embedding PCA components, for 138 dimensions. With
all 12,648 eligible samples, the sample-size rules resolve to ETC taste budget
543 and IGW gamma 112.463327356076. Explicit command-line values still override
these derived defaults.

```powershell
simulate-llm-routing `
  --cache .\boolq-routing-cache-full.zip `
  --context-profile all-features `
  --outcome-source cached `
  --experiment all `
  --l01-values 1.8182 1.9149 2.0225 2.1429 2.2785 2.4324 2.6087 2.8125 3.0508 3.3333 `
  --hgb-max-leaf-nodes 15 `
  --cbpside-matrix-regularization 1 `
  --cbpside-beta-scale 0.25 `
  --cbpside-max-confidence-radius 0.5 `
  --output-dir .\boolq-all-features-138-scaled-exploration-results
```

The context-profile default remains `non-prompt`, so this all-feature run names
`--context-profile all-features` explicitly. The default loss values are the
same four-decimal values used in the earlier BoolQ runs. Block positions and
order are always read from `manifest.json`; no column offsets are hardcoded.

For a quick installation check, run only the supervised path on a prefix:

```powershell
simulate-llm-routing `
  --cache .\boolq-routing-cache-full.zip `
  --context-profile non-prompt `
  --outcome-source cached `
  --experiment skyline `
  --hgb-max-leaf-nodes 15 `
  --limit 500 `
  --output-dir .\smoke-test-results
```

The smoke test checks execution only; its small-sample metrics are not research
results.

## Active algorithm settings

| Policy | Probability model | Exploration and fitting |
|---|---|---|
| ETC | HGB with 15 maximum leaves | Route the first `ceil(n^(2/3))` rounds, fit once, then freeze |
| IGW | Online-refitted 15-leaf HGB | `gamma=sqrt(n)`, `mu=2`, no forced tastes or class bootstrap |
| CBPSide | Regularized linear logistic regression | No forced tastes or class bootstrap; confidence-based routing |
| Random | No model | Matched separately to each ETC profile's realized traffic |

HGB uses 15 maximum leaves, 50 boosting iterations, learning rate 0.05, minimum
leaf size 20, L2 regularization 1.0, and no early stopping. IGW
inverse-propensity weights are capped at 10. A common base seed is used across
loss thresholds.

The online implementations cache append-only revealed history. CBPSide refits
its logistic coefficients and updates its design matrix only after a newly
revealed taste; action-0 rounds reuse the previous fitted state. IGW likewise
refits HGB only after a new taste, while ETC fits once after its derived taste
budget and remains frozen. Predictions are still made sequentially on every
context.

The loss grid contains ten evenly spaced decision thresholds:

```text
alpha = 0.5500, 0.5222, 0.4944, 0.4667, 0.4389,
        0.4111, 0.3833, 0.3556, 0.3278, 0.3000
```

Because `l11=1`, the simulator uses `l01=1/alpha`.

### CBPSide confidence scaling

CBPSide L2-normalizes each context using `x / max(1, ||x||_2)`, prepends an
intercept, and forms `V = lambda I + sum(x x^T)`. The active empirical
confidence radius is:

```text
leverage = sqrt(x^T V^-1 x)
radius = min(0.25 * leverage, 0.5)
```

Here `lambda=1`, the empirical scale is 0.25, and the final radius is capped at
0.5. This is a heuristic confidence rule, not the full Proposition 1 bound.
The theoretical-bound variant was retired from the active run because it hit
the 0.5 cap on every decision in the 142D ARC experiment.

## Supervised skyline versus online routing

These are separate evaluations:

- `--experiment skyline` makes one stratified 80/20 train-validation split. It
  fits logistic and the 15-leaf HGB on the training 80% and evaluates
  only validation predictions.
- `--experiment online` sends all 12,648 eligible BoolQ samples sequentially to ETC,
  IGW, and CBPSide. There is no supervised pretraining subset, and action 0
  hides its outcome from the player.
- `--experiment all` runs both evaluations against the same selected outcome
  source.

Before an online estimator has enough revealed observations from both classes,
it returns a Laplace-smoothed constant probability. It does not inspect hidden
or future outcomes.

## Outputs

The requested output directory contains:

| File | Contents |
|---|---|
| `summary.json` | Cache identity, feature block, parameters, loss grid, and skyline summary |
| `online_results.csv/json` | Per-policy routing rate, accuracy, realized total/per-example cost, and model-refit count at every loss point |
| `online_trajectories.jsonl` | Round-level actions, revealed feedback, predictions, and diagnostics |
| `supervised_model_comparison.csv/json` | Holdout AUC, log loss, Brier score, ECE, and model settings |
| `supervised_skyline.csv/json` | Threshold-level supervised routing curves |
| `supervised_holdout_predictions.csv` | Validation outcomes and model probabilities |
| `routing_comparison.png` | Separate online-routing and supervised-skyline panels |
| `online_routing_accuracy.png` | Standalone online strong-routing-rate versus accuracy comparison |
| `online_cost_vs_alpha.png` | All online policies' realized total cost versus alpha, with `alpha=1/l01` noted |
| `simulation-results.zip` | Portable bundle of the generated outputs |

Result directories and result ZIPs are ignored by Git. The source cache is the
one explicit ZIP exception.

## Synthetic positive control

To reproduce the earlier artificial forest-label sanity check:

```powershell
simulate-llm-routing `
  --cache .\llm-routing-cache-full.zip `
  --context-profile prompt-only `
  --outcome-source synthetic `
  --experiment all `
  --prompt-components 64 `
  --igw-gamma-values 16 `
  --hgb-max-leaf-nodes 15 `
  --l01-values 1.82 2.22 2.67 3.33 `
  --output-dir .\prompt-forest-sanity-results
```

The generated labels replace real disagreement for this run. They use no ARC
gold answer, cached model answer, or real disagreement label. See
[EXPERIMENTS.md](EXPERIMENTS.md) before interpreting this control.

## Branch guide

| Branch | Purpose |
|---|---|
| `main` | Current shared repository state |
| `experiment/prompt-routing` | Frozen BoolQ all-feature baseline with empirical confidence scale 0.25 |
| `experiment/boolq-cbpside-beta1` | BoolQ context studies plus the 138D sample-size-scaled exploration follow-up |
| `experiment/prompt-embedding` | External semantic prompt augmentation and residual correction |
| `experiment/residual-diagnostics` | Logistic/HGB/MLP residual and specification diagnostics |
| `backup/current-combined` | Recovery snapshot of the earlier combined workflow |

Run an experiment only from its corresponding branch. Historical commands,
metrics, and decisions are recorded in [EXPERIMENTS.md](EXPERIMENTS.md), while
implementation responsibilities are summarized in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Source layout

- `src/llm_routing_simulation/cache.py`: cache validation and manifest-defined
  feature selection.
- `src/llm_routing_simulation/environment.py`: partial-feedback environment.
- `src/llm_routing_simulation/player.py`: act-then-update player interface.
- `src/llm_routing_simulation/algorithm.py`: ETC, IGW, and CBPSide policies.
- `src/llm_routing_simulation/skyline.py`: supervised models and skylines.
- `src/llm_routing_simulation/run.py`: command-line orchestration and outputs.
- `src/llm_routing_simulation/synthetic_prompt.py`: synthetic-label positive
  control.

When changing code, preserve the feedback boundary, add an offline test, update
[EXPERIMENTS.md](EXPERIMENTS.md), and keep generated results untracked.
