# Offline LLM-routing simulation

This repository replays weak-versus-strong LLM routing policies from collected
benchmark caches. It is CPU-only: the simulation does not load an LLM, contact
Hugging Face, require an `HF_TOKEN`, or need a GPU.

The active work on `experiment/boolq-cbpside-beta1` compares the complete 64D
BoolQ prompt-embedding block before repeating the 138D all-feature study. Both
use CBPSide scale 0.5 and cap 1.0. The real routing target is cached weak/strong
disagreement. A separate synthetic-label positive control is available for
implementation sanity checks; it must not be interpreted as real benchmark
routing performance.

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

## Active BoolQ 64-dimensional prompt-only experiment

The next experiment uses only the complete BoolQ `prompt_embedding_pca` block
located through `manifest.context_blocks`. It excludes uncertainty and hidden
states, retains all 64 prompt components, and keeps the rounded loss grid,
gamma 64, beta scale 0.5, cap 1.0, and selected 15-leaf HGB model.

```powershell
simulate-llm-routing `
  --cache .\boolq-routing-cache-full.zip `
  --context-profile prompt-only `
  --prompt-components 64 `
  --outcome-source cached `
  --experiment all `
  --l01-values 1.8182 1.9149 2.0225 2.1429 2.2785 2.4324 2.6087 2.8125 3.0508 3.3333 `
  --igw-gamma-values 64 `
  --hgb-max-leaf-nodes 15 `
  --cbpside-matrix-regularization 1 `
  --cbpside-beta-scale 0.5 `
  --cbpside-max-confidence-radius 1 `
  --output-dir .\boolq-prompt-only-64-beta05-matched-results
```

The active defaults are `prompt-only`, all 64 prompt components, and the same
four-decimal loss values used in the earlier BoolQ runs. Earlier 20D and 32D
studies remain reproducible with explicit `--prompt-components`. The subsequent
matched 138D run uses `--context-profile all-features`, which always selects
every complete manifest block and ignores prompt-component truncation. Block
positions and order are read from `manifest.json`; no column offsets are
hardcoded.

For a quick installation check, run only the supervised path on a prefix:

```powershell
simulate-llm-routing `
  --cache .\boolq-routing-cache-full.zip `
  --context-profile all-features `
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
| ETC | HGB with 15 maximum leaves | Route the first 300 rounds, fit once, then freeze |
| IGW | Online-refitted 15-leaf HGB | `gamma=64`, `mu=2`, no forced tastes or class bootstrap |
| CBPSide | Regularized linear logistic regression | No forced tastes or class bootstrap; confidence-based routing |
| Random | No model | Matched separately to each ETC profile's realized traffic |

HGB uses 15 maximum leaves, 50 boosting iterations, learning rate 0.05, minimum
leaf size 20, L2 regularization 1.0, and no early stopping. IGW
inverse-propensity weights are capped at 10. A common base seed is used across
loss thresholds.

The online implementations cache append-only revealed history. CBPSide refits
its logistic coefficients and updates its design matrix only after a newly
revealed taste; action-0 rounds reuse the previous fitted state. IGW likewise
refits HGB only after a new taste, while ETC still fits once after its 300 tastes
and remains frozen. Predictions are still made sequentially on every context.

The loss grid contains ten evenly spaced decision thresholds:

```text
alpha = 0.5500, 0.5222, 0.4944, 0.4667, 0.4389,
        0.4111, 0.3833, 0.3556, 0.3278, 0.3000
```

Because `l11=1`, the simulator uses `l01=1/alpha`.

### CBPSide confidence scaling

CBPSide L2-normalizes each context using `x / max(1, ||x||_2)`, prepends an
intercept, and forms `V = lambda I + sum(x x^T)`. The active empirical
confidence radius restores the implementation used before the Proposition 1
guardrail experiment:

```text
leverage = sqrt(x^T V^-1 x)
radius = min(0.5 * leverage, 1.0)
```

Here `lambda=1`, the empirical scale is 0.5, and the final radius is capped at
1.0. This intermediate scale follows the scale-1.0 run, which removed the
cold-start collapse but routed too aggressively. This is a heuristic confidence
rule, not the full Proposition 1 bound.
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
| `online_results.csv/json` | Per-policy routing rate, accuracy, and model-refit count at every loss point |
| `online_trajectories.jsonl` | Round-level actions, revealed feedback, predictions, and diagnostics |
| `supervised_model_comparison.csv/json` | Holdout AUC, log loss, Brier score, ECE, and model settings |
| `supervised_skyline.csv/json` | Threshold-level supervised routing curves |
| `supervised_holdout_predictions.csv` | Validation outcomes and model probabilities |
| `routing_comparison.png` | Separate online-routing and supervised-skyline panels |
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
| `experiment/boolq-cbpside-beta1` | BoolQ prompt-only 64D then all-feature 138D comparison at scale 0.5 |
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
