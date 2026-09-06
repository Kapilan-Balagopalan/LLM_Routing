# Offline LLM-routing simulation

This repository replays weak-versus-strong LLM routing policies from a collected
ARC-Easy cache. It is CPU-only: the simulation does not load an LLM, contact
Hugging Face, require an `HF_TOKEN`, or need a GPU.

The active work on `experiment/prompt-routing` asks whether combining
uncertainty features with prompt embeddings supports a useful nonlinear routing
policy. The real routing target is cached weak/strong disagreement. A separate
synthetic-label positive control is available for implementation sanity checks;
it must not be interpreted as real ARC routing performance.

For experiment history and conclusions, read [EXPERIMENTS.md](EXPERIMENTS.md).
For module boundaries and data flow, read [ARCHITECTURE.md](ARCHITECTURE.md).

## Routing definitions

- `Y=1`: the extracted weak- and strong-model answers disagree.
- `Y=0`: their extracted answers agree.
- Action `0`: retain the weak answer; the player receives no outcome feedback.
- Action `1`: route to the strong model; the disagreement outcome is revealed.
- A **taste** is an action-1 round whose outcome becomes available for learning.
- Routing accuracy is agreement with the cached strong-model answer. ARC gold
  answers are not routing labels, online feedback, or the reported accuracy
  reference.

For `l11=1`, the empirical asymmetric decision loss reported in analysis is

```text
decision loss = routing_rate + l01 * (1 - routing_accuracy)
```

Raw routing accuracy should not be used alone to compare policies with different
routing rates.

## Data

`llm-routing-cache-full.zip` is versioned in the repository and is available on
the active branches. It uses schema v2 and contains 5,173 collected rows, of
which 5,138 are eligible. Rows are ineligible when either model answer could not
be parsed.

The cache stores a 142-dimensional context:

| Block | Dimensions | Description |
|---|---:|---|
| Uncertainty | 14 | Weak-model choice and token-probability summaries |
| Hidden-state PCA | 64 | Whitened PCA of weak-model hidden states |
| Prompt-embedding PCA | 64 | Whitened PCA of context-free prompt embeddings |

The prompt-routing code finds `prompt_embedding_pca` through
`manifest.json -> context_blocks`; it never hardcodes column numbers. The PCA
is fixed, transductive, and outcome-free.

## Setup

Clone the repository and select the prompt-routing branch:

```powershell
git clone https://github.com/Kapilan-Balagopalan/LLM_Routing.git
Set-Location LLM_Routing
git switch experiment/prompt-routing
```

Create the environment and install the project:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
```

## Active 142-dimensional all-feature experiment

The next experiment uses every complete block in `manifest.context_blocks`, in
manifest order: 14 uncertainty features, 64 hidden-state PCA components, and 64
prompt-embedding PCA components. This produces the complete 142-dimensional
cached context. It retains the ten loss points and gamma 64, and uses only the
selected 15-leaf HGB model.

```powershell
simulate-llm-routing `
  --cache .\llm-routing-cache-full.zip `
  --context-profile all-features `
  --outcome-source cached `
  --experiment all `
  --igw-gamma-values 64 `
  --hgb-max-leaf-nodes 15 `
  --cbpside-matrix-regularization 1 `
  --cbpside-delta 0.05 `
  --cbpside-c-max 3 `
  --cbpside-max-confidence-radius 0.5 `
  --output-dir .\all-features-142-beta-guardrail-results
```

The context-profile argument is important. The defaults remain `prompt-only`
and 20 prompt components so the completed prompt-only experiments can still be
reproduced. `all-features` always selects every complete manifest block, so
`--prompt-components` does not truncate it. Block positions and order are read
from `manifest.json`; the command does not assume numeric column offsets.

For a quick installation check, run only the supervised path on a prefix:

```powershell
simulate-llm-routing `
  --cache .\llm-routing-cache-full.zip `
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

The loss grid contains ten evenly spaced decision thresholds:

```text
alpha = 0.5500, 0.5222, 0.4944, 0.4667, 0.4389,
        0.4111, 0.3833, 0.3556, 0.3278, 0.3000
```

Because `l11=1`, the simulator uses `l01=1/alpha`.

### CBPSide confidence scaling

CBPSide L2-normalizes each context using `x / max(1, ||x||_2)`, prepends an
intercept, and forms `V = lambda I + sum(x x^T)`. It uses the Proposition 1
confidence beta

```text
beta = (2 k_sigma R_max / c_sigma)
       * sqrt(x^T V^-1 x)
       * sqrt((3 + 2 log(1 + 2/lambda))
              * 2 d log(N)
              * log(d/delta_t))

radius = min(beta, 0.5)
```

The fixed settings are `k_sigma=1/4`, `R_max=1`, `lambda=1`, and
`C_max=3`, with
`c_sigma=exp(C_max)/(1+exp(C_max))^2`. The dimension `d` includes the
intercept. The implementation uses `delta_t=min(0.05,0.5/d)` and
`N=max(2, number of revealed tastes)` to satisfy the dimension guardrail and
handle startup. No additional beta multiplier is applied.

## Supervised skyline versus online routing

These are separate evaluations:

- `--experiment skyline` makes one stratified 80/20 train-validation split. It
  fits logistic and the 15-leaf HGB on the training 80% and evaluates
  only validation predictions.
- `--experiment online` sends all 5,138 eligible samples sequentially to ETC,
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
| `online_results.csv/json` | Per-policy routing rate and accuracy at every loss point |
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
| `experiment/prompt-routing` | Prompt and uncertainty-plus-prompt real-label routing, plus the synthetic positive control |
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
