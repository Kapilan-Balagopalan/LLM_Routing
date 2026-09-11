# Offline LLM-routing simulation

This repository replays weak-versus-strong LLM routing policies from collected
benchmark caches. It is CPU-only: the simulation does not load an LLM, contact
Hugging Face, require an `HF_TOKEN`, or need a GPU.

The active work on `experiment/boolq-cbpside-beta1` performs a complete 138D
BoolQ robustness study over ten independently shuffled online orders. CBPSide
uses empirical scale 0.5 and cap 0.5, ETC uses `ceil(n^(2/3))` forced tastes,
and IGW uses `gamma=sqrt(n)`. CBPSide and IGW batch adaptive model refits after
every five additional tastes. The real routing target is cached weak/strong
disagreement. A separate synthetic-label positive control is available for
implementation sanity checks; it must not be interpreted as real benchmark
routing performance. A separate resumable multiplier tuner is now available
for 20-order studies. Completed revision-3 pure-doubling, revision-4
Fibonacci, and revision-5 capped-doubling gap-500 artifacts remain in their
original fingerprinted directories, but their numerical results were not
analyzed during the later schedule code changes. The current planned, unrun
revision-5 gap-100 study retains ETCLinear beside the fixed 15-leaf HGB ETC and
the matched IGW Linear versus IGW Tree comparison. Adaptive tuner refits now
default to capped exponential doubling: the next global-round boundary is
`min(2b, b+100)`.
This does not alter the established simulator.

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
python -m venv .routing-venv
.\.routing-venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
```

To include the optional incremental River tree used by the tuning sensitivity
study, install both extras instead:

```powershell
python -m pip install -e ".[test,online-tree]"
python -m pytest -q
```

## Active BoolQ shuffled-order experiment

The next experiment selects all complete blocks through
`manifest.context_blocks`: 10 uncertainty features, 64 hidden-state PCA
components, and 64 prompt-embedding PCA components, for 138 dimensions. With
all 12,648 eligible samples, the sample-size rules resolve to ETC taste budget
543 and IGW gamma 112.463327356076. Explicit command-line values still override
these derived defaults. Every method and loss value receives the same
permutation within an order run. The policy seed stays fixed, so variation
across runs measures sensitivity to arrival order rather than a separate model
seed study.

```powershell
simulate-llm-routing `
  --cache .\boolq-routing-cache-full.zip `
  --context-profile all-features `
  --outcome-source cached `
  --experiment all `
  --l01-values 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.3 `
  --hgb-max-leaf-nodes 15 `
  --cbpside-matrix-regularization 1 `
  --cbpside-beta-scale 0.5 `
  --cbpside-max-confidence-radius 0.5 `
  --online-refit-every-tastes 5 `
  --online-order-repeats 10 `
  --online-trajectory-mode none `
  --output-dir .\boolq-all-features-138-order10-beta05-refit5-results
```

The context-profile default remains `non-prompt`, so this all-feature run names
`--context-profile all-features` explicitly. The new loss grid uses short,
ascending decimal values over essentially the same range as the earlier
alpha-derived grid. Block positions and order are always read from
`manifest.json`; no column offsets are hardcoded.

## Pointwise multiplier tuning study

The separate `tune-llm-routing` entry point performs the planned large sweep;
the equivalent module form is `python -m llm_routing_simulation.tuning`. It
uses all 138 BoolQ features discovered from `manifest.context_blocks`, cached
weak/strong disagreement, and all 12,648 eligible rows unless `--limit` is
given. It does not run the supervised skyline.

For every policy and every ascending loss value
`l01 = 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.3`, the tuner compares
multipliers `0.1, 0.3, 1, 3, 10` on the same 20 shuffled orders:

| Policy | Base parameter | Multiplier rule |
|---|---:|---|
| CBPSide | beta scale 0.5 | `beta(x_t) = min((0.5 * multiplier) * sqrt(x_t^T V^-1 x_t), 0.5)`; the cap stays 0.5 |
| ETC HGB | `n^(2/3)` tastes | `tastes = ceil(multiplier * n^(2/3))`, bounded to the online horizon; fixed 15-leaf HGB |
| ETC Linear | `n^(2/3)` tastes | The identical forced-taste candidate with a weighted standardized linear logistic estimator |
| IGW Tree | `gamma=sqrt(n)` | `gamma = multiplier * sqrt(n)` with a nonlinear HGB probability estimator |
| IGW Linear | `gamma=sqrt(n)` | The identical gamma candidate and IGW rule, with only the estimator changed to linear logistic |

The 20 order seeds are paired across policies, losses, and multipliers. Model
snapshots for CBPSide, IGW Tree, and IGW Linear change immediately before
capped-doubling global rounds by default. Starting at `b=1`, each next boundary
is `min(2b, b+100)`, so the schedule doubles early and then limits boundary
gaps to 100 rounds. Each snapshot uses only feedback through `t-1`, and the
policy is still evaluated on every round. CBPSide freezes
`theta_hat` and `V^-1` within each epoch, while its context-dependent beta is
still evaluated for every current `x_t`. IGW Tree refits the default HGB on its
complete revealed history at each eligible boundary; IGW Linear refits a
revealed-history weighted `StandardScaler` plus IPS-weighted L2 logistic model
(`C=1`, `lbfgs`) on its own revealed rows at the same boundaries. This affine
preprocessing preserves a linear decision surface. Both IGW variants use the
same 138 features, gamma candidate, policy random numbers,
selective-feedback protocol, capped-doubling schedule, and capped
inverse-propensity-weighting rule. At a fixed gamma, the probability estimator
is the only configured difference. It is not the only realized difference:
once their actions diverge, each policy observes its own feedback rows and
computes its own propensities and IPS weights.

ETC HGB and ETC Linear use the identical shuffled order, forced prefix, taste
budget, and unit training weights for a given order and multiplier. When the
prefix contains at least two examples from each class, each fits once, freezes,
and reuses its fitted probabilities across every `l01`. If that feasibility
gate fails, the candidate performs no model fit and uses the Laplace-smoothed
prefix prevalence as a constant tail probability; the saved row records this
fallback and both prefix class counts. The two policies independently select
their pointwise taste multiplier. Neither ETC variant is affected by the
adaptive update schedule. Pass `--adaptive-update-schedule fibonacci` to select
Fibonacci boundaries for a new revision-5 run, or
`--adaptive-update-schedule doubling` to select the original `1,2,4,8,...`
boundary rule for a new revision-5 run. These options do not make revision-5
checkpoints compatible with the completed revision-4 Fibonacci or revision-3
four-policy directories; resume and `--plot-only` for those archived artifacts
require their original code revision.

For each learned policy/`l01` pair, the selected multiplier minimizes mean realized
total cost over the 20 orders. In particular, IGW Tree and IGW Linear select
their gamma multipliers independently. Their selected comparison is therefore
best-vs-best and the two selected gammas can differ; it is not a matched-gamma
estimator-only contrast. Separate matched-gamma exports compare the estimators
at each common multiplier. Even there, action-dependent feedback histories can
diverge. Ties prefer the multiplier closest to 1 and then the smaller
multiplier. The figures show mean plus or minus one sample SD.
Because the same 20 orders are used to select and display the winner, this is
an exploratory, optimistic oracle envelope. A later confirmatory study should
evaluate preselected multipliers on fresh order seeds.

### Full HGB/ETC-linear capped-doubling gap-100 sweep

HGB with 15 maximum leaves remains the default so this sweep is directly
comparable with the established routing experiments and remains the nonlinear
primary for ETC HGB and IGW Tree. The same run evaluates ETC Linear and IGW
Linear automatically. Across CBPSide, ETC HGB, ETC Linear, IGW Tree, and IGW
Linear, the full design produces 4,500 learned candidate rows
(`5 * 9 * 5 * 20`) before adding analytic Random. Final selection retains five
learned policies plus Random. The base gamma and ETC taste count are
intentionally omitted below so they are derived from the eligible online
horizon. This 2026-09-11 revision is implemented but has not been run as a full
experiment.

```powershell
.\.routing-venv\Scripts\python.exe -m llm_routing_simulation.tuning `
  --cache .\boolq-routing-cache-full.zip `
  --output-dir .\boolq-138d-multiplier-sweep-hgb-etc-linear-capped-doubling-gap100-results `
  --context-profile all-features `
  --l01-values 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.3 `
  --multipliers 0.1 0.3 1 3 10 `
  --online-order-repeats 20 `
  --adaptive-update-schedule capped-doubling `
  --adaptive-max-round-gap 100 `
  --cbpside-base-beta-scale 0.5 `
  --cbpside-max-confidence-radius 0.5 `
  --igw-mu 2 `
  --tree-estimator hgb `
  --hgb-max-leaf-nodes 15 `
  --jobs 4 `
  --seed 0 `
  --policy-seed 0
```

Relative to the earlier four-policy design, ETC Linear adds 900 candidate rows
(`9 * 5 * 20`). It uses at most one unit-weight prefix fit per order/multiplier
and then reuses frozen probabilities across losses. At `n=12,648`, capped
doubling with a 100-round gap has 133 boundaries and permits at most 132
adaptive refits after feedback exists. Its repeated full-history row-work upper
bound is 803,622, which is `28.06x` the Fibonacci upper bound, `49.09x` pure
doubling, and `4.92x` the completed gap-500 configuration. In exchange, its
final potentially stale tail is only 21 rounds, versus 137 for gap 500, 1,703
for Fibonacci, and 4,457 for pure doubling. This is a very large runtime
increase;
the actual number of fits can be lower when no new tastes arrive or both
classes are not yet available. Use the fresh capped-doubling output directory
shown above; older outputs have a different configuration fingerprint and
cannot be mixed with this run. The completed gap-500 directory remains a valid
revision-5 result: use its original directory and explicitly pass
`--adaptive-max-round-gap 500` to resume or rebuild its plots. Once the gap-100
run is started, rerun the identical gap-100 command to resume its completed
candidate checkpoints.

Each completed policy/`l01`/multiplier/order candidate is saved atomically.
If the run is interrupted, repeat the exact command above with the same output
directory; complete checkpoints are skipped. Do not change a scientific option
when resuming because the output directory is protected by a configuration
fingerprint.

To estimate the five-policy runtime before committing to the full sweep, use
this reduced execution pilot in its own output directory. It is an execution
check, not a research result, and implementation work does not run it:

```powershell
.\.routing-venv\Scripts\python.exe -m llm_routing_simulation.tuning `
  --cache .\boolq-routing-cache-full.zip `
  --output-dir .\boolq-138d-multiplier-sweep-hgb-etc-linear-capped-doubling-gap100-pilot `
  --context-profile all-features `
  --limit 500 `
  --l01-values 1.8 2.6 3.3 `
  --multipliers 0.3 1 3 `
  --online-order-repeats 2 `
  --adaptive-update-schedule capped-doubling `
  --adaptive-max-round-gap 100 `
  --tree-estimator hgb `
  --hgb-max-leaf-nodes 15 `
  --jobs 1 `
  --seed 0 `
  --policy-seed 0
```

After all checkpoints exist, tables and figures can be rebuilt without running
any policy again:

```powershell
.\.routing-venv\Scripts\python.exe -m llm_routing_simulation.tuning `
  --cache .\boolq-routing-cache-full.zip `
  --output-dir .\boolq-138d-multiplier-sweep-hgb-etc-linear-capped-doubling-gap100-results `
  --context-profile all-features `
  --l01-values 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.3 `
  --multipliers 0.1 0.3 1 3 10 `
  --online-order-repeats 20 `
  --adaptive-update-schedule capped-doubling `
  --adaptive-max-round-gap 100 `
  --cbpside-base-beta-scale 0.5 `
  --cbpside-max-confidence-radius 0.5 `
  --igw-mu 2 `
  --tree-estimator hgb `
  --hgb-max-leaf-nodes 15 `
  --seed 0 `
  --policy-seed 0 `
  --plot-only
```

### Optional River Hoeffding-tree sensitivity

`river-hoeffding` is an explicitly different tree-family sensitivity check,
not a drop-in speed claim. It replaces HGB only for IGW Tree, uses River 0.21.2,
accepts IGW inverse-propensity sample weights, and defaults to maximum depth 4
with grace period 200. ETC HGB remains the fixed 15-leaf HGB reference;
ETC Linear and IGW Linear remain linear logistic. A Mondrian forest is not
included because its River update API does not accept the required per-example
weights.

First measure correctness and runtime on this non-scientific pilot:

```powershell
.\.routing-venv\Scripts\python.exe -m llm_routing_simulation.tuning `
  --cache .\boolq-routing-cache-full.zip `
  --output-dir .\boolq-138d-multiplier-sweep-river-igw-etc-linear-capped-doubling-gap100-pilot `
  --context-profile all-features `
  --limit 500 `
  --l01-values 1.8 2.6 3.3 `
  --multipliers 0.3 1 3 `
  --online-order-repeats 2 `
  --adaptive-update-schedule capped-doubling `
  --adaptive-max-round-gap 100 `
  --tree-estimator river-hoeffding `
  --river-max-depth 4 `
  --river-grace-period 200 `
  --jobs 1 `
  --seed 0 `
  --policy-seed 0
```

If that pilot is satisfactory, run the complete River sensitivity study in a
new output directory:

```powershell
.\.routing-venv\Scripts\python.exe -m llm_routing_simulation.tuning `
  --cache .\boolq-routing-cache-full.zip `
  --output-dir .\boolq-138d-multiplier-sweep-river-igw-etc-linear-capped-doubling-gap100-results `
  --context-profile all-features `
  --l01-values 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.3 `
  --multipliers 0.1 0.3 1 3 10 `
  --online-order-repeats 20 `
  --adaptive-update-schedule capped-doubling `
  --adaptive-max-round-gap 100 `
  --cbpside-base-beta-scale 0.5 `
  --cbpside-max-confidence-radius 0.5 `
  --igw-mu 2 `
  --tree-estimator river-hoeffding `
  --river-max-depth 4 `
  --river-grace-period 200 `
  --jobs 4 `
  --seed 0 `
  --policy-seed 0
```

River IGW Tree receives buffered weighted tastes only at the same global
capped-doubling boundaries, so its predictions remain frozen within each
epoch. IGW Linear uses its matching full-history refit at those boundaries. Both ETC
variants still fit once and freeze. Rerun the same command to resume, or add
`--plot-only` after completion while keeping all scientific options unchanged.

### Multiplier-sweep outputs

| File | Contents |
|---|---|
| `sweep_manifest.json` | Complete design, derived bases, epoch semantics, data fingerprint, and selection warning |
| `checkpoints/` | Atomic candidate-level rows used for interruption-safe resume |
| `online_order_permutations.npz` | The exact 20 paired permutations and seeds |
| `candidate_results_by_order.csv/json` | All 4,500 learned policy/loss/multiplier/order results |
| `candidate_results.csv/json` | Candidate means, sample SDs, and standard errors |
| `selected_multipliers.csv/json` | Pointwise winning multiplier and effective parameter for all five learned policies, including independent ETC HGB/ETC Linear and IGW Tree/IGW Linear choices |
| `selected_results_by_order.csv/json` | Five selected learned-policy rows plus analytic Random matched only to selected ETC HGB traffic |
| `selected_results.csv/json` | Final across-order summaries used for plots |
| `igw_tree_vs_linear_by_order.csv/json` | Separately tuned best-vs-best IGW differences on each paired order; selected gammas may differ |
| `igw_tree_vs_linear.csv/json` | Across-order mean, SD, and SEM for that separately tuned comparison |
| `igw_tree_vs_linear_matched_by_order.csv/json` | Tree-versus-linear differences at each common gamma multiplier and paired order |
| `igw_tree_vs_linear_matched.csv/json` | Across-order matched-gamma means, SDs, and SEMs by `l01` and multiplier |
| `selected_routing_accuracy.png` | CBPSide, ETC HGB, ETC Linear, IGW Tree, IGW Linear, and Random routing rate versus cached-strong-reference accuracy with order-SD bars |
| `selected_cost_vs_l01.png` | The same six curves' selected realized total cost versus ascending `l01` with order-SD bars |
| `selected_multiplier_vs_l01.png` | Selected multiplier for each of the five learned policies at every loss point |
| `igw_tree_vs_linear_cost_difference.png` | Separately tuned best-vs-best `linear cost - tree cost`; positive values favor IGW Tree |
| `igw_tree_vs_linear_matched_cost_difference.png` | Matched-gamma `linear cost - tree cost` for all five common multipliers |
| `summary.json` | Compact five-policy-plus-Random selected results, both IGW comparisons, and oracle-selection warning |
| `multiplier-sweep-results.zip` | Portable top-level tables, figures, manifest, and summary |

The bundle contains five figures: three selected-policy plots, the separately
tuned IGW cost-difference plot, and the matched-gamma IGW cost-difference plot.
The matched-gamma files isolate the configured estimator choice more directly,
but they do not force the two policies to take the same actions or observe the
same feedback.

Random is calculated analytically only after ETC HGB selection and is not
matched to ETC Linear. There is no inner `--random-repeats` loop in this tuner,
and adding either linear comparator does not change the Random construction.
The checkpoint directory is retained for resume but is not copied into the
portable ZIP.

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

## Established simulator algorithm settings

These settings apply to `simulate-llm-routing`. The separate pointwise tuner
uses the five learned policies and capped-doubling schedule documented above.

| Policy | Probability model | Exploration and fitting |
|---|---|---|
| ETC | HGB with 15 maximum leaves | Route the first `ceil(n^(2/3))` rounds, fit once, then freeze |
| IGW | Online-refitted 15-leaf HGB | `gamma=sqrt(n)`, `mu=2`, no forced tastes or class bootstrap; refit after every five additional tastes |
| CBPSide | Regularized linear logistic regression | No forced tastes or class bootstrap; confidence-based routing; refit after every five additional tastes |
| Random | No model | Matched separately to each ETC profile's realized traffic |

HGB uses 15 maximum leaves, 50 boosting iterations, learning rate 0.05, minimum
leaf size 20, L2 regularization 1.0, and no early stopping. IGW
inverse-propensity weights are capped at 10. A common base seed is used across
loss thresholds.

The online implementations cache append-only revealed history. CBPSide fits its
logistic coefficients after the first taste and then only when five additional
tastes have accumulated. Its design matrix `V` still incorporates every taste
immediately, preserving the confidence-radius definition. IGW fits HGB as soon
as both classes have at least two revealed labels, then waits for five
additional tastes between refits. Each batched refit uses all revealed tastes,
so no feedback is dropped; predictions between refits use a model that is at
most four tastes behind. Action-0 rounds never trigger a refit. ETC still fits
once after its derived taste budget and remains frozen. Predictions and routing
decisions remain sequential on every context. Override the shared adaptive
cadence with `--online-refit-every-tastes`; it does not affect ETC or the
supervised skyline.

The loss grid is plotted in ascending `l01` order:

```text
l01 = 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.3
```

Because `l11=1`, the corresponding decision threshold remains `alpha=1/l01`;
alpha is retained in CSV/JSON output but is no longer the cost plot's x-axis.

### CBPSide confidence scaling

CBPSide L2-normalizes each context using `x / max(1, ||x||_2)`, prepends an
intercept, and forms `V = lambda I + sum(x x^T)`. The active empirical
confidence radius is:

```text
leverage = sqrt(x^T V^-1 x)
radius = min(0.5 * leverage, 0.5)
```

Here `lambda=1`, the empirical scale is 0.5, and the final radius is capped at
0.5. This is a heuristic confidence rule, not the full Proposition 1 bound.
The theoretical-bound variant was retired from the active run because it hit
the 0.5 cap on every decision in the 142D ARC experiment.

## Supervised skyline versus online routing

These are separate evaluations:

- `--experiment skyline` makes one stratified 80/20 train-validation split. It
  fits logistic and the 15-leaf HGB on the training 80% and evaluates
  only validation predictions.
- `--experiment online` sends all 12,648 eligible BoolQ samples sequentially to
  ETC, IGW, and CBPSide for each requested online ordering. There is no
  supervised pretraining subset, and action 0 hides its outcome from the
  player.
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
| `online_results.csv/json` | Across-order means, sample SDs, and standard errors for each policy and loss point |
| `online_results_by_order.csv/json` | One reusable summary row per order, policy, and loss point |
| `online_order_permutations.npz` | Exact row-index permutations and seeds used by the repeated online study |
| `online_trajectories.jsonl` | Optional round-level diagnostics when trajectory mode is `first` or `all` |
| `supervised_model_comparison.csv/json` | Holdout AUC, log loss, Brier score, ECE, and model settings |
| `supervised_skyline.csv/json` | Threshold-level supervised routing curves |
| `supervised_holdout_predictions.csv` | Validation outcomes and model probabilities |
| `routing_comparison.png` | Separate online-routing and supervised-skyline panels |
| `online_routing_accuracy.png` | Mean routing rate versus accuracy, with horizontal and vertical order-SD bars |
| `online_cost_vs_l01.png` | Mean realized total cost versus ascending `l01`, with order-SD bars |
| `simulation-results.zip` | Portable bundle of the generated outputs |

Result directories and result ZIPs are ignored by Git. The source cache is the
one explicit ZIP exception.

`--random-repeats 100` remains an inner Monte Carlo average for the Random
baseline within each online order. It is distinct from the ten outer shuffled
orders used for the plotted error bars. Detailed 138D trajectories are disabled
in the active command because saving them for all ten orders would create a
multi-gigabyte artifact; the compact per-order tables retain everything needed
to restyle the requested plots.

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
- `src/llm_routing_simulation/online_tree.py`: HGB, weighted standardized
  logistic, and optional weighted River Hoeffding probability backends.
- `src/llm_routing_simulation/tuning.py`: resumable pointwise multiplier sweep,
  matched ETC HGB/ETC Linear protocols, matched IGW Tree/IGW Linear comparison,
  capped doubling with a configurable maximum gap plus Fibonacci and pure
  doubling reproduction schedules, selection, analytic Random baseline, and
  final plots.
- `src/llm_routing_simulation/synthetic_prompt.py`: synthetic-label positive
  control.

When changing code, preserve the feedback boundary, add an offline test, update
[EXPERIMENTS.md](EXPERIMENTS.md), and keep generated results untracked.
