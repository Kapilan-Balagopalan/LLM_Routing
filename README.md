# Offline LLM-routing simulation

This repository replays weak-versus-strong LLM routing policies from collected
benchmark caches. It is CPU-only: the simulation does not load an LLM, contact
Hugging Face, require an `HF_TOKEN`, or need a GPU.

The active work on `experiment/boolq-cbpside-beta1` plans a complete 138D
BoolQ pointwise tuning comparison over 20 paired shuffled online orders. The
current revision-9 design tunes CBPSide linear logistic,
SquareCB.PMSide + linear logistic, SquareCB.PMSide + HGB, and scheduled PG-TS
Bayesian logistic over multipliers `0.1, 0.3, 1, 3, 10`. ETC HGB and ETC
Linear are disabled.
The adaptive estimators use pure-doubling global-round snapshots
`1,2,4,8,...`; their routing policy is still evaluated on every round. Random
is calculated analytically and matched to the selected SquareCB.PMSide + HGB
traffic.

Revision 9 retains revision 8's fixed row-aligned learning-curve reference
mapping built from five-fold stratified out-of-fold HGB-15 probabilities. It
plots both cumulative empirical excess cost and `R_t/t`, using pointwise 95%
Student-`t` confidence intervals across online-order trials. The earlier
outcome-aware clairvoyant excess is retained only as a named diagnostic, not as
the primary regret comparator.

All current tuning figures follow one publication style: no plot titles or
figure footnotes, consistent labels/legends/fonts/sizes, and both vector PDF and
400-DPI PNG output. The central contract is
`src/llm_routing_simulation/plot_style.py`.

With `--include-pgts`, a scheduled PG-TS approximation is the fourth tuned
policy. It performs its first 15-transition Gibbs draw from the prior at
round 1, then runs 15 Pólya-Gamma Gibbs transitions at a later eligible
pure-doubling boundary only when new action-1 feedback has arrived; its sampled
parameter is reused throughout the next epoch. The zero-mean isotropic prior
has effective standard deviation
`--pgts-base-prior-std * multiplier`. Revision 9 evaluates all five common
multipliers on the same 20 paired orders and selects the lowest-mean-cost prior
scale separately at every `l01`. This boundary-scheduled variant is not literal
every-round Algorithm 1; the change is explicitly for runtime. The
real routing target remains cached weak/strong disagreement, and benchmark gold
answers are never routing labels.

Earlier revision-3 through revision-8 studies and their fingerprinted result
directories remain documented in `EXPERIMENTS.md` as history. Do not reuse an
older result directory for the revision-9 design. This tuner change does not
alter the established `simulate-llm-routing` experiment runner.

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

The multiplier tuner also records a learning curve for every selected policy.
For disagreement outcome `y_t` and route `a_t` (`1` means strong), its realized
per-round cost is

```text
cost_t = 1                 if a_t = 1
cost_t = l01 * y_t         if a_t = 0
```

The primary comparator is a fixed row-aligned HGB-15 reference mapping stitched
from five fold-specific models. It obtains one out-of-fold probability
`p_oof,t` for every eligible row using stratified shuffled cross-fitting. Each
probability is produced by a model that did not train on that row. At loss
`l01`, it routes
strong when `p_oof,t >= 1/l01`, giving reference cost `cost_t^ref` under the
same realized cost definition. Therefore the primary learning metrics are

```text
cumulative_reference_regret_t = sum_{s<=t} (cost_s - cost_s^ref)
average_reference_regret_t    = cumulative_reference_regret_t / t
```

This is empirical excess cost relative to the fixed cross-fitted HGB mapping. It
may be negative, and its shape is not by itself a proof of a theoretical
`sqrt(T)` regret bound. The earlier quantity
`sum_{s<=t}(cost_s-y_s)` is still saved as
`cumulative_clairvoyant_excess_cost`, an explicitly named outcome-aware
diagnostic; it is not the primary regret.

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

To include the tuned PG-TS policy, install its Pólya-Gamma sampler as well:

```powershell
python -m pip install -e ".[test,pgts]"
python -m pytest -q
```

The dependency marker installs `polyagamma==1.3.6` on Python 3.9 and a 2.x
release on Python 3.10 or newer.

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
| SquareCB.PMSide + HGB | `gamma=sqrt(n)` | `gamma = multiplier * sqrt(n)` with a nonlinear HGB probability estimator |
| SquareCB.PMSide + linear logistic | `gamma=sqrt(n)` | The identical gamma candidate and SquareCB.PMSide rule, with only the estimator changed to linear logistic |
| PG-TS (Bayesian logistic) | prior standard deviation 1 | `prior_std = multiplier`; equivalently, `--pgts-base-prior-std * multiplier` for a configurable base |

The canonical tuner options are `--squarecb-pmside-base-gamma`,
`--squarecb-pmside-mu`, and `--squarecb-pmside-min-propensity`. The command
below omits `--squarecb-pmside-base-gamma` so the base remains the derived
`sqrt(n)` value; an explicit value is an override.

ETC HGB and ETC Linear are not candidates in this revision. The 20 order seeds
are paired across the four tuned policies, losses, and multipliers. Estimator
snapshots change immediately before pure-doubling global rounds
`1,2,4,8,...`. Each snapshot uses only feedback through `t-1`, while every
policy still makes a routing decision on every round. CBPSide freezes
`theta_hat` and `V^-1` within each epoch but evaluates its context-dependent
beta for every current `x_t`. SquareCB.PMSide + HGB refits the default 15-leaf
HGB on its complete revealed history at each eligible boundary;
SquareCB.PMSide + linear logistic refits a
revealed-history weighted `StandardScaler` plus IPS-weighted L2 logistic model
(`C=1`, `lbfgs`) on its own revealed rows. Both SquareCB.PMSide variants use the same 138
features, gamma candidate, policy random numbers, selective-feedback protocol,
pure-doubling schedule, and capped inverse-propensity weighting. Once their
actions diverge, however, their revealed histories and realized weights may
also differ.

For each learned policy/`l01` pair, the selected multiplier minimizes mean realized
total cost over the 20 orders. In particular, SquareCB.PMSide + HGB and
SquareCB.PMSide + linear logistic select
their gamma multipliers independently. Their selected comparison is therefore
best-vs-best and the two selected gammas can differ; it is not a matched-gamma
estimator-only contrast. Separate matched-gamma exports compare the estimators
at each common multiplier. Even there, action-dependent feedback histories can
diverge. Ties prefer the multiplier closest to 1 and then the smaller
multiplier.
When `--include-pgts` is present, PG-TS participates in the same pointwise
selection: its multiplier sets
`effective_prior_std = --pgts-base-prior-std * multiplier` and the selected value is
the lowest-mean-total-cost prior scale for that `l01`.
Because the same 20 orders are used to select and display the winner, this is
an exploratory, optimistic selection envelope. A later confirmatory study
should evaluate preselected multipliers on fresh order seeds.

Random is not tuned. After selecting the SquareCB.PMSide + HGB multiplier for
each loss, its expected metrics are computed analytically at that selected
policy's routing rate for each paired order. This avoids an inner Monte Carlo
loop and keeps the baseline tied to the nonlinear policy of primary interest.

Finalization always materializes complete learning curves for the selected
policies. CBPSide, both SquareCB.PMSide variants, and scheduled PG-TS use their
pointwise winning multiplier. For
analytic Random, if selected SquareCB.PMSide + HGB routed fraction `q` on that
loss/order,
the expected cost at round `t` is `q + (1-q)*l01*y_t`.

Revision 9 retains the shared five-fold stratified out-of-fold HGB-15 reference
introduced in revision 8. It is built from all eligible 138D contexts and
cached disagreement labels. The
split uses `--seed`; each row is evaluated by a fold model that did not train on
that row. The reference route for loss `l01` is
`1[p_oof >= 1/l01]`. Its row-aligned predictions and actions are fixed before
any online policy replay, merely reordered by each online permutation, and used
only for evaluation. The primary regret
increment is online realized cost minus the reference mapping's realized action
cost.
The compact NPZ and aggregate CSV preserve cumulative cost, cumulative
reference-policy regret, average reference-policy regret `R_t/t`, and the old
outcome-aware diagnostic under `cumulative_clairvoyant_excess_cost`. Each loss
receives both a cumulative-reference-regret plot and an `R_t/t` plot. These
curves inherit the same optimistic-selection warning as the final selected points, and their
empirical shape is not itself a theoretical rate guarantee.

Every tuning figure with replicated-online-order uncertainty uses the same
fixed, pointwise, two-sided 95% Student-`t` interval. This applies to both
learning-curve families, selected routing-rate versus accuracy, selected total
cost, and the separately tuned and matched tree-versus-linear cost differences.
The selected-multiplier plot is a deterministic display of selected values and
has no uncertainty bars. At each plotted point, with `n` trial values, sample
SD `s`, and `df=n-1`, the margin is
`t.ppf(0.975, df)*s/sqrt(n)` and the plotted limits are `mean-margin` and
`mean+margin`. There is no additional 0.5 factor. These intervals are pointwise,
not simultaneous or post-selection adjusted. Non-learning aggregate tables keep
their raw means, SDs, SEMs, and `online_order_repeats`; their plot intervals are
computed at render time rather than stored as additional CI columns. The
learning-curve CSV retains its explicit CI fields. The uncertainty and
publication style are presentation-only and excluded from the candidate
checkpoint fingerprint. A completed compatible revision-9 sweep can therefore
regenerate the CSV plus every PDF/PNG pair with the identical command plus
`--plot-only`; the online policies are not rerun.

For publication, all tuning panels omit titles and figure footnotes. The method
labels are exactly `PG-TS (Bayesian logistic)` and `Random`. The selected-cost
figure uses `x=$\ell_{01}$` and `y=Total cost`. The cumulative-reference-regret
figures display `Round (t)` and `Regret (excess cost)`. Labels, legend and font
sizes, figure sizes, and the 400-DPI PNG setting come from
`src/llm_routing_simulation/plot_style.py`; every plot is also written as a
same-stem vector PDF.
That module exposes `PLOT_CONFIDENCE_LEVEL=0.95`, `AXIS_LABELS`,
`METHOD_LABELS`, `LEGEND_FONT_SIZE=8.0`,
`PUBLICATION_FIGSIZE=(7.0, 4.25)`,
`PUBLICATION_FIGSIZE_SHORT=(7.0, 3.8)`, and `PUBLICATION_PNG_DPI=400`.
`publication_pyplot` supplies the shared Matplotlib settings, while
`save_publication_figure` writes both output formats. The manifest and summary
record that the intervals are neither simultaneous nor post-selection adjusted.

The full revision-9 design contains 3,600 candidate checkpoints
(`4 * 9 * 5 * 20`), 180 execution groups and candidate aggregates, 36
pointwise multiplier selections, 900 selected order rows after adding analytic
Random, and 45 selected summaries. PG-TS contributes 900 checkpoints and 45
groups/aggregates; none of its rows are fixed candidates. Use the fresh
revision-9 directory:

The manifest design is exactly
`pointwise-online-parameter-multiplier-sweep-v9` with implementation revision
9. It is intentionally incompatible with revision-8 and earlier output
directories.

The manifest's `candidate_counts` records
`multiplier_tuned_checkpoints=3600`, `pgts_prior_tuned_checkpoints=900`,
`total_checkpoints=3600`, `multiplier_tuned_execution_groups=180`,
`pgts_prior_tuned_execution_groups=45`, and `total_execution_groups=180`.

```powershell
.\.routing-venv\Scripts\python.exe -m llm_routing_simulation.tuning `
  --cache .\boolq-routing-cache-full.zip `
  --output-dir .\boolq-138d-cbpside-squarecb-pmside-pgts-prior-tuned-results `
  --context-profile all-features `
  --l01-values 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.3 `
  --multipliers 0.1 0.3 1 3 10 `
  --online-order-repeats 20 `
  --reference-folds 5 `
  --adaptive-update-schedule doubling `
  --cbpside-base-beta-scale 0.5 `
  --cbpside-max-confidence-radius 0.5 `
  --squarecb-pmside-mu 2 `
  --squarecb-pmside-min-propensity 0.1 `
  --tree-estimator hgb `
  --hgb-max-leaf-nodes 15 `
  --include-pgts `
  --pgts-gibbs-steps 15 `
  --pgts-base-prior-std 1 `
  --jobs 4 `
  --seed 0 `
  --policy-seed 0
```

### Historical HGB/ETC-linear gap-32 seven-multiplier sweep

This superseded revision-5 design is retained for provenance. HGB with 15
maximum leaves was the default so this sweep was directly
comparable with the established routing experiments and remains the nonlinear
primary for ETC HGB and IGW Tree. The same run evaluates ETC Linear and IGW
Linear automatically. Across CBPSide, ETC HGB, ETC Linear, IGW Tree, and IGW
Linear, the full design produces 6,300 learned candidate rows/checkpoints
(`5 * 9 * 7 * 20`) before adding analytic Random. The runner reports 203
candidate groups: seven groups for ETC HGB, seven for ETC Linear, and 63
loss/multiplier groups for each of CBPSide, IGW Tree, and IGW Linear. Final
selection retains five learned policies plus Random. The base gamma and ETC
taste count are intentionally omitted below so they are derived from the
eligible online horizon. This 2026-09-15 revision is implemented but has not
been run as a full experiment.

```powershell
.\.routing-venv\Scripts\python.exe -m llm_routing_simulation.tuning `
  --cache .\boolq-routing-cache-full.zip `
  --output-dir .\boolq-138d-multiplier-sweep-hgb-etc-linear-capped-doubling-gap32-multiplier7-results `
  --context-profile all-features `
  --l01-values 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.3 `
  --multipliers 0.03 0.1 0.3 1 3 10 30 `
  --online-order-repeats 20 `
  --adaptive-update-schedule capped-doubling `
  --adaptive-max-round-gap 32 `
  --cbpside-base-beta-scale 0.5 `
  --cbpside-max-confidence-radius 0.5 `
  --igw-mu 2 `
  --tree-estimator hgb `
  --hgb-max-leaf-nodes 15 `
  --jobs 4 `
  --seed 0 `
  --policy-seed 0
```

Relative to a corresponding four-policy seven-multiplier design, ETC Linear
adds 1,260 candidate rows (`9 * 7 * 20`). It uses at most one unit-weight
prefix fit per order/multiplier and then reuses frozen probabilities across
losses. At `n=12,648`, capped doubling with a 32-round gap has 400 boundaries
and permits at most 399 adaptive refits after feedback exists. Its last
boundary is round 12,640, the final potentially stale tail is nine rounds, and
the repeated full-history row-work upper bound is 2,502,351. The actual number
of fits can be lower when no new tastes arrive or both classes are not yet
available.

The new endpoint `m=0.03` gives ETC 17 forced tastes. That is below HGB's
20-sample minimum leaf size, so ETC HGB cannot split and acts as a constant
prefix-prevalence predictor; ETC Linear can still fit when the prefix passes
the existing two-per-class feasibility gate. At `m=30`, the computed ETC
budget exceeds the horizon and is capped at all 12,648 rounds. Both ETC variants
then route every example to the strong model and have no post-prefix routing
phase. With `l11=1`, both have routing rate 1, accuracy 1, total cost 12,648,
and zero order variation. This saturated endpoint is retained as an explicit
upper-bound sensitivity candidate.

Use the fresh output directory shown above. The completed five-multiplier
gap-32 directory has a different multiplier grid and must not be reused. The
`boolq-138d-multiplier-sweep-hgb-etc-linear-capped-doubling-gap08-results`
directory is also preserved: a read-only check on 2026-09-15 found 3,902 of
4,500 checkpoints, `sweep_manifest.json`, and the order permutations, but no
final tables, figures, summary, or ZIP. It remains resumable only with its
original five multipliers and explicit `--adaptive-max-round-gap 8`; it is not
part of the new seven-multiplier study.

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
  --output-dir .\boolq-138d-multiplier-sweep-hgb-etc-linear-capped-doubling-gap32-multiplier7-pilot `
  --context-profile all-features `
  --limit 500 `
  --l01-values 1.8 2.6 3.3 `
  --multipliers 0.03 1 30 `
  --online-order-repeats 2 `
  --adaptive-update-schedule capped-doubling `
  --adaptive-max-round-gap 32 `
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
  --output-dir .\boolq-138d-multiplier-sweep-hgb-etc-linear-capped-doubling-gap32-multiplier7-results `
  --context-profile all-features `
  --l01-values 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.3 `
  --multipliers 0.03 0.1 0.3 1 3 10 30 `
  --online-order-repeats 20 `
  --adaptive-update-schedule capped-doubling `
  --adaptive-max-round-gap 32 `
  --cbpside-base-beta-scale 0.5 `
  --cbpside-max-confidence-radius 0.5 `
  --igw-mu 2 `
  --tree-estimator hgb `
  --hgb-max-leaf-nodes 15 `
  --seed 0 `
  --policy-seed 0 `
  --plot-only
```

### Tuned scheduled PG-TS approximation

Pass `--include-pgts` to add a runtime-oriented approximation to PG-TS from
*Apple Tasting Revisited*. For each loss and shuffled order, it performs its
first `M=15`-transition Gibbs draw from a zero-mean Gaussian prior with
covariance `prior_std^2 I` at round 1. At a later configured estimator
boundary, it makes `M=15`
Pólya-Gamma Gibbs transitions using action-1 feedback through the preceding
round, but only when new feedback has arrived since its prior posterior draw.
Otherwise it keeps the current draw. The final sampled parameter is frozen and
reused for every decision until the next actual update. With the default
schedule, eligible boundaries are pure doubling: `1,2,4,8,...`.

The context is row-L2-normalized in the same outcome-free way as CBPSide and an
intercept is prepended. With `l11=1`, scheduled PG-TS routes to the strong model
when the sampled disagreement probability is at least `1/l01`. Only action-1
outcomes enter its posterior, and no inverse-propensity weights are used. This
is explicitly not literal Algorithm 1, which draws from its approximate
posterior on every round; scheduling is an intentional runtime approximation.

Revision 9 uses the common multiplier grid to tune the Gaussian prior scale:

```text
effective_prior_std = --pgts-base-prior-std * multiplier
```

`--pgts-prior-std` remains a backward-compatible alias, but new revision-9
commands use the canonical base-scale name above.

With base value 1, the five effective candidates are `0.1, 0.3, 1, 3, 10`.
They are evaluated on the same 20 paired shuffled orders at every `l01`, and
the multiplier with the lowest mean total cost is selected pointwise using the
common tie rule. PG-TS therefore appears in `selected_multipliers.*` and in the
selected-multiplier figure. Candidate rows record `parameter_name=prior_std`,
the base value, multiplier, effective value, and effective `pgts_prior_std`.
The ordinary Gaussian draw in Appendix D is used; no unspecified truncation
procedure is invented. `--pgts-gibbs-steps` remains 15, and `--policy-seed`
seeds both Gaussian and Pólya-Gamma draws. Its public plot label remains exactly
`PG-TS (Bayesian logistic)`.

After installing the optional dependency, this small pilot remains available
as an execution check:

```powershell
.\.routing-venv\Scripts\python.exe -m pip install -e ".[test,pgts]"
.\.routing-venv\Scripts\python.exe -m llm_routing_simulation.tuning `
  --cache .\boolq-routing-cache-full.zip `
  --output-dir .\boolq-revision9-pgts-prior-tuning-pilot `
  --context-profile all-features `
  --limit 25 `
  --l01-values 2.6 `
  --multipliers 1 `
  --online-order-repeats 2 `
  --adaptive-update-schedule doubling `
  --squarecb-pmside-mu 2 `
  --squarecb-pmside-min-propensity 0.1 `
  --include-pgts `
  --pgts-gibbs-steps 15 `
  --pgts-base-prior-std 1 `
  --jobs 1 `
  --seed 0 `
  --policy-seed 0
```

This pilot is an execution and timing check, not a research result. With 25
rounds there are five eligible pure-doubling boundaries, so there are at most
five posterior draws/model updates and 75 Gibbs transitions; four of those
draws occur after the initial round-1 draw. A boundary with no new feedback
performs no redraw. The historical revision-6 design in
`EXPERIMENTS.md` records the substantially more expensive faithful every-round
Algorithm 1 comparator; the current design deliberately does not reproduce it.

The complete revision-9 study is the 3,600-checkpoint command above. Omitting
`--include-pgts` produces a different three-policy design and must use a
different output directory; it is not the requested PG-TS prior-tuning study.

### Optional River Hoeffding-tree sensitivity

`river-hoeffding` is an explicitly different tree-family sensitivity check,
not a drop-in speed claim. It replaces HGB only for SquareCB.PMSide + HGB,
uses River 0.21.2, accepts SquareCB.PMSide inverse-propensity sample weights,
and defaults to maximum depth 4 with grace period 200. CBPSide and
SquareCB.PMSide + linear logistic remain linear logistic; the
current tuner has no ETC policy. A Mondrian forest is not included because
its River update API does not accept the required per-example weights.

First measure correctness and runtime on this non-scientific pilot:

```powershell
.\.routing-venv\Scripts\python.exe -m llm_routing_simulation.tuning `
  --cache .\boolq-routing-cache-full.zip `
  --output-dir .\boolq-138d-cbpside-squarecb-pmside-river-doubling-multiplier5-pilot `
  --context-profile all-features `
  --limit 500 `
  --l01-values 1.8 2.6 3.3 `
  --multipliers 0.1 1 10 `
  --online-order-repeats 2 `
  --adaptive-update-schedule doubling `
  --squarecb-pmside-mu 2 `
  --squarecb-pmside-min-propensity 0.1 `
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
  --output-dir .\boolq-138d-cbpside-squarecb-pmside-river-doubling-multiplier5-results `
  --context-profile all-features `
  --l01-values 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.3 `
  --multipliers 0.1 0.3 1 3 10 `
  --online-order-repeats 20 `
  --adaptive-update-schedule doubling `
  --cbpside-base-beta-scale 0.5 `
  --cbpside-max-confidence-radius 0.5 `
  --squarecb-pmside-mu 2 `
  --squarecb-pmside-min-propensity 0.1 `
  --tree-estimator river-hoeffding `
  --river-max-depth 4 `
  --river-grace-period 200 `
  --jobs 4 `
  --seed 0 `
  --policy-seed 0
```

River SquareCB.PMSide + HGB receives buffered weighted tastes only at the same
global pure-doubling boundaries, so its predictions remain frozen within each
epoch. SquareCB.PMSide + linear logistic uses its matching full-history refit
at those boundaries. Rerun the
same command to resume, or add `--plot-only` after completion while keeping all
scientific options unchanged.

### Multiplier-sweep outputs

| File | Contents |
|---|---|
| `sweep_manifest.json` | Complete revision-9 design, derived bases, epoch semantics, PG-TS prior-scale rule, fixed-reference specification, data fingerprint, selection warning, and top-level `plot_presentation` contract (`formats=[png,pdf]`, 400 DPI, no titles/footnotes, labels, and CI metadata) |
| `checkpoints/` | All 3,600 tuned candidate rows used for interruption-safe resume |
| `online_order_permutations.npz` | The exact 20 paired permutations and seeds |
| `candidate_results_by_order.csv/json` | All 3,600 policy/loss/multiplier/order results, including the 900 PG-TS prior-scale candidates (five at every loss/order) |
| `candidate_results.csv/json` | 180 tuned aggregates with means, sample SDs, and standard errors |
| `selected_multipliers.csv/json` | 36 pointwise winners: nine losses for each of CBPSide, SquareCB.PMSide + HGB, SquareCB.PMSide + linear logistic, and PG-TS; Random is excluded |
| `selected_results_by_order.csv/json` | 900 rows: four selected tuned policies plus analytic Random over nine losses and 20 orders |
| `selected_results.csv/json` | 45 summaries retaining means, sample SDs, SEMs, and repeat counts used to derive plot CIs at render time |
| `cross_fitted_hgb_reference.npz` | Row-aligned example IDs, cached outcomes, fold assignments, and float64 five-fold OOF HGB-15 disagreement probabilities |
| `cross_fitted_hgb_reference_results.csv/json` | Reference routing rate, cached-strong agreement, and realized cost for every `l01` |
| `selected_learning_curves_by_order.npz` | Compact per-policy/loss/order/round arrays: `cumulative_cost`, `cumulative_reference_cost`, `cumulative_reference_regret`, `average_reference_regret`, and diagnostic-only `cumulative_clairvoyant_excess_cost`, plus axes and metadata |
| `selected_learning_curves.csv` | Per-policy/loss/round raw mean, sample SD, and SEM for all retained learning-curve metrics across orders; also `confidence_level`, `confidence_df`, and `confidence_t_critical`, plus `cumulative_reference_regret_ci95_lower`, `cumulative_reference_regret_ci95_upper`, `average_reference_regret_ci95_lower`, and `average_reference_regret_ci95_upper` for the two plotted metrics |
| `selected_cumulative_reference_regret_l01-<float_slug>.{png,pdf}` | One cumulative excess-cost comparison against the fixed cross-fitted HGB-15 reference per loss, as 400-DPI PNG and vector PDF |
| `selected_average_reference_regret_l01-<float_slug>.{png,pdf}` | One average reference-policy regret `R_t/t` comparison per loss, as 400-DPI PNG and vector PDF |
| `squarecb_pmside_tree_vs_linear_by_order.csv/json` | 180 separately tuned best-vs-best SquareCB.PMSide estimator differences; selected gammas may differ |
| `squarecb_pmside_tree_vs_linear.csv/json` | Nine across-order mean, SD, SEM, and repeat-count rows used to derive the separately tuned comparison's plot CIs |
| `squarecb_pmside_tree_vs_linear_matched_by_order.csv/json` | 900 HGB-versus-linear differences across nine losses, five common gamma multipliers, and 20 orders |
| `squarecb_pmside_tree_vs_linear_matched.csv/json` | 45 across-order matched-gamma mean/SD/SEM/repeat-count summaries by `l01` and multiplier, used to derive plot CIs |
| `selected_routing_accuracy.{png,pdf}` | CBPSide, SquareCB.PMSide + HGB, SquareCB.PMSide + linear logistic, `PG-TS (Bayesian logistic)`, and `Random`, with pointwise two-sided 95% Student-`t` intervals on both axes |
| `selected_cost_vs_l01.{png,pdf}` | Final-horizon selected realized total cost versus `l01`, with pointwise two-sided 95% Student-`t` intervals and exact axes `$\ell_{01}$` / `Total cost` |
| `selected_multiplier_vs_l01.{png,pdf}` | Selected multiplier for all four tuned policies, including PG-TS prior scale, at every loss point; no uncertainty display |
| `squarecb_pmside_tree_vs_linear_cost_difference.{png,pdf}` | Separately tuned best-vs-best `linear cost - HGB cost` with pointwise two-sided 95% Student-`t` intervals; positive values favor SquareCB.PMSide + HGB |
| `squarecb_pmside_tree_vs_linear_matched_cost_difference.{png,pdf}` | Matched-gamma `linear cost - HGB cost` for all five common multipliers, with pointwise two-sided 95% Student-`t` intervals |
| `summary.json` | Compact selected results, both SquareCB.PMSide estimator comparisons, fixed-reference metadata and plot lists, tuned PG-TS prior-scale settings, optimistic-selection warning, echoed `plot_presentation`, and learning-curve PDF/PNG pairing metadata |
| `multiplier-sweep-results.zip` | Portable top-level tables, figures, manifest, and summary |

Learning curves are intentionally not duplicated into a large JSON artifact.
The NPZ preserves compact by-order trajectories for restyling, while the CSV
is the portable aggregate table.

The bundle retains the five existing summary figure families—three
selected-policy plots, the separately tuned SquareCB.PMSide cost-difference
plot, and the matched-gamma SquareCB.PMSide cost-difference plot—and adds two
reference-policy-regret figure families per configured loss. Every family is
included as both a vector PDF and a 400-DPI PNG.
The matched-gamma files isolate the configured estimator choice more directly,
but they do not force the two policies to take the same actions or observe the
same feedback.

Random is calculated analytically only after SquareCB.PMSide + HGB selection
and is matched to that selected policy's traffic for each loss/order. There is
no inner
`--random-repeats` loop in this tuner. The checkpoint directory is retained for
resume but is not copied into the portable ZIP.

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
uses the four tuned policies and pure-doubling schedule documented above.

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
| `experiment/boolq-cbpside-beta1` | BoolQ context studies, 138D SquareCB.PMSide/CBPSide tuning, and fixed-reference regret evaluation |
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
- `src/llm_routing_simulation/pgts.py`: warm-started Pólya-Gamma Gibbs sampler
  and Thompson decision rule used by both the historical faithful formulation
  and the active boundary-scheduled PG-TS approximation.
- `src/llm_routing_simulation/plot_style.py`: shared publication labels,
  Matplotlib settings, Student-`t` interval helper, figure sizes, legend/font
  sizes, and paired vector-PDF/400-DPI-PNG export for the tuning figures.
- `src/llm_routing_simulation/tuning.py`: resumable pointwise multiplier sweep,
  matched SquareCB.PMSide + HGB versus SquareCB.PMSide + linear logistic
  comparison, optional multiplier-tuned PG-TS prior-scale candidates,
  pure-doubling snapshots with
  Fibonacci and capped-doubling reproduction choices, selection, analytic
  Random matched to selected SquareCB.PMSide + HGB, fixed five-fold cross-fitted
  HGB-15 reference-policy evaluation, and final plots plus compact selected-policy
  cumulative cost/reference-regret curves. ETC policies are disabled in the
  current tuner.
- `src/llm_routing_simulation/synthetic_prompt.py`: synthetic-label positive
  control.

When changing code, preserve the feedback boundary, add an offline test, update
[EXPERIMENTS.md](EXPERIMENTS.md), and keep generated results untracked.
