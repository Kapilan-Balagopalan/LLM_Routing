# Architecture

## Purpose

This CPU-only project replays routing algorithms against a previously collected
LLM cache. It does not load an LLM, download models, call Hugging Face, or need
`HF_TOKEN`. LLM generation is owned by the separate
`Data_collection_LLM_routing` project. On the prompt-embedding branch, one
explicit preprocessing command may download a small public frozen encoder; no
weak- or strong-model generation is repeated.

## Data flow

```text
boolq-routing-cache-full.zip
    -> cache.py validates records and arrays
    -> run.py finds requested feature blocks from manifest context_blocks
    -> run.py selects prompt-only, non-prompt, uncertainty-prompt, or all-feature context
    -> run.py selects cached disagreement or the optional synthetic positive control
    -> run.py creates paired online-order permutations when repeats are requested
    -> environment.py emits the current context and weak answer
    -> player.py defines the act-then-update protocol
    -> algorithm.py chooses action 0 or 1 from revealed history
    -> environment.py reveals the active outcome only when action 1 was selected
    -> run.py aggregates per-order metrics, tables, and plots
```

The much larger pointwise multiplier study has a separate orchestration path so
the established single-configuration simulator remains stable:

```text
boolq-routing-cache-full.zip
    -> tuning.py selects all manifest-defined context blocks and cached disagreement
    -> tuning.py creates 20 paired shuffled orders
    -> tuning.py creates one fixed five-fold cross-fitted HGB-15 reference mapping before policy replay
    -> tuning.py evaluates CBPSide, SquareCB.PMSide + HGB, SquareCB.PMSide + linear logistic, and PG-TS multiplier candidates
    -> pgts.py supplies boundary-scheduled Bayesian-logistic draws for each prior-scale candidate
    -> online_tree.py supplies batch HGB, batch logistic, or incremental Hoeffding estimators
    -> tuning.py checkpoints every completed policy/l01/multiplier/order candidate
    -> tuning.py selects the lowest-mean-cost multiplier at each policy/l01 point
    -> tuning.py adds analytic Random matched to the selected SquareCB.PMSide + HGB traffic
    -> tuning.py reconstructs cost and fixed-reference regret curves without exposing the reference to players
    -> plot_style.py applies the shared publication contract and emits PDF/400-DPI-PNG pairs
    -> tuning.py writes reusable tables, compact NPZ curves, figures, and a ZIP bundle
```

At round `t`, a player receives only the current context. It selects:

- `0`: retain the weak answer; disagreement remains hidden.
- `1`: use the strong answer; disagreement is revealed as `0` or `1`.

The environment owns the complete cached stream. Players store only the action,
context, and feedback that the environment actually revealed.

On `experiment/prompt-routing`, cached weak/strong disagreement is the default
outcome. A deliberately synthetic positive-control outcome remains selectable
to test whether the same supervised and online code recovers known nonlinear
prompt-feature signal.

## Cache and context

The active BoolQ v2 cache manifest records:

- schema: `llm-routing-cache-v2`;
- dataset: `google/boolq`, train and validation splits;
- 12,697 collected rows and 12,648 eligible rows;
- weak model: `Qwen/Qwen2.5-0.5B-Instruct`;
- strong model: `meta-llama/Llama-2-13b-chat-hf`;
- outcome: `1` exactly when extracted weak and strong answers differ;
- raw concatenated hidden dimension: 4,480;
- fixed, transductive, outcome-free PCA: 64 components;
- full saved context: 10 uncertainty features, 64 hidden-state PCA features,
  and 64 prompt-embedding PCA features, standardized to 138 dimensions.

The 10 uncertainty features contain choice entropy, normalized choice entropy,
top probability, top-two margin, one-minus-top probability, next-token
vocabulary entropy, two option probabilities, and two option log likelihoods.

On `experiment/prompt-routing`, `run.py` obtains every block boundary and its
order from `manifest.context_blocks`. The active BoolQ context ablation excludes
the prompt block and selects 10 uncertainty plus 64 hidden-state PCA features,
for 74 dimensions. The complementary prompt-only profile selects the complete
64D prompt-embedding PCA block. The subsequent matched run selects all three
blocks for 138 dimensions. The block ranges, rather than the manifest's
descriptive prose, are authoritative.
The PCA was constructed during collection across all prompts without outcomes.
An optional experiment limit is applied only after block selection.

## Core modules

### `cache.py`

Validates the ZIP manifest, records, and NumPy arrays. It filters ineligible
extraction rows and can reconstruct a smaller context using saved PCA axes.

### `environment.py`

`LLMCascadeEnvironment` enforces partial monitoring. The strong answer and
active routing outcome are returned only following action `1`. A round can
carry an explicit synthetic outcome override; action `0` still hides it.

### `player.py`

`HistoryBasedPlayer` defines the stateful interface:

1. `next_action(current_context)`;
2. environment transition;
3. `update(action, context, revealed_outcome)`.

It checks that updates correspond to the pending action and context.

### `algorithm.py`

- `HGBETCPlayer`: routes strongly for `ceil(n^(2/3))` rounds, fits one histogram
  gradient boosting classifier from those revealed labels, and freezes that
  estimator. With 12,648 online samples this is 543 tastes. Future studies use
  15 maximum leaves per boosting tree.
- `LogCBPSideATPlayer`: estimates a regularized linear logistic disagreement
  model and applies the restored empirical Mahalanobis-leverage confidence
  radius without forced tastes. The active scale is 0.5 and the final radius
  is capped at 0.5. The player incrementally caches revealed feature rows and
  updates `V` after every taste. It fits from the same zero initialization after
  the first taste and then after each batch of five additional tastes.
- `IGWPlayer`: estimates disagreement using an online-refitted histogram
  gradient boosting classifier and samples an arm using inverse-gap weighting.
  The active study uses `mu=2`, `gamma=sqrt(n)` (112.463327356076 at
  `n=12,648`), the same 15-leaf HGB profile as ETC, and five-taste refit
  batches after its first feasible fit.
- `RevealedFeedbackEstimator`: extracts only action-1 observations and applies
  capped inverse-propensity weights when supplied. It processes only newly
  appended history rows. Batched refits use the complete cached revealed
  history, while action-0 rounds neither add a taste nor trigger a refit.

CBPSide and IGW have no forced tastes or hidden class bootstrap. Before enough
revealed observations exist to fit both classes, their estimators return a
Laplace-smoothed constant probability. IGW can therefore obtain initial labels
through its ordinary stochastic policy without privileged feedback.

### `synthetic_prompt.py`

Defines the frozen nonlinear positive-control environment. A 50-tree,
depth-four random-forest teacher is fitted to a deterministic nonlinear target
constructed from the first 12 standardized prompt PCs. The teacher produces a
probability for every eligible prompt, and a separate seeded Bernoulli draw
produces one outcome vector shared by the online and supervised evaluations.

Fitting the teacher on all prompt contexts is a transductive, outcome-free
environment-definition step: no real disagreement labels, weak/strong answers,
ARC gold answers, or train/validation labels enter it. Teacher probabilities
and unrevealed synthetic labels are never given to an online player.

### `skyline.py`

For the active prompt study, performs one stratified 80/20 split and fits linear
logistic plus the 15-leaf HGB used by the online nonlinear players on the
training portion. Classification metrics and threshold skylines
are evaluated only on validation predictions. The module also retains broader
cross-validated model-comparison functions used by earlier branches.

### `run.py`

The `simulate-llm-routing` entry point selects a manifest-defined context
profile and runs online experiments, supervised skylines, or both. For repeated
online studies it creates one deterministic permutation per order seed and
reuses that exact ordering across every method and loss value. It writes raw
per-order summaries, across-order means/SDs/standard errors, the exact compact
permutations, optional trajectories, validation predictions,
routing-rate/accuracy and cost-versus-l01 plots, and a ZIP bundle. With no
`--limit`, all 12,648 eligible BoolQ cache rows are online rounds in every
permutation. The supervised skyline runs once as a separate 4:1
train-validation task on the canonical sample collection.

### `online_tree.py`

Defines the probability-estimator boundary used only by the multiplier tuner.
The current revision-9 tuner has no ETC policy. The `--tree-estimator` option controls
SquareCB.PMSide + HGB only: its default 15-leaf HGB refits from complete
revealed history, while optional
`river-hoeffding` uses a River 0.21.2 `HoeffdingTreeClassifier` with maximum
depth 4, grace period 200, and weighted `learn_one` updates that preserve
SquareCB.PMSide inverse-propensity weights. The shared logistic backend fits a
`StandardScaler` followed by L2 logistic regression using every feature.
SquareCB.PMSide + linear logistic supplies inverse-propensity weights from its
own revealed history and is unaffected by the tree-backend option. Aggregated
Mondrian forests were
not added because River's implementation does not accept the per-example
weights required by the SquareCB.PMSide estimator.

### `pgts.py`

Implements the Pólya-Gamma Gibbs kernel underlying Algorithm 1 of *Apple
Tasting Revisited*. The prior is `N(0, prior_std^2 I)` and the default is
`prior_std=1`. Each posterior-update call performs exactly `M=15` complete
Pólya-Gamma/Gaussian Gibbs transitions, warm-started from the preceding final
draw, and returns only the final parameter sample. A proper prior keeps the
conditional precision positive definite; the Gaussian draw is computed from
its Cholesky factor without explicitly forming a covariance inverse.

The module imports `polyagamma` lazily and accepts an injected sampler for
offline tests. It sees only the feature rows and binary disagreement outcomes
that earlier action-1 rounds revealed. It does not own the full outcome stream,
perform inverse-propensity weighting, or apply the tuner's adaptive snapshot
schedule. The tuner exposes `M` through `--pgts-gibbs-steps`, the base prior
scale through `--pgts-base-prior-std`, and uses `--policy-seed` for both
Gaussian and Pólya-Gamma draws. Revision 9 tunes PG-TS pointwise by calling the
kernel with `prior_std = --pgts-base-prior-std * multiplier` for every common
multiplier, loss, and paired online order. Candidate rows identify the tuned
quantity as `parameter_name=prior_std` and retain its base, multiplier, and
effective value; `pgts_prior_std` repeats that effective value. The kernel is
still called only at eligible pure-doubling boundaries
when new action-1 feedback has arrived; the revision-6 historical design called
it every round.
The legacy `--pgts-prior-std` spelling remains an alias for the canonical
revision-9 base-scale option.

### `plot_style.py`

Defines the publication contract shared by every figure from `tuning.py`:
method and axis labels, Matplotlib defaults, legend/font sizes, regular and
short figure sizes, 400-DPI PNG export, matching vector-PDF export, and the
pointwise two-sided Student-`t` half-width helper. It is presentation-only and
does not alter a policy, selection, checkpoint, or scientific fingerprint.

### `tuning.py`

The `tune-llm-routing` entry point owns the exploratory pointwise multiplier
sweep. Its module form is `python -m llm_routing_simulation.tuning`. It uses all
138 manifest-defined BoolQ features, the ascending `l01` grid, 20 paired order
seeds, and multipliers `0.1, 0.3, 1, 3, 10`. Revision 9 tunes four policies:
CBPSide with base beta scale 0.5 and a separately fixed cap of 0.5,
SquareCB.PMSide + HGB with base `gamma=sqrt(n)`, and SquareCB.PMSide + linear
logistic with the same gamma rule, plus scheduled PG-TS with effective prior
standard deviation `--pgts-base-prior-std * multiplier`. ETC HGB and ETC
Linear are disabled rather than silently retained as untuned baselines.

The tuner exposes two SquareCB.PMSide curves for a controlled estimator
comparison. SquareCB.PMSide + HGB uses the nonlinear HGB primary by default;
SquareCB.PMSide + linear logistic uses regularized linear logistic regression.
They share the complete 138D context, paired
order, gamma candidate, `mu`, policy random numbers, cold-start rule,
pure-doubling schedule, and capped inverse-propensity-weighting rule. At
matched gamma, only the configured probability estimator differs. Their
realized actions can diverge, however, so they need not reveal the same rows or
realize the same propensities and weights.

The canonical command-line controls are `--squarecb-pmside-base-gamma`,
`--squarecb-pmside-mu`, and `--squarecb-pmside-min-propensity`. Omitting the
base-gamma option derives `sqrt(n)` after eligibility filtering and any limit.

Adaptive snapshots for the three non-Bayesian tuned policies change immediately
before global rounds `1,2,4,8,...`. Each snapshot uses feedback through `t-1`, and
every policy is still evaluated on every round. CBPSide freezes `theta_hat`
and `V^-1` within an epoch but evaluates
`min((0.5 * multiplier) * sqrt(x_t^T V^-1 x_t), 0.5)` for every current
context. SquareCB.PMSide + HGB refits HGB on its revealed history, or applies
buffered weighted River updates, at the same boundaries. SquareCB.PMSide +
linear logistic refits its weighted logistic estimator on its own revealed
history. Fibonacci and capped doubling
remain explicit reproduction/sensitivity choices; `--adaptive-max-round-gap`
has an effect only when capped doubling is selected.

With `--include-pgts`, the tuner adds PG-TS as a fourth tuned policy. For each
`l01`, common multiplier, and paired order, its Gaussian prior standard
deviation is the base `--pgts-base-prior-std` times that multiplier. It supplies
row-normalized context plus an intercept and performs its first `M`-transition
Gibbs draw from the candidate prior at round 1. At each later estimator
boundary, it performs the configured number of complete Gibbs transitions using
action-1 feedback through the preceding round only if new feedback has arrived
since the previous PG-TS draw. It otherwise retains its current parameter. The
final draw is frozen and reused throughout the epoch. PG-TS participates in
pointwise lowest-mean-cost selection and the multiplier plot. Its default
`M=15` and base prior standard deviation 1 are exposed, and `--policy-seed`
controls its random stream. Its public method label remains
`PG-TS (Bayesian logistic)`. This is a scheduled approximation for runtime, not
literal every-round Algorithm 1.

Every completed candidate is written atomically under `checkpoints/`; an
identical command skips complete candidates. Selection minimizes mean realized
total cost separately for each of the four tuned policies and `l01`. The two
SquareCB.PMSide variants
can select different gamma multipliers. The tuner exports both this
best-vs-best comparison and matched-multiplier comparisons. Ties favor the
value closest to 1 and then the smaller value.

The comparison artifacts use the `squarecb_pmside_tree_vs_linear` stem for
by-order and aggregate best-vs-best CSV/JSON, corresponding `_matched` tables,
and both separately tuned and matched `_cost_difference.{png,pdf}` figure
pairs.

Random is computed analytically after SquareCB.PMSide + HGB selection and
matched to that selected policy's routing rate for each loss/order; it is not
multiplier tuned.
The same 20 orders are used for selection and error bars, so the selected curves
are optimistic exploratory selection envelopes rather than unbiased evaluations
of policies fixed in advance.

Finalization also writes complete selected-policy learning curves. For outcome
`y_t`, a strong route costs 1 and a weak route costs `l01*y_t`. Tuned policies
including scheduled PG-TS use the multiplier already selected at that loss,
and analytic Random uses expected per-round cost
`q + (1-q)*l01*y_t`, where `q` is the selected SquareCB.PMSide + HGB routing
rate for that loss/order. Thus the Random trajectory is an analytic
expectation, not sampled random actions.

Before the online-order loop, revision 9 retains revision 8's shared fixed
reference probability per eligible row from five-fold stratified shuffled
cross-fitting.
The splitter uses `--seed`; every fold trains the exact 15-leaf
`ONLINE_HGB_PROFILE` on the complete selected 138D context and cached
weak/strong-disagreement labels. Consequently, each row's probability comes
from a model that did not train on that row. For each `l01`, the row-aligned
reference action is `1[p_oof >= 1/l01]`. These probabilities and actions are
fixed once and only reordered for each online permutation. They are strictly an
offline evaluation comparator: no reference prediction, action, or outcome is
provided to an online estimator. `cross_fitted_hgb_reference.npz` preserves the
row-aligned float64 probabilities, fold assignments, IDs, and outcomes;
`cross_fitted_hgb_reference_results.csv/json` summarizes its policy at each
loss. The study fixes `--reference-folds 5`; fold count and split seed are part
of the revision-9 manifest and fingerprint.

The primary empirical regret curve is
`sum_{s<=t}(cost_s-cost_s^reference)`. It can be negative. Finalization also
computes `R_t/t` and retains `sum_{s<=t}(cost_s-y_s)` only as
`cumulative_clairvoyant_excess_cost`, an explicitly named outcome-aware
diagnostic. The latter is not the primary regret.
`selected_learning_curves_by_order.npz` stores compact
per-policy, loss, order, and round arrays plus reference metadata;
`selected_learning_curves.csv` stores per-policy/loss/round means, sample SDs,
SEMs, pointwise CI limits for both plotted regret metrics, and
`confidence_level`, `confidence_df`, and `confidence_t_critical`. The exact CI
fields are `cumulative_reference_regret_ci95_lower`,
`cumulative_reference_regret_ci95_upper`,
`average_reference_regret_ci95_lower`, and
`average_reference_regret_ci95_upper`. There is deliberately no aggregate JSON
copy. Each loss receives
`selected_cumulative_reference_regret_l01-<float_slug>.{png,pdf}` and
`selected_average_reference_regret_l01-<float_slug>.{png,pdf}`.
All tuning figures with replicated-online-order uncertainty use fixed,
pointwise, two-sided 95% Student-`t` intervals. This includes both learning
curves, selected routing-rate versus accuracy, selected total cost, and both
tree-versus-linear difference figures; the selected-multiplier figure has no
uncertainty display. At each plotted point, `df=n_trials-1` and
`margin=t.ppf(0.975,df)*sample_sd/sqrt(n_trials)`; the plotted limits are the
mean plus or minus this margin, with no 0.5 multiplier. The intervals are
pointwise rather than simultaneous or post-selection adjusted. Non-learning
aggregate tables retain mean, sample SD, SEM, and `online_order_repeats`; the
renderer derives their CI half-widths without adding new aggregate-table CI
fields. The per-round learning CSV retains the explicit bounds listed above.

`src/llm_routing_simulation/plot_style.py` is the single publication-style
boundary. It exposes `AXIS_LABELS`, `METHOD_LABELS`,
`PLOT_CONFIDENCE_LEVEL=0.95`, `LEGEND_FONT_SIZE=8.0`,
`PUBLICATION_FIGSIZE=(7.0, 4.25)`,
`PUBLICATION_FIGSIZE_SHORT=(7.0, 3.8)`, and `PUBLICATION_PNG_DPI=400`;
`publication_pyplot` centralizes Matplotlib settings and
`save_publication_figure` emits both formats. The public labels are exactly
`PG-TS (Bayesian logistic)` and `Random`. Every tuning figure is title-free and
footnote-free and is saved as a same-stem vector PDF and 400-DPI PNG. The
selected-cost axes render as `$\ell_{01}$` and `Total cost`; the cumulative
reference-regret axes render as `Round (t)` and `Regret (excess cost)`. This
presentation metadata is outside the candidate-checkpoint fingerprint, so a
compatible completed revision-9 sweep can regenerate its tables and publication
figures with `--plot-only` without replaying an online policy.
The final selected cost-versus-`l01` figure remains part of the output set.
Because selection and trajectories reuse the same 20 orders, these are not holdout
learning curves; the cross-fitting pertains to the fixed reference predictor,
not hyperparameter selection.

The full revision-9 design has 3,600 checkpoints, 180 execution groups and
candidate aggregates, 36 multiplier selections, 900 selected order rows after
Random, and 45 selected summaries. PG-TS contributes 900 checkpoints and 45
groups/aggregates, with no fixed-policy rows. Omitting `--include-pgts` creates
a distinct three-policy design rather than fixed PG-TS rows. The matched-gamma
SquareCB.PMSide estimator comparison still contains 900 order rows and 45
aggregates.
The manifest design is exactly
`pointwise-online-parameter-multiplier-sweep-v9` with implementation revision
9. Its `candidate_counts` records
`multiplier_tuned_checkpoints=3600`, `pgts_prior_tuned_checkpoints=900`,
`total_checkpoints=3600`, `multiplier_tuned_execution_groups=180`,
`pgts_prior_tuned_execution_groups=45`, and `total_execution_groups=180`.

Pure doubling has 14 boundaries through round 8,192 for `n=12,648`, permitting
at most 13 post-feedback snapshot refits and leaving a potentially stale final
epoch of 4,457 rounds. Scheduled PG-TS has at most 14 posterior draws/model
updates in total: the round-1 draw plus 13 later draws. It can have fewer later
draws when an epoch reveals no new action-1 feedback. With `M=15`, this is at
most 210 Gibbs transitions per trajectory rather than `15 * 12,648` in the
historical faithful design.

Revision-9 fingerprints are intentionally incompatible with revision-8 and
earlier output directories. Completed revision-3 doubling, revision-4
Fibonacci, and revision-5 capped-doubling artifacts, the incomplete gap-8
attempt, the superseded gap-32 seven-multiplier design, revision-7
outcome-oracle plots, and the superseded revision-8 fixed-PG-TS reference-regret
plan remain historical records described in `EXPERIMENTS.md`. The full study
uses the fresh
`boolq-138d-cbpside-squarecb-pmside-pgts-prior-tuned-results` directory.

### `prompt_embeddings.py`

Builds semantic text from only the question and labeled choices, runs a frozen
sentence encoder locally, and writes a versioned ZIP sidecar aligned by example
ID. Its manifest records the encoder, hashes, dimensionality, and the guarantee
that answer/outcome features were not used.

### `prompt_experiment.py`

Fits a 32-dimensional fixed transductive outcome-free PCA to prompt embeddings,
then evaluates the current 78D context, prompt-only 32D context, and compact 46D
hybrid (14 uncertainty plus 32 prompt features) on identical out-of-fold splits.
It also uses prompt features alone to predict residuals from the current-context
logistic baseline. A separate nested cross-fitted diagnostic adds that predicted
residual to the base probability and evaluates the corrected probability on
classification metrics and a routing skyline. Multiple split seeds quantify
cross-validation sensitivity; this diagnostic does not alter the online player
or environment interfaces.

## Branch-specific additions

- `experiment/synthetic-sanity` adds `synthetic.py` and the
  `simulate-synthetic-nonlinear` entry point for
  `P(Y=1|X)=sigmoid(x1*x2)`.
- `experiment/residual-diagnostics` adds out-of-fold raw, Pearson, and deviance
  residual tables plus binned residual plots for logistic, selected HGB, and
  selected MLP models.
- `experiment/prompt-embedding` adds a frozen semantic sidecar, a three-context
  supervised comparison, an incremental prompt-residual test, and a two-stage
  corrected-probability skyline with split-seed stability results.
- `experiment/prompt-routing` supports prompt-only, uncertainty-plus-prompt,
  and complete 142D manifest contexts with cached disagreement for a separate
  supervised skyline and full-stream online routing evaluation. It retains the
  multifeature forest-generated synthetic positive control.
- `experiment/boolq-cbpside-beta1` preserves those context studies and adds a
  complete 138D BoolQ follow-up with CBPSide scale 0.5/cap 0.5, ETC
  `ceil(n^(2/3))` tastes, IGW `gamma=sqrt(n)`, and a ten-order robustness study
  whose plots show sample variability across paired online permutations.
  Adaptive CBPSide and IGW model fits are batched every five new tastes; ETC is
  still fitted once and frozen. Its separate pointwise tuning path uses 20
  paired orders and candidate-level resume. Revision-3 pure-doubling and
  revision-4 Fibonacci artifacts were completed, although their numerical
  results were not analyzed during the revision-5 code change. Its completed
  revision-5 gap-500, gap-100, and gap-32 five-multiplier artifacts are retained.
  A later gap-8 attempt is incomplete at 3,902 of 4,500 checkpoints, and the
  gap-32 seven-multiplier design is retained as a superseded plan. The current
  revision-9 study disables both ETC variants, retains the five-value
  multiplier grid, and tunes CBPSide, SquareCB.PMSide + HGB,
  SquareCB.PMSide + linear logistic, and scheduled PG-TS. It exports
  separately tuned best-vs-best and fixed-multiplier matched-gamma
  SquareCB.PMSide estimator comparisons; action-dependent histories may differ
  in either view. The optional weighted River sensitivity changes only
  SquareCB.PMSide + HGB.
  PG-TS retains boundary-scheduled posterior updates and 15 Gibbs transitions,
  while tuning `--pgts-base-prior-std * multiplier` pointwise. Analytic Random is
  still matched to selected SquareCB.PMSide + HGB, not to an ETC policy. The revision-6
  every-round Algorithm 1 implementation remains historical provenance only.
  Finalization always records compact by-order and aggregate per-round cost,
  fixed cross-fitted HGB-15 reference-policy regret, average regret `R_t/t`,
  and explicitly named outcome-aware excess-cost diagnostics. Per-loss
  PDF/400-DPI-PNG pairs plot both cumulative and average reference-policy
  regret.

Refer to `EXPERIMENTS.md` for motivations, results, and exact decisions rather
than inferring research intent from implementation details alone.
