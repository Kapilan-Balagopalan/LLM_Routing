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
    -> tuning.py evaluates CBPSide, ETC HGB, ETC Linear, IGW Tree, and IGW Linear candidates
    -> online_tree.py supplies batch HGB, batch logistic, or incremental Hoeffding estimators
    -> tuning.py checkpoints every completed policy/l01/multiplier/order candidate
    -> tuning.py selects the lowest-mean-cost multiplier at each policy/l01 point
    -> tuning.py adds analytic Random matched only to the selected ETC HGB traffic
    -> tuning.py writes reusable tables, figures, and a ZIP bundle
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
The ETC HGB backend is always the established scikit-learn 15-leaf HGB and uses
unit-weight prefix data. The `--tree-estimator` option controls IGW Tree only:
its default HGB refits from complete revealed history, while optional
`river-hoeffding` uses a River 0.21.2 `HoeffdingTreeClassifier` with maximum
depth 4, grace period 200, and weighted `learn_one` updates that preserve IGW
inverse-propensity weights. The shared logistic backend fits a weighted
`StandardScaler` followed by L2 logistic regression using every feature.
ETC Linear supplies unit weights from the forced prefix; IGW Linear supplies
inverse-propensity weights from its own revealed history. Neither linear policy
is affected by the tree-backend option. Aggregated Mondrian forests were not
added because River's implementation does not accept the per-example weights
required by the IGW estimator.

### `tuning.py`

The `tune-llm-routing` entry point owns the exploratory pointwise multiplier
sweep. Its module form is `python -m llm_routing_simulation.tuning`. It uses all
138 manifest-defined BoolQ features, the ascending `l01` grid, 20 paired order
seeds, and multipliers `0.1, 0.3, 1, 3, 10`. The three base rules are CBPSide
beta scale 0.5 with a separately fixed cap of 0.5, IGW `gamma=sqrt(n)`, and the
shared ETC HGB/ETC Linear budget `n^(2/3)` tastes with
`ceil(multiplier * base)` applied afterward.

The tuner exposes two IGW curves for a controlled estimator comparison. IGW
Tree uses the nonlinear HGB primary by default; IGW Linear uses regularized
linear logistic regression. They share the complete 138D context, paired order,
gamma candidate, `mu`, policy random numbers, cold-start rule, capped-doubling
schedule, and capped inverse-propensity-weighting rule. At matched gamma, only
the configured probability estimator differs. Their realized actions can
diverge, however, so they need not reveal the same feedback rows or realize the
same propensities and IPS weights. The complete design contains 4,500 learned
candidate rows: five policies times nine losses times five multipliers times 20
orders.

Adaptive snapshots for CBPSide, IGW Tree, and IGW Linear change only immediately
before capped-doubling global-round boundaries by default. Starting at `b=1`,
the next boundary is `min(2b, b+100)`: the schedule doubles early and then uses
a maximum boundary gap of 100 rounds. Each snapshot uses revealed feedback
through `t-1`, and every policy is still evaluated on every round. CBPSide
freezes both `theta_hat` and `V^-1` within an epoch but evaluates
`min((0.5 * multiplier) * sqrt(x_t^T V^-1 x_t), 0.5)` on every current context.
IGW Tree either refits HGB on the complete revealed history or applies buffered
weighted River updates at the same boundaries. IGW Linear refits its weighted
logistic estimator on its own revealed history at those boundaries. Passing
`--adaptive-update-schedule fibonacci` selects Fibonacci boundaries for a new
revision-5 run, and `--adaptive-update-schedule doubling` selects the original
`t=1,2,4,8,...` boundary rule for a new revision-5 run.
`--adaptive-max-round-gap` controls only capped doubling. Neither ETC variant
uses the adaptive schedule. Completed revision-3 doubling and revision-4
Fibonacci artifacts retain older configuration fingerprints; current
revision-5 code cannot resume or plot-only those directories, which require
their original code revision.

ETC HGB and ETC Linear receive the identical shuffled order, forced prefix,
taste budget, and unit training weights for a given order and multiplier. If
the prefix contains at least two rows from each class, each fits once and
freezes; its probabilities are reused across all `l01` values. If the shared
prefix fails that feasibility gate, neither estimator is fit and the tail uses
the Laplace-smoothed prefix prevalence. Candidate rows record the two class
counts, feasibility flag, and fallback. The two ETC policies independently
tune and select their pointwise taste multiplier. ETC HGB remains fixed to
15-leaf HGB even when IGW Tree uses the River sensitivity backend.

Every completed policy/`l01`/multiplier/order candidate is written atomically
under `checkpoints/`; repeating the same command skips complete candidates.
After all candidates exist, the tuner chooses the multiplier with the lowest
mean realized total cost separately for each policy and `l01`; IGW Tree and IGW
Linear never share a forced winner and their winning gammas may differ. The
`igw_tree_vs_linear_by_order.csv/json`, `igw_tree_vs_linear.csv/json`, and
`igw_tree_vs_linear_cost_difference.png` are therefore a separately tuned
best-vs-best comparison. The matched-gamma exports
`igw_tree_vs_linear_matched_by_order.csv/json`,
`igw_tree_vs_linear_matched.csv/json`, and
`igw_tree_vs_linear_matched_cost_difference.png` pair the candidates at every
common multiplier for a configured estimator contrast. Both views pair the
outer online order, but neither forces identical realized feedback histories.
Ties favor the value closest to 1 and then the smaller value. Random is computed
analytically after ETC HGB selection rather than by an inner Monte Carlo loop;
it is not matched to ETC Linear. The selected output therefore contains five
learned policies plus Random. The selection and plotted error bars reuse the
same 20 orders, so these figures are an optimistic exploratory oracle envelope,
not an unbiased evaluation of a preselected policy.

Relative to the earlier four-policy design, ETC Linear adds 900 candidate rows.
At `n=12,648`, capped doubling with a 100-round maximum gap has 133 boundaries
and permits at most 132 adaptive refits after feedback exists. Its full-history
row-work upper bound is 803,622, `28.06x` Fibonacci, `49.09x` pure doubling,
and `4.92x` the completed gap-500 configuration, while its final potentially
stale tail is 21 rounds instead of 137, 1,703, or 4,457. This is a very large
runtime increase. Candidate-level checkpoints make an identical-command resume
safe; this configuration must use a fresh output directory. Finalization writes
five figures, including both the separately tuned and matched-gamma IGW
cost-difference plots. The gap-500 revision-5 sweep completed all 4,500
candidate rows; its numerical results were not analyzed during the gap-100
code change. The gap-100 configuration is planned and implemented but has not
been run as a full experiment.

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
  revision-5 gap-500 artifacts are retained, while the current planned, unrun
  revision-5 study uses default capped-doubling global-round
  epochs with a 100-round maximum gap for CBPSide and both IGW variants, keeps
  explicit Fibonacci and pure-doubling boundary choices for new revision-5
  comparisons, retains independently tuned ETC Linear beside fixed 15-leaf
  ETC HGB, and keeps the
  matched IGW Tree/IGW Linear comparison. The optional weighted River
  Hoeffding sensitivity changes IGW Tree only. It exports both separately tuned
  best-vs-best and fixed-multiplier matched-gamma IGW comparisons;
  action-dependent histories may differ in either view.

Refer to `EXPERIMENTS.md` for motivations, results, and exact decisions rather
than inferring research intent from implementation details alone.
