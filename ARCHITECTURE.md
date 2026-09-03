# Architecture

## Purpose

This CPU-only project replays routing algorithms against a previously collected
LLM cache. It does not load an LLM, download models, call Hugging Face, or need
`HF_TOKEN`. LLM generation is owned by the separate
`Data_collection_LLM_routing` project.

## Data flow

```text
llm-routing-cache.zip
    -> cache.py validates records and arrays
    -> environment.py emits the current context and weak answer
    -> player.py defines the act-then-update protocol
    -> algorithm.py chooses action 0 or 1 from revealed history
    -> environment.py reveals feedback only when action 1 was selected
    -> run.py records trajectories, metrics, tables, and plots
```

At round `t`, a player receives only the current context. It selects:

- `0`: retain the weak answer; disagreement remains hidden.
- `1`: use the strong answer; disagreement is revealed as `0` or `1`.

The environment owns the complete cached stream. Players store only the action,
context, and feedback that the environment actually revealed.

## Cache and context

The current cache manifest records:

- schema: `llm-routing-cache-v1`;
- dataset: `allenai/ai2_arc`, `ARC-Easy`, test split;
- 2,365 collected rows and 2,351 eligible rows;
- weak model: `Qwen/Qwen2.5-0.5B-Instruct`;
- strong model: `meta-llama/Llama-2-13b-chat-hf`;
- outcome: `1` exactly when extracted weak and strong answers differ;
- raw concatenated hidden dimension: 4,480;
- fixed, transductive, outcome-free PCA: 64 components;
- final context: 14 uncertainty features plus 64 whitened PCA features,
  featurewise standardized to 78 dimensions.

The 14 uncertainty features contain choice entropy, normalized choice entropy,
top probability, top-two margin, one-minus-top probability, next-token
vocabulary entropy, four option probabilities, and four option log likelihoods.

## Core modules

### `cache.py`

Validates the ZIP manifest, records, and NumPy arrays. It filters ineligible
extraction rows and can reconstruct a smaller context using saved PCA axes.

### `environment.py`

`LLMCascadeEnvironment` enforces partial monitoring. The strong answer and
disagreement outcome are returned only following action `1`.

### `player.py`

`HistoryBasedPlayer` defines the stateful interface:

1. `next_action(current_context)`;
2. environment transition;
3. `update(action, context, revealed_outcome)`.

It checks that updates correspond to the pending action and context.

### `algorithm.py`

- `XGBoostETCPlayer`: routes strongly for its exploration period, fits XGBoost
  once from revealed feedback, and freezes that estimator.
- `LogCBPSideATPlayer`: estimates a regularized linear logistic disagreement
  model and applies the CBPSide confidence rule.
- `IGWPlayer`: estimates disagreement using weighted XGBoost and samples an arm
  using inverse-gap weighting.
- `RevealedFeedbackEstimator`: extracts only action-1 observations and applies
  capped inverse-propensity weights when supplied.

### `skyline.py`

Produces stratified out-of-fold disagreement probabilities and threshold
skylines. The real-data comparison includes standard logistic, several HGB
capacities, MLP-4/MLP-8, elastic-net logistic, Extra Trees, calibrated RBF SVM,
XGBoost, and optionally CatBoost. On the residual branch it also runs a nested
cross-fitted HGB regression test for feature-predictable structure remaining in
the logistic residuals, with a shuffled-training-residual permutation reference.

### `run.py`

The `simulate-llm-routing` entry point runs online experiments, supervised
skylines, or both. It writes reproducibility metadata, per-method summaries,
full online trajectories, plots, and a ZIP bundle.

### `residual_experiment.py`

The `analyze-holdout-residuals` entry point reads schema-v2 manifest block
boundaries and uses one stratified 50/50 split across all eligible rows. It
compares validation-set binned logistic residuals for the combined uncertainty
and hidden-state blocks, the prompt-only block, and fake labels produced by a
probabilistic depth-two decision-tree teacher. Its thresholds are training-set
feature medians, while alternating near-deterministic leaf probabilities create
a strong gated interaction. Its Bernoulli labels are generated without real
outcomes or validation information. No ARC gold answers are used.

The diagnostic uses approximately equal-size bins ordered by validation
predicted probability, with `round(sqrt(n_validation))` bins. It saves
pointwise raw, Pearson, and deviance residuals, bin means and intervals, tree
leaf summaries with intervals, separate probability-bin and tree-leaf plots,
metadata, and a ZIP bundle.

## Branch-specific additions

- `experiment/synthetic-sanity` adds `synthetic.py` and the
  `simulate-synthetic-nonlinear` entry point for
  `P(Y=1|X)=sigmoid(x1*x2)`.
- `experiment/residual-diagnostics` adds out-of-fold raw, Pearson, and deviance
  residual tables plus binned residual plots for logistic, selected HGB, and
  selected MLP models. It also contains the separate schema-v2 1:1 holdout
  comparison across uncertainty+hidden, prompt-only, and tree-generated-label
  settings.

Refer to `EXPERIMENTS.md` for motivations, results, and exact decisions rather
than inferring research intent from implementation details alone.
