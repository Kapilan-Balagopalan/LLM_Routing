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
llm-routing-cache-full.zip
    -> cache.py validates records and arrays
    -> run.py finds prompt_embedding_pca from manifest context_blocks
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

The v2 cache manifest records:

- schema: `llm-routing-cache-v2`;
- dataset: `allenai/ai2_arc`, `ARC-Easy`, train, validation, and test splits;
- 5,173 collected rows and 5,138 eligible rows;
- weak model: `Qwen/Qwen2.5-0.5B-Instruct`;
- strong model: `meta-llama/Llama-2-13b-chat-hf`;
- outcome: `1` exactly when extracted weak and strong answers differ;
- raw concatenated hidden dimension: 4,480;
- fixed, transductive, outcome-free PCA: 64 components;
- full saved context: 14 uncertainty features, 64 hidden-state PCA features,
  and 64 prompt-embedding PCA features, standardized to 142 dimensions.

The 14 uncertainty features contain choice entropy, normalized choice entropy,
top probability, top-two margin, one-minus-top probability, next-token
vocabulary entropy, four option probabilities, and four option log likelihoods.

On `experiment/prompt-routing`, `run.py` obtains the block boundaries by finding
`prompt_embedding_pca` in `manifest.context_blocks`. It exposes all 64 prompt
components to the players and excludes the uncertainty and hidden-state blocks.
The PCA was constructed during collection across all prompts without outcomes.
An optional experiment limit is applied only after block selection.

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

- `RBFSVMETCPlayer`: routes strongly for 300 rounds, fits a calibrated RBF SVM
  once from revealed feedback, and freezes that estimator.
- `LogCBPSideATPlayer`: estimates a regularized linear logistic disagreement
  model and applies the CBPSide confidence rule without forced tastes.
- `IGWPlayer`: estimates disagreement using an online-refitted calibrated RBF
  SVM and samples an arm using inverse-gap weighting with `mu=2`, `gamma=16`.
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
- `experiment/prompt-routing` uses the fixed prompt-only 64D context for both
  supervised skylines and the online routing players.

Refer to `EXPERIMENTS.md` for motivations, results, and exact decisions rather
than inferring research intent from implementation details alone.
