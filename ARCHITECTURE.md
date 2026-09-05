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
    -> run.py finds requested feature blocks from manifest context_blocks
    -> run.py selects prompt-only or uncertainty-then-prompt context
    -> run.py selects cached disagreement or the optional synthetic positive control
    -> environment.py emits the current context and weak answer
    -> player.py defines the act-then-update protocol
    -> algorithm.py chooses action 0 or 1 from revealed history
    -> environment.py reveals the active outcome only when action 1 was selected
    -> run.py records trajectories, metrics, tables, and plots
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

On `experiment/prompt-routing`, `run.py` obtains every block boundary by name
from `manifest.context_blocks`. The active real-label study concatenates the
complete 14-dimensional `uncertainty` block with the first 32 components of
`prompt_embedding_pca`, in that order, for a 46-dimensional context. It excludes
the hidden-state block. The prompt-only profile remains available to reproduce
earlier 20D, 32D, and 64D studies.
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

- `HGBETCPlayer`: routes strongly for 300 rounds, fits one histogram gradient
  boosting classifier from those revealed labels, and freezes that estimator.
  The active capacity study repeats this policy with 3, 7, and 15 maximum
  leaves per boosting tree while holding every other HGB setting fixed.
- `LogCBPSideATPlayer`: estimates a regularized linear logistic disagreement
  model and applies the CBPSide confidence rule without forced tastes.
- `IGWPlayer`: estimates disagreement using an online-refitted histogram
  gradient boosting classifier and samples an arm using inverse-gap weighting.
  The active study uses `mu=2`, fixed `gamma=64`, and the same 3-, 7-, and
  15-leaf HGB capacity profiles as ETC.
- `RevealedFeedbackEstimator`: extracts only action-1 observations and applies
  capped inverse-propensity weights when supplied.

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
logistic plus the three exact HGB capacity profiles used by the online nonlinear
players on the training portion. Classification metrics and threshold skylines
are evaluated only on validation predictions. The module also retains broader
cross-validated model-comparison functions used by earlier branches.

### `run.py`

The `simulate-llm-routing` entry point selects a manifest-defined context
profile and runs online experiments, supervised skylines, or both. It writes
reproducibility metadata, per-method summaries,
optional synthetic outcome metadata, validation predictions, full online
trajectories, plots, and a ZIP bundle. With no `--limit`, every one of the 5,138
eligible cache rows is an online round. The supervised skyline remains a
separate 4:1 train-validation task over the same selected outcome source.

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
- `experiment/prompt-routing` supports prompt-only and 14D-uncertainty plus
  prompt-PCA contexts with cached disagreement for a separate supervised
  skyline and full-stream online routing evaluation. It retains the
  multifeature forest-generated synthetic positive control.

Refer to `EXPERIMENTS.md` for motivations, results, and exact decisions rather
than inferring research intent from implementation details alone.
