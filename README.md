# Offline LLM-routing algorithm simulation

This project reads the one-time cache produced by `Data_collection_LLM_routing`.
It does not load an LLM, contact Hugging Face, need an `HF_TOKEN`, or require a
GPU. On `experiment/prompt-routing`, the active study uses the first 20
components of the cached prompt-PCA block to predict true cached weak/strong
disagreement. A committed synthetic-label positive control remains available
through `--outcome-source synthetic`.

## Set up on the laptop

Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

Copy `llm-routing-cache-full.zip` into this folder. It already contains the
64-dimensional prompt-PCA block, so no separate embedding sidecar is needed for
the routing experiment. Run every current experiment with:

```powershell
simulate-llm-routing `
  --cache llm-routing-cache-full.zip `
  --outcome-source cached `
  --experiment all
```

For a quick check, use a prefix:

```powershell
simulate-llm-routing `
  --cache llm-routing-cache-full.zip `
  --experiment skyline `
  --limit 100
```

The useful choices are:

- `--experiment skyline`: a separate supervised 80/20 train-validation
  comparison of only linear logistic and the same HGB profile used online.
- `--experiment online`: HGB ETC, linear-logistic CBPSide, five HGB IGW gamma
  settings, and random routing matched to ETC traffic. With no `--limit`, all
  5,138 eligible samples are online rounds; there is no supervised training
  subset.
- `--experiment all`: run both separate evaluations against the cached
  weak/strong disagreement labels (the default outcome source).

Results go to `simulation-results/`, including CSV/JSON tables, the full online
trajectory, `routing_comparison.png`, and a portable `simulation-results.zip`.
Each execution refits all estimators from the cached samples; no LLM generation
is repeated. You can run new seeds, loss values, thresholds and algorithms many
times against the same cache.

`routing_comparison.png` keeps the two tasks in separate panels. The supervised
panel plots only logistic and HGB. Exact validation probabilities are saved in
`supervised_holdout_predictions.csv`; all skyline points and model metrics
remain in CSV and JSON. Synthetic mode additionally writes
`synthetic_outcomes.csv`.

## Current defaults

- Context: the first 20 components of `prompt_embedding_pca`, selected from the
  v2 manifest block in PCA order. Weak-model uncertainty and hidden-state PCA
  blocks are not used.
- Outcome: true cached weak/strong disagreement. The cached strong-model answer
  is the routing reference; ARC gold answers are never routing labels.
- Loss grid: ten values obtained from evenly spaced decision thresholds
  `alpha = 0.55, ..., 0.30` using `l01 = 1 / alpha` because `l11 = 1`.
- ETC: 300 initial tastes, then one frozen HGB estimator.
- CBPSide: regularized linear logistic regression with no forced tastes and no
  class bootstrap.
- IGW: five online-refitted HGB policies with
  `gamma = 8, 16, 32, 64, 128`, `mu = 2`, no forced tastes or class bootstrap,
  and inverse-propensity weights capped at 10.
- Online accuracy: agreement with the cached strong-model reference, not ARC
  gold-answer accuracy.
- Supervised skyline: one stratified 4:1 split, with HGB and logistic fit only on
  the training 80% and evaluated only on the validation 20%.

Pass alternatives on the command line, for example:

```powershell
simulate-llm-routing `
  --cache llm-routing-cache-full.zip `
  --l01-values 1 2 4 8 `
  --seed 7
```

The prompt dimension defaults to the first 20 of 64 available components. The simulator
locates `prompt_embedding_pca` through `manifest.json -> context_blocks`; it
does not hardcode column numbers. `--prompt-components` can select a smaller
prefix for a later ablation.

To reproduce the earlier synthetic positive control, use:

```powershell
simulate-llm-routing `
  --cache llm-routing-cache-full.zip `
  --outcome-source synthetic `
  --prompt-components 64 `
  --igw-gamma-values 16 `
  --l01-values 1.82 2.22 2.67 3.33 `
  --output-dir prompt-forest-sanity-results
```

Its teacher uses 50 depth-four random-forest trees and prompt PCs 1--12. It uses
no real disagreement label, model answer, or ARC gold answer.

## Earlier external prompt-embedding diagnostic

This earlier diagnostic uses a separate MiniLM sidecar to compare prompt
features with the old weak-model context. It is retained for reproducibility;
the current prompt-routing command instead reads the 64D prompt block already
stored in the v2 full cache.

Install the optional local encoder dependency:

```powershell
python -m pip install -e ".[embedding,test]"
```

Encode each question and its labeled choices using the default public frozen
encoder, `sentence-transformers/all-MiniLM-L6-v2`:

```powershell
collect-prompt-embeddings `
  --cache llm-routing-cache.zip `
  --output prompt-embeddings.zip `
  --device cpu
```

The first invocation may download the encoder weights from Hugging Face. The
embedding cache contains no outcomes, answer labels, or model responses and can
be reused for every later simulation. The semantic text is exactly the question
plus labeled choices; repeated generation instructions are excluded.

Run the controlled CPU comparison:

```powershell
simulate-prompt-embedding-context `
  --cache llm-routing-cache.zip `
  --prompt-embeddings prompt-embeddings.zip `
  --prompt-components 32 `
  --residual-permutations 100 `
  --robustness-seeds 5
```

It uses identical folds to compare the current 78-dimensional context, the
32-dimensional prompt PCA context, and a compact 46-dimensional context made
from 14 uncertainty features plus the 32 prompt features. The 64-dimensional
weak-model hidden-state PCA is deliberately excluded from the compact hybrid.
It also directly tests whether prompt features predict held-out residuals from
the current logistic model. Prompt PCA is fixed, transductive, and outcome-free.
The nested cross-fitted two-stage diagnostic adds the predicted prompt residual
to the current-context logistic probability, clips the result to a valid
probability, and compares it with the base probability on AUC, log loss, Brier
score, 50% routing accuracy, and the full routing skyline. By default it repeats
the cross-fitting with five consecutive split seeds; only the primary seed runs
the permutation test. These repeated seeds measure split stability on the same
data, not performance on independent datasets. This remains a supervised
diagnostic and is not used by the online routing players.
Results are written to `prompt-embedding-results/` and bundled as
`prompt-embedding-results.zip`. The prompt-component count remains configurable.
The initial primary run used 16 components; 32 is a documented sensitivity run
motivated by the initial PCA retaining only 37.05% of embedding variance.

## Where to edit

- `src/llm_routing_simulation/algorithm.py`: ETC, CBPSide and IGW players.
- `src/llm_routing_simulation/skyline.py`: supervised logistic/HGB skylines.
- `src/llm_routing_simulation/environment.py`: partial-feedback rules.
- `src/llm_routing_simulation/run.py`: experiment parameters, result tables and
  plots.
- `src/llm_routing_simulation/cache.py`: cache validation and context selection.
- `src/llm_routing_simulation/prompt_embeddings.py`: outcome-free prompt text,
  frozen encoding, and portable sidecar validation.
- `src/llm_routing_simulation/prompt_experiment.py`: three-context supervised
  comparison, incremental prompt-residual test, and two-stage stability check.

Run tests with `python -m pytest`.
