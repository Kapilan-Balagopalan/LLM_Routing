# Project instructions for coding agents

## Start every task

1. Read `README.md`, `ARCHITECTURE.md`, and `EXPERIMENTS.md` before editing.
2. Inspect `git status --short --branch` and `git log --oneline -5`.
3. Confirm that the checked-out branch matches the requested experiment.
4. If the branch is wrong or the working tree has unrelated changes, stop and
   explain the situation instead of moving or overwriting user work.

## Branch roles

- `main`: real ARC-Easy cached-data routing and supervised skyline baseline.
- `experiment/synthetic-sanity`: nonlinear `x1*x2` synthetic sanity check.
- `experiment/residual-diagnostics`: real-data Logistic/HGB/MLP residual work.
- `experiment/prompt-embedding`: outcome-free semantic prompt augmentation and
  its controlled supervised/residual comparison.
- `experiment/prompt-routing`: frozen prompt-routing and BoolQ empirical-scale
  0.25 baseline at commit `a95a3ae`.
- `experiment/boolq-cbpside-beta1`: BoolQ confidence-scale and context
  follow-up; the active study uses all 138 features, CBPSide scale 0.5/cap
  0.5, ETC `ceil(n^(2/3))` tastes, IGW `gamma=sqrt(n)`, and ten paired shuffled
  online orders with sample-SD error bars. CBPSide and IGW refit after every
  five additional tastes; ETC still fits once and freezes. The separate
  `tune-llm-routing` study extends this baseline to 20 paired orders and
  pointwise multipliers `0.03, 0.1, 0.3, 1, 3, 10, 30`, using resumable
  candidate checkpoints. Completed revision-3 pure-doubling and revision-4
  Fibonacci artifacts are retained in their original fingerprinted output
  directories;
  their numerical results were not analyzed during the revision-5 code change,
  and current revision-5 code cannot resume or plot-only those directories.
  Use the corresponding original code revision for those archived artifacts.
  Completed revision-5 capped-doubling gap-500, gap-100, and gap-32 artifacts
  are also retained; each directory contains all 4,500 candidate checkpoints
  and its finalized tables, figures, summary, and ZIP bundle. Their numerical
  results were not analyzed during the subsequent schedule code changes.
  Current revision-5 code can resume or plot-only a completed result only with
  its original output directory and the corresponding explicit
  `--adaptive-max-round-gap 500`, `--adaptive-max-round-gap 100`, or
  `--adaptive-max-round-gap 32`. A later gap-8 attempt contains 3,902 of its
  planned 4,500 checkpoints and no finalized tables, summary, figures, or ZIP;
  preserve it as an incomplete five-multiplier artifact. The current planned,
  unrun gap-32/seven-multiplier configuration has five tuned policies:
  CBPSide, fixed 15-leaf HGB ETC, ETCLinear, IGW Tree, and IGW Linear. The two
  ETC variants share each shuffled order, forced prefix, taste budget, and unit
  training weights; each feasible prefix fits once, freezes, reuses
  probabilities across `l01`, and independently tunes its taste multiplier.
  A prefix with fewer than two rows from either class uses the recorded
  Laplace-smoothed prevalence fallback without fitting. Random is matched only
  to selected HGB ETC traffic. CBPSide and both IGW variants now default to
  capped-doubling global-round boundaries: start at 1 and set the next boundary
  to `min(2b, b+32)`. Fit before boundary `b` using feedback through `b-1`,
  while evaluating the policy every round. The ETC variants are unaffected by
  this schedule. Use `--adaptive-update-schedule fibonacci` or
  `--adaptive-update-schedule doubling` to select those boundary rules for a
  new revision-5 run, not to resume the older revision-4 or revision-3 outputs.
  Report both separately tuned best-vs-best and fixed-multiplier
  matched-gamma IGW comparisons; actions and realized feedback may diverge.
  HGB remains the nonlinear primary and `river-hoeffding` changes IGW Tree
  only. The design has 6,300 candidate rows/checkpoints and selects five learned
  policies plus Random. Execution reports 203 candidate groups: seven for each
  ETC estimator and 63 for each of the three adaptive policies. At `n=12,648`,
  capped doubling with gap 32 has 400 boundaries and at most 399 adaptive
  refits. Its last boundary is 12,640, so its final potentially stale tail is
  nine rounds. Multiplier 30 makes each ETC forced-taste budget saturate at the
  12,648-round horizon. Use the fresh
  `boolq-138d-multiplier-sweep-hgb-etc-linear-capped-doubling-gap32-multiplier7-results`
  directory; never reuse the completed five-multiplier gap-32 directory.
- `backup/current-combined`: recovery snapshot made before branch separation.

Do not mix an experiment into another branch. Shared bug fixes should be made
on `main` and then deliberately merged or cherry-picked.

## Scientific invariants

- Outcome `1` means the extracted weak and strong answers disagree; outcome `0`
  means they agree.
- Action `0` uses the weak answer and reveals no outcome to the player.
- Action `1` routes to the strong model and reveals the disagreement outcome.
- The cached strong-model answer is the evaluation reference. ARC ground truth
  is not the online routing feedback or the reported routing accuracy target.
- Online players may use only past revealed feedback and the current context.
  Never expose future outcomes, future actions, or the entire outcome vector to
  an online estimator.
- Fixed PCA is transductive but outcome-free: its axes may use collected hidden
  states, never disagreement labels.
- Exclude rows whose weak or strong answer extraction failed.

Scoped exception: `experiment/prompt-routing` can define an explicit synthetic
outcome override with `--outcome-source synthetic`. In that mode,
the fake label replaces cached disagreement for both evaluation and revealed
action-1 feedback. It must be identified as synthetic in every result; the
teacher probabilities and unrevealed labels must remain hidden from players.

## Current real-data baseline

- Dataset: ARC-Easy test split.
- Weak model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Strong reference: `meta-llama/Llama-2-13b-chat-hf`.
- Context: 14 uncertainty features plus 64 whitened PCA hidden-state features,
  then featurewise standardization (78 dimensions total).
- ETC: frozen XGBoost after 100 forced tastes.
- CBPSide: linear logistic model; 10 outcomes per class or a 50-taste cap.
- IGW: online-refitted XGBoost, `mu=2`, fixed `gamma=32`, 10 outcomes per class
  or a 50-taste cap, and inverse-propensity weights capped at 10.

## Commands and verification

Install locally with:

```powershell
python -m pip install -e ".[test]"
```

The optional incremental-tree tuning study requires:

```powershell
python -m pip install -e ".[test,online-tree]"
```

Run tests with:

```powershell
python -m pytest -q
```

Run real cached experiments with:

```powershell
simulate-llm-routing --cache llm-routing-cache.zip --experiment all
```

Generated caches, ZIP files, environments, packaging metadata, and result
directories must remain untracked. Never commit credentials or print tokens.

## Change and commit workflow

- Preserve unrelated user work and use minimal edits.
- Add or update offline tests for algorithmic changes.
- Report the active branch, changed files, and test result after each change.
- Do not commit, push, merge, or switch branches unless the user explicitly
  requests it. The user normally performs commits. Provide an exact suggested
  commit message and wait for confirmation before switching branches.
- Record research conclusions and exact run settings in `EXPERIMENTS.md`.
