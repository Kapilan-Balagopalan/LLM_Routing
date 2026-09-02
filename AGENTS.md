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
