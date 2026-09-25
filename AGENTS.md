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
  follow-up. The current planned revision-9 tuning study uses all 138
  manifest-defined features and four multiplier-tuned policies: CBPSide linear
  logistic, SquareCB.PMSide + linear logistic, SquareCB.PMSide + HGB, and
  scheduled PG-TS Bayesian logistic. ETC HGB and ETC Linear are
  disabled in this design. The common multiplier grid is
  `0.1, 0.3, 1, 3, 10`, evaluated at nine `l01` values over 20 paired shuffled
  orders. CBPSide and both SquareCB.PMSide variants make a decision every
  round but update estimator snapshots only before pure-doubling global rounds
  `1,2,4,8,...`, using feedback through the preceding round.
  SquareCB.PMSide + HGB remains the nonlinear 15-leaf HGB primary;
  SquareCB.PMSide + linear logistic is its regularized linear comparison.
  Random is computed analytically after selection and is matched to the
  selected SquareCB.PMSide + HGB routing rate for each loss/order.
  Use the canonical options `--squarecb-pmside-base-gamma`,
  `--squarecb-pmside-mu`, and `--squarecb-pmside-min-propensity`; comparison
  artifacts use the `squarecb_pmside_tree_vs_linear` stem.
  The full design has 3,600 candidate checkpoints, 180 execution groups and
  candidate aggregates, and 36 pointwise multiplier selections. PG-TS accounts
  for 900 checkpoints and 45 groups/aggregates; there are no fixed-PG-TS rows.
  Its manifest design is exactly
  `pointwise-online-parameter-multiplier-sweep-v9` with implementation
  revision 9.

  Finalization always records selected-policy learning curves at every online
  round for every `l01`. Each of the four tuned policies uses its pointwise
  selected multiplier, and analytic Random is
  matched to selected SquareCB.PMSide + HGB traffic. Before any online-order
  evaluation, build one fixed five-fold stratified out-of-fold HGB-15 reference
  probability for every eligible row. Use `--reference-folds 5`, split seed
  `--seed`, the complete
  selected 138D context, cached weak/strong disagreement labels, and the exact
  `ONLINE_HGB_PROFILE`. Each row's reference probability must come from a fold
  model that did not train on that row. For loss `l01`, the reference routes
  strong exactly when `p_oof >= 1/l01`; its row-aligned actions are fixed once
  and only reordered with each online permutation. The reference is an offline
  evaluation comparator and must never provide feedback or predictions to an
  online player. Persist its row-level probabilities as float64 in
  `cross_fitted_hgb_reference.npz` with IDs, outcomes, and fold assignments.

  For binary disagreement `y_t`, realized action cost is `1` after a strong
  route and `l01*y_t` after a weak route. The primary empirical reference-policy
  regret is `sum_{s<=t}(cost_s-cost_s^reference)` and may be negative. Plot both
  `selected_cumulative_reference_regret_l01-<float_slug>.{png,pdf}` and
  `selected_average_reference_regret_l01-<float_slug>.{png,pdf}` for every
  loss.
  Every tuning figure with replicated-online-order uncertainty uses a fixed,
  pointwise, two-sided 95% Student-`t` mean confidence interval. This includes
  both learning-curve families, selected routing-rate versus accuracy, selected
  total cost, and the separately tuned and matched tree-versus-linear cost
  differences. The selected-multiplier plot has no uncertainty display. At
  each plotted point let `n` be the number of trial values, `df=n-1`, and `s`
  their sample SD; use
  `margin=t.ppf(0.975, df)*s/sqrt(n)`, with lower/upper limits
  `mean-margin` and `mean+margin`. Do not multiply the margin by 0.5. These are
  pointwise bands, not simultaneous or post-selection-adjusted intervals.
  Keep mean/SD/SEM and `online_order_repeats` in non-learning aggregate tables;
  calculate those plot intervals at render time without adding CI columns.
  The learning-curve CSV alone retains explicit pointwise CI bounds.
  Retain the earlier
  `sum_{s<=t}(cost_s-y_s)` series only under an explicit
  `cumulative_clairvoyant_excess_cost` diagnostic name; never present it as the
  primary regret or as evidence of a theoretical square-root rate. Save
  compact by-order curves in `selected_learning_curves_by_order.npz` and
  aggregate per-round mean/SD/SEM series in `selected_learning_curves.csv`,
  along with `confidence_level`, `confidence_df`, `confidence_t_critical`, and
  `cumulative_reference_regret_ci95_lower`,
  `cumulative_reference_regret_ci95_upper`,
  `average_reference_regret_ci95_lower`, and
  `average_reference_regret_ci95_upper`; do not create a large learning-curve
  JSON file.

  All tuning figures use the centralized publication contract in
  `src/llm_routing_simulation/plot_style.py`, including method/axis labels,
  legend and font sizes, figure sizes, and export DPI. Figures have no titles
  and no figure footnotes. Use the exact public method labels
  `PG-TS (Bayesian logistic)` and `Random`. The selected-cost axes are
  `x=$\ell_{01}$` and `y=Total cost`; cumulative-reference-regret axes are
  visually `Round (t)` and `Regret (excess cost)`. Save every tuning figure as
  both a vector PDF and a same-stem 400-DPI PNG. A compatible completed sweep
  can regenerate tables and both figure formats by repeating the identical
  command with `--plot-only`; no online policy is rerun.

  `--include-pgts` adds PG-TS as the fourth multiplier-tuned policy. For every
  `l01`, multiplier, and one of the same 20 paired orders, set
  `effective_prior_std = --pgts-base-prior-std * multiplier`. With base prior scale
  1 and multipliers `0.1, 0.3, 1, 3, 10`, the effective prior-standard-
  deviation candidates are those same five values. Select the lowest-mean-
  total-cost multiplier pointwise at each `l01`, with the common tie rule. At
  round 1 PG-TS performs its first 15-transition Gibbs draw. At a later
  configured estimator boundary it makes 15 complete Gibbs
  transitions using action-1 feedback through the preceding round, but only if
  new feedback arrived since its last posterior draw; otherwise it retains the
  existing sample. The sampled `theta` is frozen and reused within the epoch.
  The prior is zero-mean isotropic Gaussian; expose the base standard deviation
  and transition count as `--pgts-base-prior-std` and `--pgts-gibbs-steps`.
  `--pgts-prior-std` remains only a backward-compatible alias. Candidate
  rows record `parameter_name=prior_std`, the base value, multiplier, effective
  value, and effective `pgts_prior_std`. The random stream is determined by the
  recorded policy seed. PG-TS uses no inverse-propensity weights. This
  schedule-aware variant is explicitly not literal every-round Algorithm 1; it
  is a runtime approximation. The full design has 900 selected order rows after
  adding analytic Random and 45 selected summaries. The public figure label
  remains exactly `PG-TS (Bayesian logistic)`.
  The documented 25-round pilot remains available as an optional execution
  check; implementation work must not launch the full experiment.

  Preserve all earlier study artifacts as history. These include completed
  revision-3 pure-doubling, revision-4 Fibonacci, and revision-5
  capped-doubling gap-500/gap-100/gap-32 runs; the incomplete gap-8 attempt
  with 3,902 of 4,500 checkpoints; and the planned gap-32 seven-multiplier
  design. Never resume or plot an older fingerprinted directory with the
  revision-9 configuration. Revision 8 and all earlier outputs are historical;
  use the fresh
  `boolq-138d-cbpside-squarecb-pmside-pgts-prior-tuned-results` directory. The
  revision-9 design/fingerprint must reject revision-8 output directories.
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
