# Offline LLM-routing algorithm simulation

This project reads the one-time cache produced by `Data_collection_LLM_routing`.
It does not load an LLM, contact Hugging Face, need an `HF_TOKEN`, or require a
GPU. The cached strong answer is retained as the evaluation reference, while
the environment still hides that answer and the disagreement outcome whenever
an online player chooses action 0.

## Set up on the laptop

Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

Copy the downloaded `llm-routing-cache.zip` into this folder. Run every current
experiment with:

```powershell
simulate-llm-routing --cache llm-routing-cache.zip --experiment all
```

For a quick check, use a prefix:

```powershell
simulate-llm-routing --cache llm-routing-cache.zip --experiment skyline --limit 100
```

The useful choices are:

- `--experiment skyline`: out-of-fold logistic and five HGB capacity skylines
  (HGB-30 through HGB-350), plus compact MLP-4 and MLP-8 neural skylines,
  elastic-net logistic, Extra Trees, calibrated RBF SVM, and—when the
  `benchmark` extra is installed—XGBoost and CatBoost. It also includes the
  analytic random-routing reference.
- `--experiment online`: ETC, CBPSide, fixed-gamma-32 IGW, and random routing
  matched to ETC traffic.
- `--experiment all`: both groups (the default).

Results go to `simulation-results/`, including CSV/JSON tables, the full online
trajectory, `routing_comparison.png`, and a portable `simulation-results.zip`.
Each execution refits all estimators from the cached samples; no LLM generation
is repeated. You can run new seeds, loss values, thresholds and algorithms many
times against the same cache.

Skyline runs also create `supervised_residual_plots.png`. Its three panels use
strictly out-of-fold predictions from linear logistic, the HGB capacity selected
by out-of-fold AUC, and the MLP capacity selected by out-of-fold AUC. Individual
binary residuals use `y - predicted_probability`; red points show equal-frequency
bin means with 95% intervals. A model is better specified when those red points
remain near the zero line without systematic curvature. The underlying raw,
Pearson, and deviance residuals are saved in `supervised_residuals.csv`, and the
bin summaries are saved in `supervised_residual_bins.csv`.

## Current defaults

- Context: 14 weak-generation uncertainty features plus the collected fixed PCA
  projection of five Qwen hidden layers.
- Loss values: `l01 = 1.82, 2.22, 2.67, 3.33`, `l11 = 1`.
- ETC: 100 initial tastes, then a frozen XGBoost estimator.
- CBPSide: logistic regression with the same class-balanced bootstrap as IGW:
  force strong routing until 10 agreements and 10 disagreements are observed,
  capped at 50 tastes. Its additional fixed minimum is zero by default.
- IGW: online-refitted XGBoost model, `mu = 2`, fixed `gamma = 32`, class
  bootstrap 10/10 with a
  maximum of 50 bootstrap tastes, inverse-propensity weights capped at 10.
- Accuracy: agreement with the strong-model answer, not ARC ground truth.
- Neural skyline diagnostics: MLP-4 has 321 trainable parameters and MLP-8 has
  641 for the default 78-dimensional context. They are not used by IGW unless
  a later out-of-fold result justifies implementing weighted online training.
- Supervised model comparison reports ROC AUC, log loss, Brier score,
  ten-bin calibration error, and routing accuracy at 50% traffic. Install the
  optional tree libraries with `python -m pip install -e ".[benchmark,test]"`.
- XGBoost is connected to both ETC and IGW after winning the supervised
  comparison; CBPSide remains logistic. CatBoost, Extra Trees, SVM,
  elastic-net, and the MLPs remain supervised diagnostics only.

Pass alternatives on the command line, for example:

```powershell
simulate-llm-routing --cache llm-routing-cache.zip --l01-values 1 2 4 8 --seed 7
```

To rebuild a lower-dimensional context using the fixed PCA axes saved in the
cache, add `--pca-components 20` (or another available positive value).

## Where to edit

- `src/llm_routing_simulation/algorithm.py`: ETC, CBPSide and IGW players.
- `src/llm_routing_simulation/skyline.py`: supervised logistic/HGB skylines.
- `src/llm_routing_simulation/environment.py`: partial-feedback rules.
- `src/llm_routing_simulation/run.py`: experiment parameters, result tables and
  plots.
- `src/llm_routing_simulation/cache.py`: cache validation and context selection.

Run tests with `python -m pytest`.

## Separate nonlinear synthetic sanity check

The synthetic workflow is isolated from the LLM-cache workflow. It generates
two-dimensional contexts independently from `Uniform(-3, 3)` and samples

```text
P(Y = 1 | X) = sigmoid(x1 * x2),
```

where `Y = 1` represents weak/strong disagreement. Because the sign pattern is
quadrant-dependent, a linear logistic model is deliberately misspecified while
a small tree ensemble can represent the relationship.

Run the complete synthetic experiment with:

```powershell
python -m pip install -e ".[test]"
simulate-synthetic-nonlinear
```

The install command only needs to be repeated once after adding this new CLI
entry point to an existing virtual environment.

Its defaults generate 3,000 supervised training samples and an independent
2,000-sample test/online stream. The left plot compares held-out linear
logistic, small HGB, the known Bayes probability, and random routing. The right
plot compares:

- ETC with a small HGB fitted once after 500 initial tastes;
- IGW with a small HGB refitted from revealed feedback, including its
  inverse-propensity weights;
- CBPSide with the existing linear logistic model;
- random routing matched to ETC's routing rate.

Only ETC has forced initial tasting. CBPSide and IGW both start with zero fixed
tastes and no class-balanced bootstrap; their normal decision rules determine
whether feedback is revealed from the first round. Results are written to
`synthetic-nonlinear-results/`, including the generated data, CSV/JSON tables,
the complete online trajectories, the combined plot, and
`synthetic-results.zip`.

The online plot uses 10 default `l01` values corresponding to evenly spaced
decision thresholds `alpha = 0.95, 0.85, ..., 0.05`. With the default `l11 = 1`,
the values range from approximately `l01 = 1.0526` to `20`. Spacing the
thresholds rather than the losses gives substantially more uniform coverage of
the routing-rate axis.

For a quick smoke test:

```powershell
simulate-synthetic-nonlinear --train-samples 1000 --online-samples 300 --etc-tastes 50 --l01-values 1.82
```

This command uses only local CPU computation. To return to the real cached LLM
experiment, continue using `simulate-llm-routing`; neither workflow changes the
other's data or defaults.
