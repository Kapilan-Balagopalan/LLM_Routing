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

The residual branch additionally asks whether the original context contains
nonlinear signal left over after logistic regression. A nested cross-fitted HGB
regressor predicts logistic residuals without seeing the evaluated outer fold.
Its held-out mean-squared error is compared with predicting a zero residual, and
100 shuffled-training-residual refits provide a one-sided permutation reference.
Results are saved in `residual_predictability_summary.json`, three accompanying
CSV files, and `residual_predictability.png`. Use
`--residual-permutations 0` for a fast diagnostic without the permutation test,
or another nonnegative count to control its precision and runtime.

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
