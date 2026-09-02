"""Out-of-fold supervised skylines for weak/strong disagreement prediction."""

from __future__ import annotations

import numpy as np


HGB_CAPACITY_PROFILES = (
    {
        "name": "HGB-30",
        "max_iter": 10,
        "max_leaf_nodes": 3,
        "max_leaf_values": 30,
    },
    {
        "name": "HGB-60",
        "max_iter": 20,
        "max_leaf_nodes": 3,
        "max_leaf_values": 60,
    },
    {
        "name": "HGB-100",
        "max_iter": 20,
        "max_leaf_nodes": 5,
        "max_leaf_values": 100,
    },
    {
        "name": "HGB-150",
        "max_iter": 30,
        "max_leaf_nodes": 5,
        "max_leaf_values": 150,
    },
    {
        "name": "HGB-350",
        "max_iter": 50,
        "max_leaf_nodes": 7,
        "max_leaf_values": 350,
    },
)
HGB_MIN_SAMPLES_LEAF = 50
HGB_L2_REGULARIZATION = 5.0

RESIDUAL_LEARNER_CONFIGURATION = {
    "name": "HGB residual regressor",
    "learning_rate": 0.05,
    "max_iter": 50,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 50,
    "l2_regularization": 5.0,
}

# Compact neural models for supervised diagnostic comparison. With a
# 78-dimensional context, MLP-4 has 321 trainable parameters and MLP-8 has 641.
MLP_CAPACITY_PROFILES = (
    {
        "name": "MLP-4",
        "hidden_units": 4,
        "activation": "relu",
        "solver": "adam",
        "alpha": 1.0,
        "max_iter": 1000,
        "learning_rate_init": 0.001,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 25,
    },
    {
        "name": "MLP-8",
        "hidden_units": 8,
        "activation": "relu",
        "solver": "adam",
        "alpha": 1.0,
        "max_iter": 1000,
        "learning_rate_init": 0.001,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 25,
    },
)

SUPERVISED_BENCHMARK_CONFIGURATIONS = {
    "Elastic-net logistic (out-of-fold)": {
        "penalty": "elasticnet",
        "solver": "saga",
        "C": 1.0,
        "l1_ratio": 0.5,
        "max_iter": 5000,
        "preprocessing": "same row normalization as logistic baseline",
    },
    "Extra Trees (out-of-fold)": {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 10,
        "class_weight": "balanced",
    },
    "RBF SVM (out-of-fold)": {
        "C": 1.0,
        "gamma": "scale",
        "probability_calibration": "training-fold internal Platt scaling",
        "preprocessing": "fold-local StandardScaler",
    },
    "XGBoost (out-of-fold)": {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_lambda": 5.0,
        "reg_alpha": 0.1,
    },
    "CatBoost (out-of-fold)": {
        "iterations": 300,
        "depth": 4,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
    },
}


def _expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10
) -> float:
    """Equal-width expected calibration error for a compact diagnostic."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(outcomes)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (
            (probabilities >= lower) & (probabilities <= upper)
            if index == bins - 1
            else (probabilities >= lower) & (probabilities < upper)
        )
        if np.any(selected):
            error += float(np.mean(selected)) * abs(
                float(np.mean(probabilities[selected]))
                - float(np.mean(outcomes[selected]))
            )
    return float(error)


def _routing_accuracy_at_rate(
    probabilities: np.ndarray, outcomes: np.ndarray, routing_rate: float
) -> float:
    count = int(round(routing_rate * len(outcomes)))
    routed = np.zeros(len(outcomes), dtype=bool)
    if count:
        routed[np.argsort(probabilities)[-count:]] = True
    return float(np.mean((outcomes == 0) | routed))


def random_routing_reference(
    baseline_agreement: float, *, point_count: int = 101
) -> list[dict]:
    """Return the exact expected random-routing line under the strong reference."""
    if not 0.0 <= baseline_agreement <= 1.0:
        raise ValueError("baseline_agreement must be in [0, 1]")
    if point_count < 2:
        raise ValueError("point_count must be at least two")
    rows = []
    for routing_rate in np.linspace(0.0, 1.0, point_count):
        rows.append(
            {
                "model": "Random routing (expected)",
                "fit_scope": "analytic_random_reference",
                "threshold": None,
                "routing_rate": float(routing_rate),
                "accuracy": float(
                    baseline_agreement
                    + routing_rate * (1.0 - baseline_agreement)
                ),
                "evaluation_auc": None,
                "folds": None,
                "configuration": None,
                "selected": False,
            }
        )
    return rows


def threshold_skyline(
    probabilities: np.ndarray,
    disagreements: np.ndarray,
    model_name: str,
    *,
    threshold_count: int = 101,
    fit_scope: str = "provided_probabilities",
    evaluation_auc: float | None = None,
    folds: int | None = None,
    configuration: dict | None = None,
    selected: bool = False,
) -> list[dict]:
    """Convert disagreement probabilities into a strong-routing curve."""
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    outcomes = np.asarray(disagreements, dtype=np.int64).reshape(-1)
    if scores.size == 0 or scores.size != outcomes.size:
        raise ValueError("Probabilities and outcomes must be nonempty and aligned")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Every probability must be finite and in [0, 1]")
    if np.any((outcomes != 0) & (outcomes != 1)):
        raise ValueError("Every disagreement outcome must be binary")
    if threshold_count < 2:
        raise ValueError("threshold_count must be at least two")

    thresholds = np.concatenate(
        ([1.0 + 1e-12], np.linspace(1.0, 0.0, threshold_count), [-1e-12])
    )
    agreement = outcomes == 0
    rows = []
    for threshold in thresholds:
        routed = scores >= threshold
        rows.append(
            {
                "model": model_name,
                "fit_scope": fit_scope,
                "threshold": float(threshold),
                "routing_rate": float(np.mean(routed)),
                "accuracy": float(np.mean(agreement | routed)),
                "evaluation_auc": evaluation_auc,
                "folds": folds,
                "configuration": configuration,
                "selected": selected,
            }
        )
    return rows


def fit_supervised_skylines(
    contexts: np.ndarray,
    disagreements: np.ndarray,
    *,
    seed: int = 0,
    requested_folds: int = 5,
) -> tuple[list[dict], dict, dict[str, np.ndarray]]:
    """Generate skylines and retain selected out-of-fold residual predictions."""
    try:
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        raise RuntimeError(
            "Supervised skylines require the project's scikit-learn dependency"
        ) from exc

    optional_models_skipped = []
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None
        optional_models_skipped.append("XGBoost")
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        CatBoostClassifier = None
        optional_models_skipped.append("CatBoost")

    X = np.asarray(contexts, dtype=np.float64)
    y = np.asarray(disagreements, dtype=np.int64).reshape(-1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[0] != y.size:
        raise ValueError("Contexts must be a nonempty matrix aligned with outcomes")
    if not np.all(np.isfinite(X)) or np.any((y != 0) & (y != 1)):
        raise ValueError("Contexts must be finite and outcomes binary")
    if requested_folds < 2:
        raise ValueError("requested_folds must be at least two")

    classes = np.unique(y)
    logistic_name = "Logistic (out-of-fold)"
    hgb_names = [profile["name"] + " (out-of-fold)" for profile in HGB_CAPACITY_PROFILES]
    mlp_names = [profile["name"] + " (out-of-fold)" for profile in MLP_CAPACITY_PROFILES]
    benchmark_names = [
        "Elastic-net logistic (out-of-fold)",
        "Extra Trees (out-of-fold)",
        "RBF SVM (out-of-fold)",
    ]
    if XGBClassifier is not None:
        benchmark_names.append("XGBoost (out-of-fold)")
    if CatBoostClassifier is not None:
        benchmark_names.append("CatBoost (out-of-fold)")
    if classes.size == 1:
        constant = np.full(y.size, float(classes[0]), dtype=np.float64)
        probability_sets = {logistic_name: constant}
        probability_sets.update({name: constant for name in hgb_names})
        probability_sets.update({name: constant for name in mlp_names})
        probability_sets.update({name: constant for name in benchmark_names})
        aucs = {name: None for name in probability_sets}
        fold_count = 0
    else:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        logistic_X = X / np.maximum(1.0, norms)
        class_counts = np.bincount(y, minlength=2)
        fold_count = min(requested_folds, int(np.min(class_counts)))
        if fold_count < 2:
            raise ValueError(
                "Out-of-fold skylines require at least two examples per class"
            )
        splitter = StratifiedKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=seed,
        )
        logistic_probabilities = np.empty(y.size, dtype=np.float64)
        hgb_probability_sets = {
            name: np.empty(y.size, dtype=np.float64) for name in hgb_names
        }
        mlp_probability_sets = {
            name: np.empty(y.size, dtype=np.float64) for name in mlp_names
        }
        benchmark_probability_sets = {
            name: np.empty(y.size, dtype=np.float64) for name in benchmark_names
        }
        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(X, y)
        ):
            logistic = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
                random_state=seed + fold_index,
            )
            logistic.fit(logistic_X[train_indices], y[train_indices])
            logistic_probabilities[test_indices] = logistic.predict_proba(
                logistic_X[test_indices]
            )[:, 1]

            for profile, name in zip(
                HGB_CAPACITY_PROFILES, hgb_names
            ):
                hgb = HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=0.05,
                    max_iter=profile["max_iter"],
                    max_leaf_nodes=profile["max_leaf_nodes"],
                    min_samples_leaf=HGB_MIN_SAMPLES_LEAF,
                    l2_regularization=HGB_L2_REGULARIZATION,
                    early_stopping=False,
                    random_state=seed + fold_index,
                )
                hgb.fit(X[train_indices], y[train_indices])
                hgb_probability_sets[name][test_indices] = hgb.predict_proba(
                    X[test_indices]
                )[:, 1]
            for profile, name in zip(MLP_CAPACITY_PROFILES, mlp_names):
                mlp = make_pipeline(
                    StandardScaler(),
                    MLPClassifier(
                        hidden_layer_sizes=(profile["hidden_units"],),
                        activation=profile["activation"],
                        solver=profile["solver"],
                        alpha=profile["alpha"],
                        max_iter=profile["max_iter"],
                        learning_rate_init=profile["learning_rate_init"],
                        early_stopping=profile["early_stopping"],
                        validation_fraction=profile["validation_fraction"],
                        n_iter_no_change=profile["n_iter_no_change"],
                        random_state=seed + fold_index,
                    ),
                )
                mlp.fit(X[train_indices], y[train_indices])
                mlp_probability_sets[name][test_indices] = mlp.predict_proba(
                    X[test_indices]
                )[:, 1]

            elastic_config = SUPERVISED_BENCHMARK_CONFIGURATIONS[
                "Elastic-net logistic (out-of-fold)"
            ]
            elastic = LogisticRegression(
                penalty=elastic_config["penalty"],
                solver=elastic_config["solver"],
                C=elastic_config["C"],
                l1_ratio=elastic_config["l1_ratio"],
                max_iter=elastic_config["max_iter"],
                random_state=seed + fold_index,
            )
            elastic.fit(logistic_X[train_indices], y[train_indices])
            benchmark_probability_sets[
                "Elastic-net logistic (out-of-fold)"
            ][test_indices] = elastic.predict_proba(logistic_X[test_indices])[:, 1]

            trees_config = SUPERVISED_BENCHMARK_CONFIGURATIONS[
                "Extra Trees (out-of-fold)"
            ]
            extra_trees = ExtraTreesClassifier(
                n_estimators=trees_config["n_estimators"],
                max_features=trees_config["max_features"],
                min_samples_leaf=trees_config["min_samples_leaf"],
                class_weight=trees_config["class_weight"],
                random_state=seed + fold_index,
                n_jobs=1,
            )
            extra_trees.fit(X[train_indices], y[train_indices])
            benchmark_probability_sets["Extra Trees (out-of-fold)"][
                test_indices
            ] = extra_trees.predict_proba(X[test_indices])[:, 1]

            svm_config = SUPERVISED_BENCHMARK_CONFIGURATIONS[
                "RBF SVM (out-of-fold)"
            ]
            rbf_svm = make_pipeline(
                StandardScaler(),
                SVC(
                    C=svm_config["C"],
                    gamma=svm_config["gamma"],
                    probability=True,
                    random_state=seed + fold_index,
                ),
            )
            rbf_svm.fit(X[train_indices], y[train_indices])
            benchmark_probability_sets["RBF SVM (out-of-fold)"][
                test_indices
            ] = rbf_svm.predict_proba(X[test_indices])[:, 1]

            if XGBClassifier is not None:
                xgb_config = SUPERVISED_BENCHMARK_CONFIGURATIONS[
                    "XGBoost (out-of-fold)"
                ]
                xgboost = XGBClassifier(
                    **xgb_config,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=seed + fold_index,
                    n_jobs=1,
                    verbosity=0,
                )
                xgboost.fit(X[train_indices], y[train_indices])
                benchmark_probability_sets["XGBoost (out-of-fold)"][
                    test_indices
                ] = xgboost.predict_proba(X[test_indices])[:, 1]

            if CatBoostClassifier is not None:
                cat_config = SUPERVISED_BENCHMARK_CONFIGURATIONS[
                    "CatBoost (out-of-fold)"
                ]
                catboost = CatBoostClassifier(
                    **cat_config,
                    loss_function="Logloss",
                    random_seed=seed + fold_index,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=1,
                )
                catboost.fit(X[train_indices], y[train_indices])
                benchmark_probability_sets["CatBoost (out-of-fold)"][
                    test_indices
                ] = catboost.predict_proba(X[test_indices])[:, 1]
        probability_sets = {logistic_name: logistic_probabilities}
        probability_sets.update(hgb_probability_sets)
        probability_sets.update(mlp_probability_sets)
        probability_sets.update(benchmark_probability_sets)
        aucs = {
            name: float(roc_auc_score(y, probabilities))
            for name, probabilities in probability_sets.items()
        }

    if classes.size == 1:
        probability_metrics = {
            name: {
                "roc_auc": None,
                "log_loss": None,
                "brier_score": None,
                "expected_calibration_error_10_bins": None,
                "routing_accuracy_at_50_percent": None,
            }
            for name in probability_sets
        }
    else:
        probability_metrics = {
            name: {
                "roc_auc": aucs[name],
                "log_loss": float(
                    log_loss(y, np.clip(probabilities, 1e-7, 1.0 - 1e-7))
                ),
                "brier_score": float(brier_score_loss(y, probabilities)),
                "expected_calibration_error_10_bins": _expected_calibration_error(
                    probabilities, y
                ),
                "routing_accuracy_at_50_percent": _routing_accuracy_at_rate(
                    probabilities, y, 0.5
                ),
            }
            for name, probabilities in probability_sets.items()
        }

    selectable_aucs = {
        name: aucs[name] for name in hgb_names if aucs[name] is not None
    }
    selected_hgb = (
        max(selectable_aucs, key=selectable_aucs.get)
        if selectable_aucs
        else hgb_names[0]
    )
    selectable_mlp_aucs = {
        name: aucs[name] for name in mlp_names if aucs[name] is not None
    }
    selected_mlp = (
        max(selectable_mlp_aucs, key=selectable_mlp_aucs.get)
        if selectable_mlp_aucs
        else mlp_names[0]
    )
    hgb_configurations = {
        profile["name"] + " (out-of-fold)": {
            **profile,
            "min_samples_leaf": HGB_MIN_SAMPLES_LEAF,
            "l2_regularization": HGB_L2_REGULARIZATION,
            "learning_rate": 0.05,
        }
        for profile in HGB_CAPACITY_PROFILES
    }
    mlp_configurations = {
        profile["name"] + " (out-of-fold)": {
            **profile,
            "input_dimension": int(X.shape[1]),
            "trainable_parameters": int(
                X.shape[1] * profile["hidden_units"]
                + profile["hidden_units"]
                + profile["hidden_units"]
                + 1
            ),
            "preprocessing": "fold-local StandardScaler",
        }
        for profile in MLP_CAPACITY_PROFILES
    }
    benchmark_configurations = {
        name: SUPERVISED_BENCHMARK_CONFIGURATIONS[name]
        for name in benchmark_names
    }
    configurations = {
        **hgb_configurations,
        **mlp_configurations,
        **benchmark_configurations,
    }
    baseline_metrics = probability_metrics[logistic_name]
    comparison = []
    for name, metrics in probability_metrics.items():
        comparison.append(
            {
                "model": name,
                **metrics,
                "delta_auc_vs_logistic": (
                    None
                    if metrics["roc_auc"] is None
                    else float(metrics["roc_auc"] - baseline_metrics["roc_auc"])
                ),
                "delta_log_loss_vs_logistic": (
                    None
                    if metrics["log_loss"] is None
                    else float(metrics["log_loss"] - baseline_metrics["log_loss"])
                ),
                "delta_brier_vs_logistic": (
                    None
                    if metrics["brier_score"] is None
                    else float(
                        metrics["brier_score"] - baseline_metrics["brier_score"]
                    )
                ),
                "beats_logistic_on_auc_logloss_brier": bool(
                    metrics["roc_auc"] is not None
                    and metrics["roc_auc"] > baseline_metrics["roc_auc"]
                    and metrics["log_loss"] < baseline_metrics["log_loss"]
                    and metrics["brier_score"] < baseline_metrics["brier_score"]
                ),
                "configuration": configurations.get(name),
            }
        )
    comparison.sort(
        key=lambda row: (
            row["roc_auc"] is not None,
            row["roc_auc"] if row["roc_auc"] is not None else -np.inf,
        ),
        reverse=True,
    )
    selected_overall = comparison[0]["model"]
    rows = []
    for name, probabilities in probability_sets.items():
        rows.extend(
            threshold_skyline(
                probabilities,
                y,
                name,
                fit_scope="stratified_out_of_fold",
                evaluation_auc=aucs[name],
                folds=fold_count,
                configuration=configurations.get(name),
                selected=name in {selected_hgb, selected_mlp, selected_overall},
            )
        )
    summary = {
        "fit_scope": "stratified_out_of_fold",
        "reference": "strong_model_answer",
        "target": "weak_strong_disagreement",
        "requested_folds": int(requested_folds),
        "folds": int(fold_count),
        "examples": int(y.size),
        "context_dimension": int(X.shape[1]),
        "disagreement_rate": float(np.mean(y)),
        "out_of_fold_auc": aucs,
        "hgb_capacity_profiles": hgb_configurations,
        "hgb_selection_metric": "out_of_fold_auc",
        "hgb_selection_scope": "exploratory_same_oof_predictions",
        "selected_hgb_model": selected_hgb,
        "mlp_capacity_profiles": mlp_configurations,
        "mlp_selection_metric": "out_of_fold_auc",
        "mlp_selection_scope": "exploratory_same_oof_predictions",
        "selected_mlp_model": selected_mlp,
        "supervised_benchmark_configurations": benchmark_configurations,
        "optional_models_skipped": optional_models_skipped,
        "model_comparison": comparison,
        "selected_overall_model_by_auc": selected_overall,
        "selection_warning": (
            "Exploratory comparison on shared out-of-fold predictions; use repeated "
            "or nested validation before final model selection."
        ),
        "residual_models": [logistic_name, selected_hgb, selected_mlp],
    }
    residual_probability_sets = {
        logistic_name: probability_sets[logistic_name],
        selected_hgb: probability_sets[selected_hgb],
        selected_mlp: probability_sets[selected_mlp],
    }
    return rows, summary, residual_probability_sets


def binary_residual_diagnostics(
    probability_sets: dict[str, np.ndarray],
    outcomes: np.ndarray,
    *,
    bin_count: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Create pointwise and equal-frequency-bin residual diagnostics."""
    y = np.asarray(outcomes, dtype=np.int64).reshape(-1)
    if y.size == 0 or np.any((y != 0) & (y != 1)):
        raise ValueError("Residual outcomes must be a nonempty binary vector")
    if bin_count < 2:
        raise ValueError("bin_count must be at least two")
    point_rows: list[dict] = []
    bin_rows: list[dict] = []
    for model, raw_probabilities in probability_sets.items():
        probabilities = np.asarray(raw_probabilities, dtype=np.float64).reshape(-1)
        if probabilities.size != y.size:
            raise ValueError("Every residual prediction must align with outcomes")
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ValueError("Residual probabilities must be finite and in [0, 1]")

        clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        raw_residual = y - probabilities
        pearson_residual = raw_residual / np.sqrt(clipped * (1.0 - clipped))
        deviance_residual = np.where(
            y == 1,
            np.sqrt(-2.0 * np.log(clipped)),
            -np.sqrt(-2.0 * np.log(1.0 - clipped)),
        )
        assignments = np.empty(y.size, dtype=np.int64)
        ordered_indices = np.argsort(probabilities, kind="stable")
        chunks = np.array_split(ordered_indices, min(bin_count, y.size))
        for bin_index, indices in enumerate(chunks, start=1):
            assignments[indices] = bin_index
            residuals = raw_residual[indices]
            standard_error = (
                float(np.std(residuals, ddof=1) / np.sqrt(indices.size))
                if indices.size > 1
                else 0.0
            )
            bin_rows.append(
                {
                    "model": model,
                    "bin": bin_index,
                    "count": int(indices.size),
                    "mean_predicted_probability": float(
                        np.mean(probabilities[indices])
                    ),
                    "observed_disagreement_rate": float(np.mean(y[indices])),
                    "mean_raw_residual": float(np.mean(residuals)),
                    "standard_error": standard_error,
                    "ci95_lower": float(np.mean(residuals) - 1.96 * standard_error),
                    "ci95_upper": float(np.mean(residuals) + 1.96 * standard_error),
                }
            )
        for index in range(y.size):
            point_rows.append(
                {
                    "model": model,
                    "example_index": index,
                    "observed_disagreement": int(y[index]),
                    "predicted_disagreement_probability": float(
                        probabilities[index]
                    ),
                    "raw_residual": float(raw_residual[index]),
                    "pearson_residual": float(pearson_residual[index]),
                    "deviance_residual": float(deviance_residual[index]),
                    "quantile_bin": int(assignments[index]),
                }
            )
    return point_rows, bin_rows


def plot_binary_residuals(
    output_path,
    point_rows: list[dict],
    bin_rows: list[dict],
) -> None:
    """Plot binary residual bands and binned calibration residuals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(row["model"] for row in point_rows))
    if not models:
        raise ValueError("Residual plot requires at least one model")
    figure, axes = plt.subplots(
        1,
        len(models),
        figsize=(5.2 * len(models), 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, models):
        selected_points = [row for row in point_rows if row["model"] == model]
        selected_bins = [row for row in bin_rows if row["model"] == model]
        axis.scatter(
            [row["predicted_disagreement_probability"] for row in selected_points],
            [row["raw_residual"] for row in selected_points],
            s=8,
            alpha=0.10,
            color="#4c78a8",
            linewidths=0,
            label="Individual binary residuals",
        )
        axis.errorbar(
            [row["mean_predicted_probability"] for row in selected_bins],
            [row["mean_raw_residual"] for row in selected_bins],
            yerr=[1.96 * row["standard_error"] for row in selected_bins],
            color="#e45756",
            marker="o",
            linewidth=2,
            capsize=3,
            label="Bin mean and 95% interval",
        )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(model.replace(" (out-of-fold)", ""))
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-1.05, 1.05)
        axis.set_xlabel("Out-of-fold predicted disagreement probability")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"Raw residual $y - \hat{p}$")
    axes[0].legend(fontsize=8, loc="upper right")
    figure.suptitle("Out-of-fold binary residual diagnostics")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def cross_fitted_residual_predictability(
    contexts: np.ndarray,
    outcomes: np.ndarray,
    *,
    residual_contexts: np.ndarray | None = None,
    base_context_label: str = "original context",
    residual_context_label: str | None = None,
    seed: int = 0,
    requested_folds: int = 5,
    permutation_repeats: int = 100,
    bin_count: int = 10,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Test whether nonlinear structure remains after the logistic model.

    Each outer fold is absent from both the logistic fit and the residual-
    learner fit used to predict that fold. Logistic residual targets for an
    outer training fold are themselves generated by an inner cross-fit. The
    permutation test repeats the residual fit after shuffling only those inner
    training residuals, while keeping the held-out evaluation residuals fixed.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
    except ImportError as exc:
        raise RuntimeError(
            "Residual predictability diagnostics require scikit-learn"
        ) from exc

    X = np.asarray(contexts, dtype=np.float64)
    Z = X if residual_contexts is None else np.asarray(
        residual_contexts, dtype=np.float64
    )
    y = np.asarray(outcomes, dtype=np.int64).reshape(-1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[0] != y.size:
        raise ValueError("Contexts must be a nonempty matrix aligned with outcomes")
    if not np.all(np.isfinite(X)) or np.any((y != 0) & (y != 1)):
        raise ValueError("Contexts must be finite and outcomes binary")
    if Z.ndim != 2 or Z.shape[0] != y.size or not np.all(np.isfinite(Z)):
        raise ValueError(
            "Residual contexts must be a finite matrix aligned with outcomes"
        )
    if requested_folds < 2:
        raise ValueError("requested_folds must be at least two")
    if permutation_repeats < 0:
        raise ValueError("permutation_repeats must be nonnegative")
    if bin_count < 2:
        raise ValueError("bin_count must be at least two")

    class_counts = np.bincount(y, minlength=2)
    fold_count = min(requested_folds, int(np.min(class_counts)))
    if fold_count < 2:
        raise ValueError(
            "Residual predictability requires at least two examples per class"
        )

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    logistic_X = X / np.maximum(1.0, norms)
    outer = StratifiedKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=seed,
    )
    outer_splits = list(outer.split(X, y))
    logistic_probabilities = np.empty(y.size, dtype=np.float64)
    predicted_residuals = np.empty(y.size, dtype=np.float64)
    fold_assignments = np.empty(y.size, dtype=np.int64)
    residual_training_sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def logistic_model(random_state: int):
        return LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=random_state,
        )

    def residual_model(random_state: int, training_size: int):
        minimum_leaf = min(
            RESIDUAL_LEARNER_CONFIGURATION["min_samples_leaf"],
            max(2, training_size // 5),
        )
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=RESIDUAL_LEARNER_CONFIGURATION["learning_rate"],
            max_iter=RESIDUAL_LEARNER_CONFIGURATION["max_iter"],
            max_leaf_nodes=RESIDUAL_LEARNER_CONFIGURATION["max_leaf_nodes"],
            min_samples_leaf=minimum_leaf,
            l2_regularization=RESIDUAL_LEARNER_CONFIGURATION[
                "l2_regularization"
            ],
            early_stopping=False,
            random_state=random_state,
        )

    for fold_index, (train_indices, test_indices) in enumerate(outer_splits):
        base = logistic_model(seed + fold_index)
        base.fit(logistic_X[train_indices], y[train_indices])
        logistic_probabilities[test_indices] = base.predict_proba(
            logistic_X[test_indices]
        )[:, 1]

        train_y = y[train_indices]
        inner_class_counts = np.bincount(train_y, minlength=2)
        inner_fold_count = min(requested_folds, int(np.min(inner_class_counts)))
        if inner_fold_count < 2:
            raise ValueError(
                "Nested residual cross-fitting requires two training examples "
                "per class in every outer fold"
            )
        inner = StratifiedKFold(
            n_splits=inner_fold_count,
            shuffle=True,
            random_state=seed + 10_000 + fold_index,
        )
        inner_probabilities = np.empty(train_indices.size, dtype=np.float64)
        for inner_index, (inner_train, inner_test) in enumerate(
            inner.split(logistic_X[train_indices], train_y)
        ):
            inner_base = logistic_model(
                seed + 20_000 + fold_index * requested_folds + inner_index
            )
            inner_base.fit(
                logistic_X[train_indices][inner_train], train_y[inner_train]
            )
            inner_probabilities[inner_test] = inner_base.predict_proba(
                logistic_X[train_indices][inner_test]
            )[:, 1]
        training_residuals = train_y.astype(np.float64) - inner_probabilities

        learner = residual_model(seed + 30_000 + fold_index, train_indices.size)
        learner.fit(Z[train_indices], training_residuals)
        predicted_residuals[test_indices] = learner.predict(Z[test_indices])
        fold_assignments[test_indices] = fold_index + 1
        residual_training_sets.append(
            (train_indices, test_indices, training_residuals)
        )

    observed_residuals = y.astype(np.float64) - logistic_probabilities
    zero_squared_errors = np.square(observed_residuals)
    learner_squared_errors = np.square(observed_residuals - predicted_residuals)
    zero_mse = float(np.mean(zero_squared_errors))
    learner_mse = float(np.mean(learner_squared_errors))
    observed_improvement = zero_mse - learner_mse
    relative_improvement = (
        observed_improvement / zero_mse if zero_mse > 0.0 else 0.0
    )
    correlation = (
        float(np.corrcoef(observed_residuals, predicted_residuals)[0, 1])
        if np.std(observed_residuals) > 0.0
        and np.std(predicted_residuals) > 0.0
        else 0.0
    )

    point_rows = [
        {
            "example_index": index,
            "outer_fold": int(fold_assignments[index]),
            "observed_disagreement": int(y[index]),
            "logistic_probability": float(logistic_probabilities[index]),
            "observed_logistic_residual": float(observed_residuals[index]),
            "predicted_logistic_residual": float(predicted_residuals[index]),
            "zero_baseline_squared_error": float(zero_squared_errors[index]),
            "residual_learner_squared_error": float(learner_squared_errors[index]),
            "squared_error_improvement": float(
                zero_squared_errors[index] - learner_squared_errors[index]
            ),
        }
        for index in range(y.size)
    ]

    ordered_indices = np.argsort(predicted_residuals, kind="stable")
    chunks = np.array_split(ordered_indices, min(bin_count, y.size))
    bin_rows = []
    for bin_index, indices in enumerate(chunks, start=1):
        selected_residuals = observed_residuals[indices]
        standard_error = (
            float(np.std(selected_residuals, ddof=1) / np.sqrt(indices.size))
            if indices.size > 1
            else 0.0
        )
        bin_rows.append(
            {
                "bin": bin_index,
                "count": int(indices.size),
                "mean_predicted_residual": float(
                    np.mean(predicted_residuals[indices])
                ),
                "mean_observed_residual": float(np.mean(selected_residuals)),
                "standard_error": standard_error,
                "ci95_lower": float(
                    np.mean(selected_residuals) - 1.96 * standard_error
                ),
                "ci95_upper": float(
                    np.mean(selected_residuals) + 1.96 * standard_error
                ),
            }
        )

    rng = np.random.default_rng(seed + 40_000)
    permutation_rows = []
    for repeat in range(permutation_repeats):
        null_predictions = np.empty(y.size, dtype=np.float64)
        for fold_index, (
            train_indices,
            test_indices,
            training_residuals,
        ) in enumerate(residual_training_sets):
            shuffled_residuals = rng.permutation(training_residuals)
            null_learner = residual_model(
                seed + 50_000 + repeat * fold_count + fold_index,
                train_indices.size,
            )
            null_learner.fit(Z[train_indices], shuffled_residuals)
            null_predictions[test_indices] = null_learner.predict(Z[test_indices])
        null_mse = float(np.mean(np.square(observed_residuals - null_predictions)))
        null_improvement = zero_mse - null_mse
        permutation_rows.append(
            {
                "permutation": repeat + 1,
                "zero_baseline_mse": zero_mse,
                "permuted_residual_learner_mse": null_mse,
                "mse_improvement": null_improvement,
                "relative_mse_improvement": (
                    null_improvement / zero_mse if zero_mse > 0.0 else 0.0
                ),
            }
        )

    if permutation_rows:
        null_improvements = np.asarray(
            [row["mse_improvement"] for row in permutation_rows],
            dtype=np.float64,
        )
        permutation_p_value = float(
            (1 + np.count_nonzero(null_improvements >= observed_improvement))
            / (permutation_repeats + 1)
        )
        null_95_percentile = float(np.quantile(null_improvements, 0.95))
    else:
        permutation_p_value = None
        null_95_percentile = None

    residual_label = residual_context_label or base_context_label
    summary = {
        "question": (
            f"Can {residual_label} predict held-out logistic residuals from "
            f"{base_context_label} better than the zero-residual baseline?"
        ),
        "method": "nested cross-fitted HGB residual regression",
        "examples": int(y.size),
        "base_context": base_context_label,
        "context_dimension": int(X.shape[1]),
        "base_context_dimension": int(X.shape[1]),
        "residual_context": residual_label,
        "residual_context_dimension": int(Z.shape[1]),
        "outer_folds": int(fold_count),
        "requested_inner_folds": int(requested_folds),
        "residual_definition": "observed disagreement minus logistic probability",
        "zero_baseline_mse": zero_mse,
        "residual_learner_mse": learner_mse,
        "mse_improvement": observed_improvement,
        "relative_mse_improvement": relative_improvement,
        "residual_correlation": correlation,
        "mean_observed_residual": float(np.mean(observed_residuals)),
        "predicted_residual_standard_deviation": float(
            np.std(predicted_residuals)
        ),
        "permutation_repeats": int(permutation_repeats),
        "permutation_p_value_one_sided": permutation_p_value,
        "null_mse_improvement_95_percentile": null_95_percentile,
        "significant_at_0.05": bool(
            permutation_p_value is not None and permutation_p_value <= 0.05
        ),
        "configuration": RESIDUAL_LEARNER_CONFIGURATION,
        "interpretation": (
            "Positive relative improvement with a small permutation p-value "
            "indicates predictable structure left in the logistic residuals."
        ),
    }
    return point_rows, bin_rows, permutation_rows, summary


def plot_residual_predictability(
    output_path,
    point_rows: list[dict],
    bin_rows: list[dict],
    permutation_rows: list[dict],
    summary: dict,
) -> None:
    """Plot held-out residual prediction and its permutation reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not point_rows or not bin_rows:
        raise ValueError("Residual predictability plot requires diagnostic rows")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axes[0].scatter(
        [row["predicted_logistic_residual"] for row in point_rows],
        [row["observed_logistic_residual"] for row in point_rows],
        s=8,
        alpha=0.1,
        color="#4c78a8",
        linewidths=0,
    )
    axes[0].errorbar(
        [row["mean_predicted_residual"] for row in bin_rows],
        [row["mean_observed_residual"] for row in bin_rows],
        yerr=[1.96 * row["standard_error"] for row in bin_rows],
        marker="o",
        linewidth=2,
        capsize=3,
        color="#e45756",
        label="Bin mean and 95% interval",
    )
    bounds = [
        row[key]
        for row in point_rows
        for key in ("predicted_logistic_residual", "observed_logistic_residual")
    ]
    lower, upper = min(bounds), max(bounds)
    axes[0].plot([lower, upper], [lower, upper], "k--", linewidth=1, label="Ideal")
    axes[0].axhline(0.0, color="grey", linewidth=1)
    axes[0].set_xlabel("Cross-fitted predicted logistic residual")
    axes[0].set_ylabel("Held-out observed logistic residual")
    axes[0].set_title("Residual signal in original context")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    observed = summary["mse_improvement"]
    if permutation_rows:
        axes[1].hist(
            [row["mse_improvement"] for row in permutation_rows],
            bins=min(20, max(5, len(permutation_rows) // 5)),
            color="#72b7b2",
            alpha=0.8,
            label="Shuffled training residuals",
        )
        axes[1].axvline(
            observed,
            color="#e45756",
            linewidth=2,
            label="Observed learner",
        )
        axes[1].set_title(
            f"Permutation reference (p={summary['permutation_p_value_one_sided']:.3g})"
        )
    else:
        axes[1].axvline(
            observed,
            color="#e45756",
            linewidth=2,
            label="Observed learner",
        )
        axes[1].text(
            0.5,
            0.5,
            "Permutation test disabled",
            transform=axes[1].transAxes,
            ha="center",
        )
        axes[1].set_title("Permutation reference")
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("MSE improvement over zero-residual baseline")
    axes[1].set_ylabel("Permutation count")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    figure.suptitle("Nested cross-fitted residual predictability diagnostic")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
