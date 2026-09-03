"""Holdout logistic-residual analyses for manifest-defined cache contexts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from llm_routing_simulation.cache import RoutingCache, load_cache
from llm_routing_simulation.skyline import binary_residual_diagnostics


BINNED_RESIDUAL_REFERENCE = (
    "https://www2.stat.duke.edu/courses/Fall19/sta210.001/slides/"
    "lec-slides/18-logistic-pt3.html#12"
)
LOGISTIC_CONFIGURATION = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 5000,
    "preprocessing": "training-split StandardScaler",
}
RESIDUAL_INNER_FOLDS = 5
RESIDUAL_FOREST_CONFIGURATION = {
    "estimator": "ExtraTreesRegressor",
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 20,
    "max_features": 0.75,
    "criterion": "squared_error",
}
LINEAR_RESIDUAL_CONFIGURATION = {
    "estimator": "Ridge",
    "alpha": 1.0,
    "preprocessing": "training-split StandardScaler",
}
SYNTHETIC_FOREST_CONFIGURATION = {
    "estimator": "RandomForestRegressor",
    "n_estimators": 50,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "max_features": 0.75,
    "prompt_feature_count": 12,
    "probability_logit_scale": 2.5,
    "probability_clip": [0.02, 0.98],
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare validation-set binned logistic residuals for the cached "
            "uncertainty+hidden context, prompt context, and synthetic "
            "random-forest labels."
        )
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("holdout-residual-results")
    )
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--residual-bin-count", type=int, default=10)
    return parser


def _fit_logistic(
    train_contexts: np.ndarray,
    train_outcomes: np.ndarray,
    validation_contexts: np.ndarray,
):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=LOGISTIC_CONFIGURATION["C"],
            solver=LOGISTIC_CONFIGURATION["solver"],
            max_iter=LOGISTIC_CONFIGURATION["max_iter"],
        ),
    )
    model.fit(train_contexts, train_outcomes)
    probabilities = model.predict_proba(validation_contexts)[:, 1]
    return model, probabilities


def _model_metrics(outcomes: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities >= 0.5
    auc = (
        float(roc_auc_score(outcomes, probabilities))
        if np.unique(outcomes).size == 2
        else None
    )
    return {
        "roc_auc": auc,
        "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(outcomes, probabilities)),
        "accuracy_at_0_5": float(np.mean(predictions == outcomes)),
        "observed_positive_rate": float(np.mean(outcomes)),
        "mean_predicted_probability": float(np.mean(probabilities)),
        "mean_raw_residual": float(np.mean(outcomes - probabilities)),
    }


def _diagnose_scenario(
    scenario: str,
    train_contexts: np.ndarray,
    validation_contexts: np.ndarray,
    train_outcomes: np.ndarray,
    validation_outcomes: np.ndarray,
    validation_positions: np.ndarray,
    example_ids: list[str],
    bin_count: int,
    residual_bin_count: int,
    label_source: str,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    _, probabilities = _fit_logistic(
        train_contexts,
        train_outcomes,
        validation_contexts,
    )
    points, bins = binary_residual_diagnostics(
        {scenario: probabilities}, validation_outcomes, bin_count=bin_count
    )
    for row in points:
        local_index = int(row["example_index"])
        eligible_position = int(validation_positions[local_index])
        row["scenario"] = scenario
        row["eligible_position"] = eligible_position
        row["example_id"] = example_ids[eligible_position]
        row["label_source"] = label_source
    for row in bins:
        row["scenario"] = scenario
        row["label_source"] = label_source
    metrics = _model_metrics(validation_outcomes, probabilities)
    residual_points, residual_bins, residual_summary = (
        _heldout_residual_predictability(
            train_contexts,
            validation_contexts,
            train_outcomes,
            validation_outcomes,
            probabilities,
            bin_count=residual_bin_count,
            seed=seed,
        )
    )
    for row in residual_points:
        local_index = int(row["example_index"])
        eligible_position = int(validation_positions[local_index])
        row["scenario"] = scenario
        row["eligible_position"] = eligible_position
        row["example_id"] = example_ids[eligible_position]
        row["label_source"] = label_source
    for row in residual_bins:
        row["scenario"] = scenario
        row["label_source"] = label_source
    metrics.update(
        {
            "scenario": scenario,
            "label_source": label_source,
            "context_dimension": int(train_contexts.shape[1]),
            "train_examples": int(train_contexts.shape[0]),
            "validation_examples": int(validation_contexts.shape[0]),
            "binned_residual_count": bin_count,
            "largest_absolute_bin_mean_residual": float(
                max(abs(row["mean_raw_residual"]) for row in bins)
            ),
            "bins_with_95_percent_interval_excluding_zero": int(
                sum(
                    row["ci95_lower"] > 0.0 or row["ci95_upper"] < 0.0
                    for row in bins
                )
            ),
            "logistic_configuration": LOGISTIC_CONFIGURATION,
            "residual_predictability": residual_summary,
        }
    )
    return points, bins, residual_points, residual_bins, metrics


def _training_oof_logistic_probabilities(
    contexts: np.ndarray,
    outcomes: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Build honest logistic residual targets inside the training split."""
    class_counts = np.bincount(outcomes, minlength=2)
    fold_count = min(RESIDUAL_INNER_FOLDS, int(np.min(class_counts)))
    if fold_count < 2:
        raise ValueError(
            "Residual prediction requires two training examples per class"
        )
    splitter = StratifiedKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=seed,
    )
    probabilities = np.empty(outcomes.size, dtype=np.float64)
    for train_indices, test_indices in splitter.split(contexts, outcomes):
        _, probabilities[test_indices] = _fit_logistic(
            contexts[train_indices],
            outcomes[train_indices],
            contexts[test_indices],
        )
    return probabilities, fold_count


def _heldout_residual_predictability(
    train_contexts: np.ndarray,
    validation_contexts: np.ndarray,
    train_outcomes: np.ndarray,
    validation_outcomes: np.ndarray,
    validation_logistic_probabilities: np.ndarray,
    *,
    bin_count: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    """Predict logistic residuals without training on validation outcomes."""
    training_probabilities, fold_count = (
        _training_oof_logistic_probabilities(
            train_contexts,
            train_outcomes,
            seed=seed + 1_000,
        )
    )
    training_residuals = train_outcomes.astype(np.float64) - training_probabilities
    minimum_leaf = min(
        RESIDUAL_FOREST_CONFIGURATION["min_samples_leaf"],
        max(2, train_contexts.shape[0] // 10),
    )
    forest = ExtraTreesRegressor(
        n_estimators=RESIDUAL_FOREST_CONFIGURATION["n_estimators"],
        max_depth=RESIDUAL_FOREST_CONFIGURATION["max_depth"],
        min_samples_leaf=minimum_leaf,
        max_features=RESIDUAL_FOREST_CONFIGURATION["max_features"],
        criterion=RESIDUAL_FOREST_CONFIGURATION["criterion"],
        random_state=seed + 2_000,
        n_jobs=1,
    )
    forest.fit(train_contexts, training_residuals)
    predicted_residuals = forest.predict(validation_contexts)

    linear = make_pipeline(
        StandardScaler(),
        Ridge(alpha=LINEAR_RESIDUAL_CONFIGURATION["alpha"]),
    )
    linear.fit(train_contexts, training_residuals)
    linear_predicted_residuals = linear.predict(validation_contexts)

    observed_residuals = (
        validation_outcomes.astype(np.float64)
        - validation_logistic_probabilities
    )
    zero_squared_errors = np.square(observed_residuals)
    forest_squared_errors = np.square(observed_residuals - predicted_residuals)
    linear_squared_errors = np.square(
        observed_residuals - linear_predicted_residuals
    )
    assignments = np.empty(observed_residuals.size, dtype=np.int64)
    ordered_indices = np.argsort(predicted_residuals, kind="stable")
    chunks = np.array_split(
        ordered_indices, min(bin_count, observed_residuals.size)
    )
    bin_rows = []
    for bin_index, indices in enumerate(chunks, start=1):
        assignments[indices] = bin_index
        selected_residuals = observed_residuals[indices]
        mean_observed = float(np.mean(selected_residuals))
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
                "mean_linear_predicted_residual": float(
                    np.mean(linear_predicted_residuals[indices])
                ),
                "mean_observed_residual": mean_observed,
                "standard_error": standard_error,
                "ci95_lower": mean_observed - 1.96 * standard_error,
                "ci95_upper": mean_observed + 1.96 * standard_error,
            }
        )

    point_rows = [
        {
            "example_index": index,
            "observed_disagreement": int(validation_outcomes[index]),
            "logistic_probability": float(
                validation_logistic_probabilities[index]
            ),
            "observed_logistic_residual": float(observed_residuals[index]),
            "forest_predicted_residual": float(predicted_residuals[index]),
            "linear_predicted_residual": float(
                linear_predicted_residuals[index]
            ),
            "residual_prediction_bin": int(assignments[index]),
            "zero_baseline_squared_error": float(zero_squared_errors[index]),
            "forest_squared_error": float(forest_squared_errors[index]),
            "linear_squared_error": float(linear_squared_errors[index]),
        }
        for index in range(observed_residuals.size)
    ]
    zero_mse = float(np.mean(zero_squared_errors))
    forest_mse = float(np.mean(forest_squared_errors))
    linear_mse = float(np.mean(linear_squared_errors))
    correlation = (
        float(np.corrcoef(observed_residuals, predicted_residuals)[0, 1])
        if np.std(observed_residuals) > 0.0
        and np.std(predicted_residuals) > 0.0
        else 0.0
    )
    summary = {
        "method": (
            "training-only cross-fitted logistic residual targets; Extra Trees "
            "and linear Ridge residual learners; untouched holdout evaluation"
        ),
        "training_residual_inner_folds": fold_count,
        "binning": (
            "equal-size validation bins ordered by signed forest-predicted "
            "logistic residual"
        ),
        "bin_count": int(len(chunks)),
        "zero_baseline_mse": zero_mse,
        "forest_residual_mse": forest_mse,
        "linear_residual_mse": linear_mse,
        "forest_mse_improvement_vs_zero": zero_mse - forest_mse,
        "forest_relative_mse_improvement_vs_zero": (
            (zero_mse - forest_mse) / zero_mse if zero_mse > 0.0 else 0.0
        ),
        "forest_mse_improvement_vs_linear": linear_mse - forest_mse,
        "forest_observed_residual_correlation": correlation,
        "forest_predicted_residual_standard_deviation": float(
            np.std(predicted_residuals)
        ),
        "bins_with_95_percent_interval_excluding_zero": int(
            sum(
                row["ci95_lower"] > 0.0 or row["ci95_upper"] < 0.0
                for row in bin_rows
            )
        ),
        "forest_configuration": {
            **RESIDUAL_FOREST_CONFIGURATION,
            "effective_min_samples_leaf": minimum_leaf,
        },
        "linear_configuration": LINEAR_RESIDUAL_CONFIGURATION,
    }
    return point_rows, bin_rows, summary


def _nonlinear_teacher_target(standardized_contexts: np.ndarray) -> np.ndarray:
    """Construct an outcome-free multifeature target for the forest teacher."""
    feature_count = standardized_contexts.shape[1]
    signal = np.zeros(standardized_contexts.shape[0], dtype=np.float64)
    for index in range(feature_count):
        partner = (index + 1) % feature_count
        sign = 1.0 if index % 2 == 0 else -1.0
        signal += sign * np.tanh(
            standardized_contexts[:, index]
            * standardized_contexts[:, partner]
        )
        signal += 0.25 * np.sin(
            (index + 1) * standardized_contexts[:, index]
        )
    scale = max(float(np.std(signal)), 1e-8)
    return 1.0 / (1.0 + np.exp(-signal / scale))


def _fit_synthetic_forest_teacher(
    train_contexts: np.ndarray,
    validation_contexts: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a multifeature forest teacher without using real outcomes."""
    feature_count = min(
        SYNTHETIC_FOREST_CONFIGURATION["prompt_feature_count"],
        train_contexts.shape[1],
    )
    train_features = np.asarray(
        train_contexts[:, :feature_count], dtype=np.float64
    )
    validation_features = np.asarray(
        validation_contexts[:, :feature_count], dtype=np.float64
    )
    means = np.mean(train_features, axis=0)
    scales = np.std(train_features, axis=0)
    scales[scales == 0.0] = 1.0
    standardized_train = (train_features - means) / scales
    latent_target = _nonlinear_teacher_target(standardized_train)
    minimum_leaf = min(
        SYNTHETIC_FOREST_CONFIGURATION["min_samples_leaf"],
        max(2, train_features.shape[0] // 10),
    )
    teacher = RandomForestRegressor(
        n_estimators=SYNTHETIC_FOREST_CONFIGURATION["n_estimators"],
        max_depth=SYNTHETIC_FOREST_CONFIGURATION["max_depth"],
        min_samples_leaf=minimum_leaf,
        max_features=SYNTHETIC_FOREST_CONFIGURATION["max_features"],
        criterion="squared_error",
        random_state=seed,
        n_jobs=1,
    )
    teacher.fit(train_features, latent_target)
    train_scores = teacher.predict(train_features)
    validation_scores = teacher.predict(validation_features)
    score_center = float(np.mean(train_scores))
    score_scale = max(float(np.std(train_scores)), 1e-8)

    def probabilities(scores: np.ndarray) -> np.ndarray:
        logits = (
            SYNTHETIC_FOREST_CONFIGURATION["probability_logit_scale"]
            * (scores - score_center)
            / score_scale
        )
        values = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        lower, upper = SYNTHETIC_FOREST_CONFIGURATION["probability_clip"]
        return np.clip(values, lower, upper)

    train_probabilities = probabilities(train_scores)
    validation_probabilities = probabilities(validation_scores)
    summary = {
        "purpose": (
            "Generate controlled multidimensional nonlinear fake labels from "
            "a random-forest teacher, then fit a logistic student."
        ),
        "real_outcomes_used": False,
        "fit_scope": "training prompt features only",
        "input_features": [
            f"prompt_pc_{index + 1}" for index in range(feature_count)
        ],
        "input_feature_count": feature_count,
        "latent_target": (
            "deterministic sum of alternating pairwise tanh interactions and "
            "univariate sine terms over standardized prompt components"
        ),
        "configuration": {
            **SYNTHETIC_FOREST_CONFIGURATION,
            "effective_min_samples_leaf": minimum_leaf,
        },
        "feature_importances": teacher.feature_importances_.tolist(),
        "training_score_center": score_center,
        "training_score_scale": score_scale,
        "train_probability_range": [
            float(np.min(train_probabilities)),
            float(np.max(train_probabilities)),
        ],
        "validation_probability_range": [
            float(np.min(validation_probabilities)),
            float(np.max(validation_probabilities)),
        ],
    }
    return train_probabilities, validation_probabilities, summary


def analyze_holdout_residuals(
    cache: RoutingCache,
    *,
    validation_fraction: float = 0.5,
    seed: int = 0,
    residual_bin_count: int = 10,
) -> dict:
    """Run the three requested analyses without writing result files."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if residual_bin_count < 2:
        raise ValueError("residual_bin_count must be at least two")
    uncertainty, uncertainty_block = cache.context_block("uncertainty")
    hidden, hidden_block = cache.context_block("hidden_state_pca")
    prompt, prompt_block = cache.context_block("prompt_embedding_pca")
    eligible_indices = cache.eligible_indices
    outcomes = np.asarray(cache.arrays["outcomes"], dtype=np.int64)[eligible_indices]
    if outcomes.size < 10 or np.unique(outcomes).tolist() != [0, 1]:
        raise ValueError("Eligible outcomes must contain both binary classes")

    uncertainty_hidden = np.concatenate(
        (uncertainty[eligible_indices], hidden[eligible_indices]), axis=1
    )
    prompt = prompt[eligible_indices]
    example_ids = [str(cache.records[int(index)]["id"]) for index in eligible_indices]
    positions = np.arange(outcomes.size)
    train_positions, validation_positions = train_test_split(
        positions,
        test_size=validation_fraction,
        random_state=seed,
        stratify=outcomes,
    )
    bin_count = max(2, int(round(math.sqrt(validation_positions.size))))

    all_points: list[dict] = []
    all_bins: list[dict] = []
    all_residual_prediction_points: list[dict] = []
    all_residual_prediction_bins: list[dict] = []
    scenarios: list[dict] = []
    for scenario_index, (name, contexts) in enumerate(
        (
            ("uncertainty_hidden_logistic", uncertainty_hidden),
            ("prompt_logistic", prompt),
        )
    ):
        points, bins, residual_points, residual_bins, metrics = _diagnose_scenario(
            name,
            contexts[train_positions],
            contexts[validation_positions],
            outcomes[train_positions],
            outcomes[validation_positions],
            validation_positions,
            example_ids,
            bin_count,
            residual_bin_count,
            "cached_weak_strong_disagreement",
            seed + scenario_index * 100,
        )
        all_points.extend(points)
        all_bins.extend(bins)
        all_residual_prediction_points.extend(residual_points)
        all_residual_prediction_bins.extend(residual_bins)
        scenarios.append(metrics)

    train_teacher_probabilities, validation_teacher_probabilities, teacher_summary = (
        _fit_synthetic_forest_teacher(
            prompt[train_positions],
            prompt[validation_positions],
            seed=seed + 9_000,
        )
    )
    synthetic_seed = seed + 10_000
    synthetic_rng = np.random.default_rng(synthetic_seed)
    fake_train_outcomes = synthetic_rng.binomial(
        1, train_teacher_probabilities
    ).astype(np.int64)
    fake_validation_outcomes = synthetic_rng.binomial(
        1, validation_teacher_probabilities
    ).astype(np.int64)
    if np.unique(fake_train_outcomes).size != 2:
        raise ValueError("The synthetic tree generated only one training class")
    (
        synthetic_points,
        synthetic_bins,
        synthetic_residual_points,
        synthetic_residual_bins,
        synthetic_metrics,
    ) = _diagnose_scenario(
        "synthetic_forest_labels_logistic",
        prompt[train_positions],
        prompt[validation_positions],
        fake_train_outcomes,
        fake_validation_outcomes,
        validation_positions,
        example_ids,
        bin_count,
        residual_bin_count,
        "synthetic_probabilistic_forest",
        seed + 200,
    )
    all_points.extend(synthetic_points)
    all_bins.extend(synthetic_bins)
    all_residual_prediction_points.extend(synthetic_residual_points)
    all_residual_prediction_bins.extend(synthetic_residual_bins)
    scenarios.append(synthetic_metrics)

    teacher_summary.update(
        {
            "label_sampling_seed": synthetic_seed,
            "train_fake_positive_rate": float(np.mean(fake_train_outcomes)),
            "validation_fake_positive_rate": float(
                np.mean(fake_validation_outcomes)
            ),
        }
    )
    summary = {
        "cache_schema_version": cache.manifest["schema_version"],
        "cache_examples": int(cache.manifest["examples"]),
        "eligible_examples": int(outcomes.size),
        "dataset": cache.manifest.get("dataset"),
        "dataset_config": cache.manifest.get("dataset_config"),
        "dataset_splits": cache.manifest.get("splits"),
        "routing_reference": cache.manifest.get("routing_reference"),
        "outcome_definition": cache.manifest.get("outcome_definition"),
        "split": {
            "method": "stratified random holdout",
            "seed": seed,
            "validation_fraction": validation_fraction,
            "train_examples": int(train_positions.size),
            "validation_examples": int(validation_positions.size),
        },
        "context_blocks": {
            "uncertainty": uncertainty_block,
            "hidden_state_pca": hidden_block,
            "prompt_embedding_pca": prompt_block,
        },
        "binning": {
            "method": (
                "approximately equal-size validation bins ordered by predicted "
                "probability"
            ),
            "bin_count_rule": "round(sqrt(validation_examples))",
            "bin_count": bin_count,
            "residual": "observed binary label minus predicted probability",
            "reference": BINNED_RESIDUAL_REFERENCE,
        },
        "residual_predictability": {
            "purpose": (
                "Visually test whether a nonlinear learner can predict the "
                "direction of held-out logistic residuals without cancellation."
            ),
            "training_targets": (
                "logistic residuals made out-of-fold inside the training split"
            ),
            "validation_usage": (
                "validation outcomes are used only for final evaluation and "
                "confidence intervals"
            ),
            "bin_count": residual_bin_count,
            "binning": (
                "equal-size validation bins ordered by signed Extra Trees "
                "predicted residual"
            ),
            "forest_configuration": RESIDUAL_FOREST_CONFIGURATION,
            "linear_comparator_configuration": LINEAR_RESIDUAL_CONFIGURATION,
        },
        "scenarios": scenarios,
        "synthetic_forest_teacher": teacher_summary,
    }
    return {
        "summary": summary,
        "point_rows": all_points,
        "bin_rows": all_bins,
        "residual_prediction_point_rows": all_residual_prediction_points,
        "residual_prediction_bin_rows": all_residual_prediction_bins,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_binned_residuals(path: Path, bin_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(dict.fromkeys(row["scenario"] for row in bin_rows))
    figure, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(5.2 * len(scenarios), 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    titles = {
        "uncertainty_hidden_logistic": "Uncertainty + hidden state",
        "prompt_logistic": "Prompt features",
        "synthetic_forest_labels_logistic": "Forest-generated fake labels",
    }
    for axis, scenario in zip(axes, scenarios):
        selected = [row for row in bin_rows if row["scenario"] == scenario]
        axis.errorbar(
            [row["mean_predicted_probability"] for row in selected],
            [row["mean_raw_residual"] for row in selected],
            yerr=[1.96 * row["standard_error"] for row in selected],
            color="#e45756",
            marker="o",
            linewidth=1.8,
            capsize=3,
        )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(titles.get(scenario, scenario))
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Mean predicted probability in bin")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"Mean raw residual $y - \hat{p}$")
    figure.suptitle("Validation-set binned logistic residuals")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_residual_predictability(
    path: Path,
    bin_rows: list[dict],
    scenarios: list[dict],
) -> None:
    """Plot held-out residual means after sorting by nonlinear corrections."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario_names = list(dict.fromkeys(row["scenario"] for row in bin_rows))
    if not scenario_names:
        raise ValueError("Residual predictability plot requires bin rows")
    metric_lookup = {row["scenario"]: row for row in scenarios}
    figure, axes = plt.subplots(
        1,
        len(scenario_names),
        figsize=(5.2 * len(scenario_names), 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    titles = {
        "uncertainty_hidden_logistic": "Uncertainty + hidden state",
        "prompt_logistic": "Prompt features",
        "synthetic_forest_labels_logistic": "Forest-generated fake labels",
    }
    bounds = [0.0]
    for row in bin_rows:
        bounds.extend(
            [
                row["mean_predicted_residual"],
                row["ci95_lower"],
                row["ci95_upper"],
            ]
        )
    limit = max(0.05, max(abs(value) for value in bounds) * 1.08)
    for axis, scenario_name in zip(axes, scenario_names):
        selected = [
            row for row in bin_rows if row["scenario"] == scenario_name
        ]
        axis.errorbar(
            [row["mean_predicted_residual"] for row in selected],
            [row["mean_observed_residual"] for row in selected],
            yerr=[1.96 * row["standard_error"] for row in selected],
            color="#e45756",
            marker="o",
            linewidth=2,
            capsize=4,
            label="Bin mean and 95% interval",
        )
        axis.plot(
            [-limit, limit],
            [-limit, limit],
            color="#4c78a8",
            linestyle=":",
            linewidth=1.5,
            label="Ideal correction",
        )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.axvline(0.0, color="grey", linewidth=0.8, alpha=0.7)
        metric = metric_lookup[scenario_name]["residual_predictability"]
        axis.set_title(titles.get(scenario_name, scenario_name))
        axis.text(
            0.03,
            0.97,
            (
                f"Forest ΔMSE vs zero: "
                f"{metric['forest_mse_improvement_vs_zero']:+.4f}\n"
                f"Forest ΔMSE vs linear: "
                f"{metric['forest_mse_improvement_vs_linear']:+.4f}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_xlabel("Mean forest-predicted logistic residual")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean observed held-out logistic residual")
    axes[0].legend(fontsize=8, loc="lower right")
    figure.suptitle("Held-out nonlinear residual predictability")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _bundle(output_dir: Path) -> Path:
    destination = output_dir / "holdout-residual-results.zip"
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path != destination:
                archive.write(path, path.name)
    return destination


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cache = load_cache(args.cache)
    result = analyze_holdout_residuals(
        cache,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        residual_bin_count=args.residual_bin_count,
    )
    result["summary"]["cache"] = str(args.cache.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8"
    )
    _write_csv(output / "validation_residuals.csv", result["point_rows"])
    _write_csv(output / "binned_residuals.csv", result["bin_rows"])
    _write_csv(
        output / "residual_predictability.csv",
        result["residual_prediction_point_rows"],
    )
    _write_csv(
        output / "residual_predictability_bins.csv",
        result["residual_prediction_bin_rows"],
    )
    _plot_binned_residuals(output / "binned_residuals.png", result["bin_rows"])
    _plot_residual_predictability(
        output / "residual_predictability.png",
        result["residual_prediction_bin_rows"],
        result["summary"]["scenarios"],
    )
    bundle = _bundle(output)
    print(
        f"Finished three held-out residual analyses. Results: {bundle}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
