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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
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
SYNTHETIC_TREE_LEAF_PROBABILITIES = (0.02, 0.98, 0.98, 0.02)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare validation-set binned logistic residuals for the cached "
            "uncertainty+hidden context, prompt context, and synthetic "
            "decision-tree labels."
        )
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("holdout-residual-results")
    )
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
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
    label_source: str,
) -> tuple[list[dict], list[dict], dict, object]:
    model, probabilities = _fit_logistic(
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
        }
    )
    return points, bins, metrics, model


def _synthetic_tree_outputs(
    contexts: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return leaf IDs and probabilities from a fixed depth-two tree."""
    root_right = contexts[:, 0] > thresholds[0]
    left_right = contexts[:, 1] > thresholds[1]
    right_right = contexts[:, 2] > thresholds[2]
    leaf_ids = np.where(
        root_right,
        np.where(right_right, 3, 2),
        np.where(left_right, 1, 0),
    ).astype(np.int64)
    probabilities = np.asarray(SYNTHETIC_TREE_LEAF_PROBABILITIES)[leaf_ids]
    return leaf_ids, probabilities


def _synthetic_tree_rules(thresholds: np.ndarray) -> str:
    probabilities = SYNTHETIC_TREE_LEAF_PROBABILITIES
    return "\n".join(
        [
            f"if prompt_pc_1 <= {thresholds[0]:.8g}:",
            (
                f"  if prompt_pc_2 <= {thresholds[1]:.8g}: "
                f"P(fake_y=1) = {probabilities[0]:.2f}"
            ),
            f"  else: P(fake_y=1) = {probabilities[1]:.2f}",
            "else:",
            (
                f"  if prompt_pc_3 <= {thresholds[2]:.8g}: "
                f"P(fake_y=1) = {probabilities[2]:.2f}"
            ),
            f"  else: P(fake_y=1) = {probabilities[3]:.2f}",
        ]
    ) + "\n"


def _tree_leaf_residuals(
    validation_leaf_ids: np.ndarray,
    validation_teacher_probabilities: np.ndarray,
    point_rows: list[dict],
) -> list[dict]:
    rows = []
    for leaf_id in sorted(np.unique(validation_leaf_ids)):
        indices = np.flatnonzero(validation_leaf_ids == leaf_id)
        selected = [point_rows[int(index)] for index in indices]
        residuals = np.asarray(
            [row["raw_residual"] for row in selected], dtype=np.float64
        )
        standard_error = (
            float(np.std(residuals, ddof=1) / np.sqrt(residuals.size))
            if residuals.size > 1
            else 0.0
        )
        mean_residual = float(np.mean(residuals))
        rows.append(
            {
                "tree_leaf_id": int(leaf_id),
                "count": int(indices.size),
                "teacher_positive_probability": float(
                    np.mean(validation_teacher_probabilities[indices])
                ),
                "fake_positive_rate": float(
                    np.mean([row["observed_disagreement"] for row in selected])
                ),
                "mean_logistic_probability": float(
                    np.mean(
                        [
                            row["predicted_disagreement_probability"]
                            for row in selected
                        ]
                    )
                ),
                "mean_raw_residual": mean_residual,
                "standard_error": standard_error,
                "ci95_lower": mean_residual - 1.96 * standard_error,
                "ci95_upper": mean_residual + 1.96 * standard_error,
            }
        )
    return rows


def analyze_holdout_residuals(
    cache: RoutingCache,
    *,
    validation_fraction: float = 0.5,
    seed: int = 0,
) -> dict:
    """Run the three requested analyses without writing result files."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
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
    scenarios: list[dict] = []
    for name, contexts in (
        ("uncertainty_hidden_logistic", uncertainty_hidden),
        ("prompt_logistic", prompt),
    ):
        points, bins, metrics, _ = _diagnose_scenario(
            name,
            contexts[train_positions],
            contexts[validation_positions],
            outcomes[train_positions],
            outcomes[validation_positions],
            validation_positions,
            example_ids,
            bin_count,
            "cached_weak_strong_disagreement",
        )
        all_points.extend(points)
        all_bins.extend(bins)
        scenarios.append(metrics)

    if prompt.shape[1] < 3:
        raise ValueError("The synthetic tree requires at least three prompt features")
    tree_thresholds = np.median(prompt[train_positions, :3], axis=0)
    train_leaf_ids, train_teacher_probabilities = _synthetic_tree_outputs(
        prompt[train_positions], tree_thresholds
    )
    validation_leaf_ids, validation_teacher_probabilities = (
        _synthetic_tree_outputs(prompt[validation_positions], tree_thresholds)
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
    synthetic_points, synthetic_bins, synthetic_metrics, _ = _diagnose_scenario(
        "synthetic_tree_labels_logistic",
        prompt[train_positions],
        prompt[validation_positions],
        fake_train_outcomes,
        fake_validation_outcomes,
        validation_positions,
        example_ids,
        bin_count,
        "synthetic_probabilistic_tree",
    )
    all_points.extend(synthetic_points)
    all_bins.extend(synthetic_bins)
    scenarios.append(synthetic_metrics)

    tree_leaf_rows = _tree_leaf_residuals(
        validation_leaf_ids,
        validation_teacher_probabilities,
        synthetic_points,
    )
    teacher_summary = {
        "purpose": (
            "Generate controlled nonlinear fake labels from a probabilistic "
            "depth-two tree over prompt features, then fit a logistic student."
        ),
        "real_outcomes_used": False,
        "threshold_fit_scope": "training prompt features only",
        "depth": 2,
        "nonlinearity": (
            "alternating near-deterministic leaf probabilities create a "
            "gated interaction that has no additive linear-logit form"
        ),
        "split_features": ["prompt_pc_1", "prompt_pc_2", "prompt_pc_3"],
        "split_thresholds": tree_thresholds.tolist(),
        "leaf_positive_probabilities": list(SYNTHETIC_TREE_LEAF_PROBABILITIES),
        "label_sampling_seed": synthetic_seed,
        "train_fake_positive_rate": float(np.mean(fake_train_outcomes)),
        "validation_fake_positive_rate": float(np.mean(fake_validation_outcomes)),
        "train_leaf_counts": np.bincount(train_leaf_ids, minlength=4).tolist(),
        "validation_leaf_counts": np.bincount(
            validation_leaf_ids, minlength=4
        ).tolist(),
    }
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
        "scenarios": scenarios,
        "synthetic_tree_teacher": teacher_summary,
    }
    return {
        "summary": summary,
        "point_rows": all_points,
        "bin_rows": all_bins,
        "tree_leaf_rows": tree_leaf_rows,
        "tree_rules": _synthetic_tree_rules(tree_thresholds),
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
        "synthetic_tree_labels_logistic": "Tree-generated fake labels",
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


def _plot_tree_leaf_residuals(path: Path, leaf_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    leaf_ids = [row["tree_leaf_id"] for row in leaf_rows]
    residuals = [row["mean_raw_residual"] for row in leaf_rows]
    intervals = [1.96 * row["standard_error"] for row in leaf_rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.errorbar(
        leaf_ids,
        residuals,
        yerr=intervals,
        color="#e45756",
        marker="o",
        markersize=7,
        linewidth=2,
        capsize=5,
    )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(leaf_ids)
    axis.set_xlabel("Synthetic generating-tree leaf")
    axis.set_ylabel(r"Mean raw residual $y - \hat{p}$")
    axis.set_title("Logistic residuals within synthetic tree leaves")
    axis.grid(alpha=0.2)
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
        output / "synthetic_tree_leaf_residuals.csv", result["tree_leaf_rows"]
    )
    (output / "synthetic_tree_rules.txt").write_text(
        result["tree_rules"], encoding="utf-8"
    )
    _plot_binned_residuals(output / "binned_residuals.png", result["bin_rows"])
    _plot_tree_leaf_residuals(
        output / "synthetic_tree_leaf_residuals.png",
        result["tree_leaf_rows"],
    )
    bundle = _bundle(output)
    print(
        f"Finished three holdout residual analyses. Results: {bundle}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
