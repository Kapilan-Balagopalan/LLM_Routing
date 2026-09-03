"""Compare prompt embeddings with the existing LLM-derived routing context."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np

from llm_routing_simulation.cache import load_cache
from llm_routing_simulation.prompt_embeddings import (
    load_prompt_embedding_cache,
    validate_prompt_embedding_source,
)
from llm_routing_simulation.skyline import (
    cross_fitted_residual_predictability,
    evaluate_probability_predictions,
    fit_supervised_skylines,
    plot_residual_predictability,
    random_routing_reference,
    threshold_skyline,
)


CONTEXT_LABELS = {
    "current": "Current 78D",
    "prompt": "Prompt PCA",
    "compact": "Uncertainty + prompt PCA",
}


def align_prompt_embeddings(rounds, embedding_cache) -> np.ndarray:
    """Align a sidecar matrix to eligible routing rounds by stable example ID."""
    ids = embedding_cache.example_ids.astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("Prompt embedding cache contains duplicate example IDs")
    positions = {example_id: index for index, example_id in enumerate(ids)}
    try:
        indices = [positions[item.example_id] for item in rounds]
    except KeyError as exc:
        raise ValueError(f"Prompt embedding is missing example ID {exc.args[0]}") from exc
    return np.asarray(embedding_cache.embeddings[indices], dtype=np.float64)


def prompt_pca_context(
    embeddings: np.ndarray, components: int
) -> tuple[np.ndarray, dict]:
    """Fit a fixed transductive, outcome-free PCA to frozen prompt embeddings."""
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise RuntimeError("Prompt PCA requires scikit-learn") from exc

    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("Prompt embeddings must be a finite nonempty matrix")
    available = min(matrix.shape)
    if not 1 <= components <= available:
        raise ValueError(f"prompt PCA components must be between 1 and {available}")
    projector = PCA(n_components=components, whiten=True, svd_solver="full")
    projected = projector.fit_transform(matrix)
    mean = projected.mean(axis=0)
    scale = projected.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (projected - mean) / scale
    return standardized, {
        "scope": "fixed transductive, outcome-free, selected eligible prompts",
        "components": int(components),
        "raw_embedding_dimension": int(matrix.shape[1]),
        "explained_variance_ratio_sum": float(
            np.sum(projector.explained_variance_ratio_)
        ),
        "whitened": True,
        "featurewise_standardized": True,
    }


def build_context_matrices(
    current_context: np.ndarray,
    prompt_context: np.ndarray,
    uncertainty_dimension: int,
) -> dict[str, np.ndarray]:
    """Build the baseline, prompt-only, and compact hybrid contexts."""
    current = np.asarray(current_context, dtype=np.float64)
    prompt = np.asarray(prompt_context, dtype=np.float64)
    if (
        current.ndim != 2
        or prompt.ndim != 2
        or current.shape[0] != prompt.shape[0]
    ):
        raise ValueError("Current and prompt contexts must be aligned matrices")
    if not 1 <= uncertainty_dimension <= current.shape[1]:
        raise ValueError("Invalid uncertainty feature dimension")
    compact = np.concatenate(
        (current[:, :uncertainty_dimension], prompt), axis=1
    )
    return {
        CONTEXT_LABELS["current"]: current,
        CONTEXT_LABELS["prompt"]: prompt,
        CONTEXT_LABELS["compact"]: compact,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _plot_model_comparison(output_path: Path, comparison_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preferred_models = [
        "Logistic (out-of-fold)",
        "HGB-350 (out-of-fold)",
        "MLP-8 (out-of-fold)",
        "RBF SVM (out-of-fold)",
        "XGBoost (out-of-fold)",
    ]
    present = {row["model"] for row in comparison_rows}
    models = [model for model in preferred_models if model in present]
    contexts = list(CONTEXT_LABELS.values())
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(contexts), dtype=np.float64)
    width = 0.8 / max(1, len(models))
    for model_index, model in enumerate(models):
        selected = {
            row["context"]: row
            for row in comparison_rows
            if row["model"] == model
        }
        offset = (model_index - (len(models) - 1) / 2.0) * width
        label = model.replace(" (out-of-fold)", "")
        axes[0].bar(
            x + offset,
            [selected[context]["roc_auc"] for context in contexts],
            width,
            label=label,
        )
        axes[1].bar(
            x + offset,
            [selected[context]["log_loss"] for context in contexts],
            width,
            label=label,
        )
    for axis, title, ylabel in (
        (axes[0], "Disagreement ranking", "Out-of-fold ROC AUC"),
        (axes[1], "Probability quality", "Out-of-fold log loss (lower is better)"),
    ):
        axis.set_xticks(x, contexts)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].legend(fontsize=8)
    figure.suptitle("Prompt-embedding context comparison on identical folds")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def evaluate_two_stage_correction(
    residual_rows: list[dict],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Evaluate base plus residual predictions from one nested cross-fit."""
    if not residual_rows:
        raise ValueError("Residual rows must be nonempty")
    ordered = sorted(residual_rows, key=lambda row: int(row["example_index"]))
    indices = [int(row["example_index"]) for row in ordered]
    if indices != list(range(len(ordered))):
        raise ValueError("Residual rows must contain each example index exactly once")

    outcomes = np.asarray(
        [row["observed_disagreement"] for row in ordered], dtype=np.int64
    )
    base = np.asarray(
        [row["logistic_probability"] for row in ordered], dtype=np.float64
    )
    correction = np.asarray(
        [row["predicted_logistic_residual"] for row in ordered],
        dtype=np.float64,
    )
    corrected = np.clip(base + correction, 1e-7, 1.0 - 1e-7)
    base_metrics = evaluate_probability_predictions(base, outcomes)
    corrected_metrics = evaluate_probability_predictions(corrected, outcomes)
    deltas = {
        key: float(corrected_metrics[key] - base_metrics[key])
        for key in base_metrics
    }
    summary = {
        "construction": (
            "clip(cross-fitted current-context logistic probability + "
            "cross-fitted prompt-HGB predicted residual, 1e-7, 1 - 1e-7)"
        ),
        "base_model": "Current 78D logistic (nested cross-fitted)",
        "correction_model": "Prompt PCA HGB residual regressor (nested cross-fitted)",
        "base_metrics": base_metrics,
        "corrected_metrics": corrected_metrics,
        "corrected_minus_base": deltas,
        "improves_auc": bool(deltas["roc_auc"] > 0.0),
        "improves_log_loss": bool(deltas["log_loss"] < 0.0),
        "improves_brier": bool(deltas["brier_score"] < 0.0),
        "improves_all_auc_logloss_brier": bool(
            deltas["roc_auc"] > 0.0
            and deltas["log_loss"] < 0.0
            and deltas["brier_score"] < 0.0
        ),
    }
    return corrected, outcomes, summary


def summarize_two_stage_seeds(seed_rows: list[dict]) -> dict:
    """Summarize repeated split seeds without treating them as new datasets."""
    if not seed_rows:
        raise ValueError("At least one seed result is required")
    delta_keys = [
        "delta_roc_auc",
        "delta_log_loss",
        "delta_brier_score",
        "delta_routing_accuracy_at_50_percent",
    ]
    aggregate = {}
    for key in delta_keys:
        values = np.asarray([row[key] for row in seed_rows], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=0)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return {
        "seeds": [int(row["seed"]) for row in seed_rows],
        "split_repetitions": len(seed_rows),
        "aggregate_deltas_corrected_minus_base": aggregate,
        "seeds_improving_auc": int(
            sum(row["delta_roc_auc"] > 0.0 for row in seed_rows)
        ),
        "seeds_improving_log_loss": int(
            sum(row["delta_log_loss"] < 0.0 for row in seed_rows)
        ),
        "seeds_improving_brier": int(
            sum(row["delta_brier_score"] < 0.0 for row in seed_rows)
        ),
        "seeds_improving_all_auc_logloss_brier": int(
            sum(row["improves_all_auc_logloss_brier"] for row in seed_rows)
        ),
        "interpretation": (
            "Repeated seeds measure sensitivity to cross-validation splits on the "
            "same dataset; they are not independent replications or a final test set."
        ),
    }


def _plot_two_stage_skyline(
    output_path: Path,
    baseline_rows: list[dict],
    corrected_rows: list[dict],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_names = {
        "Logistic (out-of-fold)",
        "XGBoost (out-of-fold)",
        "Random routing (expected)",
    }
    selected = [
        row
        for row in baseline_rows
        if row["context"] == CONTEXT_LABELS["current"]
        and row["model"] in selected_names
    ]
    selected.extend({"context": "Two-stage", **row} for row in corrected_rows)
    styles = {
        "Logistic (out-of-fold)": ("#2563eb", "--"),
        "XGBoost (out-of-fold)": ("#16a34a", ":"),
        "Random routing (expected)": ("#6b7280", "--"),
        "Two-stage logistic + prompt residual": ("#dc2626", "-"),
    }
    figure, axis = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    for name in styles:
        rows = [row for row in selected if row["model"] == name]
        if not rows:
            continue
        rows.sort(key=lambda row: row["routing_rate"])
        color, line_style = styles[name]
        axis.plot(
            [row["routing_rate"] for row in rows],
            [row["accuracy"] for row in rows],
            label=name.replace(" (out-of-fold)", ""),
            color=color,
            linestyle=line_style,
            linewidth=2.2,
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Strong-model routing rate")
    axis.set_ylabel("Accuracy under cached strong-answer reference")
    axis.set_title("Nested cross-fitted two-stage routing skyline")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _bundle(output_dir: Path) -> Path:
    destination = output_dir / "prompt-embedding-results.zip"
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path != destination:
                archive.write(path, path.name)
    return destination


def run_prompt_embedding_experiment(
    *,
    cache_path: str | Path,
    prompt_embedding_path: str | Path,
    output_dir: str | Path,
    prompt_components: int = 32,
    requested_folds: int = 5,
    residual_permutations: int = 100,
    robustness_seeds: int = 5,
    seed: int = 0,
    limit: int | None = None,
) -> Path:
    """Run three supervised contexts and the incremental prompt residual test."""
    if requested_folds < 2 or residual_permutations < 0 or robustness_seeds < 1:
        raise ValueError(
            "folds must be >= 2, residual permutations nonnegative, and "
            "robustness seeds positive"
        )
    cache = load_cache(cache_path)
    rounds = cache.eligible_rounds()
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rounds = rounds[:limit]
    if not rounds:
        raise ValueError("The routing cache has no eligible examples")

    embedding_cache = load_prompt_embedding_cache(prompt_embedding_path)
    validate_prompt_embedding_source(embedding_cache, cache)
    raw_prompt_embeddings = align_prompt_embeddings(rounds, embedding_cache)
    prompt_context, pca_summary = prompt_pca_context(
        raw_prompt_embeddings, prompt_components
    )
    current_context = np.stack([item.context for item in rounds]).astype(np.float64)
    uncertainty_dimension = int(cache.arrays["uncertainty_features"].shape[1])
    outcomes = np.asarray(
        [item.weak_answer != item.strong_answer for item in rounds], dtype=np.int64
    )
    context_matrices = build_context_matrices(
        current_context,
        prompt_context,
        uncertainty_dimension,
    )

    all_skyline_rows: list[dict] = []
    all_comparison_rows: list[dict] = []
    context_summaries = {}
    baseline_agreement = float(np.mean(outcomes == 0))
    for context_label, matrix in context_matrices.items():
        skyline_rows, context_summary, _ = fit_supervised_skylines(
            matrix,
            outcomes,
            seed=seed,
            requested_folds=requested_folds,
        )
        skyline_rows.extend(random_routing_reference(baseline_agreement))
        all_skyline_rows.extend(
            {"context": context_label, **row} for row in skyline_rows
        )
        all_comparison_rows.extend(
            {"context": context_label, **row}
            for row in context_summary["model_comparison"]
        )
        context_summaries[context_label] = context_summary

    (
        residual_rows,
        residual_bin_rows,
        permutation_rows,
        residual_summary,
    ) = cross_fitted_residual_predictability(
        current_context,
        outcomes,
        residual_contexts=prompt_context,
        base_context_label=CONTEXT_LABELS["current"],
        residual_context_label=CONTEXT_LABELS["prompt"],
        seed=seed,
        requested_folds=requested_folds,
        permutation_repeats=residual_permutations,
    )
    corrected_probabilities, corrected_outcomes, two_stage_primary = (
        evaluate_two_stage_correction(residual_rows)
    )
    two_stage_skyline = threshold_skyline(
        corrected_probabilities,
        corrected_outcomes,
        "Two-stage logistic + prompt residual",
        fit_scope="nested_cross_fitted",
        evaluation_auc=two_stage_primary["corrected_metrics"]["roc_auc"],
        folds=residual_summary["outer_folds"],
        configuration={
            "base": two_stage_primary["base_model"],
            "correction": two_stage_primary["correction_model"],
        },
        selected=True,
    )

    two_stage_seed_rows = []
    for split_seed in range(seed, seed + robustness_seeds):
        if split_seed == seed:
            seed_residual_rows = residual_rows
            seed_residual_summary = residual_summary
            seed_two_stage = two_stage_primary
        else:
            seed_residual_rows, _, _, seed_residual_summary = (
                cross_fitted_residual_predictability(
                    current_context,
                    outcomes,
                    residual_contexts=prompt_context,
                    base_context_label=CONTEXT_LABELS["current"],
                    residual_context_label=CONTEXT_LABELS["prompt"],
                    seed=split_seed,
                    requested_folds=requested_folds,
                    permutation_repeats=0,
                )
            )
            _, _, seed_two_stage = evaluate_two_stage_correction(
                seed_residual_rows
            )
        base_metrics = seed_two_stage["base_metrics"]
        corrected_metrics = seed_two_stage["corrected_metrics"]
        deltas = seed_two_stage["corrected_minus_base"]
        two_stage_seed_rows.append(
            {
                "seed": split_seed,
                "examples": len(rounds),
                "base_roc_auc": base_metrics["roc_auc"],
                "corrected_roc_auc": corrected_metrics["roc_auc"],
                "delta_roc_auc": deltas["roc_auc"],
                "base_log_loss": base_metrics["log_loss"],
                "corrected_log_loss": corrected_metrics["log_loss"],
                "delta_log_loss": deltas["log_loss"],
                "base_brier_score": base_metrics["brier_score"],
                "corrected_brier_score": corrected_metrics["brier_score"],
                "delta_brier_score": deltas["brier_score"],
                "base_routing_accuracy_at_50_percent": base_metrics[
                    "routing_accuracy_at_50_percent"
                ],
                "corrected_routing_accuracy_at_50_percent": corrected_metrics[
                    "routing_accuracy_at_50_percent"
                ],
                "delta_routing_accuracy_at_50_percent": deltas[
                    "routing_accuracy_at_50_percent"
                ],
                "residual_mse_improvement": seed_residual_summary[
                    "mse_improvement"
                ],
                "residual_correlation": seed_residual_summary[
                    "residual_correlation"
                ],
                "improves_all_auc_logloss_brier": seed_two_stage[
                    "improves_all_auc_logloss_brier"
                ],
            }
        )
    two_stage_stability = summarize_two_stage_seeds(two_stage_seed_rows)
    two_stage_summary = {
        "primary_seed": seed,
        "primary_seed_evaluation": two_stage_primary,
        "primary_seed_residual_permutation_test": residual_summary,
        "split_seed_stability": two_stage_stability,
        "scope_warning": (
            "This is a supervised nested cross-fitted diagnostic. It does not "
            "replace or evaluate the partial-feedback online routing players."
        ),
    }

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "prompt_context_model_comparison.csv", all_comparison_rows)
    _write_csv(output / "prompt_context_skylines.csv", all_skyline_rows)
    _write_csv(output / "prompt_incremental_residuals.csv", residual_rows)
    _write_csv(output / "prompt_incremental_residual_bins.csv", residual_bin_rows)
    _write_csv(output / "prompt_incremental_permutations.csv", permutation_rows)
    _write_csv(output / "two_stage_seed_results.csv", two_stage_seed_rows)
    _write_csv(output / "two_stage_skyline.csv", two_stage_skyline)
    (output / "prompt_incremental_residual_summary.json").write_text(
        json.dumps(residual_summary, indent=2), encoding="utf-8"
    )
    (output / "two_stage_summary.json").write_text(
        json.dumps(two_stage_summary, indent=2), encoding="utf-8"
    )
    summary = {
        "cache": str(Path(cache_path).resolve()),
        "prompt_embedding_cache": str(Path(prompt_embedding_path).resolve()),
        "examples": len(rounds),
        "outcome": "weak/strong answer disagreement",
        "reference": "cached strong-model answer",
        "embedding_manifest": embedding_cache.manifest,
        "prompt_pca": pca_summary,
        "contexts": {
            label: {
                "dimension": int(matrix.shape[1]),
                "supervised": context_summaries[label],
            }
            for label, matrix in context_matrices.items()
        },
        "incremental_prompt_residual_test": residual_summary,
        "two_stage_cross_fitted_evaluation": two_stage_summary,
        "seed": seed,
        "folds": requested_folds,
        "residual_permutations": residual_permutations,
        "robustness_seeds": robustness_seeds,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot_model_comparison(
        output / "prompt_context_model_comparison.png", all_comparison_rows
    )
    plot_residual_predictability(
        output / "prompt_incremental_residual_test.png",
        residual_rows,
        residual_bin_rows,
        permutation_rows,
        residual_summary,
    )
    _plot_two_stage_skyline(
        output / "two_stage_skyline.png",
        all_skyline_rows,
        two_stage_skyline,
    )
    return _bundle(output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare current, prompt-only, and combined routing contexts."
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--prompt-embeddings", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("prompt-embedding-results")
    )
    parser.add_argument("--prompt-components", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--residual-permutations", type=int, default=100)
    parser.add_argument(
        "--robustness-seeds",
        type=int,
        default=5,
        help="Consecutive cross-validation split seeds for the two-stage check",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_prompt_embedding_experiment(
        cache_path=args.cache,
        prompt_embedding_path=args.prompt_embeddings,
        output_dir=args.output_dir,
        prompt_components=args.prompt_components,
        requested_folds=args.folds,
        residual_permutations=args.residual_permutations,
        robustness_seeds=args.robustness_seeds,
        seed=args.seed,
        limit=args.limit,
    )
    print(f"Finished. Results: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
