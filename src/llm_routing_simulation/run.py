"""CPU-only replay of routing algorithms from a collected cache."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from llm_routing_simulation.algorithm import (
    DEFAULT_HGB_MAX_LEAF_NODES,
    HGBETCPlayer,
    IGWPlayer,
    LogCBPSideATConfig,
    LogCBPSideATPlayer,
)
from llm_routing_simulation.cache import RoutingCache, load_cache
from llm_routing_simulation.environment import LLMCascadeEnvironment
from llm_routing_simulation.skyline import fit_holdout_prompt_skylines
from llm_routing_simulation.synthetic_prompt import (
    generate_synthetic_prompt_outcomes,
)


SKYLINE_PLOT_MODELS = (
    "Logistic (80/20 holdout)",
    *(
        f"HGB leaves={max_leaf_nodes} (80/20 holdout)"
        for max_leaf_nodes in DEFAULT_HGB_MAX_LEAF_NODES
    ),
)

DEFAULT_ALPHA_VALUES = tuple(np.linspace(0.55, 0.30, 10))
DEFAULT_L01_VALUES = tuple(1.0 / alpha for alpha in DEFAULT_ALPHA_VALUES)
DEFAULT_IGW_GAMMA_VALUES = (64.0,)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay online routing algorithms and supervised skylines offline."
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument(
        "--context-profile",
        choices=("prompt-only", "uncertainty-prompt"),
        default="prompt-only",
        help=(
            "Use only prompt-embedding PCA components, or concatenate the full "
            "manifest-defined uncertainty block before those prompt components"
        ),
    )
    parser.add_argument("--prompt-components", type=int, default=20)
    parser.add_argument(
        "--outcome-source",
        choices=("cached", "synthetic"),
        default="cached",
        help=(
            "Use cached weak/strong disagreement for the real experiment or "
            "the committed forest positive control"
        ),
    )
    parser.add_argument(
        "--experiment", choices=("all", "online", "skyline"), default="all"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("simulation-results"))
    parser.add_argument("--limit", type=int, help="Optional prefix of eligible rows")
    parser.add_argument(
        "--l01-values", type=float, nargs="+", default=list(DEFAULT_L01_VALUES)
    )
    parser.add_argument("--l11", type=float, default=1.0)
    parser.add_argument("--etc-tastes", type=int, default=300)
    parser.add_argument("--cbpside-tastes", type=int, default=0)
    parser.add_argument("--cbpside-bootstrap-per-class", type=int, default=0)
    parser.add_argument("--cbpside-bootstrap-max-tastes", type=int, default=0)
    parser.add_argument("--igw-min-tastes", type=int, default=0)
    parser.add_argument("--igw-bootstrap-per-class", type=int, default=0)
    parser.add_argument("--igw-bootstrap-max-tastes", type=int, default=0)
    parser.add_argument(
        "--igw-gamma-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_IGW_GAMMA_VALUES),
    )
    parser.add_argument(
        "--hgb-max-leaf-nodes",
        type=int,
        nargs="+",
        default=list(DEFAULT_HGB_MAX_LEAF_NODES),
        help="HGB capacity profiles; every other HGB setting is held fixed",
    )
    parser.add_argument("--igw-mu", type=float, default=2.0)
    parser.add_argument("--igw-min-propensity", type=float, default=0.1)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--skyline-validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _prompt_context_rounds(
    cache: RoutingCache,
    prompt_components: int,
    limit: int | None,
    seed: int = 0,
    outcome_source: str = "cached",
    context_profile: str = "prompt-only",
):
    """Attach manifest-selected contexts and the requested outcome source."""
    if outcome_source not in {"cached", "synthetic"}:
        raise ValueError("outcome_source must be 'cached' or 'synthetic'")
    if context_profile not in {"prompt-only", "uncertainty-prompt"}:
        raise ValueError(
            "context_profile must be 'prompt-only' or 'uncertainty-prompt'"
        )
    all_rounds = cache.eligible_rounds()
    all_prompt_contexts, prompt_block = cache.context_block(
        "prompt_embedding_pca", components=prompt_components
    )
    eligible_prompt_contexts = all_prompt_contexts[cache.eligible_indices]
    uncertainty_block = None
    if context_profile == "uncertainty-prompt":
        all_uncertainty_contexts, uncertainty_block = cache.context_block(
            "uncertainty"
        )
        eligible_uncertainty_contexts = all_uncertainty_contexts[
            cache.eligible_indices
        ]
        eligible_contexts = np.concatenate(
            (eligible_uncertainty_contexts, eligible_prompt_contexts), axis=1
        )
    else:
        eligible_contexts = eligible_prompt_contexts
    teacher_summary = None
    synthetic_rows: list[dict] = []
    teacher_probabilities = None
    synthetic_outcomes = None
    if outcome_source == "synthetic":
        teacher_probabilities, synthetic_outcomes, teacher_summary = (
            generate_synthetic_prompt_outcomes(
                eligible_prompt_contexts, seed=seed
            )
        )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        all_rounds = all_rounds[:limit]
        eligible_contexts = eligible_contexts[:limit]
        if teacher_probabilities is not None and synthetic_outcomes is not None:
            teacher_probabilities = teacher_probabilities[:limit]
            synthetic_outcomes = synthetic_outcomes[:limit]
    if not all_rounds:
        raise ValueError("The cache has no eligible weak/strong answer pairs")
    context_rounds = [
        replace(
            item,
            context=eligible_contexts[index].copy(),
            outcome_override=(
                int(synthetic_outcomes[index])
                if synthetic_outcomes is not None
                else None
            ),
        )
        for index, item in enumerate(all_rounds)
    ]
    context_blocks = (
        [uncertainty_block, prompt_block]
        if uncertainty_block is not None
        else [prompt_block]
    )
    context_summary = {
        "profile": context_profile,
        "source": cache.manifest.get("prompt_embedding_definition"),
        # Retained for compatibility with existing prompt-only result readers.
        "context_block": prompt_block,
        "context_blocks": context_blocks,
        "context_block_order": [block["name"] for block in context_blocks],
        "prompt_context_block": prompt_block,
        "uncertainty_context_block": uncertainty_block,
        "context_dimension": int(eligible_contexts.shape[1]),
        "pca_scope": cache.manifest.get("pca_scope"),
        "fit_scope": "precomputed across all collected prompts",
        "uncertainty_features_used": uncertainty_block is not None,
        "hidden_state_features_used": False,
        "outcome_or_answer_features_used": False,
        "component_selection": {
            "uncertainty": (
                "all components in the manifest-defined block"
                if uncertainty_block is not None
                else "excluded"
            ),
            "prompt_embedding_pca": (
                "first components in manifest-defined PCA order"
            ),
        },
    }
    if teacher_summary is not None:
        teacher_summary["probabilities_stored_in_player_observations"] = False
        teacher_summary["selected_examples"] = len(context_rounds)
        teacher_summary["selected_probability_mean"] = float(
            np.mean(teacher_probabilities)
        )
        synthetic_rows = [
            {
                "t": index + 1,
                "id": item.example_id,
                "teacher_probability": float(teacher_probabilities[index]),
                "synthetic_outcome": int(synthetic_outcomes[index]),
            }
            for index, item in enumerate(context_rounds)
        ]
    return context_rounds, context_summary, teacher_summary, synthetic_rows


def _run_one_player(
    method: str,
    player,
    config,
    rounds,
    progress_label: str,
    metric: str,
):
    environment = LLMCascadeEnvironment(rounds)
    correct = 0
    routed = 0
    trajectories = []
    progress_every = max(1, len(rounds) // 20)
    while not environment.done:
        observation = environment.observe()
        decision_wrapper = player.next_action(observation.context)
        decision = decision_wrapper.diagnostics
        transition = environment.step(decision_wrapper.action)
        player.update(transition.action, observation.context, transition.outcome)

        # Evaluation may inspect the routing outcome after acting, while the
        # player still receives it only when action 1 was selected.
        evaluation_outcome = rounds[transition.t - 1].routing_outcome
        correct += int(transition.action == 1 or evaluation_outcome == 0)
        routed += transition.action
        trajectories.append(
            {
                "method": method,
                "l01": config.loss_reject_disagreement,
                "l11": config.loss_route_disagreement,
                "alpha": getattr(decision, "threshold", None),
                "t": transition.t,
                "id": transition.example_id,
                "context": observation.context.tolist(),
                "weak_answer": observation.weak_answer,
                "action": transition.action,
                "feedback_revealed_to_player": transition.outcome,
                "evaluation_only_routing_outcome": evaluation_outcome,
                "predicted_disagreement": getattr(
                    decision, "predicted_disagreement", None
                ),
                "reason": getattr(decision, "reason", None),
                "confidence_radius": getattr(decision, "confidence_radius", None),
                "theoretical_confidence_radius": getattr(
                    decision, "theoretical_confidence_radius", None
                ),
                "estimator": getattr(decision, "estimator", "logistic_regression"),
                "estimator_fitted": getattr(decision, "estimator_fitted", None),
                "training_count": getattr(decision, "training_count", None),
                "probability_1": getattr(decision, "probability_1", None),
                "igw_gap": getattr(decision, "gap", None),
                "igw_gamma": getattr(decision, "gamma", None),
                "effective_sample_size": getattr(
                    decision, "effective_sample_size", None
                ),
                "hgb_max_leaf_nodes": getattr(
                    getattr(player, "estimator", None),
                    "max_leaf_nodes",
                    None,
                ),
            }
        )
        if transition.t % progress_every == 0 or transition.t == len(rounds):
            print(f"[{progress_label}] {transition.t}/{len(rounds)}", flush=True)

    return (
        {
            "method": method,
            "l01": config.loss_reject_disagreement,
            "l11": config.loss_route_disagreement,
            "alpha": 1.0
            / (
                1.0
                + config.loss_reject_disagreement
                - config.loss_route_disagreement
            ),
            "routing_rate": routed / len(rounds),
            "accuracy": correct / len(rounds),
            "metric": metric,
            "examples": len(rounds),
            "min_tastes": getattr(player, "min_tastes", config.min_tastes),
            "bootstrap_per_class": getattr(player, "bootstrap_per_class", None),
            "bootstrap_max_tastes": getattr(
                player, "bootstrap_max_tastes", None
            ),
            "probability_estimator": getattr(
                getattr(player, "estimator", None),
                "estimator_name",
                "logistic_regression",
            ),
            "igw_gamma": getattr(player, "fixed_gamma", None),
            "igw_mu": getattr(player, "mu", None),
            "hgb_max_leaf_nodes": getattr(
                getattr(player, "estimator", None),
                "max_leaf_nodes",
                None,
            ),
        },
        trajectories,
    )


def _random_matched(
    rounds,
    target_rate: float,
    l01: float,
    l11: float,
    repeats: int,
    seed: int,
    metric: str,
    hgb_max_leaf_nodes: int,
):
    rng = np.random.default_rng(seed)
    outcomes = np.asarray(
        [item.routing_outcome for item in rounds], dtype=bool
    )
    rates, accuracies = [], []
    for _ in range(repeats):
        routed = rng.random(len(rounds)) < target_rate
        rates.append(float(np.mean(routed)))
        accuracies.append(float(np.mean(routed | ~outcomes)))
    return {
        "method": f"Random (matched ETC HGB leaves={hgb_max_leaf_nodes})",
        "l01": l01,
        "l11": l11,
        "alpha": 1.0 / (1.0 + l01 - l11),
        "routing_rate": float(np.mean(rates)),
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "metric": metric,
        "examples": len(rounds),
        "random_repeats": repeats,
        "hgb_max_leaf_nodes": hgb_max_leaf_nodes,
    }


def run_online(rounds, args) -> tuple[list[dict], list[dict]]:
    """Run HGB ETC/IGW, linear-logistic CBPSide, and matched random."""
    context_dim = int(rounds[0].context.size)
    metric = (
        "synthetic_routing_accuracy"
        if args.outcome_source == "synthetic"
        else "agreement_with_strong_llm"
    )
    rows, trajectories = [], []
    for l01 in args.l01_values:
        base = LogCBPSideATConfig(
            loss_reject_disagreement=l01,
            loss_route_disagreement=args.l11,
            beta_scale=0.5,
            max_confidence_radius=0.5,
            min_tastes=args.etc_tastes,
            use_confidence_bound=False,
        )
        cbpside = replace(
            base,
            min_tastes=args.cbpside_tastes,
            bootstrap_per_class=args.cbpside_bootstrap_per_class,
            bootstrap_max_tastes=args.cbpside_bootstrap_max_tastes,
            use_confidence_bound=True,
            beta_scale=base.beta_scale * 0.5,
        )
        cbpside_result, cbpside_path = _run_one_player(
            "CBPSide",
            LogCBPSideATPlayer(context_dim, cbpside),
            cbpside,
            rounds,
            f"l01={l01:g} CBPSide",
            metric,
        )
        rows.append(cbpside_result)
        trajectories.extend(cbpside_path)

        for profile_index, max_leaf_nodes in enumerate(args.hgb_max_leaf_nodes):
            etc_method = f"ETC HGB leaves={max_leaf_nodes}"
            etc_result, etc_path = _run_one_player(
                etc_method,
                HGBETCPlayer(
                    context_dim,
                    base,
                    hgb_max_leaf_nodes=max_leaf_nodes,
                    seed=args.seed,
                ),
                base,
                rounds,
                f"l01={l01:g} {etc_method}",
                metric,
            )
            rows.append(etc_result)
            trajectories.extend(etc_path)

            for gamma in args.igw_gamma_values:
                igw_method = (
                    f"IGW gamma={gamma:g} HGB leaves={max_leaf_nodes}"
                )
                igw_result, igw_path = _run_one_player(
                    igw_method,
                    IGWPlayer(
                        context_dim,
                        base,
                        len(rounds),
                        min_tastes=args.igw_min_tastes,
                        bootstrap_per_class=args.igw_bootstrap_per_class,
                        bootstrap_max_tastes=args.igw_bootstrap_max_tastes,
                        mu=args.igw_mu,
                        fixed_gamma=gamma,
                        min_propensity=args.igw_min_propensity,
                        hgb_max_leaf_nodes=max_leaf_nodes,
                        seed=args.seed,
                    ),
                    base,
                    rounds,
                    f"l01={l01:g} {igw_method}",
                    metric,
                )
                rows.append(igw_result)
                trajectories.extend(igw_path)

            rows.append(
                _random_matched(
                    rounds,
                    etc_result["routing_rate"],
                    l01,
                    args.l11,
                    args.random_repeats,
                    args.seed + 10_000 + profile_index,
                    metric,
                    max_leaf_nodes,
                )
            )
    return rows, trajectories


def run_skyline(
    rounds,
    validation_fraction: float,
    seed: int,
    outcome_source: str,
    hgb_max_leaf_nodes,
):
    contexts = np.stack([item.context for item in rounds])
    outcomes = np.asarray(
        [item.routing_outcome for item in rounds], dtype=np.int64
    )
    rows, summary, prediction_rows = fit_holdout_prompt_skylines(
        contexts,
        outcomes,
        validation_fraction=validation_fraction,
        seed=seed,
        outcome_source=outcome_source,
        hgb_max_leaf_nodes=hgb_max_leaf_nodes,
    )
    for row in prediction_rows:
        row["id"] = rounds[row["example_index"]].example_id
    return rows, summary, prediction_rows


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


def _plot(
    output: Path,
    online_rows: list[dict],
    skyline_rows: list[dict],
    outcome_source: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    if online_rows:
        methods = list(dict.fromkeys(row["method"] for row in online_rows))
        for method in methods:
            selected = [row for row in online_rows if row["method"] == method]
            selected.sort(key=lambda row: row["routing_rate"])
            axes[0].plot(
                [row["routing_rate"] for row in selected],
                [row["accuracy"] for row in selected],
                marker="o",
                label=method,
            )
        axes[0].set_title(
            "Online routing on synthetic outcomes"
            if outcome_source == "synthetic"
            else "Online routing on cached disagreement"
        )
        axes[0].legend(fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "Online experiment not requested", ha="center")

    if skyline_rows:
        present = {row["model"] for row in skyline_rows}
        models = [name for name in SKYLINE_PLOT_MODELS if name in present]
        for model in models:
            selected = [row for row in skyline_rows if row["model"] == model]
            selected.sort(key=lambda row: row["routing_rate"])
            axes[1].plot(
                [row["routing_rate"] for row in selected],
                [row["accuracy"] for row in selected],
                linewidth=2.5 if any(row.get("selected") for row in selected) else 1.2,
                label=model,
            )
        axes[1].set_title(
            "Supervised 80/20 synthetic skyline"
            if outcome_source == "synthetic"
            else "Supervised 80/20 real-label skyline"
        )
        axes[1].legend(fontsize=7)
    else:
        axes[1].text(0.5, 0.5, "Skyline experiment not requested", ha="center")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel(
            "Action-1 routing rate"
            if outcome_source == "synthetic"
            else "Strong-model routing rate"
        )
        axis.set_ylabel(
            "Synthetic routing accuracy"
            if outcome_source == "synthetic"
            else "Agreement with cached strong-model reference"
        )
        axis.grid(alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _bundle(output_dir: Path) -> Path:
    destination = output_dir / "simulation-results.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path != destination:
                archive.write(path, path.name)
    return destination


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if any(value < args.l11 for value in args.l01_values):
        raise SystemExit("Every l01 must be greater than or equal to l11")
    if args.random_repeats < 1:
        raise SystemExit("Random repeats must be positive")
    if any(value <= 0.0 for value in args.igw_gamma_values):
        raise SystemExit("Every IGW gamma value must be positive")
    if any(value < 2 for value in args.hgb_max_leaf_nodes):
        raise SystemExit("Every HGB max_leaf_nodes value must be at least two")
    if len(set(args.hgb_max_leaf_nodes)) != len(args.hgb_max_leaf_nodes):
        raise SystemExit("HGB max_leaf_nodes values must be unique")
    if not 0.0 < args.skyline_validation_fraction < 1.0:
        raise SystemExit(
            "Skyline validation fraction must be strictly between zero and one"
        )
    if min(
        args.etc_tastes,
        args.cbpside_tastes,
        args.cbpside_bootstrap_per_class,
        args.cbpside_bootstrap_max_tastes,
        args.igw_min_tastes,
        args.igw_bootstrap_per_class,
        args.igw_bootstrap_max_tastes,
    ) < 0:
        raise SystemExit("Taste and bootstrap settings must be nonnegative")

    cache = load_cache(args.cache)
    rounds, context_summary, teacher_summary, synthetic_rows = (
        _prompt_context_rounds(
            cache,
            args.prompt_components,
            args.limit,
            args.seed,
            args.outcome_source,
            args.context_profile,
        )
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if synthetic_rows:
        _write_csv(output / "synthetic_outcomes.csv", synthetic_rows)
    print(
        f"Loaded {len(rounds)} eligible rows; context dimension "
        f"{rounds[0].context.size}.",
        flush=True,
    )

    online_rows: list[dict] = []
    trajectories: list[dict] = []
    skyline_rows: list[dict] = []
    skyline_summary: dict = {}
    skyline_prediction_rows: list[dict] = []
    if args.experiment in {"all", "online"}:
        online_rows, trajectories = run_online(rounds, args)
        _write_csv(output / "online_results.csv", online_rows)
        (output / "online_results.json").write_text(
            json.dumps(_jsonable(online_rows), indent=2), encoding="utf-8"
        )
        with (output / "online_trajectories.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in trajectories:
                handle.write(json.dumps(_jsonable(row)) + "\n")
    if args.experiment in {"all", "skyline"}:
        skyline_rows, skyline_summary, skyline_prediction_rows = run_skyline(
            rounds,
            args.skyline_validation_fraction,
            args.seed,
            args.outcome_source,
            args.hgb_max_leaf_nodes,
        )
        _write_csv(output / "supervised_skyline.csv", skyline_rows)
        (output / "supervised_skyline.json").write_text(
            json.dumps(_jsonable(skyline_rows), indent=2), encoding="utf-8"
        )
        comparison = skyline_summary.get("model_comparison", [])
        _write_csv(output / "supervised_model_comparison.csv", comparison)
        (output / "supervised_model_comparison.json").write_text(
            json.dumps(_jsonable(comparison), indent=2), encoding="utf-8"
        )
        _write_csv(
            output / "supervised_holdout_predictions.csv",
            skyline_prediction_rows,
        )

    summary = {
        "cache": str(args.cache.resolve()),
        "cache_schema_version": cache.manifest["schema_version"],
        "examples": len(rounds),
        "context_dimension": int(rounds[0].context.size),
        "context": context_summary,
        "outcome_source": args.outcome_source,
        "routing_reference": (
            "synthetic_prompt_forest_outcome"
            if args.outcome_source == "synthetic"
            else "cached_strong_model_answer"
        ),
        "arc_gold_answers_used_as_routing_labels": False,
        "purpose": (
            "positive-control sanity check for routing implementations"
            if args.outcome_source == "synthetic"
            else "prompt-only routing study on cached weak/strong disagreement"
        ),
        "synthetic_teacher": teacher_summary,
        "loss_grid": [
            {
                "l01": float(value),
                "alpha": float(1.0 / (1.0 + value - args.l11)),
            }
            for value in args.l01_values
        ],
        "experiment": args.experiment,
        "parameters": vars(args)
        | {"cache": str(args.cache), "output_dir": str(args.output_dir)},
        "skyline": skyline_summary,
    }
    (output / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    _plot(
        output / "routing_comparison.png",
        online_rows,
        skyline_rows,
        args.outcome_source,
    )
    bundle = _bundle(output)
    print(f"Finished. Results: {bundle}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
