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
    HGBETCPlayer,
    IGWPlayer,
    LogCBPSideATConfig,
    LogCBPSideATPlayer,
)
from llm_routing_simulation.cache import RoutingCache, load_cache
from llm_routing_simulation.environment import LLMCascadeEnvironment
from llm_routing_simulation.skyline import (
    fit_holdout_prompt_skylines,
    random_routing_reference,
)
from llm_routing_simulation.synthetic_prompt import (
    generate_synthetic_prompt_outcomes,
)


SKYLINE_PLOT_MODELS = (
    "Logistic (80/20 holdout)",
    "HGB (80/20 holdout)",
    "Random routing (expected)",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay online routing algorithms and supervised skylines offline."
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--prompt-components", type=int, default=64)
    parser.add_argument(
        "--experiment", choices=("all", "online", "skyline"), default="all"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("simulation-results"))
    parser.add_argument("--limit", type=int, help="Optional prefix of eligible rows")
    parser.add_argument(
        "--l01-values", type=float, nargs="+", default=[1.82, 2.22, 2.67, 3.33]
    )
    parser.add_argument("--l11", type=float, default=1.0)
    parser.add_argument("--etc-tastes", type=int, default=300)
    parser.add_argument("--cbpside-tastes", type=int, default=0)
    parser.add_argument("--cbpside-bootstrap-per-class", type=int, default=0)
    parser.add_argument("--cbpside-bootstrap-max-tastes", type=int, default=0)
    parser.add_argument("--igw-min-tastes", type=int, default=0)
    parser.add_argument("--igw-bootstrap-per-class", type=int, default=0)
    parser.add_argument("--igw-bootstrap-max-tastes", type=int, default=0)
    parser.add_argument("--igw-gamma", type=float, default=16.0)
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
):
    """Attach prompt contexts and one frozen synthetic outcome to every round."""
    all_rounds = cache.eligible_rounds()
    all_contexts, block = cache.context_block(
        "prompt_embedding_pca", components=prompt_components
    )
    eligible_contexts = all_contexts[cache.eligible_indices]
    teacher_probabilities, synthetic_outcomes, teacher_summary = (
        generate_synthetic_prompt_outcomes(eligible_contexts, seed=seed)
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        all_rounds = all_rounds[:limit]
        eligible_contexts = eligible_contexts[:limit]
        teacher_probabilities = teacher_probabilities[:limit]
        synthetic_outcomes = synthetic_outcomes[:limit]
    if not all_rounds:
        raise ValueError("The cache has no eligible weak/strong answer pairs")
    prompt_rounds = [
        replace(
            item,
            context=eligible_contexts[index].copy(),
            outcome_override=int(synthetic_outcomes[index]),
        )
        for index, item in enumerate(all_rounds)
    ]
    context_summary = {
        "source": cache.manifest.get("prompt_embedding_definition"),
        "context_block": block,
        "context_dimension": int(eligible_contexts.shape[1]),
        "pca_scope": cache.manifest.get("pca_scope"),
        "fit_scope": "precomputed across all collected prompts",
        "outcome_or_answer_features_used": False,
    }
    teacher_summary["probabilities_stored_in_player_observations"] = False
    teacher_summary["selected_examples"] = len(prompt_rounds)
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
        for index, item in enumerate(prompt_rounds)
    ]
    return prompt_rounds, context_summary, teacher_summary, synthetic_rows


def _run_one_player(method: str, player, config, rounds, progress_label: str):
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

        # Evaluation can inspect the synthetic outcome after acting, while the
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
                "evaluation_only_synthetic_outcome": evaluation_outcome,
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
            "metric": "synthetic_routing_accuracy",
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
        },
        trajectories,
    )


def _random_matched(
    rounds, target_rate: float, l01: float, l11: float, repeats: int, seed: int
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
        "method": "Random (matched ETC)",
        "l01": l01,
        "l11": l11,
        "alpha": 1.0 / (1.0 + l01 - l11),
        "routing_rate": float(np.mean(rates)),
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "metric": "synthetic_routing_accuracy",
        "examples": len(rounds),
        "random_repeats": repeats,
    }


def run_online(rounds, args) -> tuple[list[dict], list[dict]]:
    """Run HGB ETC/IGW, linear-logistic CBPSide, and matched random."""
    context_dim = int(rounds[0].context.size)
    rows, trajectories = [], []
    for loss_index, l01 in enumerate(args.l01_values):
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
        players = [
            (
                "ETC",
                HGBETCPlayer(
                    context_dim, base, seed=args.seed + loss_index
                ),
                base,
            ),
            ("CBPSide", LogCBPSideATPlayer(context_dim, cbpside), cbpside),
            (
                f"IGW gamma={args.igw_gamma:g}",
                IGWPlayer(
                    context_dim,
                    base,
                    len(rounds),
                    min_tastes=args.igw_min_tastes,
                    bootstrap_per_class=args.igw_bootstrap_per_class,
                    bootstrap_max_tastes=args.igw_bootstrap_max_tastes,
                    mu=args.igw_mu,
                    fixed_gamma=args.igw_gamma,
                    min_propensity=args.igw_min_propensity,
                    seed=args.seed + loss_index,
                ),
                base,
            ),
        ]
        etc_row = None
        for method, player, config in players:
            result, path = _run_one_player(
                method, player, config, rounds, f"l01={l01:g} {method}"
            )
            rows.append(result)
            trajectories.extend(path)
            if method == "ETC":
                etc_row = result
        rows.append(
            _random_matched(
                rounds,
                etc_row["routing_rate"],
                l01,
                args.l11,
                args.random_repeats,
                args.seed + 10_000 + loss_index,
            )
        )
    return rows, trajectories


def run_skyline(
    rounds,
    validation_fraction: float,
    seed: int,
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
    )
    rows.extend(
        random_routing_reference(summary["validation_agreement_rate"])
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


def _plot(output: Path, online_rows: list[dict], skyline_rows: list[dict]) -> None:
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
        axes[0].set_title("Online routing on synthetic outcomes")
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
        axes[1].set_title("Supervised 80/20 synthetic skyline")
        axes[1].legend(fontsize=7)
    else:
        axes[1].text(0.5, 0.5, "Skyline experiment not requested", ha="center")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Action-1 routing rate")
        axis.set_ylabel("Synthetic routing accuracy")
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
        )
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
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
        "routing_reference": "synthetic_prompt_forest_outcome",
        "purpose": "positive-control sanity check for routing implementations",
        "synthetic_teacher": teacher_summary,
        "experiment": args.experiment,
        "parameters": vars(args) | {"cache": str(args.cache), "output_dir": str(args.output_dir)},
        "skyline": skyline_summary,
    }
    (output / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    _plot(output / "routing_comparison.png", online_rows, skyline_rows)
    bundle = _bundle(output)
    print(f"Finished. Results: {bundle}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
