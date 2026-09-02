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
    IGWPlayer,
    LogCBPSideATConfig,
    LogCBPSideATPlayer,
    XGBoostETCPlayer,
)
from llm_routing_simulation.cache import RoutingCache, load_cache
from llm_routing_simulation.environment import LLMCascadeEnvironment
from llm_routing_simulation.skyline import (
    binary_residual_diagnostics,
    fit_supervised_skylines,
    plot_binary_residuals,
    random_routing_reference,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay online routing algorithms and supervised skylines offline."
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument(
        "--experiment", choices=("all", "online", "skyline"), default="all"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("simulation-results"))
    parser.add_argument("--limit", type=int, help="Optional prefix of eligible rows")
    parser.add_argument(
        "--pca-components",
        type=int,
        help="Use this many saved PCA axes; defaults to the collected context",
    )
    parser.add_argument(
        "--l01-values", type=float, nargs="+", default=[1.82, 2.22, 2.67, 3.33]
    )
    parser.add_argument("--l11", type=float, default=1.0)
    parser.add_argument("--etc-tastes", type=int, default=100)
    parser.add_argument("--cbpside-tastes", type=int, default=0)
    parser.add_argument("--cbpside-bootstrap-per-class", type=int, default=10)
    parser.add_argument("--cbpside-bootstrap-max-tastes", type=int, default=50)
    parser.add_argument("--igw-min-tastes", type=int, default=0)
    parser.add_argument("--igw-bootstrap-per-class", type=int, default=10)
    parser.add_argument("--igw-bootstrap-max-tastes", type=int, default=50)
    parser.add_argument("--igw-gamma", type=float, default=32.0)
    parser.add_argument("--igw-mu", type=float, default=2.0)
    parser.add_argument("--igw-min-propensity", type=float, default=0.1)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--skyline-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _eligible(cache: RoutingCache, pca_components: int | None, limit: int | None):
    rounds = cache.eligible_rounds(pca_components)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        rounds = rounds[:limit]
    if not rounds:
        raise ValueError("The cache has no eligible weak/strong answer pairs")
    return rounds


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

        # Evaluation can inspect the cached strong reference, but action 0 still
        # reveals no outcome to the player through EnvironmentTransition.
        strong_answer = rounds[transition.t - 1].strong_answer
        final_answer = strong_answer if transition.action == 1 else observation.weak_answer
        correct += int(final_answer == strong_answer)
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
            "metric": "agreement_with_strong_llm",
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
        [item.weak_answer != item.strong_answer for item in rounds], dtype=bool
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
        "metric": "agreement_with_strong_llm",
        "examples": len(rounds),
        "random_repeats": repeats,
    }


def run_online(rounds, args) -> tuple[list[dict], list[dict]]:
    """Run ETC, CBPSide, IGW gamma=32, and matched-random experiments."""
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
                XGBoostETCPlayer(
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


def run_skyline(rounds, requested_folds: int, seed: int):
    contexts = np.stack([item.context for item in rounds])
    outcomes = np.asarray(
        [item.weak_answer != item.strong_answer for item in rounds], dtype=np.int64
    )
    rows, summary, residual_probability_sets = fit_supervised_skylines(
        contexts, outcomes, seed=seed, requested_folds=requested_folds
    )
    rows.extend(random_routing_reference(float(np.mean(outcomes == 0))))
    residual_rows, residual_bin_rows = binary_residual_diagnostics(
        residual_probability_sets, outcomes
    )
    return rows, summary, residual_rows, residual_bin_rows


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
        axes[0].set_title("Online players")
        axes[0].legend(fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "Online experiment not requested", ha="center")

    if skyline_rows:
        models = list(dict.fromkeys(row["model"] for row in skyline_rows))
        for model in models:
            selected = [row for row in skyline_rows if row["model"] == model]
            selected.sort(key=lambda row: row["routing_rate"])
            axes[1].plot(
                [row["routing_rate"] for row in selected],
                [row["accuracy"] for row in selected],
                linewidth=2.5 if any(row.get("selected") for row in selected) else 1.2,
                label=model,
            )
        axes[1].set_title("Out-of-fold supervised skylines")
        axes[1].legend(fontsize=7)
    else:
        axes[1].text(0.5, 0.5, "Skyline experiment not requested", ha="center")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Strong-model routing rate")
        axis.set_ylabel("Agreement with strong-model reference")
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
    if args.random_repeats < 1 or args.skyline_folds < 2:
        raise SystemExit("Random repeats must be positive and skyline folds >= 2")
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
    rounds = _eligible(cache, args.pca_components, args.limit)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"Loaded {len(rounds)} eligible rows; context dimension "
        f"{rounds[0].context.size}.",
        flush=True,
    )

    online_rows: list[dict] = []
    trajectories: list[dict] = []
    skyline_rows: list[dict] = []
    skyline_summary: dict = {}
    residual_rows: list[dict] = []
    residual_bin_rows: list[dict] = []
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
        (
            skyline_rows,
            skyline_summary,
            residual_rows,
            residual_bin_rows,
        ) = run_skyline(
            rounds, args.skyline_folds, args.seed
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
        _write_csv(output / "supervised_residuals.csv", residual_rows)
        _write_csv(output / "supervised_residual_bins.csv", residual_bin_rows)
        plot_binary_residuals(
            output / "supervised_residual_plots.png",
            residual_rows,
            residual_bin_rows,
        )

    summary = {
        "cache": str(args.cache.resolve()),
        "cache_schema_version": cache.manifest["schema_version"],
        "examples": len(rounds),
        "context_dimension": int(rounds[0].context.size),
        "routing_reference": "strong_model_answer",
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
