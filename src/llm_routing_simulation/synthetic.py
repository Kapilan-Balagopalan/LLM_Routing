"""Nonlinear synthetic sanity check for partial-feedback routing algorithms."""

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
    RevealedFeedbackEstimator,
    XGBoostETCPlayer,
)
from llm_routing_simulation.skyline import (
    random_routing_reference,
    threshold_skyline,
)


SMALL_HGB_PROFILE = {
    "name": "Small HGB",
    "learning_rate": 0.08,
    "max_iter": 30,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
}

# Ten approximately uniform decision thresholds across (0, 1). With l11=1,
# alpha = 1 / l01, so these loss values correspond to alpha=0.95,...,0.05.
SYNTHETIC_L01_VALUES = [
    1.052632,
    1.176471,
    1.333333,
    1.538462,
    1.818182,
    2.222222,
    2.857143,
    4.0,
    6.666667,
    20.0,
]


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Numerically stable logistic link for the ground-truth probability."""
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def generate_nonlinear_data(
    sample_count: int,
    *,
    seed: int,
    feature_bound: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate X and Y where P(Y=1|X)=sigmoid(X_1 X_2)."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if feature_bound <= 0:
        raise ValueError("feature_bound must be positive")
    rng = np.random.default_rng(seed)
    contexts = rng.uniform(
        -feature_bound, feature_bound, size=(sample_count, 2)
    )
    probabilities = sigmoid(contexts[:, 0] * contexts[:, 1])
    outcomes = (rng.random(sample_count) < probabilities).astype(np.int64)
    return contexts, outcomes, probabilities


class SmallHGBEstimator(RevealedFeedbackEstimator):
    """Small histogram-gradient-boosting disagreement estimator."""

    @property
    def estimator_name(self) -> str:
        return "small_hgb"

    def _new_model(self):
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError as exc:
            raise RuntimeError(
                "The synthetic experiment requires scikit-learn"
            ) from exc
        config = {
            key: value
            for key, value in SMALL_HGB_PROFILE.items()
            if key != "name"
        }
        return HistGradientBoostingClassifier(
            **config,
            loss="log_loss",
            early_stopping=False,
            random_state=self.seed,
        )


class SmallHGBETCPlayer(XGBoostETCPlayer):
    """Explore-then-commit with one frozen small-HGB fit."""

    def __init__(
        self,
        context_dim: int,
        config: LogCBPSideATConfig,
        *,
        seed: int,
    ) -> None:
        super().__init__(context_dim, config, seed=seed)
        self.estimator = SmallHGBEstimator(context_dim, seed=seed)


class SmallHGBIGWPlayer(IGWPlayer):
    """IGW with a small HGB refitted from revealed, weighted feedback."""

    def __init__(
        self,
        context_dim: int,
        config: LogCBPSideATConfig,
        total_samples: int,
        *,
        min_tastes: int,
        bootstrap_per_class: int,
        bootstrap_max_tastes: int,
        mu: float,
        fixed_gamma: float,
        min_propensity: float,
        seed: int,
    ) -> None:
        super().__init__(
            context_dim,
            config,
            total_samples,
            min_tastes=min_tastes,
            bootstrap_per_class=bootstrap_per_class,
            bootstrap_max_tastes=bootstrap_max_tastes,
            mu=mu,
            fixed_gamma=fixed_gamma,
            min_propensity=min_propensity,
            seed=seed,
        )
        self.estimator = SmallHGBEstimator(context_dim, seed=seed)


def fit_supervised_synthetic_skylines(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    oracle_probabilities: np.ndarray,
    *,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Fit logistic and small HGB on training data and score fresh test data."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "The synthetic experiment requires scikit-learn"
        ) from exc

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        ),
    )
    logistic.fit(train_x, train_y)
    logistic_probabilities = logistic.predict_proba(test_x)[:, 1]

    hgb_config = {
        key: value
        for key, value in SMALL_HGB_PROFILE.items()
        if key != "name"
    }
    hgb = HistGradientBoostingClassifier(
        **hgb_config,
        loss="log_loss",
        early_stopping=False,
        random_state=seed,
    )
    hgb.fit(train_x, train_y)
    hgb_probabilities = hgb.predict_proba(test_x)[:, 1]

    probability_sets = {
        "Linear logistic (held-out)": logistic_probabilities,
        "Small HGB (held-out)": hgb_probabilities,
        "Bayes oracle sigma(x1*x2)": oracle_probabilities,
    }
    rows: list[dict] = []
    comparison: list[dict] = []
    for name, probabilities in probability_sets.items():
        auc = float(roc_auc_score(test_y, probabilities))
        configuration = (
            SMALL_HGB_PROFILE
            if name == "Small HGB (held-out)"
            else {"features": "x1,x2", "linear_predictor": True}
            if name == "Linear logistic (held-out)"
            else {"known_probability": "sigmoid(x1*x2)"}
        )
        rows.extend(
            threshold_skyline(
                probabilities,
                test_y,
                name,
                fit_scope=(
                    "known_data_generating_probability"
                    if name.startswith("Bayes")
                    else "independent_train_test"
                ),
                evaluation_auc=auc,
                configuration=configuration,
                selected=name == "Small HGB (held-out)",
            )
        )
        clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
        comparison.append(
            {
                "model": name,
                "test_auc": auc,
                "test_log_loss": float(log_loss(test_y, clipped)),
                "test_brier_score": float(
                    brier_score_loss(test_y, probabilities)
                ),
                "train_samples": int(len(train_y)),
                "test_samples": int(len(test_y)),
                "configuration": configuration,
            }
        )
    rows.extend(random_routing_reference(float(np.mean(test_y == 0))))
    return rows, comparison


def _run_player(
    method: str,
    player,
    contexts: np.ndarray,
    outcomes: np.ndarray,
    config: LogCBPSideATConfig,
) -> tuple[dict, list[dict]]:
    correct = 0
    routed = 0
    trajectory: list[dict] = []
    progress_every = max(1, len(outcomes) // 10)
    for index, (context, true_outcome) in enumerate(
        zip(contexts, outcomes), start=1
    ):
        wrapper = player.next_action(context)
        action = wrapper.action
        revealed = int(true_outcome) if action == 1 else None
        player.update(action, context, revealed)
        correct += int(action == 1 or true_outcome == 0)
        routed += action
        diagnostics = wrapper.diagnostics
        trajectory.append(
            {
                "method": method,
                "l01": config.loss_reject_disagreement,
                "l11": config.loss_route_disagreement,
                "alpha": getattr(diagnostics, "threshold", None),
                "t": index,
                "x1": float(context[0]),
                "x2": float(context[1]),
                "outcome": int(true_outcome),
                "action": action,
                "feedback_revealed_to_player": revealed,
                "predicted_disagreement": getattr(
                    diagnostics, "predicted_disagreement", None
                ),
                "reason": getattr(diagnostics, "reason", None),
                "estimator": getattr(
                    diagnostics, "estimator", "linear_logistic"
                ),
                "training_count": getattr(
                    diagnostics, "training_count", None
                ),
                "probability_1": getattr(
                    diagnostics, "probability_1", None
                ),
            }
        )
        if index % progress_every == 0 or index == len(outcomes):
            print(f"[{method}, l01={config.loss_reject_disagreement:g}] "
                  f"{index}/{len(outcomes)}", flush=True)

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
            "routing_rate": routed / len(outcomes),
            "accuracy": correct / len(outcomes),
            "metric": "agreement_with_synthetic_strong_reference",
            "examples": len(outcomes),
            "min_tastes": getattr(player, "min_tastes", config.min_tastes),
            "bootstrap_per_class": getattr(
                player, "bootstrap_per_class", None
            ),
            "bootstrap_max_tastes": getattr(
                player, "bootstrap_max_tastes", None
            ),
            "probability_estimator": getattr(
                getattr(player, "estimator", None),
                "estimator_name",
                "linear_logistic",
            ),
        },
        trajectory,
    )


def _random_matched(
    outcomes: np.ndarray,
    *,
    target_rate: float,
    l01: float,
    l11: float,
    repeats: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    rates: list[float] = []
    accuracies: list[float] = []
    for _ in range(repeats):
        routed = rng.random(len(outcomes)) < target_rate
        rates.append(float(np.mean(routed)))
        accuracies.append(float(np.mean(routed | (outcomes == 0))))
    return {
        "method": "Random (matched ETC)",
        "l01": l01,
        "l11": l11,
        "alpha": 1.0 / (1.0 + l01 - l11),
        "routing_rate": float(np.mean(rates)),
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "metric": "agreement_with_synthetic_strong_reference",
        "examples": len(outcomes),
        "random_repeats": repeats,
    }


def run_online_synthetic(
    contexts: np.ndarray,
    outcomes: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    """Run HGB ETC/IGW, logistic CBPSide, and ETC-matched random."""
    rows: list[dict] = []
    trajectories: list[dict] = []
    for loss_index, l01 in enumerate(args.l01_values):
        base = LogCBPSideATConfig(
            loss_reject_disagreement=l01,
            loss_route_disagreement=args.l11,
            min_tastes=args.etc_tastes,
            use_confidence_bound=False,
        )
        cbpside_config = replace(
            base,
            min_tastes=args.cbpside_tastes,
            bootstrap_per_class=args.bootstrap_per_class,
            bootstrap_max_tastes=args.bootstrap_max_tastes,
            use_confidence_bound=True,
        )
        players = (
            (
                "ETC (small HGB)",
                SmallHGBETCPlayer(
                    2, base, seed=args.seed + 100 + loss_index
                ),
                base,
            ),
            (
                "CBPSide (linear logistic)",
                LogCBPSideATPlayer(2, cbpside_config),
                cbpside_config,
            ),
            (
                "IGW (small HGB)",
                SmallHGBIGWPlayer(
                    2,
                    base,
                    len(outcomes),
                    min_tastes=args.igw_min_tastes,
                    bootstrap_per_class=args.bootstrap_per_class,
                    bootstrap_max_tastes=args.bootstrap_max_tastes,
                    mu=args.igw_mu,
                    fixed_gamma=args.igw_gamma,
                    min_propensity=args.igw_min_propensity,
                    seed=args.seed + 200 + loss_index,
                ),
                base,
            ),
        )
        etc_result = None
        for method, player, config in players:
            result, trajectory = _run_player(
                method, player, contexts, outcomes, config
            )
            rows.append(result)
            trajectories.extend(trajectory)
            if method.startswith("ETC"):
                etc_result = result
        assert etc_result is not None
        rows.append(
            _random_matched(
                outcomes,
                target_rate=etc_result["routing_rate"],
                l01=l01,
                l11=args.l11,
                repeats=args.random_repeats,
                seed=args.seed + 10_000 + loss_index,
            )
        )
    return rows, trajectories


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
    if isinstance(value, Path):
        return str(value)
    return value


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
    skyline_rows: list[dict],
    online_rows: list[dict],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for model in dict.fromkeys(row["model"] for row in skyline_rows):
        selected = [row for row in skyline_rows if row["model"] == model]
        selected.sort(key=lambda row: row["routing_rate"])
        axes[0].plot(
            [row["routing_rate"] for row in selected],
            [row["accuracy"] for row in selected],
            linewidth=2.2 if "HGB" in model or "oracle" in model else 1.5,
            label=model,
        )
    axes[0].set_title("Held-out supervised skyline")

    for method in dict.fromkeys(row["method"] for row in online_rows):
        selected = [row for row in online_rows if row["method"] == method]
        selected.sort(key=lambda row: row["routing_rate"])
        axes[1].plot(
            [row["routing_rate"] for row in selected],
            [row["accuracy"] for row in selected],
            marker="o",
            label=method,
        )
    axes[1].set_title("Online partial-feedback routing")

    for axis in axes:
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Action-1 routing rate")
        axis.set_ylabel("Accuracy vs. strong reference")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _bundle(output_dir: Path) -> Path:
    destination = output_dir / "synthetic-results.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path != destination:
                archive.write(path, path.name)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a nonlinear sanity check with P(Y=1|X)=sigmoid(x1*x2)."
        )
    )
    parser.add_argument("--train-samples", type=int, default=3000)
    parser.add_argument("--online-samples", type=int, default=2000)
    parser.add_argument("--feature-bound", type=float, default=3.0)
    parser.add_argument(
        "--l01-values",
        type=float,
        nargs="+",
        default=SYNTHETIC_L01_VALUES,
    )
    parser.add_argument("--l11", type=float, default=1.0)
    parser.add_argument("--etc-tastes", type=int, default=500)
    parser.add_argument("--cbpside-tastes", type=int, default=0)
    parser.add_argument("--igw-min-tastes", type=int, default=0)
    parser.add_argument("--bootstrap-per-class", type=int, default=0)
    parser.add_argument("--bootstrap-max-tastes", type=int, default=0)
    parser.add_argument("--igw-gamma", type=float, default=32.0)
    parser.add_argument("--igw-mu", type=float, default=2.0)
    parser.add_argument("--igw-min-propensity", type=float, default=0.1)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("synthetic-nonlinear-results")
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.train_samples < 2 or args.online_samples < 2:
        raise SystemExit("Training and online sample counts must be at least two")
    if args.feature_bound <= 0:
        raise SystemExit("--feature-bound must be positive")
    if any(value < args.l11 for value in args.l01_values):
        raise SystemExit("Every l01 must be greater than or equal to l11")
    if min(
        args.etc_tastes,
        args.cbpside_tastes,
        args.igw_min_tastes,
        args.bootstrap_per_class,
        args.bootstrap_max_tastes,
    ) < 0:
        raise SystemExit("Taste settings must be nonnegative")
    if args.random_repeats < 1:
        raise SystemExit("--random-repeats must be positive")

    train_x, train_y, train_probabilities = generate_nonlinear_data(
        args.train_samples,
        seed=args.seed,
        feature_bound=args.feature_bound,
    )
    online_x, online_y, online_probabilities = generate_nonlinear_data(
        args.online_samples,
        seed=args.seed + 1,
        feature_bound=args.feature_bound,
    )
    print(
        f"Generated {len(train_y)} training and {len(online_y)} independent "
        "online/test samples.",
        flush=True,
    )

    skyline_rows, model_comparison = fit_supervised_synthetic_skylines(
        train_x,
        train_y,
        online_x,
        online_y,
        online_probabilities,
        seed=args.seed,
    )
    online_rows, trajectories = run_online_synthetic(
        online_x, online_y, args
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "synthetic_data.npz",
        train_contexts=train_x,
        train_outcomes=train_y,
        train_true_probabilities=train_probabilities,
        online_contexts=online_x,
        online_outcomes=online_y,
        online_true_probabilities=online_probabilities,
    )
    _write_csv(output / "supervised_skyline.csv", skyline_rows)
    _write_csv(output / "supervised_model_comparison.csv", model_comparison)
    _write_csv(output / "online_results.csv", online_rows)
    (output / "supervised_model_comparison.json").write_text(
        json.dumps(_jsonable(model_comparison), indent=2), encoding="utf-8"
    )
    (output / "online_results.json").write_text(
        json.dumps(_jsonable(online_rows), indent=2), encoding="utf-8"
    )
    with (output / "online_trajectories.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in trajectories:
            handle.write(json.dumps(_jsonable(row)) + "\n")
    summary = {
        "data_generating_process": "P(Y=1|X)=sigmoid(x1*x2)",
        "feature_distribution": (
            f"independent Uniform(-{args.feature_bound:g}, "
            f"{args.feature_bound:g})"
        ),
        "outcome_interpretation": "1=weak/strong disagreement",
        "parameters": vars(args),
        "small_hgb_profile": SMALL_HGB_PROFILE,
        "training_disagreement_rate": float(np.mean(train_y)),
        "online_disagreement_rate": float(np.mean(online_y)),
    }
    (output / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    _plot(output / "synthetic_routing_comparison.png", skyline_rows, online_rows)
    bundle = _bundle(output)
    print("Supervised comparison:", flush=True)
    for row in model_comparison:
        print(
            f"  {row['model']}: AUC={row['test_auc']:.4f}, "
            f"log-loss={row['test_log_loss']:.4f}",
            flush=True,
        )
    print(f"Finished. Results: {bundle}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
