"""Resumable pointwise multiplier tuning for the BoolQ online study.

This entry point is intentionally separate from :mod:`run`.  The established
single-configuration simulator stays unchanged, while this module owns the much
larger paired-order tuning design and its candidate-level checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from llm_routing_simulation.algorithm import (
    ONLINE_HGB_PROFILE,
    LogCBPSideAT,
    LogCBPSideATConfig,
)
from llm_routing_simulation.cache import load_cache
from llm_routing_simulation.online_tree import (
    ONLINE_LOGISTIC_PROFILE,
    make_tree_backend as _make_tree_backend_impl,
)
from llm_routing_simulation.run import (
    DEFAULT_L01_VALUES,
    _jsonable,
    _prompt_context_rounds,
    _realized_cost_metrics,
    _write_csv,
)


DEFAULT_MULTIPLIERS = (0.1, 0.3, 1.0, 3.0, 10.0)
TUNING_IMPLEMENTATION_REVISION = 3
POLICY_CBPSIDE = "CBPSide"
POLICY_ETC = "ETC"
POLICY_IGW_TREE = "IGW"
POLICY_IGW_LINEAR = "IGWLinear"
TUNED_POLICIES = (
    POLICY_CBPSIDE,
    POLICY_ETC,
    POLICY_IGW_LINEAR,
    POLICY_IGW_TREE,
)
AGGREGATE_METRICS = (
    "routing_rate",
    "accuracy",
    "routed_to_strong",
    "unrouted_disagreements",
    "realized_cost_per_example",
    "realized_total_cost",
    "model_updates",
    "last_model_training_count",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune CBPSide beta, IGW gamma, and ETC tastes pointwise over "
            "paired shuffled online orders."
        )
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--context-profile",
        choices=("prompt-only", "uncertainty-prompt", "non-prompt", "all-features"),
        default="all-features",
    )
    parser.add_argument("--prompt-components", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--l01-values", type=float, nargs="+", default=list(DEFAULT_L01_VALUES)
    )
    parser.add_argument("--l11", type=float, default=1.0)
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=list(DEFAULT_MULTIPLIERS),
    )
    parser.add_argument("--online-order-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=0,
        help="Fixed policy seed shared by every order and multiplier candidate",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild tables, selections, plots, and ZIP from complete checkpoints",
    )

    parser.add_argument("--cbpside-base-beta-scale", type=float, default=0.5)
    parser.add_argument("--cbpside-max-confidence-radius", type=float, default=0.5)
    parser.add_argument("--cbpside-matrix-regularization", type=float, default=1.0)
    parser.add_argument("--cbpside-theta-regularization", type=float, default=1.0)
    parser.add_argument(
        "--cbpside-zero-start",
        action="store_true",
        help=(
            "Start each epoch's Newton solve at zero instead of the preceding "
            "solution; slower but matches the legacy finite-iteration path"
        ),
    )

    parser.add_argument(
        "--igw-base-gamma",
        type=float,
        help="Base gamma; default is sqrt(the selected online horizon)",
    )
    parser.add_argument("--igw-mu", type=float, default=2.0)
    parser.add_argument("--igw-min-propensity", type=float, default=0.1)
    parser.add_argument(
        "--etc-base-tastes",
        type=float,
        help="Base forced tastes; default is n^(2/3) before multiplier and ceiling",
    )

    parser.add_argument(
        "--tree-estimator",
        choices=("hgb", "river-hoeffding"),
        default="hgb",
        help=(
            "Use the established HGB oracle or an explicitly different weighted "
            "incremental Hoeffding tree for ETC and IGW Tree; IGW Linear is "
            "always evaluated with weighted logistic regression"
        ),
    )
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=15)
    parser.add_argument("--river-max-depth", type=int, default=4)
    parser.add_argument("--river-grace-period", type=int, default=200)
    parser.add_argument("--river-delta", type=float, default=1e-7)
    parser.add_argument("--river-tau", type=float, default=0.05)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.online_order_repeats < 2:
        raise SystemExit("Multiplier tuning requires at least two shuffled orders")
    if args.jobs == 0:
        raise SystemExit("--jobs must be positive or -1")
    if args.jobs < -1:
        raise SystemExit("--jobs must be positive or -1")
    if not args.l01_values or any(
        not math.isfinite(value) or value < args.l11
        for value in args.l01_values
    ):
        raise SystemExit("Every l01 must be greater than or equal to l11")
    if args.l11 != 1.0:
        raise SystemExit(
            "This tuning study fixes l11=1: every strong-model route costs one"
        )
    if len(set(args.l01_values)) != len(args.l01_values):
        raise SystemExit("l01 values must be unique")
    if not args.multipliers or any(
        not math.isfinite(value) or value <= 0.0
        for value in args.multipliers
    ):
        raise SystemExit("Every multiplier must be positive")
    if len(set(args.multipliers)) != len(args.multipliers):
        raise SystemExit("Multipliers must be unique")
    if not math.isfinite(args.cbpside_base_beta_scale) or (
        args.cbpside_base_beta_scale <= 0.0
    ):
        raise SystemExit("The CBPSide base beta scale must be positive")
    if not math.isfinite(args.cbpside_max_confidence_radius) or (
        args.cbpside_max_confidence_radius <= 0.0
    ):
        raise SystemExit("The CBPSide confidence cap must be positive")
    if not math.isfinite(args.cbpside_matrix_regularization) or (
        args.cbpside_matrix_regularization <= 0.0
    ):
        raise SystemExit("The CBPSide matrix regularization must be positive")
    if not math.isfinite(args.cbpside_theta_regularization) or (
        args.cbpside_theta_regularization < 0.0
    ):
        raise SystemExit("The CBPSide theta regularization must be nonnegative")
    if args.igw_base_gamma is not None and (
        not math.isfinite(args.igw_base_gamma) or args.igw_base_gamma <= 0.0
    ):
        raise SystemExit("The IGW base gamma must be positive")
    if not math.isfinite(args.igw_mu) or args.igw_mu < 2.0:
        raise SystemExit("IGW mu must be at least two")
    if not 0.0 < args.igw_min_propensity <= 1.0:
        raise SystemExit("IGW minimum propensity must be in (0, 1]")
    if args.etc_base_tastes is not None and (
        not math.isfinite(args.etc_base_tastes)
        or args.etc_base_tastes <= 0.0
    ):
        raise SystemExit("The ETC base taste budget must be positive")
    if args.hgb_max_leaf_nodes != 15:
        raise SystemExit(
            "This tuning study is fixed to the agreed 15-leaf HGB profile"
        )
    if args.river_max_depth < 1 or args.river_grace_period < 1:
        raise SystemExit("River depth and grace period must be positive")
    if not 0.0 < args.river_delta < 1.0:
        raise SystemExit("River delta must be in (0, 1)")
    if not math.isfinite(args.river_tau) or not 0.0 <= args.river_tau <= 1.0:
        raise SystemExit("River tau must be in [0, 1]")


def _doubling_epochs(total_samples: int) -> Iterable[tuple[int, int, int]]:
    """Yield `(boundary_round, start_index, stop_index)` for 1,2,4,8,... ."""
    boundary = 1
    while boundary <= total_samples:
        yield boundary, boundary - 1, min(total_samples, 2 * boundary - 1)
        boundary *= 2


def _normalized_cbpside_features(contexts: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(contexts, axis=1)
    denominators = np.maximum(1.0, norms)
    features = np.empty(
        (contexts.shape[0], contexts.shape[1] + 1), dtype=np.float64
    )
    features[:, 0] = 1.0
    features[:, 1:] = contexts / denominators[:, None]
    return features


def _fit_logistic_theta(
    features: np.ndarray,
    outcomes: np.ndarray,
    config: LogCBPSideATConfig,
    initial_theta: np.ndarray | None,
) -> np.ndarray:
    """Match the CBPSide Newton solve, optionally warm-started across epochs."""
    dimension = features.shape[1]
    theta = (
        np.zeros(dimension, dtype=np.float64)
        if initial_theta is None
        else np.asarray(initial_theta, dtype=np.float64).copy()
    )
    if features.shape[0] == 0:
        return theta

    regularizer = np.eye(dimension, dtype=np.float64)
    regularizer[0, 0] = 0.0
    theta_bound = config.theta_norm_bound or config.c_max
    y = np.asarray(outcomes, dtype=np.float64)
    for _ in range(config.max_newton_steps):
        probabilities = np.asarray(LogCBPSideAT.sigmoid(features @ theta))
        gradient = (
            features.T @ (y - probabilities)
            - config.theta_regularization * (regularizer @ theta)
        )
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
        information = (
            features.T @ (weights[:, None] * features)
            + config.theta_regularization * regularizer
        )
        proposal = theta + np.linalg.lstsq(information, gradient, rcond=None)[0]
        slope_norm = float(np.linalg.norm(proposal[1:]))
        if slope_norm > theta_bound:
            proposal[1:] *= theta_bound / slope_norm
        proposal[0] = np.clip(proposal[0], -config.c_max, config.c_max)
        change = proposal - theta
        theta = proposal
        if np.linalg.norm(change) <= config.tolerance * (
            1.0 + np.linalg.norm(theta)
        ):
            break
    return theta


def _tree_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "kind": args.tree_estimator,
        "hgb_max_leaf_nodes": args.hgb_max_leaf_nodes,
        "river_max_depth": args.river_max_depth,
        "river_grace_period": args.river_grace_period,
        "river_delta": args.river_delta,
        "river_tau": args.river_tau,
    }


def _linear_settings() -> dict[str, Any]:
    """Return the fixed linear oracle used only by the IGW comparison."""
    return {"kind": "logistic", **ONLINE_LOGISTIC_PROFILE}


def make_tree_backend(
    settings: dict[str, Any], *, seed: int
):
    """Translate the sweep's recorded settings to the backend factory API."""
    if settings["kind"] == "hgb":
        return _make_tree_backend_impl("hgb", seed=seed)
    if settings["kind"] == "logistic":
        return _make_tree_backend_impl("logistic", seed=seed)
    return _make_tree_backend_impl(
        "river-hoeffding",
        seed=seed,
        max_depth=int(settings["river_max_depth"]),
        grace_period=int(settings["river_grace_period"]),
        delta=float(settings["river_delta"]),
        tau=float(settings["river_tau"]),
    )


def _method_label(policy: str, tree_settings: dict[str, Any]) -> str:
    if policy == POLICY_CBPSIDE:
        return "CBPSide (linear logistic)"
    if policy == POLICY_IGW_LINEAR:
        return "IGW + linear logistic"
    if tree_settings["kind"] == "hgb":
        estimator = f"HGB leaves={tree_settings['hgb_max_leaf_nodes']}"
    else:
        estimator = f"Hoeffding tree depth<={tree_settings['river_max_depth']}"
    if policy == POLICY_IGW_TREE:
        return f"IGW + {estimator}"
    return f"{policy} ({estimator})"


def _base_row(
    *,
    policy: str,
    l01: float,
    l11: float,
    multiplier: float,
    parameter_name: str,
    base_parameter: float,
    effective_parameter: float,
    order_index: int,
    order_seed: int,
    policy_seed: int,
    examples: int,
    tree_settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "policy": policy,
        "method": _method_label(policy, tree_settings),
        "l01": float(l01),
        "l11": float(l11),
        "alpha": float(1.0 / (1.0 + l01 - l11)),
        "multiplier": float(multiplier),
        "parameter_name": parameter_name,
        "base_parameter": float(base_parameter),
        "effective_parameter": float(effective_parameter),
        "order_run": order_index + 1,
        "order_seed": int(order_seed),
        "policy_seed": int(policy_seed),
        "order_was_shuffled": True,
        "examples": int(examples),
        "update_schedule": "global_round_doubling_before_action",
        "probability_estimator": (
            "linear-logistic"
            if policy in {POLICY_CBPSIDE, POLICY_IGW_LINEAR}
            else tree_settings["kind"]
        ),
        "tree_estimator": (
            tree_settings["kind"]
            if tree_settings["kind"] in {"hgb", "river-hoeffding"}
            else None
        ),
    }


def _finish_row(
    row: dict[str, Any],
    *,
    routed: int,
    correct: int,
    model_updates: int,
    last_training_count: int,
) -> dict[str, Any]:
    examples = int(row["examples"])
    row.update(
        {
            "routing_rate": float(routed / examples),
            "accuracy": float(correct / examples),
            "model_updates": int(model_updates),
            "last_model_training_count": int(last_training_count),
        }
    )
    row.update(
        _realized_cost_metrics(
            l01=float(row["l01"]),
            l11=float(row["l11"]),
            routing_rate=float(row["routing_rate"]),
            accuracy=float(row["accuracy"]),
            examples=examples,
        )
    )
    return row


def _simulate_cbpside_candidate(
    normalized_features: np.ndarray,
    outcomes: np.ndarray,
    permutation: np.ndarray,
    *,
    l01: float,
    l11: float,
    multiplier: float,
    base_beta_scale: float,
    confidence_cap: float,
    matrix_regularization: float,
    theta_regularization: float,
    zero_start: bool,
    order_index: int,
    order_seed: int,
    policy_seed: int,
    tree_settings: dict[str, Any],
) -> dict[str, Any]:
    """Run strict doubling epochs with beta evaluated for every context."""
    x_all = normalized_features[permutation]
    y_all = outcomes[permutation]
    n, dimension = x_all.shape
    beta_scale = base_beta_scale * multiplier
    config = LogCBPSideATConfig(
        matrix_regularization=matrix_regularization,
        loss_reject_disagreement=l01,
        loss_route_disagreement=l11,
        beta_scale=beta_scale,
        max_confidence_radius=confidence_cap,
        theta_regularization=theta_regularization,
        min_tastes=0,
        bootstrap_per_class=0,
        bootstrap_max_tastes=0,
        use_confidence_bound=True,
    )
    threshold = 1.0 / (1.0 + l01 - l11)
    theta = np.zeros(dimension, dtype=np.float64)
    accumulated_v = matrix_regularization * np.eye(dimension, dtype=np.float64)
    active_v_inverse = np.eye(dimension, dtype=np.float64) / matrix_regularization
    tasted_x_chunks: list[np.ndarray] = []
    tasted_y_chunks: list[np.ndarray] = []
    tasted_count = 0
    state_dirty = False
    model_updates = 0
    last_training_count = 0
    routed = 0
    correct = 0

    for boundary, start, stop in _doubling_epochs(n):
        del boundary
        if state_dirty:
            tasted_x = np.concatenate(tasted_x_chunks, axis=0)
            tasted_y = np.concatenate(tasted_y_chunks, axis=0)
            theta = _fit_logistic_theta(
                tasted_x,
                tasted_y,
                config,
                None if zero_start else theta,
            )
            # V and theta are snapshots of feedback through the preceding round.
            active_v_inverse = np.linalg.inv(accumulated_v)
            active_v_inverse = 0.5 * (
                active_v_inverse + active_v_inverse.T
            )
            model_updates += 1
            last_training_count = tasted_count
            state_dirty = False

        epoch_x = x_all[start:stop]
        predicted = np.asarray(LogCBPSideAT.sigmoid(epoch_x @ theta))
        leverage_squared = np.einsum(
            "ij,jk,ik->i", epoch_x, active_v_inverse, epoch_x, optimize=True
        )
        leverage = np.sqrt(np.maximum(0.0, leverage_squared))
        radius = np.minimum(beta_scale * leverage, confidence_cap)
        confident = np.abs(predicted - threshold) >= radius
        actions = np.logical_or(~confident, predicted >= threshold)

        epoch_y = y_all[start:stop]
        routed += int(np.count_nonzero(actions))
        correct += int(np.count_nonzero(actions | (epoch_y == 0)))
        if np.any(actions):
            revealed_x = epoch_x[actions]
            revealed_y = epoch_y[actions]
            tasted_x_chunks.append(revealed_x)
            tasted_y_chunks.append(revealed_y)
            accumulated_v += revealed_x.T @ revealed_x
            tasted_count += int(revealed_y.size)
            state_dirty = True

    row = _base_row(
        policy=POLICY_CBPSIDE,
        l01=l01,
        l11=l11,
        multiplier=multiplier,
        parameter_name="beta_scale",
        base_parameter=base_beta_scale,
        effective_parameter=beta_scale,
        order_index=order_index,
        order_seed=order_seed,
        policy_seed=policy_seed,
        examples=n,
        tree_settings=tree_settings,
    )
    row["confidence_cap"] = float(confidence_cap)
    row["confidence_matrix_state"] = "frozen_within_each_doubling_epoch"
    row["theta_warm_started"] = not zero_start
    return _finish_row(
        row,
        routed=routed,
        correct=correct,
        model_updates=model_updates,
        last_training_count=last_training_count,
    )


def _tree_is_feasible(label_counts: np.ndarray) -> bool:
    return bool(np.all(label_counts >= 2))


def _simulate_igw_candidate(
    contexts: np.ndarray,
    outcomes: np.ndarray,
    permutation: np.ndarray,
    *,
    policy: str,
    l01: float,
    l11: float,
    multiplier: float,
    base_gamma: float,
    mu: float,
    min_propensity: float,
    order_index: int,
    order_seed: int,
    policy_seed: int,
    estimator_settings: dict[str, Any],
) -> dict[str, Any]:
    """Run IGW with a predictor snapshot updated at global doubling rounds."""
    if policy not in {POLICY_IGW_TREE, POLICY_IGW_LINEAR}:
        raise ValueError(f"Unsupported IGW policy identifier: {policy}")
    if (policy == POLICY_IGW_LINEAR) != (
        estimator_settings["kind"] == "logistic"
    ):
        raise ValueError("IGW policy identifier and estimator kind do not match")
    x_all = contexts[permutation]
    y_all = outcomes[permutation]
    n = x_all.shape[0]
    gamma = base_gamma * multiplier
    uniforms = np.random.default_rng(policy_seed).random(n)
    label_counts = np.zeros(2, dtype=np.int64)
    tasted_x_chunks: list[np.ndarray] = []
    tasted_y_chunks: list[np.ndarray] = []
    tasted_w_chunks: list[np.ndarray] = []
    tasted_count = 0
    trained_count = 0
    state_dirty = False
    backend = None
    model_updates = 0
    routed = 0
    correct = 0

    for boundary, start, stop in _doubling_epochs(n):
        del boundary
        if state_dirty and _tree_is_feasible(label_counts):
            if backend is None:
                backend = make_tree_backend(
                    estimator_settings, seed=policy_seed
                )
            if backend.supports_incremental:
                all_x = np.concatenate(tasted_x_chunks, axis=0)
                all_y = np.concatenate(tasted_y_chunks, axis=0)
                all_w = np.concatenate(tasted_w_chunks, axis=0)
                backend.update_many(
                    all_x[trained_count:],
                    all_y[trained_count:],
                    all_w[trained_count:],
                )
            else:
                backend.fit_all(
                    np.concatenate(tasted_x_chunks, axis=0),
                    np.concatenate(tasted_y_chunks, axis=0),
                    np.concatenate(tasted_w_chunks, axis=0),
                )
            trained_count = tasted_count
            model_updates += 1
            state_dirty = False

        epoch_x = x_all[start:stop]
        if backend is None:
            probability = (label_counts[1] + 1.0) / (tasted_count + 2.0)
            predicted = np.full(epoch_x.shape[0], probability, dtype=np.float64)
        else:
            predicted = np.clip(backend.predict_proba(epoch_x), 0.0, 1.0)

        loss_0 = l01 * predicted
        loss_1 = 1.0 + (l11 - 1.0) * predicted
        gap = np.abs(loss_0 - loss_1)
        worse_probability = 1.0 / (mu + gamma * gap)
        probability_1 = np.where(
            loss_1 <= loss_0,
            1.0 - worse_probability,
            worse_probability,
        )
        actions = uniforms[start:stop] < probability_1
        epoch_y = y_all[start:stop]
        routed += int(np.count_nonzero(actions))
        correct += int(np.count_nonzero(actions | (epoch_y == 0)))
        if np.any(actions):
            revealed_x = epoch_x[actions]
            revealed_y = epoch_y[actions]
            revealed_w = 1.0 / np.maximum(
                probability_1[actions], min_propensity
            )
            tasted_x_chunks.append(revealed_x)
            tasted_y_chunks.append(revealed_y)
            tasted_w_chunks.append(revealed_w)
            label_counts += np.bincount(revealed_y, minlength=2)
            tasted_count += int(revealed_y.size)
            state_dirty = True

    row = _base_row(
        policy=policy,
        l01=l01,
        l11=l11,
        multiplier=multiplier,
        parameter_name="gamma",
        base_parameter=base_gamma,
        effective_parameter=gamma,
        order_index=order_index,
        order_seed=order_seed,
        policy_seed=policy_seed,
        examples=n,
        tree_settings=estimator_settings,
    )
    row.update(
        {
            "igw_mu": float(mu),
            "igw_min_propensity": float(min_propensity),
            "inverse_propensity_weight_cap": float(1.0 / min_propensity),
            "estimator_feedback_update": (
                "buffered_incremental_at_doubling_boundaries"
                if estimator_settings["kind"] == "river-hoeffding"
                else "full_history_refit_at_doubling_boundaries"
            ),
            "estimator_profile": estimator_settings,
            "comparison_role": (
                "linear_oracle"
                if policy == POLICY_IGW_LINEAR
                else "nonlinear_tree_oracle"
            ),
        }
    )
    return _finish_row(
        row,
        routed=routed,
        correct=correct,
        model_updates=model_updates,
        last_training_count=trained_count,
    )


def _simulate_etc_candidates(
    contexts: np.ndarray,
    outcomes: np.ndarray,
    permutation: np.ndarray,
    *,
    l01_values: Sequence[float],
    l11: float,
    multiplier: float,
    base_tastes: float,
    order_index: int,
    order_seed: int,
    policy_seed: int,
    tree_settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fit ETC once per order/multiplier and reuse its tail predictions by l01."""
    x_all = contexts[permutation]
    y_all = outcomes[permutation]
    n = x_all.shape[0]
    forced_tastes = min(n, max(1, int(math.ceil(multiplier * base_tastes))))
    prefix_y = y_all[:forced_tastes]
    label_counts = np.bincount(prefix_y, minlength=2)
    backend = None
    model_updates = 0
    if _tree_is_feasible(label_counts):
        backend = make_tree_backend(tree_settings, seed=policy_seed)
        weights = np.ones(forced_tastes, dtype=np.float64)
        if backend.supports_incremental:
            backend.update_many(x_all[:forced_tastes], prefix_y, weights)
        else:
            backend.fit_all(x_all[:forced_tastes], prefix_y, weights)
        model_updates = 1

    if forced_tastes < n:
        if backend is None:
            probability = (label_counts[1] + 1.0) / (forced_tastes + 2.0)
            tail_probability = np.full(n - forced_tastes, probability)
        else:
            tail_probability = np.clip(
                backend.predict_proba(x_all[forced_tastes:]), 0.0, 1.0
            )
    else:
        tail_probability = np.asarray([], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for l01 in l01_values:
        threshold = 1.0 / (1.0 + l01 - l11)
        tail_actions = tail_probability >= threshold
        routed = forced_tastes + int(np.count_nonzero(tail_actions))
        correct = forced_tastes + int(
            np.count_nonzero(tail_actions | (y_all[forced_tastes:] == 0))
        )
        row = _base_row(
            policy=POLICY_ETC,
            l01=l01,
            l11=l11,
            multiplier=multiplier,
            parameter_name="forced_tastes",
            base_parameter=base_tastes,
            effective_parameter=forced_tastes,
            order_index=order_index,
            order_seed=order_seed,
            policy_seed=policy_seed,
            examples=n,
            tree_settings=tree_settings,
        )
        row["forced_taste_rounding"] = "ceil(multiplier * base_tastes), capped at n"
        row["tree_feedback_update"] = "unit_weight_prefix_fit_then_freeze"
        rows.append(
            _finish_row(
                row,
                routed=routed,
                correct=correct,
                model_updates=model_updates,
                last_training_count=(forced_tastes if model_updates else 0),
            )
        )
    return rows


def _parallel_map(function, tasks: list[dict[str, Any]], jobs: int):
    if jobs == 1:
        # Yield immediately so the caller writes one atomic order checkpoint
        # before starting the next.  An interrupted default run therefore
        # loses at most the currently executing trajectory.
        for task in tasks:
            yield function(**task)
        return
    from joblib import Parallel, delayed, parallel_backend

    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(
            n_jobs=jobs,
            max_nbytes="2M",
            mmap_mode="r",
            return_as="generator_unordered",
        )(delayed(function)(**task) for task in tasks)
        # The caller writes each yielded result atomically, so completed worker
        # trajectories survive an interruption without waiting for the batch.
        for result in results:
            yield result


def _float_slug(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _checkpoint_path(
    output: Path,
    policy: str,
    l01: float,
    multiplier: float,
    order_index: int,
) -> Path:
    return (
        output
        / "checkpoints"
        / policy.lower()
        / f"l01-{_float_slug(l01)}"
        / f"multiplier-{_float_slug(multiplier)}"
        / f"order-{order_index + 1:02d}.json"
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _load_checkpoint(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("config_fingerprint") != fingerprint:
        raise RuntimeError(f"Checkpoint has a different configuration: {path}")
    return row


def _save_checkpoint(path: Path, row: dict[str, Any], fingerprint: str) -> None:
    payload = dict(row)
    payload["config_fingerprint"] = fingerprint
    _atomic_write_json(path, payload)


def _aggregate_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["policy"]),
            float(row["l01"]),
            float(row["l11"]),
            float(row["multiplier"]),
        )
        groups.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: (item[0], item[1], item[3])):
        group = sorted(groups[key], key=lambda row: int(row["order_run"]))
        first = group[0]
        result = {
            name: value
            for name, value in first.items()
            if name
            not in {
                *AGGREGATE_METRICS,
                "order_run",
                "order_seed",
                "order_was_shuffled",
                "config_fingerprint",
            }
        }
        result["online_order_repeats"] = len(group)
        result["order_seeds"] = [int(row["order_seed"]) for row in group]
        result["error_bar_definition"] = (
            "sample standard deviation across paired shuffled online orders"
        )
        for metric in AGGREGATE_METRICS:
            values = [row.get(metric) for row in group]
            if any(value is None for value in values):
                continue
            array = np.asarray(values, dtype=np.float64)
            mean = float(np.mean(array))
            std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
            result[metric] = mean
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
            result[f"{metric}_sem"] = float(std / np.sqrt(len(array)))
        aggregated.append(result)
    return aggregated


def _select_pointwise_multipliers(
    candidate_aggregates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in candidate_aggregates:
        key = (str(row["policy"]), float(row["l01"]), float(row["l11"]))
        groups.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: (item[0], item[1])):
        candidates = groups[key]
        winner = min(
            candidates,
            key=lambda row: (
                float(row["realized_total_cost_mean"]),
                abs(float(row["multiplier"]) - 1.0),
                float(row["multiplier"]),
            ),
        )
        selected.append(
            {
                "policy": winner["policy"],
                "method": winner["method"],
                "l01": float(winner["l01"]),
                "l11": float(winner["l11"]),
                "alpha": float(winner["alpha"]),
                "selected_multiplier": float(winner["multiplier"]),
                "parameter_name": winner["parameter_name"],
                "base_parameter": float(winner["base_parameter"]),
                "selected_effective_parameter": float(
                    winner["effective_parameter"]
                ),
                "selection_mean_realized_total_cost": float(
                    winner["realized_total_cost_mean"]
                ),
                "selection_rule": (
                    "lowest mean realized total cost across the same shuffled "
                    "orders; ties prefer multiplier closest to 1, then smaller"
                ),
                "selection_evaluation_reuse": True,
                "selection_interpretation": (
                    "exploratory pointwise oracle envelope; optimistically selected "
                    "and evaluated on the same orders"
                ),
            }
        )
    return selected


def _selected_order_rows(
    candidate_rows: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["policy"]), float(row["l01"]), float(row["l11"])): float(
            row["selected_multiplier"]
        )
        for row in selections
    }
    selected: list[dict[str, Any]] = []
    for row in candidate_rows:
        key = (str(row["policy"]), float(row["l01"]), float(row["l11"]))
        if float(row["multiplier"]) == lookup[key]:
            copied = dict(row)
            copied["selected_multiplier"] = float(row["multiplier"])
            copied["selection_evaluation_reuse"] = True
            selected.append(copied)
    return selected


def _expected_random_rows(
    selected_order_rows: list[dict[str, Any]], outcomes: np.ndarray
) -> list[dict[str, Any]]:
    disagreement_count = int(np.count_nonzero(outcomes))
    rows: list[dict[str, Any]] = []
    for etc in selected_order_rows:
        if etc["policy"] != POLICY_ETC:
            continue
        n = int(etc["examples"])
        rate = float(etc["routing_rate"])
        accuracy = 1.0 - (1.0 - rate) * disagreement_count / n
        row = {
            "policy": "Random",
            "method": "Random (expected, matched to selected ETC)",
            "l01": float(etc["l01"]),
            "l11": float(etc["l11"]),
            "alpha": float(etc["alpha"]),
            "multiplier": None,
            "selected_multiplier": None,
            "parameter_name": "matched_ETC_routing_rate",
            "base_parameter": None,
            "effective_parameter": rate,
            "order_run": int(etc["order_run"]),
            "order_seed": int(etc["order_seed"]),
            "policy_seed": None,
            "order_was_shuffled": True,
            "examples": n,
            "routing_rate": rate,
            "accuracy": float(accuracy),
            "model_updates": 0,
            "last_model_training_count": 0,
            "random_baseline": "analytic expectation conditional on ETC traffic",
            "matched_etc_multiplier": float(etc["selected_multiplier"]),
            "selection_evaluation_reuse": True,
        }
        row.update(
            _realized_cost_metrics(
                l01=float(row["l01"]),
                l11=float(row["l11"]),
                routing_rate=rate,
                accuracy=float(accuracy),
                examples=n,
            )
        )
        rows.append(row)
    return rows


def _aggregate_selected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["policy"]), float(row["l01"]), float(row["l11"]))
        groups.setdefault(key, []).append(row)
    aggregated: list[dict[str, Any]] = []
    policy_order = {
        POLICY_CBPSIDE: 0,
        POLICY_ETC: 1,
        POLICY_IGW_LINEAR: 2,
        POLICY_IGW_TREE: 3,
        "Random": 4,
    }
    for key in sorted(
        groups, key=lambda item: (policy_order.get(item[0], 99), item[1])
    ):
        group = sorted(groups[key], key=lambda row: int(row["order_run"]))
        first = group[0]
        result = {
            name: value
            for name, value in first.items()
            if name
            not in {
                *AGGREGATE_METRICS,
                "order_run",
                "order_seed",
                "order_was_shuffled",
                "config_fingerprint",
            }
        }
        result["online_order_repeats"] = len(group)
        result["order_seeds"] = [int(row["order_seed"]) for row in group]
        result["error_bar_definition"] = (
            "sample standard deviation across paired shuffled online orders"
        )
        for metric in AGGREGATE_METRICS:
            values = [row.get(metric) for row in group]
            if any(value is None for value in values):
                continue
            array = np.asarray(values, dtype=np.float64)
            mean = float(np.mean(array))
            std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
            result[metric] = mean
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
            result[f"{metric}_sem"] = float(std / np.sqrt(len(array)))
        aggregated.append(result)
    return aggregated


IGW_COMPARISON_METRICS = (
    "tree_realized_total_cost",
    "linear_realized_total_cost",
    "nonlinear_tree_cost_reduction",
    "nonlinear_tree_cost_reduction_per_example",
    "nonlinear_tree_accuracy_gain",
    "nonlinear_tree_routing_rate_change",
)


def _igw_difference_row(
    tree: dict[str, Any],
    linear: dict[str, Any],
    *,
    comparison_scope: str,
) -> dict[str, Any]:
    """Return one paired order-level IGW estimator difference."""
    if tree["order_seed"] != linear["order_seed"]:
        raise RuntimeError("Paired IGW rows have different online orders")
    if int(tree["examples"]) != int(linear["examples"]):
        raise RuntimeError("Paired IGW rows have different online horizons")
    for name in ("l01", "l11"):
        if not np.isclose(float(tree[name]), float(linear[name])):
            raise RuntimeError(f"Paired IGW rows have different {name} values")

    tree_cost = float(tree["realized_total_cost"])
    linear_cost = float(linear["realized_total_cost"])
    examples = int(tree["examples"])
    reduction = linear_cost - tree_cost
    tree_multiplier = float(
        tree["selected_multiplier"]
        if "selected_multiplier" in tree
        else tree["multiplier"]
    )
    linear_multiplier = float(
        linear["selected_multiplier"]
        if "selected_multiplier" in linear
        else linear["multiplier"]
    )
    tree_gamma = float(tree["effective_parameter"])
    linear_gamma = float(linear["effective_parameter"])
    return {
        "comparison_scope": comparison_scope,
        "l01": float(tree["l01"]),
        "l11": float(tree["l11"]),
        "alpha": float(tree["alpha"]),
        "order_run": int(tree["order_run"]),
        "order_seed": int(tree["order_seed"]),
        "examples": examples,
        "tree_method": tree["method"],
        "linear_method": linear["method"],
        "tree_gamma_multiplier": tree_multiplier,
        "linear_gamma_multiplier": linear_multiplier,
        "tree_effective_gamma": tree_gamma,
        "linear_effective_gamma": linear_gamma,
        "same_effective_gamma": bool(
            np.isclose(tree_gamma, linear_gamma, rtol=1e-12, atol=1e-12)
        ),
        "tree_realized_total_cost": tree_cost,
        "linear_realized_total_cost": linear_cost,
        "nonlinear_tree_cost_reduction": reduction,
        "nonlinear_tree_cost_reduction_per_example": reduction / examples,
        "nonlinear_tree_accuracy_gain": float(tree["accuracy"])
        - float(linear["accuracy"]),
        "nonlinear_tree_routing_rate_change": float(tree["routing_rate"])
        - float(linear["routing_rate"]),
        "positive_cost_reduction_favors": "IGW Tree",
        "selection_evaluation_reuse": bool(
            tree.get("selection_evaluation_reuse", False)
            or linear.get("selection_evaluation_reuse", False)
        ),
    }


def _igw_comparison_by_order(
    selected_order_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair separately tuned IGW variants on each shared online order."""
    rows_by_policy = {
        policy: {
            (float(row["l01"]), int(row["order_run"])): row
            for row in selected_order_rows
            if row["policy"] == policy
        }
        for policy in (POLICY_IGW_TREE, POLICY_IGW_LINEAR)
    }
    tree_rows = rows_by_policy[POLICY_IGW_TREE]
    linear_rows = rows_by_policy[POLICY_IGW_LINEAR]
    if tree_rows.keys() != linear_rows.keys():
        raise RuntimeError(
            "Selected IGW Tree and IGW Linear rows are not paired by loss/order"
        )

    paired: list[dict[str, Any]] = []
    for key in sorted(tree_rows):
        tree = tree_rows[key]
        linear = linear_rows[key]
        paired.append(
            _igw_difference_row(
                tree,
                linear,
                comparison_scope="separately_tuned_best_vs_best",
            )
        )
    return paired


def _igw_matched_comparison_by_order(
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair tree and linear IGW candidates at the same gamma multiplier."""
    rows_by_policy = {
        policy: {
            (
                float(row["l01"]),
                float(row["multiplier"]),
                int(row["order_run"]),
            ): row
            for row in candidate_rows
            if row["policy"] == policy
        }
        for policy in (POLICY_IGW_TREE, POLICY_IGW_LINEAR)
    }
    tree_rows = rows_by_policy[POLICY_IGW_TREE]
    linear_rows = rows_by_policy[POLICY_IGW_LINEAR]
    if tree_rows.keys() != linear_rows.keys():
        raise RuntimeError(
            "IGW Tree and IGW Linear candidates are not paired by "
            "loss/multiplier/order"
        )

    paired: list[dict[str, Any]] = []
    for key in sorted(tree_rows):
        row = _igw_difference_row(
            tree_rows[key],
            linear_rows[key],
            comparison_scope="matched_gamma_estimator_contrast",
        )
        if not row["same_effective_gamma"]:
            raise RuntimeError("Matched IGW candidates have different gamma values")
        paired.append(row)
    return paired


def _aggregate_igw_comparison(
    rows: list[dict[str, Any]],
    *,
    group_by_multiplier: bool = False,
) -> list[dict[str, Any]]:
    groups: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["l01"]),)
        if group_by_multiplier:
            key += (float(row["tree_gamma_multiplier"]),)
        groups.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        group = sorted(groups[group_key], key=lambda row: int(row["order_run"]))
        first = group[0]
        result = {
            key: first[key]
            for key in (
                "l01",
                "l11",
                "alpha",
                "examples",
                "comparison_scope",
                "tree_method",
                "linear_method",
                "tree_gamma_multiplier",
                "linear_gamma_multiplier",
                "tree_effective_gamma",
                "linear_effective_gamma",
                "same_effective_gamma",
                "positive_cost_reduction_favors",
                "selection_evaluation_reuse",
            )
        }
        result["online_order_repeats"] = len(group)
        result["order_seeds"] = [int(row["order_seed"]) for row in group]
        result["error_bar_definition"] = (
            "sample standard deviation of paired order-level differences"
        )
        reductions = np.asarray(
            [row["nonlinear_tree_cost_reduction"] for row in group],
            dtype=np.float64,
        )
        tied = np.isclose(reductions, 0.0, rtol=1e-12, atol=1e-12)
        result["tree_lower_cost_orders"] = int(
            np.count_nonzero((reductions > 0) & ~tied)
        )
        result["linear_lower_cost_orders"] = int(
            np.count_nonzero((reductions < 0) & ~tied)
        )
        result["equal_cost_orders"] = int(np.count_nonzero(tied))
        for metric in IGW_COMPARISON_METRICS:
            values = np.asarray(
                [row[metric] for row in group], dtype=np.float64
            )
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
            result[f"{metric}_sem"] = float(std / np.sqrt(len(values)))
        aggregated.append(result)
    return aggregated


def _plot_selected_routing_accuracy(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for method in dict.fromkeys(str(row["method"]) for row in rows):
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: float(row["l01"]),
        )
        axis.errorbar(
            [row["routing_rate_mean"] for row in selected],
            [row["accuracy_mean"] for row in selected],
            xerr=[row["routing_rate_std"] for row in selected],
            yerr=[row["accuracy_std"] for row in selected],
            fmt="-o",
            capsize=3,
            label=method,
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Strong-model routing rate")
    axis.set_ylabel("Agreement with cached strong-model reference")
    repeats = int(rows[0]["online_order_repeats"])
    axis.set_title(
        "Pointwise multiplier-selected routing rate versus accuracy\n"
        f"Mean +/- 1 SD across {repeats} paired shuffled orders"
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_selected_cost(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for method in dict.fromkeys(str(row["method"]) for row in rows):
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: float(row["l01"]),
        )
        axis.errorbar(
            [row["l01"] for row in selected],
            [row["realized_total_cost_mean"] for row in selected],
            yerr=[row["realized_total_cost_std"] for row in selected],
            fmt="-o",
            capsize=3,
            label=method,
        )
    ticks = sorted({float(row["l01"]) for row in rows})
    axis.set_xticks(ticks, [f"{value:g}" for value in ticks])
    axis.set_xlabel(r"Unrouted-disagreement cost $\ell_{01}$")
    axis.set_ylabel(
        f"Total cost over {int(rows[0]['examples']):,} online samples"
    )
    repeats = int(rows[0]["online_order_repeats"])
    axis.set_title(
        "Pointwise multiplier-selected total cost "
        "(learned policies realized; Random expected)\n"
        f"Mean +/- 1 SD across {repeats} paired shuffled orders"
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    l11_values = {float(row["l11"]) for row in rows}
    relation = (
        r"$\alpha=1/\ell_{01}$ because $\ell_{11}=1$"
        if l11_values == {1.0}
        else r"$\alpha=1/(1+\ell_{01}-\ell_{11})$"
    )
    figure.text(
        0.5,
        0.005,
        relation + "; exploratory oracle selection uses these same orders",
        ha="center",
        fontsize=9,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_selected_multipliers(path: Path, selections: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    labels = {
        POLICY_CBPSIDE: "CBPSide",
        POLICY_ETC: "ETC",
        POLICY_IGW_LINEAR: "IGW Linear",
        POLICY_IGW_TREE: "IGW Tree",
    }
    for policy in TUNED_POLICIES:
        selected = sorted(
            (row for row in selections if row["policy"] == policy),
            key=lambda row: float(row["l01"]),
        )
        axis.plot(
            [row["l01"] for row in selected],
            [row["selected_multiplier"] for row in selected],
            "-o",
            label=labels[policy],
        )
    ticks = sorted({float(row["l01"]) for row in selections})
    axis.set_xticks(ticks, [f"{value:g}" for value in ticks])
    axis.set_yscale("log")
    axis.set_xlabel(r"Unrouted-disagreement cost $\ell_{01}$")
    axis.set_ylabel("Selected multiplier (log scale)")
    axis.set_title("Pointwise multiplier selected by lowest mean total cost")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_igw_estimator_comparison(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = sorted(rows, key=lambda row: float(row["l01"]))
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.errorbar(
        [row["l01"] for row in selected],
        [row["nonlinear_tree_cost_reduction_mean"] for row in selected],
        yerr=[row["nonlinear_tree_cost_reduction_std"] for row in selected],
        fmt="-o",
        capsize=3,
        color="tab:purple",
    )
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ticks = [float(row["l01"]) for row in selected]
    axis.set_xticks(ticks, [f"{value:g}" for value in ticks])
    axis.set_xlabel(r"Unrouted-disagreement cost $\ell_{01}$")
    axis.set_ylabel("Linear IGW cost - tree IGW cost\n(positive favors tree)")
    repeats = int(selected[0]["online_order_repeats"])
    axis.set_title(
        "Exploratory separately tuned IGW comparison\n"
        f"Mean +/- 1 SD across {repeats} shuffled orders"
    )
    axis.grid(alpha=0.25)
    figure.text(
        0.5,
        0.005,
        "Each variant uses its separately selected gamma multiplier on these "
        "same orders",
        ha="center",
        fontsize=9,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_igw_matched_estimator_comparison(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    multipliers = sorted({float(row["tree_gamma_multiplier"]) for row in rows})
    for multiplier in multipliers:
        selected = sorted(
            (
                row
                for row in rows
                if float(row["tree_gamma_multiplier"]) == multiplier
            ),
            key=lambda row: float(row["l01"]),
        )
        axis.errorbar(
            [row["l01"] for row in selected],
            [row["nonlinear_tree_cost_reduction_mean"] for row in selected],
            yerr=[row["nonlinear_tree_cost_reduction_std"] for row in selected],
            fmt="-o",
            capsize=3,
            label=f"{multiplier:g}",
        )
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ticks = sorted({float(row["l01"]) for row in rows})
    axis.set_xticks(ticks, [f"{value:g}" for value in ticks])
    axis.set_xlabel(r"Unrouted-disagreement cost $\ell_{01}$")
    axis.set_ylabel("Linear IGW cost - tree IGW cost\n(positive favors tree)")
    repeats = int(rows[0]["online_order_repeats"])
    axis.set_title(
        "Matched-gamma IGW estimator comparison\n"
        f"Mean +/- 1 SD across {repeats} shuffled orders"
    )
    axis.grid(alpha=0.25)
    axis.legend(title=r"Gamma multiplier")
    figure.text(
        0.5,
        0.005,
        "Within each line, both IGW variants use the same gamma and online "
        "protocol; action-dependent histories may diverge",
        ha="center",
        fontsize=9,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _bundle(output: Path) -> Path:
    destination = output / "multiplier-sweep-results.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path != destination:
                archive.write(path, path.name)
    return destination


def _manifest_and_fingerprint(
    args: argparse.Namespace,
    *,
    contexts: np.ndarray,
    outcomes: np.ndarray,
    example_ids: Sequence[str],
    context_summary: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    n = contexts.shape[0]
    base_gamma = (
        float(args.igw_base_gamma)
        if args.igw_base_gamma is not None
        else float(np.sqrt(n))
    )
    base_tastes = (
        float(args.etc_base_tastes)
        if args.etc_base_tastes is not None
        else float(n ** (2.0 / 3.0))
    )
    identity = hashlib.sha256()
    for example_id in example_ids:
        identity.update(example_id.encode("utf-8"))
        identity.update(b"\0")
    identity.update(np.ascontiguousarray(contexts).tobytes())
    identity.update(np.ascontiguousarray(outcomes).tobytes())
    manifest = {
        "design": "pointwise-online-parameter-multiplier-sweep-v3",
        "implementation_revision": TUNING_IMPLEMENTATION_REVISION,
        "cache": str(args.cache.resolve()),
        "examples": int(n),
        "context_dimension": int(contexts.shape[1]),
        "context": context_summary,
        "context_profile": args.context_profile,
        "outcome_source": "cached_weak_strong_disagreement",
        "routing_reference": "cached_strong_model_answer",
        "benchmark_gold_answers_used_as_routing_labels": False,
        "l01_values": [float(value) for value in args.l01_values],
        "l11": float(args.l11),
        "multipliers": [float(value) for value in args.multipliers],
        "online_order_repeats": int(args.online_order_repeats),
        "order_seeds": [
            int(args.seed + index) for index in range(args.online_order_repeats)
        ],
        "policy_seed": int(args.policy_seed),
        "tuned_policies": list(TUNED_POLICIES),
        "base_parameters": {
            "cbpside_beta_scale": float(args.cbpside_base_beta_scale),
            "cbpside_confidence_cap": float(
                args.cbpside_max_confidence_radius
            ),
            "cbpside_matrix_regularization": float(
                args.cbpside_matrix_regularization
            ),
            "cbpside_theta_regularization": float(
                args.cbpside_theta_regularization
            ),
            "igw_gamma": base_gamma,
            "igw_gamma_rule": (
                "command_line" if args.igw_base_gamma is not None else "sqrt(n)"
            ),
            "etc_tastes": base_tastes,
            "etc_tastes_rule": (
                "command_line"
                if args.etc_base_tastes is not None
                else "n^(2/3), then ceil after multiplication"
            ),
        },
        "effective_etc_tastes": {
            format(float(multiplier), ".12g"): min(
                n, max(1, int(math.ceil(multiplier * base_tastes)))
            )
            for multiplier in args.multipliers
        },
        "update_schedule": {
            "boundaries": "before global rounds 1,2,4,8,...",
            "history_cutoff": "feedback through boundary_round-1",
            "cbpside_theta": "refit only at boundary when new tastes exist",
            "cbpside_V_inverse": "recompute only at boundary and freeze in epoch",
            "cbpside_beta": "evaluate on every current context using epoch V inverse",
            "igw_tree": (
                "buffered incremental update only at boundary"
                if args.tree_estimator == "river-hoeffding"
                else "full revealed-history refit only at boundary"
            ),
            "igw_linear": "full revealed-history refit only at boundary",
        },
        "tree": _tree_settings(args),
        "igw_linear_profile": _linear_settings(),
        "igw_estimator_comparison": {
            "shared_protocol": [
                "contexts",
                "selective-feedback protocol (each policy realizes its own history)",
                "capped inverse-propensity weighting rule",
                "gamma multiplier grid",
                "policy random numbers",
                "global-round doubling schedule",
            ],
            "fixed_multiplier_contrast": (
                "same gamma; probability-estimator family is the only configured "
                "difference, although action-dependent histories may diverge"
            ),
            "selected_contrast": (
                "separately tuned best-vs-best policies; selected gamma multipliers "
                "may differ"
            ),
            "tree_policy": POLICY_IGW_TREE,
            "linear_policy": POLICY_IGW_LINEAR,
        },
        "hgb_profile": (
            ONLINE_HGB_PROFILE if args.tree_estimator == "hgb" else None
        ),
        "cbpside_theta_warm_started": not args.cbpside_zero_start,
        "igw_mu": float(args.igw_mu),
        "igw_min_propensity": float(args.igw_min_propensity),
        "selection": {
            "objective": "lowest mean realized total cost over all order runs",
            "tie_break": "closest multiplier to 1, then smaller multiplier",
            "pointwise_by": ["policy", "l01"],
            "selection_and_evaluation_orders_are_the_same": True,
            "interpretation": "exploratory optimistic oracle envelope",
            "error_bars": "plus/minus one sample SD across shuffled orders",
        },
        "random": "analytic expected routing matched per order to selected ETC traffic",
        "data_identity_sha256": identity.hexdigest(),
    }
    encoded = json.dumps(_jsonable(manifest), sort_keys=True).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    manifest["config_fingerprint"] = fingerprint
    return manifest, fingerprint


def _prepare_output(
    output: Path, manifest: dict[str, Any], fingerprint: str
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "sweep_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise SystemExit(
                "The output directory contains a different sweep configuration. "
                "Choose a new directory rather than mixing checkpoints."
            )
    else:
        if (output / "checkpoints").exists():
            raise SystemExit(
                "Checkpoint directory exists without a sweep manifest; choose a "
                "new output directory or restore its manifest."
            )
        _atomic_write_json(manifest_path, manifest)


def _expected_checkpoint_count(args: argparse.Namespace) -> int:
    candidates_per_policy = (
        len(args.l01_values)
        * len(args.multipliers)
        * args.online_order_repeats
    )
    return len(TUNED_POLICIES) * candidates_per_policy


def _load_all_expected_checkpoints(
    output: Path, args: argparse.Namespace, fingerprint: str
) -> tuple[list[dict[str, Any]], list[tuple[str, float, float, int]]]:
    rows: list[dict[str, Any]] = []
    missing: list[tuple[str, float, float, int]] = []
    for policy in TUNED_POLICIES:
        for l01 in args.l01_values:
            for multiplier in args.multipliers:
                for order_index in range(args.online_order_repeats):
                    path = _checkpoint_path(
                        output, policy, l01, multiplier, order_index
                    )
                    row = _load_checkpoint(path, fingerprint)
                    if row is None:
                        missing.append((policy, l01, multiplier, order_index))
                    else:
                        rows.append(row)
    return rows, missing


def _run_sweep(
    *,
    contexts: np.ndarray,
    normalized_features: np.ndarray,
    outcomes: np.ndarray,
    permutations: np.ndarray,
    args: argparse.Namespace,
    output: Path,
    fingerprint: str,
    base_gamma: float,
    base_tastes: float,
) -> None:
    tree_settings = _tree_settings(args)
    linear_settings = _linear_settings()
    total_candidate_groups = len(args.multipliers) * (
        1 + 3 * len(args.l01_values)
    )
    group_number = 0

    # ETC is fit once for each order/multiplier and reused across every l01.
    for multiplier in args.multipliers:
        group_number += 1
        missing_orders = []
        for order_index in range(args.online_order_repeats):
            if any(
                _load_checkpoint(
                    _checkpoint_path(
                        output, POLICY_ETC, l01, multiplier, order_index
                    ),
                    fingerprint,
                )
                is None
                for l01 in args.l01_values
            ):
                missing_orders.append(order_index)
        print(
            f"[{group_number}/{total_candidate_groups}] ETC multiplier={multiplier:g}; "
            f"{len(missing_orders)} order run(s) remaining",
            flush=True,
        )
        tasks = [
            {
                "contexts": contexts,
                "outcomes": outcomes,
                "permutation": permutations[order_index],
                "l01_values": args.l01_values,
                "l11": args.l11,
                "multiplier": multiplier,
                "base_tastes": base_tastes,
                "order_index": order_index,
                "order_seed": args.seed + order_index,
                "policy_seed": args.policy_seed,
                "tree_settings": tree_settings,
            }
            for order_index in missing_orders
        ]
        for result_rows in _parallel_map(
            _simulate_etc_candidates, tasks, args.jobs
        ):
            for row in result_rows:
                path = _checkpoint_path(
                    output,
                    POLICY_ETC,
                    float(row["l01"]),
                    multiplier,
                    int(row["order_run"]) - 1,
                )
                _save_checkpoint(path, row, fingerprint)

    for policy in (POLICY_CBPSIDE, POLICY_IGW_LINEAR, POLICY_IGW_TREE):
        for l01 in args.l01_values:
            for multiplier in args.multipliers:
                group_number += 1
                missing_orders = []
                for order_index in range(args.online_order_repeats):
                    path = _checkpoint_path(
                        output, policy, l01, multiplier, order_index
                    )
                    if _load_checkpoint(path, fingerprint) is None:
                        missing_orders.append(order_index)
                print(
                    f"[{group_number}/{total_candidate_groups}] {policy} "
                    f"l01={l01:g}, multiplier={multiplier:g}; "
                    f"{len(missing_orders)} order run(s) remaining",
                    flush=True,
                )
                common = {
                    "outcomes": outcomes,
                    "l01": l01,
                    "l11": args.l11,
                    "multiplier": multiplier,
                    "policy_seed": args.policy_seed,
                }
                if policy == POLICY_CBPSIDE:
                    tasks = [
                        common
                        | {
                            "normalized_features": normalized_features,
                            "permutation": permutations[order_index],
                            "base_beta_scale": args.cbpside_base_beta_scale,
                            "confidence_cap": args.cbpside_max_confidence_radius,
                            "matrix_regularization": args.cbpside_matrix_regularization,
                            "theta_regularization": args.cbpside_theta_regularization,
                            "zero_start": args.cbpside_zero_start,
                            "order_index": order_index,
                            "order_seed": args.seed + order_index,
                            "tree_settings": tree_settings,
                        }
                        for order_index in missing_orders
                    ]
                    results = _parallel_map(
                        _simulate_cbpside_candidate, tasks, args.jobs
                    )
                else:
                    estimator_settings = (
                        linear_settings
                        if policy == POLICY_IGW_LINEAR
                        else tree_settings
                    )
                    tasks = [
                        common
                        | {
                            "contexts": contexts,
                            "permutation": permutations[order_index],
                            "policy": policy,
                            "base_gamma": base_gamma,
                            "mu": args.igw_mu,
                            "min_propensity": args.igw_min_propensity,
                            "order_index": order_index,
                            "order_seed": args.seed + order_index,
                            "estimator_settings": estimator_settings,
                        }
                        for order_index in missing_orders
                    ]
                    results = _parallel_map(
                        _simulate_igw_candidate, tasks, args.jobs
                    )
                for row in results:
                    path = _checkpoint_path(
                        output,
                        policy,
                        l01,
                        multiplier,
                        int(row["order_run"]) - 1,
                    )
                    _save_checkpoint(path, row, fingerprint)


def _write_final_outputs(
    *,
    output: Path,
    args: argparse.Namespace,
    fingerprint: str,
    outcomes: np.ndarray,
    manifest: dict[str, Any],
) -> Path:
    candidate_rows, missing = _load_all_expected_checkpoints(
        output, args, fingerprint
    )
    if missing:
        first = missing[0]
        raise SystemExit(
            f"Sweep is incomplete: {len(missing)} of "
            f"{_expected_checkpoint_count(args)} checkpoints are missing; "
            f"first missing={first}. Resume without --plot-only."
        )
    candidate_rows.sort(
        key=lambda row: (
            str(row["policy"]),
            float(row["l01"]),
            float(row["multiplier"]),
            int(row["order_run"]),
        )
    )
    candidate_aggregates = _aggregate_candidates(candidate_rows)
    selections = _select_pointwise_multipliers(candidate_aggregates)
    selected_order_rows = _selected_order_rows(candidate_rows, selections)
    igw_comparison_by_order = _igw_comparison_by_order(selected_order_rows)
    igw_comparison = _aggregate_igw_comparison(igw_comparison_by_order)
    igw_matched_by_order = _igw_matched_comparison_by_order(candidate_rows)
    igw_matched = _aggregate_igw_comparison(
        igw_matched_by_order, group_by_multiplier=True
    )
    selected_order_rows.extend(_expected_random_rows(selected_order_rows, outcomes))
    selected_aggregates = _aggregate_selected(selected_order_rows)

    _write_csv(output / "candidate_results_by_order.csv", candidate_rows)
    _atomic_write_json(output / "candidate_results_by_order.json", candidate_rows)
    _write_csv(output / "candidate_results.csv", candidate_aggregates)
    _atomic_write_json(output / "candidate_results.json", candidate_aggregates)
    _write_csv(output / "selected_multipliers.csv", selections)
    _atomic_write_json(output / "selected_multipliers.json", selections)
    _write_csv(output / "selected_results_by_order.csv", selected_order_rows)
    _atomic_write_json(output / "selected_results_by_order.json", selected_order_rows)
    _write_csv(output / "selected_results.csv", selected_aggregates)
    _atomic_write_json(output / "selected_results.json", selected_aggregates)
    _write_csv(
        output / "igw_tree_vs_linear_by_order.csv", igw_comparison_by_order
    )
    _atomic_write_json(
        output / "igw_tree_vs_linear_by_order.json", igw_comparison_by_order
    )
    _write_csv(output / "igw_tree_vs_linear.csv", igw_comparison)
    _atomic_write_json(output / "igw_tree_vs_linear.json", igw_comparison)
    _write_csv(
        output / "igw_tree_vs_linear_matched_by_order.csv", igw_matched_by_order
    )
    _atomic_write_json(
        output / "igw_tree_vs_linear_matched_by_order.json", igw_matched_by_order
    )
    _write_csv(output / "igw_tree_vs_linear_matched.csv", igw_matched)
    _atomic_write_json(output / "igw_tree_vs_linear_matched.json", igw_matched)

    _plot_selected_routing_accuracy(
        output / "selected_routing_accuracy.png", selected_aggregates
    )
    _plot_selected_cost(output / "selected_cost_vs_l01.png", selected_aggregates)
    _plot_selected_multipliers(
        output / "selected_multiplier_vs_l01.png", selections
    )
    _plot_igw_estimator_comparison(
        output / "igw_tree_vs_linear_cost_difference.png", igw_comparison
    )
    _plot_igw_matched_estimator_comparison(
        output / "igw_tree_vs_linear_matched_cost_difference.png", igw_matched
    )
    summary = {
        "sweep": manifest,
        "candidate_checkpoint_count": len(candidate_rows),
        "selection_count": len(selections),
        "selected_results": selected_aggregates,
        "igw_tree_vs_linear_separately_tuned": igw_comparison,
        "igw_tree_vs_linear_matched_gamma": igw_matched,
        "igw_comparison_interpretation": {
            "difference": (
                "linear IGW realized total cost minus tree IGW realized total "
                "cost; positive values favor the nonlinear tree"
            ),
            "separately_tuned": (
                "best-vs-best comparison; selected gamma multipliers may differ"
            ),
            "matched_gamma": (
                "same gamma multiplier and configured IGW protocol; estimator "
                "family is the only configured difference, while realized "
                "action-dependent histories may diverge"
            ),
        },
        "important_interpretation": (
            "Each point on a selected curve is the best of five multipliers on "
            "these same 20 orders. Those curves are exploratory optimistic oracle "
            "envelopes, not unbiased estimates of preselected policies. The "
            "matched-gamma comparison retains every multiplier without selecting "
            "a winner."
        ),
        "alpha_relation": (
            "alpha = 1/l01 because l11 = 1"
            if {float(row["l11"]) for row in selected_aggregates} == {1.0}
            else "alpha = 1/(1+l01-l11)"
        ),
    }
    _atomic_write_json(output / "summary.json", summary)
    return _bundle(output)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    cache = load_cache(args.cache)
    rounds, context_summary, _, _ = _prompt_context_rounds(
        cache,
        args.prompt_components,
        args.limit,
        args.seed,
        "cached",
        args.context_profile,
    )
    contexts = np.ascontiguousarray(
        np.stack([round_.context for round_ in rounds]), dtype=np.float64
    )
    outcomes = np.asarray(
        [round_.routing_outcome for round_ in rounds], dtype=np.int8
    )
    example_ids = [round_.example_id for round_ in rounds]
    normalized_features = _normalized_cbpside_features(contexts)
    manifest, fingerprint = _manifest_and_fingerprint(
        args,
        contexts=contexts,
        outcomes=outcomes,
        example_ids=example_ids,
        context_summary=context_summary,
    )
    output = args.output_dir.resolve()
    _prepare_output(output, manifest, fingerprint)
    permutations = np.stack(
        [
            np.random.default_rng(args.seed + index)
            .permutation(len(rounds))
            .astype(np.int32)
            for index in range(args.online_order_repeats)
        ]
    )
    permutation_path = output / "online_order_permutations.npz"
    if permutation_path.exists():
        with np.load(permutation_path) as saved:
            if not np.array_equal(saved["permutation_indices"], permutations):
                raise SystemExit(
                    "Saved permutations do not match the sweep manifest"
                )
    else:
        temporary_permutations = permutation_path.with_name(
            permutation_path.name + ".tmp.npz"
        )
        np.savez_compressed(
            temporary_permutations,
            permutation_indices=permutations,
            order_seeds=np.asarray(manifest["order_seeds"], dtype=np.int64),
        )
        temporary_permutations.replace(permutation_path)

    base_gamma = float(manifest["base_parameters"]["igw_gamma"])
    base_tastes = float(manifest["base_parameters"]["etc_tastes"])
    print(
        f"Loaded {len(rounds):,} eligible rows with {contexts.shape[1]} features. "
        f"Tree={args.tree_estimator}; IGW Linear=enabled; "
        f"{args.online_order_repeats} paired orders; "
        f"base gamma={base_gamma:.12g}; base ETC tastes={base_tastes:.12g}.",
        flush=True,
    )
    if not args.plot_only:
        _run_sweep(
            contexts=contexts,
            normalized_features=normalized_features,
            outcomes=outcomes,
            permutations=permutations,
            args=args,
            output=output,
            fingerprint=fingerprint,
            base_gamma=base_gamma,
            base_tastes=base_tastes,
        )
    bundle = _write_final_outputs(
        output=output,
        args=args,
        fingerprint=fingerprint,
        outcomes=outcomes,
        manifest=manifest,
    )
    print(f"Finished. Results: {bundle}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
