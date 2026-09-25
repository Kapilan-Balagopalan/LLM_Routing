"""Resumable pointwise multiplier tuning for the BoolQ online study.

This entry point is intentionally separate from :mod:`run`.  The established
single-configuration simulator stays unchanged, while this module owns the much
larger paired-order tuning design and its candidate-level checkpoints.
"""

from __future__ import annotations

import argparse
import base64
import csv
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
from llm_routing_simulation.plot_style import (
    AXIS_LABELS,
    METHOD_LABELS,
    PLOT_CONFIDENCE_LEVEL,
    PUBLICATION_FIGSIZE,
    PUBLICATION_FIGSIZE_SHORT,
    PUBLICATION_PNG_DPI,
    publication_pyplot,
    save_publication_figure,
    student_t_critical_value,
    student_t_half_width,
)
from llm_routing_simulation.run import (
    DEFAULT_L01_VALUES,
    _jsonable,
    _prompt_context_rounds,
    _realized_cost_metrics,
    _write_csv,
)


DEFAULT_MULTIPLIERS = (0.1, 0.3, 1.0, 3.0, 10.0)
ADAPTIVE_UPDATE_SCHEDULES = ("capped-doubling", "fibonacci", "doubling")
DEFAULT_ADAPTIVE_UPDATE_SCHEDULE = "doubling"
DEFAULT_ADAPTIVE_MAX_ROUND_GAP = 32
DEFAULT_REFERENCE_FOLDS = 5
TUNING_IMPLEMENTATION_REVISION = 9
POLICY_CBPSIDE = "CBPSide"
POLICY_IGW_TREE = "SquareCB.PMSide"
POLICY_IGW_LINEAR = "SquareCB.PMSideLinear"
POLICY_PGTS = "PGTS"
BASE_TUNED_POLICIES = (
    POLICY_CBPSIDE,
    POLICY_IGW_LINEAR,
    POLICY_IGW_TREE,
)
TUNED_POLICIES = (*BASE_TUNED_POLICIES, POLICY_PGTS)
# Retained as an explicit contract: revision 9 has no untuned learned policy.
FIXED_POLICIES: tuple[str, ...] = ()
POLICY_OUTPUT_ORDER = (*TUNED_POLICIES, "Random")
POLICY_OUTPUT_RANK = {
    policy: index for index, policy in enumerate(POLICY_OUTPUT_ORDER)
}
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
INTERNAL_ACTIONS_KEY = "_internal_actions_packbits_base64"
INTERNAL_ACTION_COUNT_KEY = "_internal_actions_count"
INTERNAL_ACTION_BITORDER_KEY = "_internal_actions_bitorder"
INTERNAL_ACTION_FIELDS = (
    INTERNAL_ACTIONS_KEY,
    INTERNAL_ACTION_COUNT_KEY,
    INTERNAL_ACTION_BITORDER_KEY,
)
ACTION_BITORDER = "little"
LEARNING_CURVE_NPZ = "selected_learning_curves_by_order.npz"
LEARNING_CURVE_CSV = "selected_learning_curves.csv"
REFERENCE_PREDICTIONS_NPZ = "cross_fitted_hgb_reference.npz"
REFERENCE_RESULTS_CSV = "cross_fitted_hgb_reference_results.csv"
REFERENCE_RESULTS_JSON = "cross_fitted_hgb_reference_results.json"
LEARNING_CURVE_PLOT_PREFIX = "selected_cumulative_reference_regret_l01-"
AVERAGE_REGRET_PLOT_PREFIX = "selected_average_reference_regret_l01-"
LEGACY_LEARNING_CURVE_PLOT_PREFIX = "selected_cumulative_cost_l01-"
REVISION7_REGRET_PLOT_PREFIX = "selected_cumulative_regret_l01-"
REALIZED_COST_INCREMENT_DEFINITION = (
    "a_t + (1-a_t) * l01 * y_t, where a_t=1 routes strong and y_t=1 "
    "means weak/strong disagreement"
)
CLAIRVOYANT_COST_INCREMENT_DEFINITION = (
    "y_t for l11=1 and l01>=1 (clairvoyant per-round minimum)"
)
REGRET_INCREMENT_DEFINITION = (
    "actual cost increment minus the realized cost of the fixed stratified "
    "cross-fitted HGB-15 reference action; cumulative regret is its prefix sum"
)
CLAIRVOYANT_EXCESS_INCREMENT_DEFINITION = (
    "actual cost increment minus y_t; retained only as an explicitly named "
    "outcome-aware diagnostic, not as the primary regret comparator"
)
RANDOM_COST_INCREMENT_DEFINITION = (
    "q * 1 + (1-q) * l01 * y_t, with q equal to the selected "
    "SquareCB.PMSide tree routing rate for the same order and l01"
)
LEARNING_CURVE_SELECTION_WARNING = (
    "For multiplier-tuned policies, selection and curve evaluation use the "
    "same shuffled orders; their selected learning curves are optimistic "
    "exploratory envelopes."
)


def _active_tuned_policies(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the multiplier-tuned policies enabled for this sweep."""
    return TUNED_POLICIES if args.include_pgts else BASE_TUNED_POLICIES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune CBPSide beta and SquareCB.PMSide gamma pointwise over paired "
            "shuffled online orders, with optional PG-TS prior-scale tuning."
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
        help=(
            "Common grid applied to CBPSide beta, SquareCB.PMSide gamma, and "
            "the PG-TS prior standard deviation when PG-TS is enabled"
        ),
    )
    parser.add_argument("--online-order-repeats", type=int, default=20)
    parser.add_argument(
        "--reference-folds",
        type=int,
        default=DEFAULT_REFERENCE_FOLDS,
        help=(
            "Stratified folds for the fixed out-of-fold HGB-15 reference "
            "policy used by learning-regret plots (default: 5)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=0,
        help="Fixed policy seed shared by every order and multiplier candidate",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--adaptive-update-schedule",
        choices=ADAPTIVE_UPDATE_SCHEDULES,
        default=DEFAULT_ADAPTIVE_UPDATE_SCHEDULE,
        help=(
            "Global-round refit/draw boundaries for CBPSide, both "
            "SquareCB.PMSide variants, and scheduled PG-TS when enabled; pure "
            "doubling (the default) updates at 1,2,4,8,...; capped doubling "
            "and Fibonacci remain available as explicit comparison schedules"
        ),
    )
    parser.add_argument(
        "--adaptive-max-round-gap",
        type=int,
        default=DEFAULT_ADAPTIVE_MAX_ROUND_GAP,
        help="Maximum boundary gap for capped doubling (default: 32 rounds)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild tables, selections, plots, and ZIP from complete checkpoints",
    )

    parser.add_argument(
        "--include-pgts",
        action="store_true",
        help=(
            "Include PG-TS and tune its Gaussian prior standard deviation over "
            "the common multiplier grid. Its posterior is sampled at the "
            "configured adaptive-update boundaries; polyagamma is required."
        ),
    )
    parser.add_argument(
        "--pgts-gibbs-steps",
        type=int,
        default=15,
        help="Gibbs transitions per scheduled PG-TS posterior draw (default: 15)",
    )
    parser.add_argument(
        "--pgts-base-prior-std",
        "--pgts-prior-std",
        dest="pgts_prior_std",
        metavar="STD",
        type=float,
        default=1.0,
        help=(
            "Base isotropic zero-mean Gaussian PG-TS prior standard deviation; "
            "each candidate multiplies it by --multipliers (default: 1)"
        ),
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
        "--squarecb-pmside-base-gamma",
        "--igw-base-gamma",
        dest="igw_base_gamma",
        metavar="GAMMA",
        type=float,
        help="Base gamma; default is sqrt(the selected online horizon)",
    )
    parser.add_argument(
        "--squarecb-pmside-mu",
        "--igw-mu",
        dest="igw_mu",
        metavar="MU",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--squarecb-pmside-min-propensity",
        "--igw-min-propensity",
        dest="igw_min_propensity",
        metavar="MIN_PROPENSITY",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--tree-estimator",
        choices=("hgb", "river-hoeffding"),
        default="hgb",
        help=(
            "Use the established HGB estimator or an explicitly different weighted "
            "incremental Hoeffding tree for SquareCB.PMSide; its linear "
            "variant always uses weighted logistic regression"
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
    if args.reference_folds < 2:
        raise SystemExit("--reference-folds must be at least two")
    if args.jobs == 0:
        raise SystemExit("--jobs must be positive or -1")
    if args.jobs < -1:
        raise SystemExit("--jobs must be positive or -1")
    if args.adaptive_max_round_gap < 1:
        raise SystemExit("--adaptive-max-round-gap must be positive")
    if args.pgts_gibbs_steps < 1:
        raise SystemExit("--pgts-gibbs-steps must be positive")
    if not math.isfinite(args.pgts_prior_std) or args.pgts_prior_std <= 0.0:
        raise SystemExit("--pgts-prior-std must be positive")
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
        raise SystemExit("The SquareCB.PMSide base gamma must be positive")
    if not math.isfinite(args.igw_mu) or args.igw_mu < 2.0:
        raise SystemExit("SquareCB.PMSide mu must be at least two")
    if not 0.0 < args.igw_min_propensity <= 1.0:
        raise SystemExit(
            "SquareCB.PMSide minimum propensity must be in (0, 1]"
        )
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


def _require_pgts_dependency() -> None:
    """Fail before a long sweep when the opt-in Gibbs dependency is absent."""
    try:
        __import__("polyagamma")
    except ImportError as exc:
        raise SystemExit(
            "PG-TS requires its optional dependency. Install it with "
            "python -m pip install -e \".[test,pgts]\", or omit --include-pgts."
        ) from exc


def _schedule_boundaries(
    total_samples: int,
    schedule: str,
    max_round_gap: int = DEFAULT_ADAPTIVE_MAX_ROUND_GAP,
) -> Iterable[int]:
    """Yield unique before-action boundary rounds for an adaptive schedule."""
    if total_samples < 0:
        raise ValueError("The online horizon must be nonnegative")
    if schedule not in ADAPTIVE_UPDATE_SCHEDULES:
        raise ValueError(f"Unknown adaptive update schedule: {schedule!r}")
    if max_round_gap < 1:
        raise ValueError("The adaptive maximum round gap must be positive")
    if schedule in {"capped-doubling", "doubling"}:
        boundary = 1
        while boundary <= total_samples:
            yield boundary
            doubled = 2 * boundary
            boundary = (
                min(doubled, boundary + max_round_gap)
                if schedule == "capped-doubling"
                else doubled
            )
        return

    previous, boundary = 1, 2
    while previous <= total_samples:
        yield previous
        previous, boundary = boundary, previous + boundary


def _adaptive_epochs(
    total_samples: int,
    schedule: str,
    max_round_gap: int = DEFAULT_ADAPTIVE_MAX_ROUND_GAP,
) -> Iterable[tuple[int, int, int]]:
    """Yield `(boundary_round, start_index, stop_index)` without gaps."""
    boundaries = list(
        _schedule_boundaries(total_samples, schedule, max_round_gap)
    )
    for index, boundary in enumerate(boundaries):
        next_boundary = (
            boundaries[index + 1]
            if index + 1 < len(boundaries)
            else total_samples + 1
        )
        yield boundary, boundary - 1, next_boundary - 1


def _doubling_epochs(total_samples: int) -> Iterable[tuple[int, int, int]]:
    """Backward-compatible wrapper for the historical doubling schedule."""
    return _adaptive_epochs(total_samples, "doubling")


def _schedule_slug(schedule: str, max_round_gap: int) -> str:
    """Return an unambiguous compact schedule label for candidate rows."""
    normalized = schedule.replace("-", "_")
    if schedule == "capped-doubling":
        return f"{normalized}_gap_{max_round_gap}"
    return normalized


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
    """Return the fixed linear oracle shared by the linear policy variants."""
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
    if policy == POLICY_PGTS:
        return METHOD_LABELS["pgts"]
    if policy == POLICY_IGW_LINEAR:
        return "SquareCB.PMSide + linear logistic"
    if tree_settings["kind"] == "hgb":
        estimator = f"HGB leaves={tree_settings['hgb_max_leaf_nodes']}"
    else:
        estimator = f"Hoeffding tree depth<={tree_settings['river_max_depth']}"
    if policy == POLICY_IGW_TREE:
        return f"SquareCB.PMSide + {estimator}"
    return f"{policy} ({estimator})"


def _base_row(
    *,
    policy: str,
    l01: float,
    l11: float,
    multiplier: float | None,
    parameter_name: str,
    base_parameter: float | None,
    effective_parameter: float | None,
    order_index: int,
    order_seed: int,
    policy_seed: int,
    examples: int,
    tree_settings: dict[str, Any],
    update_schedule: str,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "method": _method_label(policy, tree_settings),
        "l01": float(l01),
        "l11": float(l11),
        "alpha": float(1.0 / (1.0 + l01 - l11)),
        "multiplier": None if multiplier is None else float(multiplier),
        "parameter_name": parameter_name,
        "base_parameter": (
            None if base_parameter is None else float(base_parameter)
        ),
        "effective_parameter": (
            None if effective_parameter is None else float(effective_parameter)
        ),
        "order_run": order_index + 1,
        "order_seed": int(order_seed),
        "policy_seed": int(policy_seed),
        "order_was_shuffled": True,
        "examples": int(examples),
        "update_schedule": update_schedule,
        "probability_estimator": (
            "bayesian-logistic-polya-gamma-gibbs"
            if policy == POLICY_PGTS
            else (
                "linear-logistic"
                if policy in {POLICY_CBPSIDE, POLICY_IGW_LINEAR}
                else tree_settings["kind"]
            )
        ),
        "tree_estimator": (
            tree_settings["kind"]
            if policy == POLICY_IGW_TREE
            and tree_settings["kind"] in {"hgb", "river-hoeffding"}
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


def _encode_action_payload(actions: np.ndarray) -> dict[str, Any]:
    """Return a compact JSON-safe internal representation of binary actions."""
    array = np.asarray(actions)
    if array.ndim != 1:
        raise ValueError("Online actions must be a one-dimensional array")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("Online actions must be binary")
    packed = np.packbits(
        array.astype(np.uint8, copy=False), bitorder=ACTION_BITORDER
    )
    return {
        INTERNAL_ACTIONS_KEY: base64.b64encode(packed.tobytes()).decode("ascii"),
        INTERNAL_ACTION_COUNT_KEY: int(array.size),
        INTERNAL_ACTION_BITORDER_KEY: ACTION_BITORDER,
    }


def _decode_action_payload(row: dict[str, Any]) -> np.ndarray:
    """Decode and validate a checkpoint's compact binary action trajectory."""
    try:
        encoded = str(row[INTERNAL_ACTIONS_KEY])
        count = int(row[INTERNAL_ACTION_COUNT_KEY])
        bitorder = str(row[INTERNAL_ACTION_BITORDER_KEY])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Checkpoint is missing its internal action payload") from exc
    if count < 0 or bitorder != ACTION_BITORDER:
        raise RuntimeError("Checkpoint has invalid internal action metadata")
    try:
        packed = np.frombuffer(
            base64.b64decode(encoded, validate=True), dtype=np.uint8
        )
    except (ValueError, TypeError) as exc:
        raise RuntimeError("Checkpoint has an invalid base64 action payload") from exc
    required_bytes = (count + 7) // 8
    if packed.size != required_bytes:
        raise RuntimeError("Checkpoint action payload length is inconsistent")
    actions = np.unpackbits(packed, bitorder=bitorder)[:count].astype(bool)
    if count != int(row.get("examples", count)):
        raise RuntimeError("Checkpoint action count differs from its examples")
    return actions


def _attach_action_payload(
    row: dict[str, Any], actions: np.ndarray
) -> dict[str, Any]:
    row.update(_encode_action_payload(actions))
    return row


def _strip_internal_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Remove checkpoint-only data before writing normal result tables."""
    return {
        name: value
        for name, value in row.items()
        if name not in INTERNAL_ACTION_FIELDS
    }


def _simulate_pgts_candidate(
    normalized_features: np.ndarray,
    outcomes: np.ndarray,
    permutation: np.ndarray,
    *,
    l01: float,
    l11: float,
    gibbs_steps: int,
    prior_std: float,
    multiplier: float,
    order_index: int,
    order_seed: int,
    policy_seed: int,
    update_schedule: str,
    update_max_round_gap: int = DEFAULT_ADAPTIVE_MAX_ROUND_GAP,
) -> dict[str, Any]:
    """Run scheduled PG-TS posterior draws with action-1-only feedback."""
    # Keep the optional dependency out of legacy tuner imports and executions.
    from llm_routing_simulation.pgts import PolyaGammaThompsonSampler

    x_all = normalized_features[permutation]
    y_all = outcomes[permutation]
    n, dimension = x_all.shape
    effective_prior_std = float(prior_std * multiplier)
    sampler = PolyaGammaThompsonSampler(
        dimension,
        gibbs_steps=gibbs_steps,
        prior_std=effective_prior_std,
        seed=policy_seed,
    )
    revealed_x = np.empty((n, dimension), dtype=np.float64)
    revealed_y = np.empty(n, dtype=np.int8)
    actions_all = np.zeros(n, dtype=bool)
    revealed_count = 0
    routed = 0
    correct = 0
    theta: np.ndarray | None = None
    state_dirty = False
    posterior_eligible_boundary_rounds: list[int] = []
    posterior_draw_rounds: list[int] = []
    last_posterior_training_count = 0

    try:
        for boundary, start, stop in _adaptive_epochs(
            n, update_schedule, update_max_round_gap
        ):
            posterior_eligible_boundary_rounds.append(int(boundary))
            # This is a runtime-saving approximation to Algorithm 1: sample
            # from feedback available before the boundary, then freeze that
            # sampled theta for every action in the following epoch.
            if theta is None or state_dirty:
                theta = sampler.draw_theta(
                    revealed_x[:revealed_count], revealed_y[:revealed_count]
                )
                posterior_draw_rounds.append(int(boundary))
                last_posterior_training_count = revealed_count
                state_dirty = False
            epoch_x = x_all[start:stop]
            probability = np.asarray(LogCBPSideAT.sigmoid(epoch_x @ theta))
            loss_0 = l01 * probability
            loss_1 = 1.0 + (l11 - 1.0) * probability
            actions = loss_1 <= loss_0
            actions_all[start:stop] = actions
            epoch_y = y_all[start:stop]
            action_count = int(np.count_nonzero(actions))
            routed += action_count
            correct += int(np.count_nonzero(actions | (epoch_y == 0)))
            if action_count:
                next_count = revealed_count + action_count
                revealed_x[revealed_count:next_count] = epoch_x[actions]
                revealed_y[revealed_count:next_count] = epoch_y[actions]
                revealed_count = next_count
                state_dirty = True
    except ImportError as exc:
        raise RuntimeError(
            "PG-TS requires its optional dependency. Install it with "
            "python -m pip install -e \".[test,pgts]\", or rerun without "
            "--include-pgts."
        ) from exc

    schedule_slug = _schedule_slug(update_schedule, update_max_round_gap)
    draw_count = len(posterior_draw_rounds)
    row = _base_row(
        policy=POLICY_PGTS,
        l01=l01,
        l11=l11,
        multiplier=multiplier,
        parameter_name="prior_std",
        base_parameter=float(prior_std),
        effective_parameter=effective_prior_std,
        order_index=order_index,
        order_seed=order_seed,
        policy_seed=policy_seed,
        examples=n,
        tree_settings={},
        update_schedule=(
            f"global_round_{schedule_slug}_posterior_draw_before_action"
        ),
    )
    row.update(
        {
            "pgts_gibbs_steps": int(gibbs_steps),
            "pgts_prior_mean": 0.0,
            "pgts_base_prior_std": float(prior_std),
            "pgts_prior_std_multiplier": float(multiplier),
            "pgts_prior_std": effective_prior_std,
            "pgts_context_preprocessing": (
                "row_l2_normalized_with_max_one_denominator_then_intercept"
            ),
            "pgts_posterior_draw": (
                "final_draw_after_M_gibbs_transitions_at_each_actual_"
                "scheduled_update"
            ),
            "pgts_algorithm1_exact": False,
            "pgts_schedule_approximation": (
                "initial prior draw at round 1; at later configured boundaries "
                "resample only when new action-1 feedback exists; theta frozen "
                "for every action within each epoch"
            ),
            "feedback_protocol": "action_1_only_binary_disagreement",
            "inverse_propensity_weighting": False,
            "posterior_eligible_boundary_rounds": (
                posterior_eligible_boundary_rounds
            ),
            "posterior_eligible_boundary_count": int(
                len(posterior_eligible_boundary_rounds)
            ),
            "posterior_draw_rounds": posterior_draw_rounds,
            "posterior_boundary_draw_count": int(draw_count),
            "posterior_skipped_clean_boundary_count": int(
                len(posterior_eligible_boundary_rounds) - draw_count
            ),
            "total_gibbs_transitions": int(draw_count * gibbs_steps),
            "last_posterior_training_count": int(
                last_posterior_training_count
            ),
            "final_revealed_count": int(revealed_count),
        }
    )
    return _attach_action_payload(
        _finish_row(
            row,
            routed=routed,
            correct=correct,
            model_updates=draw_count,
            last_training_count=last_posterior_training_count,
        ),
        actions_all,
    )


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
    update_schedule: str,
    update_max_round_gap: int = DEFAULT_ADAPTIVE_MAX_ROUND_GAP,
) -> dict[str, Any]:
    """Run scheduled epochs with beta evaluated for every current context."""
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
    actions_all = np.zeros(n, dtype=bool)

    for boundary, start, stop in _adaptive_epochs(
        n, update_schedule, update_max_round_gap
    ):
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
        actions_all[start:stop] = actions

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

    schedule_slug = _schedule_slug(update_schedule, update_max_round_gap)
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
        update_schedule=f"global_round_{schedule_slug}_before_action",
    )
    row["confidence_cap"] = float(confidence_cap)
    row["confidence_matrix_state"] = (
        f"frozen_within_each_{schedule_slug}_epoch"
    )
    row["theta_warm_started"] = not zero_start
    return _attach_action_payload(
        _finish_row(
            row,
            routed=routed,
            correct=correct,
            model_updates=model_updates,
            last_training_count=last_training_count,
        ),
        actions_all,
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
    update_schedule: str,
    update_max_round_gap: int = DEFAULT_ADAPTIVE_MAX_ROUND_GAP,
) -> dict[str, Any]:
    """Run SquareCB.PMSide with scheduled predictor snapshots."""
    if policy not in {POLICY_IGW_TREE, POLICY_IGW_LINEAR}:
        raise ValueError(
            f"Unsupported SquareCB.PMSide policy identifier: {policy}"
        )
    if (policy == POLICY_IGW_LINEAR) != (
        estimator_settings["kind"] == "logistic"
    ):
        raise ValueError(
            "SquareCB.PMSide policy identifier and estimator kind do not match"
        )
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
    actions_all = np.zeros(n, dtype=bool)

    for boundary, start, stop in _adaptive_epochs(
        n, update_schedule, update_max_round_gap
    ):
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
        actions_all[start:stop] = actions
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

    schedule_slug = _schedule_slug(update_schedule, update_max_round_gap)
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
        update_schedule=f"global_round_{schedule_slug}_before_action",
    )
    row.update(
        {
            "squarecb_pmside_mu": float(mu),
            "squarecb_pmside_min_propensity": float(min_propensity),
            "inverse_propensity_weight_cap": float(1.0 / min_propensity),
            "estimator_feedback_update": (
                f"buffered_incremental_at_{schedule_slug}_boundaries"
                if estimator_settings["kind"] == "river-hoeffding"
                else f"full_history_refit_at_{schedule_slug}_boundaries"
            ),
            "estimator_profile": estimator_settings,
            "comparison_role": (
                "linear_oracle"
                if policy == POLICY_IGW_LINEAR
                else "nonlinear_tree_oracle"
            ),
        }
    )
    return _attach_action_payload(
        _finish_row(
            row,
            routed=routed,
            correct=correct,
            model_updates=model_updates,
            last_training_count=trained_count,
        ),
        actions_all,
    )


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
    multiplier: float | None,
    order_index: int,
) -> Path:
    parameter_directory = (
        "fixed" if multiplier is None else f"multiplier-{_float_slug(multiplier)}"
    )
    return (
        output
        / "checkpoints"
        / policy.lower()
        / f"l01-{_float_slug(l01)}"
        / parameter_directory
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
    try:
        _decode_action_payload(row)
    except RuntimeError:
        # A pre-learning-curve or partially written checkpoint is incomplete
        # for this design.  Treat it as missing so a normal resume reruns only
        # that trajectory; --plot-only will report it as missing.
        return None
    return row


def _save_checkpoint(path: Path, row: dict[str, Any], fingerprint: str) -> None:
    _decode_action_payload(row)
    payload = dict(row)
    payload["config_fingerprint"] = fingerprint
    _atomic_write_json(path, payload)


def _aggregate_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, float, float, float | None], list[dict[str, Any]]
    ] = {}
    for row in rows:
        multiplier = row.get("multiplier")
        key = (
            str(row["policy"]),
            float(row["l01"]),
            float(row["l11"]),
            None if multiplier is None else float(multiplier),
        )
        groups.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for key in sorted(
        groups,
        key=lambda item: (
            POLICY_OUTPUT_RANK.get(item[0], len(POLICY_OUTPUT_RANK)),
            item[1],
            math.inf if item[3] is None else item[3],
        ),
    ):
        group = sorted(groups[key], key=lambda row: int(row["order_run"]))
        first = group[0]
        result = {
            name: value
            for name, value in first.items()
            if name not in INTERNAL_ACTION_FIELDS
            and name
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
            "pointwise two-sided 95% Student-t confidence interval for the "
            "mean across paired shuffled online orders"
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
        if row["policy"] not in TUNED_POLICIES:
            continue
        key = (str(row["policy"]), float(row["l01"]), float(row["l11"]))
        groups.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(
        groups,
        key=lambda item: (
            POLICY_OUTPUT_RANK.get(item[0], len(POLICY_OUTPUT_RANK)),
            item[1],
        ),
    ):
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
                    "exploratory pointwise selection envelope; optimistically "
                    "selected and evaluated on the same orders"
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
        if key in lookup and float(row["multiplier"]) == lookup[key]:
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
    for igw_tree in selected_order_rows:
        if igw_tree["policy"] != POLICY_IGW_TREE:
            continue
        n = int(igw_tree["examples"])
        rate = float(igw_tree["routing_rate"])
        accuracy = 1.0 - (1.0 - rate) * disagreement_count / n
        row = {
            "policy": "Random",
            "method": METHOD_LABELS["random"],
            "l01": float(igw_tree["l01"]),
            "l11": float(igw_tree["l11"]),
            "alpha": float(igw_tree["alpha"]),
            "multiplier": None,
            "selected_multiplier": None,
            "parameter_name": (
                "matched_squarecb_pmside_tree_routing_rate"
            ),
            "base_parameter": None,
            "effective_parameter": rate,
            "order_run": int(igw_tree["order_run"]),
            "order_seed": int(igw_tree["order_seed"]),
            "policy_seed": None,
            "order_was_shuffled": True,
            "examples": n,
            "routing_rate": rate,
            "accuracy": float(accuracy),
            "model_updates": 0,
            "last_model_training_count": 0,
            "random_baseline": (
                "analytic expectation conditional on selected "
                "SquareCB.PMSide tree traffic"
            ),
            "matched_squarecb_pmside_policy": POLICY_IGW_TREE,
            "matched_squarecb_pmside_probability_estimator": igw_tree[
                "probability_estimator"
            ],
            "matched_squarecb_pmside_multiplier": float(
                igw_tree["selected_multiplier"]
            ),
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
    for key in sorted(
        groups,
        key=lambda item: (
            POLICY_OUTPUT_RANK.get(item[0], len(POLICY_OUTPUT_RANK)),
            item[1],
        ),
    ):
        group = sorted(groups[key], key=lambda row: int(row["order_run"]))
        first = group[0]
        result = {
            name: value
            for name, value in first.items()
            if name not in INTERNAL_ACTION_FIELDS
            and name
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
            "pointwise two-sided 95% Student-t confidence interval for the "
            "mean across paired shuffled online orders"
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
    """Return one paired order-level SquareCB.PMSide estimator difference."""
    if tree["order_seed"] != linear["order_seed"]:
        raise RuntimeError(
            "Paired SquareCB.PMSide rows have different online orders"
        )
    if int(tree["examples"]) != int(linear["examples"]):
        raise RuntimeError(
            "Paired SquareCB.PMSide rows have different online horizons"
        )
    for name in ("l01", "l11"):
        if not np.isclose(float(tree[name]), float(linear[name])):
            raise RuntimeError(
                f"Paired SquareCB.PMSide rows have different {name} values"
            )

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
        "positive_cost_reduction_favors": "SquareCB.PMSide tree",
        "selection_evaluation_reuse": bool(
            tree.get("selection_evaluation_reuse", False)
            or linear.get("selection_evaluation_reuse", False)
        ),
    }


def _igw_comparison_by_order(
    selected_order_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair separately tuned SquareCB.PMSide variants on each order."""
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
            "Selected SquareCB.PMSide tree and linear rows are not paired "
            "by loss/order"
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
    """Pair tree and linear SquareCB.PMSide at the same gamma multiplier."""
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
            "SquareCB.PMSide tree and linear candidates are not paired by "
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
            raise RuntimeError(
                "Matched SquareCB.PMSide candidates have different gamma values"
            )
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
            "pointwise two-sided 95% Student-t confidence interval for the "
            "mean paired order-level difference"
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


def _cross_fitted_hgb_reference(
    contexts: np.ndarray,
    outcomes: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    """Return row-aligned probabilities from a fixed OOF HGB-15 reference."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X = np.asarray(contexts, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.int8)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
        raise RuntimeError("Reference contexts and outcomes have invalid shapes")
    if not np.all(np.isfinite(X)):
        raise RuntimeError("Reference contexts must be finite")
    if np.any((y != 0) & (y != 1)):
        raise RuntimeError("Reference outcomes must be binary")
    class_counts = np.bincount(y, minlength=2)
    if folds < 2 or np.any(class_counts < folds):
        raise RuntimeError(
            "The cross-fitted reference requires at least one example from "
            "each class in every fold"
        )

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    probabilities = np.full(y.size, np.nan, dtype=np.float64)
    fold_index = np.full(y.size, -1, dtype=np.int16)
    for fold, (train_index, held_out_index) in enumerate(splitter.split(X, y)):
        backend = make_tree_backend({"kind": "hgb"}, seed=seed)
        backend.fit_all(X[train_index], y[train_index])
        probabilities[held_out_index] = backend.predict_proba(
            X[held_out_index]
        )
        fold_index[held_out_index] = fold

    if np.any(fold_index < 0) or not np.all(np.isfinite(probabilities)):
        raise RuntimeError("Cross-fitted reference did not predict every row")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return {
        "probability": probabilities,
        "fold_index": fold_index,
        "folds": int(folds),
        "seed": int(seed),
        "method": f"{folds}-fold cross-fitted HGB leaves=15 reference",
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probabilities)),
    }


def _reference_policy_rows(
    reference: dict[str, Any],
    outcomes: np.ndarray,
    l01_values: Sequence[float],
    l11: float,
) -> list[dict[str, Any]]:
    """Summarize the fixed cross-fitted reference at every loss threshold."""
    probabilities = np.asarray(reference["probability"], dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.int8)
    rows: list[dict[str, Any]] = []
    for l01 in l01_values:
        alpha = 1.0 / (1.0 + float(l01) - l11)
        actions = probabilities >= alpha
        routing_rate = float(np.mean(actions))
        accuracy = float(np.mean(actions | (outcomes == 0)))
        row = {
            "method": reference["method"],
            "l01": float(l01),
            "l11": float(l11),
            "alpha": float(alpha),
            "examples": int(outcomes.size),
            "routing_rate": routing_rate,
            "accuracy": accuracy,
            "reference_folds": int(reference["folds"]),
            "reference_seed": int(reference["seed"]),
            "reference_probability_tie_rule": "route strong when p_hat >= alpha",
            "reference_each_row_held_out": True,
        }
        row.update(
            _realized_cost_metrics(
                l01=float(l01),
                l11=float(l11),
                routing_rate=routing_rate,
                accuracy=accuracy,
                examples=outcomes.size,
            )
        )
        rows.append(row)
    return rows


def _write_reference_predictions_npz(
    path: Path,
    reference: dict[str, Any],
    outcomes: np.ndarray,
    example_ids: Sequence[str],
) -> None:
    """Persist the fixed row-aligned reference needed to audit the comparator."""
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(1, dtype=np.int16),
        example_id=np.asarray(example_ids, dtype=np.str_),
        outcome=np.asarray(outcomes, dtype=np.int8),
        fold_index=np.asarray(reference["fold_index"], dtype=np.int16),
        oof_disagreement_probability=np.asarray(
            reference["probability"], dtype=np.float64
        ),
        folds=np.asarray(reference["folds"], dtype=np.int16),
        seed=np.asarray(reference["seed"], dtype=np.int64),
        method=np.asarray(reference["method"]),
        hgb_profile=np.asarray(json.dumps(ONLINE_HGB_PROFILE, sort_keys=True)),
        outcome_source=np.asarray("cached_weak_strong_disagreement"),
        each_row_held_out=np.asarray(True),
    )
    temporary.replace(path)


def _build_selected_learning_curves(
    selected_order_rows: list[dict[str, Any]],
    outcomes: np.ndarray,
    permutations: np.ndarray,
    reference_probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reconstruct selected curves against a fixed cross-fitted reference."""
    outcomes = np.asarray(outcomes, dtype=np.int8)
    permutations = np.asarray(permutations)
    reference_probabilities = np.asarray(
        reference_probabilities, dtype=np.float64
    )
    if permutations.ndim != 2 or permutations.shape[1] != outcomes.size:
        raise RuntimeError("Learning-curve permutations have an invalid shape")
    if np.any((outcomes != 0) & (outcomes != 1)):
        raise RuntimeError("Learning curves require binary disagreement outcomes")
    if (
        reference_probabilities.ndim != 1
        or reference_probabilities.size != outcomes.size
        or not np.all(np.isfinite(reference_probabilities))
        or np.any(
            (reference_probabilities < 0.0)
            | (reference_probabilities > 1.0)
        )
    ):
        raise RuntimeError("Reference probabilities are invalid or misaligned")

    sources: list[tuple[dict[str, Any], bool]] = [
        (row, False) for row in selected_order_rows
    ]
    sources.extend(
        (row, True)
        for row in selected_order_rows
        if row["policy"] == POLICY_IGW_TREE
    )
    sources.sort(
        key=lambda item: (
            POLICY_OUTPUT_RANK.get(
                "Random" if item[1] else str(item[0]["policy"]),
                len(POLICY_OUTPUT_RANK),
            ),
            float(item[0]["l01"]),
            int(item[0]["order_run"]),
        )
    )

    trajectory_count = len(sources)
    horizon = outcomes.size
    cumulative_cost = np.empty((trajectory_count, horizon), dtype=np.float32)
    cumulative_reference_cost = np.empty(
        (trajectory_count, horizon), dtype=np.float32
    )
    cumulative_reference_regret = np.empty_like(cumulative_reference_cost)
    average_reference_regret = np.empty_like(cumulative_reference_cost)
    cumulative_clairvoyant_excess_cost = np.empty_like(
        cumulative_reference_cost
    )
    policies: list[str] = []
    methods: list[str] = []
    trajectory_kinds: list[str] = []
    l01_values = np.empty(trajectory_count, dtype=np.float64)
    l11_values = np.empty(trajectory_count, dtype=np.float64)
    order_runs = np.empty(trajectory_count, dtype=np.int32)
    order_seeds = np.empty(trajectory_count, dtype=np.int64)
    selected_multipliers = np.full(trajectory_count, np.nan, dtype=np.float64)
    effective_parameters = np.full(trajectory_count, np.nan, dtype=np.float64)
    matched_routing_rates = np.full(trajectory_count, np.nan, dtype=np.float64)

    for index, (source, is_random) in enumerate(sources):
        l01 = float(source["l01"])
        l11 = float(source["l11"])
        if l11 != 1.0 or l01 < 1.0:
            raise RuntimeError(
                "Learning-curve study requires l11=1 and l01>=1"
            )
        order_index = int(source["order_run"]) - 1
        if not 0 <= order_index < permutations.shape[0]:
            raise RuntimeError("Learning-curve order index is out of range")
        y_ordered = outcomes[permutations[order_index]].astype(
            np.float64, copy=False
        )
        reference_probability_ordered = reference_probabilities[
            permutations[order_index]
        ]
        reference_threshold = 1.0 / (1.0 + l01 - l11)
        reference_actions = reference_probability_ordered >= reference_threshold
        reference_action_values = reference_actions.astype(
            np.float64, copy=False
        )
        reference_increments = reference_action_values + (
            (1.0 - reference_action_values) * l01 * y_ordered
        )

        if is_random:
            policy = "Random"
            method = METHOD_LABELS["random"]
            trajectory_kind = "analytic_expected"
            routing_rate = float(source["routing_rate"])
            increments = (
                routing_rate + (1.0 - routing_rate) * l01 * y_ordered
            )
            effective_parameters[index] = routing_rate
            matched_routing_rates[index] = routing_rate
        else:
            policy = str(source["policy"])
            method = str(source["method"])
            trajectory_kind = "realized"
            actions = _decode_action_payload(source)
            if actions.size != horizon:
                raise RuntimeError(
                    "Selected action trajectory differs from the online horizon"
                )
            action_values = actions.astype(np.float64, copy=False)
            increments = action_values + (
                (1.0 - action_values) * l01 * y_ordered
            )
            selected = source.get("selected_multiplier")
            if selected is not None:
                selected_multipliers[index] = float(selected)
            effective = source.get("effective_parameter")
            if effective is not None:
                effective_parameters[index] = float(effective)

        cost_curve = np.cumsum(increments, dtype=np.float64)
        reference_cost_curve = np.cumsum(
            reference_increments, dtype=np.float64
        )
        reference_regret_curve = np.cumsum(
            increments - reference_increments, dtype=np.float64
        )
        average_regret_curve = reference_regret_curve / np.arange(
            1, horizon + 1, dtype=np.float64
        )
        clairvoyant_excess_curve = np.cumsum(
            increments - y_ordered, dtype=np.float64
        )
        if not is_random and horizon and not np.isclose(
            cost_curve[-1],
            float(source["realized_total_cost"]),
            rtol=1e-10,
            atol=1e-8,
        ):
            raise RuntimeError(
                "Decoded actions do not reproduce selected realized total cost"
            )
        if horizon and not np.isclose(
            reference_regret_curve[-1],
            cost_curve[-1] - reference_cost_curve[-1],
            rtol=1e-12,
            atol=1e-8,
        ):
            raise RuntimeError("Reference-regret terminal identity failed")

        cumulative_cost[index] = cost_curve.astype(np.float32)
        cumulative_reference_cost[index] = reference_cost_curve.astype(
            np.float32
        )
        cumulative_reference_regret[index] = reference_regret_curve.astype(
            np.float32
        )
        average_reference_regret[index] = average_regret_curve.astype(
            np.float32
        )
        cumulative_clairvoyant_excess_cost[index] = (
            clairvoyant_excess_curve.astype(np.float32)
        )
        policies.append(policy)
        methods.append(method)
        trajectory_kinds.append(trajectory_kind)
        l01_values[index] = l01
        l11_values[index] = l11
        order_runs[index] = int(source["order_run"])
        order_seeds[index] = int(source["order_seed"])

    return {
        "round": np.arange(1, horizon + 1, dtype=np.int32),
        "policy": np.asarray(policies, dtype=np.str_),
        "method": np.asarray(methods, dtype=np.str_),
        "trajectory_kind": np.asarray(trajectory_kinds, dtype=np.str_),
        "l01": l01_values,
        "l11": l11_values,
        "order_run": order_runs,
        "order_seed": order_seeds,
        "selected_multiplier": selected_multipliers,
        "effective_parameter": effective_parameters,
        "matched_routing_rate": matched_routing_rates,
        "cumulative_cost": cumulative_cost,
        "cumulative_reference_cost": cumulative_reference_cost,
        "cumulative_reference_regret": cumulative_reference_regret,
        "average_reference_regret": average_reference_regret,
        "cumulative_clairvoyant_excess_cost": (
            cumulative_clairvoyant_excess_cost
        ),
    }


def _write_learning_curves_npz(
    path: Path, curves: dict[str, np.ndarray]
) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(2, dtype=np.int16),
        storage_dtype=np.asarray("float32"),
        cost_increment_definition=np.asarray(
            REALIZED_COST_INCREMENT_DEFINITION
        ),
        clairvoyant_cost_increment_definition=np.asarray(
            CLAIRVOYANT_COST_INCREMENT_DEFINITION
        ),
        regret_increment_definition=np.asarray(
            REGRET_INCREMENT_DEFINITION
        ),
        clairvoyant_excess_increment_definition=np.asarray(
            CLAIRVOYANT_EXCESS_INCREMENT_DEFINITION
        ),
        random_increment_definition=np.asarray(
            RANDOM_COST_INCREMENT_DEFINITION
        ),
        selection_warning=np.asarray(LEARNING_CURVE_SELECTION_WARNING),
        **curves,
    )
    temporary.replace(path)


def _aggregate_learning_curves(
    curves: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float, float], list[int]] = {}
    for index, (policy, method, l01, l11) in enumerate(
        zip(
            curves["policy"],
            curves["method"],
            curves["l01"],
            curves["l11"],
        )
    ):
        key = (str(policy), str(method), float(l01), float(l11))
        groups.setdefault(key, []).append(index)

    aggregated: list[dict[str, Any]] = []
    for key in sorted(
        groups,
        key=lambda item: (
            POLICY_OUTPUT_RANK.get(item[0], len(POLICY_OUTPUT_RANK)),
            item[2],
        ),
        ):
        indices = np.asarray(groups[key], dtype=np.int64)
        repeats = indices.size
        confidence_df = int(repeats - 1)
        confidence_t_critical = (
            student_t_critical_value(
                int(repeats),
                confidence_level=PLOT_CONFIDENCE_LEVEL,
            )
            if repeats > 1
            else math.nan
        )
        curve_names = (
            "cumulative_cost",
            "cumulative_reference_cost",
            "cumulative_reference_regret",
            "average_reference_regret",
            "cumulative_clairvoyant_excess_cost",
        )
        curve_statistics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in curve_names:
            values = curves[name][indices]
            standard_deviation = (
                np.std(values, axis=0, ddof=1, dtype=np.float64)
                if repeats > 1
                else np.zeros(values.shape[1], dtype=np.float64)
            )
            curve_statistics[name] = (
                np.mean(values, axis=0, dtype=np.float64),
                standard_deviation,
            )
        effective = curves["effective_parameter"][indices]
        matched = curves["matched_routing_rate"][indices]
        selected = curves["selected_multiplier"][indices]
        trajectory_kinds = np.unique(curves["trajectory_kind"][indices])
        if trajectory_kinds.size != 1:
            raise RuntimeError(
                "A learning-curve group mixes realized and expected trajectories"
            )
        finite_selected = selected[np.isfinite(selected)]
        if finite_selected.size and not np.allclose(
            finite_selected, finite_selected[0]
        ):
            raise RuntimeError(
                "Selected multiplier differs across paired learning curves"
            )

        def finite_mean_std(values: np.ndarray) -> tuple[float, float]:
            finite = values[np.isfinite(values)]
            if not finite.size:
                return math.nan, math.nan
            mean = float(np.mean(finite))
            std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            return mean, std

        effective_mean, effective_std = finite_mean_std(effective)
        matched_mean, matched_std = finite_mean_std(matched)
        result = {
                "policy": key[0],
                "method": key[1],
                "trajectory_kind": str(trajectory_kinds[0]),
                "l01": key[2],
                "l11": key[3],
                "online_order_repeats": int(repeats),
                "confidence_level": PLOT_CONFIDENCE_LEVEL,
                "confidence_df": confidence_df,
                "confidence_t_critical": confidence_t_critical,
                "selected_multiplier": (
                    float(finite_selected[0])
                    if finite_selected.size
                    else math.nan
                ),
                "effective_parameter_mean": effective_mean,
                "effective_parameter_std": effective_std,
                "matched_routing_rate_mean": matched_mean,
                "matched_routing_rate_std": matched_std,
                "round": curves["round"],
            }
        for name, (mean, standard_deviation) in curve_statistics.items():
            sem = standard_deviation / np.sqrt(repeats)
            result[f"{name}_mean"] = mean
            result[f"{name}_std"] = standard_deviation
            result[f"{name}_sem"] = sem
            if name in {
                "cumulative_reference_regret",
                "average_reference_regret",
            }:
                margin = confidence_t_critical * sem
                result[f"{name}_ci95_lower"] = mean - margin
                result[f"{name}_ci95_upper"] = mean + margin
        aggregated.append(result)
    return aggregated


def _write_learning_curve_aggregate_csv(
    path: Path, aggregated: list[dict[str, Any]]
) -> None:
    fieldnames = [
        "policy",
        "method",
        "trajectory_kind",
        "l01",
        "l11",
        "round",
        "online_order_repeats",
        "confidence_level",
        "confidence_df",
        "confidence_t_critical",
        "selected_multiplier",
        "effective_parameter_mean",
        "effective_parameter_std",
        "matched_routing_rate_mean",
        "matched_routing_rate_std",
        "cumulative_cost_mean",
        "cumulative_cost_std",
        "cumulative_cost_sem",
        "cumulative_reference_cost_mean",
        "cumulative_reference_cost_std",
        "cumulative_reference_cost_sem",
        "cumulative_reference_regret_mean",
        "cumulative_reference_regret_std",
        "cumulative_reference_regret_sem",
        "cumulative_reference_regret_ci95_lower",
        "cumulative_reference_regret_ci95_upper",
        "average_reference_regret_mean",
        "average_reference_regret_std",
        "average_reference_regret_sem",
        "average_reference_regret_ci95_lower",
        "average_reference_regret_ci95_upper",
        "cumulative_clairvoyant_excess_cost_mean",
        "cumulative_clairvoyant_excess_cost_std",
        "cumulative_clairvoyant_excess_cost_sem",
    ]

    def csv_scalar(value: Any) -> Any:
        if isinstance(value, (float, np.floating)) and not math.isfinite(
            float(value)
        ):
            return ""
        return value

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for group in aggregated:
            scalar = {
                name: csv_scalar(group[name])
                for name in fieldnames
                if name
                not in {
                    "round",
                    "cumulative_cost_mean",
                    "cumulative_cost_std",
                    "cumulative_cost_sem",
                    "cumulative_reference_cost_mean",
                    "cumulative_reference_cost_std",
                    "cumulative_reference_cost_sem",
                    "cumulative_reference_regret_mean",
                    "cumulative_reference_regret_std",
                    "cumulative_reference_regret_sem",
                    "cumulative_reference_regret_ci95_lower",
                    "cumulative_reference_regret_ci95_upper",
                    "average_reference_regret_mean",
                    "average_reference_regret_std",
                    "average_reference_regret_sem",
                    "average_reference_regret_ci95_lower",
                    "average_reference_regret_ci95_upper",
                    "cumulative_clairvoyant_excess_cost_mean",
                    "cumulative_clairvoyant_excess_cost_std",
                    "cumulative_clairvoyant_excess_cost_sem",
                }
            }
            for round_index in range(len(group["round"])):
                writer.writerow(
                    scalar
                    | {
                        "round": int(group["round"][round_index]),
                        "cumulative_cost_mean": float(
                            group["cumulative_cost_mean"][round_index]
                        ),
                        "cumulative_cost_std": float(
                            group["cumulative_cost_std"][round_index]
                        ),
                        "cumulative_cost_sem": float(
                            group["cumulative_cost_sem"][round_index]
                        ),
                        "cumulative_reference_cost_mean": float(
                            group["cumulative_reference_cost_mean"][round_index]
                        ),
                        "cumulative_reference_cost_std": float(
                            group["cumulative_reference_cost_std"][round_index]
                        ),
                        "cumulative_reference_cost_sem": float(
                            group["cumulative_reference_cost_sem"][round_index]
                        ),
                        "cumulative_reference_regret_mean": float(
                            group["cumulative_reference_regret_mean"][round_index]
                        ),
                        "cumulative_reference_regret_std": float(
                            group["cumulative_reference_regret_std"][round_index]
                        ),
                        "cumulative_reference_regret_sem": float(
                            group["cumulative_reference_regret_sem"][round_index]
                        ),
                        "cumulative_reference_regret_ci95_lower": float(
                            group[
                                "cumulative_reference_regret_ci95_lower"
                            ][round_index]
                        ),
                        "cumulative_reference_regret_ci95_upper": float(
                            group[
                                "cumulative_reference_regret_ci95_upper"
                            ][round_index]
                        ),
                        "average_reference_regret_mean": float(
                            group["average_reference_regret_mean"][round_index]
                        ),
                        "average_reference_regret_std": float(
                            group["average_reference_regret_std"][round_index]
                        ),
                        "average_reference_regret_sem": float(
                            group["average_reference_regret_sem"][round_index]
                        ),
                        "average_reference_regret_ci95_lower": float(
                            group[
                                "average_reference_regret_ci95_lower"
                            ][round_index]
                        ),
                        "average_reference_regret_ci95_upper": float(
                            group[
                                "average_reference_regret_ci95_upper"
                            ][round_index]
                        ),
                        "cumulative_clairvoyant_excess_cost_mean": float(
                            group[
                                "cumulative_clairvoyant_excess_cost_mean"
                            ][round_index]
                        ),
                        "cumulative_clairvoyant_excess_cost_std": float(
                            group[
                                "cumulative_clairvoyant_excess_cost_std"
                            ][round_index]
                        ),
                        "cumulative_clairvoyant_excess_cost_sem": float(
                            group[
                                "cumulative_clairvoyant_excess_cost_sem"
                            ][round_index]
                        ),
                    }
                )
    temporary.replace(path)


def _plot_selected_cumulative_reference_regret(
    output: Path, aggregated: list[dict[str, Any]]
) -> list[Path]:
    plt = publication_pyplot()

    for prefix in (
        LEGACY_LEARNING_CURVE_PLOT_PREFIX,
        REVISION7_REGRET_PLOT_PREFIX,
    ):
        for suffix in ("png", "pdf"):
            for legacy_path in output.glob(f"{prefix}*.{suffix}"):
                legacy_path.unlink()

    paths: list[Path] = []
    for l01 in sorted({float(group["l01"]) for group in aggregated}):
        figure, axis = plt.subplots(
            figsize=PUBLICATION_FIGSIZE,
            constrained_layout=True,
        )
        for group in (
            item for item in aggregated if float(item["l01"]) == l01
        ):
            rounds = group["round"]
            mean = group["cumulative_reference_regret_mean"]
            lower = group["cumulative_reference_regret_ci95_lower"]
            upper = group["cumulative_reference_regret_ci95_upper"]
            (line,) = axis.plot(rounds, mean, label=group["method"])
            axis.fill_between(
                rounds,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.15,
                linewidth=0,
            )
        axis.set_xlabel(AXIS_LABELS["round"])
        axis.set_ylabel(AXIS_LABELS["cumulative_reference_regret"])
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axis.grid(alpha=0.25)
        axis.legend()
        path = output / (
            f"{LEARNING_CURVE_PLOT_PREFIX}{_float_slug(l01)}.png"
        )
        save_publication_figure(figure, path)
        plt.close(figure)
        paths.append(path)
    return paths


def _plot_selected_average_reference_regret(
    output: Path, aggregated: list[dict[str, Any]]
) -> list[Path]:
    plt = publication_pyplot()

    paths: list[Path] = []
    for l01 in sorted({float(group["l01"]) for group in aggregated}):
        figure, axis = plt.subplots(
            figsize=PUBLICATION_FIGSIZE,
            constrained_layout=True,
        )
        for group in (
            item for item in aggregated if float(item["l01"]) == l01
        ):
            rounds = group["round"]
            mean = group["average_reference_regret_mean"]
            lower = group["average_reference_regret_ci95_lower"]
            upper = group["average_reference_regret_ci95_upper"]
            (line,) = axis.plot(rounds, mean, label=group["method"])
            axis.fill_between(
                rounds,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.15,
                linewidth=0,
            )
        axis.set_xlabel(AXIS_LABELS["round"])
        axis.set_ylabel(AXIS_LABELS["average_reference_regret"])
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axis.grid(alpha=0.25)
        axis.legend()
        path = output / (
            f"{AVERAGE_REGRET_PLOT_PREFIX}{_float_slug(l01)}.png"
        )
        save_publication_figure(figure, path)
        plt.close(figure)
        paths.append(path)
    return paths


def _plot_selected_routing_accuracy(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    plt = publication_pyplot()

    figure, axis = plt.subplots(
        figsize=PUBLICATION_FIGSIZE,
        constrained_layout=True,
    )
    for method in dict.fromkeys(str(row["method"]) for row in rows):
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: float(row["l01"]),
        )
        axis.errorbar(
            [row["routing_rate_mean"] for row in selected],
            [row["accuracy_mean"] for row in selected],
            xerr=[
                student_t_half_width(
                    row["routing_rate_std"],
                    int(row["online_order_repeats"]),
                )
                for row in selected
            ],
            yerr=[
                student_t_half_width(
                    row["accuracy_std"],
                    int(row["online_order_repeats"]),
                )
                for row in selected
            ],
            fmt="-o",
            capsize=3,
            label=method,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(AXIS_LABELS["routing_rate"])
    axis.set_ylabel(AXIS_LABELS["accuracy"])
    axis.grid(alpha=0.25)
    axis.legend()
    save_publication_figure(figure, path)
    plt.close(figure)


def _plot_selected_cost(path: Path, rows: list[dict[str, Any]]) -> None:
    plt = publication_pyplot()

    figure, axis = plt.subplots(
        figsize=PUBLICATION_FIGSIZE,
        constrained_layout=True,
    )
    for method in dict.fromkeys(str(row["method"]) for row in rows):
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: float(row["l01"]),
        )
        axis.errorbar(
            [row["l01"] for row in selected],
            [row["realized_total_cost_mean"] for row in selected],
            yerr=[
                student_t_half_width(
                    row["realized_total_cost_std"],
                    int(row["online_order_repeats"]),
                )
                for row in selected
            ],
            fmt="-o",
            capsize=3,
            label=method,
    )
    ticks = sorted({float(row["l01"]) for row in rows})
    axis.set_xticks(ticks, [f"{value:g}" for value in ticks])
    axis.set_xlabel(AXIS_LABELS["l01"])
    axis.set_ylabel(AXIS_LABELS["total_cost"])
    axis.grid(alpha=0.25)
    axis.legend()
    save_publication_figure(figure, path)
    plt.close(figure)


def _plot_selected_multipliers(path: Path, selections: list[dict[str, Any]]) -> None:
    plt = publication_pyplot()

    figure, axis = plt.subplots(
        figsize=PUBLICATION_FIGSIZE_SHORT,
        constrained_layout=True,
    )
    labels = {
        POLICY_CBPSIDE: "CBPSide",
        POLICY_IGW_LINEAR: "SquareCB.PMSide Linear",
        POLICY_IGW_TREE: "SquareCB.PMSide Tree",
        POLICY_PGTS: METHOD_LABELS["pgts"],
    }
    for policy in TUNED_POLICIES:
        selected = sorted(
            (row for row in selections if row["policy"] == policy),
            key=lambda row: float(row["l01"]),
        )
        if not selected:
            continue
        axis.plot(
            [row["l01"] for row in selected],
            [row["selected_multiplier"] for row in selected],
            "-o",
            label=labels[policy],
        )
    ticks = sorted({float(row["l01"]) for row in selections})
    axis.set_xticks(ticks, [f"{value:g}" for value in ticks])
    axis.set_yscale("log")
    axis.set_xlabel(AXIS_LABELS["l01"])
    axis.set_ylabel(AXIS_LABELS["selected_multiplier"])
    axis.grid(alpha=0.25)
    axis.legend()
    save_publication_figure(figure, path)
    plt.close(figure)


def _plot_igw_estimator_comparison(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    plt = publication_pyplot()

    selected = sorted(rows, key=lambda row: float(row["l01"]))
    figure, axis = plt.subplots(
        figsize=PUBLICATION_FIGSIZE_SHORT,
        constrained_layout=True,
    )
    axis.errorbar(
        [row["l01"] for row in selected],
        [row["nonlinear_tree_cost_reduction_mean"] for row in selected],
        yerr=[
            student_t_half_width(
                row["nonlinear_tree_cost_reduction_std"],
                int(row["online_order_repeats"]),
            )
            for row in selected
        ],
        fmt="-o",
        capsize=3,
        color="tab:purple",
    )
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ticks = [float(row["l01"]) for row in selected]
    axis.set_xticks(ticks, [f"{value:g}" for value in ticks])
    axis.set_xlabel(AXIS_LABELS["l01"])
    axis.set_ylabel(AXIS_LABELS["squarecb_cost_difference"])
    axis.grid(alpha=0.25)
    save_publication_figure(figure, path)
    plt.close(figure)


def _plot_igw_matched_estimator_comparison(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    plt = publication_pyplot()

    figure, axis = plt.subplots(
        figsize=PUBLICATION_FIGSIZE,
        constrained_layout=True,
    )
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
            yerr=[
                student_t_half_width(
                    row["nonlinear_tree_cost_reduction_std"],
                    int(row["online_order_repeats"]),
                )
                for row in selected
            ],
            fmt="-o",
            capsize=3,
            label=f"{multiplier:g}",
        )
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ticks = sorted({float(row["l01"]) for row in rows})
    axis.set_xticks(ticks, [f"{value:g}" for value in ticks])
    axis.set_xlabel(AXIS_LABELS["l01"])
    axis.set_ylabel(AXIS_LABELS["squarecb_cost_difference"])
    axis.grid(alpha=0.25)
    axis.legend(title=r"Gamma multiplier")
    save_publication_figure(figure, path)
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
    active_tuned_policies = _active_tuned_policies(args)
    base_gamma = (
        float(args.igw_base_gamma)
        if args.igw_base_gamma is not None
        else float(np.sqrt(n))
    )
    identity = hashlib.sha256()
    for example_id in example_ids:
        identity.update(example_id.encode("utf-8"))
        identity.update(b"\0")
    identity.update(np.ascontiguousarray(contexts).tobytes())
    identity.update(np.ascontiguousarray(outcomes).tobytes())
    adaptive_boundaries = list(
        _schedule_boundaries(
            n,
            args.adaptive_update_schedule,
            args.adaptive_max_round_gap,
        )
    )
    if args.adaptive_update_schedule == "capped-doubling":
        boundary_rule = (
            "before global rounds starting at 1, with "
            f"next=min(2*current, current+{args.adaptive_max_round_gap})"
        )
    elif args.adaptive_update_schedule == "fibonacci":
        boundary_rule = "before global rounds 1,2,3,5,8,..."
    else:
        boundary_rule = "before global rounds 1,2,4,8,..."
    manifest = {
        "design": "pointwise-online-parameter-multiplier-sweep-v9",
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
        "tuned_policies": list(active_tuned_policies),
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
            "squarecb_pmside_gamma": base_gamma,
            "squarecb_pmside_gamma_rule": (
                "command_line" if args.igw_base_gamma is not None else "sqrt(n)"
            ),
            "pgts_prior_std": (
                float(args.pgts_prior_std) if args.include_pgts else None
            ),
        },
        "update_schedule": {
            "name": args.adaptive_update_schedule,
            "maximum_round_gap": (
                int(args.adaptive_max_round_gap)
                if args.adaptive_update_schedule == "capped-doubling"
                else None
            ),
            "boundary_rule": boundary_rule,
            "boundary_rounds": adaptive_boundaries,
            "boundary_count": len(adaptive_boundaries),
            "maximum_model_updates": max(0, len(adaptive_boundaries) - 1),
            "last_boundary_round": (
                adaptive_boundaries[-1] if adaptive_boundaries else None
            ),
            "final_frozen_epoch_rounds": (
                n - adaptive_boundaries[-1] + 1 if adaptive_boundaries else 0
            ),
            "history_cutoff": "feedback through boundary_round-1",
            "cbpside_theta": "refit only at boundary when new tastes exist",
            "cbpside_V_inverse": "recompute only at boundary and freeze in epoch",
            "cbpside_beta": "evaluate on every current context using epoch V inverse",
            "squarecb_pmside_tree": (
                "buffered incremental update only at boundary"
                if args.tree_estimator == "river-hoeffding"
                else "full revealed-history refit only at boundary"
            ),
            "squarecb_pmside_linear": (
                "full revealed-history refit only at boundary"
            ),
        },
        "tree": _tree_settings(args),
        "squarecb_pmside_tree_profile": _tree_settings(args),
        "squarecb_pmside_linear_profile": _linear_settings(),
        "squarecb_pmside_estimator_comparison": {
            "shared_protocol": [
                "contexts",
                "selective-feedback protocol (each policy realizes its own history)",
                "capped inverse-propensity weighting rule",
                "gamma multiplier grid",
                "policy random numbers",
                "global-round "
                f"{_schedule_slug(args.adaptive_update_schedule, args.adaptive_max_round_gap)} "
                "schedule",
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
        "squarecb_pmside_mu": float(args.igw_mu),
        "squarecb_pmside_min_propensity": float(args.igw_min_propensity),
        "selection": {
            "objective": "lowest mean realized total cost over all order runs",
            "tie_break": "closest multiplier to 1, then smaller multiplier",
            "pointwise_by": ["policy", "l01"],
            "policies": list(active_tuned_policies),
            "selection_and_evaluation_orders_are_the_same": True,
            "interpretation": "exploratory optimistic selection envelope",
            "error_bars": (
                "pointwise two-sided 95% Student-t confidence intervals for "
                "the mean across shuffled orders"
            ),
        },
        "random": (
            "analytic expected routing matched per order and l01 to selected "
            "SquareCB.PMSide tree traffic"
        ),
        "regret_reference": {
            "type": "fixed_row_aligned_cross_fitted_probability_reference",
            "method": (
                f"{args.reference_folds}-fold stratified out-of-fold HGB "
                "leaves=15"
            ),
            "folds": int(args.reference_folds),
            "splitter": "StratifiedKFold(shuffle=True)",
            "split_seed": int(args.seed),
            "model_seed": int(args.seed),
            "hgb_profile": ONLINE_HGB_PROFILE,
            "context_profile": args.context_profile,
            "context_dimension": int(contexts.shape[1]),
            "outcome_source": "cached_weak_strong_disagreement",
            "benchmark_gold_answers_used": False,
            "each_row_predicted_by_model_excluding_that_row": True,
            "action_rule": "route strong when p_hat >= 1/l01",
            "shared_across_online_policies_and_permutations": True,
            "interpretation": (
                "offline empirical reference assembled from fold models; "
                "not an outcome-aware oracle and not by itself a theorem test"
            ),
            "prediction_artifact": REFERENCE_PREDICTIONS_NPZ,
            "threshold_summary_csv": REFERENCE_RESULTS_CSV,
            "threshold_summary_json": REFERENCE_RESULTS_JSON,
        },
        "learning_curves": {
            "enabled": True,
            "selection_scope": "pointwise selected policy per l01",
            "checkpoint_action_encoding": (
                "np.packbits uint8 with bitorder=little, then base64"
            ),
            "checkpoint_action_payload_internal": True,
            "by_order_artifact": LEARNING_CURVE_NPZ,
            "by_order_curve_dtype": "float32",
            "aggregate_artifact": LEARNING_CURVE_CSV,
            "aggregate_curve_dtype": "float64",
            "cumulative_regret_plot_pattern": (
                f"{LEARNING_CURVE_PLOT_PREFIX}<float_slug>.png"
            ),
            "average_regret_plot_pattern": (
                f"{AVERAGE_REGRET_PLOT_PREFIX}<float_slug>.png"
            ),
            "primary_regret_metric": "cumulative_reference_regret",
            "normalized_diagnostic": "average_reference_regret=R_t/t",
            "cost_increment_definition": REALIZED_COST_INCREMENT_DEFINITION,
            "clairvoyant_cost_increment_definition": (
                CLAIRVOYANT_COST_INCREMENT_DEFINITION
            ),
            "regret_increment_definition": REGRET_INCREMENT_DEFINITION,
            "clairvoyant_excess_increment_definition": (
                CLAIRVOYANT_EXCESS_INCREMENT_DEFINITION
            ),
            "outcome_aware_diagnostic_retained_but_not_plotted": True,
            "random_increment_definition": RANDOM_COST_INCREMENT_DEFINITION,
            "selection_warning": LEARNING_CURVE_SELECTION_WARNING,
            "json_artifact": None,
        },
        "disabled_policies": ["ETC", "ETCLinear"],
        "fixed_policies": [],
        "candidate_policies": list(active_tuned_policies),
        "candidate_counts": {
            "multiplier_tuned_checkpoints": (
                len(active_tuned_policies)
                * len(args.l01_values)
                * len(args.multipliers)
                * args.online_order_repeats
            ),
            "pgts_prior_tuned_checkpoints": (
                len(args.l01_values)
                * len(args.multipliers)
                * args.online_order_repeats
                if args.include_pgts
                else 0
            ),
            "total_checkpoints": _expected_checkpoint_count(args),
            "multiplier_tuned_execution_groups": (
                len(active_tuned_policies)
                * len(args.l01_values)
                * len(args.multipliers)
            ),
            "pgts_prior_tuned_execution_groups": (
                len(args.l01_values) * len(args.multipliers)
                if args.include_pgts
                else 0
            ),
            "total_execution_groups": (
                len(active_tuned_policies)
                * len(args.l01_values)
                * len(args.multipliers)
            ),
        },
        "data_identity_sha256": identity.hexdigest(),
    }
    if args.include_pgts:
        manifest.update(
            {
                "pgts": {
                    "enabled": True,
                    "policy": POLICY_PGTS,
                    "algorithm": (
                        "PG-TS Algorithm 1 scheduled-update approximation"
                    ),
                    "algorithm1_exact": False,
                    "gibbs_steps_per_posterior_draw": int(
                        args.pgts_gibbs_steps
                    ),
                    "prior_mean": 0.0,
                    "base_prior_std": float(args.pgts_prior_std),
                    "prior_std_multiplier_grid": [
                        float(value) for value in args.multipliers
                    ],
                    "effective_prior_std_values": [
                        float(args.pgts_prior_std * value)
                        for value in args.multipliers
                    ],
                    "effective_prior_std_rule": (
                        "base_prior_std * selected_multiplier"
                    ),
                    "prior_covariance": (
                        "effective_prior_std^2 * identity"
                    ),
                    "context_preprocessing": (
                        "row L2 normalization using max(1, norm), then prepend "
                        "intercept"
                    ),
                    "posterior_draw": (
                        "final draw after M Gibbs transitions at an eligible "
                        "configured boundary"
                    ),
                    "update_schedule": (
                        "global-round "
                        f"{_schedule_slug(args.adaptive_update_schedule, args.adaptive_max_round_gap)} "
                        "boundaries"
                    ),
                    "update_rule": (
                        "initial prior draw at round 1; later boundary draws "
                        "only when new action-1 feedback arrived since the "
                        "previous draw; reuse theta within each epoch"
                    ),
                    "eligible_boundary_rounds": adaptive_boundaries,
                    "eligible_boundary_count": len(adaptive_boundaries),
                    "feedback": (
                        "only action-1 disagreement outcomes are revealed"
                    ),
                    "inverse_propensity_weighting": False,
                    "multiplier_tuned": True,
                    "selection_objective": (
                        "lowest mean realized total cost pointwise by l01"
                    ),
                    "optional_dependency": "polyagamma",
                },
            }
        )
        manifest["update_schedule"]["pgts"] = (
            "initial prior draw at round 1; thereafter draw only at configured "
            "boundaries with new action-1 feedback, and freeze theta within epoch"
        )
    encoded = json.dumps(_jsonable(manifest), sort_keys=True).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    # Plot uncertainty is presentation-only: keeping it outside the checkpoint
    # fingerprint lets completed revision-9 sweeps be regenerated with the new
    # interval style via --plot-only, without rerunning any online policy.
    manifest["learning_curves"]["plotted_uncertainty"] = {
        "type": "pointwise_student_t_confidence_interval_for_mean",
        "confidence_level": PLOT_CONFIDENCE_LEVEL,
        "degrees_of_freedom": "online_order_repeats - 1",
        "formula": (
            "mean +/- t.ppf((1 + confidence_level) / 2, df) * "
            "sample_sd / sqrt(n)"
        ),
        "simultaneous_band": False,
        "post_selection_adjusted": False,
    }
    manifest["plot_presentation"] = {
        "style_module": "llm_routing_simulation.plot_style",
        "formats": ["png", "pdf"],
        "png_dpi": PUBLICATION_PNG_DPI,
        "figure_titles": False,
        "figure_footnotes": False,
        "axis_labels": dict(AXIS_LABELS),
        "method_labels": dict(METHOD_LABELS),
        "replicated_run_uncertainty": {
            "type": "student_t_confidence_interval_for_mean",
            "confidence_level": PLOT_CONFIDENCE_LEVEL,
            "degrees_of_freedom": "online_order_repeats - 1",
            "simultaneous_band": False,
            "post_selection_adjusted": False,
        },
    }
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
        if existing != manifest:
            _atomic_write_json(manifest_path, manifest)
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
    return len(_active_tuned_policies(args)) * candidates_per_policy


def _load_all_expected_checkpoints(
    output: Path, args: argparse.Namespace, fingerprint: str
) -> tuple[
    list[dict[str, Any]], list[tuple[str, float, float | None, int]]
]:
    rows: list[dict[str, Any]] = []
    missing: list[tuple[str, float, float | None, int]] = []
    for policy in _active_tuned_policies(args):
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
) -> None:
    tree_settings = _tree_settings(args)
    linear_settings = _linear_settings()
    active_tuned_policies = _active_tuned_policies(args)
    total_candidate_groups = (
        len(active_tuned_policies)
        * len(args.l01_values)
        * len(args.multipliers)
    )
    group_number = 0

    # PG-TS tunes its Gaussian-prior scale over the common multiplier grid.
    # Each loss/multiplier/order has its own actions and revealed history.
    if args.include_pgts:
        for l01 in args.l01_values:
            for multiplier in args.multipliers:
                group_number += 1
                missing_orders = []
                for order_index in range(args.online_order_repeats):
                    path = _checkpoint_path(
                        output, POLICY_PGTS, l01, multiplier, order_index
                    )
                    if _load_checkpoint(path, fingerprint) is None:
                        missing_orders.append(order_index)
                print(
                    f"[{group_number}/{total_candidate_groups}] "
                    f"{_method_label(POLICY_PGTS, {})} l01={l01:g}, "
                    f"prior-std multiplier={multiplier:g}; "
                    f"{len(missing_orders)} order run(s) remaining",
                    flush=True,
                )
                tasks = [
                    {
                        "normalized_features": normalized_features,
                        "outcomes": outcomes,
                        "permutation": permutations[order_index],
                        "l01": l01,
                        "l11": args.l11,
                        "gibbs_steps": args.pgts_gibbs_steps,
                        "prior_std": args.pgts_prior_std,
                        "multiplier": multiplier,
                        "order_index": order_index,
                        "order_seed": args.seed + order_index,
                        "policy_seed": args.policy_seed,
                        "update_schedule": args.adaptive_update_schedule,
                        "update_max_round_gap": args.adaptive_max_round_gap,
                    }
                    for order_index in missing_orders
                ]
                for row in _parallel_map(
                    _simulate_pgts_candidate, tasks, args.jobs
                ):
                    path = _checkpoint_path(
                        output,
                        POLICY_PGTS,
                        l01,
                        multiplier,
                        int(row["order_run"]) - 1,
                    )
                    _save_checkpoint(path, row, fingerprint)

    for policy in BASE_TUNED_POLICIES:
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
                progress_settings = (
                    linear_settings
                    if policy == POLICY_IGW_LINEAR
                    else tree_settings
                )
                print(
                    f"[{group_number}/{total_candidate_groups}] "
                    f"{_method_label(policy, progress_settings)} "
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
                    "update_schedule": args.adaptive_update_schedule,
                    "update_max_round_gap": args.adaptive_max_round_gap,
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
    example_ids: Sequence[str],
    permutations: np.ndarray,
    manifest: dict[str, Any],
    reference: dict[str, Any],
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
            POLICY_OUTPUT_RANK.get(
                str(row["policy"]), len(POLICY_OUTPUT_RANK)
            ),
            float(row["l01"]),
            (
                math.inf
                if row.get("multiplier") is None
                else float(row["multiplier"])
            ),
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
    _write_reference_predictions_npz(
        output / REFERENCE_PREDICTIONS_NPZ,
        reference,
        outcomes,
        example_ids,
    )
    reference_rows = _reference_policy_rows(
        reference, outcomes, args.l01_values, args.l11
    )
    _write_csv(output / REFERENCE_RESULTS_CSV, reference_rows)
    _atomic_write_json(output / REFERENCE_RESULTS_JSON, reference_rows)
    learning_curves = _build_selected_learning_curves(
        selected_order_rows,
        outcomes,
        permutations,
        reference["probability"],
    )
    _write_learning_curves_npz(
        output / LEARNING_CURVE_NPZ, learning_curves
    )
    learning_curve_aggregates = _aggregate_learning_curves(learning_curves)
    _write_learning_curve_aggregate_csv(
        output / LEARNING_CURVE_CSV, learning_curve_aggregates
    )
    cumulative_reference_regret_plots = (
        _plot_selected_cumulative_reference_regret(
            output, learning_curve_aggregates
        )
    )
    average_reference_regret_plots = _plot_selected_average_reference_regret(
        output, learning_curve_aggregates
    )
    selected_order_rows.extend(_expected_random_rows(selected_order_rows, outcomes))
    selected_aggregates = _aggregate_selected(selected_order_rows)

    public_candidate_rows = [
        _strip_internal_fields(row) for row in candidate_rows
    ]
    public_selected_order_rows = [
        _strip_internal_fields(row) for row in selected_order_rows
    ]
    _write_csv(output / "candidate_results_by_order.csv", public_candidate_rows)
    _atomic_write_json(
        output / "candidate_results_by_order.json", public_candidate_rows
    )
    _write_csv(output / "candidate_results.csv", candidate_aggregates)
    _atomic_write_json(output / "candidate_results.json", candidate_aggregates)
    _write_csv(output / "selected_multipliers.csv", selections)
    _atomic_write_json(output / "selected_multipliers.json", selections)
    _write_csv(
        output / "selected_results_by_order.csv", public_selected_order_rows
    )
    _atomic_write_json(
        output / "selected_results_by_order.json", public_selected_order_rows
    )
    _write_csv(output / "selected_results.csv", selected_aggregates)
    _atomic_write_json(output / "selected_results.json", selected_aggregates)
    _write_csv(
        output / "squarecb_pmside_tree_vs_linear_by_order.csv",
        igw_comparison_by_order,
    )
    _atomic_write_json(
        output / "squarecb_pmside_tree_vs_linear_by_order.json",
        igw_comparison_by_order,
    )
    _write_csv(
        output / "squarecb_pmside_tree_vs_linear.csv", igw_comparison
    )
    _atomic_write_json(
        output / "squarecb_pmside_tree_vs_linear.json", igw_comparison
    )
    _write_csv(
        output / "squarecb_pmside_tree_vs_linear_matched_by_order.csv",
        igw_matched_by_order,
    )
    _atomic_write_json(
        output / "squarecb_pmside_tree_vs_linear_matched_by_order.json",
        igw_matched_by_order,
    )
    _write_csv(
        output / "squarecb_pmside_tree_vs_linear_matched.csv", igw_matched
    )
    _atomic_write_json(
        output / "squarecb_pmside_tree_vs_linear_matched.json", igw_matched
    )

    _plot_selected_routing_accuracy(
        output / "selected_routing_accuracy.png", selected_aggregates
    )
    _plot_selected_cost(output / "selected_cost_vs_l01.png", selected_aggregates)
    _plot_selected_multipliers(
        output / "selected_multiplier_vs_l01.png", selections
    )
    _plot_igw_estimator_comparison(
        output / "squarecb_pmside_tree_vs_linear_cost_difference.png",
        igw_comparison,
    )
    _plot_igw_matched_estimator_comparison(
        output / "squarecb_pmside_tree_vs_linear_matched_cost_difference.png",
        igw_matched,
    )
    summary = {
        "sweep": manifest,
        "plot_presentation": manifest["plot_presentation"],
        "candidate_checkpoint_count": len(candidate_rows),
        "selection_count": len(selections),
        "selected_results": selected_aggregates,
        "squarecb_pmside_tree_vs_linear_separately_tuned": igw_comparison,
        "squarecb_pmside_tree_vs_linear_matched_gamma": igw_matched,
        "squarecb_pmside_comparison_interpretation": {
            "difference": (
                "linear SquareCB.PMSide realized total cost minus tree "
                "SquareCB.PMSide realized total cost; positive values favor "
                "the nonlinear tree"
            ),
            "separately_tuned": (
                "best-vs-best comparison; selected gamma multipliers may differ"
            ),
            "matched_gamma": (
                "same gamma multiplier and configured SquareCB.PMSide protocol; "
                "estimator family is the only configured difference, while "
                "realized action-dependent histories may diverge"
            ),
        },
        "important_interpretation": (
            f"Each point on a multiplier-tuned policy curve is the best of "
            f"{len(args.multipliers)} multipliers on these same "
            f"{args.online_order_repeats} orders. Those curves are exploratory "
            "optimistic selection envelopes, not unbiased estimates of preselected "
            "policies. The matched-gamma comparison retains every multiplier "
            "without selecting a winner."
        ),
        "alpha_relation": (
            "alpha = 1/l01 because l11 = 1"
            if {float(row["l11"]) for row in selected_aggregates} == {1.0}
            else "alpha = 1/(1+l01-l11)"
        ),
        "regret_reference": {
            "method": reference["method"],
            "folds": int(reference["folds"]),
            "seed": int(reference["seed"]),
            "each_row_held_out": True,
            "roc_auc": float(reference["roc_auc"]),
            "log_loss": float(reference["log_loss"]),
            "brier_score": float(reference["brier_score"]),
            "prediction_artifact": REFERENCE_PREDICTIONS_NPZ,
            "threshold_results": reference_rows,
            "interpretation": (
                f"fixed row-aligned {int(reference['folds'])}-fold OOF HGB-15 "
                "probability reference; "
                "shared across policies and online orders, but assembled from "
                "fold models rather than one fitted policy"
            ),
        },
        "learning_curves": {
            "by_order_npz": LEARNING_CURVE_NPZ,
            "aggregate_csv": LEARNING_CURVE_CSV,
            "plot_formats": ["png", "pdf"],
            "png_dpi": PUBLICATION_PNG_DPI,
            "paired_pdf_for_every_png": True,
            "cumulative_reference_regret_plots": [
                path.name for path in cumulative_reference_regret_plots
            ],
            "average_reference_regret_plots": [
                path.name for path in average_reference_regret_plots
            ],
            "trajectory_count": int(learning_curves["policy"].size),
            "aggregate_policy_loss_groups": len(
                learning_curve_aggregates
            ),
            "aggregate_csv_rows": int(
                len(learning_curve_aggregates)
                * learning_curves["round"].size
            ),
            "online_rounds": int(learning_curves["round"].size),
            "policies": list(
                dict.fromkeys(str(value) for value in learning_curves["policy"])
            ),
            "l01_values": sorted(
                {float(value) for value in learning_curves["l01"]}
            ),
            "by_order_curve_dtype": "float32",
            "aggregate_curve_dtype": "float64",
            "uncertainty": {
                "raw_statistics": (
                    "mean, sample SD, and SEM across paired shuffled online "
                    "orders"
                ),
                "plotted_interval": (
                    "pointwise 95% Student-t confidence interval for the mean"
                ),
                "confidence_level": PLOT_CONFIDENCE_LEVEL,
                "degrees_of_freedom": args.online_order_repeats - 1,
                "t_critical": float(
                    learning_curve_aggregates[0]["confidence_t_critical"]
                ),
                "formula": (
                    "mean +/- t.ppf((1 + confidence_level) / 2, df) * "
                    "sample_sd / sqrt(n)"
                ),
                "simultaneous_band": False,
                "post_selection_adjusted": False,
            },
            "cost_increment_definition": REALIZED_COST_INCREMENT_DEFINITION,
            "clairvoyant_cost_increment_definition": (
                CLAIRVOYANT_COST_INCREMENT_DEFINITION
            ),
            "regret_increment_definition": REGRET_INCREMENT_DEFINITION,
            "clairvoyant_excess_increment_definition": (
                CLAIRVOYANT_EXCESS_INCREMENT_DEFINITION
            ),
            "primary_regret_comparator": reference["method"],
            "outcome_aware_diagnostic_retained_but_not_plotted": True,
            "random_increment_definition": RANDOM_COST_INCREMENT_DEFINITION,
            "selection_warning": LEARNING_CURVE_SELECTION_WARNING,
            "normal_result_tables_include_internal_action_payload": False,
            "json_artifact": None,
        },
    }
    if args.include_pgts:
        summary.update(
            {
                "pgts_prior_tuned_checkpoint_count": (
                    len(args.l01_values)
                    * len(args.multipliers)
                    * args.online_order_repeats
                ),
                "fixed_policies": [],
                "pgts_prior_std_tuning": {
                    "base_prior_std": float(args.pgts_prior_std),
                    "multiplier_grid": [
                        float(value) for value in args.multipliers
                    ],
                    "effective_prior_std_values": [
                        float(args.pgts_prior_std * value)
                        for value in args.multipliers
                    ],
                    "selection": (
                        "lowest mean realized total cost pointwise by l01"
                    ),
                },
                "pgts_interpretation": (
                    "prior-standard-deviation-tuned scheduled-update "
                    "approximation: draw initially and at configured boundaries "
                    "with new revealed feedback, then reuse theta within the epoch"
                ),
            }
        )
    _atomic_write_json(output / "summary.json", summary)
    return _bundle(output)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    if args.include_pgts and not args.plot_only:
        _require_pgts_dependency()
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
    reference_class_counts = np.bincount(outcomes, minlength=2)
    if np.any(reference_class_counts < args.reference_folds):
        raise SystemExit(
            "The cross-fitted reference needs at least --reference-folds "
            "eligible examples from each outcome class"
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

    base_gamma = float(
        manifest["base_parameters"]["squarecb_pmside_gamma"]
    )
    pgts_status = (
        f"PG-TS=prior-std-tuned scheduled approximation "
        f"M={args.pgts_gibbs_steps}, base prior_std={args.pgts_prior_std:g}; "
        if args.include_pgts
        else ""
    )
    print(
        f"Loaded {len(rounds):,} eligible rows with {contexts.shape[1]} features. "
        f"SquareCB.PMSide tree={args.tree_estimator}; "
        "SquareCB.PMSide linear=enabled; "
        "ETC HGB=disabled; ETC Linear=disabled; "
        f"{pgts_status}"
        f"regret reference={args.reference_folds}-fold OOF HGB-15; "
        "adaptive schedule="
        f"{_schedule_slug(args.adaptive_update_schedule, args.adaptive_max_round_gap)}; "
        f"{args.online_order_repeats} paired orders; "
        f"base gamma={base_gamma:.12g}.",
        flush=True,
    )
    # Build the offline comparator once, before any online policy replay.  It
    # remains evaluation-only and is never passed into a routing algorithm.
    reference = _cross_fitted_hgb_reference(
        contexts,
        outcomes,
        folds=args.reference_folds,
        seed=args.seed,
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
        )
    bundle = _write_final_outputs(
        output=output,
        args=args,
        fingerprint=fingerprint,
        outcomes=outcomes,
        example_ids=example_ids,
        permutations=permutations,
        manifest=manifest,
        reference=reference,
    )
    print(f"Finished. Results: {bundle}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
