"""Synthetic multifeature prompt outcomes for routing positive controls."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


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


def _nonlinear_target(standardized_contexts: np.ndarray) -> np.ndarray:
    """Create a deterministic nonlinear target without answer information."""
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


def generate_synthetic_prompt_outcomes(
    contexts: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate one frozen synthetic outcome vector from prompt contexts.

    The teacher is part of the simulated environment. It may use the complete
    outcome-free prompt-context matrix, but its probabilities and labels are
    never exposed to an online player. Players receive only the current context
    and outcomes revealed by strong routing.
    """
    X = np.asarray(contexts, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("Synthetic prompt contexts must be a nonempty matrix")
    if not np.all(np.isfinite(X)):
        raise ValueError("Synthetic prompt contexts must be finite")

    feature_count = min(
        SYNTHETIC_FOREST_CONFIGURATION["prompt_feature_count"], X.shape[1]
    )
    features = X[:, :feature_count]
    means = np.mean(features, axis=0)
    scales = np.std(features, axis=0)
    scales[scales == 0.0] = 1.0
    latent_target = _nonlinear_target((features - means) / scales)
    minimum_leaf = min(
        SYNTHETIC_FOREST_CONFIGURATION["min_samples_leaf"],
        max(2, features.shape[0] // 10),
    )
    teacher_seed = seed + 9_000
    label_seed = seed + 10_000
    teacher = RandomForestRegressor(
        n_estimators=SYNTHETIC_FOREST_CONFIGURATION["n_estimators"],
        max_depth=SYNTHETIC_FOREST_CONFIGURATION["max_depth"],
        min_samples_leaf=minimum_leaf,
        max_features=SYNTHETIC_FOREST_CONFIGURATION["max_features"],
        criterion="squared_error",
        random_state=teacher_seed,
        n_jobs=1,
    )
    teacher.fit(features, latent_target)
    scores = teacher.predict(features)
    score_center = float(np.mean(scores))
    score_scale = max(float(np.std(scores)), 1e-8)
    logits = (
        SYNTHETIC_FOREST_CONFIGURATION["probability_logit_scale"]
        * (scores - score_center)
        / score_scale
    )
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    lower, upper = SYNTHETIC_FOREST_CONFIGURATION["probability_clip"]
    probabilities = np.clip(probabilities, lower, upper)
    outcomes = np.random.default_rng(label_seed).binomial(1, probabilities).astype(
        np.int8
    )
    if np.unique(outcomes).size != 2:
        raise ValueError("Synthetic forest generated only one outcome class")

    summary = {
        "purpose": "positive-control outcome generator for routing sanity checks",
        "outcome_source": "synthetic_probabilistic_prompt_forest",
        "real_outcomes_used": False,
        "arc_gold_answers_used": False,
        "teacher_fit_scope": (
            "all selected eligible prompt contexts; transductive and outcome-free"
        ),
        "online_access": (
            "teacher probabilities and unrevealed labels remain environment-only"
        ),
        "input_features": [
            f"prompt_pc_{index + 1}" for index in range(feature_count)
        ],
        "input_feature_count": feature_count,
        "latent_target": (
            "alternating pairwise tanh interactions plus univariate sine terms"
        ),
        "configuration": {
            **SYNTHETIC_FOREST_CONFIGURATION,
            "effective_min_samples_leaf": minimum_leaf,
        },
        "teacher_seed": teacher_seed,
        "label_sampling_seed": label_seed,
        "feature_importances": teacher.feature_importances_.tolist(),
        "score_center": score_center,
        "score_scale": score_scale,
        "probability_range": [
            float(np.min(probabilities)),
            float(np.max(probabilities)),
        ],
        "positive_rate": float(np.mean(outcomes)),
        "examples": int(outcomes.size),
    }
    return probabilities, outcomes, summary
