"""Probability-estimator backends used by online tuning experiments.

The batch backend deliberately rebuilds scikit-learn's HGB estimator from all
revealed feedback.  The River backend instead consumes only the newly revealed
rows.  Keeping those semantics explicit prevents a sweep driver from silently
dropping old observations when it switches estimators.

River is an optional dependency and is imported only when the Hoeffding backend
first receives training data.  The constructor arguments below target River
0.21.2's :class:`river.tree.HoeffdingTreeClassifier` API.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


HGB_BACKEND_KIND = "hgb"
HOEFFDING_BACKEND_KIND = "river_hoeffding"
LOGISTIC_BACKEND_KIND = "logistic"

ONLINE_LOGISTIC_PROFILE = {
    "name": "Logistic",
    "preprocessing": "training-history weighted StandardScaler",
    "uses_all_features": True,
    "scaler_with_mean": True,
    "scaler_with_std": True,
    "scaler_uses_sample_weight": True,
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 5000,
    "fit_intercept": True,
    "class_weight": None,
}


@runtime_checkable
class TreeProbabilityBackend(Protocol):
    """Small common interface needed by the tuning-sweep runner."""

    estimator_name: str
    supports_incremental: bool
    fit_count: int
    training_count: int

    def fit_all(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """Reset or refit the estimator using a complete revealed dataset."""

    def update_many(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """Apply a training batch according to the backend's stated semantics."""

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return one class-1 probability for every input row."""


def _training_batch(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(features, dtype=np.float64)
    raw_y = np.asarray(labels)
    if X.ndim != 2:
        raise ValueError("Training features must be a two-dimensional array")
    if raw_y.ndim != 1:
        raise ValueError("Training labels must be a one-dimensional array")
    if X.shape[0] == 0:
        raise ValueError("A training batch must contain at least one row")
    if X.shape[0] != raw_y.shape[0]:
        raise ValueError("Training features and labels must have equal lengths")
    if not np.all(np.isfinite(X)):
        raise ValueError("Training features must be finite")
    if not np.all(np.isin(raw_y, (0, 1))):
        raise ValueError("Training labels must be binary")
    y = raw_y.astype(np.int8, copy=False)

    if sample_weights is None:
        weights = np.ones(X.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.ndim != 1 or weights.shape[0] != X.shape[0]:
            raise ValueError(
                "Sample weights must be one-dimensional and align with training rows"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("Sample weights must be finite and strictly positive")
    return X, y, weights


def _prediction_batch(
    features: np.ndarray,
    *,
    expected_features: int | None,
) -> np.ndarray:
    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Prediction features must be a two-dimensional array")
    if expected_features is not None and X.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, received {X.shape[1]}"
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("Prediction features must be finite")
    return X


class HGBBatchTreeBackend:
    """Batch HGB backend with the project's fixed 15-leaf online profile.

    ``update_many`` intentionally has the same full-refit semantics as
    ``fit_all``.  A caller using this backend must therefore pass *all*
    accumulated revealed rows, not merely the latest buffer.
    """

    estimator_name = "sklearn_hgb_15_leaf_batch"
    supports_incremental = False

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self.fit_count = 0
        self.training_count = 0
        self._feature_count: int | None = None
        self._model: Any | None = None

    @property
    def model(self) -> Any | None:
        """Expose the fitted estimator for diagnostics, never for online labels."""

        return self._model

    @staticmethod
    def _profile() -> dict[str, Any]:
        # Import lazily to avoid a module cycle if algorithm.py later adopts
        # this backend.  The single project constant remains authoritative.
        from llm_routing_simulation.algorithm import ONLINE_HGB_PROFILE

        if int(ONLINE_HGB_PROFILE["max_leaf_nodes"]) != 15:
            raise RuntimeError(
                "The tuning backend requires ONLINE_HGB_PROFILE to use 15 leaves"
            )
        return ONLINE_HGB_PROFILE

    def _new_model(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingClassifier

        profile = self._profile()
        return HistGradientBoostingClassifier(
            loss=profile["loss"],
            learning_rate=profile["learning_rate"],
            max_iter=profile["max_iter"],
            max_leaf_nodes=profile["max_leaf_nodes"],
            min_samples_leaf=profile["min_samples_leaf"],
            l2_regularization=profile["l2_regularization"],
            early_stopping=profile["early_stopping"],
            random_state=self.seed,
        )

    def fit_all(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        X, y, weights = _training_batch(features, labels, sample_weights)
        if self._feature_count is not None and X.shape[1] != self._feature_count:
            raise ValueError(
                f"Expected {self._feature_count} features, received {X.shape[1]}"
            )
        model = self._new_model()
        model.fit(X, y, sample_weight=weights)
        self._model = model
        self._feature_count = int(X.shape[1])
        self.fit_count += 1
        # A batch rebuild represents the latest complete revealed dataset.
        self.training_count = int(X.shape[0])

    def update_many(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        self.fit_all(features, labels, sample_weights)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("The HGB backend must be fitted before prediction")
        X = _prediction_batch(features, expected_features=self._feature_count)
        probabilities = np.asarray(self._model.predict_proba(X), dtype=np.float64)
        classes = np.asarray(self._model.classes_)
        positive_columns = np.flatnonzero(classes == 1)
        if positive_columns.size == 0:
            return np.zeros(X.shape[0], dtype=np.float64)
        return probabilities[:, int(positive_columns[0])].reshape(-1)


class LogisticBatchProbabilityBackend:
    """Full-refit standardized linear-logistic probability backend.

    Every refit learns both scaling statistics and logistic coefficients from
    the supplied revealed-history rows only.  Inverse-propensity weights are
    applied to the scaler and the logistic objective.  No feature selection or
    prefix truncation is performed.
    """

    estimator_name = "sklearn_standardized_logistic_l2_batch"
    supports_incremental = False

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self.fit_count = 0
        self.training_count = 0
        self._feature_count: int | None = None
        self._scaler: Any | None = None
        self._model: Any | None = None

    @property
    def profile(self) -> dict[str, Any]:
        """Return a copy of the fixed estimator profile for result metadata."""

        return dict(ONLINE_LOGISTIC_PROFILE)

    @property
    def metadata(self) -> dict[str, Any]:
        """Describe the estimator and its online-information boundary."""

        return {
            "kind": LOGISTIC_BACKEND_KIND,
            "estimator_name": self.estimator_name,
            "supports_incremental": self.supports_incremental,
            "preprocessing_fit_scope": "supplied_revealed_history_only",
            "sample_weight_usage": "scaler_and_logistic_objective",
            "profile": self.profile,
        }

    @property
    def scaler(self) -> Any | None:
        return self._scaler

    @property
    def model(self) -> Any | None:
        return self._model

    @staticmethod
    def _new_scaler() -> Any:
        from sklearn.preprocessing import StandardScaler

        return StandardScaler(
            with_mean=ONLINE_LOGISTIC_PROFILE["scaler_with_mean"],
            with_std=ONLINE_LOGISTIC_PROFILE["scaler_with_std"],
        )

    def _new_model(self) -> Any:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            penalty=ONLINE_LOGISTIC_PROFILE["penalty"],
            C=ONLINE_LOGISTIC_PROFILE["C"],
            solver=ONLINE_LOGISTIC_PROFILE["solver"],
            max_iter=ONLINE_LOGISTIC_PROFILE["max_iter"],
            fit_intercept=ONLINE_LOGISTIC_PROFILE["fit_intercept"],
            class_weight=ONLINE_LOGISTIC_PROFILE["class_weight"],
            random_state=self.seed,
        )

    def fit_all(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        X, y, weights = _training_batch(features, labels, sample_weights)
        if self._feature_count is not None and X.shape[1] != self._feature_count:
            raise ValueError(
                f"Expected {self._feature_count} features, received {X.shape[1]}"
            )
        scaler = self._new_scaler()
        scaler.fit(X, sample_weight=weights)
        transformed = scaler.transform(X)
        model = self._new_model()
        model.fit(transformed, y, sample_weight=weights)

        # Update externally visible state only after both fits succeed.
        self._scaler = scaler
        self._model = model
        self._feature_count = int(X.shape[1])
        self.fit_count += 1
        self.training_count = int(X.shape[0])

    def update_many(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        self.fit_all(features, labels, sample_weights)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._model is None or self._scaler is None:
            raise RuntimeError("The logistic backend must be fitted before prediction")
        X = _prediction_batch(features, expected_features=self._feature_count)
        transformed = self._scaler.transform(X)
        probabilities = np.asarray(
            self._model.predict_proba(transformed), dtype=np.float64
        )
        classes = np.asarray(self._model.classes_)
        positive_columns = np.flatnonzero(classes == 1)
        if positive_columns.size == 0:
            return np.zeros(X.shape[0], dtype=np.float64)
        return probabilities[:, int(positive_columns[0])].reshape(-1)


def _import_river_tree() -> Any:
    try:
        from river import tree
    except ImportError as exc:  # pragma: no cover - exact message tested indirectly
        raise ImportError(
            "The River Hoeffding backend requires the optional dependency "
            "river==0.21.2"
        ) from exc
    return tree


class RiverHoeffdingTreeBackend:
    """Incremental River 0.21.2 Hoeffding-tree probability backend."""

    estimator_name = "river_hoeffding_tree_depth4_incremental"
    supports_incremental = True

    def __init__(
        self,
        *,
        seed: int = 0,
        max_depth: int = 4,
        grace_period: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
    ) -> None:
        if max_depth < 1:
            raise ValueError("Hoeffding-tree max_depth must be positive")
        if grace_period < 1:
            raise ValueError("Hoeffding-tree grace_period must be positive")
        if not 0.0 < delta < 1.0:
            raise ValueError("Hoeffding-tree delta must lie strictly between 0 and 1")
        if not 0.0 <= tau <= 1.0:
            raise ValueError("Hoeffding-tree tau must lie between 0 and 1")
        # HoeffdingTreeClassifier has no stochastic seed in River 0.21.2.  Keep
        # the value as run metadata so both backends share one factory API.
        self.seed = int(seed)
        self.max_depth = int(max_depth)
        self.grace_period = int(grace_period)
        self.delta = float(delta)
        self.tau = float(tau)
        self.fit_count = 0
        self.training_count = 0
        self._feature_count: int | None = None
        self._model: Any | None = None

    @property
    def model(self) -> Any | None:
        """Expose the fitted estimator for diagnostics, never for online labels."""

        return self._model

    def _new_model(self) -> Any:
        tree = _import_river_tree()
        try:
            return tree.HoeffdingTreeClassifier(
                max_depth=self.max_depth,
                grace_period=self.grace_period,
                delta=self.delta,
                tau=self.tau,
                leaf_prediction="mc",
                binary_split=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Installed River is incompatible with the supported 0.21.2 "
                "HoeffdingTreeClassifier API"
            ) from exc

    @staticmethod
    def _row_mapping(row: np.ndarray) -> dict[int, float]:
        return {index: float(value) for index, value in enumerate(row)}

    def _learn_validated_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        if self._model is None:
            self._model = self._new_model()
        for row, label, weight in zip(X, y, weights):
            self._model.learn_one(
                self._row_mapping(row),
                int(label),
                w=float(weight),
            )

    def fit_all(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        X, y, weights = _training_batch(features, labels, sample_weights)
        self._model = None
        self._feature_count = int(X.shape[1])
        self.training_count = 0
        self._learn_validated_batch(X, y, weights)
        self.fit_count += 1
        self.training_count = int(X.shape[0])

    def update_many(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        X, y, weights = _training_batch(features, labels, sample_weights)
        if self._feature_count is not None and X.shape[1] != self._feature_count:
            raise ValueError(
                f"Expected {self._feature_count} features, received {X.shape[1]}"
            )
        if self._feature_count is None:
            self._feature_count = int(X.shape[1])
        self._learn_validated_batch(X, y, weights)
        self.fit_count += 1
        self.training_count += int(X.shape[0])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(
                "The River Hoeffding backend must receive training data before prediction"
            )
        X = _prediction_batch(features, expected_features=self._feature_count)
        return np.asarray(
            [
                float(self._model.predict_proba_one(self._row_mapping(row)).get(1, 0.0))
                for row in X
            ],
            dtype=np.float64,
        )


def make_tree_backend(
    kind: str,
    *,
    seed: int = 0,
    **backend_options: Any,
) -> TreeProbabilityBackend:
    """Build an explicit HGB, logistic, or incremental-Hoeffding backend."""

    normalized = kind.strip().lower().replace("-", "_")
    if normalized in {HGB_BACKEND_KIND, "hgb_batch"}:
        return HGBBatchTreeBackend(seed=seed, **backend_options)
    if normalized in {
        LOGISTIC_BACKEND_KIND,
        "linear_logistic",
        "logistic_batch",
    }:
        return LogisticBatchProbabilityBackend(seed=seed, **backend_options)
    if normalized in {
        HOEFFDING_BACKEND_KIND,
        "hoeffding",
        "hoeffding_tree",
    }:
        return RiverHoeffdingTreeBackend(seed=seed, **backend_options)
    raise ValueError(
        f"Unknown tree backend {kind!r}; choose 'hgb', 'logistic', "
        "or 'river_hoeffding'"
    )
