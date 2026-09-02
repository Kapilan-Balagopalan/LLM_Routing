"""Online players: XGBoost ETC/IGW and logistic CBPSide."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from llm_routing_simulation.player import HistoryBasedPlayer, PlayerDecision


# This matches the supervised XGBoost model that achieved the strongest
# out-of-fold AUC. ETC freezes it after exploration, while IGW refits it with
# inverse-propensity sample weights on revealed outcomes.
ONLINE_XGBOOST_PROFILE = {
    "name": "XGBoost",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_lambda": 5.0,
    "reg_alpha": 0.1,
}


@dataclass(frozen=True)
class LogCBPSideATConfig:
    matrix_regularization: float = 1.0
    delta: float = 0.05
    loss_reject_disagreement: float = 2.0
    loss_route_disagreement: float = 1.0
    c_max: float = 3.0
    beta_scale: float = 0.05
    max_confidence_radius: float = 0.5
    theta_regularization: float = 1.0
    theta_norm_bound: float | None = None
    max_newton_steps: int = 50
    tolerance: float = 1e-8
    min_tastes: int = 100
    bootstrap_per_class: int = 0
    bootstrap_max_tastes: int = 0
    use_confidence_bound: bool = True


@dataclass(frozen=True)
class LogCBPSideATDecision:
    action: int  # 0 = weak model, 1 = strong model
    predicted_disagreement: float
    threshold: float
    confidence_radius: float
    confident: bool
    reason: str
    theoretical_confidence_radius: float
    scaled_confidence_radius: float
    theta: np.ndarray
    V: np.ndarray
    tasted_count: int
    projection_events: int
    unprojected_theta_norm: float
    max_abs_logit: float
    score_residual_norm: float


class LogCBPSideAT:
    """Compute a routing action from complete history and the current context."""

    def __init__(self, config: LogCBPSideATConfig | None = None) -> None:
        self.config = config or LogCBPSideATConfig()
        self._validate_config()

    @property
    def threshold(self) -> float:
        cfg = self.config
        return 1.0 / (
            1.0 + cfg.loss_reject_disagreement - cfg.loss_route_disagreement
        )

    def choose_action(
        self,
        past_actions: Sequence[int],
        past_contexts: Sequence[np.ndarray],
        past_outcomes: Sequence[int | None],
        current_context: np.ndarray,
    ) -> LogCBPSideATDecision:
        """Fit on revealed history and return action 0 (weak) or 1 (strong)."""
        if not (len(past_actions) == len(past_contexts) == len(past_outcomes)):
            raise ValueError(
                "Past actions, contexts, and outcomes must have equal length"
            )
        x = np.asarray(current_context, dtype=np.float64).reshape(-1)
        if x.size == 0:
            raise ValueError("Current context must not be empty")

        tasted_contexts, tasted_outcomes = self._revealed_history(
            past_actions, past_contexts, past_outcomes, x.size
        )
        model_dim = x.size + 1
        theta, fit_diagnostics = self._estimate_theta(
            tasted_contexts, tasted_outcomes, model_dim
        )
        V = self._design_matrix(tasted_contexts, model_dim)
        model_x = self.features(x)
        predicted = float(self.sigmoid(model_x @ theta))
        theoretical = self._confidence_radius(
            model_x, V, len(tasted_outcomes)
        )
        scaled = self.config.beta_scale * theoretical
        radius = min(scaled, self.config.max_confidence_radius)

        agreements = tasted_outcomes.count(0)
        disagreements = tasted_outcomes.count(1)
        bootstrap_complete = (
            self.config.bootstrap_per_class == 0
            or (
                agreements >= self.config.bootstrap_per_class
                and disagreements >= self.config.bootstrap_per_class
            )
            or (
                self.config.bootstrap_max_tastes > 0
                and len(tasted_outcomes) >= self.config.bootstrap_max_tastes
            )
        )
        if len(tasted_outcomes) < self.config.min_tastes:
            action, confident, reason = 1, False, "forced_exploration"
        elif not bootstrap_complete:
            action, confident, reason = 1, False, "adaptive_bootstrap"
        elif not self.config.use_confidence_bound:
            action = int(predicted >= self.threshold)
            confident = True
            reason = "greedy_threshold"
        else:
            confident = abs(predicted - self.threshold) >= radius
            # EDIT HERE: LogCBPSide-AT action-selection rule.
            action = int((not confident) or predicted >= self.threshold)
            reason = (
                "confidence_exploration"
                if not confident
                else "predicted_disagreement" if action else "confident_agreement"
            )

        return LogCBPSideATDecision(
            action=action,
            predicted_disagreement=predicted,
            threshold=self.threshold,
            confidence_radius=radius,
            confident=confident,
            reason=reason,
            theoretical_confidence_radius=theoretical,
            scaled_confidence_radius=scaled,
            theta=theta,
            V=V,
            tasted_count=len(tasted_outcomes),
            projection_events=fit_diagnostics[0],
            unprojected_theta_norm=fit_diagnostics[1],
            max_abs_logit=fit_diagnostics[2],
            score_residual_norm=fit_diagnostics[3],
        )

    def _revealed_history(
        self,
        actions: Sequence[int],
        contexts: Sequence[np.ndarray],
        outcomes: Sequence[int | None],
        context_dim: int,
    ) -> tuple[list[np.ndarray], list[int]]:
        revealed_contexts: list[np.ndarray] = []
        revealed_outcomes: list[int] = []
        for action, context, outcome in zip(actions, contexts, outcomes):
            if action not in (0, 1):
                raise ValueError("Every past action must be 0 or 1")
            if outcome is not None and outcome not in (0, 1):
                raise ValueError("Every observed outcome must be 0, 1, or None")
            if action == 1 and outcome is not None:
                x = np.asarray(context, dtype=np.float64).reshape(-1)
                if x.size != context_dim:
                    raise ValueError("All contexts must have the same dimension")
                revealed_contexts.append(self.features(x))
                revealed_outcomes.append(outcome)
        return revealed_contexts, revealed_outcomes

    def _estimate_theta(
        self,
        contexts: list[np.ndarray],
        outcomes: list[int],
        context_dim: int,
    ) -> tuple[np.ndarray, tuple[int, float, float, float]]:
        theta = np.zeros(context_dim, dtype=np.float64)
        if not contexts:
            return theta, (0, 0.0, 0.0, 0.0)

        cfg = self.config
        X = np.stack(contexts)
        y = np.asarray(outcomes, dtype=np.float64)
        # Penalize contextual slopes but not the intercept.
        regularizer = np.eye(context_dim)
        regularizer[0, 0] = 0.0
        theta_bound = cfg.theta_norm_bound or cfg.c_max
        projection_events = 0
        largest_proposal_norm = 0.0

        for _ in range(cfg.max_newton_steps):
            probabilities = np.asarray(self.sigmoid(X @ theta))
            penalty = regularizer @ theta
            gradient = (
                X.T @ (y - probabilities)
                - cfg.theta_regularization * penalty
            )
            weights = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
            information = (
                X.T @ (weights[:, None] * X)
                + cfg.theta_regularization * regularizer
            )
            proposal = theta + np.linalg.lstsq(information, gradient, rcond=None)[0]
            proposal_norm = float(np.linalg.norm(proposal))
            largest_proposal_norm = max(largest_proposal_norm, proposal_norm)
            slope_norm = float(np.linalg.norm(proposal[1:]))
            if slope_norm > theta_bound:
                proposal[1:] *= theta_bound / slope_norm
                projection_events += 1
            # The intercept represents baseline disagreement and does not
            # consume the contextual slope norm budget.
            proposal[0] = np.clip(proposal[0], -cfg.c_max, cfg.c_max)
            change = proposal - theta
            theta = proposal
            if np.linalg.norm(change) <= cfg.tolerance * (1.0 + np.linalg.norm(theta)):
                break

        probabilities = np.asarray(self.sigmoid(X @ theta))
        score = (
            X.T @ (y - probabilities)
            - cfg.theta_regularization * (regularizer @ theta)
        )
        return theta, (
            projection_events,
            largest_proposal_norm,
            float(np.max(np.abs(X @ theta))),
            float(np.linalg.norm(score)),
        )

    def _design_matrix(
        self, contexts: list[np.ndarray], context_dim: int
    ) -> np.ndarray:
        V = self.config.matrix_regularization * np.eye(context_dim)
        for x in contexts:
            V += np.outer(x, x)
        return V

    def _confidence_radius(
        self, x: np.ndarray, V: np.ndarray, tasted_count: int
    ) -> float:
        cfg = self.config
        mahalanobis = float(np.sqrt(max(0.0, x @ np.linalg.solve(V, x))))
        # The intercept and slope have separate bounds, so their worst-case
        # absolute logit is the sum of those bounds.
        max_logit = cfg.c_max + (cfg.theta_norm_bound or cfg.c_max)
        edge_probability = float(self.sigmoid(max_logit))
        c_sigma = edge_probability * (1.0 - edge_probability)
        n = max(2, tasted_count)
        beta_squared = (
            (3.0 + 2.0 * np.log(1.0 + 2.0 / cfg.matrix_regularization))
            * 2.0
            * x.size
            * np.log(n)
            * np.log(x.size / cfg.delta)
        )
        return float(
            mahalanobis #* np.sqrt(max(0.0, beta_squared)) / (2.0 * c_sigma)
        )

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.matrix_regularization <= 0 or cfg.c_max <= 0:
            raise ValueError("matrix_regularization and c_max must be positive")
        if not 0 < cfg.delta < 1:
            raise ValueError("delta must be between 0 and 1")
        if cfg.loss_reject_disagreement < cfg.loss_route_disagreement:
            raise ValueError("loss_reject_disagreement must be >= loss_route_disagreement")
        if cfg.beta_scale <= 0 or cfg.max_confidence_radius <= 0:
            raise ValueError("confidence scale and cap must be positive")
        if cfg.theta_regularization < 0:
            raise ValueError("theta_regularization must be nonnegative")
        if cfg.min_tastes < 0:
            raise ValueError("min_tastes must be nonnegative")
        if cfg.bootstrap_per_class < 0 or cfg.bootstrap_max_tastes < 0:
            raise ValueError("bootstrap settings must be nonnegative")
        if (cfg.theta_norm_bound or cfg.c_max) <= 0:
            raise ValueError("theta_norm_bound must be positive")

    @staticmethod
    def normalize(context: np.ndarray) -> np.ndarray:
        return context / max(1.0, float(np.linalg.norm(context)))

    @classmethod
    def features(cls, context: np.ndarray) -> np.ndarray:
        """Prepend an intercept to the normalized external LLM context."""
        return np.concatenate(([1.0], cls.normalize(context)))

    @staticmethod
    def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


class LogCBPSideATPlayer(HistoryBasedPlayer):
    """Store online history and call LogCBPSideAT each round."""

    def __init__(self, context_dim: int, config: LogCBPSideATConfig) -> None:
        super().__init__(context_dim)
        self.algorithm = LogCBPSideAT(config)
        self.min_tastes = config.min_tastes
        self.bootstrap_per_class = config.bootstrap_per_class
        self.bootstrap_max_tastes = config.bootstrap_max_tastes
        self.last_decision: LogCBPSideATDecision | None = None

    def next_action(self, context: np.ndarray) -> PlayerDecision:
        if self._pending_action is not None:
            raise RuntimeError("Previous action must be updated before acting again")
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.size != self.context_dim:
            raise ValueError(f"Expected context dimension {self.context_dim}")
        decision = self.algorithm.choose_action(
            self.actions, self.contexts, self.outcomes, x
        )
        self._pending_context = x.copy()
        self._pending_action = decision.action
        self.last_decision = decision
        return PlayerDecision(decision.action, decision)


@dataclass(frozen=True)
class ETCDecision:
    action: int
    predicted_disagreement: float
    threshold: float
    reason: str
    tasted_count: int
    estimator: str
    estimator_fitted: bool
    observed_classes: int
    training_count: int


class RevealedFeedbackEstimator:
    """Fit disagreement from revealed feedback only; never sees future outcomes."""

    def __init__(
        self,
        context_dim: int,
        *,
        max_features: int | None = None,
        seed: int = 0,
    ) -> None:
        self.context_dim = context_dim
        self.model_context_dim = min(context_dim, max_features or context_dim)
        self.seed = seed
        self.model = None
        self.fitted_count = -1
        self.last_probability = 0.5
        self.observed_classes = 0
        self.last_max_sample_weight = 1.0
        self.last_effective_sample_size = 0.0

    @property
    def estimator_name(self) -> str:
        raise NotImplementedError

    def _new_model(self):
        raise NotImplementedError

    def transform(self, context: np.ndarray) -> np.ndarray:
        """Select a fixed feature prefix without changing the external context."""
        row = np.asarray(context, dtype=np.float64).reshape(-1)
        if row.size != self.context_dim:
            raise ValueError(f"Expected context dimension {self.context_dim}")
        return row[: self.model_context_dim]

    def predict(
        self,
        actions: Sequence[int],
        contexts: Sequence[np.ndarray],
        outcomes: Sequence[int | None],
        current_context: np.ndarray,
        *,
        allow_fit: bool = True,
        freeze_after_fit: bool = False,
        sampling_probabilities: Sequence[float] | None = None,
        min_sampling_probability: float = 0.1,
    ) -> tuple[float, int, bool, int]:
        if not (len(actions) == len(contexts) == len(outcomes)):
            raise ValueError("Past actions, contexts, and outcomes must have equal length")
        if sampling_probabilities is not None and len(sampling_probabilities) != len(actions):
            raise ValueError("Sampling probabilities must align with history")
        if not 0 < min_sampling_probability <= 1:
            raise ValueError("Minimum sampling probability must be in (0, 1]")
        current = self.transform(current_context)
        revealed_x: list[np.ndarray] = []
        revealed_y: list[int] = []
        revealed_weights: list[float] = []
        probabilities = (
            sampling_probabilities
            if sampling_probabilities is not None
            else [1.0] * len(actions)
        )
        for action, context, outcome, sampling_probability in zip(
            actions, contexts, outcomes, probabilities
        ):
            if action not in (0, 1) or outcome not in (0, 1, None):
                raise ValueError("Actions must be binary and outcomes binary or None")
            if not 0 < sampling_probability <= 1:
                raise ValueError("Every sampling probability must be in (0, 1]")
            if action == 1 and outcome is not None:
                row = self.transform(context)
                revealed_x.append(row)
                revealed_y.append(outcome)
                revealed_weights.append(
                    1.0 / max(float(sampling_probability), min_sampling_probability)
                )

        tasted_count = len(revealed_y)
        classes = len(set(revealed_y))
        self.observed_classes = classes
        if revealed_weights:
            weights = np.asarray(revealed_weights, dtype=np.float64)
            self.last_max_sample_weight = float(weights.max())
            self.last_effective_sample_size = float(
                weights.sum() ** 2 / np.dot(weights, weights)
            )
        else:
            weights = np.asarray([], dtype=np.float64)
            self.last_max_sample_weight = 1.0
            self.last_effective_sample_size = 0.0
        class_counts = [revealed_y.count(label) for label in set(revealed_y)]
        if not allow_fit or classes < 2 or min(class_counts) < 2:
            # Laplace smoothing avoids unjustified probabilities of exactly 0 or 1.
            probability = (sum(revealed_y) + 1.0) / (tasted_count + 2.0)
            self.model = None
            self.last_probability = float(probability)
            return float(probability), tasted_count, False, classes

        should_fit = self.model is None or (
            not freeze_after_fit and tasted_count != self.fitted_count
        )
        if should_fit:
            self.model = self._new_model()
            self.model.fit(
                np.stack(revealed_x),
                np.asarray(revealed_y),
                sample_weight=weights,
            )
            self.fitted_count = tasted_count
        probability = float(self.model.predict_proba(current[None, :])[0, 1])
        self.last_probability = probability
        return probability, tasted_count, True, classes


class XGBoostEstimator(RevealedFeedbackEstimator):
    """XGBoost disagreement estimator with IGW inverse-propensity weights."""

    @property
    def estimator_name(self) -> str:
        return "xgboost"

    def _new_model(self):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError(
                "ETC and IGW require XGBoost; reinstall the project dependencies"
            ) from exc
        config = {
            key: value
            for key, value in ONLINE_XGBOOST_PROFILE.items()
            if key != "name"
        }
        return XGBClassifier(
            **config,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.seed,
            n_jobs=1,
            verbosity=0,
        )


class XGBoostETCPlayer(HistoryBasedPlayer):
    """Explore-then-commit using a frozen XGBoost disagreement model."""

    def __init__(
        self,
        context_dim: int,
        config: LogCBPSideATConfig,
        *,
        estimator_max_features: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(context_dim)
        self.config = config
        self.min_tastes = config.min_tastes
        self.estimator = XGBoostEstimator(
            context_dim, max_features=estimator_max_features, seed=seed
        )
        self.last_decision: ETCDecision | None = None

    @property
    def threshold(self) -> float:
        return 1.0 / (
            1.0
            + self.config.loss_reject_disagreement
            - self.config.loss_route_disagreement
        )

    def next_action(self, context: np.ndarray) -> PlayerDecision:
        if self._pending_action is not None:
            raise RuntimeError("Previous action must be updated before acting again")
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        tasted_before_action = sum(
            outcome is not None for outcome in self.outcomes
        )
        predicted, tasted_count, fitted, classes = self.estimator.predict(
            self.actions,
            self.contexts,
            self.outcomes,
            x,
            allow_fit=tasted_before_action >= self.min_tastes,
            freeze_after_fit=True,
        )
        if tasted_count < self.min_tastes:
            action, reason = 1, "forced_exploration"
        else:
            action = int(predicted >= self.threshold)
            reason = f"{self.estimator.estimator_name}_threshold"
        decision = ETCDecision(
            action=action,
            predicted_disagreement=predicted,
            threshold=self.threshold,
            reason=reason,
            tasted_count=tasted_count,
            estimator=self.estimator.estimator_name,
            estimator_fitted=fitted,
            observed_classes=classes,
            training_count=max(0, self.estimator.fitted_count),
        )
        self._pending_context = x.copy()
        self._pending_action = action
        self.last_decision = decision
        return PlayerDecision(action, decision)


@dataclass(frozen=True)
class IGWDecision:
    action: int
    predicted_disagreement: float
    threshold: float
    estimated_loss_0: float
    estimated_loss_1: float
    gap: float
    probability_0: float
    probability_1: float
    reason: str
    theta: np.ndarray
    tasted_count: int
    gamma: float
    gamma_mode: str
    configured_gamma: float | None
    inverse_mahalanobis_norm: float | None
    estimator: str
    estimator_fitted: bool
    observed_classes: int
    training_count: int
    gamma_multiplier: float | None
    action1_probability: float
    max_inverse_propensity_weight: float
    effective_sample_size: float
    bootstrap_agreements: int
    bootstrap_disagreements: int
    bootstrap_complete: bool


class IGWPlayer(HistoryBasedPlayer):
    """Two-arm inverse-gap-weighting player with partial disagreement feedback."""

    def __init__(
        self,
        context_dim: int,
        config: LogCBPSideATConfig,
        total_samples: int,
        *,
        min_tastes: int = 0,
        bootstrap_per_class: int = 10,
        bootstrap_max_tastes: int = 50,
        mu: float = 2.0,
        gamma_multiplier: float = 2.0,
        fixed_gamma: float | None = None,
        min_propensity: float = 0.1,
        estimator_max_features: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(context_dim)
        if (
            total_samples < 1
            or min_tastes < 0
            or bootstrap_per_class < 0
            or bootstrap_max_tastes < 0
            or mu < 2.0
            or gamma_multiplier <= 0
            or (fixed_gamma is not None and fixed_gamma <= 0)
            or not 0 < min_propensity <= 1
        ):
            raise ValueError(
                "IGW requires N>0, min_tastes>=0, mu>=2, positive gamma, "
                "nonnegative bootstrap settings, and min propensity in (0,1]"
            )
        self.estimator = XGBoostEstimator(
            context_dim, max_features=estimator_max_features, seed=seed
        )
        self.config = config
        self.total_samples = total_samples
        self.min_tastes = min_tastes
        self.bootstrap_per_class = bootstrap_per_class
        self.bootstrap_max_tastes = bootstrap_max_tastes
        self.mu = mu
        self.gamma_multiplier = gamma_multiplier
        self.fixed_gamma = fixed_gamma
        self.min_propensity = min_propensity
        self.rng = np.random.default_rng(seed)
        self.action1_probabilities: list[float] = []
        self._pending_probability_1: float | None = None
        self.last_decision: IGWDecision | None = None

    def next_action(self, context: np.ndarray) -> PlayerDecision:
        if self._pending_action is not None:
            raise RuntimeError("Previous action must be updated before acting again")
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.size != self.context_dim:
            raise ValueError(f"Expected context dimension {self.context_dim}")
        tasted_before_action = sum(
            outcome is not None for outcome in self.outcomes
        )
        agreements = sum(outcome == 0 for outcome in self.outcomes)
        disagreements = sum(outcome == 1 for outcome in self.outcomes)
        bootstrap_complete = (
            self.bootstrap_per_class == 0
            or (
                agreements >= self.bootstrap_per_class
                and disagreements >= self.bootstrap_per_class
            )
            or (
                self.bootstrap_max_tastes > 0
                and tasted_before_action >= self.bootstrap_max_tastes
            )
        )
        predicted, tasted_count, fitted, classes = self.estimator.predict(
            self.actions,
            self.contexts,
            self.outcomes,
            x,
            allow_fit=(
                tasted_before_action >= self.min_tastes
                and bootstrap_complete
            ),
            sampling_probabilities=self.action1_probabilities,
            min_sampling_probability=self.min_propensity,
        )
        loss_0 = self.config.loss_reject_disagreement * predicted
        loss_1 = 1.0 + (
            self.config.loss_route_disagreement - 1.0
        ) * predicted
        if self.fixed_gamma is not None:
            gamma = self.fixed_gamma
            gamma_mode = "fixed"
            inverse_mahalanobis_norm = None
        else:
            model_x = LogCBPSideAT.features(self.estimator.transform(x))
            V = self.config.matrix_regularization * np.eye(model_x.size)
            for past_context, outcome in zip(
                self.contexts, self.outcomes
            ):
                if outcome is not None:
                    past_x = LogCBPSideAT.features(
                        self.estimator.transform(past_context)
                    )
                    V += np.outer(past_x, past_x)
            inverse_mahalanobis_norm = float(
                np.sqrt(max(0.0, model_x @ np.linalg.solve(V, model_x)))
            )
            gamma = self.gamma_multiplier * np.sqrt(model_x.size) / max(
                inverse_mahalanobis_norm, 1e-12
            )
            gamma_mode = "adaptive"
        gap = abs(loss_0 - loss_1)
        worse_probability = 1.0 / (self.mu + gamma * gap)
        if loss_1 <= loss_0:
            probability_1 = 1.0 - worse_probability
        else:
            probability_1 = worse_probability
        if tasted_count < self.min_tastes:
            action = 1
            probability_1 = 1.0
            reason = "forced_exploration"
        elif not bootstrap_complete:
            action = 1
            probability_1 = 1.0
            reason = "adaptive_bootstrap"
        else:
            action = int(self.rng.random() < probability_1)
            reason = "inverse_gap_weighting"
        decision = IGWDecision(
            action=action,
            predicted_disagreement=predicted,
            threshold=1.0 / (
                1.0
                + self.config.loss_reject_disagreement
                - self.config.loss_route_disagreement
            ),
            estimated_loss_0=float(loss_0),
            estimated_loss_1=float(loss_1),
            gap=float(gap),
            probability_0=float(1.0 - probability_1),
            probability_1=float(probability_1),
            reason=reason,
            theta=np.array([], dtype=np.float64),
            tasted_count=tasted_count,
            gamma=gamma,
            gamma_mode=gamma_mode,
            configured_gamma=self.fixed_gamma,
            inverse_mahalanobis_norm=inverse_mahalanobis_norm,
            estimator=self.estimator.estimator_name,
            estimator_fitted=fitted,
            observed_classes=classes,
            training_count=max(0, self.estimator.fitted_count),
            gamma_multiplier=(
                None if self.fixed_gamma is not None else self.gamma_multiplier
            ),
            action1_probability=float(probability_1),
            max_inverse_propensity_weight=self.estimator.last_max_sample_weight,
            effective_sample_size=self.estimator.last_effective_sample_size,
            bootstrap_agreements=agreements,
            bootstrap_disagreements=disagreements,
            bootstrap_complete=bootstrap_complete,
        )
        self._pending_context = x.copy()
        self._pending_action = action
        self._pending_probability_1 = float(probability_1)
        self.last_decision = decision
        return PlayerDecision(action, decision)

    def update(
        self, action: int, context: np.ndarray, outcome: int | None
    ) -> None:
        if self._pending_probability_1 is None:
            raise RuntimeError("IGW update must follow next_action")
        probability = self._pending_probability_1
        super().update(action, context, outcome)
        self.action1_probabilities.append(probability)
        self._pending_probability_1 = None
