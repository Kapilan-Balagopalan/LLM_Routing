"""Pólya--Gamma Thompson sampling for logistic apple tasting.

This module implements the posterior-sampling core used by Algorithm 1 of
Grant and Leslie, *Apple Tasting Revisited*.  It deliberately does not own an
environment or a vector of hidden outcomes.  Callers pass only the contexts
and binary outcomes revealed on earlier action-1 rounds.

The paper denotes the Gaussian prior covariance by ``B``.  Here the prior is
fixed to ``N(0, tau^2 I)`` with ``tau = prior_std``; consequently the matrix
added to the likelihood precision is ``B^{-1} = tau^{-2} I``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
PolyaGammaSampler = Callable[[FloatArray, np.random.Generator], ArrayLike]


def sigmoid(value: Union[float, ArrayLike]) -> Union[float, FloatArray]:
    """Evaluate the logistic sigmoid without overflowing for large inputs."""

    values = np.asarray(value, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    if values.ndim == 0:
        return float(result)
    return result


def _default_polya_gamma_sampler(
    linear_predictor: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw ``PG(1, linear_predictor)`` values using the optional package."""

    try:
        from polyagamma import random_polyagamma
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "PG-TS requires the optional 'polyagamma' dependency once "
            "revealed feedback is available. Install it with "
            "`python -m pip install -e \".[pgts]\"`."
        ) from exc

    return np.asarray(
        random_polyagamma(
            1.0,
            linear_predictor,
            random_state=rng,
        ),
        dtype=np.float64,
    )


def _precision_cholesky(precision: FloatArray) -> FloatArray:
    """Return a numerically stable lower Cholesky factor of a precision."""

    symmetric = 0.5 * (precision + precision.T)
    dimension = symmetric.shape[0]
    identity = np.eye(dimension, dtype=np.float64)
    diagonal_scale = max(1.0, float(np.max(np.diag(symmetric))))

    # The proper Gaussian prior makes the precision positive definite in exact
    # arithmetic.  The retries only protect against roundoff after a large
    # weighted Gram-matrix calculation; they do not regularise the model.
    for relative_jitter in (0.0, 1e-12, 1e-10, 1e-8):
        try:
            return np.linalg.cholesky(
                symmetric + relative_jitter * diagonal_scale * identity
            )
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError(
        "PG-TS posterior precision is not numerically positive definite."
    )


def _readonly_copy(values: FloatArray) -> FloatArray:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class PGTSDecision:
    """One Thompson-sampling decision and its sampled expected losses."""

    action: int
    disagreement_probability: float
    loss_action_0: float
    loss_action_1: float
    theta: FloatArray


class PolyaGammaThompsonSampler:
    """Warm-started Pólya--Gamma Gibbs sampler for PG-TS.

    Parameters
    ----------
    dimension:
        Number of features in every context.  Include an intercept in the
        supplied contexts explicitly if one is desired.
    gibbs_steps:
        Number ``M`` of complete Pólya--Gamma/Gaussian Gibbs transitions made
        for every posterior draw.  Algorithm 1 uses the final transition as
        the Thompson sample.  The default is 15.
    prior_std:
        ``tau`` in the zero-mean prior covariance ``B = tau^2 I``.
    seed:
        Seed for both Gaussian and default Pólya--Gamma draws.
    pg_sampler:
        Optional test or alternative sampler.  It receives a one-dimensional
        vector of linear predictors and this instance's NumPy generator, and
        must return one positive ``PG(1, psi_i)`` draw per predictor.  When it
        is omitted, :mod:`polyagamma` is imported lazily on the first draw
        with non-empty revealed history.

    Notes
    -----
    ``draw_theta`` has no access to unrevealed outcomes.  Its two history
    arguments should be assembled only from earlier action-1 rounds.  The
    final sample is retained as the initial state of the next round, exactly
    as in Algorithm 1.
    """

    def __init__(
        self,
        dimension: int,
        *,
        gibbs_steps: int = 15,
        prior_std: float = 1.0,
        seed: Optional[int] = 0,
        pg_sampler: Optional[PolyaGammaSampler] = None,
    ) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
            raise TypeError("dimension must be an integer")
        if int(dimension) <= 0:
            raise ValueError("dimension must be positive")
        if isinstance(gibbs_steps, bool) or not isinstance(
            gibbs_steps, (int, np.integer)
        ):
            raise TypeError("gibbs_steps must be an integer")
        if int(gibbs_steps) <= 0:
            raise ValueError("gibbs_steps must be positive")
        if not np.isfinite(prior_std) or float(prior_std) <= 0.0:
            raise ValueError("prior_std must be finite and positive")

        self.dimension = int(dimension)
        self.gibbs_steps = int(gibbs_steps)
        self.prior_std = float(prior_std)
        self._rng = np.random.default_rng(seed)
        self._pg_sampler = pg_sampler or _default_polya_gamma_sampler
        self._prior_precision = 1.0 / (self.prior_std * self.prior_std)

        # Algorithm 1 initialises theta_0^(M) from the prior.  Every later
        # posterior call starts at the final sample retained here.
        self._theta = self._draw_prior()
        self.draw_count = 0
        self.transition_count = 0

    @property
    def prior_mean(self) -> FloatArray:
        """A defensive copy of the zero prior mean."""

        return np.zeros(self.dimension, dtype=np.float64)

    @property
    def prior_covariance(self) -> FloatArray:
        """A defensive copy of ``B = prior_std^2 I`` (a covariance)."""

        return (self.prior_std * self.prior_std) * np.eye(
            self.dimension, dtype=np.float64
        )

    @property
    def theta(self) -> FloatArray:
        """A defensive copy of the current warm-start state."""

        return self._theta.copy()

    def _draw_prior(self) -> FloatArray:
        return np.asarray(
            self._rng.normal(0.0, self.prior_std, size=self.dimension),
            dtype=np.float64,
        )

    def _validated_history(
        self,
        revealed_features: Optional[ArrayLike],
        revealed_outcomes: Optional[ArrayLike],
    ) -> Tuple[FloatArray, FloatArray]:
        if revealed_features is None and revealed_outcomes is None:
            return (
                np.empty((0, self.dimension), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        if revealed_features is None or revealed_outcomes is None:
            raise ValueError(
                "revealed_features and revealed_outcomes must be supplied together"
            )

        features = np.asarray(revealed_features, dtype=np.float64)
        outcomes = np.asarray(revealed_outcomes, dtype=np.float64)
        if features.size == 0 and features.ndim == 1:
            features = features.reshape(0, self.dimension)
        if features.ndim != 2 or features.shape[1] != self.dimension:
            raise ValueError(
                "revealed_features must have shape (n_observations, dimension)"
            )
        if outcomes.ndim != 1 or outcomes.shape[0] != features.shape[0]:
            raise ValueError(
                "revealed_outcomes must be one-dimensional and match the number "
                "of revealed feature rows"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("revealed_features must contain only finite values")
        if not np.all(np.isfinite(outcomes)) or not np.all(
            (outcomes == 0.0) | (outcomes == 1.0)
        ):
            raise ValueError("revealed_outcomes must contain only binary 0/1 values")
        return features, outcomes

    def draw_theta(
        self,
        revealed_features: Optional[ArrayLike] = None,
        revealed_outcomes: Optional[ArrayLike] = None,
    ) -> FloatArray:
        """Draw the round's Thompson parameter from revealed feedback only.

        Exactly ``gibbs_steps`` Gaussian draws are made on every call.  With
        empty history, each transition is an independent prior draw.  With
        history, each transition first draws the Pólya--Gamma latent variables
        and then draws from their conditional Gaussian posterior.
        """

        features, outcomes = self._validated_history(
            revealed_features, revealed_outcomes
        )
        theta = self._theta.copy()

        if features.shape[0] == 0:
            for _ in range(self.gibbs_steps):
                theta = self._draw_prior()
                self.transition_count += 1
        else:
            prior_precision = self._prior_precision * np.eye(
                self.dimension, dtype=np.float64
            )
            kappa = outcomes - 0.5
            posterior_rhs = features.T @ kappa  # prior mean is exactly zero

            for _ in range(self.gibbs_steps):
                linear_predictor = features @ theta
                omega = np.asarray(
                    self._pg_sampler(linear_predictor.copy(), self._rng),
                    dtype=np.float64,
                )
                if omega.shape != (features.shape[0],):
                    raise ValueError(
                        "pg_sampler must return one value per revealed observation"
                    )
                if not np.all(np.isfinite(omega)) or np.any(omega <= 0.0):
                    raise ValueError(
                        "pg_sampler must return finite, strictly positive values"
                    )

                precision = prior_precision + features.T @ (
                    omega[:, np.newaxis] * features
                )
                factor = _precision_cholesky(precision)
                mean = np.linalg.solve(
                    factor.T,
                    np.linalg.solve(factor, posterior_rhs),
                )
                standard_normal = self._rng.standard_normal(self.dimension)
                theta = mean + np.linalg.solve(factor.T, standard_normal)
                self.transition_count += 1

        self._theta = np.asarray(theta, dtype=np.float64)
        self.draw_count += 1
        return self._theta.copy()

    def sample_posterior(
        self,
        revealed_features: Optional[ArrayLike] = None,
        revealed_outcomes: Optional[ArrayLike] = None,
    ) -> FloatArray:
        """Alias for :meth:`draw_theta`."""

        return self.draw_theta(revealed_features, revealed_outcomes)

    def decide(
        self,
        context: ArrayLike,
        *,
        l01: float,
        l11: float = 1.0,
        revealed_features: Optional[ArrayLike] = None,
        revealed_outcomes: Optional[ArrayLike] = None,
    ) -> PGTSDecision:
        """Make one Algorithm-1 action from the revealed history.

        The expected losses are ``l01 * p`` for action 0 and
        ``1 + (l11 - 1) * p`` for action 1.  An exact tie is resolved in
        favour of informative action 1.
        """

        current_context = np.asarray(context, dtype=np.float64)
        if current_context.shape != (self.dimension,):
            raise ValueError("context must have shape (dimension,)")
        if not np.all(np.isfinite(current_context)):
            raise ValueError("context must contain only finite values")
        if not np.isfinite(l01) or not np.isfinite(l11):
            raise ValueError("l01 and l11 must be finite")
        if float(l11) < 0.0 or float(l01) < float(l11):
            raise ValueError("losses must satisfy l01 >= l11 >= 0")

        theta = self.draw_theta(revealed_features, revealed_outcomes)
        probability = float(sigmoid(float(current_context @ theta)))
        loss_action_0 = float(l01) * probability
        loss_action_1 = 1.0 + (float(l11) - 1.0) * probability
        action = int(loss_action_1 <= loss_action_0)
        return PGTSDecision(
            action=action,
            disagreement_probability=probability,
            loss_action_0=loss_action_0,
            loss_action_1=loss_action_1,
            theta=_readonly_copy(theta),
        )
