import builtins

import numpy as np
import pytest

from llm_routing_simulation.pgts import (
    PolyaGammaThompsonSampler,
    sigmoid,
)


def _constant_pg(value, calls=None):
    def sample(linear_predictor, rng):
        del rng
        if calls is not None:
            calls.append(np.asarray(linear_predictor).copy())
        return np.full(np.asarray(linear_predictor).shape, value, dtype=float)

    return sample


def test_defaults_use_zero_mean_unit_covariance_and_fifteen_steps():
    sampler = PolyaGammaThompsonSampler(3, seed=9, pg_sampler=_constant_pg(0.25))

    assert sampler.dimension == 3
    assert sampler.gibbs_steps == 15
    np.testing.assert_array_equal(sampler.prior_mean, np.zeros(3))
    np.testing.assert_array_equal(sampler.prior_covariance, np.eye(3))


def test_no_history_draws_from_prior_for_every_transition_reproducibly():
    def should_not_be_called(linear_predictor, rng):
        raise AssertionError("PG sampler is not used with empty history")

    first = PolyaGammaThompsonSampler(
        4, gibbs_steps=3, prior_std=2.0, seed=17, pg_sampler=should_not_be_called
    )
    second = PolyaGammaThompsonSampler(
        4, gibbs_steps=3, prior_std=2.0, seed=17, pg_sampler=should_not_be_called
    )

    first_draw = first.draw_theta()
    second_draw = second.draw_theta([], [])

    np.testing.assert_array_equal(first_draw, second_draw)
    assert first.draw_count == 1
    assert first.transition_count == 3
    assert not np.shares_memory(first_draw, first.theta)


def test_injected_sampler_runs_m_transitions_and_next_round_warm_starts():
    calls = []
    sampler = PolyaGammaThompsonSampler(
        2,
        gibbs_steps=3,
        seed=4,
        pg_sampler=_constant_pg(0.25, calls),
    )
    features = np.array([[1.0, -0.5], [0.25, 2.0]])
    outcomes = np.array([1, 0])
    initial_theta = sampler.theta

    first_final = sampler.draw_theta(features, outcomes)
    np.testing.assert_allclose(calls[0], features @ initial_theta)
    assert len(calls) == 3

    sampler.draw_theta(features, outcomes)
    np.testing.assert_allclose(calls[3], features @ first_final)
    assert len(calls) == 6
    assert sampler.draw_count == 2
    assert sampler.transition_count == 6


def test_same_seed_and_history_produce_identical_posterior_sequence():
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    outcomes = np.array([1, 0, 1])
    samplers = [
        PolyaGammaThompsonSampler(
            2, gibbs_steps=4, seed=123, pg_sampler=_constant_pg(0.2)
        )
        for _ in range(2)
    ]

    for _ in range(3):
        left = samplers[0].draw_theta(features, outcomes)
        right = samplers[1].draw_theta(features, outcomes)
        np.testing.assert_array_equal(left, right)


def test_decide_uses_the_apple_tasting_expected_losses():
    sampler = PolyaGammaThompsonSampler(
        2, gibbs_steps=2, seed=5, pg_sampler=_constant_pg(0.25)
    )

    route = sampler.decide(np.zeros(2), l01=3.0, l11=1.0)
    keep = sampler.decide(np.zeros(2), l01=1.5, l11=1.0)

    assert route.disagreement_probability == pytest.approx(0.5)
    assert route.loss_action_0 == pytest.approx(1.5)
    assert route.loss_action_1 == pytest.approx(1.0)
    assert route.action == 1
    assert keep.action == 0
    assert not route.theta.flags.writeable


def test_default_sampler_import_is_lazy_and_has_a_clear_install_message(monkeypatch):
    real_import = builtins.__import__

    def import_without_polyagamma(name, *args, **kwargs):
        if name == "polyagamma":
            raise ModuleNotFoundError("test-controlled missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_polyagamma)
    sampler = PolyaGammaThompsonSampler(1, gibbs_steps=1, seed=0)

    # The dependency is not needed until revealed feedback enters the Gibbs
    # likelihood; empty-history Thompson sampling remains available.
    sampler.draw_theta()
    with pytest.raises(ImportError, match=r"\.\[pgts\]"):
        sampler.draw_theta([[1.0]], [1])


@pytest.mark.parametrize(
    "features,outcomes,message",
    [
        ([[1.0]], [1], "shape"),
        ([[1.0, 2.0]], [0, 1], "match"),
        ([[1.0, np.nan]], [1], "finite"),
        ([[1.0, 2.0]], [2], "binary"),
    ],
)
def test_revealed_history_is_validated(features, outcomes, message):
    sampler = PolyaGammaThompsonSampler(
        2, gibbs_steps=1, seed=0, pg_sampler=_constant_pg(0.25)
    )

    with pytest.raises(ValueError, match=message):
        sampler.draw_theta(features, outcomes)


def test_injected_sampler_output_is_validated():
    wrong_shape = PolyaGammaThompsonSampler(
        2,
        gibbs_steps=1,
        seed=0,
        pg_sampler=lambda linear_predictor, rng: np.ones(2),
    )
    nonpositive = PolyaGammaThompsonSampler(
        2,
        gibbs_steps=1,
        seed=0,
        pg_sampler=lambda linear_predictor, rng: np.zeros_like(linear_predictor),
    )

    with pytest.raises(ValueError, match="one value"):
        wrong_shape.draw_theta([[1.0, 2.0]], [1])
    with pytest.raises(ValueError, match="strictly positive"):
        nonpositive.draw_theta([[1.0, 2.0]], [1])


@pytest.mark.parametrize("value", [-1000.0, -2.0, 0.0, 2.0, 1000.0])
def test_sigmoid_is_finite_and_symmetric(value):
    probability = sigmoid(value)

    assert np.isfinite(probability)
    assert 0.0 <= probability <= 1.0
    assert probability + sigmoid(-value) == pytest.approx(1.0)
