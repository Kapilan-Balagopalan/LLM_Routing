import numpy as np

from llm_routing_simulation.algorithm import LogCBPSideATConfig
from llm_routing_simulation.synthetic import (
    SmallHGBETCPlayer,
    SmallHGBIGWPlayer,
    SYNTHETIC_L01_VALUES,
    _parser,
    fit_supervised_synthetic_skylines,
    generate_nonlinear_data,
    sigmoid,
)


def test_nonlinear_generator_is_reproducible_and_uses_product_logit():
    contexts, outcomes, probabilities = generate_nonlinear_data(100, seed=7)
    repeated = generate_nonlinear_data(100, seed=7)
    assert contexts.shape == (100, 2)
    assert outcomes.shape == probabilities.shape == (100,)
    assert np.array_equal(contexts, repeated[0])
    assert np.array_equal(outcomes, repeated[1])
    assert np.allclose(
        probabilities, sigmoid(contexts[:, 0] * contexts[:, 1])
    )


def test_small_hgb_resolves_relationship_that_linear_logistic_misses():
    train_x, train_y, _ = generate_nonlinear_data(2000, seed=0)
    test_x, test_y, true_probabilities = generate_nonlinear_data(1000, seed=1)
    _, comparison = fit_supervised_synthetic_skylines(
        train_x,
        train_y,
        test_x,
        test_y,
        true_probabilities,
        seed=0,
    )
    aucs = {row["model"]: row["test_auc"] for row in comparison}
    assert aucs["Small HGB (held-out)"] > aucs["Linear logistic (held-out)"] + 0.2
    assert aucs["Bayes oracle sigma(x1*x2)"] >= aucs["Small HGB (held-out)"]


def test_synthetic_online_players_use_small_hgb_and_expected_defaults():
    args = _parser().parse_args([])
    assert args.train_samples == 3000
    assert args.online_samples == 2000
    assert args.etc_tastes == 500
    assert args.cbpside_tastes == 0
    assert args.igw_min_tastes == 0
    assert args.bootstrap_per_class == 0
    assert args.bootstrap_max_tastes == 0
    assert args.l01_values == SYNTHETIC_L01_VALUES
    thresholds = np.asarray([1.0 / value for value in args.l01_values])
    assert len(thresholds) == 10
    assert np.allclose(thresholds, np.linspace(0.95, 0.05, 10), atol=1e-6)

    config = LogCBPSideATConfig(min_tastes=args.etc_tastes)
    etc = SmallHGBETCPlayer(2, config, seed=0)
    igw = SmallHGBIGWPlayer(
        2,
        config,
        args.online_samples,
        min_tastes=args.igw_min_tastes,
        bootstrap_per_class=args.bootstrap_per_class,
        bootstrap_max_tastes=args.bootstrap_max_tastes,
        mu=args.igw_mu,
        fixed_gamma=args.igw_gamma,
        min_propensity=args.igw_min_propensity,
        seed=0,
    )
    assert etc.estimator.estimator_name == "small_hgb"
    assert igw.estimator.estimator_name == "small_hgb"
