import numpy as np

from llm_routing_simulation.algorithm import (
    ONLINE_HGB_PROFILE,
    HGBETCPlayer,
    HGBEstimator,
    IGWPlayer,
    LogCBPSideAT,
    LogCBPSideATConfig,
)
from llm_routing_simulation.skyline import (
    HGB_CAPACITY_PROFILES,
    MLP_CAPACITY_PROFILES,
    binary_residual_diagnostics,
    cross_fitted_residual_predictability,
    fit_holdout_prompt_skylines,
    plot_binary_residuals,
    plot_residual_predictability,
)
from llm_routing_simulation.run import SKYLINE_PLOT_MODELS, _parser


def test_skyline_stops_at_hgb_350():
    profiles = {profile["name"]: profile for profile in HGB_CAPACITY_PROFILES}
    assert profiles["HGB-350"]["max_iter"] == 50
    assert profiles["HGB-350"]["max_leaf_nodes"] == 7
    assert len(profiles) == 5


def test_main_skyline_plot_is_limited_to_research_comparison():
    assert SKYLINE_PLOT_MODELS == (
        "Logistic (80/20 holdout)",
        "HGB (80/20 holdout)",
        "Random routing (expected)",
    )


def test_online_etc_and_igw_use_hgb_configuration():
    assert ONLINE_HGB_PROFILE == {
        "name": "HGB",
        "loss": "log_loss",
        "learning_rate": 0.05,
        "max_iter": 50,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "early_stopping": False,
    }
    config = LogCBPSideATConfig()
    etc = HGBETCPlayer(8, config)
    igw = IGWPlayer(8, config, 100, fixed_gamma=16.0)
    assert isinstance(etc.estimator, HGBEstimator)
    assert etc.estimator.estimator_name == "hist_gradient_boosting"
    assert isinstance(igw.estimator, HGBEstimator)
    assert igw.estimator.estimator_name == "hist_gradient_boosting"


def test_hgb_etc_fits_after_exact_taste_budget_and_then_freezes():
    config = LogCBPSideATConfig(min_tastes=4, use_confidence_bound=False)
    player = HGBETCPlayer(2, config, seed=5)
    contexts = [
        np.asarray([0.1, 0.2]),
        np.asarray([-0.2, 0.3]),
        np.asarray([0.4, -0.1]),
        np.asarray([-0.3, -0.2]),
    ]
    for context, outcome in zip(contexts, [0, 1, 0, 1]):
        decision = player.next_action(context)
        assert decision.action == 1
        assert decision.diagnostics.reason == "forced_exploration"
        player.update(1, context, outcome)

    post_taste = np.asarray([0.5, 0.5])
    decision = player.next_action(post_taste)
    assert decision.diagnostics.estimator_fitted is True
    assert decision.diagnostics.training_count == 4
    fitted_model = player.estimator.model
    player.update(
        decision.action,
        post_taste,
        1 if decision.action == 1 else None,
    )
    next_context = np.asarray([0.6, -0.4])
    next_decision = player.next_action(next_context)
    assert next_decision.diagnostics.training_count == 4
    assert player.estimator.model is fitted_model


def test_igw_has_no_implicit_forced_taste_when_configured_zero():
    config = LogCBPSideATConfig(min_tastes=0)
    player = IGWPlayer(
        2,
        config,
        total_samples=20,
        min_tastes=0,
        bootstrap_per_class=0,
        bootstrap_max_tastes=0,
        fixed_gamma=16.0,
        seed=8,
    )
    decision = player.next_action(np.asarray([0.1, -0.2])).diagnostics
    assert decision.reason == "inverse_gap_weighting"
    assert 0.0 < decision.probability_1 < 1.0


def test_cbpside_class_bootstrap_until_balanced_or_capped():
    config = LogCBPSideATConfig(
        min_tastes=0,
        bootstrap_per_class=1,
        bootstrap_max_tastes=3,
    )
    algorithm = LogCBPSideAT(config)
    x = np.asarray([0.2, -0.1])
    empty = algorithm.choose_action([], [], [], x)
    assert empty.action == 1 and empty.reason == "adaptive_bootstrap"
    one_class = algorithm.choose_action([1], [x], [0], x)
    assert one_class.action == 1 and one_class.reason == "adaptive_bootstrap"
    balanced = algorithm.choose_action([1, 1], [x, x], [0, 1], x)
    assert balanced.reason != "adaptive_bootstrap"
    capped = algorithm.choose_action([1, 1, 1], [x, x, x], [0, 0, 0], x)
    assert capped.reason != "adaptive_bootstrap"


def test_online_exploration_defaults():
    args = _parser().parse_args(["--cache", "fixture.zip"])
    assert args.prompt_components == 64
    assert args.etc_tastes == 300
    assert args.cbpside_tastes == 0
    assert args.cbpside_bootstrap_per_class == 0
    assert args.cbpside_bootstrap_max_tastes == 0
    assert args.igw_min_tastes == 0
    assert args.igw_bootstrap_per_class == 0
    assert args.igw_bootstrap_max_tastes == 0
    assert args.igw_mu == 2.0
    assert args.igw_gamma == 16.0
    assert args.skyline_validation_fraction == 0.2


def test_holdout_prompt_skyline_uses_four_to_one_split():
    rng = np.random.default_rng(41)
    contexts = rng.normal(size=(200, 8))
    outcomes = ((contexts[:, 0] * contexts[:, 1]) > 0.0).astype(np.int64)

    rows, summary, predictions = fit_holdout_prompt_skylines(
        contexts,
        outcomes,
        validation_fraction=0.2,
        seed=4,
    )

    assert summary["train_examples"] == 160
    assert summary["validation_examples"] == 40
    assert summary["fit_scope"] == "single stratified 80/20 train-validation holdout"
    assert len(predictions) == 40
    assert {row["model"] for row in rows} == {
        "Logistic (80/20 holdout)",
        "HGB (80/20 holdout)",
    }
    assert {
        row["model"] for row in summary["model_comparison"]
    } == {
        "Logistic (80/20 holdout)",
        "HGB (80/20 holdout)",
    }


def test_compact_mlp_capacity_sweep():
    profiles = {profile["name"]: profile for profile in MLP_CAPACITY_PROFILES}
    assert profiles["MLP-4"]["hidden_units"] == 4
    assert profiles["MLP-8"]["hidden_units"] == 8
    assert all(profile["alpha"] == 1.0 for profile in profiles.values())
    assert all(profile["solver"] == "adam" for profile in profiles.values())
    assert all(profile["early_stopping"] for profile in profiles.values())


def test_binary_residual_diagnostics_and_plot(tmp_path):
    outcomes = np.asarray([0, 0, 1, 1, 0, 1])
    probabilities = {
        "Logistic": np.asarray([0.1, 0.2, 0.6, 0.8, 0.4, 0.9]),
        "HGB": np.asarray([0.2, 0.3, 0.7, 0.9, 0.1, 0.8]),
        "MLP": np.asarray([0.3, 0.4, 0.6, 0.7, 0.2, 0.9]),
    }
    points, bins = binary_residual_diagnostics(
        probabilities, outcomes, bin_count=3
    )
    assert len(points) == len(outcomes) * len(probabilities)
    assert len(bins) == 3 * len(probabilities)
    first = points[0]
    assert np.isclose(first["raw_residual"], -0.1)
    assert first["deviance_residual"] < 0
    assert all(np.isfinite(row["pearson_residual"]) for row in points)
    assert all(row["count"] == 2 for row in bins)

    output = tmp_path / "residuals.png"
    plot_binary_residuals(output, points, bins)
    assert output.exists() and output.stat().st_size > 0


def test_cross_fitted_residual_predictability_and_plot(tmp_path):
    rng = np.random.default_rng(17)
    contexts = rng.normal(size=(90, 6))
    nonlinear_score = contexts[:, 0] * contexts[:, 1]
    probabilities = 1.0 / (1.0 + np.exp(-nonlinear_score))
    outcomes = (rng.random(90) < probabilities).astype(np.int64)

    points, bins, permutations, summary = cross_fitted_residual_predictability(
        contexts,
        outcomes,
        seed=3,
        requested_folds=3,
        permutation_repeats=2,
        bin_count=5,
    )

    assert len(points) == 90
    assert len(bins) == 5
    assert len(permutations) == 2
    assert {row["outer_fold"] for row in points} == {1, 2, 3}
    assert summary["outer_folds"] == 3
    assert summary["zero_baseline_mse"] > 0.0
    assert np.isfinite(summary["residual_learner_mse"])
    assert 0.0 < summary["permutation_p_value_one_sided"] <= 1.0

    output = tmp_path / "residual_predictability.png"
    plot_residual_predictability(output, points, bins, permutations, summary)
    assert output.exists() and output.stat().st_size > 0
