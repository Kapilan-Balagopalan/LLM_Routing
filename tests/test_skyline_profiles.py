import numpy as np
import pytest

import llm_routing_simulation.run as run_module
from llm_routing_simulation.algorithm import (
    DEFAULT_HGB_MAX_LEAF_NODES,
    ONLINE_HGB_PROFILE,
    HGBETCPlayer,
    HGBEstimator,
    IGWPlayer,
    LogCBPSideAT,
    LogCBPSideATConfig,
    LogCBPSideATPlayer,
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
from llm_routing_simulation.run import (
    DEFAULT_ALPHA_VALUES,
    DEFAULT_IGW_GAMMA_VALUES,
    SKYLINE_PLOT_MODELS,
    _parser,
)
from llm_routing_simulation.environment import CascadeRound


def test_skyline_stops_at_hgb_350():
    profiles = {profile["name"]: profile for profile in HGB_CAPACITY_PROFILES}
    assert profiles["HGB-350"]["max_iter"] == 50
    assert profiles["HGB-350"]["max_leaf_nodes"] == 7
    assert len(profiles) == 5


def test_main_skyline_plot_is_limited_to_research_comparison():
    assert SKYLINE_PLOT_MODELS == (
        "Logistic (80/20 holdout)",
        "HGB leaves=15 (80/20 holdout)",
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
    assert DEFAULT_HGB_MAX_LEAF_NODES == (15,)


def test_cbpside_uses_scaled_mahalanobis_leverage_radius():
    config = LogCBPSideATConfig(
        matrix_regularization=1.0,
        beta_scale=0.25,
        max_confidence_radius=0.5,
    )
    algorithm = LogCBPSideAT(config)
    x = np.asarray([1.0, 0.6, 0.8])
    V = np.diag([2.0, 3.0, 4.0])

    leverage = algorithm._confidence_radius(x, V, tasted_count=10)
    expected = np.sqrt(x @ np.linalg.solve(V, x))

    assert np.isclose(leverage, expected)


def test_cbpside_empirical_radius_applies_scale_and_final_cap():
    algorithm = LogCBPSideAT(
        LogCBPSideATConfig(beta_scale=1.0, max_confidence_radius=0.5)
    )
    decision = algorithm.choose_action([], [], [], np.ones(138))

    assert np.isclose(decision.theoretical_confidence_radius, np.sqrt(2.0))
    assert np.isclose(decision.scaled_confidence_radius, np.sqrt(2.0))
    assert decision.confidence_radius == 0.5


def test_cbpside_refits_only_after_a_new_taste_and_matches_fresh_fit():
    config = LogCBPSideATConfig(
        beta_scale=0.25,
        loss_reject_disagreement=1.25,
        min_tastes=2,
        use_confidence_bound=False,
    )
    player = LogCBPSideATPlayer(2, config)
    round_contexts = [
        np.asarray([1.0, 0.0]),
        np.asarray([-1.0, 0.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([0.2, 0.0]),
    ]
    routed_outcomes = [0, 1, 0, 1]
    observed_fit_counts = []

    for current, routed_outcome in zip(round_contexts, routed_outcomes):
        reference = LogCBPSideAT(config).choose_action(
            player.actions,
            player.contexts,
            player.outcomes,
            current,
        )
        cached = player.next_action(current).diagnostics
        observed_fit_counts.append(player.theta_fit_count)

        assert cached.action == reference.action
        assert np.array_equal(cached.theta, reference.theta)
        assert np.array_equal(cached.V, reference.V)
        assert cached.predicted_disagreement == reference.predicted_disagreement
        assert cached.confidence_radius == reference.confidence_radius

        revealed = routed_outcome if cached.action == 1 else None
        player.update(cached.action, current, revealed)

    assert observed_fit_counts == [0, 1, 2, 2]
    assert player.actions[:2] == [1, 1]
    assert player.actions[2:] == [0, 0]


def test_cbpside_cached_diagnostics_are_read_only_and_failed_update_is_atomic():
    config = LogCBPSideATConfig(min_tastes=1)
    player = LogCBPSideATPlayer(2, config)
    context = np.asarray([0.2, -0.1])
    decision = player.next_action(context).diagnostics

    with pytest.raises(ValueError):
        decision.theta[0] = 100.0
    with pytest.raises(ValueError):
        decision.V[0, 0] = 100.0

    with pytest.raises(RuntimeError, match="context differs"):
        player.update(decision.action, np.asarray([0.3, -0.1]), 0)
    assert player.actions == []
    assert player._tasted_outcomes == []
    assert player._fit_dirty is False

    player.update(decision.action, context, 0)
    assert player._tasted_outcomes == [0]


def test_cbpside_cached_player_rejects_empty_context():
    player = LogCBPSideATPlayer(0, LogCBPSideATConfig())
    with pytest.raises(ValueError, match="must not be empty"):
        player.next_action(np.asarray([]))


def test_hgb_cache_refits_only_for_new_tastes_and_preserves_freeze():
    class FakeModel:
        def fit(self, features, labels, sample_weight):
            fit_sizes.append(len(labels))
            self.probability = float(np.average(labels, weights=sample_weight))

        def predict_proba(self, features):
            return np.asarray(
                [[1.0 - self.probability, self.probability]] * len(features)
            )

    fit_sizes: list[int] = []
    estimator = HGBEstimator(2, seed=3)
    estimator._new_model = FakeModel
    actions = [1, 1, 1, 1]
    contexts = [
        np.asarray([0.1, 0.0]),
        np.asarray([0.2, 0.1]),
        np.asarray([-0.1, 0.2]),
        np.asarray([-0.2, -0.1]),
    ]
    outcomes = [0, 0, 1, 1]
    propensities = [1.0, 1.0, 1.0, 1.0]
    current = np.asarray([0.3, -0.2])

    estimator.predict(
        actions,
        contexts,
        outcomes,
        current,
        sampling_probabilities=propensities,
    )
    fitted_model = estimator.model
    assert estimator.fit_count == 1
    assert fit_sizes == [4]

    actions.append(0)
    contexts.append(np.asarray([0.7, 0.4]))
    outcomes.append(None)
    propensities.append(0.4)
    estimator.predict(
        actions,
        contexts,
        outcomes,
        current,
        sampling_probabilities=propensities,
    )
    assert estimator.fit_count == 1
    assert estimator.model is fitted_model
    assert estimator.history_rows_processed == 5

    actions.append(1)
    contexts.append(np.asarray([-0.4, 0.5]))
    outcomes.append(1)
    propensities.append(0.2)
    estimator.predict(
        actions,
        contexts,
        outcomes,
        current,
        sampling_probabilities=propensities,
    )
    assert estimator.fit_count == 2
    assert fit_sizes == [4, 5]
    assert estimator.last_max_sample_weight == 5.0

    actions.extend([1, 2])
    contexts.extend(
        [np.asarray([0.6, -0.3]), np.asarray([0.8, 0.2])]
    )
    outcomes.extend([0, None])
    propensities.extend([0.5, 0.5])
    with pytest.raises(ValueError, match="Actions must be binary"):
        estimator.predict(
            actions,
            contexts,
            outcomes,
            current,
            sampling_probabilities=propensities,
        )

    actions[-1] = 0
    _, tasted_count, _, _ = estimator.predict(
        actions,
        contexts,
        outcomes,
        current,
        sampling_probabilities=propensities,
    )
    assert tasted_count == 6
    assert estimator.fit_count == 3
    assert fit_sizes == [4, 5, 6]
    assert estimator.history_rows_processed == len(actions)

    frozen = HGBEstimator(2, seed=4)
    frozen._new_model = FakeModel
    frozen_actions = actions[:4]
    frozen_contexts = contexts[:4]
    frozen_outcomes = outcomes[:4]
    frozen.predict(
        frozen_actions,
        frozen_contexts,
        frozen_outcomes,
        current,
        freeze_after_fit=True,
    )
    frozen_model = frozen.model
    frozen.predict(
        list(frozen_actions),
        [context.copy() for context in frozen_contexts],
        list(frozen_outcomes),
        current.copy(),
        freeze_after_fit=True,
    )
    assert frozen.fit_count == 1
    assert frozen.model is frozen_model

    frozen_actions.append(1)
    frozen_contexts.append(contexts[-1])
    frozen_outcomes.append(1)
    frozen.predict(
        frozen_actions,
        frozen_contexts,
        frozen_outcomes,
        current,
        freeze_after_fit=True,
    )
    assert frozen.fit_count == 1
    assert frozen.model is frozen_model


def test_online_hgb_capacity_is_configurable_without_changing_other_settings():
    estimator = HGBEstimator(8, max_leaf_nodes=7, seed=12)
    model = estimator._new_model()
    assert estimator.max_leaf_nodes == 7
    assert model.max_leaf_nodes == 7
    assert model.max_iter == ONLINE_HGB_PROFILE["max_iter"]
    assert model.learning_rate == ONLINE_HGB_PROFILE["learning_rate"]
    assert model.min_samples_leaf == ONLINE_HGB_PROFILE["min_samples_leaf"]
    assert model.l2_regularization == ONLINE_HGB_PROFILE["l2_regularization"]


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
    assert args.context_profile == "prompt-only"
    assert args.prompt_components == 20
    assert args.outcome_source == "cached"
    assert args.etc_tastes == 300
    assert args.cbpside_tastes == 0
    assert args.cbpside_bootstrap_per_class == 0
    assert args.cbpside_bootstrap_max_tastes == 0
    assert args.cbpside_matrix_regularization == 1.0
    assert args.cbpside_beta_scale == 0.25
    assert args.cbpside_max_confidence_radius == 0.5
    assert args.igw_min_tastes == 0
    assert args.igw_bootstrap_per_class == 0
    assert args.igw_bootstrap_max_tastes == 0
    assert args.igw_mu == 2.0
    assert args.igw_gamma_values == list(DEFAULT_IGW_GAMMA_VALUES)
    assert DEFAULT_IGW_GAMMA_VALUES == (64.0,)
    assert args.hgb_max_leaf_nodes == [15]
    assert len(args.l01_values) == 10
    assert np.allclose(
        [1.0 / value for value in args.l01_values],
        DEFAULT_ALPHA_VALUES,
    )
    assert np.allclose(DEFAULT_ALPHA_VALUES, np.linspace(0.55, 0.30, 10))
    assert args.skyline_validation_fraction == 0.2


def test_uncertainty_prompt_context_profile_is_explicitly_selectable():
    args = _parser().parse_args(
        [
            "--cache",
            "fixture.zip",
            "--context-profile",
            "uncertainty-prompt",
            "--prompt-components",
            "32",
        ]
    )
    assert args.context_profile == "uncertainty-prompt"
    assert args.prompt_components == 32


def test_all_feature_context_profile_is_explicitly_selectable():
    args = _parser().parse_args(
        ["--cache", "fixture.zip", "--context-profile", "all-features"]
    )
    assert args.context_profile == "all-features"


def test_online_sweep_runs_every_hgb_capacity_with_fixed_gamma(monkeypatch):
    args = _parser().parse_args(["--cache", "fixture.zip"])
    args.l01_values = [2.0]
    round_ = CascadeRound(
        example_id="arc-1",
        prompt="prompt",
        context=np.asarray([0.1, -0.2]),
        weak_answer="A",
        strong_answer="B",
        gold_answer="B",
    )

    def fake_run(method, player, config, rounds, progress_label, metric):
        return {"method": method, "routing_rate": 0.5}, []

    monkeypatch.setattr(run_module, "_run_one_player", fake_run)
    rows, trajectories = run_module.run_online([round_], args)

    assert trajectories == []
    assert [row["method"] for row in rows] == [
        "CBPSide",
        "ETC HGB leaves=15",
        "IGW gamma=64 HGB leaves=15",
        "Random (matched ETC HGB leaves=15)",
    ]


def test_online_hgb_seed_is_constant_across_loss_points(monkeypatch):
    args = _parser().parse_args(["--cache", "fixture.zip", "--seed", "19"])
    args.l01_values = [2.0, 3.0]
    args.hgb_max_leaf_nodes = [7]
    round_ = CascadeRound(
        example_id="arc-1",
        prompt="prompt",
        context=np.asarray([0.1, -0.2]),
        weak_answer="A",
        strong_answer="B",
        gold_answer="B",
    )
    observed_seeds = []

    def fake_run(method, player, config, rounds, progress_label, metric):
        if hasattr(player, "estimator"):
            observed_seeds.append(player.estimator.seed)
        return {"method": method, "routing_rate": 0.5}, []

    monkeypatch.setattr(run_module, "_run_one_player", fake_run)
    run_module.run_online([round_], args)

    assert observed_seeds == [19, 19, 19, 19]


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
    assert all("routing_outcome" in row for row in predictions)
    assert summary["outcome_source"] == "cached_weak_strong_disagreement"
    assert summary["plot_models"] == [
        "Logistic (80/20 holdout)",
        "HGB leaves=15 (80/20 holdout)",
    ]
    assert {row["model"] for row in rows} == {
        "Logistic (80/20 holdout)",
        "HGB leaves=15 (80/20 holdout)",
    }
    assert {
        row["model"] for row in summary["model_comparison"]
    } == {
        "Logistic (80/20 holdout)",
        "HGB leaves=15 (80/20 holdout)",
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
