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
    DEFAULT_L01_VALUES,
    SKYLINE_PLOT_MODELS,
    _aggregate_online_order_rows,
    _parser,
    _plot_online_cost_vs_l01,
    _plot_online_routing_accuracy,
    _realized_cost_metrics,
    _resolve_online_parameters,
    _shuffled_online_rounds,
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


def test_cbpside_refits_after_each_five_additional_tastes():
    config = LogCBPSideATConfig(min_tastes=100, use_confidence_bound=False)
    player = LogCBPSideATPlayer(2, config, refit_every_tastes=5)
    fit_sizes = []
    original_estimate = player.algorithm._estimate_theta

    def recording_estimate(contexts, outcomes, model_dim):
        fit_sizes.append(len(outcomes))
        return original_estimate(contexts, outcomes, model_dim)

    player.algorithm._estimate_theta = recording_estimate
    previous_V = player._V.copy()
    for index in range(12):
        context = np.asarray([index / 10.0, (-1.0) ** index])
        decision = player.next_action(context)
        assert decision.action == 1
        player.update(1, context, index % 2)
        assert not np.array_equal(player._V, previous_V)
        previous_V = player._V.copy()

    assert fit_sizes == [1, 6, 11]
    assert player.theta_fit_count == 3
    assert player.last_model_training_count == 11
    assert player._fit_dirty is True


def test_online_refit_interval_must_be_positive():
    config = LogCBPSideATConfig()
    with pytest.raises(ValueError, match="refit_every_tastes"):
        LogCBPSideATPlayer(2, config, refit_every_tastes=0)
    with pytest.raises(ValueError, match="refit_every_tastes"):
        HGBEstimator(2, refit_every_tastes=0)


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


def test_hgb_refits_after_each_five_additional_tastes():
    class FakeModel:
        def fit(self, features, labels, sample_weight):
            fit_sizes.append(len(labels))
            self.probability = float(np.average(labels, weights=sample_weight))

        def predict_proba(self, features):
            return np.asarray(
                [[1.0 - self.probability, self.probability]] * len(features)
            )

    fit_sizes = []
    estimator = HGBEstimator(2, refit_every_tastes=5, seed=3)
    estimator._new_model = FakeModel
    actions = [1, 1, 1, 1]
    contexts = [
        np.asarray([0.1, 0.0]),
        np.asarray([0.2, 0.1]),
        np.asarray([-0.1, 0.2]),
        np.asarray([-0.2, -0.1]),
    ]
    outcomes = [0, 0, 1, 1]
    propensities = [1.0] * 4
    current = np.asarray([0.3, -0.2])

    estimator.predict(
        actions,
        contexts,
        outcomes,
        current,
        sampling_probabilities=propensities,
    )
    assert fit_sizes == [4]

    for index, outcome in enumerate([0, 1, 0, 1], start=5):
        actions.append(1)
        contexts.append(np.asarray([index / 10.0, -index / 10.0]))
        outcomes.append(outcome)
        propensities.append(1.0)
        estimator.predict(
            actions,
            contexts,
            outcomes,
            current,
            sampling_probabilities=propensities,
        )
    assert fit_sizes == [4]
    assert estimator.fitted_count == 4

    actions.append(1)
    contexts.append(np.asarray([0.9, -0.9]))
    outcomes.append(0)
    propensities.append(1.0)
    estimator.predict(
        actions,
        contexts,
        outcomes,
        current,
        sampling_probabilities=propensities,
    )
    assert fit_sizes == [4, 9]
    assert estimator.fitted_count == 9


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
    assert args.context_profile == "non-prompt"
    assert args.prompt_components == 64
    assert args.outcome_source == "cached"
    assert args.etc_tastes is None
    assert args.cbpside_tastes == 0
    assert args.cbpside_bootstrap_per_class == 0
    assert args.cbpside_bootstrap_max_tastes == 0
    assert args.cbpside_matrix_regularization == 1.0
    assert args.cbpside_beta_scale == 0.5
    assert LogCBPSideATConfig().beta_scale == 0.5
    assert args.cbpside_max_confidence_radius == 0.5
    assert LogCBPSideATConfig().max_confidence_radius == 0.5
    assert args.igw_min_tastes == 0
    assert args.igw_bootstrap_per_class == 0
    assert args.igw_bootstrap_max_tastes == 0
    assert args.igw_mu == 2.0
    assert args.igw_gamma_values is None
    assert args.hgb_max_leaf_nodes == [15]
    assert args.online_order_repeats == 1
    assert args.online_trajectory_mode == "none"
    assert args.online_refit_every_tastes == 5
    assert len(args.l01_values) == 9
    assert args.l01_values == list(DEFAULT_L01_VALUES)
    assert DEFAULT_L01_VALUES == (
        1.8,
        2.0,
        2.2,
        2.4,
        2.6,
        2.8,
        3.0,
        3.2,
        3.3,
    )
    assert np.allclose(
        [1.0 / value for value in args.l01_values], DEFAULT_ALPHA_VALUES
    )
    assert args.skyline_validation_fraction == 0.2


def test_online_parameters_resolve_from_actual_horizon():
    args = _parser().parse_args(["--cache", "fixture.zip"])

    resolution = _resolve_online_parameters(args, 12_648)

    assert args.etc_tastes == 543
    assert np.isclose(args.igw_gamma_values, [np.sqrt(12_648)]).all()
    assert resolution == {
        "total_samples": 12_648,
        "etc_tastes": 543,
        "etc_tastes_source": "ceil(n^(2/3))",
        "igw_gamma_values": [np.sqrt(12_648)],
        "igw_gamma_source": "sqrt(n)",
    }


def test_online_parameter_overrides_are_preserved():
    args = _parser().parse_args(
        [
            "--cache",
            "fixture.zip",
            "--etc-tastes",
            "300",
            "--igw-gamma-values",
            "64",
            "128",
        ]
    )

    resolution = _resolve_online_parameters(args, 12_648)

    assert args.etc_tastes == 300
    assert args.igw_gamma_values == [64.0, 128.0]
    assert resolution["etc_tastes_source"] == "command_line"
    assert resolution["igw_gamma_source"] == "command_line"


def test_shuffled_online_rounds_are_reproducible_complete_and_nonmutating():
    rounds = [
        CascadeRound(
            example_id=f"boolq-{index}",
            prompt="prompt",
            context=np.asarray([float(index)]),
            weak_answer="A",
            strong_answer="B",
            gold_answer="B",
        )
        for index in range(12)
    ]
    original_ids = [row.example_id for row in rounds]

    first, first_indices = _shuffled_online_rounds(rounds, 7)
    repeated, repeated_indices = _shuffled_online_rounds(rounds, 7)
    different, different_indices = _shuffled_online_rounds(rounds, 8)

    assert np.array_equal(first_indices, repeated_indices)
    assert not np.array_equal(first_indices, different_indices)
    assert [row.example_id for row in first] == [
        row.example_id for row in repeated
    ]
    assert [row.example_id for row in first] != [
        row.example_id for row in different
    ]
    assert sorted(row.example_id for row in first) == sorted(original_ids)
    assert [row.example_id for row in rounds] == original_ids


def test_online_order_aggregation_uses_sample_standard_deviation():
    rows = []
    for order_run, routing_rate, accuracy, total_cost in (
        (1, 0.2, 0.7, 50.0),
        (2, 0.4, 0.9, 70.0),
    ):
        rows.append(
            {
                "method": "CBPSide",
                "l01": 2.0,
                "l11": 1.0,
                "alpha": 0.5,
                "examples": 100,
                "routing_rate": routing_rate,
                "accuracy": accuracy,
                "routed_to_strong": routing_rate * 100,
                "unrouted_disagreements": (1.0 - accuracy) * 100,
                "realized_cost_per_example": total_cost / 100,
                "realized_total_cost": total_cost,
                "model_refits": 10 + order_run,
                "order_run": order_run,
                "order_seed": order_run - 1,
                "order_was_shuffled": True,
                "policy_seed": 0,
            }
        )

    [result] = _aggregate_online_order_rows(rows)

    assert result["online_order_repeats"] == 2
    assert result["order_seeds"] == [0, 1]
    assert result["routing_rate"] == pytest.approx(0.3)
    assert result["accuracy"] == pytest.approx(0.8)
    assert result["realized_total_cost"] == pytest.approx(60.0)
    assert result["realized_total_cost_std"] == pytest.approx(np.sqrt(200.0))
    assert result["realized_total_cost_sem"] == pytest.approx(10.0)


def test_online_order_repeats_use_distinct_full_permutations(monkeypatch):
    args = _parser().parse_args(
        [
            "--cache",
            "fixture.zip",
            "--online-order-repeats",
            "3",
        ]
    )
    rounds = [
        CascadeRound(
            example_id=f"boolq-{index}",
            prompt="prompt",
            context=np.asarray([float(index)]),
            weak_answer="A",
            strong_answer="B",
            gold_answer="B",
        )
        for index in range(8)
    ]
    observed_orders = []
    observed_collection_flags = []

    def fake_online(ordered_rounds, parsed_args, *, collect_trajectories):
        observed_orders.append(tuple(row.example_id for row in ordered_rounds))
        observed_collection_flags.append(collect_trajectories)
        result = {
            "method": "CBPSide",
            "l01": 2.0,
            "l11": 1.0,
            "alpha": 0.5,
            "examples": len(ordered_rounds),
            "routing_rate": 0.5,
            "accuracy": 0.75,
            "routed_to_strong": 4.0,
            "unrouted_disagreements": 2.0,
            "realized_cost_per_example": 1.0,
            "realized_total_cost": 8.0,
            "model_refits": 2,
        }
        return [result], []

    monkeypatch.setattr(run_module, "run_online", fake_online)
    aggregate, raw, trajectories, permutations, metadata = (
        run_module.run_online_order_repeats(rounds, args)
    )

    assert len(observed_orders) == 3
    assert len(set(observed_orders)) == 3
    assert all(sorted(order) == sorted(observed_orders[0]) for order in observed_orders)
    assert observed_collection_flags == [False, False, False]
    assert args.etc_tastes == 4
    assert args.igw_gamma_values == pytest.approx([np.sqrt(8)])
    assert [row["order_seed"] for row in raw] == [0, 1, 2]
    assert aggregate[0]["online_order_repeats"] == 3
    assert trajectories == []
    assert permutations.shape == (3, 8)
    assert [row["order_seed"] for row in metadata] == [0, 1, 2]


def test_online_sweep_applies_horizon_derived_parameters(monkeypatch):
    args = _parser().parse_args(["--cache", "fixture.zip"])
    args.l01_values = [2.0]
    rounds = [
        CascadeRound(
            example_id=f"boolq-{index}",
            prompt="prompt",
            context=np.asarray([0.1, -0.2]),
            weak_answer="A",
            strong_answer="B",
            gold_answer="B",
        )
        for index in range(8)
    ]
    observed = {}

    def fake_run(
        method,
        player,
        config,
        rounds,
        progress_label,
        metric,
        collect_trajectories,
    ):
        if method == "CBPSide":
            observed["cbpside_refit_every"] = player.refit_every_tastes
        if method.startswith("ETC"):
            observed["etc_tastes"] = player.min_tastes
            observed["etc_refit_every"] = player.estimator.refit_every_tastes
        if method.startswith("IGW"):
            observed["igw_gamma"] = player.fixed_gamma
            observed["igw_refit_every"] = player.refit_every_tastes
        return {"method": method, "routing_rate": 0.5}, []

    monkeypatch.setattr(run_module, "_run_one_player", fake_run)
    run_module.run_online(rounds, args)

    assert observed["etc_tastes"] == 4
    assert observed["etc_refit_every"] == 1
    assert np.isclose(observed["igw_gamma"], np.sqrt(8))
    assert observed["cbpside_refit_every"] == 5
    assert observed["igw_refit_every"] == 5


def test_realized_total_cost_matches_the_three_outcome_costs():
    metrics = _realized_cost_metrics(
        l01=3.0,
        l11=1.0,
        routing_rate=2 / 8,
        accuracy=7 / 8,
        examples=8,
    )

    assert metrics["routed_to_strong"] == 2.0
    assert metrics["unrouted_disagreements"] == 1.0
    assert metrics["realized_total_cost"] == 5.0
    assert metrics["realized_cost_per_example"] == 5 / 8


def test_online_comparison_plots_are_written(tmp_path):
    rows = []
    for method_index, method in enumerate(("CBPSide", "ETC", "IGW", "Random")):
        for alpha in (0.4, 0.5):
            l01 = 1.0 / alpha
            routing_rate = 0.2 + 0.05 * method_index
            accuracy = 0.7 + 0.04 * method_index
            row = {
                "method": method,
                "l01": l01,
                "l11": 1.0,
                "alpha": alpha,
                "routing_rate": routing_rate,
                "routing_rate_std": 0.01,
                "accuracy": accuracy,
                "accuracy_std": 0.02,
                "examples": 100,
                "online_order_repeats": 10,
            }
            row.update(
                _realized_cost_metrics(
                    l01=l01,
                    l11=1.0,
                    routing_rate=routing_rate,
                    accuracy=accuracy,
                    examples=100,
                )
            )
            row["realized_total_cost_std"] = 2.0
            rows.append(row)

    routing_path = tmp_path / "online_routing_accuracy.png"
    cost_path = tmp_path / "online_cost_vs_l01.png"
    combined_path = tmp_path / "routing_comparison.png"
    _plot_online_routing_accuracy(routing_path, rows, "cached")
    _plot_online_cost_vs_l01(cost_path, rows)
    run_module._plot(combined_path, rows, [], "cached")

    assert routing_path.stat().st_size > 0
    assert cost_path.stat().st_size > 0
    assert combined_path.stat().st_size > 0


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


def test_non_prompt_context_profile_is_explicitly_selectable():
    args = _parser().parse_args(
        ["--cache", "fixture.zip", "--context-profile", "non-prompt"]
    )
    assert args.context_profile == "non-prompt"


def test_online_sweep_runs_every_hgb_capacity_with_fixed_gamma(monkeypatch):
    args = _parser().parse_args(
        [
            "--cache",
            "fixture.zip",
            "--etc-tastes",
            "300",
            "--igw-gamma-values",
            "64",
        ]
    )
    args.l01_values = [2.0]
    round_ = CascadeRound(
        example_id="arc-1",
        prompt="prompt",
        context=np.asarray([0.1, -0.2]),
        weak_answer="A",
        strong_answer="B",
        gold_answer="B",
    )
    collection_flags = []

    def fake_run(
        method,
        player,
        config,
        rounds,
        progress_label,
        metric,
        collect_trajectories,
    ):
        collection_flags.append(collect_trajectories)
        return {"method": method, "routing_rate": 0.5}, []

    monkeypatch.setattr(run_module, "_run_one_player", fake_run)
    rows, trajectories = run_module.run_online(
        [round_], args, collect_trajectories=False
    )

    assert trajectories == []
    assert collection_flags == [False, False, False]
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

    def fake_run(
        method,
        player,
        config,
        rounds,
        progress_label,
        metric,
        collect_trajectories,
    ):
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
