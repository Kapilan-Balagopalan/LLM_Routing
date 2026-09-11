import json

import numpy as np
import pytest

import llm_routing_simulation.tuning as tuning
from llm_routing_simulation.environment import CascadeRound


class _FakeIncrementalTree:
    supports_incremental = True

    def __init__(self):
        self.update_sizes = []
        self.update_records = []
        self.predict_records = []

    def update_many(self, features, labels, weights):
        assert len(features) == len(labels) == len(weights)
        self.update_sizes.append(len(labels))
        self.update_records.append(
            (
                np.asarray(features).copy(),
                np.asarray(labels).copy(),
                np.asarray(weights).copy(),
            )
        )

    def fit_all(self, features, labels, weights):
        raise AssertionError("incremental fake must not receive fit_all")

    def predict_proba(self, features):
        self.predict_records.append(np.asarray(features).copy())
        return np.full(len(features), 0.5)


class _RecordingBatchEstimator:
    supports_incremental = False

    def __init__(self):
        self.fit_records = []
        self.predict_records = []

    def update_many(self, features, labels, weights):
        raise AssertionError("batch fake must not receive update_many")

    def fit_all(self, features, labels, weights):
        self.fit_records.append(
            (
                np.asarray(features).copy(),
                np.asarray(labels).copy(),
                np.asarray(weights).copy(),
            )
        )

    def predict_proba(self, features):
        self.predict_records.append(np.asarray(features).copy())
        return np.full(len(features), 0.5)


def _tree_settings(kind="river-hoeffding"):
    return {
        "kind": kind,
        "hgb_max_leaf_nodes": 15,
        "river_max_depth": 4,
        "river_grace_period": 200,
        "river_delta": 1e-7,
        "river_tau": 0.05,
    }


@pytest.mark.parametrize(
    ("schedule", "expected_boundaries", "expected_epochs"),
    [
        (
            "capped-doubling",
            [1, 2, 4, 8],
            [(1, 0, 1), (2, 1, 3), (4, 3, 7), (8, 7, 10)],
        ),
        (
            "fibonacci",
            [1, 2, 3, 5, 8],
            [(1, 0, 1), (2, 1, 2), (3, 2, 4), (5, 4, 7), (8, 7, 10)],
        ),
        (
            "doubling",
            [1, 2, 4, 8],
            [(1, 0, 1), (2, 1, 3), (4, 3, 7), (8, 7, 10)],
        ),
    ],
)
def test_adaptive_epochs_update_before_boundaries_without_gaps(
    schedule, expected_boundaries, expected_epochs
):
    assert list(tuning._schedule_boundaries(10, schedule)) == expected_boundaries
    epochs = list(tuning._adaptive_epochs(10, schedule))
    assert epochs == expected_epochs
    assert [index for _, start, stop in epochs for index in range(start, stop)] == (
        list(range(10))
    )
    assert all(boundary - 1 == start for boundary, start, _ in epochs)


def test_capped_doubling_limits_late_boundary_gaps_to_100_rounds():
    expected_boundaries = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        228,
        328,
        428,
        528,
    ]
    expected_epochs = [
        (1, 0, 1),
        (2, 1, 3),
        (4, 3, 7),
        (8, 7, 15),
        (16, 15, 31),
        (32, 31, 63),
        (64, 63, 127),
        (128, 127, 227),
        (228, 227, 327),
        (328, 327, 427),
        (428, 427, 527),
        (528, 527, 600),
    ]

    assert list(tuning._schedule_boundaries(600, "capped-doubling")) == (
        expected_boundaries
    )
    assert list(tuning._adaptive_epochs(600, "capped-doubling")) == (
        expected_epochs
    )
    assert max(
        later - earlier
        for earlier, later in zip(expected_boundaries, expected_boundaries[1:])
    ) == 100


def test_capped_doubling_honors_a_custom_maximum_round_gap():
    assert list(tuning._schedule_boundaries(16, "capped-doubling", 3)) == [
        1,
        2,
        4,
        7,
        10,
        13,
        16,
    ]


def test_schedule_boundaries_handle_empty_horizon_and_reject_invalid_inputs():
    assert list(tuning._schedule_boundaries(0, "capped-doubling")) == []
    assert list(tuning._schedule_boundaries(0, "fibonacci")) == []
    assert list(tuning._adaptive_epochs(0, "doubling")) == []
    with pytest.raises(ValueError, match="nonnegative"):
        list(tuning._schedule_boundaries(-1, "fibonacci"))
    with pytest.raises(ValueError, match="Unknown adaptive update schedule"):
        list(tuning._schedule_boundaries(3, "geometric"))
    with pytest.raises(ValueError, match="maximum round gap must be positive"):
        list(tuning._schedule_boundaries(3, "capped-doubling", 0))


def test_doubling_epochs_remains_a_backward_compatible_explicit_schedule():
    assert list(tuning._doubling_epochs(10)) == list(
        tuning._adaptive_epochs(10, "doubling")
    )


@pytest.mark.parametrize(
    (
        "schedule",
        "max_round_gap",
        "expected_fit_sizes",
        "expected_slug",
    ),
    [
        (
            "capped-doubling",
            100,
            [1, 3, 7],
            "capped_doubling_gap_100",
        ),
        ("capped-doubling", 3, [1, 3, 6], "capped_doubling_gap_3"),
        ("fibonacci", 500, [1, 2, 4, 7], "fibonacci"),
        ("doubling", 500, [1, 3, 7], "doubling"),
    ],
)
def test_cbpside_uses_feedback_only_at_next_scheduled_boundary(
    monkeypatch,
    schedule,
    max_round_gap,
    expected_fit_sizes,
    expected_slug,
):
    fit_sizes = []

    def fake_fit(features, outcomes, config, initial_theta):
        fit_sizes.append(len(outcomes))
        return np.zeros(features.shape[1])

    monkeypatch.setattr(tuning, "_fit_logistic_theta", fake_fit)
    contexts = np.asarray([[1.0, (-1.0) ** index] for index in range(8)])
    normalized = tuning._normalized_cbpside_features(contexts)
    outcomes = np.asarray([0, 1] * 4, dtype=np.int8)

    row = tuning._simulate_cbpside_candidate(
        normalized,
        outcomes,
        np.arange(8),
        l01=2.0,
        l11=1.0,
        multiplier=1.0,
        base_beta_scale=0.5,
        confidence_cap=0.5,
        matrix_regularization=1.0,
        theta_regularization=1.0,
        zero_start=False,
        order_index=0,
        order_seed=0,
        policy_seed=0,
        tree_settings=_tree_settings(),
        update_schedule=schedule,
        update_max_round_gap=max_round_gap,
    )

    assert fit_sizes == expected_fit_sizes
    assert row["model_updates"] == len(expected_fit_sizes)
    assert row["last_model_training_count"] == expected_fit_sizes[-1]
    assert row["update_schedule"] == f"global_round_{expected_slug}_before_action"
    assert row["confidence_matrix_state"] == (
        f"frozen_within_each_{expected_slug}_epoch"
    )


@pytest.mark.parametrize(
    (
        "schedule",
        "max_round_gap",
        "expected_update_sizes",
        "expected_training_count",
        "expected_slug",
    ),
    [
        ("capped-doubling", 100, [7], 7, "capped_doubling_gap_100"),
        ("capped-doubling", 3, [6], 6, "capped_doubling_gap_3"),
        ("fibonacci", 500, [4, 3], 7, "fibonacci"),
        ("doubling", 500, [7], 7, "doubling"),
    ],
)
def test_igw_first_feasible_update_waits_for_next_scheduled_boundary(
    monkeypatch,
    schedule,
    max_round_gap,
    expected_update_sizes,
    expected_training_count,
    expected_slug,
):
    fake = _FakeIncrementalTree()
    monkeypatch.setattr(tuning, "make_tree_backend", lambda settings, seed: fake)
    contexts = np.asarray([[index, -index] for index in range(8)], dtype=float)
    outcomes = np.asarray([0, 1] * 4, dtype=np.int8)

    row = tuning._simulate_igw_candidate(
        contexts,
        outcomes,
        np.arange(8),
        policy=tuning.POLICY_IGW_TREE,
        l01=10.0,
        l11=1.0,
        multiplier=1.0,
        base_gamma=1e6,
        mu=2.0,
        min_propensity=0.1,
        order_index=0,
        order_seed=0,
        policy_seed=1,
        estimator_settings=_tree_settings(),
        update_schedule=schedule,
        update_max_round_gap=max_round_gap,
    )

    assert fake.update_sizes == expected_update_sizes
    assert row["model_updates"] == len(expected_update_sizes)
    assert row["last_model_training_count"] == expected_training_count
    assert row["update_schedule"] == f"global_round_{expected_slug}_before_action"


def test_igw_variants_receive_matching_histories_when_predictions_match(
    monkeypatch,
):
    created = {}

    def factory(settings, seed):
        del seed
        estimator = _RecordingBatchEstimator()
        created[settings["kind"]] = estimator
        return estimator

    monkeypatch.setattr(tuning, "make_tree_backend", factory)
    contexts = np.column_stack(
        (np.arange(32, dtype=float), (-1.0) ** np.arange(32))
    )
    outcomes = np.asarray([0, 1] * 16, dtype=np.int8)
    common = {
        "contexts": contexts,
        "outcomes": outcomes,
        "permutation": np.arange(32),
        "l01": 2.0,
        "l11": 1.0,
        "multiplier": 1.0,
        "base_gamma": 1.0,
        "mu": 2.0,
        "min_propensity": 0.1,
        "order_index": 0,
        "order_seed": 0,
        "policy_seed": 7,
        "update_schedule": "fibonacci",
    }
    tree_row = tuning._simulate_igw_candidate(
        **common,
        policy=tuning.POLICY_IGW_TREE,
        estimator_settings=_tree_settings("hgb"),
    )
    linear_row = tuning._simulate_igw_candidate(
        **common,
        policy=tuning.POLICY_IGW_LINEAR,
        estimator_settings=tuning._linear_settings(),
    )

    assert tree_row["routing_rate"] == linear_row["routing_rate"]
    assert tree_row["accuracy"] == linear_row["accuracy"]
    assert tree_row["model_updates"] == linear_row["model_updates"]
    assert tree_row["comparison_role"] == "nonlinear_tree_oracle"
    assert linear_row["comparison_role"] == "linear_oracle"
    tree_fits = created["hgb"].fit_records
    linear_fits = created["logistic"].fit_records
    assert len(tree_fits) == len(linear_fits) > 0
    for tree_fit, linear_fit in zip(tree_fits, linear_fits):
        for tree_value, linear_value in zip(tree_fit, linear_fit):
            assert np.array_equal(tree_value, linear_value)


def test_igw_unrevealed_outcome_does_not_change_later_policy_state(monkeypatch):
    created = []

    def factory(settings, seed):
        del settings, seed
        estimator = _RecordingBatchEstimator()
        created.append(estimator)
        return estimator

    monkeypatch.setattr(tuning, "make_tree_backend", factory)
    contexts = np.column_stack(
        (np.arange(32, dtype=float), (-1.0) ** np.arange(32))
    )
    outcomes = np.asarray([0, 1] * 16, dtype=np.int8)
    common = {
        "contexts": contexts,
        "permutation": np.arange(32),
        "policy": tuning.POLICY_IGW_LINEAR,
        "l01": 2.0,
        "l11": 1.0,
        "multiplier": 1.0,
        "base_gamma": 1.0,
        "mu": 2.0,
        "min_propensity": 0.1,
        "order_index": 0,
        "order_seed": 0,
        "policy_seed": 7,
        "estimator_settings": tuning._linear_settings(),
        "update_schedule": "fibonacci",
    }

    original = tuning._simulate_igw_candidate(outcomes=outcomes, **common)
    first_estimator = created[0]
    final_training_features = first_estimator.fit_records[-1][0]
    revealed_before_last_boundary = {
        int(value) for value in final_training_features[:, 0]
    }
    final_boundary = list(tuning._schedule_boundaries(32, "fibonacci"))[-1]
    unrevealed_before_later_fit = [
        index
        for index in range(final_boundary - 1)
        if index not in revealed_before_last_boundary
    ]
    assert unrevealed_before_later_fit
    unrevealed_index = unrevealed_before_later_fit[0]

    changed = outcomes.copy()
    changed[unrevealed_index] = 1 - changed[unrevealed_index]
    counterfactual = tuning._simulate_igw_candidate(outcomes=changed, **common)
    second_estimator = created[1]

    assert original["routing_rate"] == counterfactual["routing_rate"]
    assert original["model_updates"] == counterfactual["model_updates"]
    assert len(first_estimator.fit_records) == len(second_estimator.fit_records)
    for first_fit, second_fit in zip(
        first_estimator.fit_records, second_estimator.fit_records
    ):
        for first_value, second_value in zip(first_fit, second_fit):
            assert np.array_equal(first_value, second_value)


def test_etc_estimators_share_prefix_freeze_and_reuse_tail_across_losses(
    monkeypatch,
):
    created = {}
    factory_calls = []

    def factory(settings, seed):
        estimator = _RecordingBatchEstimator()
        factory_calls.append((settings.copy(), seed))
        created[settings["kind"]] = estimator
        return estimator

    monkeypatch.setattr(tuning, "make_tree_backend", factory)
    contexts = np.asarray([[index, index / 2] for index in range(10)], dtype=float)
    outcomes = np.asarray([0, 1] * 5, dtype=np.int8)
    common = {
        "contexts": contexts,
        "outcomes": outcomes,
        "permutation": np.arange(10),
        "l01_values": [1.8, 2.0, 3.0],
        "l11": 1.0,
        "multiplier": 1.0,
        "base_tastes": 4.0,
        "order_index": 0,
        "order_seed": 0,
        "policy_seed": 3,
    }

    tree_rows = tuning._simulate_etc_candidates(
        **common,
        policy=tuning.POLICY_ETC,
        estimator_settings=_tree_settings("hgb"),
    )
    linear_rows = tuning._simulate_etc_candidates(
        **common,
        policy=tuning.POLICY_ETC_LINEAR,
        estimator_settings=tuning._linear_settings(),
    )

    assert len(factory_calls) == 2
    tree_fit = created["hgb"].fit_records
    linear_fit = created["logistic"].fit_records
    assert len(tree_fit) == len(linear_fit) == 1
    for tree_value, linear_value in zip(tree_fit[0], linear_fit[0]):
        assert np.array_equal(tree_value, linear_value)
    fit_features, fit_labels, fit_weights = tree_fit[0]
    assert np.array_equal(fit_features, contexts[:4])
    assert np.array_equal(fit_labels, outcomes[:4])
    assert np.array_equal(fit_weights, np.ones(4))
    assert len(created["hgb"].predict_records) == 1
    assert len(created["logistic"].predict_records) == 1
    assert np.array_equal(created["hgb"].predict_records[0], contexts[4:])
    assert np.array_equal(created["logistic"].predict_records[0], contexts[4:])

    for policy, estimator, rows in (
        (tuning.POLICY_ETC, "hgb", tree_rows),
        (tuning.POLICY_ETC_LINEAR, "linear-logistic", linear_rows),
    ):
        assert [row["l01"] for row in rows] == [1.8, 2.0, 3.0]
        assert all(row["policy"] == policy for row in rows)
        assert all(row["effective_parameter"] == 4.0 for row in rows)
        assert all(row["probability_estimator"] == estimator for row in rows)
        assert all(row["model_updates"] == 1 for row in rows)
        assert all(row["last_model_training_count"] == 4 for row in rows)
        assert all(row["forced_prefix_label_counts"] == [2, 2] for row in rows)
        assert all(row["estimator_fit_feasible"] is True for row in rows)
        assert all(row["estimator_fallback"] is None for row in rows)
        assert all(
            row["update_schedule"] == "forced_taste_prefix_fit_then_freeze"
            for row in rows
        )
        assert all(
            row["estimator_feedback_update"]
            == "unit_weight_prefix_fit_then_freeze"
            for row in rows
        )
    assert all(row["tree_estimator"] == "hgb" for row in tree_rows)
    assert all(row["tree_estimator"] is None for row in linear_rows)
    assert [row["routing_rate"] for row in tree_rows] == [0.4, 1.0, 1.0]
    assert [row["routing_rate"] for row in linear_rows] == [0.4, 1.0, 1.0]
    assert all(
        row["comparison_role"] == "nonlinear_tree_oracle" for row in tree_rows
    )
    assert all(row["comparison_role"] == "linear_oracle" for row in linear_rows)


@pytest.mark.parametrize(
    ("policy", "settings"),
    [
        (tuning.POLICY_ETC, _tree_settings("hgb")),
        (tuning.POLICY_ETC_LINEAR, tuning._linear_settings()),
    ],
)
def test_etc_records_the_infeasible_prefix_prevalence_fallback(
    monkeypatch, policy, settings
):
    def unexpected_factory(estimator_settings, seed):
        del estimator_settings, seed
        raise AssertionError("an infeasible forced prefix must not fit a model")

    monkeypatch.setattr(tuning, "make_tree_backend", unexpected_factory)
    contexts = np.arange(12, dtype=float).reshape(6, 2)
    outcomes = np.asarray([0, 0, 0, 1, 0, 1], dtype=np.int8)

    [row] = tuning._simulate_etc_candidates(
        contexts,
        outcomes,
        np.arange(6),
        policy=policy,
        l01_values=[2.0],
        l11=1.0,
        multiplier=1.0,
        base_tastes=3.0,
        order_index=0,
        order_seed=0,
        policy_seed=3,
        estimator_settings=settings,
    )

    assert row["forced_prefix_label_counts"] == [3, 0]
    assert row["estimator_fit_feasible"] is False
    assert row["estimator_fallback"] == (
        "laplace_smoothed_forced_prefix_prevalence"
    )
    assert row["model_updates"] == 0
    assert row["last_model_training_count"] == 0
    assert row["routing_rate"] == 0.5


def test_candidate_selection_is_pointwise_and_uses_deterministic_tie_break():
    candidates = []
    for policy in tuning.TUNED_POLICIES:
        for l01 in (1.8, 2.0):
            for multiplier, cost in ((0.3, 12.0), (1.0, 10.0), (3.0, 10.0)):
                candidates.append(
                    {
                        "policy": policy,
                        "method": policy,
                        "l01": l01,
                        "l11": 1.0,
                        "alpha": 1.0 / l01,
                        "multiplier": multiplier,
                        "parameter_name": "parameter",
                        "base_parameter": 1.0,
                        "effective_parameter": multiplier,
                        "realized_total_cost_mean": cost,
                    }
                )

    selected = tuning._select_pointwise_multipliers(candidates)

    assert tuning.TUNED_POLICIES == (
        tuning.POLICY_CBPSIDE,
        tuning.POLICY_ETC,
        tuning.POLICY_ETC_LINEAR,
        tuning.POLICY_IGW_LINEAR,
        tuning.POLICY_IGW_TREE,
    )
    assert len(selected) == 10
    assert [(row["policy"], row["l01"]) for row in selected] == [
        (policy, l01)
        for policy in tuning.TUNED_POLICIES
        for l01 in (1.8, 2.0)
    ]
    assert {row["selected_multiplier"] for row in selected} == {1.0}
    assert all(row["selection_evaluation_reuse"] for row in selected)

    tied = [
        {
            **candidates[0],
            "multiplier": multiplier,
            "effective_parameter": multiplier,
            "realized_total_cost_mean": 10.0,
        }
        for multiplier in (0.3, 3.0)
    ]
    [tie_winner] = tuning._select_pointwise_multipliers(tied)
    assert tie_winner["selected_multiplier"] == 0.3


def test_candidate_aggregation_uses_sample_standard_deviation():
    rows = []
    for order, cost in ((1, 10.0), (2, 14.0)):
        rows.append(
            {
                "policy": "IGW",
                "method": "IGW",
                "l01": 2.0,
                "l11": 1.0,
                "alpha": 0.5,
                "multiplier": 0.3,
                "parameter_name": "gamma",
                "base_parameter": 10.0,
                "effective_parameter": 3.0,
                "order_run": order,
                "order_seed": order - 1,
                "order_was_shuffled": True,
                "examples": 10,
                "routing_rate": 0.5,
                "accuracy": 0.8,
                "routed_to_strong": 5.0,
                "unrouted_disagreements": 2.0,
                "realized_cost_per_example": cost / 10,
                "realized_total_cost": cost,
                "model_updates": 3,
                "last_model_training_count": 5,
            }
        )

    [aggregate] = tuning._aggregate_candidates(rows)

    assert aggregate["realized_total_cost_mean"] == pytest.approx(12.0)
    assert aggregate["realized_total_cost_std"] == pytest.approx(np.sqrt(8.0))
    assert aggregate["realized_total_cost_sem"] == pytest.approx(2.0)


def test_igw_comparison_uses_paired_order_level_cost_differences():
    rows = []
    for order, tree_cost, linear_cost in ((1, 8.0, 10.0), (2, 12.0, 10.0)):
        common = {
            "l01": 2.0,
            "l11": 1.0,
            "alpha": 0.5,
            "order_run": order,
            "order_seed": order - 1,
            "examples": 10,
            "selected_multiplier": 1.0,
            "effective_parameter": 3.0,
            "routing_rate": 0.5,
            "accuracy": 0.8,
        }
        rows.extend(
            [
                {
                    **common,
                    "policy": tuning.POLICY_IGW_TREE,
                    "method": "IGW + HGB leaves=15",
                    "realized_total_cost": tree_cost,
                },
                {
                    **common,
                    "policy": tuning.POLICY_IGW_LINEAR,
                    "method": "IGW + linear logistic",
                    "realized_total_cost": linear_cost,
                },
            ]
        )

    paired = tuning._igw_comparison_by_order(rows)
    [aggregate] = tuning._aggregate_igw_comparison(paired)
    matched_rows = [dict(row, multiplier=1.0) for row in rows]
    matched_paired = tuning._igw_matched_comparison_by_order(matched_rows)
    [matched_aggregate] = tuning._aggregate_igw_comparison(
        matched_paired, group_by_multiplier=True
    )

    assert [row["nonlinear_tree_cost_reduction"] for row in paired] == [
        2.0,
        -2.0,
    ]
    assert aggregate["nonlinear_tree_cost_reduction_mean"] == 0.0
    assert aggregate["nonlinear_tree_cost_reduction_std"] == pytest.approx(
        np.sqrt(8.0)
    )
    assert aggregate["tree_lower_cost_orders"] == 1
    assert aggregate["linear_lower_cost_orders"] == 1
    assert aggregate["comparison_scope"] == "separately_tuned_best_vs_best"
    assert matched_aggregate["comparison_scope"] == (
        "matched_gamma_estimator_contrast"
    )
    assert matched_aggregate["same_effective_gamma"] is True
    assert matched_aggregate["nonlinear_tree_cost_reduction_mean"] == 0.0

    mismatched = [dict(row) for row in matched_rows]
    next(
        row for row in mismatched if row["policy"] == tuning.POLICY_IGW_LINEAR
    )["effective_parameter"] = 4.0
    with pytest.raises(RuntimeError, match="different gamma"):
        tuning._igw_matched_comparison_by_order(mismatched)


def test_checkpoint_rejects_a_different_sweep_fingerprint(tmp_path):
    path = tmp_path / "row.json"
    path.write_text(
        json.dumps({"config_fingerprint": "old", "policy": "ETC"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="different configuration"):
        tuning._load_checkpoint(path, "new")


def test_parser_keeps_established_hgb_as_explicit_default():
    base = ["--cache", "cache.zip", "--output-dir", "results"]
    args = tuning._parser().parse_args(base)
    fibonacci_args = tuning._parser().parse_args(
        base + ["--adaptive-update-schedule", "fibonacci"]
    )
    doubling_args = tuning._parser().parse_args(
        base
        + [
            "--adaptive-update-schedule",
            "doubling",
            "--tree-estimator",
            "river-hoeffding",
        ]
    )

    assert args.context_profile == "all-features"
    assert args.tree_estimator == "hgb"
    assert args.hgb_max_leaf_nodes == 15
    assert args.adaptive_update_schedule == "capped-doubling"
    assert args.adaptive_max_round_gap == 100
    assert fibonacci_args.adaptive_update_schedule == "fibonacci"
    assert doubling_args.adaptive_update_schedule == "doubling"
    assert tuning._tree_settings(doubling_args)["kind"] == "river-hoeffding"
    assert tuning._etc_hgb_settings(doubling_args) == {
        "kind": "hgb",
        "hgb_max_leaf_nodes": 15,
    }
    assert args.online_order_repeats == 20
    assert args.multipliers == [0.1, 0.3, 1.0, 3.0, 10.0]
    assert tuning._expected_checkpoint_count(args) == 4500


def test_manifest_fingerprint_covers_contexts_and_cbpside_regularization(tmp_path):
    base = [
        "--cache",
        str(tmp_path / "cache.zip"),
        "--output-dir",
        str(tmp_path / "results"),
    ]
    contexts = np.column_stack(
        (np.arange(10, dtype=float), np.arange(10, dtype=float) + 1.0)
    )
    outcomes = np.asarray([0, 1] * 5, dtype=np.int8)
    common = {
        "contexts": contexts,
        "outcomes": outcomes,
        "example_ids": [f"example-{index}" for index in range(10)],
        "context_summary": {"profile": "all-features"},
    }
    default_args = tuning._parser().parse_args(base)
    manifest, default_fingerprint = tuning._manifest_and_fingerprint(
        default_args, **common
    )
    matrix_args = tuning._parser().parse_args(
        base + ["--cbpside-matrix-regularization", "2"]
    )
    _, matrix_fingerprint = tuning._manifest_and_fingerprint(
        matrix_args, **common
    )
    theta_args = tuning._parser().parse_args(
        base + ["--cbpside-theta-regularization", "2"]
    )
    _, theta_fingerprint = tuning._manifest_and_fingerprint(
        theta_args, **common
    )
    changed_contexts = dict(common)
    changed_contexts["contexts"] = contexts + 0.25
    _, context_fingerprint = tuning._manifest_and_fingerprint(
        default_args, **changed_contexts
    )
    fibonacci_args = tuning._parser().parse_args(
        base + ["--adaptive-update-schedule", "fibonacci"]
    )
    fibonacci_manifest, fibonacci_fingerprint = (
        tuning._manifest_and_fingerprint(fibonacci_args, **common)
    )
    doubling_args = tuning._parser().parse_args(
        base + ["--adaptive-update-schedule", "doubling"]
    )
    doubling_manifest, doubling_fingerprint = tuning._manifest_and_fingerprint(
        doubling_args, **common
    )
    short_gap_args = tuning._parser().parse_args(
        base + ["--adaptive-max-round-gap", "3"]
    )
    short_gap_manifest, short_gap_fingerprint = (
        tuning._manifest_and_fingerprint(short_gap_args, **common)
    )

    assert manifest["design"] == "pointwise-online-parameter-multiplier-sweep-v5"
    assert manifest["implementation_revision"] == 5
    assert manifest["tuned_policies"] == list(tuning.TUNED_POLICIES)
    assert manifest["update_schedule"] == {
        "name": "capped-doubling",
        "maximum_round_gap": 100,
        "boundary_rule": (
            "before global rounds starting at 1, with "
            "next=min(2*current, current+100)"
        ),
        "boundary_rounds": [1, 2, 4, 8],
        "boundary_count": 4,
        "maximum_model_updates": 3,
        "last_boundary_round": 8,
        "final_frozen_epoch_rounds": 3,
        "history_cutoff": "feedback through boundary_round-1",
        "cbpside_theta": "refit only at boundary when new tastes exist",
        "cbpside_V_inverse": "recompute only at boundary and freeze in epoch",
        "cbpside_beta": "evaluate on every current context using epoch V inverse",
        "igw_tree": "full revealed-history refit only at boundary",
        "igw_linear": "full revealed-history refit only at boundary",
    }
    assert fibonacci_manifest["update_schedule"] == {
        **manifest["update_schedule"],
        "name": "fibonacci",
        "maximum_round_gap": None,
        "boundary_rule": "before global rounds 1,2,3,5,8,...",
        "boundary_rounds": [1, 2, 3, 5, 8],
        "boundary_count": 5,
        "maximum_model_updates": 4,
    }
    assert doubling_manifest["update_schedule"]["name"] == "doubling"
    assert doubling_manifest["update_schedule"]["maximum_round_gap"] is None
    assert doubling_manifest["update_schedule"]["boundary_rounds"] == [
        1,
        2,
        4,
        8,
    ]
    assert short_gap_manifest["update_schedule"]["name"] == "capped-doubling"
    assert short_gap_manifest["update_schedule"]["maximum_round_gap"] == 3
    assert short_gap_manifest["update_schedule"]["boundary_rounds"] == [
        1,
        2,
        4,
        7,
        10,
    ]
    assert manifest["etc_hgb_profile"]["settings"]["kind"] == "hgb"
    assert manifest["etc_linear_profile"]["settings"]["kind"] == "logistic"
    assert (
        manifest["etc_linear_profile"]["forced_prefix_shared_with"]
        == tuning.POLICY_ETC
    )
    assert manifest["etc_estimator_comparison"]["tree_policy"] == tuning.POLICY_ETC
    assert (
        manifest["etc_estimator_comparison"]["linear_policy"]
        == tuning.POLICY_ETC_LINEAR
    )
    assert manifest["igw_linear_profile"]["kind"] == "logistic"
    assert "fixed_multiplier_contrast" in manifest["igw_estimator_comparison"]
    assert "selected_contrast" in manifest["igw_estimator_comparison"]
    assert manifest["base_parameters"]["cbpside_matrix_regularization"] == 1.0
    assert manifest["base_parameters"]["cbpside_theta_regularization"] == 1.0
    assert len(
        {
            default_fingerprint,
            matrix_fingerprint,
            theta_fingerprint,
            context_fingerprint,
            fibonacci_fingerprint,
            doubling_fingerprint,
            short_gap_fingerprint,
        }
    ) == 7


def test_tuner_rejects_a_nonunit_strong_route_cost():
    args = tuning._parser().parse_args(
        [
            "--cache",
            "cache.zip",
            "--output-dir",
            "results",
            "--l11",
            "0.5",
        ]
    )

    with pytest.raises(SystemExit, match="fixes l11=1"):
        tuning._validate_args(args)


def test_tuner_rejects_a_nonpositive_adaptive_maximum_gap():
    args = tuning._parser().parse_args(
        [
            "--cache",
            "cache.zip",
            "--output-dir",
            "results",
            "--adaptive-max-round-gap",
            "0",
        ]
    )

    with pytest.raises(SystemExit, match="adaptive-max-round-gap must be positive"):
        tuning._validate_args(args)


def test_small_end_to_end_sweep_resumes_and_plot_only_uses_checkpoints(
    monkeypatch, tmp_path
):
    rounds = []
    for index in range(8):
        disagree = index % 2
        rounds.append(
            CascadeRound(
                example_id=f"boolq-{index}",
                prompt="prompt",
                context=np.asarray([index / 8.0, (-1.0) ** index]),
                weak_answer="weak",
                strong_answer="strong" if disagree else "weak",
                gold_answer="unused",
            )
        )
    monkeypatch.setattr(tuning, "load_cache", lambda path: object())
    monkeypatch.setattr(
        tuning,
        "_prompt_context_rounds",
        lambda *args, **kwargs: (
            rounds,
            {
                "profile": "all-features",
                "context_dimension": 2,
                "context_blocks": [],
            },
            None,
            [],
        ),
    )
    created_backends = []

    def factory(settings, seed):
        del seed
        backend = (
            _FakeIncrementalTree()
            if settings["kind"] == "river-hoeffding"
            else _RecordingBatchEstimator()
        )
        created_backends.append(backend)
        return backend

    monkeypatch.setattr(tuning, "make_tree_backend", factory)
    output = tmp_path / "sweep"
    arguments = [
        "--cache",
        str(tmp_path / "cache.zip"),
        "--output-dir",
        str(output),
        "--l01-values",
        "2",
        "--multipliers",
        "1",
        "--online-order-repeats",
        "2",
        "--tree-estimator",
        "river-hoeffding",
    ]

    assert tuning.main(arguments) == 0
    checkpoints = list((output / "checkpoints").rglob("*.json"))
    assert len(checkpoints) == 10
    checkpoint_policies = {
        path.relative_to(output / "checkpoints").parts[0] for path in checkpoints
    }
    assert checkpoint_policies == {
        "cbpside",
        "etc",
        "etclinear",
        "igwlinear",
        "igw",
    }
    manifest = json.loads(
        (output / "sweep_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["update_schedule"]["name"] == "capped-doubling"
    assert manifest["update_schedule"]["maximum_round_gap"] == 100
    assert manifest["update_schedule"]["boundary_rounds"] == [1, 2, 4, 8]
    assert manifest["tuned_policies"] == list(tuning.TUNED_POLICIES)
    candidate_rows = json.loads(
        (output / "candidate_results_by_order.json").read_text(encoding="utf-8")
    )
    assert len(candidate_rows) == 10
    assert {
        policy: sum(row["policy"] == policy for row in candidate_rows)
        for policy in tuning.TUNED_POLICIES
    } == {policy: 2 for policy in tuning.TUNED_POLICIES}
    for policy in (tuning.POLICY_ETC, tuning.POLICY_ETC_LINEAR):
        rows = [row for row in candidate_rows if row["policy"] == policy]
        assert all(
            row["update_schedule"] == "forced_taste_prefix_fit_then_freeze"
            for row in rows
        )
    for policy in (
        tuning.POLICY_CBPSIDE,
        tuning.POLICY_IGW_LINEAR,
        tuning.POLICY_IGW_TREE,
    ):
        rows = [row for row in candidate_rows if row["policy"] == policy]
        assert all(
            row["update_schedule"]
            == "global_round_capped_doubling_gap_100_before_action"
            for row in rows
        )
    selected = json.loads(
        (output / "selected_results.json").read_text(encoding="utf-8")
    )
    assert [row["policy"] for row in selected] == [
        *tuning.TUNED_POLICIES,
        "Random",
    ]
    selected_by_order = json.loads(
        (output / "selected_results_by_order.json").read_text(encoding="utf-8")
    )
    random_rows = [row for row in selected_by_order if row["policy"] == "Random"]
    assert len(random_rows) == 2
    assert all(
        row["matched_etc_policy"] == tuning.POLICY_ETC for row in random_rows
    )
    assert all(
        row["matched_etc_probability_estimator"] == "hgb" for row in random_rows
    )
    assert (output / "selected_routing_accuracy.png").stat().st_size > 0
    assert (output / "selected_cost_vs_l01.png").stat().st_size > 0
    assert (output / "igw_tree_vs_linear.csv").stat().st_size > 0
    assert (output / "igw_tree_vs_linear_matched.csv").stat().st_size > 0
    assert (
        output / "igw_tree_vs_linear_cost_difference.png"
    ).stat().st_size > 0
    assert (
        output / "igw_tree_vs_linear_matched_cost_difference.png"
    ).stat().st_size > 0
    assert (output / "multiplier-sweep-results.zip").stat().st_size > 0

    created_count = len(created_backends)
    assert tuning.main(arguments) == 0
    assert len(created_backends) == created_count
    assert tuning.main(arguments + ["--plot-only"]) == 0
    assert len(created_backends) == created_count
