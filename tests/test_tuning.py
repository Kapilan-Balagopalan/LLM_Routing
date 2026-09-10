import json

import numpy as np
import pytest

import llm_routing_simulation.tuning as tuning
from llm_routing_simulation.environment import CascadeRound


class _FakeIncrementalTree:
    supports_incremental = True

    def __init__(self):
        self.update_sizes = []

    def update_many(self, features, labels, weights):
        assert len(features) == len(labels) == len(weights)
        self.update_sizes.append(len(labels))

    def fit_all(self, features, labels, weights):
        raise AssertionError("incremental fake must not receive fit_all")

    def predict_proba(self, features):
        return np.full(len(features), 0.5)


class _RecordingBatchEstimator:
    supports_incremental = False

    def __init__(self):
        self.fit_records = []

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


def test_doubling_epochs_update_before_power_of_two_rounds():
    assert list(tuning._doubling_epochs(10)) == [
        (1, 0, 1),
        (2, 1, 3),
        (4, 3, 7),
        (8, 7, 10),
    ]


def test_cbpside_doubling_uses_feedback_only_at_next_boundary(monkeypatch):
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
    )

    assert fit_sizes == [1, 3, 7]
    assert row["model_updates"] == 3
    assert row["last_model_training_count"] == 7
    assert row["confidence_matrix_state"] == "frozen_within_each_doubling_epoch"


def test_igw_first_feasible_tree_update_waits_for_next_power(monkeypatch):
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
    )

    # Two samples per class first become available after round 4, but the
    # predictor is not updated until the next boundary, immediately before t=8.
    assert fake.update_sizes == [7]
    assert row["model_updates"] == 1
    assert row["last_model_training_count"] == 7


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
    }

    original = tuning._simulate_igw_candidate(outcomes=outcomes, **common)
    first_estimator = created[0]
    final_training_features = first_estimator.fit_records[-1][0]
    revealed_before_last_boundary = {
        int(value) for value in final_training_features[:, 0]
    }
    unrevealed_index = next(
        index for index in range(31) if index not in revealed_before_last_boundary
    )

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


def test_etc_reuses_one_prefix_fit_for_all_loss_values(monkeypatch):
    fake = _FakeIncrementalTree()
    factory_calls = []

    def factory(settings, seed):
        factory_calls.append((settings, seed))
        return fake

    monkeypatch.setattr(tuning, "make_tree_backend", factory)
    contexts = np.asarray([[index, index / 2] for index in range(10)], dtype=float)
    outcomes = np.asarray([0, 1] * 5, dtype=np.int8)

    rows = tuning._simulate_etc_candidates(
        contexts,
        outcomes,
        np.arange(10),
        l01_values=[1.8, 2.0, 3.0],
        l11=1.0,
        multiplier=1.0,
        base_tastes=4.0,
        order_index=0,
        order_seed=0,
        policy_seed=3,
        tree_settings=_tree_settings(),
    )

    assert len(factory_calls) == 1
    assert fake.update_sizes == [4]
    assert [row["l01"] for row in rows] == [1.8, 2.0, 3.0]
    assert all(row["effective_parameter"] == 4.0 for row in rows)


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

    assert len(selected) == 8
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
    args = tuning._parser().parse_args(
        ["--cache", "cache.zip", "--output-dir", "results"]
    )

    assert args.context_profile == "all-features"
    assert args.tree_estimator == "hgb"
    assert args.hgb_max_leaf_nodes == 15
    assert args.online_order_repeats == 20
    assert args.multipliers == [0.1, 0.3, 1.0, 3.0, 10.0]
    assert tuning._expected_checkpoint_count(args) == 3600


def test_manifest_fingerprint_covers_contexts_and_cbpside_regularization(tmp_path):
    base = [
        "--cache",
        str(tmp_path / "cache.zip"),
        "--output-dir",
        str(tmp_path / "results"),
    ]
    contexts = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    outcomes = np.asarray([0, 1], dtype=np.int8)
    common = {
        "contexts": contexts,
        "outcomes": outcomes,
        "example_ids": ["a", "b"],
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

    assert manifest["implementation_revision"] == 3
    assert manifest["tuned_policies"] == list(tuning.TUNED_POLICIES)
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
        }
    ) == 4


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
        backend = _FakeIncrementalTree()
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
    assert len(checkpoints) == 8
    selected = json.loads(
        (output / "selected_results.json").read_text(encoding="utf-8")
    )
    assert {row["policy"] for row in selected} == {
        "CBPSide",
        "ETC",
        "IGW",
        "IGWLinear",
        "Random",
    }
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
    assert tuning.main(arguments + ["--plot-only"]) == 0
    assert len(created_backends) == created_count
