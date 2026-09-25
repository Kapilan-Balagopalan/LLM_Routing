import csv
import json

import numpy as np
import pytest

import llm_routing_simulation.plot_style as plot_style
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

    def update_many(self, features, labels, weights=None):
        raise AssertionError("batch fake must not receive update_many")

    def fit_all(self, features, labels, weights=None):
        if weights is None:
            weights = np.ones(len(labels), dtype=np.float64)
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


class _CrossFitReferenceEstimator:
    """Expose row identifiers so the test can prove held-out prediction."""

    supports_incremental = False

    def __init__(self, records):
        self.records = records
        self.training_ids = None

    def fit_all(self, features, labels, weights=None):
        features = np.asarray(features)
        self.training_ids = set(features[:, 0].astype(int).tolist())
        self.records.append(
            {
                "training_ids": self.training_ids,
                "training_labels": np.asarray(labels).copy(),
                "weights": None if weights is None else np.asarray(weights).copy(),
            }
        )

    def update_many(self, features, labels, weights=None):
        raise AssertionError("cross-fitted HGB reference must use full fits")

    def predict_proba(self, features):
        held_out_ids = np.asarray(features)[:, 0].astype(int)
        assert self.training_ids is not None
        assert self.training_ids.isdisjoint(held_out_ids.tolist())
        self.records[-1]["held_out_ids"] = set(held_out_ids.tolist())
        return 0.05 + 0.09 * held_out_ids


def test_publication_plot_style_contract_and_dual_format_save(tmp_path):
    assert plot_style.PLOT_CONFIDENCE_LEVEL == 0.95
    assert plot_style.PUBLICATION_PNG_DPI == 400
    assert plot_style.PUBLICATION_FIGSIZE == (7.0, 4.25)
    assert plot_style.PUBLICATION_FIGSIZE_SHORT == (7.0, 3.8)
    assert plot_style.LEGEND_FONT_SIZE == 8.0
    assert plot_style.AXIS_LABELS["round"] == r"Round ($t$)"
    assert plot_style.AXIS_LABELS["cumulative_reference_regret"] == (
        "Regret (excess cost)"
    )
    assert plot_style.AXIS_LABELS["l01"] == r"$\ell_{01}$"
    assert plot_style.AXIS_LABELS["total_cost"] == "Total cost"
    assert plot_style.METHOD_LABELS == {
        "pgts": "PG-TS (Bayesian logistic)",
        "random": "Random",
    }
    expected_t_critical = 2.093024054408263
    expected_half_width = expected_t_critical / np.sqrt(20.0)
    assert plot_style.student_t_critical_value(20) == pytest.approx(
        expected_t_critical
    )
    assert plot_style.student_t_half_width(1.0, 20) == pytest.approx(
        expected_half_width
    )
    assert plot_style.student_t_half_width(1.0, 20) != pytest.approx(
        0.5 * expected_half_width
    )

    class RecordingFigure:
        def __init__(self):
            self.calls = []

        def savefig(self, path, **kwargs):
            self.calls.append((path, kwargs))

    figure = RecordingFigure()
    png = tmp_path / "figure.png"
    png_result, pdf_result = plot_style.save_publication_figure(figure, png)

    assert png_result == png
    assert pdf_result == png.with_suffix(".pdf")
    assert figure.calls == [
        (
            png,
            {
                "dpi": 400,
                "bbox_inches": "tight",
                "pad_inches": 0.03,
            },
        ),
        (
            png.with_suffix(".pdf"),
            {"bbox_inches": "tight", "pad_inches": 0.03},
        ),
    ]
    with pytest.raises(ValueError, match=".png suffix"):
        plot_style.save_publication_figure(figure, tmp_path / "figure.pdf")


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
    "actions",
    [
        np.asarray([], dtype=bool),
        np.asarray([1], dtype=np.uint8),
        np.asarray([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1], dtype=np.uint8),
    ],
)
def test_action_payload_round_trip_preserves_exact_length_and_order(actions):
    row = {"examples": int(actions.size), **tuning._encode_action_payload(actions)}

    decoded = tuning._decode_action_payload(row)

    assert decoded.dtype == np.bool_
    assert decoded.shape == actions.shape
    assert decoded.tolist() == actions.astype(bool).tolist()


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


def test_capped_doubling_limits_late_boundary_gaps_to_32_rounds():
    expected_boundaries = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        96,
        128,
        160,
    ]
    expected_epochs = [
        (1, 0, 1),
        (2, 1, 3),
        (4, 3, 7),
        (8, 7, 15),
        (16, 15, 31),
        (32, 31, 63),
        (64, 63, 95),
        (96, 95, 127),
        (128, 127, 159),
        (160, 159, 160),
    ]

    assert list(tuning._schedule_boundaries(160, "capped-doubling")) == (
        expected_boundaries
    )
    assert list(tuning._adaptive_epochs(160, "capped-doubling")) == (
        expected_epochs
    )
    assert max(
        later - earlier
        for earlier, later in zip(expected_boundaries, expected_boundaries[1:])
    ) == 32


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
            32,
            [1, 3, 7],
            "capped_doubling_gap_32",
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
        ("capped-doubling", 32, [7], 7, "capped_doubling_gap_32"),
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


def test_pgts_candidate_draws_only_at_doubling_boundaries_from_prior_feedback(
    monkeypatch,
):
    import llm_routing_simulation.pgts as pgts

    histories = []

    class FakeSampler:
        def __init__(self, dimension, *, gibbs_steps, prior_std, seed):
            assert dimension == 2
            assert gibbs_steps == 15
            assert prior_std == pytest.approx(0.6)
            assert seed == 7

        def draw_theta(self, features, outcomes):
            histories.append(
                (np.asarray(features).copy(), np.asarray(outcomes).copy())
            )
            return np.asarray([10.0, 0.0])

    monkeypatch.setattr(pgts, "PolyaGammaThompsonSampler", FakeSampler)
    features = np.column_stack((np.ones(8), np.arange(8, dtype=float)))
    outcomes = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)

    row = tuning._simulate_pgts_candidate(
        features,
        outcomes,
        np.arange(8),
        l01=2.0,
        l11=1.0,
        gibbs_steps=15,
        prior_std=2.0,
        multiplier=0.3,
        order_index=0,
        order_seed=3,
        policy_seed=7,
        update_schedule="doubling",
    )

    # Feedback generated inside an epoch is held back from posterior sampling
    # until the next global doubling boundary.  In particular, the draw at
    # round 4 sees rounds 1--3, and one theta is reused for rounds 4--7.
    assert [len(labels) for _, labels in histories] == [0, 1, 3, 7]
    assert histories[1][1].tolist() == [1]
    assert histories[2][1].tolist() == [1, 0, 1]
    assert histories[3][1].tolist() == [1, 0, 1, 0, 1, 0, 1]
    assert histories[2][0].tolist() == features[:3].tolist()
    assert row["policy"] == tuning.POLICY_PGTS
    assert row["multiplier"] == 0.3
    assert row["parameter_name"] == "prior_std"
    assert row["base_parameter"] == 2.0
    assert row["effective_parameter"] == pytest.approx(0.6)
    assert row["pgts_base_prior_std"] == 2.0
    assert row["pgts_prior_std_multiplier"] == 0.3
    assert row["pgts_prior_std"] == pytest.approx(0.6)
    assert row["routing_rate"] == 1.0
    assert row["accuracy"] == 1.0
    assert row["update_schedule"] == (
        "global_round_doubling_posterior_draw_before_action"
    )
    assert row["pgts_algorithm1_exact"] is False
    assert row["posterior_eligible_boundary_rounds"] == [1, 2, 4, 8]
    assert row["posterior_eligible_boundary_count"] == 4
    assert row["posterior_draw_rounds"] == [1, 2, 4, 8]
    assert row["posterior_boundary_draw_count"] == 4
    assert row["posterior_skipped_clean_boundary_count"] == 0
    assert row["model_updates"] == 4
    assert row["last_model_training_count"] == 7
    assert row["last_posterior_training_count"] == 7
    assert row["final_revealed_count"] == 8
    assert row["total_gibbs_transitions"] == 60


def test_pgts_reuses_theta_at_clean_doubling_boundaries(monkeypatch):
    import llm_routing_simulation.pgts as pgts

    histories = []

    class FakeSampler:
        def __init__(self, dimension, *, gibbs_steps, prior_std, seed):
            del dimension, gibbs_steps, prior_std, seed

        def draw_theta(self, features, outcomes):
            histories.append(
                (np.asarray(features).copy(), np.asarray(outcomes).copy())
            )
            return np.asarray([-10.0, 0.0])

    monkeypatch.setattr(pgts, "PolyaGammaThompsonSampler", FakeSampler)
    features = np.column_stack((np.ones(8), np.arange(8, dtype=float)))
    outcomes = np.ones(8, dtype=np.int8)

    row = tuning._simulate_pgts_candidate(
        features,
        outcomes,
        np.arange(8),
        l01=2.0,
        l11=1.0,
        gibbs_steps=15,
        prior_std=1.0,
        multiplier=1.0,
        order_index=0,
        order_seed=3,
        policy_seed=7,
        update_schedule="doubling",
    )

    # Round 1 establishes the prior draw.  With no subsequent action-1
    # feedback, the same theta is reused at rounds 2, 4, and 8 without Gibbs.
    assert len(histories) == 1
    assert histories[0][0].shape == (0, 2)
    assert histories[0][1].shape == (0,)
    assert row["multiplier"] == 1.0
    assert row["parameter_name"] == "prior_std"
    assert row["base_parameter"] == 1.0
    assert row["effective_parameter"] == 1.0
    assert row["routing_rate"] == 0.0
    assert row["accuracy"] == 0.0
    assert row["posterior_eligible_boundary_rounds"] == [1, 2, 4, 8]
    assert row["posterior_draw_rounds"] == [1]
    assert row["posterior_boundary_draw_count"] == 1
    assert row["posterior_skipped_clean_boundary_count"] == 3
    assert row["model_updates"] == 1
    assert row["last_model_training_count"] == 0
    assert row["last_posterior_training_count"] == 0
    assert row["final_revealed_count"] == 0
    assert row["total_gibbs_transitions"] == 15


def test_candidate_selection_is_pointwise_and_uses_deterministic_tie_break():
    candidates = []
    for policy in tuning.TUNED_POLICIES:
        for l01 in tuning.DEFAULT_L01_VALUES:
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
        tuning.POLICY_IGW_LINEAR,
        tuning.POLICY_IGW_TREE,
        tuning.POLICY_PGTS,
    )
    assert len(selected) == 36
    assert [(row["policy"], row["l01"]) for row in selected] == [
        (policy, l01)
        for policy in tuning.TUNED_POLICIES
        for l01 in tuning.DEFAULT_L01_VALUES
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
                "policy": "SquareCB.PMSide",
                "method": "SquareCB.PMSide + HGB leaves=15",
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


def test_random_baseline_matches_only_selected_squarecb_pmside_tree_traffic():
    common = {
        "l01": 2.0,
        "l11": 1.0,
        "alpha": 0.5,
        "order_run": 1,
        "order_seed": 0,
        "examples": 10,
        "selected_multiplier": 3.0,
        "probability_estimator": "hgb",
    }
    selected = [
        {
            **common,
            "policy": tuning.POLICY_CBPSIDE,
            "routing_rate": 0.1,
        },
        {
            **common,
            "policy": tuning.POLICY_IGW_LINEAR,
            "routing_rate": 0.2,
        },
        {
            **common,
            "policy": tuning.POLICY_IGW_TREE,
            "routing_rate": 0.6,
        },
    ]

    [row] = tuning._expected_random_rows(
        selected,
        np.asarray([0, 1] * 5, dtype=np.int8),
    )

    assert row["routing_rate"] == 0.6
    assert row["effective_parameter"] == 0.6
    assert row["method"] == "Random"
    assert row["parameter_name"] == (
        "matched_squarecb_pmside_tree_routing_rate"
    )
    assert (
        row["matched_squarecb_pmside_policy"] == tuning.POLICY_IGW_TREE
    )
    assert row["matched_squarecb_pmside_probability_estimator"] == "hgb"
    assert row["matched_squarecb_pmside_multiplier"] == 3.0
    assert "matched_etc_policy" not in row
    assert "matched_igw_policy" not in row


def test_cross_fitted_hgb_reference_predicts_every_row_out_of_fold(
    monkeypatch,
):
    contexts = np.column_stack(
        (np.arange(10, dtype=float), np.linspace(-1.0, 1.0, 10))
    )
    outcomes = np.asarray([0, 1] * 5, dtype=np.int8)
    records = []
    factory_calls = []

    def factory(settings, *, seed):
        factory_calls.append((dict(settings), seed))
        return _CrossFitReferenceEstimator(records)

    monkeypatch.setattr(tuning, "make_tree_backend", factory)

    reference = tuning._cross_fitted_hgb_reference(
        contexts,
        outcomes,
        folds=5,
        seed=13,
    )

    repeat = tuning._cross_fitted_hgb_reference(
        contexts,
        outcomes,
        folds=5,
        seed=13,
    )

    assert factory_calls == [({"kind": "hgb"}, 13)] * 10
    assert np.array_equal(reference["fold_index"], repeat["fold_index"])
    assert np.array_equal(reference["probability"], repeat["probability"])
    assert len(records) == 10
    assert set().union(*(record["held_out_ids"] for record in records)) == set(
        range(10)
    )
    assert all(
        record["training_ids"].isdisjoint(record["held_out_ids"])
        for record in records
    )
    assert np.allclose(
        reference["probability"],
        0.05 + 0.09 * np.arange(10),
    )
    assert sorted(np.bincount(reference["fold_index"]).tolist()) == [2] * 5
    assert reference["folds"] == 5
    assert reference["seed"] == 13
    assert reference["method"] == "5-fold cross-fitted HGB leaves=15 reference"
    assert all(
        np.isfinite(reference[name])
        for name in ("roc_auc", "log_loss", "brier_score")
    )


@pytest.mark.parametrize(
    ("outcomes", "message"),
    [
        (np.asarray([0, 1, 2, 0]), "binary"),
        (np.asarray([0, 0, 0, 1]), "each class"),
    ],
)
def test_cross_fitted_hgb_reference_rejects_invalid_labels_or_too_few_classes(
    outcomes,
    message,
):
    contexts = np.arange(outcomes.size * 2, dtype=float).reshape(-1, 2)

    with pytest.raises(RuntimeError, match=message):
        tuning._cross_fitted_hgb_reference(
            contexts,
            outcomes,
            folds=2,
            seed=0,
        )


def test_reference_policy_rows_and_prediction_artifact_use_oof_probabilities(
    tmp_path,
):
    outcomes = np.asarray([0, 1, 0, 1], dtype=np.int8)
    reference = {
        "probability": np.asarray([0.5, 0.49, 0.51, 0.8]),
        "fold_index": np.asarray([0, 0, 1, 1], dtype=np.int16),
        "folds": 2,
        "seed": 17,
        "method": "2-fold cross-fitted HGB leaves=15 reference",
        "roc_auc": 0.5,
        "log_loss": 0.7,
        "brier_score": 0.25,
    }

    [row] = tuning._reference_policy_rows(reference, outcomes, [2.0], 1.0)

    # The probability equal to alpha=0.5 routes strong by the documented tie
    # rule, giving actions [1, 0, 1, 1].
    assert row["alpha"] == 0.5
    assert row["routing_rate"] == 0.75
    assert row["accuracy"] == 0.75
    assert row["realized_total_cost"] == 5.0
    assert row["reference_probability_tie_rule"] == (
        "route strong when p_hat >= alpha"
    )
    assert row["reference_each_row_held_out"] is True

    path = tmp_path / tuning.REFERENCE_PREDICTIONS_NPZ
    tuning._write_reference_predictions_npz(
        path,
        reference,
        outcomes,
        ["a", "b", "c", "d"],
    )
    with np.load(path) as stored:
        assert stored["schema_version"].item() == 1
        assert stored["example_id"].tolist() == ["a", "b", "c", "d"]
        assert stored["fold_index"].tolist() == [0, 0, 1, 1]
        assert stored["oof_disagreement_probability"].dtype == np.float64
        assert np.array_equal(
            stored["oof_disagreement_probability"],
            reference["probability"],
        )
        assert stored["each_row_held_out"].item()
        assert json.loads(stored["hgb_profile"].item()) == (
            tuning.ONLINE_HGB_PROFILE
        )


def test_selected_learning_curves_use_fixed_reference_cost_and_allow_negative_regret(
    monkeypatch,
    tmp_path,
):
    outcomes = np.asarray([0, 1, 0, 1], dtype=np.int8)
    permutations = np.asarray(
        [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int32
    )

    def candidate(policy, method, multiplier, order_index, actions):
        actions = np.asarray(actions, dtype=bool)
        ordered_y = outcomes[permutations[order_index]].astype(float)
        increments = actions.astype(float) + (
            1.0 - actions.astype(float)
        ) * 2.0 * ordered_y
        row = {
            "policy": policy,
            "method": method,
            "l01": 2.0,
            "l11": 1.0,
            "alpha": 0.5,
            "multiplier": float(multiplier),
            "parameter_name": "test_parameter",
            "base_parameter": 1.0,
            "effective_parameter": float(multiplier),
            "order_run": order_index + 1,
            "order_seed": order_index,
            "examples": 4,
            "routing_rate": float(np.mean(actions)),
            "realized_total_cost": float(np.sum(increments)),
        }
        return tuning._attach_action_payload(row, actions)

    candidates = [
        candidate(
            tuning.POLICY_CBPSIDE,
            "CBPSide",
            0.3,
            0,
            [1, 1, 1, 1],
        ),
        candidate(
            tuning.POLICY_CBPSIDE,
            "CBPSide",
            0.3,
            1,
            [1, 1, 1, 1],
        ),
        candidate(
            tuning.POLICY_CBPSIDE,
            "CBPSide",
            3.0,
            0,
            [0, 0, 1, 1],
        ),
        candidate(
            tuning.POLICY_CBPSIDE,
            "CBPSide",
            3.0,
            1,
            [1, 0, 0, 1],
        ),
        candidate(
            tuning.POLICY_IGW_TREE,
            "SquareCB.PMSide + HGB leaves=15",
            1.0,
            0,
            [1, 0, 0, 0],
        ),
        candidate(
            tuning.POLICY_IGW_TREE,
            "SquareCB.PMSide + HGB leaves=15",
            1.0,
            1,
            [1, 1, 1, 0],
        ),
    ]
    selections = [
        {
            "policy": tuning.POLICY_CBPSIDE,
            "l01": 2.0,
            "l11": 1.0,
            "selected_multiplier": 3.0,
        },
        {
            "policy": tuning.POLICY_IGW_TREE,
            "l01": 2.0,
            "l11": 1.0,
            "selected_multiplier": 1.0,
        },
    ]

    selected = tuning._selected_order_rows(candidates, selections)
    assert len(selected) == 4
    assert {
        (row["policy"], row["selected_multiplier"])
        for row in selected
    } == {
        (tuning.POLICY_CBPSIDE, 3.0),
        (tuning.POLICY_IGW_TREE, 1.0),
    }
    assert all(
        key not in tuning._strip_internal_fields(candidates[0])
        for key in tuning.INTERNAL_ACTION_FIELDS
    )
    assert all(
        key not in row
        for row in tuning._aggregate_candidates(candidates)
        for key in tuning.INTERNAL_ACTION_FIELDS
    )

    reference_probabilities = np.asarray([0.6, 0.4, 0.5, 0.9])
    curves = tuning._build_selected_learning_curves(
        selected,
        outcomes,
        permutations,
        reference_probabilities,
    )
    cbpside_indices = np.flatnonzero(
        curves["policy"] == tuning.POLICY_CBPSIDE
    )
    random_indices = np.flatnonzero(curves["policy"] == "Random")
    assert curves["cumulative_cost"].dtype == np.float32
    assert curves["cumulative_reference_regret"].dtype == np.float32
    assert curves["average_reference_regret"].dtype == np.float32
    assert curves["cumulative_cost"].shape == (6, 4)
    assert curves["order_run"][cbpside_indices].tolist() == [1, 2]
    assert curves["selected_multiplier"][cbpside_indices].tolist() == [3.0, 3.0]
    assert curves["cumulative_cost"][cbpside_indices].tolist() == [
        [0.0, 2.0, 3.0, 4.0],
        [1.0, 1.0, 3.0, 4.0],
    ]
    # The fixed OOF reference actions in original row order are [1, 0, 1, 1].
    # They are replayed in each online permutation, with no access to y_t.
    assert curves["cumulative_reference_cost"][cbpside_indices].tolist() == [
        [1.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0],
    ]
    assert curves["cumulative_reference_regret"][cbpside_indices].tolist() == [
        [-1.0, -1.0, -1.0, -1.0],
        [-1.0, -2.0, -1.0, -1.0],
    ]
    np.testing.assert_allclose(
        curves["average_reference_regret"][cbpside_indices],
        [
            [-1.0, -0.5, -1.0 / 3.0, -0.25],
            [-1.0, -1.0, -1.0 / 3.0, -0.25],
        ],
        rtol=1e-6,
    )
    assert curves["cumulative_clairvoyant_excess_cost"][
        cbpside_indices
    ].tolist() == [
        [0.0, 1.0, 2.0, 2.0],
        [0.0, 0.0, 1.0, 2.0],
    ]

    assert curves["trajectory_kind"][random_indices].tolist() == [
        "analytic_expected",
        "analytic_expected",
    ]
    assert curves["matched_routing_rate"][random_indices].tolist() == [
        0.25,
        0.75,
    ]
    assert curves["cumulative_cost"][random_indices].tolist() == [
        [0.25, 2.0, 2.25, 4.0],
        [1.25, 2.0, 3.25, 4.0],
    ]
    assert curves["cumulative_reference_regret"][random_indices].tolist() == [
        [-0.75, -1.0, -1.75, -1.0],
        [-0.75, -1.0, -0.75, -1.0],
    ]
    np.testing.assert_allclose(
        curves["cumulative_reference_regret"][:, -1],
        curves["cumulative_cost"][:, -1]
        - curves["cumulative_reference_cost"][:, -1],
    )

    aggregated = tuning._aggregate_learning_curves(curves)
    cbpside = next(
        row for row in aggregated if row["policy"] == tuning.POLICY_CBPSIDE
    )
    assert cbpside["cumulative_cost_mean"].tolist() == [0.5, 1.5, 3.0, 4.0]
    assert cbpside["cumulative_cost_std"] == pytest.approx(
        [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
    )
    assert cbpside["cumulative_cost_sem"] == pytest.approx(
        [0.5, 0.5, 0.0, 0.0]
    )
    assert cbpside["cumulative_reference_cost_mean"].tolist() == [
        1.5,
        3.0,
        4.0,
        5.0,
    ]
    assert cbpside["cumulative_reference_regret_mean"].tolist() == [
        -1.0,
        -1.5,
        -1.0,
        -1.0,
    ]
    assert cbpside["cumulative_reference_regret_std"] == pytest.approx(
        [0.0, np.sqrt(0.5), 0.0, 0.0]
    )
    assert cbpside["cumulative_reference_regret_sem"] == pytest.approx(
        [0.0, 0.5, 0.0, 0.0]
    )
    assert cbpside["confidence_level"] == 0.95
    assert cbpside["confidence_df"] == 1
    assert cbpside["confidence_t_critical"] == pytest.approx(
        12.706204736432095
    )
    expected_half_width = (
        cbpside["confidence_t_critical"]
        * cbpside["cumulative_reference_regret_sem"]
    )
    assert cbpside["cumulative_reference_regret_ci95_lower"] == pytest.approx(
        cbpside["cumulative_reference_regret_mean"] - expected_half_width
    )
    assert cbpside["cumulative_reference_regret_ci95_upper"] == pytest.approx(
        cbpside["cumulative_reference_regret_mean"] + expected_half_width
    )
    assert cbpside["average_reference_regret_mean"] == pytest.approx(
        [-1.0, -0.75, -1.0 / 3.0, -0.25]
    )

    npz_path = tmp_path / tuning.LEARNING_CURVE_NPZ
    csv_path = tmp_path / tuning.LEARNING_CURVE_CSV
    tuning._write_learning_curves_npz(npz_path, curves)
    tuning._write_learning_curve_aggregate_csv(csv_path, aggregated)
    plot_groups = [
        *aggregated,
        *(dict(row, l01=3.0) for row in aggregated),
    ]
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    plotted_values = []
    filled_bounds = []
    zero_guides = []
    x_labels = []
    y_labels = []
    titles = []
    figure_texts = []
    original_plot = Axes.plot
    original_fill_between = Axes.fill_between
    original_axhline = Axes.axhline
    original_set_xlabel = Axes.set_xlabel
    original_set_ylabel = Axes.set_ylabel
    original_set_title = Axes.set_title
    original_figure_text = Figure.text

    def recording_plot(axis, x_values, y_values, *args, **kwargs):
        plotted_values.append(np.asarray(y_values).copy())
        return original_plot(axis, x_values, y_values, *args, **kwargs)

    def recording_fill_between(
        axis, x_values, lower_values, upper_values, *args, **kwargs
    ):
        filled_bounds.append(
            (
                np.asarray(lower_values).copy(),
                np.asarray(upper_values).copy(),
            )
        )
        return original_fill_between(
            axis, x_values, lower_values, upper_values, *args, **kwargs
        )

    def recording_axhline(axis, y=0, *args, **kwargs):
        zero_guides.append(float(y))
        return original_axhline(axis, y, *args, **kwargs)

    def recording_set_xlabel(axis, label, *args, **kwargs):
        x_labels.append(label)
        return original_set_xlabel(axis, label, *args, **kwargs)

    def recording_set_ylabel(axis, label, *args, **kwargs):
        y_labels.append(label)
        return original_set_ylabel(axis, label, *args, **kwargs)

    def recording_set_title(axis, label, *args, **kwargs):
        titles.append(label)
        return original_set_title(axis, label, *args, **kwargs)

    def recording_figure_text(figure, *args, **kwargs):
        figure_texts.append((args, kwargs))
        return original_figure_text(figure, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", recording_plot)
    monkeypatch.setattr(Axes, "fill_between", recording_fill_between)
    monkeypatch.setattr(Axes, "axhline", recording_axhline)
    monkeypatch.setattr(Axes, "set_xlabel", recording_set_xlabel)
    monkeypatch.setattr(Axes, "set_ylabel", recording_set_ylabel)
    monkeypatch.setattr(Axes, "set_title", recording_set_title)
    monkeypatch.setattr(Figure, "text", recording_figure_text)
    plot_paths = tuning._plot_selected_cumulative_reference_regret(
        tmp_path, plot_groups
    )
    assert len(plotted_values) == len(plot_groups)
    for plotted, group in zip(plotted_values, plot_groups):
        assert np.array_equal(
            plotted, group["cumulative_reference_regret_mean"]
        )
    for (lower, upper), group in zip(filled_bounds, plot_groups):
        assert np.array_equal(
            lower, group["cumulative_reference_regret_ci95_lower"]
        )
        assert np.array_equal(
            upper, group["cumulative_reference_regret_ci95_upper"]
        )
    assert zero_guides == [0.0, 0.0]
    assert x_labels == [plot_style.AXIS_LABELS["round"]] * 2
    assert y_labels == [
        plot_style.AXIS_LABELS["cumulative_reference_regret"]
    ] * 2
    plotted_values.clear()
    filled_bounds.clear()
    zero_guides.clear()
    x_labels.clear()
    y_labels.clear()
    average_plot_paths = tuning._plot_selected_average_reference_regret(
        tmp_path, plot_groups
    )
    assert len(plotted_values) == len(plot_groups)
    for plotted, group in zip(plotted_values, plot_groups):
        assert np.array_equal(plotted, group["average_reference_regret_mean"])
    for (lower, upper), group in zip(filled_bounds, plot_groups):
        assert np.array_equal(
            lower, group["average_reference_regret_ci95_lower"]
        )
        assert np.array_equal(
            upper, group["average_reference_regret_ci95_upper"]
        )
    assert zero_guides == [0.0, 0.0]
    assert x_labels == [plot_style.AXIS_LABELS["round"]] * 2
    assert y_labels == [plot_style.AXIS_LABELS["average_reference_regret"]] * 2
    assert titles == []
    assert figure_texts == []
    assert npz_path.stat().st_size > 0
    assert csv_path.stat().st_size > 0
    assert [path.name for path in plot_paths] == [
        "selected_cumulative_reference_regret_l01-2.png",
        "selected_cumulative_reference_regret_l01-3.png",
    ]
    assert [path.name for path in average_plot_paths] == [
        "selected_average_reference_regret_l01-2.png",
        "selected_average_reference_regret_l01-3.png",
    ]
    assert all(
        path.stat().st_size > 0 for path in [*plot_paths, *average_plot_paths]
    )
    assert all(
        path.with_suffix(".pdf").stat().st_size > 0
        for path in [*plot_paths, *average_plot_paths]
    )
    with np.load(npz_path) as stored:
        assert stored["schema_version"].item() == 2
        assert stored["cumulative_cost"].dtype == np.float32
        assert stored["cumulative_reference_regret"].shape == (6, 4)
        assert stored["average_reference_regret"].shape == (6, 4)
        assert "cross-fitted HGB-15 reference action" in stored[
            "regret_increment_definition"
        ].item()
    with csv_path.open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(aggregated) * 4
    assert {
        "cumulative_cost_mean",
        "cumulative_cost_std",
        "cumulative_cost_sem",
        "cumulative_reference_cost_mean",
        "cumulative_reference_regret_mean",
        "cumulative_reference_regret_std",
        "cumulative_reference_regret_sem",
        "cumulative_reference_regret_ci95_lower",
        "cumulative_reference_regret_ci95_upper",
        "average_reference_regret_mean",
        "average_reference_regret_std",
        "average_reference_regret_sem",
        "average_reference_regret_ci95_lower",
        "average_reference_regret_ci95_upper",
        "confidence_level",
        "confidence_df",
        "confidence_t_critical",
        "cumulative_clairvoyant_excess_cost_mean",
    }.issubset(csv_rows[0])


def test_reference_regret_plot_removes_only_superseded_learning_plots(tmp_path):
    legacy_plots = [
        tmp_path / "selected_cumulative_cost_l01-2.png",
        tmp_path / "selected_cumulative_cost_l01-3.png",
        tmp_path / "selected_cumulative_regret_l01-2.png",
        tmp_path / "selected_cumulative_regret_l01-3.png",
        tmp_path / "selected_cumulative_cost_l01-2.pdf",
        tmp_path / "selected_cumulative_cost_l01-3.pdf",
        tmp_path / "selected_cumulative_regret_l01-2.pdf",
        tmp_path / "selected_cumulative_regret_l01-3.pdf",
    ]
    unrelated_plot = tmp_path / "selected_cost_vs_l01.png"
    for path in [*legacy_plots, unrelated_plot]:
        path.write_bytes(b"existing plot")

    [regret_plot] = tuning._plot_selected_cumulative_reference_regret(
        tmp_path,
        [
            {
                "policy": tuning.POLICY_CBPSIDE,
                "method": "CBPSide (linear logistic)",
                "l01": 2.0,
                "round": np.asarray([1, 2]),
                "cumulative_reference_regret_mean": np.asarray([-0.2, 0.5]),
                "cumulative_reference_regret_std": np.asarray([0.0, 0.1]),
                "cumulative_reference_regret_ci95_lower": np.asarray(
                    [-0.2, 0.3]
                ),
                "cumulative_reference_regret_ci95_upper": np.asarray(
                    [-0.2, 0.7]
                ),
            }
        ],
    )

    assert not any(path.exists() for path in legacy_plots)
    assert unrelated_plot.read_bytes() == b"existing plot"
    assert regret_plot.name == (
        "selected_cumulative_reference_regret_l01-2.png"
    )
    assert regret_plot.stat().st_size > 0
    assert regret_plot.with_suffix(".pdf").stat().st_size > 0


def test_publication_plots_use_student_t_intervals_and_central_labels(
    monkeypatch, tmp_path
):
    class FakeLine:
        @staticmethod
        def get_color():
            return "tab:blue"

    class FakeAxis:
        def __init__(self):
            self.errorbars = []
            self.plots = []
            self.x_label = None
            self.y_label = None

        def errorbar(self, *args, **kwargs):
            self.errorbars.append((args, kwargs))

        def plot(self, *args, **kwargs):
            self.plots.append((args, kwargs))
            return (FakeLine(),)

        def fill_between(self, *args, **kwargs):
            del args, kwargs

        def set_xlabel(self, label):
            self.x_label = label

        def set_ylabel(self, label):
            self.y_label = label

        def set_title(self, *args, **kwargs):
            raise AssertionError("publication plots must not have titles")

        def set_xlim(self, *args, **kwargs):
            del args, kwargs

        def set_ylim(self, *args, **kwargs):
            del args, kwargs

        def set_xticks(self, *args, **kwargs):
            del args, kwargs

        def set_yscale(self, *args, **kwargs):
            del args, kwargs

        def axhline(self, *args, **kwargs):
            del args, kwargs

        def grid(self, *args, **kwargs):
            del args, kwargs

        def legend(self, *args, **kwargs):
            del args, kwargs

    class FakeFigure:
        def text(self, *args, **kwargs):
            raise AssertionError("publication plots must not have footnotes")

    class FakePyplot:
        def __init__(self):
            self.axes = []
            self.figsize_calls = []

        def subplots(self, *, figsize, constrained_layout):
            assert constrained_layout is True
            self.figsize_calls.append(figsize)
            axis = FakeAxis()
            self.axes.append(axis)
            return FakeFigure(), axis

        @staticmethod
        def close(figure):
            del figure

    pyplot = FakePyplot()
    saved_paths = []
    interval_calls = []

    def fake_half_width(standard_deviation, sample_count):
        interval_calls.append((float(standard_deviation), int(sample_count)))
        return 1000.0 + float(standard_deviation)

    def fake_save(figure, path):
        del figure
        saved_paths.append(path)
        return path, path.with_suffix(".pdf")

    monkeypatch.setattr(tuning, "publication_pyplot", lambda: pyplot)
    monkeypatch.setattr(tuning, "student_t_half_width", fake_half_width)
    monkeypatch.setattr(tuning, "save_publication_figure", fake_save)

    selected_rows = [
        {
            "policy": tuning.POLICY_CBPSIDE,
            "method": "CBPSide (linear logistic)",
            "l01": 2.0,
            "l11": 1.0,
            "examples": 100,
            "online_order_repeats": 20,
            "routing_rate_mean": 0.3,
            "routing_rate_std": 0.1,
            "accuracy_mean": 0.8,
            "accuracy_std": 0.2,
            "realized_total_cost_mean": 50.0,
            "realized_total_cost_std": 4.0,
        }
    ]
    comparison_rows = [
        {
            "l01": 2.0,
            "online_order_repeats": 20,
            "tree_gamma_multiplier": 1.0,
            "nonlinear_tree_cost_reduction_mean": 2.0,
            "nonlinear_tree_cost_reduction_std": 3.0,
        }
    ]
    selections = [
        {
            "policy": policy,
            "l01": 2.0,
            "selected_multiplier": 1.0,
        }
        for policy in tuning.TUNED_POLICIES
    ]

    tuning._plot_selected_routing_accuracy(
        tmp_path / "routing.png", selected_rows
    )
    tuning._plot_selected_cost(tmp_path / "cost.png", selected_rows)
    tuning._plot_selected_multipliers(
        tmp_path / "multipliers.png", selections
    )
    tuning._plot_igw_estimator_comparison(
        tmp_path / "comparison.png", comparison_rows
    )
    tuning._plot_igw_matched_estimator_comparison(
        tmp_path / "matched.png", comparison_rows
    )

    assert interval_calls == [
        (0.1, 20),
        (0.2, 20),
        (4.0, 20),
        (3.0, 20),
        (3.0, 20),
    ]
    assert pyplot.axes[0].errorbars[0][1]["xerr"] == [1000.1]
    assert pyplot.axes[0].errorbars[0][1]["yerr"] == [1000.2]
    assert pyplot.axes[1].errorbars[0][1]["yerr"] == [1004.0]
    assert pyplot.axes[3].errorbars[0][1]["yerr"] == [1003.0]
    assert pyplot.axes[4].errorbars[0][1]["yerr"] == [1003.0]
    assert [call[1]["label"] for call in pyplot.axes[2].plots] == [
        "CBPSide",
        "SquareCB.PMSide Linear",
        "SquareCB.PMSide Tree",
        "PG-TS (Bayesian logistic)",
    ]
    assert [(axis.x_label, axis.y_label) for axis in pyplot.axes] == [
        (
            plot_style.AXIS_LABELS["routing_rate"],
            plot_style.AXIS_LABELS["accuracy"],
        ),
        (r"$\ell_{01}$", "Total cost"),
        (
            plot_style.AXIS_LABELS["l01"],
            plot_style.AXIS_LABELS["selected_multiplier"],
        ),
        (
            plot_style.AXIS_LABELS["l01"],
            plot_style.AXIS_LABELS["squarecb_cost_difference"],
        ),
        (
            plot_style.AXIS_LABELS["l01"],
            plot_style.AXIS_LABELS["squarecb_cost_difference"],
        ),
    ]
    assert pyplot.figsize_calls == [
        plot_style.PUBLICATION_FIGSIZE,
        plot_style.PUBLICATION_FIGSIZE,
        plot_style.PUBLICATION_FIGSIZE_SHORT,
        plot_style.PUBLICATION_FIGSIZE_SHORT,
        plot_style.PUBLICATION_FIGSIZE,
    ]
    assert saved_paths == [
        tmp_path / "routing.png",
        tmp_path / "cost.png",
        tmp_path / "multipliers.png",
        tmp_path / "comparison.png",
        tmp_path / "matched.png",
    ]


def test_learning_curve_ci95_uses_student_t_sem_for_twenty_orders():
    repeats = 20
    cumulative_regret = np.column_stack(
        (
            np.arange(repeats, dtype=np.float64),
            2.0 * np.arange(repeats, dtype=np.float64),
        )
    )
    average_regret = cumulative_regret / np.asarray([1.0, 2.0])
    curves = {
        "policy": np.asarray([tuning.POLICY_CBPSIDE] * repeats),
        "method": np.asarray(["CBPSide (linear logistic)"] * repeats),
        "trajectory_kind": np.asarray(["realized"] * repeats),
        "l01": np.full(repeats, 2.0),
        "l11": np.ones(repeats),
        "effective_parameter": np.ones(repeats),
        "matched_routing_rate": np.full(repeats, np.nan),
        "selected_multiplier": np.ones(repeats),
        "round": np.asarray([1, 2]),
        "cumulative_cost": cumulative_regret + 10.0,
        "cumulative_reference_cost": np.full((repeats, 2), 10.0),
        "cumulative_reference_regret": cumulative_regret,
        "average_reference_regret": average_regret,
        "cumulative_clairvoyant_excess_cost": cumulative_regret + 5.0,
    }

    [group] = tuning._aggregate_learning_curves(curves)

    t_critical_95_df19 = 2.093024054408263
    expected_mean = np.mean(cumulative_regret, axis=0)
    expected_sem = np.std(cumulative_regret, axis=0, ddof=1) / np.sqrt(
        repeats
    )
    expected_half_width = t_critical_95_df19 * expected_sem
    assert group["online_order_repeats"] == repeats
    assert group["confidence_level"] == 0.95
    assert group["confidence_df"] == 19
    assert group["confidence_t_critical"] == pytest.approx(
        t_critical_95_df19
    )
    assert group["cumulative_reference_regret_ci95_lower"] == pytest.approx(
        expected_mean - expected_half_width
    )
    assert group["cumulative_reference_regret_ci95_upper"] == pytest.approx(
        expected_mean + expected_half_width
    )
    observed_half_width = (
        group["cumulative_reference_regret_ci95_upper"]
        - group["cumulative_reference_regret_mean"]
    )
    assert observed_half_width == pytest.approx(expected_half_width)
    assert not np.allclose(observed_half_width, 0.5 * expected_half_width)

    average_mean = np.mean(average_regret, axis=0)
    average_sem = np.std(average_regret, axis=0, ddof=1) / np.sqrt(repeats)
    average_half_width = t_critical_95_df19 * average_sem
    assert group["average_reference_regret_ci95_lower"] == pytest.approx(
        average_mean - average_half_width
    )
    assert group["average_reference_regret_ci95_upper"] == pytest.approx(
        average_mean + average_half_width
    )


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
                    "method": "SquareCB.PMSide + HGB leaves=15",
                    "realized_total_cost": tree_cost,
                },
                {
                    **common,
                    "policy": tuning.POLICY_IGW_LINEAR,
                    "method": "SquareCB.PMSide + linear logistic",
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
        json.dumps({"config_fingerprint": "old", "policy": "CBPSide"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="different configuration"):
        tuning._load_checkpoint(path, "new")


def test_pgts_checkpoint_path_includes_prior_std_multiplier(tmp_path):
    path = tuning._checkpoint_path(
        tmp_path,
        tuning.POLICY_PGTS,
        2.0,
        0.3,
        1,
    )

    assert path.relative_to(tmp_path).parts == (
        "checkpoints",
        "pgts",
        "l01-2",
        "multiplier-0p3",
        "order-02.json",
    )
    assert "fixed" not in path.parts


def test_prepare_output_rejects_a_revision7_output_instead_of_refreshing_it(
    tmp_path,
):
    output = tmp_path / "results"
    args = tuning._parser().parse_args(
        [
            "--cache",
            str(tmp_path / "cache.zip"),
            "--output-dir",
            str(output),
        ]
    )
    manifest, fingerprint = tuning._manifest_and_fingerprint(
        args,
        contexts=np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
        ),
        outcomes=np.asarray([0, 1, 0, 1], dtype=np.int8),
        example_ids=["example-0", "example-1", "example-2", "example-3"],
        context_summary={"profile": "all-features"},
    )
    assert manifest["implementation_revision"] == 9

    existing = json.loads(json.dumps(manifest))
    existing["implementation_revision"] = 7
    existing["design"] = "pointwise-online-parameter-multiplier-sweep-v7"
    existing["learning_curves"]["plot_pattern"] = (
        f"{tuning.LEGACY_LEARNING_CURVE_PLOT_PREFIX}<float_slug>.png"
    )
    existing["learning_curves"]["plot_metric"] = "cumulative_cost"
    existing["config_fingerprint"] = "revision-7-fingerprint"
    output.mkdir()
    manifest_path = output / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(existing), encoding="utf-8")

    with pytest.raises(SystemExit, match="different sweep configuration"):
        tuning._prepare_output(output, manifest, fingerprint)

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == existing


def test_prepare_output_refreshes_only_presentation_metadata(tmp_path):
    output = tmp_path / "results"
    args = tuning._parser().parse_args(
        [
            "--cache",
            str(tmp_path / "cache.zip"),
            "--output-dir",
            str(output),
        ]
    )
    manifest, fingerprint = tuning._manifest_and_fingerprint(
        args,
        contexts=np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
        ),
        outcomes=np.asarray([0, 1, 0, 1], dtype=np.int8),
        example_ids=["example-0", "example-1", "example-2", "example-3"],
        context_summary={"profile": "all-features"},
    )
    prior_manifest = json.loads(json.dumps(manifest))
    prior_manifest["learning_curves"].pop("plotted_uncertainty")
    prior_manifest.pop("plot_presentation")
    output.mkdir()
    manifest_path = output / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(prior_manifest), encoding="utf-8")

    tuning._prepare_output(output, manifest, fingerprint)

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_parser_uses_five_multipliers_pure_doubling_and_hgb_defaults():
    base = ["--cache", "cache.zip", "--output-dir", "results"]
    args = tuning._parser().parse_args(base)
    canonical_squarecb_args = tuning._parser().parse_args(
        base
        + [
            "--squarecb-pmside-base-gamma",
            "17",
            "--squarecb-pmside-mu",
            "3",
            "--squarecb-pmside-min-propensity",
            "0.2",
        ]
    )
    legacy_igw_args = tuning._parser().parse_args(
        base
        + [
            "--igw-base-gamma",
            "17",
            "--igw-mu",
            "3",
            "--igw-min-propensity",
            "0.2",
        ]
    )
    canonical_pgts_args = tuning._parser().parse_args(
        base + ["--pgts-base-prior-std", "2"]
    )
    legacy_pgts_args = tuning._parser().parse_args(
        base + ["--pgts-prior-std", "2"]
    )
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
    assert args.reference_folds == 5
    assert args.adaptive_update_schedule == "doubling"
    assert args.adaptive_max_round_gap == 32
    assert fibonacci_args.adaptive_update_schedule == "fibonacci"
    assert doubling_args.adaptive_update_schedule == "doubling"
    assert tuning._tree_settings(doubling_args)["kind"] == "river-hoeffding"
    assert args.online_order_repeats == 20
    assert args.multipliers == [0.1, 0.3, 1.0, 3.0, 10.0]
    assert args.include_pgts is False
    assert args.pgts_gibbs_steps == 15
    assert args.pgts_prior_std == 1.0
    assert canonical_pgts_args.pgts_prior_std == 2.0
    assert legacy_pgts_args.pgts_prior_std == 2.0
    assert tuning.TUNED_POLICIES == (
        "CBPSide",
        "SquareCB.PMSideLinear",
        "SquareCB.PMSide",
        "PGTS",
    )
    assert tuning.FIXED_POLICIES == ()
    assert (
        canonical_squarecb_args.igw_base_gamma,
        canonical_squarecb_args.igw_mu,
        canonical_squarecb_args.igw_min_propensity,
    ) == (17.0, 3.0, 0.2)
    assert (
        legacy_igw_args.igw_base_gamma,
        legacy_igw_args.igw_mu,
        legacy_igw_args.igw_min_propensity,
    ) == (
        canonical_squarecb_args.igw_base_gamma,
        canonical_squarecb_args.igw_mu,
        canonical_squarecb_args.igw_min_propensity,
    )
    assert tuning._method_label(tuning.POLICY_CBPSIDE, {}) == (
        "CBPSide (linear logistic)"
    )
    assert tuning._method_label(tuning.POLICY_IGW_LINEAR, {}) == (
        "SquareCB.PMSide + linear logistic"
    )
    assert tuning._method_label(
        tuning.POLICY_IGW_TREE, tuning._tree_settings(args)
    ) == "SquareCB.PMSide + HGB leaves=15"
    assert tuning._method_label(tuning.POLICY_PGTS, {}) == (
        "PG-TS (Bayesian logistic)"
    )
    assert tuning._expected_checkpoint_count(args) == 2700
    pgts_args = tuning._parser().parse_args(base + ["--include-pgts"])
    assert tuning._active_tuned_policies(args) == tuning.BASE_TUNED_POLICIES
    assert tuning._active_tuned_policies(pgts_args) == tuning.TUNED_POLICIES
    assert tuning._expected_checkpoint_count(pgts_args) == 3600


def test_manifest_fingerprint_covers_contexts_and_cbpside_regularization(
    monkeypatch, tmp_path
):
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
        base
        + [
            "--adaptive-update-schedule",
            "capped-doubling",
            "--adaptive-max-round-gap",
            "3",
        ]
    )
    short_gap_manifest, short_gap_fingerprint = (
        tuning._manifest_and_fingerprint(short_gap_args, **common)
    )
    pgts_args = tuning._parser().parse_args(
        base
        + [
            "--include-pgts",
            "--pgts-gibbs-steps",
            "7",
            "--pgts-base-prior-std",
            "2",
        ]
    )
    pgts_manifest, pgts_fingerprint = tuning._manifest_and_fingerprint(
        pgts_args, **common
    )
    reference_args = tuning._parser().parse_args(
        base + ["--reference-folds", "2"]
    )
    reference_manifest, reference_fingerprint = (
        tuning._manifest_and_fingerprint(reference_args, **common)
    )

    assert manifest["design"] == "pointwise-online-parameter-multiplier-sweep-v9"
    assert manifest["implementation_revision"] == 9
    assert manifest["tuned_policies"] == list(tuning.TUNED_POLICIES[:-1])
    assert "pgts" not in manifest
    assert manifest["disabled_policies"] == ["ETC", "ETCLinear"]
    assert manifest["fixed_policies"] == []
    assert manifest["candidate_policies"] == list(tuning.TUNED_POLICIES[:-1])
    assert manifest["candidate_counts"] == {
        "multiplier_tuned_checkpoints": 2700,
        "pgts_prior_tuned_checkpoints": 0,
        "total_checkpoints": 2700,
        "multiplier_tuned_execution_groups": 135,
        "pgts_prior_tuned_execution_groups": 0,
        "total_execution_groups": 135,
    }
    assert manifest["update_schedule"] == {
        "name": "doubling",
        "maximum_round_gap": None,
        "boundary_rule": "before global rounds 1,2,4,8,...",
        "boundary_rounds": [1, 2, 4, 8],
        "boundary_count": 4,
        "maximum_model_updates": 3,
        "last_boundary_round": 8,
        "final_frozen_epoch_rounds": 3,
        "history_cutoff": "feedback through boundary_round-1",
        "cbpside_theta": "refit only at boundary when new tastes exist",
        "cbpside_V_inverse": "recompute only at boundary and freeze in epoch",
        "cbpside_beta": "evaluate on every current context using epoch V inverse",
        "squarecb_pmside_tree": "full revealed-history refit only at boundary",
        "squarecb_pmside_linear": (
            "full revealed-history refit only at boundary"
        ),
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
    assert "etc_hgb_profile" not in manifest
    assert "etc_linear_profile" not in manifest
    assert "etc_estimator_comparison" not in manifest
    assert manifest["squarecb_pmside_linear_profile"]["kind"] == "logistic"
    assert (
        "fixed_multiplier_contrast"
        in manifest["squarecb_pmside_estimator_comparison"]
    )
    assert (
        "selected_contrast"
        in manifest["squarecb_pmside_estimator_comparison"]
    )
    assert manifest["base_parameters"]["squarecb_pmside_gamma"] == pytest.approx(
        np.sqrt(10)
    )
    assert manifest["base_parameters"]["squarecb_pmside_gamma_rule"] == "sqrt(n)"
    assert "igw_gamma" not in manifest["base_parameters"]
    assert "igw_gamma_rule" not in manifest["base_parameters"]
    assert manifest["squarecb_pmside_mu"] == 2.0
    assert manifest["squarecb_pmside_min_propensity"] == 0.1
    assert "igw_linear_profile" not in manifest
    assert "igw_estimator_comparison" not in manifest
    assert "IGW" not in json.dumps(manifest, sort_keys=True)
    assert '"igw_' not in json.dumps(manifest, sort_keys=True)
    assert manifest["base_parameters"]["cbpside_matrix_regularization"] == 1.0
    assert manifest["base_parameters"]["cbpside_theta_regularization"] == 1.0
    assert manifest["regret_reference"] == {
        "type": "fixed_row_aligned_cross_fitted_probability_reference",
        "method": "5-fold stratified out-of-fold HGB leaves=15",
        "folds": 5,
        "splitter": "StratifiedKFold(shuffle=True)",
        "split_seed": 0,
        "model_seed": 0,
        "hgb_profile": tuning.ONLINE_HGB_PROFILE,
        "context_profile": "all-features",
        "context_dimension": 2,
        "outcome_source": "cached_weak_strong_disagreement",
        "benchmark_gold_answers_used": False,
        "each_row_predicted_by_model_excluding_that_row": True,
        "action_rule": "route strong when p_hat >= 1/l01",
        "shared_across_online_policies_and_permutations": True,
        "interpretation": (
            "offline empirical reference assembled from fold models; "
            "not an outcome-aware oracle and not by itself a theorem test"
        ),
        "prediction_artifact": tuning.REFERENCE_PREDICTIONS_NPZ,
        "threshold_summary_csv": tuning.REFERENCE_RESULTS_CSV,
        "threshold_summary_json": tuning.REFERENCE_RESULTS_JSON,
    }
    assert reference_manifest["regret_reference"]["folds"] == 2
    assert manifest["learning_curves"] == {
        "enabled": True,
        "selection_scope": "pointwise selected policy per l01",
        "checkpoint_action_encoding": (
            "np.packbits uint8 with bitorder=little, then base64"
        ),
        "checkpoint_action_payload_internal": True,
        "by_order_artifact": tuning.LEARNING_CURVE_NPZ,
        "by_order_curve_dtype": "float32",
        "aggregate_artifact": tuning.LEARNING_CURVE_CSV,
        "aggregate_curve_dtype": "float64",
        "cumulative_regret_plot_pattern": (
            f"{tuning.LEARNING_CURVE_PLOT_PREFIX}<float_slug>.png"
        ),
        "average_regret_plot_pattern": (
            f"{tuning.AVERAGE_REGRET_PLOT_PREFIX}<float_slug>.png"
        ),
        "primary_regret_metric": "cumulative_reference_regret",
        "normalized_diagnostic": "average_reference_regret=R_t/t",
        "plotted_uncertainty": {
            "type": "pointwise_student_t_confidence_interval_for_mean",
            "confidence_level": 0.95,
            "degrees_of_freedom": "online_order_repeats - 1",
            "formula": (
                "mean +/- t.ppf((1 + confidence_level) / 2, df) * "
                "sample_sd / sqrt(n)"
            ),
            "simultaneous_band": False,
            "post_selection_adjusted": False,
        },
        "cost_increment_definition": tuning.REALIZED_COST_INCREMENT_DEFINITION,
        "clairvoyant_cost_increment_definition": (
            tuning.CLAIRVOYANT_COST_INCREMENT_DEFINITION
        ),
        "regret_increment_definition": tuning.REGRET_INCREMENT_DEFINITION,
        "clairvoyant_excess_increment_definition": (
            tuning.CLAIRVOYANT_EXCESS_INCREMENT_DEFINITION
        ),
        "outcome_aware_diagnostic_retained_but_not_plotted": True,
        "random_increment_definition": tuning.RANDOM_COST_INCREMENT_DEFINITION,
        "selection_warning": tuning.LEARNING_CURVE_SELECTION_WARNING,
        "json_artifact": None,
    }
    assert manifest["plot_presentation"] == {
        "style_module": "llm_routing_simulation.plot_style",
        "formats": ["png", "pdf"],
        "png_dpi": 400,
        "figure_titles": False,
        "figure_footnotes": False,
        "axis_labels": plot_style.AXIS_LABELS,
        "method_labels": plot_style.METHOD_LABELS,
        "replicated_run_uncertainty": {
            "type": "student_t_confidence_interval_for_mean",
            "confidence_level": 0.95,
            "degrees_of_freedom": "online_order_repeats - 1",
            "simultaneous_band": False,
            "post_selection_adjusted": False,
        },
    }
    assert pgts_manifest["design"] == (
        "pointwise-online-parameter-multiplier-sweep-v9"
    )
    assert pgts_manifest["implementation_revision"] == 9
    assert pgts_manifest["tuned_policies"] == list(tuning.TUNED_POLICIES)
    assert pgts_manifest["fixed_policies"] == []
    assert pgts_manifest["pgts"] == {
        "enabled": True,
        "policy": tuning.POLICY_PGTS,
        "algorithm": "PG-TS Algorithm 1 scheduled-update approximation",
        "algorithm1_exact": False,
        "gibbs_steps_per_posterior_draw": 7,
        "prior_mean": 0.0,
        "base_prior_std": 2.0,
        "prior_std_multiplier_grid": [0.1, 0.3, 1.0, 3.0, 10.0],
        "effective_prior_std_values": [0.2, 0.6, 2.0, 6.0, 20.0],
        "effective_prior_std_rule": "base_prior_std * selected_multiplier",
        "prior_covariance": "effective_prior_std^2 * identity",
        "context_preprocessing": (
            "row L2 normalization using max(1, norm), then prepend intercept"
        ),
        "posterior_draw": (
            "final draw after M Gibbs transitions at an eligible configured "
            "boundary"
        ),
        "update_schedule": "global-round doubling boundaries",
        "update_rule": (
            "initial prior draw at round 1; later boundary draws only when "
            "new action-1 feedback arrived since the previous draw; reuse "
            "theta within each epoch"
        ),
        "eligible_boundary_rounds": [1, 2, 4, 8],
        "eligible_boundary_count": 4,
        "feedback": "only action-1 disagreement outcomes are revealed",
        "inverse_propensity_weighting": False,
        "multiplier_tuned": True,
        "selection_objective": (
            "lowest mean realized total cost pointwise by l01"
        ),
        "optional_dependency": "polyagamma",
    }
    assert pgts_manifest["update_schedule"]["pgts"] == (
        "initial prior draw at round 1; thereafter draw only at configured "
        "boundaries with new action-1 feedback, and freeze theta within epoch"
    )
    assert pgts_manifest["candidate_counts"] == {
        "multiplier_tuned_checkpoints": 3600,
        "pgts_prior_tuned_checkpoints": 900,
        "total_checkpoints": 3600,
        "multiplier_tuned_execution_groups": 180,
        "pgts_prior_tuned_execution_groups": 45,
        "total_execution_groups": 180,
    }
    assert "fixed_policies_excluded" not in pgts_manifest["selection"]
    monkeypatch.setattr(tuning, "PLOT_CONFIDENCE_LEVEL", 0.9)
    presentation_manifest, presentation_fingerprint = (
        tuning._manifest_and_fingerprint(default_args, **common)
    )
    assert presentation_fingerprint == default_fingerprint
    assert presentation_manifest["learning_curves"]["plotted_uncertainty"][
        "confidence_level"
    ] == 0.9
    assert presentation_manifest["plot_presentation"][
        "replicated_run_uncertainty"
    ]["confidence_level"] == 0.9
    assert default_fingerprint == doubling_fingerprint
    assert len(
        {
            default_fingerprint,
            matrix_fingerprint,
            theta_fingerprint,
            context_fingerprint,
            fibonacci_fingerprint,
            doubling_fingerprint,
            short_gap_fingerprint,
            pgts_fingerprint,
            reference_fingerprint,
        }
    ) == 8


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


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--pgts-gibbs-steps", "0", "pgts-gibbs-steps must be positive"),
        ("--pgts-prior-std", "0", "pgts-prior-std must be positive"),
    ],
)
def test_tuner_rejects_invalid_pgts_settings(option, value, message):
    args = tuning._parser().parse_args(
        [
            "--cache",
            "cache.zip",
            "--output-dir",
            "results",
            option,
            value,
        ]
    )

    with pytest.raises(SystemExit, match=message):
        tuning._validate_args(args)


def test_pgts_dependency_preflight_happens_before_cache_loading(
    monkeypatch, tmp_path
):
    def unavailable():
        raise SystemExit("missing optional PG-TS dependency")

    def unexpected_cache_load(path):
        del path
        raise AssertionError("cache loading must follow dependency preflight")

    monkeypatch.setattr(tuning, "_require_pgts_dependency", unavailable)
    monkeypatch.setattr(tuning, "load_cache", unexpected_cache_load)

    with pytest.raises(SystemExit, match="missing optional PG-TS dependency"):
        tuning.main(
            [
                "--cache",
                str(tmp_path / "cache.zip"),
                "--output-dir",
                str(tmp_path / "results"),
                "--include-pgts",
            ]
        )


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
    pgts_calls = []

    def fake_pgts_candidate(
        normalized_features,
        outcomes,
        permutation,
        *,
        l01,
        l11,
        gibbs_steps,
        prior_std,
        multiplier,
        order_index,
        order_seed,
        policy_seed,
        update_schedule,
        update_max_round_gap,
    ):
        assert update_schedule == "doubling"
        assert update_max_round_gap == 32
        pgts_calls.append(
            (float(l01), float(multiplier), int(order_index), update_schedule)
        )
        y_all = outcomes[permutation]
        n = len(y_all)
        actions = np.arange(n) % 2 == 0
        row = tuning._base_row(
            policy=tuning.POLICY_PGTS,
            l01=l01,
            l11=l11,
            multiplier=multiplier,
            parameter_name="prior_std",
            base_parameter=prior_std,
            effective_parameter=prior_std * multiplier,
            order_index=order_index,
            order_seed=order_seed,
            policy_seed=policy_seed,
            examples=n,
            tree_settings={},
            update_schedule=(
                "global_round_doubling_posterior_draw_before_action"
            ),
        )
        row.update(
            {
                "pgts_gibbs_steps": int(gibbs_steps),
                "pgts_base_prior_std": float(prior_std),
                "pgts_prior_std_multiplier": float(multiplier),
                "pgts_prior_std": float(prior_std * multiplier),
                "normalized_dimension": int(normalized_features.shape[1]),
                "pgts_algorithm1_exact": False,
                "posterior_eligible_boundary_rounds": [1, 2, 4, 8],
                "posterior_eligible_boundary_count": 4,
                "posterior_draw_rounds": [1, 2, 4, 8],
                "posterior_boundary_draw_count": 4,
                "posterior_skipped_clean_boundary_count": 0,
                "total_gibbs_transitions": 4 * int(gibbs_steps),
                "last_posterior_training_count": 4,
                "final_revealed_count": 4,
            }
        )
        return tuning._attach_action_payload(
            tuning._finish_row(
                row,
                routed=int(np.count_nonzero(actions)),
                correct=int(np.count_nonzero(actions | (y_all == 0))),
                model_updates=4,
                last_training_count=4,
            ),
            actions,
        )

    monkeypatch.setattr(tuning, "_require_pgts_dependency", lambda: None)
    monkeypatch.setattr(tuning, "_simulate_pgts_candidate", fake_pgts_candidate)
    reference_calls = []

    def fake_cross_fitted_reference(contexts, outcomes, *, folds, seed):
        reference_calls.append((folds, seed))
        return {
            "probability": np.linspace(0.2, 0.8, len(outcomes)),
            "fold_index": np.arange(len(outcomes), dtype=np.int16) % folds,
            "folds": folds,
            "seed": seed,
            "method": f"{folds}-fold cross-fitted HGB leaves=15 reference",
            "roc_auc": 0.75,
            "log_loss": 0.6,
            "brier_score": 0.2,
        }

    monkeypatch.setattr(
        tuning, "_cross_fitted_hgb_reference", fake_cross_fitted_reference
    )
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
        "--reference-folds",
        "2",
        "--tree-estimator",
        "river-hoeffding",
        "--include-pgts",
    ]

    assert tuning.main(arguments) == 0
    assert sorted(pgts_calls) == [
        (2.0, 1.0, 0, "doubling"),
        (2.0, 1.0, 1, "doubling"),
    ]
    checkpoints = list((output / "checkpoints").rglob("*.json"))
    assert len(checkpoints) == 8
    checkpoint_policies = {
        path.relative_to(output / "checkpoints").parts[0] for path in checkpoints
    }
    assert checkpoint_policies == {
        "cbpside",
        "squarecb.pmsidelinear",
        "squarecb.pmside",
        "pgts",
    }
    assert all(
        "multiplier-1" in checkpoint.parts for checkpoint in checkpoints
    )
    assert not any("fixed" in checkpoint.parts for checkpoint in checkpoints)
    for checkpoint in checkpoints:
        checkpoint_row = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert all(
            key in checkpoint_row for key in tuning.INTERNAL_ACTION_FIELDS
        )
    manifest = json.loads(
        (output / "sweep_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["design"] == "pointwise-online-parameter-multiplier-sweep-v9"
    assert manifest["implementation_revision"] == 9
    assert manifest["update_schedule"]["name"] == "doubling"
    assert manifest["update_schedule"]["maximum_round_gap"] is None
    assert manifest["update_schedule"]["boundary_rounds"] == [1, 2, 4, 8]
    assert manifest["tuned_policies"] == list(tuning.TUNED_POLICIES)
    assert manifest["fixed_policies"] == []
    assert manifest["candidate_counts"] == {
        "multiplier_tuned_checkpoints": 8,
        "pgts_prior_tuned_checkpoints": 2,
        "total_checkpoints": 8,
        "multiplier_tuned_execution_groups": 4,
        "pgts_prior_tuned_execution_groups": 1,
        "total_execution_groups": 4,
    }
    candidate_rows = json.loads(
        (output / "candidate_results_by_order.json").read_text(encoding="utf-8")
    )
    assert len(candidate_rows) == 8
    assert all(
        key not in row
        for row in candidate_rows
        for key in tuning.INTERNAL_ACTION_FIELDS
    )
    with (output / "candidate_results_by_order.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        candidate_csv_fields = csv.DictReader(stream).fieldnames
    assert candidate_csv_fields is not None
    assert all(
        key not in candidate_csv_fields for key in tuning.INTERNAL_ACTION_FIELDS
    )
    assert {
        policy: sum(row["policy"] == policy for row in candidate_rows)
        for policy in tuning.TUNED_POLICIES
    } == {policy: 2 for policy in tuning.TUNED_POLICIES}
    assert {row["policy"] for row in candidate_rows} == {
        "CBPSide",
        "SquareCB.PMSideLinear",
        "SquareCB.PMSide",
        "PGTS",
    }
    assert {row["method"] for row in candidate_rows} == {
        "CBPSide (linear logistic)",
        "SquareCB.PMSide + linear logistic",
        "SquareCB.PMSide + Hoeffding tree depth<=4",
        "PG-TS (Bayesian logistic)",
    }
    assert all("IGW" not in row["policy"] for row in candidate_rows)
    assert all("IGW" not in row["method"] for row in candidate_rows)
    pgts_rows = [
        row for row in candidate_rows if row["policy"] == tuning.POLICY_PGTS
    ]
    assert len(pgts_rows) == 2
    assert all(row["multiplier"] == 1.0 for row in pgts_rows)
    assert all(row["parameter_name"] == "prior_std" for row in pgts_rows)
    assert all(row["base_parameter"] == 1.0 for row in pgts_rows)
    assert all(row["effective_parameter"] == 1.0 for row in pgts_rows)
    assert all(row["pgts_base_prior_std"] == 1.0 for row in pgts_rows)
    assert all(row["pgts_prior_std_multiplier"] == 1.0 for row in pgts_rows)
    assert all(row["pgts_prior_std"] == 1.0 for row in pgts_rows)
    assert all(row["pgts_gibbs_steps"] == 15 for row in pgts_rows)
    assert all(row["normalized_dimension"] == 3 for row in pgts_rows)
    assert all(row["pgts_algorithm1_exact"] is False for row in pgts_rows)
    assert all(row["posterior_draw_rounds"] == [1, 2, 4, 8] for row in pgts_rows)
    assert all(row["model_updates"] == 4 for row in pgts_rows)
    assert all(row["total_gibbs_transitions"] == 60 for row in pgts_rows)
    assert all(
        row["update_schedule"]
        == "global_round_doubling_posterior_draw_before_action"
        for row in pgts_rows
    )
    for policy in (
        tuning.POLICY_CBPSIDE,
        tuning.POLICY_IGW_LINEAR,
        tuning.POLICY_IGW_TREE,
    ):
        rows = [row for row in candidate_rows if row["policy"] == policy]
        assert all(
            row["update_schedule"]
            == "global_round_doubling_before_action"
            for row in rows
        )
    selected = json.loads(
        (output / "selected_results.json").read_text(encoding="utf-8")
    )
    assert [row["policy"] for row in selected] == [
        *tuning.TUNED_POLICIES,
        "Random",
    ]
    selected_multipliers = json.loads(
        (output / "selected_multipliers.json").read_text(encoding="utf-8")
    )
    assert {row["policy"] for row in selected_multipliers} == set(
        tuning.TUNED_POLICIES
    )
    pgts_selection = next(
        row
        for row in selected_multipliers
        if row["policy"] == tuning.POLICY_PGTS
    )
    assert pgts_selection["selected_multiplier"] == 1.0
    assert pgts_selection["parameter_name"] == "prior_std"
    selected_by_order = json.loads(
        (output / "selected_results_by_order.json").read_text(encoding="utf-8")
    )
    assert all(
        key not in row
        for row in selected_by_order
        for key in tuning.INTERNAL_ACTION_FIELDS
    )
    random_rows = [row for row in selected_by_order if row["policy"] == "Random"]
    assert len(random_rows) == 2
    assert all(
        row["matched_squarecb_pmside_policy"] == tuning.POLICY_IGW_TREE
        for row in random_rows
    )
    assert all(
        row["matched_squarecb_pmside_probability_estimator"]
        == "river-hoeffding"
        for row in random_rows
    )
    assert all(
        row["matched_squarecb_pmside_multiplier"] == 1.0
        for row in random_rows
    )
    assert all(row["method"] == "Random" for row in random_rows)
    assert all("matched_igw_policy" not in row for row in random_rows)
    assert all("IGW" not in row["method"] for row in selected_by_order)
    learning_npz = output / tuning.LEARNING_CURVE_NPZ
    learning_csv = output / tuning.LEARNING_CURVE_CSV
    learning_plot = output / "selected_cumulative_reference_regret_l01-2.png"
    average_regret_plot = output / "selected_average_reference_regret_l01-2.png"
    reference_npz = output / tuning.REFERENCE_PREDICTIONS_NPZ
    reference_csv = output / tuning.REFERENCE_RESULTS_CSV
    reference_json = output / tuning.REFERENCE_RESULTS_JSON
    assert learning_npz.stat().st_size > 0
    assert learning_csv.stat().st_size > 0
    assert learning_plot.stat().st_size > 0
    assert learning_plot.with_suffix(".pdf").stat().st_size > 0
    assert average_regret_plot.stat().st_size > 0
    assert average_regret_plot.with_suffix(".pdf").stat().st_size > 0
    assert reference_npz.stat().st_size > 0
    assert reference_csv.stat().st_size > 0
    assert reference_json.stat().st_size > 0
    with np.load(learning_npz) as curves:
        assert curves["schema_version"].item() == 2
        assert curves["round"].tolist() == list(range(1, 9))
        assert curves["cumulative_cost"].shape == (10, 8)
        assert curves["cumulative_cost"].dtype == np.float32
        assert curves["cumulative_reference_cost"].dtype == np.float32
        assert curves["cumulative_reference_regret"].dtype == np.float32
        assert curves["average_reference_regret"].dtype == np.float32
        assert curves["cumulative_clairvoyant_excess_cost"].dtype == np.float32
        assert set(curves["policy"].tolist()) == {
            *tuning.TUNED_POLICIES,
            "Random",
        }
        assert all("IGW" not in value for value in curves["policy"].tolist())
        assert all("IGW" not in value for value in curves["method"].tolist())
    with learning_csv.open(encoding="utf-8", newline="") as stream:
        learning_csv_rows = list(csv.DictReader(stream))
    assert len(learning_csv_rows) == 5 * 8
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["plot_presentation"] == manifest["plot_presentation"]
    assert summary["selection_count"] == 4
    assert summary["learning_curves"]["trajectory_count"] == 10
    assert summary["learning_curves"]["aggregate_policy_loss_groups"] == 5
    assert summary["learning_curves"]["plot_formats"] == ["png", "pdf"]
    assert summary["learning_curves"]["png_dpi"] == 400
    assert summary["learning_curves"]["paired_pdf_for_every_png"] is True
    assert summary["learning_curves"]["cumulative_reference_regret_plots"] == [
        learning_plot.name
    ]
    assert summary["learning_curves"]["average_reference_regret_plots"] == [
        average_regret_plot.name
    ]
    assert summary["learning_curves"]["uncertainty"] == {
        "raw_statistics": (
            "mean, sample SD, and SEM across paired shuffled online orders"
        ),
        "plotted_interval": (
            "pointwise 95% Student-t confidence interval for the mean"
        ),
        "confidence_level": 0.95,
        "degrees_of_freedom": 1,
        "t_critical": pytest.approx(12.706204736432095),
        "formula": (
            "mean +/- t.ppf((1 + confidence_level) / 2, df) * "
            "sample_sd / sqrt(n)"
        ),
        "simultaneous_band": False,
        "post_selection_adjusted": False,
    }
    assert summary["learning_curves"]["primary_regret_comparator"] == (
        "2-fold cross-fitted HGB leaves=15 reference"
    )
    assert summary["regret_reference"]["prediction_artifact"] == (
        tuning.REFERENCE_PREDICTIONS_NPZ
    )
    assert summary["regret_reference"]["threshold_results"][0][
        "reference_each_row_held_out"
    ] is True
    assert "cumulative_cost_plots" not in summary["learning_curves"]
    assert "squarecb_pmside_tree_vs_linear_separately_tuned" in summary
    assert "squarecb_pmside_tree_vs_linear_matched_gamma" in summary
    assert "squarecb_pmside_comparison_interpretation" in summary
    assert "igw_tree_vs_linear_separately_tuned" not in summary
    assert "igw_tree_vs_linear_matched_gamma" not in summary
    assert "igw_comparison_interpretation" not in summary
    assert (
        summary["learning_curves"][
            "normal_result_tables_include_internal_action_payload"
        ]
        is False
    )
    assert (output / "selected_routing_accuracy.png").stat().st_size > 0
    assert (output / "selected_routing_accuracy.pdf").stat().st_size > 0
    assert (output / "selected_cost_vs_l01.png").stat().st_size > 0
    assert (output / "selected_cost_vs_l01.pdf").stat().st_size > 0
    assert (output / "selected_multiplier_vs_l01.png").stat().st_size > 0
    assert (output / "selected_multiplier_vs_l01.pdf").stat().st_size > 0
    comparison_artifacts = [
        "squarecb_pmside_tree_vs_linear_by_order.csv",
        "squarecb_pmside_tree_vs_linear_by_order.json",
        "squarecb_pmside_tree_vs_linear.csv",
        "squarecb_pmside_tree_vs_linear.json",
        "squarecb_pmside_tree_vs_linear_matched_by_order.csv",
        "squarecb_pmside_tree_vs_linear_matched_by_order.json",
        "squarecb_pmside_tree_vs_linear_matched.csv",
        "squarecb_pmside_tree_vs_linear_matched.json",
        "squarecb_pmside_tree_vs_linear_cost_difference.png",
        "squarecb_pmside_tree_vs_linear_cost_difference.pdf",
        "squarecb_pmside_tree_vs_linear_matched_cost_difference.png",
        "squarecb_pmside_tree_vs_linear_matched_cost_difference.pdf",
    ]
    assert all((output / name).stat().st_size > 0 for name in comparison_artifacts)
    assert not list(output.glob("igw_tree_vs_linear*"))
    assert not list(output.glob("selected_cumulative_cost_l01-*.png"))
    assert not list(output.glob("selected_cumulative_regret_l01-*.png"))
    for public_json in output.glob("*.json"):
        public_text = public_json.read_text(encoding="utf-8")
        assert "IGW" not in public_text
        assert '"igw_' not in public_text
    assert (output / "multiplier-sweep-results.zip").stat().st_size > 0

    created_count = len(created_backends)
    pgts_call_count = len(pgts_calls)
    assert tuning.main(arguments) == 0
    assert len(created_backends) == created_count
    assert len(pgts_calls) == pgts_call_count
    for artifact in (
        learning_npz,
        learning_csv,
        learning_plot,
        learning_plot.with_suffix(".pdf"),
        average_regret_plot,
        average_regret_plot.with_suffix(".pdf"),
        reference_npz,
        reference_csv,
        reference_json,
    ):
        artifact.unlink()
        assert not artifact.exists()
    assert tuning.main(arguments + ["--plot-only"]) == 0
    assert len(created_backends) == created_count
    assert len(pgts_calls) == pgts_call_count
    assert learning_npz.stat().st_size > 0
    assert learning_csv.stat().st_size > 0
    assert learning_plot.stat().st_size > 0
    assert learning_plot.with_suffix(".pdf").stat().st_size > 0
    assert average_regret_plot.stat().st_size > 0
    assert average_regret_plot.with_suffix(".pdf").stat().st_size > 0
    assert reference_npz.stat().st_size > 0
    assert reference_csv.stat().st_size > 0
    assert reference_json.stat().st_size > 0
    assert reference_calls == [(2, 0), (2, 0), (2, 0)]
