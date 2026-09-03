import io
import json
import zipfile

import numpy as np

from llm_routing_simulation.cache import RoutingCache
from llm_routing_simulation.residual_experiment import (
    _heldout_residual_predictability,
    _parser,
    analyze_holdout_residuals,
    main,
)


def _fixture_data(rows=120):
    rng = np.random.default_rng(12)
    uncertainty = rng.normal(size=(rows, 2))
    hidden = rng.normal(size=(rows, 3))
    prompt = rng.normal(size=(rows, 3))
    outcomes = ((prompt[:, 0] > 0.0) ^ (prompt[:, 1] > 0.0)).astype(np.int8)
    contexts = np.concatenate((uncertainty, hidden, prompt), axis=1).astype(
        np.float32
    )
    manifest = {
        "schema_version": "llm-routing-cache-v2",
        "examples": rows,
        "eligible_examples": rows,
        "routing_reference": "strong_model_answer",
        "outcome_definition": "1 iff extracted weak and strong answers differ",
        "context_blocks": [
            {"name": "uncertainty", "start": 0, "stop": 2},
            {"name": "hidden_state_pca", "start": 2, "stop": 5},
            {"name": "prompt_embedding_pca", "start": 5, "stop": 8},
        ],
    }
    arrays = {
        "contexts": contexts,
        "outcomes": outcomes,
        "eligible": np.ones(rows, dtype=bool),
    }
    records = [{"id": f"example-{index}"} for index in range(rows)]
    return manifest, arrays, records


def _routing_cache():
    manifest, arrays, records = _fixture_data()
    return RoutingCache(manifest=manifest, arrays=arrays, records=records)


def _write_fixture(path):
    manifest, arrays, records = _fixture_data()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr(
            "records.jsonl", "\n".join(json.dumps(record) for record in records)
        )
        archive.writestr("arrays.npz", buffer.getvalue())


def test_holdout_residual_analysis_uses_manifest_blocks_and_shared_split():
    result = analyze_holdout_residuals(
        _routing_cache(),
        validation_fraction=0.5,
        seed=7,
    )

    summary = result["summary"]
    assert summary["split"]["train_examples"] == 60
    assert summary["split"]["validation_examples"] == 60
    assert summary["binning"]["bin_count"] == 8
    assert summary["residual_predictability"]["bin_count"] == 10
    scenarios = {row["scenario"]: row for row in summary["scenarios"]}
    assert scenarios["uncertainty_hidden_logistic"]["context_dimension"] == 5
    assert scenarios["prompt_logistic"]["context_dimension"] == 3
    assert scenarios["synthetic_forest_labels_logistic"]["context_dimension"] == 3
    assert len(result["point_rows"]) == 3 * 60
    assert len(result["bin_rows"]) == 3 * 8
    assert len(result["residual_prediction_point_rows"]) == 3 * 60
    assert len(result["residual_prediction_bin_rows"]) == 3 * 10
    assert all(
        "ci95_lower" in row for row in result["residual_prediction_bin_rows"]
    )
    teacher = summary["synthetic_forest_teacher"]
    assert teacher["real_outcomes_used"] is False
    assert teacher["input_feature_count"] == 3
    assert teacher["configuration"]["n_estimators"] == 50
    assert teacher["configuration"]["max_depth"] == 4
    assert len(teacher["feature_importances"]) == 3
    for scenario in scenarios.values():
        diagnostic = scenario["residual_predictability"]
        assert diagnostic["training_residual_inner_folds"] >= 2
        assert np.isfinite(diagnostic["forest_residual_mse"])
        assert np.isfinite(diagnostic["linear_residual_mse"])
    assert {
        row["label_source"] for row in result["point_rows"]
    } == {
        "cached_weak_strong_disagreement",
        "synthetic_probabilistic_forest",
    }

    positions_by_scenario = {}
    for row in result["point_rows"]:
        positions_by_scenario.setdefault(row["scenario"], set()).add(
            row["eligible_position"]
        )
    assert len({frozenset(value) for value in positions_by_scenario.values()}) == 1


def test_holdout_residual_cli_defaults_and_outputs(tmp_path):
    args = _parser().parse_args(["--cache", "fixture.zip"])
    assert args.validation_fraction == 0.5
    assert args.seed == 0
    assert args.residual_bin_count == 10

    cache_path = tmp_path / "cache.zip"
    output = tmp_path / "results"
    _write_fixture(cache_path)
    assert (
        main(
            [
                "--cache",
                str(cache_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    expected = {
        "summary.json",
        "validation_residuals.csv",
        "binned_residuals.csv",
        "binned_residuals.png",
        "residual_predictability.csv",
        "residual_predictability_bins.csv",
        "residual_predictability.png",
        "holdout-residual-results.zip",
    }
    assert expected == {path.name for path in output.iterdir()}


def test_residual_predictions_do_not_fit_validation_outcomes():
    rng = np.random.default_rng(31)
    train_contexts = rng.normal(size=(80, 5))
    validation_contexts = rng.normal(size=(40, 5))
    train_outcomes = np.tile([0, 1], 40)
    validation_outcomes = np.tile([0, 1], 20)
    validation_probabilities = np.linspace(0.2, 0.8, 40)

    original, _, _ = _heldout_residual_predictability(
        train_contexts,
        validation_contexts,
        train_outcomes,
        validation_outcomes,
        validation_probabilities,
        bin_count=5,
        seed=9,
    )
    flipped, _, _ = _heldout_residual_predictability(
        train_contexts,
        validation_contexts,
        train_outcomes,
        1 - validation_outcomes,
        validation_probabilities,
        bin_count=5,
        seed=9,
    )

    assert np.allclose(
        [row["forest_predicted_residual"] for row in original],
        [row["forest_predicted_residual"] for row in flipped],
    )
    assert np.allclose(
        [row["linear_predicted_residual"] for row in original],
        [row["linear_predicted_residual"] for row in flipped],
    )
    assert not np.allclose(
        [row["observed_logistic_residual"] for row in original],
        [row["observed_logistic_residual"] for row in flipped],
    )
