from types import SimpleNamespace

import numpy as np
import pytest

from llm_routing_simulation.cache import RoutingCache
from llm_routing_simulation.prompt_embeddings import (
    PROMPT_EMBEDDING_SCHEMA_VERSION,
    build_semantic_prompt,
    load_prompt_embedding_cache,
    save_prompt_embedding_cache,
    validate_prompt_embedding_source,
)
from llm_routing_simulation.prompt_experiment import (
    align_prompt_embeddings,
    build_context_matrices,
    evaluate_two_stage_correction,
    prompt_pca_context,
    summarize_two_stage_seeds,
)
from llm_routing_simulation.run import _prompt_context_rounds
from llm_routing_simulation.skyline import (
    cross_fitted_residual_predictability,
    evaluate_probability_predictions,
)


def _routing_cache():
    records = [
        {
            "id": "arc-1",
            "question": "Which material is magnetic?",
            "choices": {"B": "wood", "A": "iron"},
            "gold_answer": "A",
            "weak_response": "Answer: B",
        },
        {
            "id": "arc-2",
            "question": "What do plants need for photosynthesis?",
            "choices": {"A": "sunlight", "B": "sand"},
            "gold_answer": "A",
            "weak_response": "Answer: A",
        },
        {
            "id": "arc-3",
            "question": "Which object conducts electricity?",
            "choices": {"A": "copper", "B": "rubber"},
            "gold_answer": "A",
            "weak_response": "Answer: B",
        },
    ]
    return RoutingCache(
        manifest={"schema_version": "llm-routing-cache-v1"},
        records=records,
        arrays={},
    )


def test_semantic_prompt_excludes_answers_and_model_responses():
    record = _routing_cache().records[0]
    text = build_semantic_prompt(record)
    assert text == "Question: Which material is magnetic?\nA. iron\nB. wood"
    assert "gold_answer" not in text
    assert "Answer:" not in text


def test_prompt_embedding_sidecar_round_trip_and_alignment(tmp_path):
    cache = _routing_cache()
    texts = [build_semantic_prompt(record) for record in cache.records]
    embeddings = np.asarray(
        [[1.0, 0.0, 0.2], [0.0, 1.0, 0.3], [0.5, 0.5, 0.4]],
        dtype=np.float32,
    )
    path = tmp_path / "prompt-embeddings.zip"
    save_prompt_embedding_cache(
        path,
        source_cache=cache,
        embedding_model="fixture-encoder",
        texts=texts,
        embeddings=embeddings,
        normalized=False,
    )
    loaded = load_prompt_embedding_cache(path)
    assert loaded.manifest["schema_version"] == PROMPT_EMBEDDING_SCHEMA_VERSION
    assert loaded.manifest["outcome_or_answer_features_used"] is False
    assert loaded.embeddings.shape == (3, 3)
    validate_prompt_embedding_source(loaded, cache)

    rounds = [SimpleNamespace(example_id="arc-3"), SimpleNamespace(example_id="arc-1")]
    aligned = align_prompt_embeddings(rounds, loaded)
    assert np.allclose(aligned, embeddings[[2, 0]])

    cache.records[0]["question"] = "A different question"
    with pytest.raises(ValueError, match="source prompt text"):
        validate_prompt_embedding_source(loaded, cache)


def test_prompt_pca_is_fixed_size_and_outcome_free():
    rng = np.random.default_rng(4)
    embeddings = rng.normal(size=(30, 12))
    context, summary = prompt_pca_context(embeddings, components=5)
    assert context.shape == (30, 5)
    assert np.all(np.isfinite(context))
    assert np.allclose(context.mean(axis=0), 0.0, atol=1e-10)
    assert summary["scope"] == "fixed transductive, outcome-free, selected eligible prompts"


def test_prompt_context_rounds_use_manifest_block_before_limit():
    records = [
        {
            "id": f"arc-{index}",
            "question": f"Question {index}?",
            "choices": {"A": "first", "B": "second"},
            "prompt": f"Cached generation prompt {index}",
            "weak_answer": "A",
            "strong_answer": "B" if index % 2 else "A",
        }
        for index in range(6)
    ]
    cache = RoutingCache(
        manifest={
            "schema_version": "llm-routing-cache-v2",
            "pca_components": 64,
            "prompt_embedding_definition": "fixture prompt representation",
            "pca_scope": "fixed transductive, outcome-free",
            "context_blocks": [
                {"name": "uncertainty", "start": 0, "stop": 2},
                {"name": "hidden_state_pca", "start": 2, "stop": 4},
                {"name": "prompt_embedding_pca", "start": 4, "stop": 8},
            ],
        },
        records=records,
        arrays={
            "eligible": np.ones(6, dtype=bool),
            "contexts": np.concatenate(
                (
                    np.column_stack(
                        (
                            100.0 + np.arange(6),
                            200.0 + np.arange(6),
                        )
                    ),
                    np.full((6, 2), -999.0),
                    np.arange(24, dtype=np.float64).reshape(6, 4),
                ),
                axis=1,
            ),
        },
    )

    rounds, summary, teacher_summary, synthetic_rows = _prompt_context_rounds(
        cache,
        prompt_components=4,
        limit=3,
        seed=0,
        outcome_source="cached",
    )
    assert len(rounds) == 3
    assert all(round_.context.shape == (4,) for round_ in rounds)
    assert np.array_equal(
        np.stack([round_.context for round_ in rounds]),
        cache.arrays["contexts"][:3, 4:8],
    )
    assert summary["context_dimension"] == 4
    assert summary["context_block"]["start"] == 4
    assert summary["context_block"]["stop"] == 8
    assert summary["outcome_or_answer_features_used"] is False
    assert summary["fit_scope"] == "precomputed across all collected prompts"
    assert summary["profile"] == "prompt-only"
    assert summary["context_block_order"] == ["prompt_embedding_pca"]
    assert summary["uncertainty_features_used"] is False
    assert summary["hidden_state_features_used"] is False
    assert summary["component_selection"]["prompt_embedding_pca"] == (
        "first components in manifest-defined PCA order"
    )
    assert teacher_summary is None
    assert synthetic_rows == []
    assert [round_.outcome_override for round_ in rounds] == [None, None, None]
    assert [round_.routing_outcome for round_ in rounds] == [0, 1, 0]

    synthetic_rounds, _, teacher_summary, synthetic_rows = _prompt_context_rounds(
        cache,
        prompt_components=4,
        limit=3,
        seed=0,
        outcome_source="synthetic",
    )
    assert teacher_summary["examples"] == 6
    assert teacher_summary["selected_examples"] == 3
    assert len(synthetic_rows) == 3
    assert [row["id"] for row in synthetic_rows] == [
        "arc-0",
        "arc-1",
        "arc-2",
    ]
    assert [round_.routing_outcome for round_ in synthetic_rounds] == [
        row["synthetic_outcome"] for row in synthetic_rows
    ]

    hybrid_rounds, hybrid_summary, _, _ = _prompt_context_rounds(
        cache,
        prompt_components=4,
        limit=3,
        seed=0,
        outcome_source="cached",
        context_profile="uncertainty-prompt",
    )
    expected_hybrid = np.concatenate(
        (cache.arrays["contexts"][:3, 0:2], cache.arrays["contexts"][:3, 4:8]),
        axis=1,
    )
    assert all(round_.context.shape == (6,) for round_ in hybrid_rounds)
    assert np.array_equal(
        np.stack([round_.context for round_ in hybrid_rounds]),
        expected_hybrid,
    )
    assert not np.any(expected_hybrid == -999.0)
    assert hybrid_summary["profile"] == "uncertainty-prompt"
    assert hybrid_summary["context_dimension"] == 6
    assert hybrid_summary["context_block_order"] == [
        "uncertainty",
        "prompt_embedding_pca",
    ]
    assert hybrid_summary["uncertainty_context_block"]["start"] == 0
    assert hybrid_summary["uncertainty_context_block"]["stop"] == 2
    assert hybrid_summary["prompt_context_block"]["start"] == 4
    assert hybrid_summary["prompt_context_block"]["stop"] == 8
    assert hybrid_summary["uncertainty_features_used"] is True
    assert hybrid_summary["hidden_state_features_used"] is False

    with pytest.raises(ValueError, match="context_profile"):
        _prompt_context_rounds(
            cache,
            prompt_components=4,
            limit=3,
            context_profile="unknown",
        )


def test_compact_context_is_14_uncertainty_plus_32_prompt_dimensions():
    current = np.zeros((20, 78), dtype=np.float64)
    prompt = np.ones((20, 32), dtype=np.float64)
    matrices = build_context_matrices(current, prompt, uncertainty_dimension=14)
    assert matrices["Current 78D"].shape == (20, 78)
    assert matrices["Prompt PCA"].shape == (20, 32)
    assert matrices["Uncertainty + prompt PCA"].shape == (20, 46)


def test_incremental_residual_test_accepts_separate_prompt_context():
    rng = np.random.default_rng(9)
    base = rng.normal(size=(90, 6))
    prompt = rng.normal(size=(90, 4))
    outcomes = np.tile(np.asarray([0, 1], dtype=np.int64), 45)
    _, _, _, summary = cross_fitted_residual_predictability(
        base,
        outcomes,
        residual_contexts=prompt,
        base_context_label="current",
        residual_context_label="prompt",
        requested_folds=3,
        permutation_repeats=0,
    )
    assert summary["base_context_dimension"] == 6
    assert summary["residual_context_dimension"] == 4
    assert summary["base_context"] == "current"
    assert summary["residual_context"] == "prompt"
    assert summary["incrementally_useful_at_0.05"] == (
        summary["mse_improvement"] > 0.0
        and summary["permutation_p_value_one_sided"] is not None
        and summary["permutation_p_value_one_sided"] <= 0.05
    )


def test_two_stage_correction_uses_aligned_clipped_probabilities():
    rows = [
        {
            "example_index": 1,
            "observed_disagreement": 1,
            "logistic_probability": 0.7,
            "predicted_logistic_residual": 0.4,
        },
        {
            "example_index": 0,
            "observed_disagreement": 0,
            "logistic_probability": 0.2,
            "predicted_logistic_residual": -0.1,
        },
        {
            "example_index": 3,
            "observed_disagreement": 1,
            "logistic_probability": 0.6,
            "predicted_logistic_residual": 0.1,
        },
        {
            "example_index": 2,
            "observed_disagreement": 0,
            "logistic_probability": 0.65,
            "predicted_logistic_residual": -0.45,
        },
    ]
    corrected, outcomes, summary = evaluate_two_stage_correction(rows)
    assert np.array_equal(outcomes, [0, 1, 0, 1])
    assert np.allclose(corrected, [0.1, 1.0 - 1e-7, 0.2, 0.7])
    assert summary["corrected_metrics"]["brier_score"] < summary["base_metrics"][
        "brier_score"
    ]
    assert summary["improves_all_auc_logloss_brier"]


def test_probability_metrics_and_seed_stability_summary():
    metrics = evaluate_probability_predictions(
        np.asarray([0.1, 0.8, 0.2, 0.9]),
        np.asarray([0, 1, 0, 1]),
    )
    assert metrics["roc_auc"] == 1.0
    assert metrics["routing_accuracy_at_50_percent"] == 1.0

    rows = [
        {
            "seed": 2,
            "delta_roc_auc": 0.01,
            "delta_log_loss": -0.02,
            "delta_brier_score": -0.01,
            "delta_routing_accuracy_at_50_percent": 0.02,
            "improves_all_auc_logloss_brier": True,
        },
        {
            "seed": 3,
            "delta_roc_auc": -0.01,
            "delta_log_loss": 0.01,
            "delta_brier_score": 0.01,
            "delta_routing_accuracy_at_50_percent": 0.0,
            "improves_all_auc_logloss_brier": False,
        },
    ]
    summary = summarize_two_stage_seeds(rows)
    assert summary["seeds"] == [2, 3]
    assert summary["seeds_improving_auc"] == 1
    assert summary["seeds_improving_all_auc_logloss_brier"] == 1
    assert np.isclose(
        summary["aggregate_deltas_corrected_minus_base"]["delta_roc_auc"]["mean"],
        0.0,
    )
