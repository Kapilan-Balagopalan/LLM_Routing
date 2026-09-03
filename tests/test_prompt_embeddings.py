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
    prompt_pca_context,
)
from llm_routing_simulation.skyline import cross_fitted_residual_predictability


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
