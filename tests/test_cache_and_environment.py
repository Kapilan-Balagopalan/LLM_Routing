import io
import json
import zipfile

import numpy as np

from llm_routing_simulation.cache import load_cache
from llm_routing_simulation.environment import LLMCascadeEnvironment


def _fixture_cache(path):
    records = [
        {
            "id": "0",
            "prompt": "Question zero",
            "weak_answer": "A",
            "strong_answer": "A",
        },
        {
            "id": "1",
            "prompt": "Question one",
            "weak_answer": "B",
            "strong_answer": "C",
        },
    ]
    manifest = {
        "schema_version": "llm-routing-cache-v1",
        "examples": 2,
        "pca_components": 2,
        "routing_reference": "strong_model_answer",
    }
    arrays = {
        "hidden_states": np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float16),
        "uncertainty_features": np.zeros((2, 14), dtype=np.float32),
        "option_probabilities": np.full((2, 4), 0.25, dtype=np.float32),
        "option_log_likelihoods": np.zeros((2, 4), dtype=np.float32),
        "contexts": np.zeros((2, 16), dtype=np.float32),
        "outcomes": np.asarray([0, 1], dtype=np.int8),
        "eligible": np.asarray([True, True]),
        "pca_mean": np.zeros(3, dtype=np.float32),
        "pca_axes": np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "pca_scales": np.ones(2, dtype=np.float32),
        "context_mean": np.zeros(16, dtype=np.float32),
        "context_scale": np.ones(16, dtype=np.float32),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr(
            "records.jsonl", "\n".join(json.dumps(record) for record in records)
        )
        archive.writestr("arrays.npz", buffer.getvalue())


def test_cache_load_and_partial_feedback(tmp_path):
    path = tmp_path / "cache.zip"
    _fixture_cache(path)
    cache = load_cache(path)
    rounds = cache.eligible_rounds()
    environment = LLMCascadeEnvironment(rounds)

    first = environment.observe()
    hidden = environment.step(0)
    assert first.weak_answer == "A"
    assert hidden.outcome is None
    assert hidden.revealed_strong_answer is None

    environment.observe()
    revealed = environment.step(1)
    assert revealed.outcome == 1
    assert revealed.revealed_strong_answer == "C"


def test_smaller_fixed_pca_context(tmp_path):
    path = tmp_path / "cache.zip"
    _fixture_cache(path)
    cache = load_cache(path)
    contexts = cache.contexts(pca_components=1)
    assert contexts.shape == (2, 15)
    assert np.all(np.isfinite(contexts))
