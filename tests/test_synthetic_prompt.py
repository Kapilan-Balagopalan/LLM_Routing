import numpy as np

from llm_routing_simulation.synthetic_prompt import (
    SYNTHETIC_FOREST_CONFIGURATION,
    generate_synthetic_prompt_outcomes,
)


def test_synthetic_prompt_forest_is_reproducible_and_multifeature():
    contexts = np.random.default_rng(52).normal(size=(240, 16))

    probabilities, outcomes, summary = generate_synthetic_prompt_outcomes(
        contexts, seed=7
    )
    repeated_probabilities, repeated_outcomes, _ = (
        generate_synthetic_prompt_outcomes(contexts, seed=7)
    )

    assert SYNTHETIC_FOREST_CONFIGURATION["n_estimators"] == 50
    assert SYNTHETIC_FOREST_CONFIGURATION["max_depth"] == 4
    assert summary["input_feature_count"] == 12
    assert len(summary["feature_importances"]) == 12
    assert np.count_nonzero(np.asarray(summary["feature_importances"]) > 0.0) > 1
    assert summary["real_outcomes_used"] is False
    assert summary["arc_gold_answers_used"] is False
    assert np.all((probabilities >= 0.02) & (probabilities <= 0.98))
    assert set(np.unique(outcomes)) == {0, 1}
    assert np.allclose(probabilities, repeated_probabilities)
    assert np.array_equal(outcomes, repeated_outcomes)
