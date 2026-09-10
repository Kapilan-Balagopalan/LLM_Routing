from types import SimpleNamespace

import numpy as np
import pytest

import llm_routing_simulation.online_tree as online_tree
from llm_routing_simulation.algorithm import ONLINE_HGB_PROFILE
from llm_routing_simulation.online_tree import (
    HGBBatchTreeBackend,
    LogisticBatchProbabilityBackend,
    ONLINE_LOGISTIC_PROFILE,
    RiverHoeffdingTreeBackend,
    TreeProbabilityBackend,
    make_tree_backend,
)


def _binary_batch(rows: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.linspace(-2.0, 2.0, rows)
    X = np.column_stack((values, values**2))
    y = (values > 0.0).astype(np.int8)
    weights = np.linspace(1.0, 2.0, rows)
    return X, y, weights


def test_hgb_backend_uses_the_fixed_online_profile_and_full_refits():
    X, y, weights = _binary_batch()
    backend = HGBBatchTreeBackend(seed=17)

    backend.fit_all(X[:30], y[:30], weights[:30])
    backend.update_many(X, y, weights)

    assert isinstance(backend, TreeProbabilityBackend)
    assert backend.estimator_name == "sklearn_hgb_15_leaf_batch"
    assert backend.supports_incremental is False
    assert backend.fit_count == 2
    assert backend.training_count == 40
    params = backend.model.get_params()
    for key in (
        "loss",
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "early_stopping",
    ):
        assert params[key] == ONLINE_HGB_PROFILE[key]
    assert params["random_state"] == 17
    probabilities = backend.predict_proba(X[:3])
    assert probabilities.shape == (3,)
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))


class _FakeHoeffdingTreeClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.learned = []

    def learn_one(self, features, label, *, w):
        self.learned.append((features, label, w))
        return self

    def predict_proba_one(self, features):
        return {0: 0.25, 1: 0.75 if features[0] >= 0.0 else 0.2}


def test_river_backend_is_lazy_incremental_and_uses_weights(monkeypatch):
    imports = []

    def fake_import():
        imports.append("river")
        return SimpleNamespace(HoeffdingTreeClassifier=_FakeHoeffdingTreeClassifier)

    monkeypatch.setattr(online_tree, "_import_river_tree", fake_import)
    backend = RiverHoeffdingTreeBackend(seed=9)
    assert imports == []
    assert backend.model is None

    X = np.asarray([[-1.0, 2.0], [0.5, 3.0], [1.0, 4.0]])
    y = np.asarray([0, 1, 1])
    weights = np.asarray([1.0, 2.5, 4.0])
    backend.update_many(X[:2], y[:2], weights[:2])
    backend.update_many(X[2:], y[2:], weights[2:])

    assert imports == ["river"]
    assert backend.estimator_name == "river_hoeffding_tree_depth4_incremental"
    assert backend.supports_incremental is True
    assert backend.fit_count == 2
    assert backend.training_count == 3
    assert backend.model.kwargs == {
        "max_depth": 4,
        "grace_period": 200,
        "delta": 1e-7,
        "tau": 0.05,
        "leaf_prediction": "mc",
        "binary_split": True,
    }
    assert backend.model.learned[1] == ({0: 0.5, 1: 3.0}, 1, 2.5)
    assert np.allclose(backend.predict_proba(X), [0.2, 0.75, 0.75])


def test_river_fit_all_resets_the_incremental_model(monkeypatch):
    monkeypatch.setattr(
        online_tree,
        "_import_river_tree",
        lambda: SimpleNamespace(
            HoeffdingTreeClassifier=_FakeHoeffdingTreeClassifier
        ),
    )
    backend = RiverHoeffdingTreeBackend()
    X, y, weights = _binary_batch(8)
    backend.update_many(X[:3], y[:3], weights[:3])
    original_model = backend.model

    backend.fit_all(X, y, weights)

    assert backend.model is not original_model
    assert backend.fit_count == 2
    assert backend.training_count == 8
    assert len(backend.model.learned) == 8


def test_logistic_backend_is_deterministic_full_refit_using_all_features():
    X, y, weights = _binary_batch(60)
    extra_feature = np.sin(np.arange(X.shape[0], dtype=np.float64))
    X = np.column_stack((X, extra_feature))
    first = LogisticBatchProbabilityBackend(seed=23)
    second = LogisticBatchProbabilityBackend(seed=23)

    first.fit_all(X[:40], y[:40], weights[:40])
    first.update_many(X, y, weights)
    second.fit_all(X, y, weights)

    assert isinstance(first, TreeProbabilityBackend)
    assert first.estimator_name == "sklearn_standardized_logistic_l2_batch"
    assert first.supports_incremental is False
    assert first.fit_count == 2
    assert first.training_count == 60
    assert first.scaler.n_features_in_ == 3
    assert first.model.n_features_in_ == 3
    assert first.model.get_params()["random_state"] == 23
    assert np.allclose(first.predict_proba(X), second.predict_proba(X))


def test_logistic_backend_profile_and_training_history_weighted_scaling():
    X = np.asarray(
        [
            [0.0, -2.0],
            [1.0, -1.0],
            [10.0, 1.0],
            [11.0, 2.0],
        ]
    )
    y = np.asarray([0, 0, 1, 1])
    weights = np.asarray([1.0, 1.0, 5.0, 5.0])
    backend = LogisticBatchProbabilityBackend(seed=5)

    backend.fit_all(X, y, weights)

    assert backend.profile == ONLINE_LOGISTIC_PROFILE
    assert backend.profile is not ONLINE_LOGISTIC_PROFILE
    assert backend.metadata["preprocessing_fit_scope"] == (
        "supplied_revealed_history_only"
    )
    assert backend.metadata["sample_weight_usage"] == (
        "scaler_and_logistic_objective"
    )
    assert np.allclose(backend.scaler.mean_, np.average(X, axis=0, weights=weights))
    probabilities = backend.predict_proba(X)
    assert probabilities.shape == (4,)
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))


def test_factory_uses_explicit_backend_kinds():
    assert isinstance(make_tree_backend("hgb", seed=3), HGBBatchTreeBackend)
    assert isinstance(
        make_tree_backend("logistic", seed=3),
        LogisticBatchProbabilityBackend,
    )
    assert isinstance(
        make_tree_backend("river-hoeffding", seed=3),
        RiverHoeffdingTreeBackend,
    )
    with pytest.raises(ValueError, match="Unknown tree backend"):
        make_tree_backend("forest", seed=3)


@pytest.mark.parametrize(
    "features, labels, weights, message",
    [
        (np.ones(3), np.asarray([0, 1, 0]), None, "two-dimensional"),
        (np.ones((3, 2)), np.asarray([0, 2, 0]), None, "binary"),
        (
            np.ones((3, 2)),
            np.asarray([0, 1, 0]),
            np.asarray([1.0, 0.0, 1.0]),
            "strictly positive",
        ),
    ],
)
def test_training_batches_are_validated(features, labels, weights, message):
    backend = HGBBatchTreeBackend()
    with pytest.raises(ValueError, match=message):
        backend.fit_all(features, labels, weights)
