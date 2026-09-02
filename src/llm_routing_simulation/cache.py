"""Read and validate a collected LLM-routing cache."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llm_routing_simulation import CACHE_SCHEMA_VERSION
from llm_routing_simulation.environment import CascadeRound


@dataclass(frozen=True)
class RoutingCache:
    manifest: dict
    records: list[dict]
    arrays: dict[str, np.ndarray]

    @property
    def eligible_indices(self) -> np.ndarray:
        return np.flatnonzero(self.arrays["eligible"])

    def contexts(self, pca_components: int | None = None) -> np.ndarray:
        """Return saved contexts or rebuild a smaller PCA context from raw states."""
        if pca_components is None or pca_components == self.manifest["pca_components"]:
            return np.asarray(self.arrays["contexts"], dtype=np.float64)
        available = int(self.arrays["pca_axes"].shape[0])
        if not 1 <= pca_components <= available:
            raise ValueError(f"pca_components must be between 1 and {available}")
        hidden = np.asarray(self.arrays["hidden_states"], dtype=np.float64)
        mean = np.asarray(self.arrays["pca_mean"], dtype=np.float64)
        axes = np.asarray(self.arrays["pca_axes"][:pca_components], dtype=np.float64)
        scales = np.asarray(self.arrays["pca_scales"][:pca_components], dtype=np.float64)
        projected = ((hidden - mean) @ axes.T) / scales
        uncertainty = np.asarray(self.arrays["uncertainty_features"], dtype=np.float64)
        raw = np.concatenate((uncertainty, projected), axis=1)
        raw_mean, raw_scale = raw.mean(axis=0), raw.std(axis=0)
        raw_scale[raw_scale < 1e-8] = 1.0
        return (raw - raw_mean) / raw_scale

    def eligible_rounds(self, pca_components: int | None = None) -> list[CascadeRound]:
        contexts = self.contexts(pca_components)
        rounds = []
        for index in self.eligible_indices:
            record = self.records[int(index)]
            rounds.append(
                CascadeRound(
                    example_id=str(record["id"]),
                    prompt=str(record["prompt"]),
                    context=contexts[index],
                    weak_answer=str(record["weak_answer"]),
                    strong_answer=str(record["strong_answer"]),
                    gold_answer=str(record["strong_answer"]),
                )
            )
        return rounds


def load_cache(path: str | Path) -> RoutingCache:
    source = Path(path)
    with zipfile.ZipFile(source) as archive:
        required = {"manifest.json", "records.jsonl", "arrays.npz"}
        missing = required.difference(archive.namelist())
        if missing:
            raise ValueError(f"Cache is missing: {sorted(missing)}")
        manifest = json.loads(archive.read("manifest.json"))
        records = [
            json.loads(line)
            for line in archive.read("records.jsonl").decode("utf-8").splitlines()
            if line.strip()
        ]
        with np.load(io.BytesIO(archive.read("arrays.npz")), allow_pickle=False) as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Expected {CACHE_SCHEMA_VERSION}, got {manifest.get('schema_version')}"
        )
    count = int(manifest["examples"])
    if len(records) != count or any(len(value) != count for key, value in arrays.items() if key in {"hidden_states", "uncertainty_features", "option_probabilities", "option_log_likelihoods", "contexts", "outcomes", "eligible"}):
        raise ValueError("Cache record and array counts do not agree")
    if manifest.get("routing_reference") != "strong_model_answer":
        raise ValueError("This simulator requires the strong model as reference")
    return RoutingCache(manifest, records, arrays)
