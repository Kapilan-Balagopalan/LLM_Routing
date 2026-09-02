"""Create and validate an outcome-free prompt-embedding sidecar cache."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from llm_routing_simulation.cache import RoutingCache, load_cache


PROMPT_EMBEDDING_SCHEMA_VERSION = "prompt-embedding-cache-v1"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class PromptEmbeddingCache:
    manifest: dict
    example_ids: np.ndarray
    embeddings: np.ndarray


def build_semantic_prompt(record: dict) -> str:
    """Build question-plus-options text without answers or model responses."""
    question = str(record.get("question", "")).strip()
    choices = record.get("choices")
    if not question or not isinstance(choices, dict) or not choices:
        raise ValueError("Every record needs a question and choice mapping")
    ordered = []
    for label in sorted(choices):
        value = str(choices[label]).strip()
        if not value:
            raise ValueError("Prompt choices must be nonempty")
        ordered.append(f"{label}. {value}")
    return "Question: " + question + "\n" + "\n".join(ordered)


def _sequence_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def save_prompt_embedding_cache(
    path: str | Path,
    *,
    source_cache: RoutingCache,
    embedding_model: str,
    texts: list[str],
    embeddings: np.ndarray,
    normalized: bool,
) -> Path:
    """Save embeddings and alignment metadata as one portable ZIP file."""
    destination = Path(path)
    ids = np.asarray(
        [str(record["id"]) for record in source_cache.records], dtype=np.str_
    )
    matrix = np.asarray(embeddings, dtype=np.float32)
    if len(texts) != len(ids) or matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise ValueError("Prompt texts, embeddings, and cache records must align")
    if matrix.shape[1] < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("Prompt embeddings must be a finite nonempty matrix")

    manifest = {
        "schema_version": PROMPT_EMBEDDING_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_cache_schema_version": source_cache.manifest["schema_version"],
        "source_dataset": source_cache.manifest.get("dataset"),
        "source_dataset_config": source_cache.manifest.get("dataset_config"),
        "source_split": source_cache.manifest.get("split"),
        "examples": int(matrix.shape[0]),
        "embedding_dimension": int(matrix.shape[1]),
        "embedding_model": embedding_model,
        "normalized": bool(normalized),
        "text_definition": "question and labeled choices only",
        "outcome_or_answer_features_used": False,
        "example_id_sha256": _sequence_digest(ids.tolist()),
        "prompt_text_sha256": _sequence_digest(texts),
    }
    array_buffer = io.BytesIO()
    np.savez_compressed(array_buffer, example_ids=ids, embeddings=matrix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr("arrays.npz", array_buffer.getvalue())
    return destination


def load_prompt_embedding_cache(path: str | Path) -> PromptEmbeddingCache:
    """Load a prompt-embedding sidecar without extracting archive paths."""
    with zipfile.ZipFile(Path(path)) as archive:
        required = {"manifest.json", "arrays.npz"}
        missing = required.difference(archive.namelist())
        if missing:
            raise ValueError(f"Prompt embedding cache is missing: {sorted(missing)}")
        manifest = json.loads(archive.read("manifest.json"))
        with np.load(
            io.BytesIO(archive.read("arrays.npz")), allow_pickle=False
        ) as arrays:
            example_ids = arrays["example_ids"].copy()
            embeddings = arrays["embeddings"].copy()

    if manifest.get("schema_version") != PROMPT_EMBEDDING_SCHEMA_VERSION:
        raise ValueError("Unsupported prompt embedding cache schema")
    count = int(manifest["examples"])
    dimension = int(manifest["embedding_dimension"])
    if example_ids.ndim != 1 or len(example_ids) != count:
        raise ValueError("Prompt embedding IDs do not match the manifest")
    if embeddings.shape != (count, dimension) or not np.all(
        np.isfinite(embeddings)
    ):
        raise ValueError("Prompt embedding matrix does not match the manifest")
    if _sequence_digest(example_ids.astype(str).tolist()) != manifest.get(
        "example_id_sha256"
    ):
        raise ValueError("Prompt embedding ID checksum failed")
    return PromptEmbeddingCache(manifest, example_ids.astype(str), embeddings)


def validate_prompt_embedding_source(
    embedding_cache: PromptEmbeddingCache, source_cache: RoutingCache
) -> None:
    """Verify that a sidecar was built from exactly this cache's prompt inputs."""
    source_ids = [str(record["id"]) for record in source_cache.records]
    source_texts = [build_semantic_prompt(record) for record in source_cache.records]
    if embedding_cache.manifest.get("source_cache_schema_version") != (
        source_cache.manifest.get("schema_version")
    ):
        raise ValueError("Prompt embeddings were created from an incompatible cache")
    if embedding_cache.manifest.get("outcome_or_answer_features_used") is not False:
        raise ValueError("Prompt embedding provenance does not exclude answer features")
    if _sequence_digest(source_ids) != embedding_cache.manifest.get(
        "example_id_sha256"
    ):
        raise ValueError("Prompt embeddings do not match the source example IDs")
    if _sequence_digest(source_texts) != embedding_cache.manifest.get(
        "prompt_text_sha256"
    ):
        raise ValueError("Prompt embeddings do not match the source prompt text")


def encode_existing_cache(
    cache_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    device: str = "cpu",
) -> Path:
    """Download/load a frozen encoder only when this command is explicitly run."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional embedding dependency with "
            "python -m pip install -e '.[embedding]'"
        ) from exc

    cache = load_cache(cache_path)
    texts = [build_semantic_prompt(record) for record in cache.records]
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return save_prompt_embedding_cache(
        output_path,
        source_cache=cache,
        embedding_model=model_name,
        texts=texts,
        embeddings=embeddings,
        normalized=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Encode question-and-choice text already stored in an LLM routing cache."
        )
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("prompt-embeddings.zip")
    )
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = encode_existing_cache(
        args.cache,
        args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Saved prompt embeddings: {result.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
