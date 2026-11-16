"""Pinned ONNX weights and content-addressed corpus embeddings."""

import importlib.metadata
import json
import sys
import time
from pathlib import Path

from .dataset import Document
from .io import download, fingerprint, sha256_file, write_json

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REPO = "Qdrant/bge-small-en-v1.5-onnx-Q"
MODEL_REVISION = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODEL_FILES = ("config.json", "model_optimized.onnx", "ort_config.json",
               "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt")


def prepare_model(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        return verify_model(directory)
    for name in MODEL_FILES:
        download(f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}",
                 directory / name)
    manifest = {
        "name": MODEL_NAME, "repository": MODEL_REPO, "revision": MODEL_REVISION,
        "files": {name: sha256_file(directory / name) for name in MODEL_FILES},
        "checksum_origin": "Observed after HTTPS download from an immutable HF revision",
        "license": "MIT", "dimensions": 384, "max_tokens": 512,
    }
    write_json(manifest_path, manifest)
    return manifest


def verify_model(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["name"], manifest["repository"], manifest["revision"]) != (
            MODEL_NAME, MODEL_REPO, MODEL_REVISION):
        raise ValueError("Unexpected model identity; use a new model directory")
    for name in MODEL_FILES:
        if sha256_file(directory / name) != manifest["files"][name]:
            raise ValueError(f"Model file changed: {name}")
    return manifest


class Encoder:
    def __init__(self, directory: Path, threads: int = 4, batch_size: int = 1):
        if threads < 1 or batch_size < 1:
            raise ValueError("threads and batch_size must be positive")
        from fastembed import TextEmbedding
        self.manifest = verify_model(directory)
        self.threads, self.batch_size = threads, batch_size
        self.model = TextEmbedding(
            model_name=MODEL_NAME, specific_model_path=str(directory.resolve()),
            cache_dir=str(directory.resolve()), local_files_only=True,
            providers=["CPUExecutionProvider"], threads=threads,
        )

    def encode(self, texts: list[str], *, query: bool = False):
        import numpy as np
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        if query:
            texts = [QUERY_PREFIX + text for text in texts]
        encoded = []
        for i, vector in enumerate(self.model.embed(texts, batch_size=self.batch_size), 1):
            encoded.append(vector)
            if len(texts) > 128 and (i % 128 == 0 or i == len(texts)):
                print(f"Encoded {i}/{len(texts)} documents", file=sys.stderr, flush=True)
        vectors = np.asarray(encoded, dtype=np.float32)
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or np.any(norms < 1e-12):
            raise ValueError("Model produced invalid embeddings")
        return (vectors / norms).astype(np.float32)

    def identity(self) -> dict:
        return {
            "model": self.manifest, "fastembed": importlib.metadata.version("fastembed"),
            "onnxruntime": importlib.metadata.version("onnxruntime"),
            "numpy": importlib.metadata.version("numpy"), "provider": "CPUExecutionProvider",
            "threads": self.threads, "batch_size": self.batch_size,
            "query_prefix": QUERY_PREFIX, "document_prefix": "",
            "normalization": "l2-float64-accumulation-float32-output-v1",
            "document_template": "title + newline + text, strip",
        }


def _checked_matrix(vectors, rows: int):
    import numpy as np
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (rows, 384):
        raise ValueError("Index matrix has an unexpected shape")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or np.any(norms < 1e-12):
        raise ValueError("Index vectors must be finite and nonzero")
    return matrix


def corpus_vectors(documents: list[Document], encoder: Encoder, index: Path):
    import numpy as np
    identity = {
        "encoder": encoder.identity(),
        "documents_sha256": fingerprint([[d.id, d.title, d.text] for d in documents]),
        "doc_order_sha256": fingerprint([d.id for d in documents]),
    }
    key = fingerprint(identity)
    manifest_path = index / "manifest.json"
    matrix_path = index / "vectors.npy"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != key:
            raise ValueError("Stale index: corpus/model/config changed. Choose a new --index path.")
        if sha256_file(matrix_path) != manifest["vectors_sha256"]:
            raise ValueError("Index checksum mismatch")
        matrix = _checked_matrix(np.load(matrix_path, allow_pickle=False), len(documents))
        return matrix, {**manifest, "cache_hit": True}
    index.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    matrix = _checked_matrix(encoder.encode([d.content for d in documents]), len(documents))
    duration = time.perf_counter() - start
    np.save(matrix_path, matrix, allow_pickle=False)
    manifest = {"fingerprint": key, **identity, "encoding_seconds": duration,
                "document_count": len(documents), "vectors_sha256": sha256_file(matrix_path)}
    write_json(manifest_path, manifest)
    return matrix, {**manifest, "cache_hit": False}
