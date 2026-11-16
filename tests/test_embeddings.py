"""Content-addressed cache and model-file integrity, using a fake encoder."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evidenceforge.dataset import Document
from evidenceforge.embeddings import MODEL_FILES, MODEL_NAME, MODEL_REPO, MODEL_REVISION, corpus_vectors, prepare_model, verify_model
from evidenceforge.io import sha256_file, write_json

try:
    import numpy as np
except ImportError:
    np = None


class FakeEncoder:
    def __init__(self):
        self.manifest = {"name": "fake-for-unit-tests", "dimensions": 384, "revision": "fixture-v1"}
        self.calls = []
        self.config = "one"

    def identity(self):
        return {"model": dict(self.manifest), "config": self.config}

    def encode(self, texts, *, query=False):
        self.calls.append((list(texts), query))
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        for index in range(len(texts)):
            matrix[index, index % 384] = 1
        return matrix


class ModelIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        for name in MODEL_FILES:
            (self.path / name).write_bytes(("fixture bytes for " + name).encode())
        self.manifest = {"name": MODEL_NAME, "repository": MODEL_REPO, "revision": MODEL_REVISION,
                         "files": {name: sha256_file(self.path / name) for name in MODEL_FILES}}
        write_json(self.path / "manifest.json", self.manifest)

    def test_existing_verified_model_does_not_download(self):
        with patch("evidenceforge.embeddings.download") as network:
            self.assertEqual(prepare_model(self.path), self.manifest)
            network.assert_not_called()

    def test_modified_model_file_is_rejected(self):
        (self.path / MODEL_FILES[0]).write_bytes(b"changed file")
        with self.assertRaisesRegex(ValueError, "Model file changed"):
            verify_model(self.path)

    def test_model_identity_is_pinned(self):
        self.manifest["revision"] = "another-revision"
        write_json(self.path / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "Unexpected model identity"):
            verify_model(self.path)


@unittest.skipIf(np is None, "NumPy is an optional dense-retrieval dependency")
class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.index = Path(self.temp.name) / "index"
        self.documents = [Document("a", "Title", "First body"), Document("b", "Other", "Second body")]
        self.encoder = FakeEncoder()

    def test_cache_hit_avoids_encoding_and_preserves_matrix(self):
        first, metadata = corpus_vectors(self.documents, self.encoder, self.index)
        second, cached = corpus_vectors(self.documents, self.encoder, self.index)
        self.assertFalse(metadata["cache_hit"])
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(len(self.encoder.calls), 1)
        self.assertEqual(self.encoder.calls[0], (["Title\nFirst body", "Other\nSecond body"], False))
        np.testing.assert_array_equal(first, second)

    def test_content_order_and_encoder_config_changes_invalidate_cache(self):
        corpus_vectors(self.documents, self.encoder, self.index)
        changed = [Document("a", "Title", "Changed text"), self.documents[1]]
        for docs in (changed, list(reversed(self.documents))):
            with self.subTest(docs=docs), self.assertRaisesRegex(ValueError, "Stale index"):
                corpus_vectors(docs, self.encoder, self.index)
        self.encoder.config = "changed"
        with self.assertRaisesRegex(ValueError, "Stale index"):
            corpus_vectors(self.documents, self.encoder, self.index)
        self.assertEqual(len(self.encoder.calls), 1)

    def test_vector_file_corruption_is_detected_before_loading(self):
        corpus_vectors(self.documents, self.encoder, self.index)
        matrix_path = self.index / "vectors.npy"
        matrix_path.write_bytes(matrix_path.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            corpus_vectors(self.documents, self.encoder, self.index)

    def test_new_encoder_invalid_matrices_are_rejected_before_persistence(self):
        invalid = [np.ones((2, 3)), np.zeros((2, 384)), np.full((2, 384), np.nan), np.full((2, 384), np.inf)]
        for index, matrix in enumerate(invalid):
            destination = self.index / str(index)
            with self.subTest(shape=matrix.shape), patch.object(self.encoder, "encode", return_value=matrix):
                with self.assertRaises(ValueError):
                    corpus_vectors(self.documents, self.encoder, destination)
            self.assertFalse((destination / "manifest.json").exists())
            self.assertFalse((destination / "vectors.npy").exists())

    def test_cached_matrix_shape_and_values_are_checked_even_with_matching_checksum(self):
        corpus_vectors(self.documents, self.encoder, self.index)
        matrix_path = self.index / "vectors.npy"
        manifest_path = self.index / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Matching hashes alone prove byte integrity, not semantic validity.
        for matrix in (np.ones((2, 5)), np.zeros((2, 384)), np.full((2, 384), np.nan)):
            np.save(matrix_path, matrix, allow_pickle=False)
            manifest["vectors_sha256"] = sha256_file(matrix_path)
            write_json(manifest_path, manifest)
            with self.subTest(shape=matrix.shape), self.assertRaises(ValueError):
                corpus_vectors(self.documents, self.encoder, self.index)


if __name__ == "__main__":
    unittest.main()
