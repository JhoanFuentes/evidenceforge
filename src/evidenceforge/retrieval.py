"""Transparent sparse retrieval, exact dense search and reciprocal rank fusion."""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .dataset import Document

TOKENIZER_VERSION = "unicode-alnum-lower-v1"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


@dataclass(frozen=True)
class Hit:
    document_id: str
    score: float

    def to_dict(self) -> dict:
        return {"document_id": self.document_id, "score": self.score}


class BM25:
    """Robertson BM25 with positive Lucene-style IDF, no stemming/stopword removal."""

    def __init__(self, documents: list[Document], k1: float = 1.2, b: float = 0.75):
        if (not documents or not math.isfinite(k1) or not math.isfinite(b)
                or k1 <= 0 or not 0 <= b <= 1):
            raise ValueError("BM25 requires documents, k1>0 and b in [0,1]")
        self.ids = [doc.id for doc in documents]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate document IDs")
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)
        lengths = []
        for i, document in enumerate(documents):
            terms = tokenize(document.content)
            lengths.append(len(terms))
            for term, count in Counter(terms).items():
                self.postings[term].append((i, count))
        avg_length = sum(lengths) / len(lengths) or 1
        self.norms = [k1 * (1 - b + b * length / avg_length) for length in lengths]

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be positive")
        scores: dict[int, float] = defaultdict(float)
        for term, frequency in Counter(tokenize(query)).items():
            postings = self.postings.get(term, [])
            idf = math.log(1 + (len(self.ids) - len(postings) + 0.5) / (len(postings) + 0.5))
            for i, count in postings:
                scores[i] += frequency * idf * count * (self.k1 + 1) / (count + self.norms[i])
        ordered = sorted(scores, key=lambda i: (-scores[i], self.ids[i]))[:k]
        return [Hit(self.ids[i], scores[i]) for i in ordered]


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = 10,
                           constant: int = 60) -> list[Hit]:
    if k < 1 or constant < 1:
        raise ValueError("k and RRF constant must be positive")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for hit in ranking:
            if hit.document_id not in seen:
                seen.add(hit.document_id)
                scores[hit.document_id] += 1 / (constant + len(seen))
    return [Hit(doc_id, scores[doc_id])
            for doc_id in sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:k]]


class ExactDense:
    """Cosine similarity over unit vectors. Stable ties; no ANN approximation."""

    def __init__(self, doc_ids: list[str], vectors):
        import numpy as np
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(doc_ids) or not len(doc_ids):
            raise ValueError("Dense matrix shape does not match document IDs")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("Duplicate document IDs")
        # Accumulate in float64: finite float32 components can overflow when
        # squared, silently turning a nonzero direction into an all-zero vector.
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
        if not np.all(np.isfinite(matrix)) or not np.isfinite(norms).all() or np.any(norms < 1e-12):
            raise ValueError("Dense vectors must be finite and nonzero")
        self.ids = list(doc_ids)
        self.matrix = (matrix / norms).astype(np.float32)

    def search(self, vector, k: int = 10) -> list[Hit]:
        import numpy as np
        q = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(q.astype(np.float64))
        if (k < 1 or q.shape != (self.matrix.shape[1],) or not np.isfinite(q).all()
                or not np.isfinite(norm) or norm < 1e-12):
            raise ValueError("Invalid query vector or k")
        scores = self.matrix @ (q / norm).astype(np.float32)
        # Stable deterministic document-ID tie breaking independent of source order.
        order = np.lexsort((np.asarray(self.ids), -scores))[:k]
        return [Hit(self.ids[i], float(scores[i])) for i in order]
