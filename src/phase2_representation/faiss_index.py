"""
Semantic Similarity Index construction and querying using FAISS.
Enables fast nearest neighbor search (cosine similarity on normalized vectors)
for large scientific corpora.
"""

import os
import logging
from typing import List, Tuple
import faiss
import numpy as np

logger = logging.getLogger(__name__)


class SemanticIndex:
    """
    Wrapper around FAISS index for normalized paper vectors.
    Computes top-k nearest semantic neighbors.
    """

    def __init__(self, dimension: int, metric: str = "inner_product"):
        self.dimension = dimension
        self.metric = metric
        self.index = None
        self.paper_ids: List[str] = []
        # Exact (brute-force) search; for normalized embeddings, inner product = cosine similarity
        if metric == "inner_product":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = faiss.IndexFlatL2(dimension)

    def add(self, embeddings: np.ndarray, paper_ids: List[str]):
        """Add embedding vectors and track corresponding paper IDs."""
        assert embeddings.shape[1] == self.dimension, (
            f"Expected dim {self.dimension}, got {embeddings.shape[1]}"
        )
        assert len(paper_ids) == embeddings.shape[0], "Length of paper_ids must match rows in embeddings"

        self.paper_ids = list(paper_ids)
        # FAISS needs a C-contiguous float32 array
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        logger.info(f"Added {len(paper_ids)} vectors to FAISS index.")

    def search(
        self,
        query_vectors: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Query the index for k nearest neighbors.
        Returns:
            scores: array of shape (N_queries, k)
            neighbor_ids: list of lists with paper IDs for each query
        """
        k = min(k, len(self.paper_ids))
        if k == 0:
            return np.empty((len(query_vectors), 0)), [[] for _ in query_vectors]

        query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
        scores, indices = self.index.search(query_vectors, k)

        # FAISS returns -1 for missing neighbors; map row indices back to paper IDs
        result_ids = [
            [self.paper_ids[idx] for idx in row if 0 <= idx < len(self.paper_ids)]
            for row in indices
        ]

        return scores, result_ids

    def range_search(
        self,
        query_vectors: np.ndarray,
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return every indexed vector with inner product > threshold for each query.
        Results for query i are sims[lims[i]:lims[i+1]] and rows[lims[i]:lims[i+1]].
        Only meaningful for the inner_product metric (higher = more similar).
        """
        assert self.metric == "inner_product", "range_search with a similarity threshold needs inner_product"
        query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
        lims, sims, rows = self.index.range_search(query_vectors, float(threshold))
        return lims, sims, rows

    def save(self, filepath: str):
        """Save FAISS index and paper ID mapping."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        faiss.write_index(self.index, filepath)
        # Save ID list alongside index
        id_path = filepath + ".ids"
        with open(id_path, "w", encoding="utf-8") as f:
            for pid in self.paper_ids:
                f.write(f"{pid}\n")
        logger.info(f"Saved semantic index to {filepath}")

    def load(self, filepath: str):
        """Load index and paper IDs from disk."""
        self.index = faiss.read_index(filepath)
        self.dimension = self.index.d

        id_path = filepath + ".ids"
        if os.path.exists(id_path):
            with open(id_path, "r", encoding="utf-8") as f:
                self.paper_ids = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded semantic index with {len(self.paper_ids)} vectors from {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    dim = 4
    np.random.seed(42)
    sample_vecs = np.random.randn(5, dim).astype(np.float32)
    sample_vecs /= np.linalg.norm(sample_vecs, axis=1, keepdims=True)
    ids = [f"paper_{i}" for i in range(5)]

    index = SemanticIndex(dimension=dim)
    index.add(sample_vecs, ids)
    scores, top_ids = index.search(sample_vecs[:2], k=3)
    print("Top Neighbor IDs:", top_ids)
    print("Scores:", scores)
