"""
Semantic Similarity Index construction and querying using FAISS.
Enables sub-millisecond nearest neighbor search and pairwise cosine similarity
computation for large scientific corpora.
"""

import os
import logging
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS library not installed. Falling back to NumPy inner product similarity.")


class SemanticIndex:
    """
    Wrapper around FAISS index for normalized paper vectors.
    Computes top-k nearest semantic neighbors and dense similarity matrices.
    """

    def __init__(self, dimension: int, metric: str = "inner_product"):
        self.dimension = dimension
        self.metric = metric
        self.index = None
        self.paper_ids: List[str] = []
        self._vectors_cache: Optional[np.ndarray] = None
        self._build_index()

    def _build_index(self):
        """Initialize the FAISS index structure."""
        if HAS_FAISS:
            # For normalized embeddings, inner product equals cosine similarity
            if self.metric == "inner_product":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = None

    def add(self, embeddings: np.ndarray, paper_ids: List[str]):
        """Add embedding vectors and track corresponding paper IDs."""
        assert embeddings.shape[1] == self.dimension, (
            f"Expected dim {self.dimension}, got {embeddings.shape[1]}"
        )
        assert len(paper_ids) == embeddings.shape[0], "Length of paper_ids must match rows in embeddings"

        self.paper_ids = list(paper_ids)
        self._vectors_cache = embeddings.astype(np.float32)

        if HAS_FAISS:
            self.index.add(self._vectors_cache)
            logger.info(f"Added {len(paper_ids)} vectors to FAISS index.")
        else:
            logger.info(f"Cached {len(paper_ids)} vectors for NumPy similarity search.")

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

        query_vectors = query_vectors.astype(np.float32)

        if HAS_FAISS:
            scores, indices = self.index.search(query_vectors, k)
        else:
            # Fallback pure NumPy dot product
            scores_matrix = np.dot(query_vectors, self._vectors_cache.T)
            # Top k indices
            indices = np.argsort(-scores_matrix, axis=1)[:, :k]
            scores = np.take_along_axis(scores_matrix, indices, axis=1)

        result_ids = []
        for row in indices:
            matched = [self.paper_ids[idx] for idx in row if 0 <= idx < len(self.paper_ids)]
            result_ids.append(matched)

        return scores, result_ids

    def get_pairwise_similarity_matrix(self) -> np.ndarray:
        """
        Compute full N x N semantic similarity matrix.
        """
        if self._vectors_cache is None or len(self._vectors_cache) == 0:
            return np.empty((0, 0))
        return np.dot(self._vectors_cache, self._vectors_cache.T)

    def save(self, filepath: str):
        """Save FAISS index and paper ID mapping."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if HAS_FAISS and self.index:
            faiss.write_index(self.index, filepath)
        # Save ID list alongside index
        id_path = filepath + ".ids"
        with open(id_path, "w", encoding="utf-8") as f:
            for pid in self.paper_ids:
                f.write(f"{pid}\n")
        logger.info(f"Saved semantic index to {filepath}")

    def load(self, filepath: str):
        """Load index and paper IDs from disk."""
        if HAS_FAISS:
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
