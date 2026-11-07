"""
Semantic Community Detection using k-means++ Clustering.
Groups scientific publications into topical clusters based on SPECTER2 / PCA embeddings
and calculates internal cohesion metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class SemanticCommunityDetector:
    """
    Detects semantic communities in literature embeddings using k-means++.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        max_iter: int = 300,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.model: Optional[KMeans] = None
        self.labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.silhouette: float = 0.0

    def fit_predict(
        self,
        embeddings: np.ndarray,
        find_optimal_k: bool = False,
        k_min: int = 2,
        k_max: int = 12
    ) -> np.ndarray:
        """
        Cluster embeddings into semantic communities.
        Optionally searches for k that maximizes the silhouette score.
        """
        n_samples = len(embeddings)
        if n_samples < 2:
            self.labels = np.zeros(n_samples, dtype=int)
            return self.labels

        target_k = min(self.n_clusters, n_samples)

        if find_optimal_k and n_samples > k_min:
            best_k = target_k
            best_score = -1.0
            search_max = min(k_max, n_samples - 1)

            for candidate_k in range(k_min, search_max + 1):
                km = KMeans(
                    n_clusters=candidate_k,
                    init=self.init,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    n_init=10
                )
                pred = km.fit_predict(embeddings)
                score = silhouette_score(embeddings, pred)
                logger.debug(f"Evaluated k={candidate_k}: Silhouette = {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_k = candidate_k

            target_k = best_k
            logger.info(f"Optimal semantic clusters selected: k={target_k} (Silhouette: {best_score:.4f})")

        self.model = KMeans(
            n_clusters=target_k,
            init=self.init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init=10
        )
        self.labels = self.model.fit_predict(embeddings)
        self.cluster_centers = self.model.cluster_centers_

        if len(set(self.labels)) > 1:
            self.silhouette = float(silhouette_score(embeddings, self.labels))
        else:
            self.silhouette = 0.0

        logger.info(
            f"Detected {target_k} semantic communities (Overall Silhouette: {self.silhouette:.4f})."
        )
        return self.labels

    def assign_to_dataframe(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray
    ) -> pd.DataFrame:
        """
        Appends 'semantic_community' column and finds closest exemplar paper to each centroid.
        """
        if self.labels is None:
            self.fit_predict(embeddings)

        df = df.copy()
        df["semantic_community"] = self.labels

        # Find exemplar (closest to cluster centroid)
        exemplars = {}
        for cluster_id in range(self.model.n_clusters):
            cluster_indices = np.where(self.labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_vecs = embeddings[cluster_indices]
            center = self.cluster_centers[cluster_id]
            # Euclidean distance to center
            dists = np.linalg.norm(cluster_vecs - center, axis=1)
            best_idx = cluster_indices[np.argmin(dists)]
            exemplars[cluster_id] = df.iloc[best_idx].get("title", f"Paper {best_idx}")

        df["is_semantic_exemplar"] = False
        for cluster_id, title in exemplars.items():
            df.loc[(df["semantic_community"] == cluster_id) & (df["title"] == title), "is_semantic_exemplar"] = True

        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42)
    dummy_emb = np.random.randn(20, 16)
    detector = SemanticCommunityDetector(n_clusters=3)
    labels = detector.fit_predict(dummy_emb)
    print("Semantic Community Labels:", labels)
    print("Silhouette Score:", detector.silhouette)
