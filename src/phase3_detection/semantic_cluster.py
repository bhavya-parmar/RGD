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

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings into n_clusters semantic communities (fixed k).
        """
        n_samples = len(embeddings)
        if n_samples < 2:
            self.labels = np.zeros(n_samples, dtype=int)
            return self.labels

        # k can't exceed the number of papers
        target_k = min(self.n_clusters, n_samples)

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
        Appends the 'semantic_community' column.
        """
        # Run k-means first if it hasn't been fitted yet
        if self.labels is None:
            self.fit_predict(embeddings)

        # Row i of df gets the cluster label of embedding row i
        df = df.copy()
        df["semantic_community"] = self.labels
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42)
    dummy_emb = np.random.randn(20, 16)
    detector = SemanticCommunityDetector(n_clusters=3)
    labels = detector.fit_predict(dummy_emb)
    print("Semantic Community Labels:", labels)
    print("Silhouette Score:", detector.silhouette)
