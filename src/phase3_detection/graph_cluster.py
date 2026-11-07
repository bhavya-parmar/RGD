"""
Citation Community Detection using Spectral Clustering and Louvain Modularity.
Partitions the citation network into structural communities reflecting citation silos,
schools of thought, and historical reference traditions.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import SpectralClustering

logger = logging.getLogger(__name__)

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    logger.warning("python-louvain not found. Will use NetworkX greedy modularity or Spectral Clustering.")


class CitationCommunityDetector:
    """
    Detects citation communities using Spectral Clustering or Modularity-based Partitioning.
    """

    def __init__(
        self,
        method: str = "spectral",
        n_clusters: int = 8,
        louvain_resolution: float = 1.0,
        random_state: int = 42
    ):
        self.method = method.lower()
        self.n_clusters = n_clusters
        self.louvain_resolution = louvain_resolution
        self.random_state = random_state
        self.community_map: Dict[str, int] = {}
        self.modularity: float = 0.0

    def fit_predict(self, graph: nx.Graph) -> Dict[str, int]:
        """
        Cluster nodes into citation communities.
        Supports both Spectral Clustering and Louvain / Greedy Modularity.
        """
        nodes = list(graph.nodes())
        n_nodes = len(nodes)

        if n_nodes < 2:
            self.community_map = {n: 0 for n in nodes}
            return self.community_map

        # Work on undirected view for symmetric community partitioning
        undirected_g = graph.to_undirected()

        if self.method == "louvain" and HAS_LOUVAIN and undirected_g.number_of_edges() > 0:
            logger.info("Detecting citation communities via Louvain modularity...")
            self.community_map = community_louvain.best_partition(
                undirected_g,
                resolution=self.louvain_resolution,
                random_state=self.random_state
            )
            self.modularity = community_louvain.modularity(self.community_map, undirected_g)
        elif self.method == "spectral" and undirected_g.number_of_edges() > 0:
            logger.info(f"Detecting citation communities via Spectral Clustering (k={self.n_clusters})...")
            k = min(self.n_clusters, n_nodes)
            adj_matrix = nx.to_numpy_array(undirected_g, nodelist=nodes)

            # Add self-loops to prevent zero-degree rows in Laplacian
            adj_matrix += np.eye(n_nodes) * 0.1

            sc = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=self.random_state
            )
            labels = sc.fit_predict(adj_matrix)
            self.community_map = {nodes[i]: int(labels[i]) for i in range(n_nodes)}
            self._compute_modularity(undirected_g)
        else:
            # Fallback: NetworkX greedy modularity or connected components
            logger.info("Detecting citation communities via NetworkX modularity communities...")
            try:
                communities = list(nx.algorithms.community.greedy_modularity_communities(undirected_g))
                for c_id, comm in enumerate(communities):
                    for n in comm:
                        self.community_map[n] = c_id
                self._compute_modularity(undirected_g)
            except Exception as e:
                logger.warning(f"Greedy modularity partition failed ({e}), grouping by connected components.")
                for c_id, comp in enumerate(nx.connected_components(undirected_g)):
                    for n in comp:
                        self.community_map[n] = c_id

        num_comm = len(set(self.community_map.values()))
        logger.info(
            f"Detected {num_comm} citation communities (Modularity: {self.modularity:.4f})."
        )
        return self.community_map

    def _compute_modularity(self, g: nx.Graph):
        """Calculate Newman modularity of the partition."""
        try:
            # Group nodes into communities list
            communities_dict: Dict[int, set] = {}
            for node, cid in self.community_map.items():
                communities_dict.setdefault(cid, set()).add(node)
            self.modularity = float(nx.algorithms.community.modularity(g, communities_dict.values()))
        except Exception:
            self.modularity = 0.0

    def assign_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'citation_community' column to paper metadata DataFrame."""
        df = df.copy()
        df["citation_community"] = df["id"].astype(str).map(self.community_map).fillna(-1).astype(int)
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    g = nx.erdos_renyi_graph(20, 0.2, seed=42)
    nx.relabel_nodes(g, {i: f"paper_{i}" for i in range(20)}, copy=False)
    detector = CitationCommunityDetector(method="spectral", n_clusters=3)
    comm = detector.fit_predict(g)
    print("Citation Communities Sample:", list(comm.items())[:5])
    print("Modularity:", detector.modularity)
