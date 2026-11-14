"""
Citation Community Detection using Louvain Modularity.
Partitions the citation network into structural communities reflecting citation silos,
schools of thought, and historical reference traditions.
"""

import logging
from typing import Dict
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


class CitationCommunityDetector:
    """
    Detects citation communities with Louvain modularity optimisation.
    The number of communities is chosen by the algorithm, not fixed in advance.
    """

    def __init__(self, resolution: float = 1.0, random_state: int = 42):
        """
        resolution: > 1 gives more, smaller communities; < 1 gives fewer, larger ones.
        """
        self.resolution = resolution
        self.random_state = random_state
        self.community_map: Dict[str, int] = {}
        self.modularity: float = 0.0

    def fit_predict(self, graph: nx.Graph) -> Dict[str, int]:
        """Cluster nodes into citation communities. Returns {paper_id: community_id}."""
        nodes = list(graph.nodes())
        if len(nodes) < 2:
            self.community_map = {n: 0 for n in nodes}
            return self.community_map

        # Citation direction doesn't matter for "who is connected to whom"
        undirected_g = graph.to_undirected()
        if undirected_g.number_of_edges() == 0:
            raise ValueError(
                "Citation graph has no edges between corpus papers; citation communities "
                "cannot be detected. Use a larger or more connected corpus."
            )

        # Louvain: greedily merges nodes into communities to maximise modularity
        communities = nx.community.louvain_communities(
            undirected_g, resolution=self.resolution, seed=self.random_state
        )
        self.community_map = {n: cid for cid, comm in enumerate(communities) for n in comm}
        # Newman modularity: how much denser links are inside communities than expected by chance
        self.modularity = float(nx.community.modularity(undirected_g, communities))

        logger.info(
            f"Detected {len(communities)} citation communities via Louvain "
            f"(Modularity: {self.modularity:.4f})."
        )
        return self.community_map

    def assign_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'citation_community' column to paper metadata DataFrame."""
        df = df.copy()
        df["citation_community"] = df["id"].astype(str).map(self.community_map).fillna(-1).astype(int)
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    g = nx.erdos_renyi_graph(20, 0.2, seed=42)
    nx.relabel_nodes(g, {i: f"paper_{i}" for i in range(20)}, copy=False)
    detector = CitationCommunityDetector()
    comm = detector.fit_predict(g)
    print("Citation Communities Sample:", list(comm.items())[:5])
    print("Modularity:", round(detector.modularity, 4))
