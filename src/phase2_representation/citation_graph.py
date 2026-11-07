"""
Citation Graph Construction and Network Centrality Analysis.
Builds directed citation networks, executes PageRank to rank seminal influence,
and computes HITS Hub and Authority scores to distinguish surveys from foundational works.
"""

import logging
from typing import Dict, List, Any, Tuple
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class CitationGraph:
    """
    Constructs and analyzes a directed citation graph where an edge (u -> v)
    denotes that paper u cites paper v.
    """

    def __init__(self, directed: bool = True):
        self.is_directed = directed
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self.pagerank_scores: Dict[str, float] = {}
        self.hub_scores: Dict[str, float] = {}
        self.authority_scores: Dict[str, float] = {}

    def build_from_dataframe(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Populate nodes and citation edges from a DataFrame containing
        'id', 'title', 'cited_by_count', and 'referenced_works'.
        """
        self.graph.clear()
        known_ids = set(df["id"].astype(str))

        # Add all nodes with attributes
        for _, row in df.iterrows():
            pid = str(row["id"])
            self.graph.add_node(
                pid,
                title=row.get("title", "Untitled"),
                year=row.get("publication_year"),
                cited_by_count=row.get("cited_by_count", 0),
                in_corpus=True
            )

        # Add directed edges: citing_paper -> referenced_paper
        edge_count = 0
        for _, row in df.iterrows():
            source_id = str(row["id"])
            refs = row.get("referenced_works", [])
            if isinstance(refs, (list, tuple, set)):
                for ref in refs:
                    target_id = str(ref)
                    # We add edge if target is in our corpus or as external reference node
                    if target_id in known_ids:
                        self.graph.add_edge(source_id, target_id)
                        edge_count += 1

        logger.info(
            f"Built citation graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges (intra-corpus citations: {edge_count})."
        )
        return self.graph

    def compute_pagerank(self, alpha: float = 0.85, max_iter: int = 100) -> Dict[str, float]:
        """
        Compute PageRank scores on the directed citation graph.
        Reflects structural influence and mitigates historical citation bias.
        """
        if self.graph.number_of_nodes() == 0:
            return {}

        try:
            self.pagerank_scores = nx.pagerank(self.graph, alpha=alpha, max_iter=max_iter)
        except Exception as e:
            logger.warning(f"PageRank convergence error ({e}), computing on undirected projection...")
            self.pagerank_scores = nx.pagerank(self.graph.to_undirected(), alpha=alpha)

        nx.set_node_attributes(self.graph, self.pagerank_scores, "pagerank")
        return self.pagerank_scores

    def compute_hits(self, max_iter: int = 100) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute HITS (Hyperlink-Induced Topic Search) Hubs and Authorities scores.
        - Authority: Highly cited, fundamental seminal paper.
        - Hub: Literature survey or synthesis citing prominent authorities.
        """
        if self.graph.number_of_nodes() == 0 or self.graph.number_of_edges() == 0:
            zeros = {n: 0.0 for n in self.graph.nodes}
            self.hub_scores, self.authority_scores = zeros, zeros
            return zeros, zeros

        try:
            self.hub_scores, self.authority_scores = nx.hits(self.graph, max_iter=max_iter, normalized=True)
        except Exception as e:
            logger.warning(f"HITS computation fallback: {e}")
            degree = dict(self.graph.degree())
            total = sum(degree.values()) or 1
            norm_deg = {k: v / total for k, v in degree.items()}
            self.hub_scores = norm_deg
            self.authority_scores = norm_deg

        nx.set_node_attributes(self.graph, self.hub_scores, "hub_score")
        nx.set_node_attributes(self.graph, self.authority_scores, "authority_score")
        return self.hub_scores, self.authority_scores

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Export node graph metrics into a unified Pandas DataFrame."""
        rows = []
        for node, data in self.graph.nodes(data=True):
            in_deg = self.graph.in_degree(node) if self.graph.is_directed() else self.graph.degree(node)
            out_deg = self.graph.out_degree(node) if self.graph.is_directed() else self.graph.degree(node)
            rows.append({
                "id": node,
                "title": data.get("title", ""),
                "year": data.get("year"),
                "in_degree": in_deg,
                "out_degree": out_deg,
                "pagerank": self.pagerank_scores.get(node, 0.0),
                "hub_score": self.hub_scores.get(node, 0.0),
                "authority_score": self.authority_scores.get(node, 0.0),
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    mock_papers = pd.DataFrame([
        {"id": "p1", "title": "Paper 1 (Foundational)", "publication_year": 2017, "cited_by_count": 500, "referenced_works": []},
        {"id": "p2", "title": "Paper 2 (Survey Hub)", "publication_year": 2020, "cited_by_count": 120, "referenced_works": ["p1"]},
        {"id": "p3", "title": "Paper 3 (Extension)", "publication_year": 2022, "cited_by_count": 40, "referenced_works": ["p1", "p2"]}
    ])
    cg = CitationGraph()
    cg.build_from_dataframe(mock_papers)
    pr = cg.compute_pagerank()
    hubs, auths = cg.compute_hits()
    summary = cg.get_summary_dataframe()
    print("Graph Centrality Summary:")
    print(summary[["id", "title", "in_degree", "pagerank", "authority_score", "hub_score"]])
