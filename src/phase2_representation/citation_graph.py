"""
Citation Graph Construction.
Builds a directed citation network over the corpus, used to detect citation
communities and measure citation density between them.
"""

import logging
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
                    # Only citations between papers in the corpus become edges
                    if target_id in known_ids:
                        self.graph.add_edge(source_id, target_id)
                        edge_count += 1

        logger.info(
            f"Built citation graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges (intra-corpus citations: {edge_count})."
        )
        return self.graph

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Export per-paper degree statistics (citations made/received within the corpus)."""
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
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    mock_papers = pd.DataFrame([
        {"id": "p1", "title": "Paper 1", "publication_year": 2017, "cited_by_count": 500, "referenced_works": []},
        {"id": "p2", "title": "Paper 2", "publication_year": 2020, "cited_by_count": 120, "referenced_works": ["p1"]},
        {"id": "p3", "title": "Paper 3", "publication_year": 2022, "cited_by_count": 40, "referenced_works": ["p1", "p2"]}
    ])
    cg = CitationGraph()
    cg.build_from_dataframe(mock_papers)
    summary = cg.get_summary_dataframe()
    print("Citation Graph Summary:")
    print(summary[["id", "title", "in_degree", "out_degree"]])
