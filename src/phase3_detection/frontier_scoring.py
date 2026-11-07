"""
Frontier Scoring and Discrepancy Analysis.
Identifies overlooked cross-disciplinary frontiers and isolated academic silos
by isolating community and paper pairs exhibiting High Semantic Similarity but Low Citation Connectivity.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


class FrontierScorer:
    """
    Computes discrepancy scores between semantic similarity and citation network density
    to discover candidate research frontiers.
    """

    def __init__(
        self,
        semantic_sim_threshold: float = 0.65,
        citation_connectivity_threshold: float = 0.10,
        min_community_size: int = 3,
        top_k_frontiers: int = 10
    ):
        self.semantic_sim_threshold = semantic_sim_threshold
        self.citation_connectivity_threshold = citation_connectivity_threshold
        self.min_community_size = min_community_size
        self.top_k_frontiers = top_k_frontiers

    def score_community_frontiers(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        citation_graph: nx.Graph,
        community_col: str = "citation_community"
    ) -> pd.DataFrame:
        """
        Evaluate pairs of citation communities for frontier potential.
        Frontier Score = Mean_Semantic_Sim * (1 - Citation_Edge_Density).
        """
        assert community_col in df.columns, f"Column '{community_col}' missing from DataFrame."

        unique_communities = [
            c for c in df[community_col].unique()
            if c != -1 and (df[community_col] == c).sum() >= self.min_community_size
        ]

        logger.info(f"Evaluating cross-community frontiers across {len(unique_communities)} valid communities...")

        results = []
        n_comm = len(unique_communities)

        for i in range(n_comm):
            c_a = unique_communities[i]
            idx_a = df.index[df[community_col] == c_a].tolist()
            nodes_a = set(df.loc[idx_a, "id"].astype(str))
            emb_a = embeddings[idx_a]

            for j in range(i + 1, n_comm):
                c_b = unique_communities[j]
                idx_b = df.index[df[community_col] == c_b].tolist()
                nodes_b = set(df.loc[idx_b, "id"].astype(str))
                emb_b = embeddings[idx_b]

                # 1. Semantic Similarity: Mean pairwise dot product between communities
                sim_matrix = np.dot(emb_a, emb_b.T)
                mean_semantic_sim = float(np.mean(sim_matrix))
                max_semantic_sim = float(np.max(sim_matrix))

                # 2. Citation Connectivity: Cross-edges between communities
                possible_edges = len(nodes_a) * len(nodes_b)
                if citation_graph.is_directed():
                    possible_edges *= 2

                cross_edges = 0
                for u in nodes_a:
                    if u in citation_graph:
                        neighbors = set(citation_graph.neighbors(u))
                        cross_edges += len(neighbors.intersection(nodes_b))

                citation_density = float(cross_edges / possible_edges) if possible_edges > 0 else 0.0

                # 3. Frontier Score formulation
                # High semantic similarity + low citation density = high frontier score
                frontier_score = mean_semantic_sim * (1.0 - citation_density)

                results.append({
                    "community_a": int(c_a),
                    "community_b": int(c_b),
                    "size_a": len(nodes_a),
                    "size_b": len(nodes_b),
                    "mean_semantic_similarity": mean_semantic_sim,
                    "max_semantic_similarity": max_semantic_sim,
                    "cross_citation_edges": cross_edges,
                    "citation_density": citation_density,
                    "frontier_score": frontier_score,
                    "is_candidate_frontier": (
                        mean_semantic_sim >= self.semantic_sim_threshold
                        and citation_density <= self.citation_connectivity_threshold
                    )
                })

        frontier_df = pd.DataFrame(results)
        if not frontier_df.empty:
            frontier_df = frontier_df.sort_values(by="frontier_score", ascending=False).reset_index(drop=True)

        logger.info(f"Calculated frontier scores for {len(frontier_df)} community pairs.")
        return frontier_df.head(self.top_k_frontiers)

    def find_isolated_paper_pairs(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        citation_graph: nx.Graph,
        top_n_pairs: int = 20
    ) -> pd.DataFrame:
        """
        Find individual paper pairs that have high semantic similarity
        but have ZERO citation link between each other (potential disconnected silos).
        """
        n_papers = len(df)
        if n_papers < 2:
            return pd.DataFrame()

        sim_matrix = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(sim_matrix, -1.0)  # Exclude self-similarity

        candidate_pairs = []
        ids = df["id"].astype(str).tolist()
        titles = df["title"].tolist()
        citations = df.get("cited_by_count", pd.Series(np.zeros(n_papers))).tolist()

        # Undirected view for checking direct citation existence
        undirected_g = citation_graph.to_undirected()

        for i in range(n_papers):
            for j in range(i + 1, n_papers):
                sim = sim_matrix[i, j]
                if sim >= self.semantic_sim_threshold:
                    u, v = ids[i], ids[j]
                    has_citation = False
                    if u in undirected_g and v in undirected_g:
                        has_citation = undirected_g.has_edge(u, v)

                    if not has_citation:
                        candidate_pairs.append({
                            "paper_1_id": u,
                            "paper_1_title": titles[i],
                            "paper_1_citations": citations[i],
                            "paper_2_id": v,
                            "paper_2_title": titles[j],
                            "paper_2_citations": citations[j],
                            "semantic_similarity": float(sim),
                            "citation_edge_exists": False
                        })

        pairs_df = pd.DataFrame(candidate_pairs)
        if not pairs_df.empty:
            pairs_df = pairs_df.sort_values(by="semantic_similarity", ascending=False).reset_index(drop=True)

        return pairs_df.head(top_n_pairs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42)
    n = 10
    mock_df = pd.DataFrame({
        "id": [f"p_{i}" for i in range(n)],
        "title": [f"Study on Method {i}" for i in range(n)],
        "citation_community": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "cited_by_count": np.random.randint(5, 100, size=n)
    })
    mock_emb = np.random.randn(n, 8)
    mock_emb /= np.linalg.norm(mock_emb, axis=1, keepdims=True)

    g = nx.DiGraph()
    for pid in mock_df["id"]:
        g.add_node(pid)
    # Add a few intra-community edges
    g.add_edge("p_0", "p_1")
    g.add_edge("p_5", "p_6")

    scorer = FrontierScorer(min_community_size=2)
    comm_frontiers = scorer.score_community_frontiers(mock_df, mock_emb, g)
    print("Top Community Frontiers:")
    print(comm_frontiers)
