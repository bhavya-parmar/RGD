"""
Frontier Scoring and Discrepancy Analysis.
Identifies overlooked cross-disciplinary frontiers and isolated academic silos
by isolating community and paper pairs exhibiting High Semantic Similarity but Low Citation Connectivity.
"""

import logging
from typing import Set
import numpy as np
import pandas as pd
import networkx as nx

from src.phase2_representation.faiss_index import SemanticIndex

logger = logging.getLogger(__name__)


class FrontierScorer:
    """
    Computes discrepancy scores between semantic similarity and citation network density
    to discover candidate research frontiers.

    Cutoffs are relative, not fixed, so they adapt to the corpus: paper similarity
    uses a percentile of observed similarities, and "low citation" compares a pair's
    density with the average density of the whole citation graph.
    """

    def __init__(
        self,
        similarity_percentile: float = 95.0,
        max_density_ratio: float = 1.0,
        min_community_size: int = 3,
        max_sampled_pairs: int = 1_000_000,
        random_state: int = 42
    ):
        """
        similarity_percentile: a paper pair counts as "highly similar" if it is in the
            top (100 - similarity_percentile)% of similarities, e.g. 95 -> top 5%.
        max_density_ratio: a community pair is "low citation" if its citation density
            is at most this multiple of the whole graph's density (1.0 = below average).
        max_sampled_pairs: above this many paper pairs, the paper-level percentile is
            estimated from a random sample instead of all pairs.
        """
        self.similarity_percentile = similarity_percentile
        self.max_density_ratio = max_density_ratio
        self.min_community_size = min_community_size
        self.max_sampled_pairs = max_sampled_pairs
        self.random_state = random_state


    @staticmethod
    def community_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """
        Mean cosine similarity over all |A| x |B| paper pairs.
        For unit vectors, mean_ab(a . b) = mean(a) . mean(b), so this costs one
        dot product instead of an |A| x |B| matrix.
        """
        return float(emb_a.mean(axis=0) @ emb_b.mean(axis=0))

    @staticmethod
    def citation_density(graph: nx.Graph, nodes_a: Set[str], nodes_b: Set[str]) -> float:
        """Fraction of possible citations between A and B that exist (both directions)."""
        possible = len(nodes_a) * len(nodes_b) * (2 if graph.is_directed() else 1)
        if possible == 0:
            return 0.0
        # cut_size counts A -> B and B -> A edges for directed graphs
        return nx.cut_size(graph, nodes_a & set(graph), nodes_b & set(graph)) / possible

    def paper_similarity_threshold(self, embeddings: np.ndarray) -> float:
        """Percentile of pairwise paper similarities (exact for small n, sampled for large n)."""
        n = len(embeddings)
        n_pairs = n * (n - 1) // 2
        if n_pairs <= self.max_sampled_pairs:
            iu = np.triu_indices(n, k=1)
            sims = (embeddings @ embeddings.T)[iu]
        else:
            rng = np.random.default_rng(self.random_state)
            i = rng.integers(0, n, self.max_sampled_pairs)
            j = rng.integers(0, n, self.max_sampled_pairs)
            keep = i != j
            sims = np.einsum("ij,ij->i", embeddings[i[keep]], embeddings[j[keep]])
        return float(np.percentile(sims, self.similarity_percentile))


    def score_community_frontiers(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        citation_graph: nx.Graph,
        community_col: str = "citation_community"
    ) -> pd.DataFrame:
        """
        Measure every pair of communities: mean semantic similarity and citation
        density relative to the whole graph. Sorted by similarity.
        All pairs are returned so that validation can correct for testing all of them.
        """
        assert community_col in df.columns, f"Column '{community_col}' missing from DataFrame."
        labels = df[community_col].to_numpy()
        ids = df["id"].astype(str).to_numpy()

        communities = [
            c for c in pd.unique(labels)
            if c != -1 and (labels == c).sum() >= self.min_community_size
        ]
        logger.info(
            f"Evaluating cross-community frontiers across {len(communities)} valid "
            f"'{community_col}' communities..."
        )

        n = citation_graph.number_of_nodes()
        possible = n * (n - 1) / (1 if citation_graph.is_directed() else 2)
        graph_density = citation_graph.number_of_edges() / possible if possible else 0.0

        members = {c: set(ids[labels == c]) for c in communities}
        embs = {c: embeddings[labels == c] for c in communities}

        results = []
        for i, c_a in enumerate(communities):
            for c_b in communities[i + 1:]:
                mean_sim = self.community_similarity(embs[c_a], embs[c_b])
                density = self.citation_density(citation_graph, members[c_a], members[c_b])
                results.append({
                    "community_a": int(c_a),
                    "community_b": int(c_b),
                    "size_a": len(members[c_a]),
                    "size_b": len(members[c_b]),
                    "mean_semantic_similarity": mean_sim,
                    "cross_citation_edges": nx.cut_size(citation_graph, members[c_a], members[c_b]),
                    "citation_density": density,
                    "density_ratio": density / graph_density if graph_density else 0.0,
                })

        frontier_df = pd.DataFrame(results)
        if frontier_df.empty:
            return frontier_df

        frontier_df["low_citation"] = frontier_df["density_ratio"] <= self.max_density_ratio
        logger.info(
            f"Scored {len(frontier_df)} community pairs (graph density {graph_density:.4f}; "
            f"{int(frontier_df['low_citation'].sum())} pairs at or below {self.max_density_ratio:g}x)."
        )
        return frontier_df.sort_values(by="mean_semantic_similarity", ascending=False).reset_index(drop=True)


    def find_isolated_paper_pairs(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        semantic_index: SemanticIndex,
        citation_graph: nx.Graph,
        top_n_pairs: int = 20
    ) -> pd.DataFrame:
        """
        Find paper pairs with high semantic similarity but no citation in either
        direction (potential disconnected silos). Uses FAISS range search, so only
        pairs above the similarity cutoff are ever examined (no n x n matrix).
        """
        ids = df["id"].astype(str).tolist()
        if len(ids) < 2:
            return pd.DataFrame()
        assert semantic_index.paper_ids == ids, "FAISS index rows must match DataFrame row order."

        threshold = self.paper_similarity_threshold(embeddings)
        lims, sims, neighbors = semantic_index.range_search(embeddings, threshold)
        logger.info(
            f"Paper similarity cutoff (p{self.similarity_percentile:g}): {threshold:.4f}; "
            f"FAISS returned {len(neighbors)} neighbor hits."
        )

        titles = df["title"].tolist()
        citations = df.get("cited_by_count", pd.Series(0, index=df.index)).tolist()

        pairs = []
        for i in range(len(ids)):
            for k in range(lims[i], lims[i + 1]):
                j = int(neighbors[k])
                if j <= i:  # skip self-match and the mirrored (j, i) duplicate
                    continue
                u, v = ids[i], ids[j]
                if citation_graph.has_edge(u, v) or citation_graph.has_edge(v, u):
                    continue
                pairs.append({
                    "paper_1_id": u,
                    "paper_1_title": titles[i],
                    "paper_1_citations": citations[i],
                    "paper_2_id": v,
                    "paper_2_title": titles[j],
                    "paper_2_citations": citations[j],
                    "semantic_similarity": float(sims[k]),
                })

        pairs_df = pd.DataFrame(pairs)
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
    mock_emb = np.random.randn(n, 8).astype(np.float32)
    mock_emb /= np.linalg.norm(mock_emb, axis=1, keepdims=True)

    g = nx.DiGraph()
    g.add_nodes_from(mock_df["id"])
    g.add_edge("p_0", "p_1")
    g.add_edge("p_5", "p_6")
    g.add_edge("p_7", "p_2")  # cross-community citation

    index = SemanticIndex(dimension=8)
    index.add(mock_emb, mock_df["id"].tolist())

    scorer = FrontierScorer(min_community_size=2)
    print("Top Community Frontiers:")
    print(scorer.score_community_frontiers(mock_df, mock_emb, g))
    print("Isolated Paper Pairs:")
    print(scorer.find_isolated_paper_pairs(mock_df, mock_emb, index, g, top_n_pairs=5))
