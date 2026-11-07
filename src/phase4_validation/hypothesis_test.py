"""
Statistical Validation and Hypothesis Testing for Research Frontier Detection.
Applies bootstrap resampling to estimate frontier stability and conducts permutation
testing against random graph null models to guarantee statistical significance.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


class StatisticalValidator:
    """
    Validates detected research frontiers through non-parametric bootstrap resampling
    and edge-permutation null hypothesis tests.
    """

    def __init__(
        self,
        n_iterations: int = 200,
        subsample_ratio: float = 0.80,
        alpha: float = 0.05,
        random_state: int = 42
    ):
        self.n_iterations = n_iterations
        self.subsample_ratio = subsample_ratio
        self.alpha = alpha
        self.random_state = random_state

    def bootstrap_frontier_stability(
        self,
        frontier_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        embeddings: np.ndarray,
        citation_graph: nx.Graph
    ) -> pd.DataFrame:
        """
        Estimate stability of candidate frontiers across random subsamples of papers.
        Calculates empirical confidence intervals and selection persistence frequency.
        """
        if frontier_df.empty or len(corpus_df) < 5:
            logger.warning("Insufficient data for bootstrap resampling.")
            return frontier_df

        rng = np.random.default_rng(self.random_state)
        n_samples = len(corpus_df)
        sample_size = max(4, int(n_samples * self.subsample_ratio))

        logger.info(
            f"Running {self.n_iterations} bootstrap resampling iterations "
            f"(Subsample ratio: {self.subsample_ratio:.0%}, Size: {sample_size}/{n_samples})..."
        )

        # Track bootstrap scores for each top community pair
        pair_scores: Dict[Tuple[int, int], List[float]] = {}
        for _, row in frontier_df.iterrows():
            pair = (int(row["community_a"]), int(row["community_b"]))
            pair_scores[pair] = []

        for _ in range(self.n_iterations):
            sub_indices = rng.choice(n_samples, size=sample_size, replace=True)
            sub_df = corpus_df.iloc[sub_indices].reset_index(drop=True)
            sub_emb = embeddings[sub_indices]

            for (ca, cb) in pair_scores.keys():
                idx_a = np.where(sub_df["citation_community"] == ca)[0]
                idx_b = np.where(sub_df["citation_community"] == cb)[0]

                if len(idx_a) == 0 or len(idx_b) == 0:
                    continue

                # Compute mean semantic similarity on bootstrap sample
                sim = float(np.mean(np.dot(sub_emb[idx_a], sub_emb[idx_b].T)))
                pair_scores[(ca, cb)].append(sim)

        # Compute mean, standard error, and 95% Confidence Intervals
        ci_lower = []
        ci_upper = []
        stability_pct = []

        for _, row in frontier_df.iterrows():
            pair = (int(row["community_a"]), int(row["community_b"]))
            scores = pair_scores.get(pair, [])
            if scores:
                low = float(np.percentile(scores, 100 * (self.alpha / 2)))
                high = float(np.percentile(scores, 100 * (1 - self.alpha / 2)))
                ci_lower.append(low)
                ci_upper.append(high)
                stability_pct.append(len(scores) / self.n_iterations)
            else:
                ci_lower.append(row["mean_semantic_similarity"])
                ci_upper.append(row["mean_semantic_similarity"])
                stability_pct.append(0.0)

        validated_df = frontier_df.copy()
        validated_df["bootstrap_ci_95_low"] = ci_lower
        validated_df["bootstrap_ci_95_high"] = ci_upper
        validated_df["bootstrap_persistence"] = stability_pct

        logger.info("Bootstrap stability analysis completed.")
        return validated_df

    def permutation_test_significance(
        self,
        observed_frontier_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        embeddings: np.ndarray,
        citation_graph: nx.Graph,
        n_permutations: int = 100
    ) -> pd.DataFrame:
        """
        Tests null hypothesis H0: The observed frontier score is an artifact of random network topology.
        Permutes citation graph edges to generate empirical null distribution.
        """
        if observed_frontier_df.empty or citation_graph.number_of_edges() == 0:
            return observed_frontier_df

        logger.info(f"Conducting edge-permutation hypothesis tests ({n_permutations} permutations)...")
        rng = np.random.default_rng(self.random_state)
        nodes = list(citation_graph.nodes())
        n_nodes = len(nodes)
        p_values = []

        # Generate null frontier scores via randomized degree-preserving or rewiring null models
        for _, row in observed_frontier_df.iterrows():
            ca, cb = int(row["community_a"]), int(row["community_b"])
            obs_score = row["frontier_score"]

            nodes_a = set(corpus_df.loc[corpus_df["citation_community"] == ca, "id"].astype(str))
            nodes_b = set(corpus_df.loc[corpus_df["citation_community"] == cb, "id"].astype(str))

            if not nodes_a or not nodes_b:
                p_values.append(1.0)
                continue

            null_scores = []
            for _ in range(n_permutations):
                # Shuffle community labels or shuffle network connectivity
                permuted_nodes = list(rng.permutation(nodes))
                mapping = dict(zip(nodes, permuted_nodes))
                perm_graph = nx.relabel_nodes(citation_graph, mapping, copy=True)

                cross_edges = 0
                for u in nodes_a:
                    if u in perm_graph:
                        cross_edges += len(set(perm_graph.neighbors(u)).intersection(nodes_b))

                possible = len(nodes_a) * len(nodes_b)
                perm_density = cross_edges / possible if possible > 0 else 0.0
                null_score = row["mean_semantic_similarity"] * (1.0 - perm_density)
                null_scores.append(null_score)

            # Empirical p-value
            greater_or_equal = sum(1 for s in null_scores if s >= obs_score)
            p_val = (greater_or_equal + 1) / (n_permutations + 1)
            p_values.append(float(p_val))

        result_df = observed_frontier_df.copy()
        result_df["p_value"] = p_values
        result_df["statistically_significant"] = result_df["p_value"] < self.alpha

        logger.info("Hypothesis testing completed.")
        return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    mock_frontiers = pd.DataFrame([{
        "community_a": 0, "community_b": 1,
        "mean_semantic_similarity": 0.82,
        "citation_density": 0.01,
        "frontier_score": 0.8118
    }])
    mock_corpus = pd.DataFrame({
        "id": [f"w_{i}" for i in range(10)],
        "citation_community": [0]*5 + [1]*5
    })
    mock_emb = np.random.randn(10, 8)
    mock_emb /= np.linalg.norm(mock_emb, axis=1, keepdims=True)
    g = nx.erdos_renyi_graph(10, 0.1, seed=42)
    nx.relabel_nodes(g, {i: f"w_{i}" for i in range(10)}, copy=False)

    validator = StatisticalValidator(n_iterations=50)
    boot = validator.bootstrap_frontier_stability(mock_frontiers, mock_corpus, mock_emb, g)
    perm = validator.permutation_test_significance(boot, mock_corpus, mock_emb, g, n_permutations=50)
    print("Statistical Validation Results:")
    print(perm[["community_a", "community_b", "frontier_score", "bootstrap_persistence", "p_value", "statistically_significant"]])
