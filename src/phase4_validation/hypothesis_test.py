"""
Statistical Validation of Research Frontiers.
1. Similarity confidence interval: bootstrap resampling of papers to measure how
   stable each frontier's mean semantic similarity is.
2. Permutation test: checks whether two communities are more similar in topic
   than chance.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx

from src.phase3_detection.frontier_scoring import FrontierScorer

logger = logging.getLogger(__name__)


class StatisticalValidator:
    """Adds confidence intervals and p-values to scored frontier pairs."""

    def __init__(
        self,
        n_iterations: int = 500,
        subsample_ratio: float = 0.80,
        confidence: float = 0.95,
        alpha: float = 0.05,
        random_state: int = 42
    ):
        """
        n_iterations: bootstrap samples for the similarity confidence interval.
        subsample_ratio: size of each bootstrap sample relative to the corpus.
        confidence: confidence level of the interval (0.95 -> 2.5th to 97.5th percentile).
        alpha: p-value cutoff for statistical significance.
        """
        self.n_iterations = n_iterations
        self.subsample_ratio = subsample_ratio
        self.confidence = confidence
        self.alpha = alpha
        self.random_state = random_state

    def similarity_confidence_interval(
        self,
        frontier_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        embeddings: np.ndarray,
        community_col: str = "citation_community"
    ) -> pd.DataFrame:
        """
        Confidence interval for each frontier's mean semantic similarity.
        Resamples papers with replacement, recomputes the similarity of the same
        (fixed) communities, and takes percentiles of the results. Communities are
        not re-detected, so this measures how stable the similarity is, not whether
        the frontier would be found again.
        """
        if frontier_df.empty or len(corpus_df) < 5:
            logger.warning("Insufficient data for similarity confidence intervals.")
            return frontier_df

        rng = np.random.default_rng(self.random_state)
        n_papers = len(corpus_df)
        sample_size = max(4, int(n_papers * self.subsample_ratio))
        labels = corpus_df[community_col].to_numpy()
        pairs = list(zip(frontier_df["community_a"].astype(int), frontier_df["community_b"].astype(int)))
        samples: Dict[Tuple[int, int], List[float]] = {p: [] for p in pairs}

        logger.info(f"Similarity CI: {self.n_iterations} bootstrap samples of {sample_size}/{n_papers} papers...")
        for _ in range(self.n_iterations):
            idx = rng.choice(n_papers, size=sample_size, replace=True)
            sub_labels, sub_emb = labels[idx], embeddings[idx]
            for ca, cb in pairs:
                in_a, in_b = sub_labels == ca, sub_labels == cb
                # Skip samples where one community has no papers
                if in_a.any() and in_b.any():
                    samples[(ca, cb)].append(FrontierScorer.community_similarity(sub_emb[in_a], sub_emb[in_b]))

        tail = 100 * (1 - self.confidence) / 2
        result = frontier_df.copy()
        result["similarity_ci_low"] = [np.percentile(samples[p], tail) if samples[p] else np.nan for p in pairs]
        result["similarity_ci_high"] = [np.percentile(samples[p], 100 - tail) if samples[p] else np.nan for p in pairs]
        # Fraction of bootstrap samples that contained both communities (low = unreliable CI)
        result["ci_sample_coverage"] = [len(samples[p]) / self.n_iterations for p in pairs]
        return result

    def permutation_test_significance(
        self,
        frontier_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        embeddings: np.ndarray,
        n_permutations: int = 100,
        community_col: str = "citation_community"
    ) -> pd.DataFrame:
        """
        H0: the two communities are no more similar in topic than two random groups
        of the same sizes.
        Shuffles which embedding belongs to which paper (communities and citations
        stay fixed), recomputes the mean similarity, and reports
        p = fraction of shuffled similarities >= observed similarity.
        p-values are Benjamini-Hochberg adjusted across all tested pairs, and a pair
        is a frontier if it is significant after adjustment and has low citation.
        Communities come from citations and the null from the embeddings, so the
        test is not circular.
        """
        if frontier_df.empty:
            return frontier_df

        logger.info(f"Permutation test: {n_permutations} shuffles of paper embeddings...")
        rng = np.random.default_rng(self.random_state)
        labels = corpus_df[community_col].to_numpy()
        # Shuffled paper order is shared by all pairs
        perms = [rng.permutation(len(embeddings)) for _ in range(n_permutations)]

        p_values = []
        for _, row in frontier_df.iterrows():
            in_a, in_b = labels == row["community_a"], labels == row["community_b"]
            null_sims = [
                FrontierScorer.community_similarity(embeddings[perm][in_a], embeddings[perm][in_b])
                for perm in perms
            ]
            # +1 in numerator and denominator: the observed arrangement counts as one permutation
            n_extreme = sum(s >= row["mean_semantic_similarity"] for s in null_sims)
            p_values.append((n_extreme + 1) / (n_permutations + 1))

        result = frontier_df.copy()
        result["p_value"] = p_values
        result["p_value_adjusted"] = benjamini_hochberg(np.array(p_values))
        result["statistically_significant"] = result["p_value_adjusted"] < self.alpha
        result["is_frontier"] = result["statistically_significant"] & result["low_citation"]
        return result


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg adjusted p-values (controls the false discovery rate).
    Sort p ascending; adjusted p_(i) = min over k >= i of p_(k) * m / k, capped at 1.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    scaled = p_values[order] * m / np.arange(1, m + 1)
    # Running minimum from the largest p downwards keeps adjusted p-values monotone
    adjusted_sorted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rng = np.random.default_rng(0)
    corpus = pd.DataFrame({"id": [f"w_{i}" for i in range(10)], "citation_community": [0] * 5 + [1] * 5})
    emb = rng.normal(size=(10, 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    g = nx.relabel_nodes(nx.erdos_renyi_graph(10, 0.2, seed=42, directed=True), lambda i: f"w_{i}")

    frontiers = FrontierScorer(min_community_size=2).score_community_frontiers(corpus, emb, g, "citation_community")
    validator = StatisticalValidator(n_iterations=50)
    out = validator.similarity_confidence_interval(frontiers, corpus, emb)
    out = validator.permutation_test_significance(out, corpus, emb, n_permutations=50)
    print(out[["community_a", "community_b", "mean_semantic_similarity", "similarity_ci_low",
               "similarity_ci_high", "p_value", "p_value_adjusted", "is_frontier"]])
