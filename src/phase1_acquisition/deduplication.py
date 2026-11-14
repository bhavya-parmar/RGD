"""
Corpus Deduplication using MinHash and Locality Sensitive Hashing (LSH).
Identifies near-identical preprints, duplicates, and republished works based on
character/word shingle overlap in titles and abstracts.
"""

import re
import logging
from typing import List, Set, Tuple, Dict, Any, Optional
import pandas as pd
from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)



def create_shingles(text: str, k: int = 3) -> Set[str]:
    """
    Generate word or character k-shingles from text.
    Uses word-level k-shingles by default.
    
    Example:
    "self-attention (bert), 2019!"
    shingles: {"self attention bert", "attention bert 2019"}
    """
    # Lowercase and replace punctuation with spaces ("physics-informed" -> "physics informed")
    clean_text = re.sub(r"[^\w\s]", " ", text.lower())
    # Split into words on whitespace
    words = clean_text.split()
    # Text shorter than k: whole text is the only shingle (empty text: no shingles)
    if len(words) < k:
        return {" ".join(words)} if words else set()
    # Slide a k-word window one word at a time; set drops repeated windows
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


class CorpusDeduplicator:
    """
    Deduplicates research papers using MinHash signatures and LSH index.
    Prefers keeping records with non-empty abstracts and higher citation counts.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        num_perm: int = 128,
        shingle_size: int = 3,
        candidate_threshold: Optional[float] = None
    ):
        """
        threshold: exact Jaccard needed to call two papers duplicates (verification).
        candidate_threshold: where the LSH S-curve is placed. Kept below `threshold`
            so true duplicates are rarely missed; false positives are removed by
            verification. Defaults to threshold - 0.15.
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.candidate_threshold = (
            candidate_threshold if candidate_threshold is not None
            else max(0.5, threshold - 0.15)
        )

    def deduplicate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Deduplicate the given DataFrame containing paper records.
        Returns (deduplicated_df, count_of_duplicates_removed).
        """
        if df.empty:
            return df, 0

        initial_count = len(df)
        logger.info(f"Starting MinHash/LSH deduplication on {initial_count} records (Threshold: {self.threshold})...")

        # Exact match deduplication on ID and Title
        df = df.drop_duplicates(subset=["id"]).copy()
        df["normalized_title"] = df["title"].astype(str).str.strip().str.lower()
        df = df.drop_duplicates(subset=["normalized_title"]).copy()

        cleaned_df = self._deduplicate_with_datasketch(df)

        removed = initial_count - len(cleaned_df)
        logger.info(f"Deduplication complete: {len(cleaned_df)} retained, {removed} removed.")
        return cleaned_df.reset_index(drop=True), removed

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        """Exact Jaccard similarity J(A, B) = |A ∩ B| / |A ∪ B|."""
        union = len(a | b)
        return len(a & b) / union if union else 0.0

    def _deduplicate_with_datasketch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        LSH-NEAR-PAIRS pipeline (lecture 6-7):
          1. shingle each document
          2. MinHash signature of k = num_perm rows
          3. banding: documents sharing a band bucket become candidate pairs
          4. verify candidates with exact Jaccard; keep only pairs >= threshold
        """
        # Step 3 setup: datasketch picks bands b and rows r (k = r * b) so the
        # S-curve threshold (1/b)^(1/r) sits near candidate_threshold
        lsh = MinHashLSH(threshold=self.candidate_threshold, num_perm=self.num_perm)
        minhashes: Dict[str, MinHash] = {}
        shingle_sets: Dict[str, Set[str]] = {}

        for _, row in df.iterrows():
            pid = str(row["id"])
            # Step 1: shingling, document -> set of k-word windows
            text = f"{row.get('title', '')} {row.get('abstract', '')}"
            shingles = create_shingles(text, k=self.shingle_size)
            if not shingles:
                continue  # empty text: nothing to compare, always kept
            shingle_sets[pid] = shingles

            # Step 2: signature, one min-hash per random hash function
            # (stands in for a random permutation); Pr[rows agree] = J
            m = MinHash(num_perm=self.num_perm)
            for s in shingles:
                m.update(s.encode("utf8"))
            minhashes[pid] = m

            # Step 3: banding, hash each band's r-tuple into that band's bucket table
            lsh.insert(pid, m)

        # Prefer papers with an abstract, then more citations
        abstracts = df["abstract"] if "abstract" in df else pd.Series("", index=df.index)
        rank = pd.DataFrame({
            "_has_abstract": abstracts.fillna("").astype(str).str.len() > 0,
            "cited_by_count": df.get("cited_by_count", 0),
        }).set_index(df["id"].astype(str))

        to_remove: Set[str] = set()
        n_candidates = n_verified = 0
        for pid, m in minhashes.items():
            if pid in to_remove:
                continue

            # Step 3: candidates = papers sharing at least one band bucket with pid
            candidates = [c for c in lsh.query(m) if c != pid and c not in to_remove]
            n_candidates += len(candidates)

            # Step 4: verify with exact Jaccard on the shingle sets; LSH false
            # positives are dropped here instead of deleting a real paper
            duplicates = [
                c for c in candidates
                if self._jaccard(shingle_sets[pid], shingle_sets[c]) >= self.threshold
            ]
            n_verified += len(duplicates)
            if not duplicates:
                continue

            # Keep the best record of the verified group, remove the rest
            group = [pid] + duplicates
            best_id = rank.loc[group].sort_values(
                by=["_has_abstract", "cited_by_count"], ascending=False
            ).index[0]
            to_remove.update(g for g in group if g != best_id)

        logger.info(
            f"LSH: {n_candidates} candidate matches, {n_verified} verified "
            f"(J >= {self.threshold}), {n_candidates - n_verified} false positives discarded."
        )
        return df[~df["id"].astype(str).isin(to_remove)].copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    test_data = [
        {"id": "W1", "title": "Physics Informed Neural Networks: A Deep Learning Framework", "abstract": "We introduce physics informed neural networks for solving forward and inverse problems involving nonlinear partial differential equations using deep learning and automatic differentiation.", "cited_by_count": 500},
        {"id": "W2", "title": "Physics-Informed Neural Networks: Deep Learning Framework", "abstract": "We introduce physics informed neural networks for solving forward and inverse problems involving nonlinear partial differential equations using deep learning and automatic differentiation.", "cited_by_count": 12},  # Near-duplicate
        {"id": "W3", "title": "Graph Convolutional Networks for Semi-Supervised Classification", "abstract": "We present a scalable approach for semi-supervised learning on graph-structured data.", "cited_by_count": 800},
    ]
    df_raw = pd.DataFrame(test_data)
    dedup = CorpusDeduplicator(threshold=0.80)
    df_clean, removed_count = dedup.deduplicate(df_raw)
    print(f"Original: {len(df_raw)} records -> Clean: {len(df_clean)} records (Removed: {removed_count})")
    print(df_clean[["id", "title", "cited_by_count"]])
