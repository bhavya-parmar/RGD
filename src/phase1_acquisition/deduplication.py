"""
Corpus Deduplication using MinHash and Locality Sensitive Hashing (LSH).
Identifies near-identical preprints, duplicates, and republished works based on
character/word shingle overlap in titles and abstracts.
"""

import re
import logging
from typing import List, Set, Tuple, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False
    logger.warning("datasketch library not found. Falling back to built-in MinHash implementation.")


def create_shingles(text: str, k: int = 3) -> Set[str]:
    """
    Generate word or character k-shingles from text.
    Uses word-level k-shingles by default.
    """
    clean_text = re.sub(r"[^\w\s]", "", text.lower())
    words = clean_text.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
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
        shingle_size: int = 3
    ):
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size

    def deduplicate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Deduplicate the given DataFrame containing paper records.
        Returns (deduplicated_df, count_of_duplicates_removed).
        """
        if df.empty:
            return df, 0

        initial_count = len(df)
        logger.info(f"Starting MinHash/LSH deduplication on {initial_count} records (Threshold: {self.threshold})...")

        # 1. Exact match deduplication on ID and Title
        df = df.drop_duplicates(subset=["id"]).copy()
        df["normalized_title"] = df["title"].astype(str).str.strip().str.lower()
        df = df.drop_duplicates(subset=["normalized_title"]).copy()

        if HAS_DATASKETCH:
            cleaned_df = self._deduplicate_with_datasketch(df)
        else:
            cleaned_df = self._deduplicate_with_fallback(df)

        removed = initial_count - len(cleaned_df)
        logger.info(f"Deduplication complete: {len(cleaned_df)} retained, {removed} removed.")
        return cleaned_df.reset_index(drop=True), removed

    def _deduplicate_with_datasketch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use datasketch MinHashLSH for near-duplicate indexing and query."""
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        minhashes: Dict[str, MinHash] = {}

        for idx, row in df.iterrows():
            text = f"{row.get('title', '')} {row.get('abstract', '')}"
            shingles = create_shingles(text, k=self.shingle_size)
            m = MinHash(num_perm=self.num_perm)
            for s in shingles:
                m.update(s.encode("utf8"))
            minhashes[str(row["id"])] = m
            lsh.insert(str(row["id"]), m)

        to_remove = set()
        for paper_id, m in minhashes.items():
            if paper_id in to_remove:
                continue
            matches = lsh.query(m)
            # If multiple matching papers found, retain the best one (highest citations)
            if len(matches) > 1:
                candidate_rows = df[df["id"].isin(matches)]
                # Sort descending by cited_by_count and non-empty abstract length
                best_id = candidate_rows.sort_values(
                    by=["cited_by_count"], ascending=False
                ).iloc[0]["id"]

                for dup_id in matches:
                    if dup_id != best_id:
                        to_remove.add(dup_id)

        return df[~df["id"].isin(to_remove)].copy()

    def _deduplicate_with_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pure Python Jaccard similarity fallback if datasketch is not installed.
        """
        records = df.to_dict(orient="records")
        kept = []
        to_skip = set()

        # Build shingles
        shingle_map = {}
        for r in records:
            pid = r["id"]
            text = f"{r.get('title', '')} {r.get('abstract', '')}"
            shingle_map[pid] = create_shingles(text, k=self.shingle_size)

        for i, r1 in enumerate(records):
            pid1 = r1["id"]
            if pid1 in to_skip:
                continue
            s1 = shingle_map[pid1]
            if not s1:
                kept.append(r1)
                continue

            for j in range(i + 1, len(records)):
                pid2 = records[j]["id"]
                if pid2 in to_skip:
                    continue
                s2 = shingle_map[pid2]
                if not s2:
                    continue
                intersection = len(s1.intersection(s2))
                union = len(s1.union(s2))
                jaccard = intersection / union if union > 0 else 0.0

                if jaccard >= self.threshold:
                    # Choose duplicate to drop
                    if r1.get("cited_by_count", 0) >= records[j].get("cited_by_count", 0):
                        to_skip.add(pid2)
                    else:
                        to_skip.add(pid1)
                        break

            if pid1 not in to_skip:
                kept.append(r1)

        return pd.DataFrame(kept)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    test_data = [
        {"id": "W1", "title": "Physics Informed Neural Networks: A Deep Learning Framework", "abstract": "We introduce physics informed neural networks for solving forward and inverse problems.", "cited_by_count": 500},
        {"id": "W2", "title": "Physics-Informed Neural Networks: Deep Learning Framework", "abstract": "We introduce physics informed neural networks for solving forward and inverse problems.", "cited_by_count": 12},  # Near-duplicate
        {"id": "W3", "title": "Graph Convolutional Networks for Semi-Supervised Classification", "abstract": "We present a scalable approach for semi-supervised learning on graph-structured data.", "cited_by_count": 800},
    ]
    df_raw = pd.DataFrame(test_data)
    dedup = CorpusDeduplicator(threshold=0.80)
    df_clean, removed_count = dedup.deduplicate(df_raw)
    print(f"Original: {len(df_raw)} records -> Clean: {len(df_clean)} records (Removed: {removed_count})")
    print(df_clean[["id", "title", "cited_by_count"]])
