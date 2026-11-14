"""
HTML Gap Report.
Writes one self-contained HTML page with the validated community frontiers and
the isolated paper pairs.
"""

import os
import html
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STYLE = """
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 30px; }
.container { max-width: 1100px; margin: 0 auto; background: #fff; border-radius: 12px; padding: 40px; box-shadow: 0 4px 6px -1px rgba(0,0,0,.1); }
h1 { color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }
h2 { color: #334155; margin-top: 32px; }
.badge { background: #e0e7ff; color: #4338ca; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
.note { color: #64748b; font-size: 14px; }
table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }
th, td { border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }
th { background: #f1f5f9; font-weight: 600; }
tr:hover { background: #f8fafc; }
"""


def _community_labels(corpus_df: pd.DataFrame, embeddings: np.ndarray, community_col: str) -> dict:
    """
    Label each community by its most typical paper: the member closest to the
    community's mean embedding (for unit vectors, highest dot product with the mean).
    """
    labels = corpus_df[community_col].to_numpy()
    titles = corpus_df["title"].astype(str).to_numpy()
    result = {}
    for c in pd.unique(labels):
        members = np.where(labels == c)[0]
        centroid = embeddings[members].mean(axis=0)
        best = members[np.argmax(embeddings[members] @ centroid)]
        result[c] = f"C{c} · {titles[best][:70]}"
    return result


def _table(df: pd.DataFrame, empty_msg: str) -> str:
    if df.empty:
        return f'<p class="note">{empty_msg}</p>'
    return df.to_html(index=False, escape=True, float_format=lambda x: f"{x:.3f}", border=0)


def write_html_report(
    output_dir: str,
    query: str,
    corpus_df: pd.DataFrame,
    embeddings: np.ndarray,
    n_citations: int,
    frontier_df: pd.DataFrame,
    isolated_pairs_df: pd.DataFrame,
    community_col: str = "citation_community",
    filename: str = "frontier_report.html"
) -> str:
    """Write the gap report and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    labels = _community_labels(corpus_df, embeddings, community_col)

    frontiers = pd.DataFrame()
    if not frontier_df.empty:
        f = frontier_df
        frontiers = pd.DataFrame({
            "Community A": f["community_a"].map(labels),
            "Community B": f["community_b"].map(labels),
            "Similarity [CI]": [
                f"{s:.3f} [{lo:.3f}, {hi:.3f}]"
                for s, lo, hi in zip(f["mean_semantic_similarity"], f["similarity_ci_low"], f["similarity_ci_high"])
            ],
            "Cross-citations": f["cross_citation_edges"],
            "Density vs graph avg": f["density_ratio"].map(lambda r: f"{r:.2f}x"),
            "Frontier score": f["frontier_score"],
            "p (adjusted)": f["p_value_adjusted"],
            "Frontier": f["is_frontier"].map({True: "yes", False: "no"}),
        })

    pairs = pd.DataFrame()
    if not isolated_pairs_df.empty:
        pairs = isolated_pairs_df[["paper_1_title", "paper_2_title", "semantic_similarity"]].rename(
            columns={"paper_1_title": "Paper 1", "paper_2_title": "Paper 2", "semantic_similarity": "Similarity"}
        )

    n_frontiers = int(frontier_df.get("is_frontier", pd.Series(dtype=bool)).sum())
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RGD Report: {html.escape(query)}</title>
<style>{STYLE}</style>
</head>
<body>
<div class="container">
<h1>Research Gap Discovery Report</h1>
<p><strong>Query:</strong> <span class="badge">{html.escape(query)}</span>
 &nbsp; <strong>Papers:</strong> {len(corpus_df)}
 &nbsp; <strong>Citation links:</strong> {n_citations}
 &nbsp; <strong>Frontiers:</strong> {n_frontiers}</p>

<h2>Community frontiers</h2>
<p class="note">Pairs of {html.escape(community_col.replace("_", " "))}s that are similar in topic but rarely cite each other.
Each community is labelled by its most typical paper (closest to the community's mean embedding). Similarity is the mean cosine over all paper pairs, with its
bootstrap confidence interval. Density is shown relative to the whole citation graph (below 1x = cite each other less
than average). The p-value tests whether the topic overlap could be chance, adjusted for testing all pairs
(Benjamini-Hochberg). Frontier = significant after adjustment and density at or below the configured ratio.</p>
{_table(frontiers, "No community frontiers found.")}

<h2>Isolated paper pairs</h2>
<p class="note">Individual papers among the most similar pairs in the corpus that do not cite each other.</p>
{_table(pairs, "No isolated paper pairs found.")}
</div>
</body>
</html>
"""
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(page)
    logger.info(f"Wrote HTML gap report to {filepath}")
    return filepath
