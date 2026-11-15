"""
HTML Gap Report.
Writes one self-contained HTML page with the citation graph (coloured by
community), the validated community frontiers and the isolated paper pairs.
Paper titles link to their DOI or OpenAlex page.
"""

import os
import html
import logging
from typing import Dict
import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)

STYLE = """
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 30px; }
.container { max-width: 1100px; margin: 0 auto; background: #fff; border-radius: 12px; padding: 40px; box-shadow: 0 4px 6px -1px rgba(0,0,0,.1); }
h1 { color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }
h2 { color: #334155; margin-top: 32px; }
a { color: #1d4ed8; text-decoration: none; }
a:hover { text-decoration: underline; }
.badge { background: #e0e7ff; color: #4338ca; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
.note { color: #64748b; font-size: 14px; }
table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }
th, td { border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }
th { background: #f1f5f9; font-weight: 600; }
tr:hover { background: #f8fafc; }
svg { width: 100%; height: auto; border: 1px solid #e2e8f0; border-radius: 8px; }
.legend { list-style: none; padding: 0; font-size: 14px; }
.legend li { margin: 4px 0; }
.swatch { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }
"""

# Colour-blind friendly palette; communities beyond it are grey
PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377",
           "#EE7733", "#009988", "#BBBBBB", "#332288"]
OTHER = "#d4d4d8"


def _paper_url(row: pd.Series) -> str:
    """DOI link if available, else the OpenAlex work page."""
    doi = row.get("doi")
    return doi if isinstance(doi, str) and doi.startswith("http") else str(row["id"])


def _link(title: str, url: str, max_len: int = 90) -> str:
    text = title if len(title) <= max_len else title[:max_len - 1] + "…"
    return f'<a href="{html.escape(url)}" target="_blank">{html.escape(text)}</a>'


def _representatives(corpus_df: pd.DataFrame, embeddings: np.ndarray, community_col: str) -> Dict[int, int]:
    """
    Row position of each community's most typical paper: the member closest to
    the community's mean embedding (for unit vectors, highest dot product with the mean).
    """
    labels = corpus_df[community_col].to_numpy()
    reps = {}
    for c in pd.unique(labels):
        members = np.where(labels == c)[0]
        centroid = embeddings[members].mean(axis=0)
        reps[c] = int(members[np.argmax(embeddings[members] @ centroid)])
    return reps


def _graph_svg(graph: nx.Graph, corpus_df: pd.DataFrame, community_col: str,
               reps: Dict[int, int], colors: Dict[int, str], size: int = 900) -> str:
    """
    Citation graph as inline SVG: nodes coloured by community, representatives of
    coloured communities enlarged. Papers with no citation link are left out, since
    they would only stretch the layout.
    """
    graph = graph.subgraph([n for n in graph.nodes() if graph.degree(n) > 0])
    pos = nx.spring_layout(graph.to_undirected(), seed=42, k=2 / np.sqrt(max(len(graph), 1)))
    xy = np.array(list(pos.values()))
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    margin = 20
    scale = lambda p: margin + (np.asarray(p) - lo) / np.where(hi > lo, hi - lo, 1) * (size - 2 * margin)
    pt = {n: scale(p) for n, p in pos.items()}

    rows = corpus_df.assign(_pos=range(len(corpus_df))).set_index(corpus_df["id"].astype(str))
    rep_positions = {reps[c] for c in colors}

    parts = [f'<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">']
    for u, v in graph.edges():
        (x1, y1), (x2, y2) = pt[u], pt[v]
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#cbd5e1" stroke-width="0.6"/>')

    # Representatives drawn last so they sit on top
    nodes = sorted(graph.nodes(), key=lambda n: rows.loc[n, "_pos"] in rep_positions)
    for n in nodes:
        row = rows.loc[n]
        comm = int(row[community_col])
        is_rep = row["_pos"] in rep_positions
        x, y = pt[n]
        radius, stroke = (9, 'stroke="#0f172a" stroke-width="2.5"') if is_rep else (4, 'stroke="#fff" stroke-width="0.8"')
        tooltip = html.escape(f"C{comm} · {row['title']}" + (" (most typical paper)" if is_rep else ""))
        parts.append(
            f'<a href="{html.escape(_paper_url(row))}" target="_blank">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{colors.get(comm, OTHER)}" {stroke}>'
            f'<title>{tooltip}</title></circle></a>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def write_html_report(
    output_dir: str,
    query: str,
    corpus_df: pd.DataFrame,
    embeddings: np.ndarray,
    citation_graph: nx.Graph,
    frontier_df: pd.DataFrame,
    isolated_pairs_df: pd.DataFrame,
    community_col: str = "citation_community",
    filename: str = "frontier_report.html"
) -> str:
    """Write the gap report and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    df = corpus_df.reset_index(drop=True)
    reps = _representatives(df, embeddings, community_col)

    # Colour the largest communities; singletons and the rest stay grey
    sizes = df[community_col].value_counts()
    colors = {c: PALETTE[i] for i, c in enumerate(sizes[sizes >= 3].index[:len(PALETTE)])}

    def community_cell(c: int) -> str:
        rep = df.iloc[reps[c]]
        swatch = f'<span class="swatch" style="background:{colors.get(c, OTHER)}"></span>'
        return f"{swatch}C{c} · {_link(str(rep['title']), _paper_url(rep), 70)}"

    frontiers = pd.DataFrame()
    if not frontier_df.empty:
        f = frontier_df
        frontiers = pd.DataFrame({
            "Community A": f["community_a"].map(community_cell),
            "Community B": f["community_b"].map(community_cell),
            "Similarity [95% CI]": [
                f"{s:.3f} [{lo:.3f}, {hi:.3f}]"
                for s, lo, hi in zip(f["mean_semantic_similarity"], f["similarity_ci_low"], f["similarity_ci_high"])
            ],
            "Cross-citations": f["cross_citation_edges"],
            "Density vs graph avg": f["density_ratio"].map(lambda r: f"{r:.2f}x"),
            "p (adjusted)": f["p_value_adjusted"].map(lambda p: f"{p:.3f}"),
            "Frontier": f["is_frontier"].map({True: "yes", False: "no"}),
        })

    urls = dict(zip(df["id"].astype(str), df.apply(_paper_url, axis=1)))
    pairs = pd.DataFrame()
    if not isolated_pairs_df.empty:
        p = isolated_pairs_df
        pairs = pd.DataFrame({
            "Paper 1": [_link(str(t), urls.get(str(i), str(i))) for t, i in zip(p["paper_1_title"], p["paper_1_id"])],
            "Paper 2": [_link(str(t), urls.get(str(i), str(i))) for t, i in zip(p["paper_2_title"], p["paper_2_id"])],
            "Similarity": p["semantic_similarity"].map(lambda s: f"{s:.3f}"),
        })

    legend = "\n".join(
        f'<li>{community_cell(c)} ({sizes[c]} papers)</li>' for c in colors
    )
    table = lambda t, msg: t.to_html(index=False, escape=False, border=0) if not t.empty else f'<p class="note">{msg}</p>'
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
 &nbsp; <strong>Papers:</strong> {len(df)}
 &nbsp; <strong>Citation links:</strong> {citation_graph.number_of_edges()}
 &nbsp; <strong>Frontiers:</strong> {n_frontiers}</p>

<h2>Citation graph</h2>
<p class="note">Each node is a paper and each line a citation. Colour = {html.escape(community_col.replace("_", " "))};
grey = communities with fewer than 3 papers; papers with no citation link are not shown. Large outlined nodes are each community's most typical paper
(closest to the community's mean embedding). Hover for the title, click to open the paper.</p>
{_graph_svg(citation_graph, df, community_col, reps, colors)}
<ul class="legend">
{legend}
</ul>

<h2>Community frontiers</h2>
<p class="note">Pairs of communities that are similar in topic but cite each other less than average. Similarity is the
mean cosine over all paper pairs with its bootstrap 95% confidence interval. Density is relative to the whole citation
graph (below 1x = cite each other less than average). The p-value tests whether the topic overlap could be chance,
adjusted for testing all pairs (Benjamini-Hochberg). Frontier = significant after adjustment and low citation.</p>
{table(frontiers, "No community frontiers found.")}

<h2>Isolated paper pairs</h2>
<p class="note">Among the most similar pairs of papers in the corpus, those with no citation in either direction.</p>
{table(pairs, "No isolated paper pairs found.")}
</div>
</body>
</html>
"""
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(page)
    logger.info(f"Wrote HTML gap report to {filepath}")
    return filepath
