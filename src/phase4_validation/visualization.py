"""
Visualization and Gap Reporting Logic.
Generates dual-view landscape representations (semantic space vs citation network),
community discrepancy heatmaps, and standalone interactive HTML analytical dashboards.
"""

import os
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

logger = logging.getLogger(__name__)


class LandscapeVisualizer:
    """
    Renders visual representations of the literature landscape,
    including 2D semantic projections, citation subgraphs, and gap heatmaps.
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_semantic_landscape(
        self,
        embeddings_2d: np.ndarray,
        df: pd.DataFrame,
        hue_col: str = "semantic_community",
        filename: str = "semantic_landscape.png"
    ) -> str:
        """
        Plot 2D projection (PCA or t-SNE) of the corpus with community coloration.
        """
        filepath = os.path.join(self.output_dir, filename)
        fig, ax = plt.subplots(figsize=(10, 7))

        communities = df.get(hue_col, pd.Series(np.zeros(len(df))))
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=communities,
            cmap="tab10",
            alpha=0.85,
            edgecolors="none",
            s=45
        )

        ax.set_title(f"2D Semantic Embedding Landscape (Colored by {hue_col})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Principal Dimension 1", fontsize=11)
        ax.set_ylabel("Principal Dimension 2", fontsize=11)
        plt.colorbar(scatter, ax=ax, label="Community Index")
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved semantic landscape visualization to {filepath}")
        return filepath

    def plot_frontier_heatmap(
        self,
        frontier_df: pd.DataFrame,
        filename: str = "frontier_discrepancy_matrix.png"
    ) -> str:
        """
        Create a heatmap visualizing the discrepancy between semantic overlap and citation density.
        """
        filepath = os.path.join(self.output_dir, filename)
        if frontier_df.empty:
            logger.warning("No frontier data available to plot heatmap.")
            return filepath

        all_comms = sorted(list(set(frontier_df["community_a"]).union(set(frontier_df["community_b"]))))
        matrix = pd.DataFrame(index=all_comms, columns=all_comms, dtype=float).fillna(0.0)

        for _, row in frontier_df.iterrows():
            ca, cb = int(row["community_a"]), int(row["community_b"])
            score = float(row["frontier_score"])
            matrix.loc[ca, cb] = score
            matrix.loc[cb, ca] = score

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Frontier Gap Score"})
        ax.set_title("Cross-Community Frontier Discrepancy Matrix", fontsize=13, fontweight="bold")
        ax.set_xlabel("Citation Community ID")
        ax.set_ylabel("Citation Community ID")
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved frontier heatmap to {filepath}")
        return filepath

    def export_interactive_network_html(
        self,
        graph: nx.Graph,
        df: pd.DataFrame,
        filename: str = "citation_network.html"
    ) -> str:
        """
        Export an interactive network using Pyvis or pure HTML fallback.
        """
        filepath = os.path.join(self.output_dir, filename)

        try:
            from pyvis.network import Network
            net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="#333333", directed=graph.is_directed())
            net.barnes_hut()

            for node, data in graph.nodes(data=True):
                title = data.get("title", node)
                comm = data.get("citation_community", 0)
                net.add_node(
                    node,
                    label=str(node)[:12],
                    title=f"{title}<br>Community: {comm}",
                    group=comm
                )

            for u, v in graph.edges():
                net.add_edge(u, v)

            net.save_graph(filepath)
            logger.info(f"Generated interactive network visualization at {filepath}")
        except Exception as e:
            logger.warning(f"Pyvis export error ({e}). Generating lightweight HTML report.")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"<html><body><h2>Citation Network Summary</h2><p>Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}</p></body></html>")

        return filepath

    def generate_html_summary_report(
        self,
        query: str,
        corpus_df: pd.DataFrame,
        frontier_df: pd.DataFrame,
        filename: str = "frontier_report.html"
    ) -> str:
        """
        Compile findings into an interactive dashboard-style HTML report.
        """
        filepath = os.path.join(self.output_dir, filename)

        frontiers_html = frontier_df.to_html(classes="table", index=False) if not frontier_df.empty else "<p>No candidate frontiers isolated.</p>"
        top_papers_html = corpus_df[["title", "publication_year", "cited_by_count"]].head(10).to_html(classes="table", index=False) if not corpus_df.empty else "<p>No papers.</p>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RGD: Research Gap Discovery Report - {query}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f8fafc; color: #1e293b; margin: 0; padding: 30px; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: #ffffff; border-radius: 12px; padding: 40px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
        h2 {{ color: #334155; margin-top: 30px; }}
        .badge {{ background-color: #e0e7ff; color: #4338ca; padding: 4px 10px; border-radius: 6px; font-weight: 600; }}
        .table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }}
        .table th, .table td {{ border: 1px solid #e2e8f0; padding: 10px 14px; text-align: left; }}
        .table th {{ background-color: #f1f5f9; font-weight: 600; }}
        .table tr:hover {{ background-color: #f8fafc; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Research Gap Discovery (RGD) Report</h1>
        <p><strong>Query:</strong> <span class="badge">{query}</span> | <strong>Analyzed Publications:</strong> {len(corpus_df)}</p>
        
        <h2>Candidate Research Frontiers (High Semantic Sim + Low Citation Density)</h2>
        <p>These community pairs demonstrate high conceptual overlap but operate in citation silos, representing high-potential cross-disciplinary frontiers.</p>
        {frontiers_html}
        
        <h2>Top Influential Seed Publications</h2>
        {top_papers_html}
    </div>
</body>
</html>
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Compiled comprehensive HTML frontier report at {filepath}")
        return filepath


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    vis = LandscapeVisualizer()
    np.random.seed(42)
    pts = np.random.randn(25, 2)
    df = pd.DataFrame({"semantic_community": np.random.randint(0, 3, size=25), "title": [f"Paper {i}" for i in range(25)]})
    p = vis.plot_semantic_landscape(pts, df)
    print("Exported plot:", p)
