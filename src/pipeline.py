"""
RGD: Research Gap Discovery End-to-End Orchestrator Pipeline.
Sequentially runs:
  Phase 1: Retrieval (OpenAlex) & Deduplication (MinHash/LSH)
  Phase 2: Representation (SPECTER2, PCA, FAISS, Citation Network, PageRank, HITS)
  Phase 3: Detection (k-means++ Semantic Clustering, Graph Clustering, Frontier Scoring)
  Phase 4: Validation & Reporting (Bootstrap Resampling, Permutation Hypothesis Testing, Visualizations)
"""

import os
import sys
import time
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ensure local packages in src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.phase1_acquisition.openalex_fetch import OpenAlexFetcher
from src.phase1_acquisition.deduplication import CorpusDeduplicator
from src.phase2_representation.embed_specter import SpecterEmbedder, DimensionalityReducer
from src.phase2_representation.faiss_index import SemanticIndex
from src.phase2_representation.citation_graph import CitationGraph
from src.phase3_detection.semantic_cluster import SemanticCommunityDetector
from src.phase3_detection.graph_cluster import CitationCommunityDetector
from src.phase3_detection.frontier_scoring import FrontierScorer
from src.phase4_validation.hypothesis_test import StatisticalValidator
from src.phase4_validation.visualization import LandscapeVisualizer

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RGD-Pipeline")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load hyperparameters from YAML configuration file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file '{config_path}' not found. Using defaults.")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_pipeline(
    query: str,
    max_seeds: int = 50,
    config_path: str = "config.yaml",
    mock_mode: bool = False
):
    """
    Execute full Research Gap Discovery sequential workflow.
    """
    total_start = time.time()
    cfg = load_config(config_path)

    raw_dir = cfg.get("paths", {}).get("raw_data_dir", "data/raw")
    proc_dir = cfg.get("paths", {}).get("processed_data_dir", "data/processed")
    emb_dir = cfg.get("paths", {}).get("embeddings_dir", "models/embeddings")
    faiss_dir = cfg.get("paths", {}).get("faiss_indices_dir", "models/faiss_indices")
    reports_dir = cfg.get("paths", {}).get("reports_dir", "reports")

    for d in [raw_dir, proc_dir, emb_dir, faiss_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"🚀 STARTING RESEARCH GAP DISCOVERY (RGD) PIPELINE")
    print(f"🎯 Target Research Query: '{query}'")
    print(f"⚙️ Mode: {'Synthetic Mock (Offline)' if mock_mode else 'Live OpenAlex API'}")
    print("=" * 80 + "\n")

    # =========================================================================
    # PHASE 1: Literature Retrieval & Preparation
    # =========================================================================
    p1_start = time.time()
    logger.info(">>> [PHASE 1] Literature Retrieval & Deduplication")

    if mock_mode:
        logger.info("Generating synthetic mock scientific corpus...")
        synthetic_papers = [
            {
                "id": f"W_{i}",
                "title": f"Scientific Investigation {i}: {query} in Domain {i % 3}",
                "publication_year": 2018 + (i % 6),
                "cited_by_count": (i * 17) % 250,
                "abstract": f"This research study addresses foundational methodologies for {query} with application to subdomain {i % 3}.",
                "referenced_works": [f"W_{(i - j) % 20}" for j in range(1, 4) if (i - j) >= 0]
            }
            for i in range(max_seeds)
        ]
        raw_df = pd.DataFrame(synthetic_papers)
    else:
        fetcher = OpenAlexFetcher()
        seeds = fetcher.fetch_papers_by_query(query, max_results=max_seeds)
        if not seeds:
            logger.error("No papers retrieved from OpenAlex. Falling back to synthetic mock data.")
            return run_pipeline(query, max_seeds, config_path, mock_mode=True)

        full_papers = fetcher.expand_corpus_via_citations(
            seeds,
            max_references_per_paper=cfg.get("acquisition", {}).get("max_references_per_paper", 15)
        )
        raw_df = fetcher.to_dataframe(full_papers)

    # Save raw snapshot
    raw_path = os.path.join(raw_dir, "corpus_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    logger.info(f"Saved raw literature corpus ({len(raw_df)} records) to {raw_path}")

    # MinHash & LSH Deduplication
    dedup_cfg = cfg.get("deduplication", {})
    deduplicator = CorpusDeduplicator(
        threshold=dedup_cfg.get("lsh_threshold", 0.85),
        num_perm=dedup_cfg.get("num_perm", 128),
        shingle_size=dedup_cfg.get("shingle_size", 3)
    )
    corpus_df, removed_count = deduplicator.deduplicate(raw_df)
    proc_path = os.path.join(proc_dir, "corpus_deduplicated.csv")
    corpus_df.to_csv(proc_path, index=False)
    logger.info(f"[Phase 1 Finished] Corpus size: {len(corpus_df)} (Removed {removed_count} duplicates) in {time.time() - p1_start:.2f}s")

    # =========================================================================
    # PHASE 2: Semantic & Citation Representation
    # =========================================================================
    p2_start = time.time()
    logger.info("\n>>> [PHASE 2] Semantic & Citation Representation")

    # SPECTER2 Embeddings
    specter_cfg = cfg.get("representation", {}).get("specter", {})
    embedder = SpecterEmbedder(
        model_name=specter_cfg.get("model_name", "allenai/specter2_base"),
        batch_size=specter_cfg.get("batch_size", 16)
    )
    raw_embeddings = embedder.embed_papers(corpus_df)
    np.save(os.path.join(emb_dir, "specter_raw.npy"), raw_embeddings)

    # Dimensionality Reduction (PCA)
    pca_cfg = cfg.get("representation", {}).get("pca", {})
    if pca_cfg.get("enabled", True):
        n_comp = min(pca_cfg.get("n_components", 64), len(corpus_df) - 1, raw_embeddings.shape[1])
        reducer = DimensionalityReducer(n_components=n_comp)
        embeddings = reducer.fit_transform(raw_embeddings)
        reducer.save(os.path.join(emb_dir, "pca_reducer.pkl"))
    else:
        embeddings = raw_embeddings
    np.save(os.path.join(emb_dir, "embeddings_final.npy"), embeddings)

    # FAISS Semantic Index
    faiss_cfg = cfg.get("representation", {}).get("faiss", {})
    dim = embeddings.shape[1]
    semantic_index = SemanticIndex(dimension=dim, metric=faiss_cfg.get("metric", "inner_product"))
    semantic_index.add(embeddings, corpus_df["id"].astype(str).tolist())
    semantic_index.save(os.path.join(faiss_dir, "semantic.index"))

    # Citation Graph with PageRank & HITS
    citation_graph = CitationGraph(directed=True)
    citation_graph.build_from_dataframe(corpus_df)
    citation_graph.compute_pagerank(alpha=cfg.get("representation", {}).get("citation_graph", {}).get("pagerank_alpha", 0.85))
    citation_graph.compute_hits()
    graph_summary = citation_graph.get_summary_dataframe()
    graph_summary.to_csv(os.path.join(proc_dir, "graph_centrality_metrics.csv"), index=False)
    logger.info(f"[Phase 2 Finished] Representation models constructed in {time.time() - p2_start:.2f}s")

    # =========================================================================
    # PHASE 3: Community & Frontier Detection
    # =========================================================================
    p3_start = time.time()
    logger.info("\n>>> [PHASE 3] Community & Frontier Detection")

    det_cfg = cfg.get("detection", {})
    sem_cfg = det_cfg.get("semantic_clustering", {})
    graph_cfg = det_cfg.get("graph_clustering", {})
    front_cfg = det_cfg.get("frontier_scoring", {})

    # Semantic Communities (k-means++)
    sem_detector = SemanticCommunityDetector(
        n_clusters=min(sem_cfg.get("n_clusters", 8), len(corpus_df)),
        init=sem_cfg.get("init", "k-means++")
    )
    corpus_df = sem_detector.assign_to_dataframe(corpus_df, embeddings)

    # Citation Communities (Spectral / Modularity)
    graph_detector = CitationCommunityDetector(
        method=graph_cfg.get("algorithm", "spectral"),
        n_clusters=min(graph_cfg.get("n_clusters", 8), len(corpus_df))
    )
    graph_detector.fit_predict(citation_graph.graph)
    corpus_df = graph_detector.assign_to_dataframe(corpus_df)

    # Frontier Scoring
    frontier_scorer = FrontierScorer(
        semantic_sim_threshold=front_cfg.get("semantic_sim_threshold", 0.65),
        citation_connectivity_threshold=front_cfg.get("citation_connectivity_threshold", 0.10),
        min_community_size=front_cfg.get("min_community_size", 2),
        top_k_frontiers=front_cfg.get("top_k_frontiers", 10)
    )
    community_frontiers = frontier_scorer.score_community_frontiers(
        corpus_df, embeddings, citation_graph.graph, community_col="citation_community"
    )
    isolated_paper_pairs = frontier_scorer.find_isolated_paper_pairs(
        corpus_df, embeddings, citation_graph.graph, top_n_pairs=15
    )

    community_frontiers.to_csv(os.path.join(reports_dir, "candidate_community_frontiers.csv"), index=False)
    isolated_paper_pairs.to_csv(os.path.join(reports_dir, "isolated_paper_pairs.csv"), index=False)
    logger.info(f"[Phase 3 Finished] Identified candidate research frontiers in {time.time() - p3_start:.2f}s")

    # =========================================================================
    # PHASE 4: Validation & Analytical Dashboard
    # =========================================================================
    p4_start = time.time()
    logger.info("\n>>> [PHASE 4] Validation & Analytical Dashboard")

    val_cfg = cfg.get("validation", {})
    validator = StatisticalValidator(
        n_iterations=val_cfg.get("bootstrap", {}).get("n_iterations", 100),
        alpha=val_cfg.get("hypothesis_test", {}).get("alpha", 0.05)
    )

    # Bootstrap Stability & Hypothesis Testing
    validated_frontiers = validator.bootstrap_frontier_stability(
        community_frontiers, corpus_df, embeddings, citation_graph.graph
    )
    validated_frontiers = validator.permutation_test_significance(
        validated_frontiers, corpus_df, embeddings, citation_graph.graph, n_permutations=100
    )
    validated_frontiers.to_csv(os.path.join(reports_dir, "validated_frontiers.csv"), index=False)

    # Visualizations
    visualizer = LandscapeVisualizer(output_dir=reports_dir)

    # 2D projection for plotting
    pca_2d = DimensionalityReducer(n_components=2)
    embeddings_2d = pca_2d.fit_transform(embeddings)

    vis_landscape = visualizer.plot_semantic_landscape(embeddings_2d, corpus_df, hue_col="citation_community")
    vis_heatmap = visualizer.plot_frontier_heatmap(validated_frontiers)
    vis_network = visualizer.export_interactive_network_html(citation_graph.graph, corpus_df)
    report_html = visualizer.generate_html_summary_report(query, corpus_df, validated_frontiers)

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"✅ RGD PIPELINE SUCCESSFULLY COMPLETED ({total_time:.2f}s)")
    print(f"📊 Processed Corpus: {proc_path}")
    print(f"🏆 Validated Frontiers: {os.path.join(reports_dir, 'validated_frontiers.csv')}")
    print(f"🌐 Interactive Dashboard: {report_html}")
    print(f"🕸️ Citation Network HTML: {vis_network}")
    print("=" * 80 + "\n")

    return {
        "corpus_df": corpus_df,
        "validated_frontiers": validated_frontiers,
        "report_html": report_html
    }


def main():
    parser = argparse.ArgumentParser(description="Research Gap Discovery (RGD) Orchestration Pipeline")
    parser.add_argument("--query", type=str, default="physics informed neural networks", help="Research domain query")
    parser.add_argument("--max-seeds", type=int, default=30, help="Maximum number of seed papers to fetch")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mock", action="store_true", help="Run with synthetic corpus for instant testing")
    args = parser.parse_args()

    run_pipeline(
        query=args.query,
        max_seeds=args.max_seeds,
        config_path=args.config,
        mock_mode=args.mock
    )


if __name__ == "__main__":
    main()
