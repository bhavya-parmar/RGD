"""
RGD: Research Gap Discovery End-to-End Orchestrator Pipeline.
Sequentially runs:
  Phase 1: Retrieval (OpenAlex) & Deduplication (MinHash/LSH)
  Phase 2: Representation (SPECTER2, PCA, FAISS, Citation Network)
  Phase 3: Detection (k-means++ Semantic Clustering, Graph Clustering, Frontier Scoring)
  Phase 4: Validation & Reporting (Similarity Confidence Interval, Permutation Test, HTML Report)
"""

import os
import sys
import time
import argparse
import logging
from typing import List, Optional
import yaml
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Make `src` importable when run as a script
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
from src.phase4_validation.report import write_html_report

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Silence per-request HTTP logs from Hugging Face model loading
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
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
    mock_mode: bool = False,
    must_include: Optional[List[str]] = None,
    filter_expanded: bool = False
):
    """
    Execute full Research Gap Discovery sequential workflow.

    must_include: term groups that must appear in the title or abstract of
        seed papers. Use "|" inside a group for alternatives.
    filter_expanded: also apply must_include to citation-expanded papers.
    """
    total_start = time.time()
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    raw_dir = paths.get("raw_data_dir", "data/raw")
    proc_dir = paths.get("processed_data_dir", "data/processed")
    emb_dir = paths.get("embeddings_dir", "models/embeddings")
    faiss_dir = paths.get("faiss_indices_dir", "models/faiss_indices")
    reports_dir = paths.get("reports_dir", "reports")
    for d in [raw_dir, proc_dir, emb_dir, faiss_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    logger.info(f"Query: '{query}' ({'mock corpus' if mock_mode else 'OpenAlex'})")

    logger.info("Phase 1/4: retrieval and deduplication")
    if mock_mode:
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
        seeds = fetcher.fetch_papers_by_query(
            query, max_results=max_seeds, must_include=must_include
        )
        if not seeds:
            raise RuntimeError(
                f"No papers retrieved from OpenAlex for '{query}'. Check the query, "
                "--must-include terms and OPENALEX_API_KEY (see the log above for HTTP errors)."
            )

        full_papers = fetcher.expand_corpus_via_citations(
            seeds,
            max_references_per_paper=cfg.get("acquisition", {}).get("max_references_per_paper", 15),
            must_include=must_include if filter_expanded else None
        )
        raw_df = fetcher.to_dataframe(full_papers)
    raw_df.to_csv(os.path.join(raw_dir, "corpus_raw.csv"), index=False)

    dedup_cfg = cfg.get("deduplication", {})
    deduplicator = CorpusDeduplicator(
        threshold=dedup_cfg.get("lsh_threshold", 0.85),
        num_perm=dedup_cfg.get("num_perm", 128),
        shingle_size=dedup_cfg.get("shingle_size", 3)
    )
    corpus_df, _ = deduplicator.deduplicate(raw_df)
    corpus_df.to_csv(os.path.join(proc_dir, "corpus_deduplicated.csv"), index=False)

    logger.info("Phase 2/4: embeddings, FAISS index and citation graph")
    specter_cfg = cfg.get("representation", {}).get("specter", {})
    embedder = SpecterEmbedder(
        model_name=specter_cfg.get("model_name", "allenai/specter2_base"),
        adapter_name=specter_cfg.get("adapter_name", "allenai/specter2"),
        batch_size=specter_cfg.get("batch_size", 16)
    )
    raw_embeddings = embedder.embed_papers(corpus_df)
    np.save(os.path.join(emb_dir, "specter_raw.npy"), raw_embeddings)

    pca_cfg = cfg.get("representation", {}).get("pca", {})
    if pca_cfg.get("enabled", True):
        # After mean-centering, n papers span at most n - 1 directions
        n_comp = min(pca_cfg.get("n_components", 64), len(corpus_df) - 1, raw_embeddings.shape[1])
        reducer = DimensionalityReducer(n_components=n_comp)
        embeddings = reducer.fit_transform(raw_embeddings)
        reducer.save(os.path.join(emb_dir, "pca_reducer.pkl"))
    else:
        embeddings = raw_embeddings
    np.save(os.path.join(emb_dir, "embeddings_final.npy"), embeddings)

    faiss_cfg = cfg.get("representation", {}).get("faiss", {})
    semantic_index = SemanticIndex(dimension=embeddings.shape[1], metric=faiss_cfg.get("metric", "inner_product"))
    semantic_index.add(embeddings, corpus_df["id"].astype(str).tolist())
    semantic_index.save(os.path.join(faiss_dir, "semantic.index"))

    citation_graph = CitationGraph(directed=True)
    citation_graph.build_from_dataframe(corpus_df)
    citation_graph.get_summary_dataframe().to_csv(os.path.join(proc_dir, "citation_graph_degrees.csv"), index=False)

    logger.info("Phase 3/4: communities and frontier scoring")
    det_cfg = cfg.get("detection", {})
    sem_cfg = det_cfg.get("semantic_clustering", {})
    graph_cfg = det_cfg.get("graph_clustering", {})
    front_cfg = det_cfg.get("frontier_scoring", {})

    sem_detector = SemanticCommunityDetector(
        n_clusters=min(sem_cfg.get("n_clusters", 8), len(corpus_df)),
        init=sem_cfg.get("init", "k-means++"),
        max_iter=sem_cfg.get("max_iter", 300),
        random_state=sem_cfg.get("random_state", 42)
    )
    corpus_df = sem_detector.assign_to_dataframe(corpus_df, embeddings)

    graph_detector = CitationCommunityDetector(
        resolution=graph_cfg.get("resolution", 1.0),
        random_state=graph_cfg.get("random_state", 42)
    )
    graph_detector.fit_predict(citation_graph.graph)
    corpus_df = graph_detector.assign_to_dataframe(corpus_df)

    community_col = front_cfg.get("community_col", "citation_community")
    frontier_scorer = FrontierScorer(
        similarity_percentile=front_cfg.get("similarity_percentile", 95),
        max_density_ratio=front_cfg.get("max_density_ratio", 1.0),
        min_community_size=front_cfg.get("min_community_size", 3)
    )
    community_frontiers = frontier_scorer.score_community_frontiers(
        corpus_df, embeddings, citation_graph.graph, community_col=community_col
    )
    isolated_paper_pairs = frontier_scorer.find_isolated_paper_pairs(
        corpus_df, embeddings, semantic_index, citation_graph.graph, top_n_pairs=15
    )
    isolated_paper_pairs.to_csv(os.path.join(reports_dir, "isolated_paper_pairs.csv"), index=False)

    logger.info("Phase 4/4: validation and report")
    val_cfg = cfg.get("validation", {})
    ci_cfg = val_cfg.get("similarity_ci", {})
    test_cfg = val_cfg.get("hypothesis_test", {})
    validator = StatisticalValidator(
        n_iterations=ci_cfg.get("n_iterations", 500),
        subsample_ratio=ci_cfg.get("subsample_ratio", 0.80),
        confidence=ci_cfg.get("confidence", 0.95),
        alpha=test_cfg.get("alpha", 0.05),
        random_state=ci_cfg.get("random_state", 42)
    )
    validated_frontiers = validator.similarity_confidence_interval(
        community_frontiers, corpus_df, embeddings, community_col=community_col
    )
    validated_frontiers = validator.permutation_test_significance(
        validated_frontiers, corpus_df, embeddings,
        n_permutations=test_cfg.get("n_permutations", 100), community_col=community_col
    )
    # Frontiers first, then by score; tests were corrected over all pairs before this cut
    validated_frontiers = validated_frontiers.sort_values(
        ["is_frontier", "frontier_score"], ascending=False
    ).head(front_cfg.get("top_k_frontiers", 10))
    validated_frontiers.to_csv(os.path.join(reports_dir, "validated_frontiers.csv"), index=False)

    report_html = write_html_report(
        reports_dir, query, corpus_df, embeddings, citation_graph.graph.number_of_edges(),
        validated_frontiers, isolated_paper_pairs, community_col=community_col
    )
    logger.info(f"Pipeline finished in {time.time() - total_start:.1f}s. Report: {report_html}")

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
    parser.add_argument(
        "--must-include", nargs="+", metavar="TERM", default=None,
        help='Terms that must appear in the title or abstract of seed papers. '
             'Each TERM is required; use "|" for alternatives and quotes for phrases. '
             'Example: --must-include anisotropy "language model|transformer"'
    )
    parser.add_argument(
        "--filter-expanded", action="store_true",
        help="Also apply --must-include to papers added by citation expansion"
    )
    args = parser.parse_args()

    if args.filter_expanded and not args.must_include:
        parser.error("--filter-expanded requires --must-include")

    run_pipeline(
        query=args.query,
        max_seeds=args.max_seeds,
        config_path=args.config,
        mock_mode=args.mock,
        must_include=args.must_include,
        filter_expanded=args.filter_expanded
    )


if __name__ == "__main__":
    main()
