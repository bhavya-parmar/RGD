"""
Phase 3: Community & Frontier Detection
Clusters literature along semantic and citation dimensions and scores discrepancies
to isolate candidate research frontiers (high semantic similarity + weak citation links).
"""

from .semantic_cluster import SemanticCommunityDetector
from .graph_cluster import CitationCommunityDetector
from .frontier_scoring import FrontierScorer

__all__ = [
    "SemanticCommunityDetector",
    "CitationCommunityDetector",
    "FrontierScorer"
]
