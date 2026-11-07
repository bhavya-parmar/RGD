"""
Phase 2: Semantic & Citation Representation
Generates SPECTER2 embeddings, performs PCA/SVD dimensionality reduction,
builds FAISS similarity indices, and constructs directed citation graphs with PageRank & HITS.
"""

from .embed_specter import SpecterEmbedder, DimensionalityReducer
from .faiss_index import SemanticIndex
from .citation_graph import CitationGraph

__all__ = ["SpecterEmbedder", "DimensionalityReducer", "SemanticIndex", "CitationGraph"]
