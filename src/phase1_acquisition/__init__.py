"""
Phase 1: Literature Retrieval & Preparation
Includes OpenAlex API querying, citation link expansion, and MinHash/LSH deduplication.
"""

from .openalex_fetch import OpenAlexFetcher
from .deduplication import CorpusDeduplicator

__all__ = ["OpenAlexFetcher", "CorpusDeduplicator"]
