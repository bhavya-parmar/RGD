"""
Literature Retrieval & Expansion using the OpenAlex API.
Fetches seed publications based on keywords/concepts and expands the corpus
via citation links (referenced_works) and semantically related works.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def reconstruct_abstract(abstract_inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """
    Reconstruct raw text from OpenAlex's abstract_inverted_index.
    OpenAlex stores abstracts as a dictionary mapping each word to list of position indices.
    """
    if not abstract_inverted_index or not isinstance(abstract_inverted_index, dict):
        return ""
    
    positions = []
    for word, indices in abstract_inverted_index.items():
        for pos in indices:
            positions.append((pos, word))
    
    positions.sort(key=lambda x: x[0])
    return " ".join([word for _, word in positions])


class OpenAlexFetcher:
    """
    Client for querying OpenAlex API with automatic polite-pool identification,
    pagination, citation expansion, and parsing into structured DataFrames.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        base_url: str = "https://api.openalex.org"
    ):
        self.api_key = api_key or os.getenv("OPENALEX_API_KEY")
        self.email = email or os.getenv("OPENALEX_EMAIL")
        self.base_url = base_url.rstrip("/")

        self.headers = {
            "User-Agent": f"RGD-ResearchGapDiscovery/1.0 (mailto:{self.email})"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def fetch_papers_by_query(
        self,
        query: str,
        max_results: int = 50,
        per_page: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Query OpenAlex works endpoint using keyword/concept search.
        """
        papers: List[Dict[str, Any]] = []
        page = 1
        url = f"{self.base_url}/works"

        logger.info(f"Querying OpenAlex for: '{query}' (Target: {max_results} papers)...")

        while len(papers) < max_results:
            params = {
                "search": query,
                "per_page": min(per_page, max_results - len(papers)),
                "page": page
            }
            if self.api_key:
                params["api_key"] = self.api_key

            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])

                if not results:
                    break

                for work in results:
                    parsed = self._parse_work(work)
                    if parsed:
                        papers.append(parsed)

                page += 1
                time.sleep(0.1)  # Respect API rate limits

            except requests.RequestException as e:
                logger.error(f"Error fetching page {page} from OpenAlex: {e}")
                break

        logger.info(f"Fetched {len(papers)} seed papers from OpenAlex.")
        return papers

    def expand_corpus_via_citations(
        self,
        seed_papers: List[Dict[str, Any]],
        max_references_per_paper: int = 15,
        max_total_expanded: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Expand the initial corpus by fetching referenced works and related works
        to bridge citation and cross-disciplinary gaps.
        """
        logger.info(f"Expanding corpus via citation links and related works...")
        expanded_ids = set()
        seed_ids = {p["id"] for p in seed_papers}

        for paper in seed_papers:
            # OpenAlex references
            refs = paper.get("referenced_works", [])[:max_references_per_paper]
            # OpenAlex related works
            related = paper.get("related_works", [])[:max_references_per_paper // 2]

            for ref_id in refs + related:
                if ref_id not in seed_ids and ref_id not in expanded_ids:
                    expanded_ids.add(ref_id)
                    if len(expanded_ids) >= max_total_expanded:
                        break
            if len(expanded_ids) >= max_total_expanded:
                break

        logger.info(f"Identified {len(expanded_ids)} linked candidate papers to retrieve.")

        expanded_papers = []
        batch_size = 25
        id_list = list(expanded_ids)

        for i in range(0, len(id_list), batch_size):
            batch = id_list[i : i + batch_size]
            # Query multiple IDs using pipe delimiter
            pipe_ids = "|".join([item.replace("https://openalex.org/", "") for item in batch])
            url = f"{self.base_url}/works?filter=openalex_id:{pipe_ids}"

            try:
                params = {"per_page": batch_size}
                if self.api_key:
                    params["api_key"] = self.api_key
                resp = requests.get(url, headers=self.headers, params=params, timeout=15)
                if resp.status_code == 200:
                    for work in resp.json().get("results", []):
                        parsed = self._parse_work(work)
                        if parsed:
                            expanded_papers.append(parsed)
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error retrieving expanded batch {i}: {e}")

        logger.info(f"Successfully retrieved {len(expanded_papers)} expanded papers.")
        return seed_papers + expanded_papers

    def _parse_work(self, work: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract and normalize relevant metadata fields.
        """
        work_id = work.get("id")
        title = work.get("title")
        if not work_id or not title:
            return None

        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

        authors = []
        for authorship in work.get("authorships", []):
            author_info = authorship.get("author", {})
            if author_info.get("display_name"):
                authors.append(author_info.get("display_name"))

        concepts = [c.get("display_name") for c in work.get("concepts", []) if c.get("display_name")]

        return {
            "id": work_id,
            "doi": work.get("doi", ""),
            "title": title,
            "publication_year": work.get("publication_year"),
            "cited_by_count": work.get("cited_by_count", 0),
            "abstract": abstract,
            "authors": authors,
            "concepts": concepts,
            "referenced_works": work.get("referenced_works", []),
            "related_works": work.get("related_works", [])
        }

    def to_dataframe(self, papers: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert list of papers to a clean Pandas DataFrame."""
        return pd.DataFrame(papers)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    fetcher = OpenAlexFetcher()
    test_query = "physics informed neural networks"
    print(f"--- Running Test Fetch for '{test_query}' ---")
    seeds = fetcher.fetch_papers_by_query(test_query, max_results=5)
    print(f"Retrieved {len(seeds)} papers.")
    if seeds:
        print("Sample Paper Title:", seeds[0]["title"])
        print("Sample Year:", seeds[0]["publication_year"])
        print("Abstract Preview:", seeds[0]["abstract"][:120] if seeds[0]["abstract"] else "[No Abstract]")
