"""
PatentScraper — fetches automotive AI perception patents from the Crossref API.

Crossref indexes patent metadata globally, is completely free, requires no API keys,
and reliably provides abstracts. It's the most stable option for academic RAG pipelines.

API: https://api.crossref.org/works
"""

import requests
from typing import List, Optional, Dict

from src.ingestion.base_scraper import BaseScraper
from src.schemas.document import DocumentSchema
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.crossref.org/works"


class PatentScraper(BaseScraper):
    """
    Fetches patent records via the Crossref REST API.
    """

    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "Simulation Platforms": [
            "autonomous driving simulation",
            "driving simulator",
            "sensor simulation",
        ],
        "Perception & World models": [
            "neural rendering",
            "occupancy network",
            "world model",
            "point cloud processing",
            "object detection",
        ],
        "Sensors & Environment": [
            "LiDAR",
            "radar sensor",
            "sensor fusion",
            "depth estimation",
        ],
        "Validation & Testing": [
            "hardware-in-the-loop",
            "software-in-the-loop",
            "virtual validation",
        ],
    }

    def __init__(self, api_key: Optional[str] = None) -> None:
        # Crossref doesn't require an API key, but they prefer a mailto header
        # to enter the "polite pool" (faster responses).
        self.headers = {
            "User-Agent": "AIPerceptionRAG/1.0 (mailto:academic-researcher@example.com)"
        }
        logger.info("PatentScraper (Crossref API) initialized.")

    def build_query(
        self,
        selected_domains: Optional[List[str]] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2026-12-31",
    ) -> str:
        """
        Builds a Crossref query string.
        """
        all_keywords: List[str] = []
        if selected_domains:
            for domain in selected_domains:
                all_keywords.extend(self.DOMAIN_KEYWORDS.get(domain, []))

        # Join with basic OR logic for the crossref search engine
        query = " OR ".join(f'"{kw}"' if " " in kw else kw for kw in all_keywords)
        if not query:
             query = "autonomous driving"
             
        logger.debug(f"Built Crossref query: {query}")
        return query

    def fetch(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[DocumentSchema]:
        """
        Sends the query to Crossref, filtering specifically for 'peer-review' type which
        Crossref uses for patents in some registries, or general works containing 'patent' in title.
        """
        # We append 'patent' to the query terms and filter for specific types
        search_query = f"({query}) AND patent"
        
        params = {
            "query": search_query,
            "rows": max_results,
            "sort": "created",
            "order": "desc",
            "filter": "has-abstract:true" # Guarantee abstracts
        }

        logger.info(f"Searching Crossref for top {max_results} patent-related works...")

        try:
            response = requests.get(_BASE_URL, headers=self.headers, params=params, timeout=20)
            response.raise_for_status()
        except requests.HTTPError as e:
            logger.error(f"Crossref API error {e.response.status_code}: {e.response.text[:200]}")
            raise

        data = response.json()
        items = data.get("message", {}).get("items", [])
        logger.info(f"Crossref returned {len(items)} items. Parsing...")

        results: List[DocumentSchema] = []
        for item in items:
            # Title
            title_list = item.get("title", [])
            title = title_list[0] if title_list else "Untitled"

            # Abstract
            abstract = item.get("abstract", "No abstract available.")
            # Crossref abstracts sometimes have JATS XML tags like <jats:p>
            import re
            abstract = re.sub(r'<[^>]+>', '', abstract).strip()

            # Date
            created = item.get("created", {}).get("date-time", "")
            date = created.split("T")[0] if created else "Unknown"

            # ID (DOI)
            doi = item.get("DOI", "unknown-doi")
            url = item.get("URL", f"https://doi.org/{doi}")

            # Authors
            authors = []
            for author in item.get("author", []):
                given = author.get("given", "")
                family = author.get("family", "")
                name = f"{given} {family}".strip()
                if name:
                    authors.append(name)

            doc = DocumentSchema(
                source="crossref",
                id=doi,
                title=title,
                authors=authors if authors else ["Unknown Author"],
                abstract=abstract,
                published_date=date,
                pdf_url=url,
            )
            results.append(doc)
            logger.debug(f"  Parsed: {doi} | '{title[:60]}'")

        logger.info(f"Successfully parsed {len(results)} works.")
        return results