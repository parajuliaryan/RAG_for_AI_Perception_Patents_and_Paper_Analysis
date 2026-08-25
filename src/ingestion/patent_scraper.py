"""
PatentScraper — fetches automotive AI perception patents from the EPO OPS API.

Uses OAuth2 Client Credentials flow.
API: https://ops.epo.org/rest-services/published-data/search
"""

import base64
import requests
from typing import List, Optional, Dict
from datetime import datetime

import src.config as cfg
from src.ingestion.base_scraper import BaseScraper
from src.schemas.document import DocumentSchema
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PatentScraper(BaseScraper):
    """
    Fetches patent records via the authoritative EPO Open Patent Services (OPS) API.
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
            "point cloud",
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

    def __init__(self) -> None:
        self.auth_url = "https://ops.epo.org/3.2/auth/accesstoken"
        self.search_url = "https://ops.epo.org/rest-services/published-data/search/biblio"
        
        self.consumer_key = cfg.EPO_CONSUMER_KEY
        self.consumer_secret = cfg.EPO_CONSUMER_SECRET

        if not self.consumer_key or not self.consumer_secret:
            logger.warning("EPO API keys missing from .env. The scraper will fail to authenticate.")
            
        self.access_token = None
        logger.info("PatentScraper (EPO OPS) initialized.")

    def _authenticate(self) -> None:
        """Exchanges Consumer Key & Secret for an OAuth2 Bearer Token."""
        logger.debug("Requesting new access token from EPO...")
        credentials = f"{self.consumer_key}:{self.consumer_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"grant_type": "client_credentials"}

        response = requests.post(self.auth_url, headers=headers, data=data, timeout=10)
        response.raise_for_status()

        token_data = response.json()
        self.access_token = token_data.get("access_token")
        logger.info("Successfully authenticated with EPO OPS API.")

    def build_query(
        self,
        selected_domains: Optional[List[str]] = None,
        start_date: str = "2020",
        end_date: str = "2026",
    ) -> str:
        """
        Builds a CQL (Contextual Query Language) string for the EPO API.
        """
        all_keywords: List[str] = []
        if selected_domains:
            for domain in selected_domains:
                all_keywords.extend(self.DOMAIN_KEYWORDS.get(domain, []))

        if not all_keywords:
            all_keywords = ["autonomous vehicle"]

        # Format keywords for CQL (abstract matches)
        keyword_clause = " OR ".join(f'"{kw}"' if " " in kw else kw for kw in all_keywords)
        
        # Restrict to US/EP patents with abstracts within the date range
        # pd = publication date, ab = abstract
        query = f'ab=({keyword_clause}) and pd within "{start_date} {end_date}"'
        logger.debug(f"Built EPO CQL query: {query}")
        return query

    def fetch(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[DocumentSchema]:
        """
        Sends the authenticated query to EPO OPS and parses the nested JSON bibliographic data.
        """
        if not self.access_token:
            self._authenticate()

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }
        
        # Range format: 1-5
        params = {
            "q": query,
            "Range": f"1-{max_results}"
        }

        logger.info(f"Searching EPO OPS for top {max_results} patent(s)...")

        try:
            response = requests.get(self.search_url, headers=headers, params=params, timeout=20)
            if response.status_code == 400 and "Token Invalid" in response.text:
                logger.warning("Token expired, re-authenticating...")
                self._authenticate()
                headers["Authorization"] = f"Bearer {self.access_token}"
                response = requests.get(self.search_url, headers=headers, params=params, timeout=20)
                
            response.raise_for_status()
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.info("EPO API returned 404 (No matching records found).")
                return []
            logger.error(f"EPO API error {e.response.status_code}: {e.response.text[:200]}")
            raise

        data = response.json()
        
        try:
            results_node = data["ops:world-patent-data"]["ops:biblio-search"]["ops:search-result"]["exchange-documents"]
        except KeyError:
            logger.warning("No exchange-documents found in EPO response.")
            return []
            
        # EPO returns a single dict if only 1 result, otherwise a list
        if isinstance(results_node, dict):
            results_node = [results_node]

        parsed_docs: List[DocumentSchema] = []
        for doc in results_node:
            biblio = doc.get("exchange-document", {}).get("bibliographic-data", {})
            
            # 1. ID / Publication Reference
            pub_ref = biblio.get("publication-reference", {}).get("document-id", [])
            if isinstance(pub_ref, list) and pub_ref:
                pub_ref = pub_ref[0]  # Usually epodoc is first
            country = pub_ref.get("country", {}).get("$", "US")
            doc_number = pub_ref.get("doc-number", {}).get("$", "unknown")
            kind = pub_ref.get("kind", {}).get("$", "A1")
            full_id = f"{country}{doc_number}{kind}"
            
            # 2. Title
            title_node = biblio.get("invention-title", [])
            if isinstance(title_node, dict):
                title_node = [title_node]
            title = "Untitled Patent"
            for t in title_node:
                if t.get("@lang", "") == "en":
                    title = t.get("$", title)
                    break
            
            # 3. Date
            date_raw = pub_ref.get("date", {}).get("$", "")
            date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}" if len(date_raw) == 8 else "Unknown"
            
            # 4. Inventors
            inventors = []
            inv_nodes = biblio.get("parties", {}).get("inventors", {}).get("inventor", [])
            if isinstance(inv_nodes, dict):
                inv_nodes = [inv_nodes]
            for inv in inv_nodes:
                name = inv.get("inventor-name", {}).get("name", {}).get("$", "")
                if name:
                    inventors.append(name.title())
            
            # 5. Abstract
            abstract = "No abstract available."
            abstract_nodes = doc.get("exchange-document", {}).get("abstract", [])
            if isinstance(abstract_nodes, dict):
                abstract_nodes = [abstract_nodes]
            for a in abstract_nodes:
                if a.get("@lang", "") == "en":
                    p_node = a.get("p", {})
                    if isinstance(p_node, str):
                        abstract = p_node
                    elif isinstance(p_node, dict):
                        abstract = p_node.get("$", abstract)
                    break
            
            # Provide standard Google Patents link for visual reference (we rely on abstract for content)
            google_link = f"https://patents.google.com/patent/{full_id}/en"

            doc_schema = DocumentSchema(
                source="epo_ops",
                id=full_id,
                title=title,
                authors=inventors if inventors else ["Unknown Inventor"],
                abstract=abstract,
                published_date=date,
                pdf_url=None  # We intentionally skip PDF extraction for patents to avoid Google captchas/paywalls
            )
            parsed_docs.append(doc_schema)
            logger.debug(f"  Parsed: {full_id} | '{title[:60]}'")

        logger.info(f"Successfully parsed {len(parsed_docs)} patent(s) from EPO.")
        return parsed_docs