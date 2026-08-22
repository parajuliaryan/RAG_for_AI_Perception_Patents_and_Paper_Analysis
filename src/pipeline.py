"""
Pipeline — thin, stateless batch runner for CLI-driven and automated operations.

Distinct from RAGService, which is stateful and long-lived (used by Streamlit).
Pipeline wraps RAGService for one-shot, scripted execution — suitable for:
  - CLI batch ingestion
  - Automated evaluation runs
  - CI/CD integration tests
  - Scheduled data refresh jobs
"""

from typing import List

from src.services.rag_service import RAGService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Pipeline:
    """
    A thin orchestration wrapper around RAGService for batch/scripted use.

    Instantiates RAGService once and exposes simple, high-level methods
    that log progress and handle output formatting — keeping main.py minimal.
    """

    def __init__(self) -> None:
        logger.info("Initializing Pipeline...")
        self._service = RAGService()
        logger.info("Pipeline ready.")

    def run_ingestion(self, domains: List[str], max_papers: int = 3) -> dict:
        """
        Runs the full ingestion workflow: scrape → standardize → chunk → embed → store.

        Args:
            domains:    List of domain names to query (must match ArxivScraper.DOMAIN_KEYWORDS).
            max_papers: Maximum number of papers to fetch per run.

        Returns:
            A stats dict with keys: papers_fetched, chunks_added.
        """
        logger.info(f"Starting ingestion | domains={domains} | max_papers={max_papers}")
        stats = self._service.ingest_papers(domains=domains, max_papers=max_papers)
        logger.info(
            f"Ingestion complete — "
            f"{stats['papers_fetched']} paper(s) → {stats['chunks_added']} chunk(s)."
        )
        return stats

    def run_query(self, query: str, top_k: int = 3) -> None:
        """
        Runs a single research query and pretty-prints the structured JSON result.

        Args:
            query:  The natural language research question.
            top_k:  Number of context chunks to retrieve.
        """
        logger.info(f"Running query: '{query}'")
        result = self._service.analyze_query(query=query, top_k=top_k)
        response = result["response"]

        print("\n" + "=" * 60)
        print("  STRUCTURED LLM ANALYSIS (JSON)")
        print("=" * 60)
        print(response.model_dump_json(indent=2))
        print("=" * 60 + "\n")

