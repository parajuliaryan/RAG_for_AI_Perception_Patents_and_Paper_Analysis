"""
RAGService — central pipeline orchestrator.

The single interface that both the CLI (via Pipeline) and the Streamlit UI
use to interact with the pipeline. Holds all components as long-lived
instance attributes so they are initialized only once per process.

Frontends and scripts must never import individual pipeline components
directly — everything flows through this service.
"""

from typing import List, Dict, Any

import src.config as cfg
from src.ingestion.arxiv_scraper import ArxivScraper
from src.processing.standardizer import DocumentStandardizer
from src.processing.chunker import StructuralChunker
from src.embeddings.ollama_embedder import OllamaEmbedder
from src.databases.vectorstore.chroma_store import ChromaStore
from src.retrieval.retriever import Retriever
from src.generation.llm_client import GenerationPipeline
from src.schemas.response import FinalOutputSchema
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGService:
    """
    Central orchestrator for the AI Perception RAG pipeline.

    Initializes and wires together all pipeline components using values
    from src/config.py. Exposes three public methods:

      - ingest_papers()  — scrape → process → embed → store
      - analyze_query()  — retrieve → generate structured JSON
      - get_db_count()   — inspect vector store size
    """

    def __init__(self) -> None:
        logger.info("Initializing RAGService...")

        self.scraper = ArxivScraper()
        self.standardizer = DocumentStandardizer()
        self.chunker = StructuralChunker(
            chunk_size=cfg.CHUNK_SIZE,
            chunk_overlap=cfg.CHUNK_OVERLAP,
        )
        self.embedder = OllamaEmbedder(model_name=cfg.EMBED_MODEL)
        self.vector_store = ChromaStore(
            embedder=self.embedder,
            collection_name=cfg.COLLECTION_NAME,
            persist_dir=str(cfg.VECTOR_STORE_DIR),
        )
        self.retriever = Retriever(vector_store=self.vector_store)
        self.generator = GenerationPipeline(model_name=cfg.LLM_MODEL)

        logger.info("RAGService initialized successfully.")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_papers(self, domains: List[str], max_papers: int = 3) -> Dict[str, int]:
        """
        Scrapes arXiv, standardizes, chunks, and indexes documents into ChromaDB.

        Args:
            domains:    Domain names to query (matched against ArxivScraper.DOMAIN_KEYWORDS).
            max_papers: Maximum number of papers to fetch per call.

        Returns:
            Dict with keys: papers_fetched (int), chunks_added (int).
        """
        logger.info(f"Starting ingestion | domains={domains} | max_papers={max_papers}")

        query = self.scraper.build_query(
            selected_domains=domains,
            start_year="2024",
            end_year="2026",
        )
        papers = self.scraper.fetch(query=query, max_results=max_papers)

        if not papers:
            logger.warning("No papers fetched — check your query or network connection.")
            return {"papers_fetched": 0, "chunks_added": 0}

        # Filter out papers we have already ingested
        new_papers = []
        cfg.RAW_ARXIV_DIR.mkdir(parents=True, exist_ok=True)
        
        for paper in papers:
            file_path = cfg.RAW_ARXIV_DIR / f"{paper.id.replace('/', '_')}.json"
            if file_path.exists():
                logger.debug(f"Paper {paper.id} already exists in dataset. Skipping.")
                continue
            
            # Save the new raw JSON
            file_path.write_text(paper.model_dump_json(indent=2), encoding="utf-8")
            new_papers.append(paper)
            
        if not new_papers:
            logger.info("No new papers found. All fetched papers are already in the database.")
            return {"papers_fetched": len(papers), "chunks_added": 0}

        logger.info(f"Found {len(new_papers)} new paper(s) to process and embed.")

        # Process → chunk → embed → store ONLY the new papers
        std_docs = self.standardizer.process(new_papers)
        chunks = self.chunker.chunk(std_docs)
        logger.info(f"Produced {len(chunks)} chunk(s) from {len(new_papers)} new paper(s).")

        self._save_chunks_for_debug(chunks)

        self.vector_store.add_documents(chunks)

        return {"papers_fetched": len(papers), "chunks_added": len(chunks)}

    def ingest_patents(self, domains: List[str], max_patents: int = 5) -> Dict[str, int]:
        """
        Scrapes Crossref for patents, standardizes, chunks, and indexes into ChromaDB.

        Args:
            domains:     Domain names to query (matched against PatentScraper.DOMAIN_KEYWORDS).
            max_patents: Maximum number of patents to fetch per call.

        Returns:
            Dict with keys: patents_fetched (int), chunks_added (int).
        """
        logger.info(f"Starting patent ingestion | domains={domains} | max_patents={max_patents}")
        
        from src.ingestion.patent_scraper import PatentScraper
        scraper = PatentScraper()
        
        query = scraper.build_query(
            selected_domains=domains,
            start_date="2020",
            end_date="2025",
        )
        patents = scraper.fetch(query=query, max_results=max_patents)

        if not patents:
            logger.warning("No patents fetched — check your query or network connection.")
            return {"patents_fetched": 0, "chunks_added": 0}

        # Filter out patents we have already ingested
        new_patents = []
        cfg.RAW_PATENTS_DIR.mkdir(parents=True, exist_ok=True)
        
        for patent in patents:
            file_path = cfg.RAW_PATENTS_DIR / f"{patent.id.replace('/', '_')}.json"
            if file_path.exists():
                logger.debug(f"Patent {patent.id} already exists in dataset. Skipping.")
                continue
                
            # Save the new raw JSON
            file_path.write_text(patent.model_dump_json(indent=2), encoding="utf-8")
            new_patents.append(patent)
            
        if not new_patents:
            logger.info("No new patents found. All fetched patents are already in the database.")
            return {"patents_fetched": len(patents), "chunks_added": 0}

        logger.info(f"Found {len(new_patents)} new patent(s) to process and embed.")

        # Process → chunk → embed → store ONLY the new patents
        std_docs = self.standardizer.process(new_patents)
        chunks = self.chunker.chunk(std_docs)
        logger.info(f"Produced {len(chunks)} chunk(s) from {len(new_patents)} new patent(s).")
        
        self._save_chunks_for_debug(chunks)

        self.vector_store.add_documents(chunks)

        return {"patents_fetched": len(patents), "chunks_added": len(chunks)}
        
    def _save_chunks_for_debug(self, chunks: List[Dict[str, Any]]) -> None:
        """Groups chunks by parent document and writes them to human-readable text files."""
        if not chunks:
            return
            
        grouped_chunks = {}
        for chunk in chunks:
            p_id = chunk.get("parent_id", "unknown_parent")
            if p_id not in grouped_chunks:
                grouped_chunks[p_id] = []
            grouped_chunks[p_id].append(chunk)
            
        for p_id, p_chunks in grouped_chunks.items():
            safe_id = p_id.replace('/', '_')
            debug_path = cfg.CHUNKS_DEBUG_DIR / f"{safe_id}_chunks.txt"
            
            p_chunks.sort(key=lambda x: x.get("metadata", {}).get("chunk_index", 0))
            
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"=== CHUNK DEBUG FOR: {p_id} ===\n")
                f.write(f"Total Chunks: {len(p_chunks)}\n\n")
                for c in p_chunks:
                    c_idx = c.get("metadata", {}).get("chunk_index", 0)
                    c_text = c.get("text", "")
                    f.write(f"{'=' * 20} CHUNK {c_idx} (Length: {len(c_text)}) {'=' * 20}\n")
                    f.write(c_text + "\n\n")

    # ------------------------------------------------------------------
    # Query & Generation
    # ------------------------------------------------------------------

    def analyze_query(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Retrieves relevant context chunks and generates a structured LLM response.

        Args:
            query:  The user's natural language research question.
            top_k:  Number of chunks to retrieve. Defaults to cfg.TOP_K_DEFAULT.

        Returns:
            Dict with keys:
              - response (FinalOutputSchema): validated Pydantic model
              - context  (List[dict]):        raw retrieved chunks
        """
        effective_top_k = top_k if top_k is not None else cfg.TOP_K_DEFAULT

        retrieved_chunks = self.retriever.retrieve(query=query, top_k=effective_top_k)
        structured_response: FinalOutputSchema = self.generator.generate(
            query=query,
            context_chunks=retrieved_chunks,
        )

        return {
            "response": structured_response,
            "context": retrieved_chunks,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_db_count(self) -> int:
        """Returns the total number of chunks stored in the ChromaDB collection."""
        return self.vector_store.collection.count()