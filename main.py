import json
from pathlib import Path
from src.ingestion.arxiv_scraper import ArxivScraper
from src.processing.standardizer import DocumentStandardizer
from src.processing.chunker import RecursiveChunker
from src.embeddings.ollama_embedder import OllamaEmbedder
from src.databases.vectorstore.chroma_store import ChromaStore
from src.generation.llm_client import GenerationPipeline

if __name__ == "__main__":
    # --- 1. Scrape arXiv ---
    scraper = ArxivScraper()
    query = scraper.build_query(
        selected_domains=["Sensors & Environment", "Perception & World models"],
        start_year="2024",
        end_year="2026"
    )
    print("Fetching arXiv papers...")
    papers = scraper.fetch(query=query, max_results=3)

    # --- 2. Persist Raw JSON ---
    raw_dir = Path("data/raw/arxiv")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for paper in papers:
        file_path = raw_dir / f"{paper.id.replace('/', '_')}.json"
        file_path.write_text(paper.model_dump_json(indent=2), encoding="utf-8")

    # --- 3. Standardize & Chunk ---
    standardizer = DocumentStandardizer()
    standardized_docs = standardizer.process(papers)

    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk(standardized_docs)

    # --- 4. Embed & Store in ChromaDB ---
    embedder = OllamaEmbedder(model_name="nomic-embed-text")
    vector_store = ChromaStore(embedder=embedder, persist_dir="data/vector_store")
    vector_store.add_documents(chunks)

    # --- 5. Retrieve Context ---
    user_query = "What are the capabilities and limitations of LiDAR simulation and occupancy networks?"
    print(f"\nSearching ChromaDB for: '{user_query}'...")
    retrieved_chunks = vector_store.search(query=user_query, top_k=3)

    # --- 6. Generate Structured Answer ---
    generator = GenerationPipeline(model_name="llama3")
    analysis = generator.generate(query=user_query, context_chunks=retrieved_chunks)

    # --- 7. Print Final Output ---
    print("\n" + "=" * 60)
    print("FINAL STRUCTURED LLM ANALYSIS (JSON):")
    print("=" * 60)
    print(json.dumps(analysis.model_dump(), indent=2))