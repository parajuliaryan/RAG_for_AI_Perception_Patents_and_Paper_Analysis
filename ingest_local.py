import os
import argparse
from pathlib import Path
from src.schemas.document import DocumentSchema
from src.services.rag_service import RAGService
from src.utils.logger import get_logger

logger = get_logger("local_ingestion")

def ingest_local_directory(dir_path: str):
    """
    Scans a local directory for PDFs and ingests them into the RAG pipeline.
    """
    directory = Path(dir_path)
    if not directory.exists() or not directory.is_dir():
        logger.error(f"Directory not found: {directory}")
        return

    documents = []
    total_files = 0
    
    # Process Papers
    papers_dir = directory / "papers"
    if papers_dir.exists():
        for pdf_path in papers_dir.glob("*.pdf"):
            total_files += 1
            doc_id = pdf_path.stem
            doc = DocumentSchema(
                source="arxiv",  # Tagged as a paper
                id=doc_id,
                title=doc_id.replace("_", " ").title(),
                authors=["Manual Upload"],
                abstract="[No abstract provided. See full text.]",
                published_date="Unknown",
                local_path=str(pdf_path.absolute())
            )
            documents.append(doc)

    # Process Patents
    patents_dir = directory / "patents"
    if patents_dir.exists():
        for pdf_path in patents_dir.glob("*.pdf"):
            total_files += 1
            doc_id = pdf_path.stem
            doc = DocumentSchema(
                source="patent",  # Tagged as a patent
                id=doc_id,
                title=doc_id.replace("_", " ").title(),
                authors=["Manual Upload"],
                abstract="[No abstract provided. See full text.]",
                published_date="Unknown",
                local_path=str(pdf_path.absolute())
            )
            documents.append(doc)

    if not documents:
        logger.warning(f"No PDFs found! Please make sure your PDFs are inside {papers_dir} or {patents_dir}")
        return

    logger.info(f"Found {total_files} local PDFs. Preparing for ingestion...")

    # Initialize the RAG pipeline components
    service = RAGService()

    logger.info("Standardizing and extracting text from local PDFs...")
    std_docs = service.standardizer.process(documents)
    
    logger.info("Applying Hybrid Structural Chunking...")
    chunks = service.chunker.chunk(std_docs)
    
    logger.info(f"Produced {len(chunks)} chunks from {total_files} manual documents.")
    
    # Save chunks to chunks_debug/ so you can verify them
    service._save_chunks_for_debug(chunks)
    
    logger.info("Embedding and indexing into ChromaDB...")
    service.vector_store.add_documents(chunks)
    
    logger.info("Successfully ingested local documents into the Vector Store!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest local PDFs into the RAG database.")
    parser.add_argument("--dir", type=str, default="data/manual_docs", help="Directory containing PDFs to ingest")
    args = parser.parse_args()
    
    # Auto-create the directories so they are ready for the user to drop files into
    os.makedirs(os.path.join(args.dir, "papers"), exist_ok=True)
    os.makedirs(os.path.join(args.dir, "patents"), exist_ok=True)
    
    ingest_local_directory(args.dir)
