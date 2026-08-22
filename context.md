# Project Context: Automotive AI Perception RAG Pipeline

## 1. Project Overview & Research Goal
This project is a Retrieval-Augmented Generation (RAG) pipeline built for a master's thesis. 
The core research objectives are to answer: *" Q1: How do different vector embedding models compare in retrieval precision when processing complex automotive engineering publications?"*
*"Q2: How can a local RAG system process both academic papers and complex patent files without losing the actual technical meaning of the text?
Q3: Can an open-source LLM reliably pull specific engineering data from these texts and format it into JSON files without hallucination? (Using AI perception as the base case)
Q4: How accurately can the automated evaluation methodologies quantify trustworthiness and accuracy without relying on ground-truth human data?
Q5: How should the pipeline be designed so it can be easily scaled and reused for other engineering topics in the future?"*
"Q6: What is the measurable hallucination rate of an open-source LLM 
(e.g., Llama-3) when extracting Technology Readiness Levels (TRL) 
from unstructured patent data?"

One task also is to evaluate open-source llms and embedding models (but that is mostly focused on report with little coding involvement I guess)

The system ingests academic literature (currently arXiv, soon Patents), processes it, stores it in a local vector database, and uses a local LLM to extract highly structured, deterministic JSON analyses of perception technologies, capabilities, and limitations.

## 2. Tech Stack & Architecture
* **Language & Env:** Python 3.12, Docker (docker-compose)
* **LLM & Embeddings Engine:** Local Ollama (`llama3` 8B for generation, `nomic-embed-text` for embeddings).
* **Frameworks:** `langchain-ollama` (for base LLM/embedder wrappers), `langchain-text-splitters`, `pydantic` (for strict schema enforcement).
* **Vector Database:** ChromaDB (Persistent local SQLite/HNSW store).
* **Frontend:** Streamlit (Object-Oriented implementation).

### Design Philosophy
1. **Clean Architecture:** Strict separation of concerns. The CLI (`main.py`) and UI (`app.py`) contain **zero** pipeline logic. They act only as interfaces that call the central `RAGService`.
2. **Object-Oriented Programming (OOP):** All modules (Scraper, Chunker, Embedder) are encapsulated classes with single responsibilities. 
3. **Deterministic Output:** The generation layer strictly utilizes `.with_structured_output()` mapped to Pydantic schemas to guarantee machine-readable JSON over conversational text.

## 3. Directory Structure
```text
.
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .gitignore
├── .env                  # (Optional: API keys, paths)
├── data/
│   ├── raw/              # Raw ingested JSON/PDFs (Ignored in Git)
│   └── vector_store/     # ChromaDB persistent storage (Ignored in Git)
└── src/
    ├── ingestion/        # arXiv scraper, patent scraper
    ├── processing/       # Document standardizer, Langchain recursive chunker
    ├── embeddings/       # Base embedder interfaces, Ollama embedder implementation
    ├── databases/        # ChromaDB setup and vector operations
    ├── generation/       # LLM client with Pydantic structured output
    ├── schemas/          # Pydantic models (e.g., FinalOutputSchema)
    ├── services/         # rag_service.py (Central pipeline orchestrator)
    └── frontend/         # app.py (Object-Oriented Streamlit dashboard)