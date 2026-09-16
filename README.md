# AutoPercept-RAG: Automated Tech Scouting for AI Perception

A local, privacy-preserving Retrieval-Augmented Generation (RAG) framework designed to ingest, process, and extract structured engineering parameters from heterogeneous technical literature.

---

## 📌 Overview

Tracking emerging trends in autonomous driving (AD) perception and synthetic sensor simulation is challenging due to the rapid influx of academic pre-prints and patent filings. **AutoPercept-RAG** automates technology scouting by parsing dense, multi-modal technical documents and transforming unstructured claims into standardized, validated JSON schemas for downstream systems engineering.

---

## ✨ Key Features

* **Localized & Private Execution:** Built entirely with open-source models and vector stores to ensure strict data sovereignty without third-party API exposure.
* **Heterogeneous Ingestion:** Handles both mathematically dense academic papers (e.g., arXiv pre-prints) and complex, legally formatted patent claims.
* **Structured Data Extraction:** Directly extracts multi-modal perception specifications—including sensor modalities, simulation platforms, testing scenarios, and evaluated KPIs—into schema-enforced JSON objects.
* **Automated Framework Evaluation:** Benchmarks retrieval precision and hallucination resistance using reference-free evaluation metrics (RAGAS).

---

## 🛠️ Architecture Pipeline

1. **Ingestion & Parsing:** Automated retrieval and document cleaning across academic papers and patent documents.
2. **Chunking & Vector Store:** Semantic-aware chunking and local embedding generation stored in a persistent vector database.
3. **Prompt Orchestration:** Constraint-guided generation forcing strict schema compliance and explicit handling of missing parameters.
4. **Evaluation:** End-to-end scoring of context relevance, precision, and generative faithfulness.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Ollama / Local LLM runtime
* Docker
