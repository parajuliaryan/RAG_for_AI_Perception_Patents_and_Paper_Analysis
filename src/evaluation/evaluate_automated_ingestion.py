"""
End-to-End Automated Ingestion & Live Tech-Scouting Validation Pipeline.

Runs in a completely isolated directory (`data/automated_ingestion_test/`) so that
the main 22-document benchmark vector store (`data/vector_store`) and evaluation
files (`data/evaluation`) are never modified.

Workflow:
  - Phase 1 (Automated Ingestion & Standardization Audit):
      Scrapes 1 arXiv paper + 1 EPO/Google patent across all 6 AI perception domains
      (6 papers + 6 patents = 12 documents), runs DocumentStandardizer and
      StructuralChunker, indexes chunks into an isolated ChromaDB collection, and
      exports `ingestion_audit_summary.csv`.
  - Phase 2 (Live Tech-Scouting Retrieval & Structured Generation):
      Executes 12 domain-specific tech-scouting queries against the newly indexed
      corpus using `qwen2.5:72b` as Generator (`top_k = 15`), verifying target
      document retrieval and `FinalOutputSchema` completeness (`perception_findings`
      and `patent_metadata`).
  - Phase 3 (Reference-Free LLM-as-a-Judge Evaluation):
      Unloads the generator from GPU VRAM and evaluates all 12 responses using
      `llama3.1:70b` as Judge on two reference-free metrics:
        1. RAGAS `Faithfulness` (hallucination / context grounding check)
        2. Opik `AnswerRelevance` (direct technical responsiveness check)
      Exports `live_evaluation_results.csv` and `automated_ingestion_report.md`.

Usage (inside Docker container):
    docker exec -it rag-app python -m src.evaluation.evaluate_automated_ingestion
"""

import asyncio
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from datasets import Dataset
from dotenv import load_dotenv

# --- RAGAS IMPORT BUG FIX ---
mock_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
mock_vertex.ChatVertexAI = None
sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex
# ----------------------------

import src.config as cfg
from src.databases.vectorstore.chroma_store import ChromaStore
from src.embeddings.ollama_embedder import OllamaEmbedder
from src.evaluation.opik_judge import OllamaOpikJudgeModel
from src.generation.llm_client import GenerationPipeline
from src.ingestion.arxiv_scraper import ArxivScraper
from src.ingestion.patent_scraper import PatentScraper
from src.processing.chunker import StructuralChunker
from src.processing.standardizer import DocumentStandardizer
from src.retrieval.retriever import Retriever
from src.schemas.document import DocumentSchema
from src.utils.logger import get_logger

from opik.evaluation.metrics import AnswerRelevance
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.run_config import RunConfig

load_dotenv()
logger = get_logger("automated_ingestion_eval")

DOMAINS = [
    "Sensor Model Validation",
    "Sim2Real",
    "Virtual Homologation",
    "Neural Rendering",
    "World Model",
    "Driver Monitoring System",
]


def _setup_isolated_dirs() -> Dict[str, Path]:
    """Creates isolated directories under `data/automated_ingestion_test/` and redirects config paths."""
    base_test_dir = cfg.DATA_DIR / "automated_ingestion_test"
    dirs = {
        "base": base_test_dir,
        "raw_arxiv": base_test_dir / "raw" / "arxiv",
        "raw_patents": base_test_dir / "raw" / "patents",
        "raw_pdfs": base_test_dir / "raw" / "pdfs",
        "texts": base_test_dir / "texts",
        "chunks_debug": base_test_dir / "chunks_debug",
        "vector_store": base_test_dir / "vector_store",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Redirect cfg paths in-memory so PDFParser and debug savers write ONLY to isolated folders
    cfg.RAW_ARXIV_DIR = dirs["raw_arxiv"]
    cfg.RAW_PATENTS_DIR = dirs["raw_patents"]
    cfg.RAW_PDF_DIR = dirs["raw_pdfs"]
    cfg.EXTRACTED_TEXT_DIR = dirs["texts"]
    cfg.CHUNKS_DEBUG_DIR = dirs["chunks_debug"]
    cfg.VECTOR_STORE_DIR = dirs["vector_store"]

    return dirs


def _save_debug_chunks(chunks: List[Dict[str, Any]], debug_dir: Path) -> None:
    """Saves human-readable debug chunk files per parent document in the isolated folder."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        p_id = c.get("parent_id", "unknown")
        grouped.setdefault(p_id, []).append(c)

    for p_id, p_chunks in grouped.items():
        safe_id = p_id.replace("/", "_")
        out_path = debug_dir / f"{safe_id}_chunks.txt"
        p_chunks.sort(key=lambda x: x.get("metadata", {}).get("chunk_index", 0))
        meta0 = p_chunks[0].get("metadata", {}) if p_chunks else {}
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"=== ISOLATED AUTOMATED INGESTION CHUNKS: {p_id} ===\n")
            f.write(f"Total Chunks: {len(p_chunks)}\n")
            f.write(f"Source Type: {meta0.get('source', 'N/A')}\n")
            f.write(f"Title: {meta0.get('title', 'N/A')}\n")
            f.write(f"Affiliated Institutions: {meta0.get('affiliated_institutions') or 'N/A'}\n")
            if meta0.get("source") in ("patent", "epo_ops"):
                f.write(f"Priority / Grant Date: {meta0.get('priority_or_grant_date') or 'N/A'}\n")
                f.write(f"Patent Classifications: {meta0.get('patent_classifications') or 'N/A'}\n")
            f.write(f"Abstract: {meta0.get('abstract', 'N/A')}\n")
            if meta0.get("independent_claims"):
                f.write(f"Independent Claims:\n{meta0.get('independent_claims')}\n")
            f.write("\n")
            for c in p_chunks:
                idx = c.get("metadata", {}).get("chunk_index", 0)
                txt = c.get("text", "")
                f.write(f"{'=' * 20} CHUNK {idx} (Length: {len(txt)}) {'=' * 20}\n{txt}\n\n")


def _format_chunk_contexts(context_chunks: List[Dict[str, Any]]) -> List[str]:
    """Formats retrieved chunks with once-per-document metadata matching `evaluate.py`."""
    chunk_texts: List[str] = []
    seen_docs = set()

    for chunk in context_chunks:
        chunk_id = chunk.get("id", "N/A")
        parent_id = chunk.get("parent_id") or (
            chunk_id.rsplit("_chunk_", 1)[0] if "_chunk_" in chunk_id else chunk_id
        )
        meta = chunk.get("metadata", {}) or {}
        pub_no = meta.get("publication_number") or parent_id
        header_parts = [f"Source ID: {pub_no}"]

        if parent_id not in seen_docs:
            seen_docs.add(parent_id)
            if meta.get("affiliated_institutions"):
                header_parts.append(
                    f"Affiliated Institutions / Assignee: {meta['affiliated_institutions']}"
                )
            if meta.get("abstract") and meta.get("abstract") != "N/A":
                header_parts.append(f"Document Abstract: {meta['abstract']}")
            if meta.get("source") in ("patent", "epo_ops"):
                if meta.get("patent_classifications"):
                    header_parts.append(
                        f"Patent Classifications (IPC/CPC): {meta['patent_classifications']}"
                    )
                if meta.get("priority_or_grant_date"):
                    header_parts.append(
                        f"Patent Priority / Grant Date: {meta['priority_or_grant_date']}"
                    )
                if meta.get("independent_claims"):
                    header_parts.append(
                        f"Independent Claims Scope:\n{meta['independent_claims']}"
                    )

        header_parts.append(chunk["text"])
        chunk_texts.append("\n".join(header_parts))

    return chunk_texts


BLIND_TECH_SCOUTING_QUERIES: Dict[str, str] = {
    "CN120876698A": (
        "How can dynamic vehicle surface rendering and multi-modal LiDAR and camera sensor "
        "simulation be combined in open-source driving simulators to evaluate perception "
        "robustness against adversarial visual camouflage?"
    ),
    "2604.03497v2": (
        "How can vision-language model reward signals trained in simulation be transferred "
        "to physical autonomous vehicles without relying on simulator-native states or "
        "coupled action spaces?"
    ),
    "CN112783199A": (
        "What domain adaptation techniques allow deep reinforcement learning navigation "
        "policies trained on synthetic RGB imagery in virtual environments to generalize "
        "to real-world autonomous aerial vehicles?"
    ),
    "2507.20034v3": (
        "How can digital twins and hardware-in-the-loop robotic testbeds be integrated to "
        "verify and validate safety-critical guidance, navigation, and control perception "
        "pipelines for close-proximity autonomous operations?"
    ),
    "CN121118363A": (
        "How can multi-source real-world traffic accident recordings be cleaned and "
        "parameterized to automatically generate edge-case testing scenarios across "
        "software-in-the-loop (SIL) and hardware-in-the-loop (HIL) simulation platforms?"
    ),
    "2609.32175v1": (
        "How can single-step causal video diffusion models mitigate temporal drift and "
        "rendering artifacts in 3D Gaussian Splatting (3DGS) autonomous driving "
        "simulations at low latency?"
    ),
    "2611.02526v1": (
        "How can action-conditioned world models explicitly capture object-level "
        "multi-agent traffic interactions from driving videos rather than relying solely "
        "on dense bird's-eye-view (BEV) feature maps?"
    ),
    "CN121145974A": (
        "What methods exist for training offline model-based reinforcement learning agents "
        "using simulated intersection traffic environments and expert trajectory datasets "
        "for autonomous vehicle decision-making?"
    ),
    "2611.07941v1": (
        "How can synthetic facial image generation frameworks decouple identity "
        "representation from image synthesis to produce privacy-preserving training "
        "datasets for facial monitoring systems?"
    ),
}


def _build_query_for_doc(doc_id: str, domain: str, source: str, title: str) -> str:
    """
    Returns a blind, title-free technical research question so semantic retrieval is
    genuinely tested on domain concepts rather than matching the injected Document Title header.
    """
    if doc_id in BLIND_TECH_SCOUTING_QUERIES:
        return BLIND_TECH_SCOUTING_QUERIES[doc_id]
    if source == "arxiv":
        return (
            f"What recent methodologies, sensor modalities, simulation frameworks, and "
            f"quantitative KPIs are proposed for {domain} in autonomous systems?"
        )
    return (
        f"What system architectures, simulation methods, and independent patent claims "
        f"are disclosed for {domain} in autonomous driving?"
    )


def run_automated_ingestion_validation() -> None:
    dirs = _setup_isolated_dirs()
    base_dir = dirs["base"]

    ingested_ckpt_path = base_dir / "checkpoint_ingested_docs.json"
    audit_csv_path = base_dir / "ingestion_audit_summary.csv"
    gen_ckpt_path = base_dir / "checkpoint_generation.json"
    eval_ckpt_path = base_dir / "checkpoint_evaluation.json"
    eval_csv_path = base_dir / "live_evaluation_results.csv"
    report_md_path = base_dir / "automated_ingestion_report.md"

    print("==================================================================")
    print("  AUTOMATED INGESTION & LIVE TECH-SCOUTING VALIDATION (ISOLATED)")
    print("==================================================================")
    print(f"Isolated Working Directory : {base_dir}")
    print(f"Generator Model            : {cfg.GENERATOR_MODEL}")
    print(f"Judge Model                : {cfg.JUDGE_MODEL}")
    print(f"Embedding Model            : {cfg.EMBED_MODEL}")

    embedder = OllamaEmbedder(model_name=cfg.EMBED_MODEL)
    vector_store = ChromaStore(
        embedder=embedder,
        collection_name="automated_ingestion_test_docs",
        persist_dir=str(dirs["vector_store"]),
    )
    standardizer = DocumentStandardizer(use_full_text=True)
    chunker = StructuralChunker(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)

    # ------------------------------------------------------------------
    # PHASE 1: Automated Scraping, Standardization, Chunking & Indexing
    # ------------------------------------------------------------------
    print("\n--- PHASE 1: AUTOMATED SCRAPING, STANDARDIZATION & INDEXING ---")
    audit_records: List[Dict[str, Any]] = []

    if ingested_ckpt_path.exists() and vector_store.collection.count() > 0:
        with open(ingested_ckpt_path, "r", encoding="utf-8") as f:
            audit_records = json.load(f)
        print(
            f"Loaded {len(audit_records)} previously ingested documents from checkpoint "
            f"(ChromaDB chunk count: {vector_store.collection.count()})."
        )
    else:
        arxiv_scraper = ArxivScraper()
        patent_scraper = PatentScraper()

        seen_paper_ids = set()
        seen_patent_ids = set()

        for domain in DOMAINS:
            print(f"\n[Domain: {domain}]")

            # 1. Fetch 1 unique arXiv Paper for this domain
            try:
                p_query = arxiv_scraper.build_query(
                    selected_domains=[domain], start_year="2023", end_year="2026"
                )
                candidates = arxiv_scraper.fetch(query=p_query, max_results=4)
                selected_paper: DocumentSchema = None
                for cand in candidates:
                    if cand.id not in seen_paper_ids:
                        selected_paper = cand
                        seen_paper_ids.add(cand.id)
                        break

                if selected_paper:
                    print(f"  -> Fetched arXiv Paper: {selected_paper.id} | {selected_paper.title[:65]}...")
                    raw_path = dirs["raw_arxiv"] / f"{selected_paper.id.replace('/', '_')}.json"
                    raw_path.write_text(selected_paper.model_dump_json(indent=2), encoding="utf-8")

                    std_docs = standardizer.process([selected_paper])
                    chunks = chunker.chunk(std_docs)
                    _save_debug_chunks(chunks, dirs["chunks_debug"])
                    vector_store.add_documents(chunks)

                    std_doc = std_docs[0]
                    meta = std_doc["metadata"]
                    audit_records.append(
                        {
                            "domain": domain,
                            "source": "arxiv",
                            "doc_id": selected_paper.id,
                            "title": meta.get("title", selected_paper.title),
                            "published_date": meta.get("published_date", ""),
                            "full_text_chars": len(std_doc.get("text", "")),
                            "full_text_success": len(std_doc.get("text", "")) > 1500,
                            "abstract_extracted": bool(
                                meta.get("abstract")
                                and "no abstract" not in meta.get("abstract", "").lower()
                            ),
                            "affiliated_institutions": meta.get("affiliated_institutions", ""),
                            "patent_classifications": "N/A (Paper)",
                            "independent_claims_extracted": "N/A (Paper)",
                            "num_chunks": len(chunks),
                        }
                    )
                else:
                    print(f"  -> Warning: No unseen arXiv paper found for {domain}.")
            except Exception as exc:
                print(f"  -> Error scraping arXiv for {domain}: {exc}")

            # 2. Fetch 1 unique Patent (EPO OPS + Google Patents full text) for this domain
            try:
                pat_query = patent_scraper.build_query(
                    selected_domains=[domain], start_date="2020", end_date="2025"
                )
                pat_candidates = patent_scraper.fetch(query=pat_query, max_results=5)
                selected_patent: DocumentSchema = None

                # Prefer a candidate that has full_text (claims/description) scraped from Google Patents
                for cand in pat_candidates:
                    if cand.id not in seen_patent_ids and cand.full_text:
                        selected_patent = cand
                        seen_patent_ids.add(cand.id)
                        break
                if not selected_patent:
                    for cand in pat_candidates:
                        if cand.id not in seen_patent_ids:
                            selected_patent = cand
                            seen_patent_ids.add(cand.id)
                            break

                if selected_patent:
                    print(f"  -> Fetched EPO Patent : {selected_patent.id} | {selected_patent.title[:65]}...")
                    raw_path = dirs["raw_patents"] / f"{selected_patent.id.replace('/', '_')}.json"
                    raw_path.write_text(selected_patent.model_dump_json(indent=2), encoding="utf-8")

                    std_docs = standardizer.process([selected_patent])
                    chunks = chunker.chunk(std_docs)
                    _save_debug_chunks(chunks, dirs["chunks_debug"])
                    vector_store.add_documents(chunks)

                    std_doc = std_docs[0]
                    meta = std_doc["metadata"]
                    indep_claims = meta.get("independent_claims", "")
                    audit_records.append(
                        {
                            "domain": domain,
                            "source": "epo_ops",
                            "doc_id": selected_patent.id,
                            "title": meta.get("title", selected_patent.title),
                            "published_date": meta.get("published_date", ""),
                            "full_text_chars": len(std_doc.get("text", "")),
                            "full_text_success": len(std_doc.get("text", "")) > 1000,
                            "abstract_extracted": bool(
                                meta.get("abstract")
                                and "no abstract" not in meta.get("abstract", "").lower()
                            ),
                            "affiliated_institutions": meta.get("affiliated_institutions", ""),
                            "patent_classifications": meta.get("patent_classifications", ""),
                            "independent_claims_extracted": bool(indep_claims and len(indep_claims) > 20),
                            "num_chunks": len(chunks),
                        }
                    )
                else:
                    print(f"  -> Warning: No unseen patent found for {domain}.")
            except Exception as exc:
                print(f"  -> Error scraping patent for {domain}: {exc}")

            # Save checkpoint after each domain
            with open(ingested_ckpt_path, "w", encoding="utf-8") as f:
                json.dump(audit_records, f, indent=2)

    audit_df = pd.DataFrame(audit_records)
    audit_df.to_csv(audit_csv_path, index=False)
    print(f"\nIngestion Audit saved to: {audit_csv_path}")
    print(f"Total Documents Indexed : {len(audit_records)}")
    print(f"Total Chunks in ChromaDB: {vector_store.collection.count()}")

    # ------------------------------------------------------------------
    # PHASE 2: Live Tech-Scouting Retrieval & Structured Generation
    # ------------------------------------------------------------------
    print("\n--- PHASE 2: LIVE TECH-SCOUTING QUERIES & GENERATION ---")
    gen_records: List[Dict[str, Any]] = []
    if gen_ckpt_path.exists():
        with open(gen_ckpt_path, "r", encoding="utf-8") as f:
            gen_records = json.load(f)
        print(f"Loaded {len(gen_records)} previously generated responses from checkpoint.")

    done_doc_ids = {r["target_doc_id"] for r in gen_records}
    pending_docs = [doc for doc in audit_records if doc["doc_id"] not in done_doc_ids]

    # Disable Opik cloud tracing during batch generation to avoid network stalls
    cfg.ENABLE_OPIK = False

    if pending_docs:
        retriever = Retriever(vector_store=vector_store)
        print(f"\nStep 2a: Pre-retrieving Top-{cfg.TOP_K_DEFAULT} chunks for all {len(pending_docs)} queries...")
        pre_retrieved: List[Dict[str, Any]] = []
        for doc_info in pending_docs:
            q = _build_query_for_doc(
                doc_id=doc_info["doc_id"],
                domain=doc_info["domain"],
                source=doc_info["source"],
                title=doc_info["title"],
            )
            chunks = retriever.retrieve(query=q, top_k=cfg.TOP_K_DEFAULT)
            pre_retrieved.append({"doc_info": doc_info, "question": q, "chunks": chunks})

        # Unload embedder (~2.8 GB VRAM) ONCE before loading the 72B generator
        try:
            print(f"Unloading embedder ({cfg.EMBED_MODEL}) from VRAM to maximize GPU memory for {cfg.GENERATOR_MODEL}...")
            requests.post(
                f"{cfg.OLLAMA_HOST}/api/generate",
                json={"model": cfg.EMBED_MODEL, "keep_alive": 0},
                timeout=15,
            )
        except Exception:
            pass

        generator = GenerationPipeline(model_name=cfg.GENERATOR_MODEL)
        from langchain_ollama import ChatOllama
        from src.schemas.response import FinalOutputSchema
        CompactLLM = ChatOllama(
            model=cfg.GENERATOR_MODEL,
            base_url=cfg.OLLAMA_HOST,
            temperature=0.0,
            num_ctx=12288,
        )
        generator.structured_llm = CompactLLM.with_structured_output(FinalOutputSchema)

        for idx, item_bundle in enumerate(pre_retrieved, start=1):
            doc_info = item_bundle["doc_info"]
            doc_id = doc_info["doc_id"]
            domain = doc_info["domain"]
            source = doc_info["source"]
            title = doc_info["title"]
            question = item_bundle["question"]
            retrieved_chunks = item_bundle["chunks"]

            print(f"\nGenerating [{idx}/{len(pre_retrieved)}] ({domain} | {source} | {doc_id}):")
            print(f"  Query: {question[:90]}...")

            try:
                structured_resp = generator.generate(query=question, context_chunks=retrieved_chunks)
                json_answer = structured_resp.model_dump_json(indent=2)
                formatted_contexts = _format_chunk_contexts(retrieved_chunks)

                # Check if target doc was retrieved in Top-K
                retrieved_parent_ids = {
                    c.get("parent_id") or c.get("id", "").rsplit("_chunk_", 1)[0]
                    for c in retrieved_chunks
                }
                target_in_topk = doc_id in retrieved_parent_ids
                resp_dict = structured_resp.model_dump()
                has_findings = len(resp_dict.get("perception_findings") or []) > 0
                has_patent_meta = resp_dict.get("patent_metadata") is not None

                gen_records.append(
                    {
                        "domain": domain,
                        "source": source,
                        "target_doc_id": doc_id,
                        "target_title": title,
                        "question": question,
                        "target_retrieved_in_topk": target_in_topk,
                        "cited_doc_id": resp_dict.get("source_document_id", ""),
                        "perception_findings_count": len(resp_dict.get("perception_findings") or []),
                        "perception_findings_populated": has_findings,
                        "patent_metadata_populated": has_patent_meta,
                        "answer": json_answer,
                        "contexts": formatted_contexts,
                    }
                )
                with open(gen_ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(gen_records, f, indent=2)
            except Exception as exc:
                print(f"  -> Error generating for {doc_id}: {exc}")

        # Unload GENERATOR_MODEL from GPU VRAM before loading JUDGE_MODEL
        if cfg.GENERATOR_MODEL != cfg.JUDGE_MODEL:
            try:
                print(f"\nUnloading generator model ({cfg.GENERATOR_MODEL}) from GPU VRAM...")
                requests.post(
                    f"{cfg.OLLAMA_HOST}/api/generate",
                    json={"model": cfg.GENERATOR_MODEL, "keep_alive": 0},
                    timeout=30,
                )
            except Exception as exc:
                print(f"Note: Could not unload generator model ({exc}).")

    # ------------------------------------------------------------------
    # PHASE 3: Reference-Free Evaluation (Opik AnswerRelevance + RAGAS Faithfulness)
    # ------------------------------------------------------------------
    print("\n--- PHASE 3: REFERENCE-FREE EVALUATION (FAITHFULNESS & ANSWER RELEVANCE) ---")
    existing_eval_map: Dict[str, Dict[str, Any]] = {}
    if eval_ckpt_path.exists():
        with open(eval_ckpt_path, "r", encoding="utf-8") as f:
            loaded_eval = json.load(f)
        for r in loaded_eval:
            existing_eval_map[r["target_doc_id"]] = r
        print(f"Loaded {len(loaded_eval)} previously evaluated entries from checkpoint.")

    scored_doc_ids = {
        doc_id
        for doc_id, r in existing_eval_map.items()
        if r.get("opik_answer_relevance") is not None and r.get("ragas_faithfulness") is not None
    }
    eval_records: List[Dict[str, Any]] = [
        existing_eval_map[r["target_doc_id"]]
        for r in gen_records
        if r["target_doc_id"] in scored_doc_ids
    ]

    to_score = [r for r in gen_records if r["target_doc_id"] not in scored_doc_ids]

    if to_score:
        import nest_asyncio
        nest_asyncio.apply()

        from langchain_ollama import ChatOllama, OllamaEmbeddings

        class SyncToAsyncOllama(ChatOllama):
            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                sync_run_manager = run_manager.get_sync() if run_manager else None
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, lambda: self._generate(messages, stop, sync_run_manager, **kwargs)
                )

        class SyncToAsyncOllamaEmbeddings(OllamaEmbeddings):
            async def aembed_documents(self, texts):
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self.embed_documents, texts)

            async def aembed_query(self, text):
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self.embed_query, text)

        ragas_judge_llm = SyncToAsyncOllama(
            model=cfg.JUDGE_MODEL,
            base_url=cfg.OLLAMA_HOST,
            temperature=0.1,
            num_ctx=16384,
            client_kwargs={"timeout": 600.0},
        )
        ragas_embeddings = SyncToAsyncOllamaEmbeddings(
            model=cfg.EMBED_MODEL,
            base_url=cfg.OLLAMA_HOST,
        )
        run_config = RunConfig(timeout=1200, max_retries=1, max_workers=1)

        opik_judge_model = OllamaOpikJudgeModel(
            model_name=cfg.JUDGE_MODEL,
            base_url=cfg.OLLAMA_HOST,
            temperature=0.0,
            num_ctx=16384,
            timeout=600.0,
        )
        opik_ar_metric = AnswerRelevance(
            model=opik_judge_model,
            require_context=False,
            track=False,
        )

        for idx, item in enumerate(to_score, start=1):
            doc_id = item["target_doc_id"]
            prev = existing_eval_map.get(doc_id, {})
            print(
                f"\nEvaluating [{idx}/{len(to_score)}] ({item['domain']} | {doc_id})..."
            )
            # 1. Opik AnswerRelevance (reuse if already computed in checkpoint)
            ar_score = prev.get("opik_answer_relevance")
            ar_reason = prev.get("opik_answer_relevance_reason", "")
            if ar_score is None:
                try:
                    res_ar = opik_ar_metric.score(
                        input=item["question"],
                        output=item["answer"],
                    )
                    ar_score = float(res_ar.value)
                    ar_reason = str(res_ar.reason or "")
                except Exception as exc:
                    print(f"  -> Opik AnswerRelevance failed: {exc}")
            else:
                print(f"  -> Reusing saved Opik AnswerRelevance: {ar_score}")

            # 2. RAGAS Faithfulness (prioritize chunks from cited/target doc up to 6 chunks so NLI prompt never overflows num_ctx)
            faith_score = prev.get("ragas_faithfulness")
            if faith_score is None:
                try:
                    cited_id = item.get("cited_doc_id") or doc_id
                    all_ctx = item["contexts"]
                    primary_ctx = [c for c in all_ctx if f"Source ID: {cited_id}" in c]
                    other_ctx = [c for c in all_ctx if f"Source ID: {cited_id}" not in c]
                    # Cap individual context strings at 5000 chars so long Google Patent claims never overflow num_ctx
                    focused_ctx = [
                        (c[:2500] + "\n...\n" + c[-2500:] if len(c) > 5500 else c)
                        for c in (primary_ctx + other_ctx)[:6]
                    ]

                    ds = Dataset.from_dict(
                        {
                            "question": [item["question"]],
                            "answer": [item["answer"]],
                            "contexts": [focused_ctx],
                        }
                    )
                    ragas_res = evaluate(
                        dataset=ds,
                        metrics=[faithfulness],
                        llm=ragas_judge_llm,
                        embeddings=ragas_embeddings,
                        run_config=run_config,
                        raise_exceptions=False,
                    )
                    df_res = ragas_res.to_pandas()
                    if "faithfulness" in df_res.columns and pd.notna(df_res["faithfulness"].iloc[0]):
                        faith_score = float(df_res["faithfulness"].iloc[0])
                except Exception as exc:
                    print(f"  -> RAGAS Faithfulness failed: {exc}")

            print(
                f"  -> RAGAS Faithfulness: {faith_score} | Opik AnswerRelevance: {ar_score}"
            )

            eval_records.append(
                {
                    **{k: v for k, v in item.items() if k != "contexts"},
                    "ragas_faithfulness": faith_score,
                    "opik_answer_relevance": ar_score,
                    "opik_answer_relevance_reason": ar_reason,
                }
            )
            with open(eval_ckpt_path, "w", encoding="utf-8") as f:
                json.dump(eval_records, f, indent=2)

    # ------------------------------------------------------------------
    # PHASE 4: Export Summary CSV & Thesis Markdown Report
    # ------------------------------------------------------------------
    eval_df = pd.DataFrame(eval_records)
    csv_cols = [
        c
        for c in [
            "domain",
            "source",
            "target_doc_id",
            "target_title",
            "target_retrieved_in_topk",
            "cited_doc_id",
            "perception_findings_count",
            "perception_findings_populated",
            "patent_metadata_populated",
            "ragas_faithfulness",
            "opik_answer_relevance",
            "opik_answer_relevance_reason",
            "question",
            "answer",
        ]
        if c in eval_df.columns
    ]
    eval_df[csv_cols].to_csv(eval_csv_path, index=False)

    mean_faith = eval_df["ragas_faithfulness"].dropna().mean() if not eval_df.empty else 0.0
    mean_ar = eval_df["opik_answer_relevance"].dropna().mean() if not eval_df.empty else 0.0
    topk_hit_rate = (
        eval_df["target_retrieved_in_topk"].mean() * 100.0 if not eval_df.empty else 0.0
    )
    findings_rate = (
        eval_df["perception_findings_populated"].mean() * 100.0 if not eval_df.empty else 0.0
    )

    md_lines = [
        "# Automated Ingestion & Live Tech-Scouting Validation Report",
        "",
        "## 1. Executive Summary",
        f"- **Total Documents Automatically Ingested:** {len(audit_records)} (across {len(DOMAINS)} AI perception domains)",
        f"- **Total Chunks Indexed in Isolated ChromaDB:** {vector_store.collection.count()}",
        f"- **Top-K Target Document Retrieval Hit Rate:** {topk_hit_rate:.1f}%",
        f"- **`perception_findings` Extraction Completeness:** {findings_rate:.1f}%",
        f"- **Mean RAGAS Faithfulness (`{cfg.JUDGE_MODEL}` Judge):** `{mean_faith:.4f}`",
        f"- **Mean Opik Answer Relevance (`{cfg.JUDGE_MODEL}` Judge):** `{mean_ar:.4f}`",
        "",
        "## 2. Automated Ingestion & Standardization Audit Table",
        "| Domain | Source | Document ID | Full-Text Chars | Abstract Extracted | Claims Extracted | Chunks Produced |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in audit_records:
        md_lines.append(
            f"| {r['domain']} | `{r['source']}` | `{r['doc_id']}` | {r['full_text_chars']} | "
            f"{r['abstract_extracted']} | {r['independent_claims_extracted']} | {r['num_chunks']} |"
        )

    md_lines.extend(
        [
            "",
            "## 3. Live Tech-Scouting Evaluation Table",
            "| Domain | Source | Target Doc ID | Retrieved in Top-K | Findings Count | RAGAS Faithfulness | Opik Answer Relevance |",
            "| :--- | :---: | :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for r in eval_records:
        f_val = f"{r['ragas_faithfulness']:.4f}" if r.get("ragas_faithfulness") is not None else "N/A"
        ar_val = (
            f"{r['opik_answer_relevance']:.4f}"
            if r.get("opik_answer_relevance") is not None
            else "N/A"
        )
        md_lines.append(
            f"| {r['domain']} | `{r['source']}` | `{r['target_doc_id']}` | "
            f"{r['target_retrieved_in_topk']} | {r['perception_findings_count']} | {f_val} | {ar_val} |"
        )

    md_lines.extend(["", "## 4. Qualitative Output Inspection (Per Document)", ""])
    for i, r in enumerate(eval_records, start=1):
        md_lines.extend(
            [
                f"### {i}. [{r['domain']}] `{r['target_doc_id']}` — {r['target_title']}",
                f"- **Query:** {r['question']}",
                f"- **RAGAS Faithfulness:** `{r.get('ragas_faithfulness')}` | **Opik Answer Relevance:** `{r.get('opik_answer_relevance')}`",
                f"- **Judge Reason:** {r.get('opik_answer_relevance_reason', '')}",
                "```json",
                r["answer"],
                "```",
                "",
            ]
        )

    report_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n==================================================================")
    print("  AUTOMATED INGESTION VALIDATION COMPLETE!")
    print("==================================================================")
    print(f"Mean RAGAS Faithfulness     : {mean_faith:.4f}")
    print(f"Mean Opik Answer Relevance  : {mean_ar:.4f}")
    print(f"Ingestion Audit CSV         : {audit_csv_path}")
    print(f"Live Evaluation CSV         : {eval_csv_path}")
    print(f"Full Thesis Markdown Report : {report_md_path}")


if __name__ == "__main__":
    run_automated_ingestion_validation()
