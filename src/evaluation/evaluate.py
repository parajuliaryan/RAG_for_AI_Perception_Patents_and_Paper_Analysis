import os
import sys
import types
import time
import json
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# --- RAGAS IMPORT BUG FIX ---
# Ragas has a hardcoded import to an old LangChain VertexAI path that was removed.
# We mock it here so Ragas doesn't crash when it boots up.
mock_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
mock_vertex.ChatVertexAI = None
sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex
# ----------------------------

import src.config as cfg
from src.services.rag_service import RAGService

# Ragas imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Local embeddings and models for fully independent Ragas

# Opik tracing (Optional but recommended)
try:
    from opik.integrations.ragas import RagasOpikTracer
    opik_tracer = RagasOpikTracer()
    callbacks = [opik_tracer]
except ImportError:
    callbacks = []

load_dotenv()

def run_evaluation(csv_path: str):
    print(f"Loading ground truth dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Please make sure you saved your Excel sheet as a CSV here.")
        return

    # Ensure required columns exist
    if not {'question', 'ground_truth'}.issubset(df.columns):
        print("Error: CSV must contain 'question' and 'ground_truth' columns.")
        return

    rag_service = RAGService()
    
    checkpoint_path = os.path.join("data", "evaluation", "checkpoint.json")
    processed_data = []
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
        print(f"Loaded {len(processed_data)} previously processed queries from checkpoint.")
        
    processed_questions = {item["question"] for item in processed_data}
    
    print("\n--- PHASE 1: GENERATING ANSWERS ---")
    for idx, row in df.iterrows():
        question = str(row['question'])
        gt = str(row['ground_truth'])
        
        if question.strip() == "nan" or not question.strip():
            continue
            
        if question in processed_questions:
            print(f"Skipping {idx + 1}/{len(df)} (Already processed): {question[:50]}...")
            continue
            
        print(f"\nProcessing Query {idx + 1}/{len(df)}: {question[:50]}...")
        
        try:
            # 1. Run RAG Pipeline
            result = rag_service.analyze_query(query=question, top_k=cfg.TOP_K_DEFAULT)
            
            # 2. Extract JSON answer and Contexts (include document-level metadata once per unique parent doc)
            json_answer = result["response"].model_dump_json(indent=2)
            chunk_texts = []
            seen_docs = set()
            for chunk in result["context"]:
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
                        header_parts.append(f"Affiliated Institutions / Assignee: {meta['affiliated_institutions']}")
                    if meta.get("abstract") and meta.get("abstract") != "N/A":
                        header_parts.append(f"Document Abstract: {meta['abstract']}")
                    if meta.get("source") in ("patent", "epo_ops"):
                        if meta.get("patent_classifications"):
                            header_parts.append(f"Patent Classifications (IPC/CPC): {meta['patent_classifications']}")
                        if meta.get("priority_or_grant_date"):
                            header_parts.append(f"Patent Priority / Grant Date: {meta['priority_or_grant_date']}")
                        if meta.get("independent_claims"):
                            header_parts.append(f"Independent Claims Scope:\n{meta['independent_claims']}")
                header_parts.append(chunk["text"])
                chunk_texts.append("\n".join(header_parts))
            
            # 3. Save to checkpoint immediately
            processed_data.append({
                "question": question,
                "answer": json_answer,
                "contexts": chunk_texts,
                "ground_truth": gt
            })
            
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=2)
                
        except Exception as e:
            print(f"Error processing query {idx + 1}: {e}")
            print("Skipping to next query. You can manually inspect the logs.")

    # Unload GENERATOR_MODEL from GPU VRAM before loading JUDGE_MODEL
    if cfg.GENERATOR_MODEL != cfg.JUDGE_MODEL:
        try:
            import requests
            print(f"\nUnloading generator model ({cfg.GENERATOR_MODEL}) from GPU VRAM before Phase 2...")
            requests.post(
                f"{cfg.OLLAMA_HOST}/api/generate",
                json={"model": cfg.GENERATOR_MODEL, "keep_alive": 0},
                timeout=30,
            )
        except Exception as exc:
            print(f"Note: Could not explicitly unload generator model ({exc}).")

    # 3. Build Ragas Dataset from checkpoint data
    data_dict = {
        "question": [item["question"] for item in processed_data],
        "answer": [item["answer"] for item in processed_data],
        "contexts": [item["contexts"] for item in processed_data],
        "ground_truth": [item["ground_truth"] for item in processed_data]
    }
    dataset = Dataset.from_dict(data_dict)
    
    # Fix for Docker Windows asyncio deadlocks in Ragas
    import nest_asyncio
    nest_asyncio.apply()
    
    print("\n--- PHASE 2: RUNNING RAGAS EVALUATION ---")
    print(f"Generator Model: {cfg.GENERATOR_MODEL}")
    print(f"Judge Model: {cfg.JUDGE_MODEL}")
    
    import asyncio
    
    # 1. Wrapper for Local Ollama to bypass async deadlocks and API rate limits
    from langchain_ollama import ChatOllama
    class SyncToAsyncOllama(ChatOllama):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            # Convert the async callback manager to a sync one so we don't get unawaited coroutine warnings
            sync_run_manager = run_manager.get_sync() if run_manager else None
            
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self._generate(messages, stop, sync_run_manager, **kwargs))
            
    # We are moving entirely to your local Llama3 instance!
    # No TPM limits, no TPD limits, 100% free and private.
    # LOCAL MODEL CONFIG:
    # judge_llm = SyncToAsyncOllama(
    #     model=cfg.LLM_MODEL, 
    #     base_url="http://host.docker.internal:11434",
    #     temperature=0.1 # Low temperature for more objective grading
    # )
    
    # UNI GPU MODEL CONFIG:
    # Setting num_ctx=16384 prevents llama3.1:70b from allocating its default 131072-token KV cache (~46 GB),
    # which otherwise pushes total model memory to 86.6 GB and spills 13.6 GB onto slow CPU RAM.
    judge_llm = SyncToAsyncOllama(
        model=cfg.JUDGE_MODEL, 
        base_url=cfg.OLLAMA_HOST,  # Inherits tunnel path from config.py
        temperature=0.1, # Low temperature for more objective grading
        num_ctx=16384,
        client_kwargs={"timeout": 300.0},
    )
    
    # 2. Wrapper for Ollama Embeddings to bypass async deadlocks
    from langchain_ollama import OllamaEmbeddings
    class SyncToAsyncOllamaEmbeddings(OllamaEmbeddings):
        async def aembed_documents(self, texts):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.embed_documents, texts)
        async def aembed_query(self, text):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.embed_query, text)
            
    judge_embeddings = SyncToAsyncOllamaEmbeddings(
        model=cfg.EMBED_MODEL,
        base_url=cfg.OLLAMA_HOST
    )
    
    # Explicitly configure Ragas to ask for only 1 response to permanently bypass the 'n' crash
    answer_relevancy.strictness = 1
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    
    from ragas.run_config import RunConfig
    # Increase job timeout to 1200s (since context_precision makes 15 sequential LLM calls per row)
    # and set max_retries=1 so timed-out threads do not queue duplicate retries on Ollama.
    run_config = RunConfig(timeout=1200, max_retries=1, max_workers=1)
    
    # Run the evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
        raise_exceptions=False,
        callbacks=callbacks
    )
    
    print("\n--- EVALUATION COMPLETE ---")
    print("Final RAGAS Scores:")
    print(result)
    
    # Save the detailed results to a CSV for your thesis
    output_path = cfg.DATA_DIR / "evaluation" / "ragas_results.csv"
    result_df = result.to_pandas()
    result_df.to_csv(output_path, index=False)
    print(f"\nDetailed evaluation results saved to {output_path}")

    if "llama" in cfg.GENERATOR_MODEL.lower() and "qwen" in cfg.JUDGE_MODEL.lower():
        import shutil
        exp_ckpt = cfg.DATA_DIR / "evaluation" / "checkpoint_Exp8_topk15_llama_gen_qwen_judge.json"
        exp_csv = cfg.DATA_DIR / "evaluation" / "ragas_results_Exp8_topk15_llama_gen_qwen_judge.csv"
        if os.path.exists(checkpoint_path):
            shutil.copyfile(checkpoint_path, exp_ckpt)
        result_df.to_csv(exp_csv, index=False)
        print(f"Named experiment copies saved to:\n  - {exp_ckpt}\n  - {exp_csv}")

if __name__ == "__main__":
    # Point this to your saved CSV
    ground_truth_file = os.path.join("data", "evaluation", "ground_truth.csv")
    run_evaluation(ground_truth_file)

