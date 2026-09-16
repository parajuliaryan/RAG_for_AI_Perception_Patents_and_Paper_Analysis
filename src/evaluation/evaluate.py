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

# Groq and local embeddings for fully independent Ragas
from langchain_groq import ChatGroq

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
            result = rag_service.analyze_query(query=question, top_k=5)
            
            # 2. Extract JSON answer and Contexts
            json_answer = result["response"].model_dump_json(indent=2)
            chunk_texts = [chunk["text"] for chunk in result["context"]]
            
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
    print(f"Using Local {cfg.LLM_MODEL} as the objective LLM-as-a-judge...")
    
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
    judge_llm = SyncToAsyncOllama(
        model=cfg.LLM_MODEL, 
        base_url="http://host.docker.internal:11434",
        temperature=0.1 # Low temperature for more objective grading
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
        model="nomic-embed-text",
        base_url="http://host.docker.internal:11434"
    )
    
    # Explicitly configure Ragas to ask for only 1 response to permanently bypass the 'n' crash
    answer_relevancy.strictness = 1
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    
    from ragas.run_config import RunConfig
    # We increase timeout to 600s but drop max_retries to 2.
    # If Groq hallucinates bad JSON, Ragas will immediately assign NaN instead of infinitely retrying and hanging.
    run_config = RunConfig(timeout=600, max_retries=2, max_workers=1)
    
    # Run the evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
        raise_exceptions=False
    )
    
    print("\n--- EVALUATION COMPLETE ---")
    print("Final RAGAS Scores:")
    print(result)
    
    # Save the detailed results to a CSV for your thesis
    output_path = cfg.DATA_DIR / "evaluation" / "ragas_results.csv"
    result_df = result.to_pandas()
    result_df.to_csv(output_path, index=False)
    print(f"\nDetailed evaluation results saved to {output_path}")

if __name__ == "__main__":
    # Point this to your saved CSV
    ground_truth_file = os.path.join("data", "evaluation", "ground_truth.csv")
    run_evaluation(ground_truth_file)

