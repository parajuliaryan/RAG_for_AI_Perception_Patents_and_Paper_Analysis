"""
Experiment 7 — Opik Evaluation Pipeline (AnswerRelevance + GEval).

Evaluates the AI Perception RAG pipeline using Comet Opik's built-in LLM-as-a-Judge
metrics (`AnswerRelevance` and `GEval`) to overcome the structural JSON penalty
of RAGAS's reverse-question `answer_relevancy` metric.

Workflow:
  - Phase 1: Loads pre-generated RAG responses from `data/evaluation/checkpoint.json`
             (or generates any missing queries via `RAGService`).
  - Phase 2: Scores each query using Opik `AnswerRelevance` and Opik `GEval`
             backed by `OllamaOpikJudgeModel` (`llama3.1:70b` with `num_ctx=8192`),
             saving incremental progress to `data/evaluation/checkpoint_Exp7_opik.json`.
  - Phase 3: Exports `data/evaluation/opik_results_Exp7.csv` and merges with
             Experiment 6 RAGAS metrics into `data/evaluation/ragas_opik_combined_Exp7.csv`.

Usage (inside Docker container or local environment):
    python -m src.evaluation.opik_evaluate
"""

import json
import os
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

import src.config as cfg
from src.evaluation.opik_judge import OllamaOpikJudgeModel
from src.evaluation.opik_metrics import OpikMetricsSuite
from src.services.rag_service import RAGService
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def _format_chunk_contexts(context_chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Formats retrieved chunks with deduplicated document-level metadata
    (once per unique parent document), matching `src/evaluation/evaluate.py`.
    """
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


def run_opik_evaluation(csv_path: str) -> None:
    print(f"Loading ground truth dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}.")
        return

    if not {"question", "ground_truth"}.issubset(df.columns):
        print("Error: CSV must contain 'question' and 'ground_truth' columns.")
        return

    eval_dir = cfg.DATA_DIR / "evaluation"
    exp_tag = getattr(cfg, "EXPERIMENT_TAG", "Exp9_foolproof_chunking_and_extraction")
    gen_checkpoint_path = eval_dir / f"checkpoint_{exp_tag}.json"
    opik_checkpoint_path = eval_dir / f"checkpoint_{exp_tag}_opik.json"
    opik_csv_path = eval_dir / f"opik_results_{exp_tag}.csv"
    combined_csv_path = eval_dir / f"ragas_opik_combined_{exp_tag}.csv"

    # ------------------------------------------------------------------
    # PHASE 1: Load or Generate RAG Answers
    # ------------------------------------------------------------------
    processed_data: List[Dict[str, Any]] = []
    if gen_checkpoint_path.exists():
        with open(gen_checkpoint_path, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
        print(
            f"Loaded {len(processed_data)} pre-generated RAG responses from {gen_checkpoint_path}."
        )

    processed_questions = {item["question"] for item in processed_data}

    missing_rows = [
        (idx, row)
        for idx, row in df.iterrows()
        if str(row["question"]).strip() not in ("", "nan")
        and str(row["question"]) not in processed_questions
    ]

    if missing_rows:
        print(f"\n--- PHASE 1: GENERATING {len(missing_rows)} MISSING ANSWERS ---")
        rag_service = RAGService()
        for idx, row in missing_rows:
            question = str(row["question"])
            gt = str(row["ground_truth"])
            print(f"\nProcessing Query {idx + 1}/{len(df)}: {question[:60]}...")
            try:
                result = rag_service.analyze_query(query=question, top_k=cfg.TOP_K_DEFAULT)
                json_answer = result["response"].model_dump_json(indent=2)
                chunk_texts = _format_chunk_contexts(result["context"])

                processed_data.append(
                    {
                        "question": question,
                        "answer": json_answer,
                        "contexts": chunk_texts,
                        "ground_truth": gt,
                    }
                )
                with open(gen_checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(processed_data, f, indent=2)
            except Exception as exc:
                print(f"Error generating answer for query {idx + 1}: {exc}")

        if cfg.GENERATOR_MODEL != cfg.JUDGE_MODEL:
            try:
                import requests
                print(
                    f"\nUnloading generator model ({cfg.GENERATOR_MODEL}) from GPU VRAM before Phase 2..."
                )
                requests.post(
                    f"{cfg.OLLAMA_HOST}/api/generate",
                    json={"model": cfg.GENERATOR_MODEL, "keep_alive": 0},
                    timeout=30,
                )
            except Exception as exc:
                print(f"Note: Could not explicitly unload generator model ({exc}).")
    else:
        print("\n--- PHASE 1: ALL ANSWERS ALREADY GENERATED (SKIPPING GENERATION) ---")

    # ------------------------------------------------------------------
    # PHASE 2: Run Opik Evaluation (AnswerRelevance + GEval)
    # ------------------------------------------------------------------
    print("\n--- PHASE 2: RUNNING EXPERIMENT 7 OPIK EVALUATION ---")
    print(f"Generator Model : {cfg.GENERATOR_MODEL}")
    print(f"Judge Model     : {cfg.JUDGE_MODEL}")
    print("Metrics         : Opik AnswerRelevance + Opik GEval")

    opik_results: List[Dict[str, Any]] = []
    if opik_checkpoint_path.exists():
        with open(opik_checkpoint_path, "r", encoding="utf-8") as f:
            opik_results = json.load(f)
        print(
            f"Loaded {len(opik_results)} previously scored Opik queries from {opik_checkpoint_path}."
        )

    scored_questions = {
        item["question"]
        for item in opik_results
        if item.get("opik_answer_relevance") is not None
        and item.get("opik_geval_relevancy") is not None
    }

    # Filter out already scored entries if re-evaluating failed ones
    opik_results = [
        item for item in opik_results if item["question"] in scored_questions
    ]

    judge_model = OllamaOpikJudgeModel(
        model_name=cfg.JUDGE_MODEL,
        base_url=cfg.OLLAMA_HOST,
        temperature=0.0,
        num_ctx=8192,
        timeout=300.0,
    )
    metrics_suite = OpikMetricsSuite(judge_model=judge_model)

    total = len(processed_data)
    for idx, item in enumerate(processed_data):
        question = item["question"]
        if question in scored_questions:
            print(f"Skipping [{idx + 1}/{total}] (Already scored): {question[:60]}...")
            continue

        print(f"\nScoring [{idx + 1}/{total}]: {question[:70]}...")
        scores = metrics_suite.evaluate_sample(
            question=question,
            answer=item["answer"],
            contexts=item["contexts"],
            ground_truth=item["ground_truth"],
        )

        print(
            f"  -> Opik AnswerRelevance : {scores['opik_answer_relevance']} | "
            f"Opik GEval : {scores['opik_geval_relevancy']}"
        )

        record = {
            "question": question,
            "ground_truth": item["ground_truth"],
            "answer": item["answer"],
            "opik_answer_relevance": scores["opik_answer_relevance"],
            "opik_answer_relevance_reason": scores["opik_answer_relevance_reason"],
            "opik_geval_relevancy": scores["opik_geval_relevancy"],
            "opik_geval_reason": scores["opik_geval_reason"],
        }
        opik_results.append(record)

        with open(opik_checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(opik_results, f, indent=2)

    # ------------------------------------------------------------------
    # PHASE 3: Compute Means, Export CSVs & Merge with Exp 6 RAGAS
    # ------------------------------------------------------------------
    opik_df = pd.DataFrame(opik_results)
    opik_df.to_csv(opik_csv_path, index=False)

    mean_ar = opik_df["opik_answer_relevance"].dropna().mean()
    mean_geval = opik_df["opik_geval_relevancy"].dropna().mean()

    print("\n============================================================")
    print("       EXPERIMENT 7: OPIK EVALUATION COMPLETE")
    print("============================================================")
    print(f"Opik AnswerRelevance (mean) : {mean_ar:.4f}")
    print(f"Opik GEval Relevancy (mean) : {mean_geval:.4f}")
    print(f"Detailed Opik results saved : {opik_csv_path}")

    # Merge with corresponding RAGAS scores if available
    ragas_exp_path = eval_dir / f"ragas_results_{exp_tag}.csv"
    if not ragas_exp_path.exists():
        ragas_exp_path = eval_dir / "ragas_results.csv"

    if ragas_exp_path.exists():
        try:
            ragas_df = pd.read_csv(ragas_exp_path)
            ragas_cols = [
                c
                for c in [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                ]
                if c in ragas_df.columns
            ]
            if len(ragas_df) == len(opik_df) and ragas_cols:
                combined_df = opik_df.copy()
                for col in ragas_cols:
                    target_col = "ragas_answer_relevancy" if col == "answer_relevancy" else col
                    combined_df[target_col] = ragas_df[col].values
                combined_df.to_csv(combined_csv_path, index=False)

                print("\n--- COMBINED THESIS SCORECARD (RAGAS + OPIK) ---")
                print(f"Faithfulness (RAGAS)            : {ragas_df['faithfulness'].mean():.4f}")
                print(f"Context Precision (RAGAS)       : {ragas_df['context_precision'].mean():.4f}")
                print(f"Context Recall (RAGAS)          : {ragas_df['context_recall'].mean():.4f}")
                print(f"Answer Relevancy (RAGAS - Old)  : {ragas_df['answer_relevancy'].mean():.4f}")
                print(f"Answer Relevance (Opik Built-in): {mean_ar:.4f}")
                print(f"Answer Relevancy (Opik GEval)   : {mean_geval:.4f}")
                print(f"Combined CSV saved to           : {combined_csv_path}")
        except Exception as exc:
            logger.warning(f"Could not merge with RAGAS CSV ({exc}).")


if __name__ == "__main__":
    ground_truth_file = os.path.join("data", "evaluation", "ground_truth.csv")
    run_opik_evaluation(ground_truth_file)

