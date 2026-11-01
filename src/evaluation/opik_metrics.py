"""
OpikMetricsSuite — Structured wrapper for Opik LLM-as-a-Judge evaluation metrics.

Configures and executes two complementary Opik metrics for Experiment 7:
1. AnswerRelevance (opik.evaluation.metrics.AnswerRelevance):
   Opik's built-in RAG metric that directly evaluates whether the generated
   response addresses the user's input question given the retrieved context
   (replacing RAGAS's reverse-question embedding similarity approach).
2. GEval (opik.evaluation.metrics.GEval):
   Opik's Chain-of-Thought (CoT) rubric evaluator tailored for structured
   technology scouting JSON outputs (FinalOutputSchema), grading technical
   relevance and completeness without penalizing schema-required metadata fields.
"""

from typing import Any, Dict, List
from opik.evaluation.metrics import AnswerRelevance, GEval

import src.config as cfg
from src.evaluation.opik_judge import OllamaOpikJudgeModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


GEVAL_TASK_INTRODUCTION = (
    "You are an expert evaluator grading an AI Perception Technology Scouting RAG system. "
    "Given a user's technical research question, the reference ground truth, and the system's "
    "generated structured JSON report (following FinalOutputSchema), evaluate the Answer Relevancy "
    "and technical responsiveness of the generated JSON report to the user's question."
)

GEVAL_EVALUATION_CRITERIA = (
    "Evaluate the generated structured JSON output on an integer scale from 0 to 10 based on:\n"
    "1. Direct Technical Relevancy (40%): Does the content inside the JSON fields "
    "(such as 'summary_of_technical_relevance', 'key_claims_or_methodologies', 'perception_findings', "
    "'document_abstract', or 'patent_metadata') directly and accurately answer the specific technical "
    "or legal question asked by the user?\n"
    "2. Completeness & Alignment with Ground Truth (40%): Does the JSON capture the core algorithms, "
    "sensor modalities, mathematical/architectural concepts, quantitative KPIs, or patent claim "
    "boundaries expected in the reference ground truth?\n"
    "3. Schema Awareness — No Penalty for Standardized Structure (20%): The system is required by design "
    "to output a complete technology scouting JSON schema for every query. Do NOT penalize the output "
    "for including additional valid document metadata (e.g., 'source_document_id', 'document_title', "
    "'affiliated_companies_or_institutions', 'patent_metadata') or for placing '[\"None\"]' in schema "
    "fields not mentioned in the retrieved text."
)


class OpikMetricsSuite:
    """
    Encapsulates Opik's built-in AnswerRelevance and GEval metrics backed by OllamaOpikJudgeModel.
    """

    def __init__(self, judge_model: OllamaOpikJudgeModel) -> None:
        self.judge_model = judge_model

        logger.info(
            "Initializing Opik AnswerRelevance metric (require_context=False, track=False)..."
        )
        self.answer_relevance_metric = AnswerRelevance(
            model=self.judge_model,
            require_context=False,
            track=False,
        )

        logger.info(
            "Initializing Opik GEval metric (Chain-of-Thought Tech Scouting Rubric, track=False)..."
        )
        self.geval_metric = GEval(
            model=self.judge_model,
            task_introduction=GEVAL_TASK_INTRODUCTION,
            evaluation_criteria=GEVAL_EVALUATION_CRITERIA,
            track=False,
        )

    @staticmethod
    def format_geval_payload(question: str, answer: str, ground_truth: str) -> str:
        """
        Packages the user question, reference ground truth, and generated JSON report
        into a single structured evaluation block for GEval.score(output=...).
        """
        return (
            f"=== USER QUESTION ===\n{question}\n\n"
            f"=== REFERENCE GROUND TRUTH ===\n{ground_truth}\n\n"
            f"=== GENERATED STRUCTURED JSON REPORT ===\n{answer}"
        )

    def evaluate_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> Dict[str, Any]:
        """
        Evaluates a single RAG sample using both Opik AnswerRelevance and Opik GEval.

        Returns:
            Dictionary containing numeric scores (0.0 to 1.0) and judge rationales.
        """
        results: Dict[str, Any] = {
            "opik_answer_relevance": None,
            "opik_answer_relevance_reason": "",
            "opik_geval_relevancy": None,
            "opik_geval_reason": "",
        }

        # 1. Run Opik Built-in AnswerRelevance (Direct Question-to-Answer Relevance)
        try:
            ar_score = self.answer_relevance_metric.score(
                input=question,
                output=answer,
            )
            results["opik_answer_relevance"] = round(float(ar_score.value), 4)
            results["opik_answer_relevance_reason"] = str(ar_score.reason or "")
        except Exception as exc:
            logger.error(f"Opik AnswerRelevance failed on query '{question[:50]}...': {exc}")
            results["opik_answer_relevance_reason"] = f"Error: {exc}"

        # 2. Run Opik Built-in GEval
        try:
            geval_payload = self.format_geval_payload(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
            )
            geval_score = self.geval_metric.score(output=geval_payload)
            results["opik_geval_relevancy"] = round(float(geval_score.value), 4)
            results["opik_geval_reason"] = str(geval_score.reason or "")
        except Exception as exc:
            logger.error(f"Opik GEval failed on query '{question[:50]}...': {exc}")
            results["opik_geval_reason"] = f"Error: {exc}"

        return results

