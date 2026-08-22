"""
GenerationPipeline — LLM client with structured output enforcement.

Uses ChatOllama with .with_structured_output() bound to FinalOutputSchema
so every response is a validated Pydantic model, never raw text.
Temperature is fixed at 0.0 for deterministic, hallucination-resistant extraction.
"""

from typing import List, Dict, Any

from langchain_ollama import ChatOllama

import src.config as cfg
from src.schemas.response import FinalOutputSchema
from src.generation.prompts import build_extraction_prompt
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GenerationPipeline:
    """
    Wraps a local Ollama LLM and enforces Pydantic structured output.

    Args:
        model_name: Ollama model tag to use. Defaults to cfg.LLM_MODEL ("llama3").
    """

    def __init__(self, model_name: str = cfg.LLM_MODEL) -> None:
        logger.info(
            f"Initializing GenerationPipeline | model='{model_name}' | host='{cfg.OLLAMA_HOST}'"
        )
        base_llm = ChatOllama(
            model=model_name,
            base_url=cfg.OLLAMA_HOST,
            temperature=0.0,  # Deterministic extraction — no creativity needed
        )
        # Bind the Pydantic schema so every call returns a validated model instance
        self.structured_llm = base_llm.with_structured_output(FinalOutputSchema)

    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> FinalOutputSchema:
        """
        Builds the grounded prompt and invokes the structured LLM.

        Args:
            query:          User's research question.
            context_chunks: Retrieved chunks from the Retriever.

        Returns:
            A validated FinalOutputSchema Pydantic instance.
        """
        prompt = build_extraction_prompt(query=query, context_chunks=context_chunks)
        logger.info("Invoking LLM for structured extraction...")
        logger.debug(f"Prompt (first 300 chars):\n{prompt[:300]}...")

        result: FinalOutputSchema = self.structured_llm.invoke(prompt)

        logger.info("LLM extraction complete.")
        return result