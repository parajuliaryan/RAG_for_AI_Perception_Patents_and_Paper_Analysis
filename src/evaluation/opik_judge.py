"""
OllamaOpikJudgeModel — Custom Opik Judge Model wrapper for local/university Ollama LLMs.

Bridges LangChain's ChatOllama with Opik's OpikBaseModel interface so built-in
Opik LLM-as-a-Judge metrics (AnswerRelevance, GEval) run deterministically on
local GPU infrastructure (e.g., llama3.1:70b on A100) without OpenAI API keys
or LiteLLM logprob parsing failures.
"""

import asyncio
from typing import Any, Dict, List, Optional, Type
from types import SimpleNamespace

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator
from opik.evaluation.models import OpikBaseModel

import src.config as cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)


class _AnswerRelevanceFallbackSchema(BaseModel):
    """Fallback Pydantic schema guaranteeing valid JSON for Opik AnswerRelevance."""
    answer_relevance_score: float = Field(
        ...,
        description="Score between 0.0 and 1.0 indicating how well the output answers the question.",
    )
    reason: str = Field(
        ...,
        description="Concise explanation justifying the assigned answer relevance score.",
    )

    @field_validator("answer_relevance_score")
    @classmethod
    def _normalize_score(cls, v: float) -> float:
        val = float(v)
        if val > 1.0 and val <= 10.0:
            val = val / 10.0
        elif val > 10.0:
            val = val / 100.0
        return max(0.0, min(1.0, val))


class _GEvalFallbackSchema(BaseModel):
    """Fallback Pydantic schema guaranteeing valid JSON for Opik GEval scoring step."""
    score: int = Field(
        ...,
        description="Integer score between 0 and 10 based on the evaluation criteria and chain-of-thought steps.",
    )
    reason: str = Field(
        ...,
        description="Detailed rationale explaining the score based on the evaluation steps.",
    )

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_int_score(cls, v: Any) -> int:
        val = int(round(float(v)))
        return max(0, min(10, val))


class OllamaOpikJudgeModel(OpikBaseModel):
    """
    OpikBaseModel implementation backed by ChatOllama.

    Configures num_ctx=16384 so 70B/72B judge models fit 100% inside 80 GB A100
    VRAM without spilling KV cache to CPU RAM, and enforces structured JSON output
    whenever Opik metrics request a JSON score payload.
    """

    def __init__(
        self,
        model_name: str = cfg.JUDGE_MODEL,
        base_url: str = cfg.OLLAMA_HOST,
        temperature: float = 0.0,
        num_ctx: int = 16384,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(model_name=model_name)
        logger.info(
            f"Initializing OllamaOpikJudgeModel | model='{model_name}' | host='{base_url}' | num_ctx={num_ctx}"
        )
        self._llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            num_ctx=num_ctx,
            client_kwargs={"timeout": timeout},
        )
        self._json_llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            num_ctx=num_ctx,
            format="json",
            client_kwargs={"timeout": timeout},
        )

    def _detect_schema(
        self,
        prompt_text: str,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Optional[Type[BaseModel]]:
        """
        Determines whether a Pydantic schema should be bound to the LLM call.
        - If Opik passes `response_format` directly, use it.
        - If `response_format` is None (older Opik versions), inspect the prompt to
          distinguish JSON scoring prompts from plain-text Chain-of-Thought generation.
        """
        if (
            response_format is not None
            and isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
        ):
            return response_format

        if "answer_relevance_score" in prompt_text:
            return _AnswerRelevanceFallbackSchema

        # GEval's final scoring prompt asks for a JSON object with "score" and "reason",
        # whereas GEval's CoT generation prompt asks for plain-text EVALUATION STEPS.
        prompt_lower = prompt_text.lower()
        if '"score"' in prompt_lower and '"reason"' in prompt_lower and "json" in prompt_lower:
            return _GEvalFallbackSchema

        return None

    def generate_string(
        self,
        input: str,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generates a string response from the Ollama judge model.
        Uses structured output enforcement when a JSON score schema is expected.
        """
        schema = self._detect_schema(input, response_format)
        if schema is not None:
            try:
                structured_llm = self._llm.with_structured_output(schema)
                result: BaseModel = structured_llm.invoke(input)
                return result.model_dump_json()
            except Exception as exc:
                logger.warning(
                    f"Structured output binding failed ({exc}); falling back to JSON mode."
                )
                response = self._json_llm.invoke(input)
                return str(response.content)

        response = self._llm.invoke(input)
        return str(response.content)

    def generate_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generates a chat completion turn (`{"role": "assistant", "content": ...}`)
        from a list of role-tagged message dictionaries.
        Used by Opik's built-in AnswerRelevance and GEval metrics.
        """
        combined_text = "\n\n".join(str(msg.get("content", "")) for msg in messages)
        schema = self._detect_schema(combined_text, response_format)

        if schema is not None:
            try:
                structured_llm = self._llm.with_structured_output(schema)
                result: BaseModel = structured_llm.invoke(messages)
                return {"role": "assistant", "content": result.model_dump_json()}
            except Exception as exc:
                logger.warning(
                    f"Structured chat completion failed ({exc}); falling back to JSON mode."
                )
                response = self._json_llm.invoke(messages)
                return {"role": "assistant", "content": str(response.content)}

        response = self._llm.invoke(messages)
        return {"role": "assistant", "content": str(response.content)}

    def generate_provider_response(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """
        Generates a provider-compatible response object for message-list calls.
        """
        response_format = kwargs.get("response_format")
        chat_turn = self.generate_chat_completion(
            messages=messages,
            response_format=response_format,
        )
        content = chat_turn["content"]

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, role="assistant"),
                    logprobs=None,
                )
            ]
        )

    async def agenerate_string(
        self,
        input: str,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> str:
        """Async wrapper around generate_string using thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_string(input=input, response_format=response_format, **kwargs),
        )

    async def agenerate_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async wrapper around generate_chat_completion using thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_chat_completion(
                messages=messages,
                response_format=response_format,
                **kwargs,
            ),
        )

    async def agenerate_provider_response(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Async wrapper around generate_provider_response using thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_provider_response(messages=messages, **kwargs),
        )
