"""
Prompt templates for the RAG generation layer.

All LLM prompts live here — this is the single location to iterate on
prompt engineering without touching business logic in llm_client.py.
"""

from typing import List, Dict, Any


def build_extraction_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Builds the grounded extraction prompt sent to the LLM.

    Formats each retrieved chunk into a numbered document block so the model
    can cite specific sources, then wraps them with a strict, hallucination-
    resistant system instruction.

    Args:
        query:          The user's natural language research question.
        context_chunks: List of retrieved chunk dicts (keys: id, text, metadata).

    Returns:
        A fully formatted prompt string ready for LLM invocation.
    """
    document_blocks = []
    for i, chunk in enumerate(context_chunks, start=1):
        source_id = chunk.get("id", "N/A")
        title = chunk.get("metadata", {}).get("title", "N/A")
        abstract = chunk.get("metadata", {}).get("abstract", "N/A")
        content = chunk.get("text", "")
        document_blocks.append(
            f"--- [Context Chunk {i}] ---\n"
            f"Source ID: {source_id}\n"
            f"Document Title: {title}\n"
            f"Document Abstract: {abstract}\n"
            f"Content:\n{content}"
        )

    context_str = "\n\n".join(document_blocks)

    prompt = (
        "You are an expert AI perception tech scout. Extract exact entities from the context.\n\n"
        "CRITICAL RULES:\n"
        "1. DO NOT invent filler text or write full sentences.\n"
        "2. If a field (like simulators or hardware) is not mentioned, you MUST output EXACTLY `[\"None\"]`.\n\n"
        "EXAMPLE OF CORRECT BEHAVIOR WHEN DATA IS MISSING:\n"
        '{\n'
        '  "simulators_used": ["None"],\n'
        '  "ecu_or_hardware_tested": ["None"]\n'
        '}\n\n'
        f"User Query:\n{query}\n"
        "\n"
        f"Context:\n{context_str}\n"
    )

    return prompt

