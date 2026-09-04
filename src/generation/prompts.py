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
        "You are an expert AI perception and automotive systems researcher.\n"
        "Answer the user query strictly using the provided context documents below.\n"
        "Do not extrapolate or assume details that are not present in the context.\n"
        "If the context does not contain enough information, state that explicitly.\n\n"
        "CRITICAL INSTRUCTIONS FOR JSON GENERATION:\n"
        "1. For `source_document_id`, `document_title`, and `document_abstract`, you MUST extract them exactly as they appear in the metadata headers of the context blocks below.\n"
        "2. If a specific data point (like hardware, simulators, weather, or companies) is not explicitly named in the text, you MUST output `['Not specified in document']` for that field. Do not leave lists empty [], and do not substitute author names for company names.\n\n"
        "Your primary goal is Tech Scouting. Carefully extract the following:\n"
        "- Affiliated Companies or Institutions\n"
        "- Simulators (e.g., CARLA, AURELION, Carmaker)\n"
        "- Sensor types (e.g., LiDAR, Radar, Camera)\n"
        "- Weather and environmental parameters tested\n"
        "- Target Use Cases (e.g., Sensor Validation, Virtual Homologation, Neural Rendering)\n"
        "- Core technologies and buzzwords\n"
        "- ECU or Hardware tested\n"
        "- Tested scenarios and Evaluated KPIs (e.g., mAP, latency)\n"
        "- Operational constraints or assumptions made\n"
        "\n"
        f"User Query:\n{query}\n"
        "\n"
        f"Context:\n{context_str}\n"
    )

    return prompt

