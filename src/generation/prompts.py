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
    seen_docs = set()

    for i, chunk in enumerate(context_chunks, start=1):
        chunk_id = chunk.get("id", "N/A")
        parent_id = chunk.get("parent_id") or (
            chunk_id.rsplit("_chunk_", 1)[0] if "_chunk_" in chunk_id else chunk_id
        )
        meta = chunk.get("metadata", {}) or {}
        pub_no = meta.get("publication_number") or parent_id
        title = meta.get("title", "N/A")
        abstract = meta.get("abstract", "N/A")
        doc_source = meta.get("source", "")
        affil = meta.get("affiliated_institutions", "")
        pat_class = meta.get("patent_classifications", "")
        pat_date = meta.get("priority_or_grant_date", "")
        indep_claims = meta.get("independent_claims", "")
        content = chunk.get("text", "")

        lines = [
            f"--- [Context Chunk {i}] ---",
            f"Source ID: {pub_no}",
            f"Document Title: {title}",
        ]

        # Inject document-level metadata only once per unique parent document to save prompt tokens
        if parent_id not in seen_docs:
            seen_docs.add(parent_id)
            if affil:
                lines.append(f"Affiliated Institutions / Assignee: {affil}")
            if abstract and abstract != "N/A":
                lines.append(f"Document Abstract: {abstract}")
            if doc_source in ("patent", "epo_ops"):
                if pat_class:
                    lines.append(f"Patent Classifications (IPC/CPC): {pat_class}")
                if pat_date:
                    lines.append(f"Patent Priority / Grant Date: {pat_date}")
                if indep_claims:
                    lines.append(f"Independent Claims Scope:\n{indep_claims}")

        lines.append(f"Content:\n{content}")
        document_blocks.append("\n".join(lines))

    context_str = "\n\n".join(document_blocks)

    prompt = (
        "You are an expert AI perception tech scout. Extract exact technical entities from the provided context.\n\n"
        "CRITICAL RULES:\n"
        "1. DO NOT invent filler text or hallucinate entities not supported by the context.\n"
        "2. Focus on the primary target document that directly answers the User Query: use its exact `Source ID`, `Document Title`, `Document Abstract`, and extract ALL institutions/companies listed in its `Affiliated Institutions / Assignee` header into `affiliated_companies_or_institutions`.\n"
        "3. Thoroughly scan ALL retrieved chunks and the abstract of the primary document to populate every `PerceptionFinding` field:\n"
        "   - `simulation_tools_or_solvers`: extract all simulators (e.g., CARLA, SUMO, dSPACE AURELION), ray-tracing/modeling libraries (e.g., SMDL), validation tools (e.g., Avelon, Vector CANoe), EM/optical solvers (e.g., ANSYS HFSS, SBR+), and radar target simulators/echo generators (e.g., dSPACE DARTS 9040-G, Rohde & Schwarz R&S AREG800A).\n"
        "   - `hardware_target_or_deployment_platform`: extract all physical sensor boards under test (e.g., Infineon 77 GHz P2S, NXP RDK-S32R274, Ouster OS1 LiDAR), reference measurement devices (e.g., Bosch GLM 120 C laser rangefinder), HIL interfaces (e.g., Vector VN1640A), compute GPUs/SoCs (e.g., NVIDIA A100, H100, RTX 3090/4090, Drive Orin), and vehicle ECUs.\n"
        "   - `weather_and_environmental_parameters`: extract all indoor/outdoor settings (e.g., Indoors, Outdoors), lighting conditions (e.g., Night, Daytime, Glare, Ambient light suppression), weather (e.g., Rain, Fog, Snow), and physical propagation/noise conditions.\n"
        "   - When multiple `PerceptionFinding` entries describe sub-modules or iterations of the same system, propagate the document's shared simulators, solvers, hardware platforms, and test conditions across each relevant finding rather than leaving sub-findings as `[\"None\"]`.\n"
        "4. Only if a list field is completely unmentioned across all chunks of the document, output `[\"None\"]`.\n"
        "5. If the primary document is an academic paper, DO NOT generate a `patent_metadata` object (leave it null/None).\n"
        "6. If the primary document is a patent, include ALL claims listed under `Independent Claims Scope` in `patent_metadata.independent_claims_scope` (preserving each `Claim X:` prefix).\n\n"
        "EXAMPLE OF FALLBACK ONLY WHEN A FIELD IS COMPLETELY UNMENTIONED IN ALL CHUNKS:\n"
        "{\n"
        '  "unmentioned_list_field": ["None"]\n'
        "}\n\n"
        f"User Query:\n{query}\n"
        "\n"
        f"Context:\n{context_str}\n"
    )

    return prompt
