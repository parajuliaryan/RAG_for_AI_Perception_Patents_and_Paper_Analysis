"""
Streamlit frontend for the AI Perception RAG Pipeline.

Responsibility: Query interface and structured JSON output display only.
Ingestion is handled exclusively via the CLI (python main.py).
"""

import json
import streamlit as st
from src.services.rag_service import RAGService


@st.cache_resource
def get_rag_service() -> RAGService:
    """Load all models and DB connections once per process."""
    return RAGService()


class RAGDashboard:
    def __init__(self):
        st.set_page_config(
            page_title="AI Perception RAG",
            page_icon="🔬",
            layout="centered",
        )
        self.service = get_rag_service()

    def _db_is_empty(self) -> bool:
        return self.service.get_db_count() == 0

    def render(self):
        # ── Header ──────────────────────────────────────────────────────────
        st.title("🔬 AI Perception RAG — Structured Extraction")
        st.caption(
            "Enter a research question. The system retrieves relevant document chunks "
            "and extracts a machine-readable JSON analysis using a local LLM."
        )
        st.divider()

        # ── Empty DB warning ─────────────────────────────────────────────────
        if self._db_is_empty():
            st.warning(
                "⚠️ The vector store is empty. "
                "Run ingestion via the CLI first:\n\n"
                "```bash\n"
                "docker-compose run app python main.py\n"
                "```",
                icon="⚠️",
            )
            return  # Nothing else to show until data is available

        db_count = self.service.get_db_count()
        st.caption(f"🗄️ Vector store: **{db_count}** chunk(s) indexed")

        # ── Query input ──────────────────────────────────────────────────────
        query = st.text_area(
            label="Research Question",
            placeholder="e.g. What are the capabilities and limitations of LiDAR simulation?",
            height=100,
        )

        col_left, col_right = st.columns([3, 1])
        with col_right:
            top_k = st.number_input(
                "Chunks to retrieve (top-k)",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="How many document chunks are passed to the LLM as context.",
            )

        run = col_left.button(
            "⚡ Extract Structured Analysis",
            type="primary",
            use_container_width=True,
            disabled=not query.strip(),
        )

        # ── Results ──────────────────────────────────────────────────────────
        if run and query.strip():
            with st.spinner("Retrieving context and running LLM extraction..."):
                result = self.service.analyze_query(query=query.strip(), top_k=int(top_k))

            response = result["response"]
            context  = result["context"]

            st.divider()

            # Primary output: structured JSON
            st.subheader("📄 Structured JSON Output")
            st.code(
                response.model_dump_json(indent=2),
                language="json",
            )

            # Download button for the JSON
            st.download_button(
                label="⬇️ Download JSON",
                data=response.model_dump_json(indent=2),
                file_name="rag_extraction.json",
                mime="application/json",
            )

            # Secondary: retrieved source chunks (collapsed by default)
            st.divider()
            with st.expander(f"📚 Retrieved Source Chunks (top-{top_k})", expanded=False):
                for idx, chunk in enumerate(context, start=1):
                    dist  = chunk.get("similarity_distance", 0.0)
                    title = chunk.get("metadata", {}).get("title", "Unknown")
                    st.markdown(f"**Chunk {idx}** — *{title}* — distance: `{dist:.4f}`")
                    st.code(chunk.get("text", ""), language="text")
                    if idx < len(context):
                        st.divider()

    def run(self):
        self.render()


if __name__ == "__main__":
    app = RAGDashboard()
    app.run()