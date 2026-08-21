import streamlit as st
from pathlib import Path

# Import your OOP backend classes
from src.ingestion.arxiv_scraper import ArxivScraper
from src.processing.standardizer import DocumentStandardizer
from src.processing.chunker import RecursiveChunker
from src.embeddings.ollama_embedder import OllamaEmbedder
from src.databases.vectorstore.chroma_store import ChromaStore
from src.generation.llm_client import GenerationPipeline

class RAGDashboard:
    """Object-Oriented Streamlit Interface for the AI Perception RAG Pipeline."""

    def __init__(self):
        # 1. Initialize page config
        st.set_page_config(page_title="AI Perception RAG", page_icon="🚗", layout="wide")
        
        # 2. Load background services (Cached so they don't reload on every UI click)
        self.vector_store = self._init_vector_store()
        self.generator = self._init_generator()

    @staticmethod
    @st.cache_resource
    def _init_vector_store():
        """Initializes the embedder and ChromaDB connection."""
        embedder = OllamaEmbedder(model_name="nomic-embed-text")
        return ChromaStore(embedder=embedder, persist_dir="data/vector_store")

    @staticmethod
    @st.cache_resource
    def _init_generator():
        """Initializes the LLM generation pipeline."""
        return GenerationPipeline(model_name="llama3")

    def _handle_ingestion(self, domains: list, max_papers: int):
        """Encapsulates the entire ingestion and storage pipeline."""
        with st.spinner("Scraping arXiv and building vector embeddings..."):
            # Scrape
            scraper = ArxivScraper()
            query = scraper.build_query(selected_domains=domains, start_year="2024", end_year="2026")
            papers = scraper.fetch(query=query, max_results=max_papers)

            # Persist raw data
            raw_dir = Path("data/raw/arxiv")
            raw_dir.mkdir(parents=True, exist_ok=True)
            for paper in papers:
                file_path = raw_dir / f"{paper.id.replace('/', '_')}.json"
                file_path.write_text(paper.model_dump_json(indent=2), encoding="utf-8")

            # Standardize & Chunk
            std_docs = DocumentStandardizer().process(papers)
            chunks = RecursiveChunker(chunk_size=500, chunk_overlap=50).chunk(std_docs)

            # Store
            self.vector_store.add_documents(chunks)
            st.sidebar.success(f"Ingested {len(papers)} papers into {len(chunks)} chunks!")

    def render_sidebar(self):
        """Renders the left control panel for data ingestion."""
        st.sidebar.header("1. Ingestion Controls")
        
        available_domains = [
            "Simulation Platforms", "Perception & World models",
            "Sensors & Environment", "Validation & Testing"
        ]
        
        selected_domains = st.sidebar.multiselect("Target Domains", options=available_domains, default=["Sensors & Environment"])
        max_papers = st.sidebar.slider("Max Papers to Fetch", 1, 10, 3)

        if st.sidebar.button("📥 Fetch & Ingest Papers", use_container_width=True):
            if not selected_domains:
                st.sidebar.error("Select at least one domain.")
            else:
                self._handle_ingestion(selected_domains, max_papers)

        # Show DB Stats
        db_count = self.vector_store.collection.count()
        st.sidebar.metric(label="Total Chunks in Vector Store", value=db_count)
        return db_count

    def render_main_area(self, db_count: int):
        """Renders the main search and results area."""
        st.title("🚗 Automotive AI Perception - RAG System")
        st.subheader("2. Ask an Engineering Question")

        user_query = st.text_input(
            "Enter your research question:", 
            value="What are the capabilities and limitations of LiDAR simulation?"
        )
        top_k = st.slider("Top Chunks to Retrieve", 1, 6, 3)

        if st.button("🔍 Search & Analyze", type="primary", use_container_width=True):
            if db_count == 0:
                st.error("Vector store is empty! Fetch papers first using the sidebar.")
                return

            # Retrieval Step
            with st.spinner("Retrieving relevant context from ChromaDB..."):
                retrieved_chunks = self.vector_store.search(query=user_query, top_k=top_k)

            # Generation Step
            with st.spinner("Analyzing context with Llama-3..."):
                structured_response = self.generator.generate(query=user_query, context_chunks=retrieved_chunks)

            self._render_results(structured_response, retrieved_chunks)

    def _render_results(self, structured_response, retrieved_chunks):
        """Formats and displays the final RAG output and context."""
        col1, col2 = st.columns([3, 2])

        with col1:
            st.success("### 📊 Structured Analysis Output")
            st.markdown(f"**Primary Paper:** `{structured_response.paper_id}`")
            st.info(f"**Relevance:**\n{structured_response.summary_of_relevance}")

            st.write("#### Extracted Perception Findings:")
            for item in structured_response.perception_findings:
                with st.expander(f"🔬 **Tech:** {item.technology_mentioned}", expanded=True):
                    c_left, c_right = st.columns(2)
                    with c_left:
                        st.markdown("**Capabilities:**")
                        for cap in item.capabilities:
                            st.markdown(f"- ✅ {cap}")
                    with c_right:
                        st.markdown("**Limitations:**")
                        for lim in item.limitations:
                            st.markdown(f"- ⚠️ {lim}")

        with col2:
            st.markdown("### 📚 Retrieved Context")
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                dist = chunk.get("similarity_distance", 0.0)
                with st.expander(f"Chunk {idx} (Distance: {dist:.4f})"):
                    st.markdown(f"```text\n{chunk.get('text')}\n```")

    def run(self):
        """Main execution flow for the Streamlit app."""
        db_count = self.render_sidebar()
        self.render_main_area(db_count)

# --- Entry Point ---
if __name__ == "__main__":
    app = RAGDashboard()
    app.run()