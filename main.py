"""
CLI entry point for the AI Perception RAG Pipeline.

Delegates all work to the Pipeline class, keeping this file to its
sole responsibility: defining what runs and in what order.
"""

from src.pipeline import Pipeline


if __name__ == "__main__":
    pipeline = Pipeline()

    # 1. Ingest papers from arXiv
    pipeline.run_ingestion(
        domains=["Sensors & Environment", "Perception & World models"],
        max_papers=2,
    )

    # 2. Run a research query and print structured JSON result
    pipeline.run_query(
        query="What are the capabilities and limitations of LiDAR simulation and occupancy networks?",
    )