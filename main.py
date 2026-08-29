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
        domains=[
            "Simulation Platforms", 
            "Perception & World models", 
            "Sensors & Environment", 
            "Validation & Testing"
        ],
        max_papers=5,
    )

    # 2. Execute Patent Ingestion
    print("\n--- Phase 1b: Patent Ingestion ---")
    pipeline.run_ingestion_patents(
        domains=[
            "Simulation Platforms", 
            "Perception & World models", 
            "Sensors & Environment", 
            "Validation & Testing"
        ],
        max_patents=5,
    )

    # 3. Interactive Query Loop
    print("\n--- Phase 2: RAG Generation (Interactive) ---")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_query = input("\n🤔 Enter your research question: ").strip()
            if user_query.lower() in ['exit', 'quit']:
                print("Exiting pipeline. Goodbye!")
                break
            if not user_query:
                continue
                
            pipeline.run_query(query=user_query)
            
        except KeyboardInterrupt:
            print("\nExiting pipeline. Goodbye!")
            break