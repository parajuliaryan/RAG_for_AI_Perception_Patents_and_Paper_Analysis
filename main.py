from src.ingestion.arxiv_scraper import ArxivScraper

if __name__ == "__main__":
    scraper = ArxivScraper()
    
    # Example: Fetch recent Computer Vision papers strictly covering 'Sensors & Environment' 
    # and 'Perception & World models' from 2024 to 2026.
    advanced_query = scraper.build_query(
        base_category="cs.CV",
        selected_domains=[
            "Sensors & Environment", 
            "Perception & World models"
        ],
        start_year="2024",
        end_year="2026"
    )
    
    print(f"Executing Query API String: \n{advanced_query}\n")
    
    papers = scraper.fetch(query=advanced_query, max_results=3)
    
    for i, paper in enumerate(papers, 1):
        print(f"{i}. [{paper.published_date}] {paper.title}")
        print(f"   URL: {paper.pdf_url}\n")