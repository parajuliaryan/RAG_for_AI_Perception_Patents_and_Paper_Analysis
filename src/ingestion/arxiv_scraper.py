import arxiv
from typing import List, Optional
from src.ingestion.base_scraper import BaseScraper
from src.schemas.document import DocumentSchema

class ArxivScraper(BaseScraper):
    OPTIMIZED_QUERIES = {
        "Sensor Model Validation": 'all:"autonomous driving" AND (all:"sensor model validation" OR all:"sensor model fidelity" OR all:"physics-based sensor simulation" OR all:"LiDAR simulation validation" OR all:"radar model validation") AND (all:"dSPACE" OR all:"AURELION" OR all:"Applied Intuition" OR all:"Spectral" OR all:"aiSim" OR all:"DYNA4" OR all:"CarMaker" OR all:"CARLA" OR all:"Cognata")',
        
        "Sim2Real": 'all:"autonomous driving" AND (all:"sim-to-real gap" OR all:"reality gap" OR all:"synthetic vs real data" OR all:"domain gap quantification") AND (all:"perception" OR all:"object detection") AND (all:"dSPACE" OR all:"Applied Intuition" OR all:"Foretellix" OR all:"DYNA4" OR all:"CarMaker" OR all:"CARLA")',
        
        "Virtual Homologation": '(all:"Hardware-in-the-Loop" OR all:"HIL" OR all:"vECU" OR all:"virtual homologation" OR all:"scenario-based testing") AND (all:"sensor injection" OR all:"closed-loop" OR all:"safety assurance" OR all:"certification") AND (all:"autonomous" OR all:"ADAS")',
        
        "Neural Rendering": '(all:"neural rendering" OR all:"NeRF" OR all:"neural radiance field" OR all:"3D Gaussian Splatting" OR all:"3DGS") AND (all:"autonomous driving" OR all:"ADAS") AND (all:"sensor simulation" OR all:"synthetic data" OR all:"closed-loop simulation" OR all:"point cloud generation" OR all:"novel view synthesis")',
        
        "World Model": '(all:"world model" OR all:"world models" OR all:"generative world model") AND (all:"autonomous driving" OR all:"end-to-end driving") AND (all:"action-conditioned" OR all:"future prediction" OR all:"neural simulator" OR all:"video generation")',
        
        "Driver Monitoring System": '(all:"synthetic data" OR all:"simulated image" OR all:"synthetic training data" OR all:"virtual environment") AND (all:"driver monitoring" OR all:"in-cabin" OR all:"occupant monitoring" OR all:"DMS" OR all:"OMS")'
    }

    def __init__(self):
        self.client = arxiv.Client()

    def build_query(self, base_category: str = "cs.CV", selected_domains: Optional[List[str]] = None, start_year: str = "2023", end_year: str = "2026") -> str:
        """
        Dynamically constructs an arXiv API query using optimized boolean strings.
        """
        query_parts = [f"cat:{base_category}"]
        
        # Append pre-formatted robust queries mapped from requested domains
        if selected_domains:
            domain_queries = []
            for domain in selected_domains:
                if domain in self.OPTIMIZED_QUERIES:
                    domain_queries.append(f"({self.OPTIMIZED_QUERIES[domain]})")
            
            if domain_queries:
                # If multiple use cases are selected, combine them with OR
                keyword_query = " OR ".join(domain_queries)
                query_parts.append(f"({keyword_query})")
        
        # Enforce date range filtering natively in the API query
        date_query = f"submittedDate:[{start_year}01010000 TO {end_year}12312359]"
        query_parts.append(date_query)
        
        # Combine all parameters
        return " AND ".join(query_parts)

    def fetch(self, query: str, max_results: int = 5) -> List[DocumentSchema]:
        # Switch to SubmittedDate and Descending to guarantee the most recent papers
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate, 
            sort_order=arxiv.SortOrder.Descending      
        )

        results = []
        for paper in self.client.results(search):
            doc = DocumentSchema(
                source="arxiv",
                id=paper.entry_id.split('/')[-1],
                title=paper.title,
                authors=[author.name for author in paper.authors],
                abstract=paper.summary.replace('\n', ' '),
                published_date=paper.published.strftime("%Y-%m-%d"),
                pdf_url=paper.pdf_url
            )
            results.append(doc)
            
        return results