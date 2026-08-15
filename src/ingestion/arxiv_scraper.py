import arxiv
from typing import List, Optional
from src.ingestion.base_scraper import BaseScraper
from src.schemas.document import DocumentSchema

class ArxivScraper(BaseScraper):
    DOMAIN_KEYWORDS = {
        "Simulation Platforms": [
            "CARLA", "dSPACE AURELION", "Carmaker", "Gazebo", "AiSim", 
            "NVIDIA Drive Sim", "Cognata", "LGSVL", "Metadrive"
        ],
        "Perception & World models": [
            "Neural Rendering", "NeRF", "World Models", "Occupancy networks", 
            "Diffusion Models", "Generative Simulation", "Closed-loop perception"
        ],
        "Sensors & Environment": [
            "LiDAR simulation", "Radar-ray tracing", "Monocular/Stereo camera", 
            "Ultrasonic", "IMU", "Weather parameters", "Physics-based sensor models"
        ],
        "Validation & Testing": [
            "Virtual Homologation", "Sensor Model Validation", "Hardware in the Loop (HIL)", 
            "Software in the Loop (SIL)", "ECU testing", "Driver Monitoring System (DMS)", 
            "Edge-case testing"
        ]
    }

    def __init__(self):
        self.client = arxiv.Client()

    def build_query(self, base_category: str = "cs.CV", selected_domains: Optional[List[str]] = None, start_year: str = "2023", end_year: str = "2026") -> str:
        """
        Dynamically constructs an arXiv API query string with categories, keyword OR-logic, and date ranges.
        """
        query_parts = [f"cat:{base_category}"]
        
        # Append buzzwords mapped from requested domains
        if selected_domains:
            all_keywords = []
            for domain in selected_domains:
                if domain in self.DOMAIN_KEYWORDS:
                    # Enclose terms in double quotes for exact phrase matching
                    all_keywords.extend([f'"{kw}"' for kw in self.DOMAIN_KEYWORDS[domain]])
            
            if all_keywords:
                # arXiv interprets "all:" as searching within title, abstract, and authors
                keyword_query = " OR ".join([f"all:{kw}" for kw in all_keywords])
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