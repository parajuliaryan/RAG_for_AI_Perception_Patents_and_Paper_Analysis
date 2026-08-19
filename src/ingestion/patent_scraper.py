import os
import requests
import json
from typing import List, Optional
from src.ingestion.base_scraper import BaseScraper
from src.schemas.document import DocumentSchema

class PatentScraper(BaseScraper):
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

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("PATENTSVIEW_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set PATENTSVIEW_API_KEY environment variable.")
            
        self.base_url = "https://search.patentsview.org/api/v1/patent/"
        self.headers = {
            "X-Api-Key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

    def build_query(self, selected_domains: Optional[List[str]] = None, start_date: str = "2023-01-01") -> dict:
        """
        Constructs a PatentsView JSON query targeting abstracts using exact phrase matches.
        """
        all_keywords = []
        if selected_domains:
            for domain in selected_domains:
                if domain in self.DOMAIN_KEYWORDS:
                    all_keywords.extend(self.DOMAIN_KEYWORDS[domain])
        
        # Build keyword OR conditions using _text_phrase for exact multi-word matches
        keyword_conditions = [
            {"_text_phrase": {"patent_abstract": kw}} for kw in all_keywords
        ]
        
        # Combine the date filter with the keyword filters using an AND operator
        query = {
            "_and": [
                {"_gte": {"patent_date": start_date}}
            ]
        }
        
        if keyword_conditions:
            query["_and"].append({"_or": keyword_conditions})
            
        return query

    def fetch(self, query: dict, max_results: int = 5) -> List[DocumentSchema]:
        fields = [
            "patent_id",
            "patent_title",
            "patent_abstract",
            "patent_date",
            "inventors"
        ]
        
        payload = {
            "q": query,
            "f": fields,
            "o": {"per_page": max_results}
        }
        
        response = requests.get(
            self.base_url, 
            headers=self.headers,
            # PatentsView requires the JSON payload dictionaries to be URL-encoded strings
            params={"q": json.dumps(payload["q"]), "f": json.dumps(payload["f"]), "o": json.dumps(payload["o"])}
        )
        
        response.raise_for_status()
        data = response.json()
        
        results = []
        for patent in data.get("patents", []):
            inventor_list = patent.get("inventors", [])
            authors = []
            
            # Safely extract first and last names, since sometimes they are missing
            for inv in inventor_list:
                first = inv.get('inventor_name_first') or ''
                last = inv.get('inventor_name_last') or ''
                full_name = f"{first} {last}".strip()
                if full_name:
                    authors.append(full_name)
                
            doc = DocumentSchema(
                source="patentsview",
                id=patent.get("patent_id"),
                title=patent.get("patent_title", "Unknown Title"),
                authors=authors if authors else ["Unknown Inventor"],
                abstract=patent.get("patent_abstract", "No abstract available"),
                published_date=patent.get("patent_date"),
                pdf_url=f"https://patents.google.com/patent/US{patent.get('patent_id')}/en"
            )
            results.append(doc)
            
        return results