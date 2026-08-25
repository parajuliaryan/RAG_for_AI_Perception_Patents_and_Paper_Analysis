from pydantic import BaseModel
from typing import List

class PerceptionFinding(BaseModel):
    technology_mentioned: str
    capabilities: List[str]
    limitations: List[str]
    technical_parameters: List[str]  # e.g. ["Range: 200m", "Dataset: nuScenes"]

class FinalOutputSchema(BaseModel):
    paper_id: str
    summary_of_relevance: str
    perception_findings: List[PerceptionFinding]