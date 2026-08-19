from pydantic import BaseModel
from typing import List

class PerceptionFinding(BaseModel):
    technology_mentioned: str
    capabilities: List[str]
    limitations: List[str]

class FinalOutputSchema(BaseModel):
    paper_id: str
    summary_of_relevance: str
    perception_findings: List[PerceptionFinding]