from pydantic import BaseModel, Field
from typing import List, Optional

class PerceptionFinding(BaseModel):
    technology_or_algorithm_name: str
    target_use_cases: List[str]
    core_technologies_and_buzzwords: List[str]
    sensor_types: List[str]
    simulation_tools_or_solvers: List[str]
    hardware_target_or_deployment_platform: List[str]
    weather_and_environmental_parameters: List[str]
    tested_scenarios: List[str]
    evaluated_kpis: List[str]
    operational_constraints_or_assumptions: List[str]

class PatentMetadata(BaseModel):
    patent_classifications: List[str]
    priority_or_grant_date: str
    independent_claims_scope: List[str]

class FinalOutputSchema(BaseModel):
    source_document_id: str
    document_title: str
    document_abstract: str
    affiliated_companies_or_institutions: List[str]
    summary_of_technical_relevance: str
    key_claims_or_methodologies: List[str]
    perception_findings: List[PerceptionFinding]
    patent_metadata: Optional[PatentMetadata] = None