from pydantic import BaseModel
from typing import List

class PerceptionFinding(BaseModel):
    technology_or_algorithm_name: str
    target_use_cases: List[str]                  #Target Use Cases (Sensor Validation, etc.)
    core_technologies_and_buzzwords: List[str]   #Core technologies / buzzwords
    sensor_types: List[str]                      #Sensor types (LiDAR, Camera, Radar, etc.)
    simulators_used: List[str]                   #Simulators used (CARLA, Carmaker, AURELION etc.)
    ecu_or_hardware_tested: List[str]            #ECU or Hardware tested
    weather_and_environmental_parameters: List[str] # Weather and environmental parameters
    tested_scenarios: List[str]                  #Tested scenarios
    evaluated_kpis: List[str]                    #Evaluated KPIs
    operational_constraints_or_assumptions: List[str] #Operational constraints/assumptions

class FinalOutputSchema(BaseModel):
    source_document_id: str
    document_title: str
    document_abstract: str
    affiliated_companies_or_institutions: List[str]  #Companies / Institutions involved
    summary_of_technical_relevance: str
    key_claims_or_methodologies: List[str]
    perception_findings: List[PerceptionFinding]