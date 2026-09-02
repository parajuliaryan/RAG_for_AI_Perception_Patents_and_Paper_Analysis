from pydantic import BaseModel
from typing import List

class PerceptionFinding(BaseModel):
    technology_or_algorithm_name: str
    target_use_cases: List[str]                  # v. Target Use Cases (Sensor Validation, etc.)
    core_technologies_and_buzzwords: List[str]   # vi. Core technologies / buzzwords
    sensor_types: List[str]                      # iii. Sensor types (LiDAR, Camera, Radar, etc.)
    simulators_used: List[str]                   # ii. Simulators used (CARLA, Carmaker, etc.)
    ecu_or_hardware_tested: List[str]            # vii. ECU or Hardware tested
    weather_and_environmental_parameters: List[str] # iv. Weather and environmental parameters
    tested_scenarios: List[str]                  # viii. Tested scenarios
    evaluated_kpis: List[str]                    # viii. Evaluated KPIs
    operational_constraints_or_assumptions: List[str] # ix. Operational constraints/assumptions

class FinalOutputSchema(BaseModel):
    source_document_id: str
    affiliated_companies_or_institutions: List[str] # i. Companies / Institutions involved
    summary_of_technical_relevance: str
    key_claims_or_methodologies: List[str]
    perception_findings: List[PerceptionFinding]