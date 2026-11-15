from pydantic import BaseModel, Field
from typing import List, Optional


class PerceptionFinding(BaseModel):
    technology_or_algorithm_name: str = Field(
        description=(
            "Exact name or acronym of the proposed method, model, framework, sensor simulation architecture, "
            "or patented invention (e.g., 'CoDa-4DGS', 'SplatAD', 'RADAR FMU Model', 'CV&V Development Method')."
        )
    )
    target_use_cases: List[str] = Field(
        description=(
            "Specific automotive, robotics, or perception applications targeted (e.g., 'Autonomous driving perception', "
            "'FMCW radar sensor validation', 'Closed-loop HIL testing', 'Driver monitoring system')."
        )
    )
    core_technologies_and_buzzwords: List[str] = Field(
        description=(
            "Key algorithms, mathematical models, neural architectures, physical principles, and technical buzzwords "
            "(e.g., '3D Gaussian Splatting', 'Ray tracing', 'Range-Doppler FFT', 'OS-CFAR', 'DBSCAN', 'Diffusion Transformer')."
        )
    )
    sensor_types: List[str] = Field(
        description=(
            "All perception or reference sensor modalities involved (e.g., 'FMCW Radar (77 GHz)', 'LiDAR', "
            "'RGB Camera', 'Event-based camera (DVS)', 'Laser rangefinder', 'Time-of-Flight (ToF) sensor')."
        )
    )
    simulation_tools_or_solvers: List[str] = Field(
        description=(
            "All simulation environments, game/physics engines, ray-tracing or sensor modeling libraries, "
            "electromagnetic/optical solvers, validation frameworks, and hardware-in-the-loop target simulators "
            "or echo generators mentioned in the document (e.g., 'CARLA', 'SUMO', 'dSPACE AURELION', "
            "'dSPACE DARTS 9040-G', 'Rohde & Schwarz R&S AREG800A', 'SMDL', 'Avelon', 'ANSYS HFSS', 'SBR+', "
            "'Vector CANoe', 'v2e', 'OpenDRIVE', 'RoadRunner'). Only output ['None'] if no simulator, solver, "
            "modeling library, or target generator is mentioned anywhere in the context."
        )
    )
    hardware_target_or_deployment_platform: List[str] = Field(
        description=(
            "All compute GPUs/CPUs/SoCs, embedded ECUs, physical test vehicles/robots, physical radar/LiDAR/camera "
            "sensor boards under test, reference measurement devices, and HIL interface hardware mentioned in the "
            "document (e.g., 'Infineon 77 GHz P2S radar board', 'NXP RDK-S32R274 radar sensor', 'Ouster OS1 32-layer LiDAR', "
            "'Bosch GLM 120 C laser rangefinder', 'NVIDIA A100 / H100 / RTX 3090 / RTX 4090 GPU', 'NVIDIA Drive Orin', "
            "'Vector VN1640A CAN interface', 'Vehicle ECU'). Only output ['None'] if no physical sensor unit, hardware "
            "board, compute GPU/ECU, or test vehicle is mentioned anywhere in the context."
        )
    )
    weather_and_environmental_parameters: List[str] = Field(
        description=(
            "All weather conditions, lighting conditions, indoor/outdoor test settings, road geometry/surface factors, "
            "or physical/electromagnetic propagation phenomena mentioned (e.g., 'Indoors', 'Outdoors', 'Night', 'Daytime', "
            "'Rain', 'Fog', 'Snow', 'Glare', 'Ambient light suppression', 'Multipath reflections', 'Phase noise', "
            "'Thermal noise'). Only output ['None'] if no environmental, lighting, indoor/outdoor, or physical clutter "
            "parameter is mentioned."
        )
    )
    tested_scenarios: List[str] = Field(
        description=(
            "All experimental validation setups, target configurations, driving maneuvers, traffic scenarios, "
            "or benchmark datasets evaluated (e.g., 'Static cyclist at 30.2 m masked by static truck at 29 m', "
            "'Corner reflector at 5.17 m', 'Square Lambertian target (80% reflectance) at 10 distances', "
            "'Follow leading vehicle', 'Cyclist crossing', 'nuScenes', 'Waymo Open Dataset', 'NAVSIM', 'Town10HD')."
        )
    )
    evaluated_kpis: List[str] = Field(
        description=(
            "All quantitative evaluation metrics, signal outputs, accuracy measures, or KPIs reported "
            "(e.g., 'Range Map (RM)', 'Range-Doppler Map (RDM)', 'Mean Absolute Percentage Error (MAPE)', "
            "'Equivalent Noise Bandwidth (ENBW)', 'PSNR', 'SSIM', 'LPIPS', 'Chamfer Distance', 'PDMS', 'EPDMS', "
            "'mAP', 'IoU', 'FPS')."
        )
    )
    operational_constraints_or_assumptions: List[str] = Field(
        description=(
            "All modeling simplifications, physical or geometric assumptions, operational limitations, or boundary "
            "conditions stated (e.g., 'Single-beam validation simplification', '1D virtual antenna array (no elevation)', "
            "'Far-field point-scatterer approximation', 'Static environment assumption', 'Real-time latency constraint')."
        )
    )


class PatentMetadata(BaseModel):
    patent_classifications: List[str] = Field(
        description="All IPC and CPC patent classification codes listed in the patent metadata."
    )
    priority_or_grant_date: str = Field(
        description="Publication/Grant date and earliest priority/filing date exactly as listed in the patent metadata."
    )
    independent_claims_scope: List[str] = Field(
        description="All independent claims from 'Independent Claims Scope', preserving each 'Claim X:' prefix."
    )


class FinalOutputSchema(BaseModel):
    source_document_id: str = Field(
        description="Exact 'Source ID' of the primary target document (e.g., DOI, arXiv ID, or Patent Publication/Grant Number)."
    )
    document_title: str = Field(
        description="Exact 'Document Title' of the primary target document."
    )
    document_abstract: str = Field(
        description="Full abstract of the primary target document from 'Document Abstract'."
    )
    affiliated_companies_or_institutions: List[str] = Field(
        description=(
            "All universities, research institutes, and corporate assignees/companies affiliated with the primary "
            "target document (extract ALL institutions listed in 'Affiliated Institutions / Assignee' plus any "
            "author affiliations in the text)."
        )
    )
    summary_of_technical_relevance: str = Field(
        description="Concise technical synthesis explaining how the document addresses the user's research query."
    )
    key_claims_or_methodologies: List[str] = Field(
        description="Core technical contributions, architectural steps, or patent claim mechanisms."
    )
    perception_findings: List[PerceptionFinding] = Field(
        description="Detailed structured technical findings extracted from the primary target document."
    )
    patent_metadata: Optional[PatentMetadata] = Field(
        default=None,
        description="Populated ONLY when the primary target document is a patent; must be null/None for academic papers.",
    )
