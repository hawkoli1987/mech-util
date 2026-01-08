"""
Simulation Agent Communication Schemas.

Request: SimulationRequest (sent to simulation agent)
Response: SimulationReport (returned by simulation agent)

Critical: All geometric regions use primitives (NO CAD internal IDs!)
"""

from enum import Enum
from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Shared Enums
# ============================================================================

class SimulationType(str, Enum):
    """Types of engineering simulations."""
    STATIC_STRESS = "static_stress"
    MODAL_ANALYSIS = "modal_analysis"
    THERMAL_STEADY_STATE = "thermal_steady_state"
    THERMAL_TRANSIENT = "thermal_transient"
    BUCKLING = "buckling"
    FATIGUE = "fatigue"


class BoundaryConditionType(str, Enum):
    """Types of boundary conditions."""
    FIXED = "fixed"
    DISPLACEMENT = "displacement"
    SYMMETRY = "symmetry"
    CONTACT = "contact"
    REMOTE_FORCE = "remote_force"


class LoadType(str, Enum):
    """Types of loads."""
    FORCE = "force"
    PRESSURE = "pressure"
    MOMENT = "moment"
    GRAVITY = "gravity"
    TEMPERATURE = "temperature"


class SimulationStatus(str, Enum):
    """Simulation completion status."""
    SUCCESS = "success"
    FAILED = "failed"
    DIVERGED = "diverged"


# ============================================================================
# Request Components
# ============================================================================

class BoundaryCondition(BaseModel):
    """
    Boundary condition specified using geometric descriptions (NO CAD IDs!).
    
    Critical: All geometric regions must be described using primitives like
    coordinates, normal vectors, planes, etc. Never use face_id, edge_id, etc.
    """
    
    bc_type: BoundaryConditionType = Field(
        ...,
        description="Type of boundary condition"
    )
    
    geometric_region: dict[str, Any] = Field(
        ...,
        description="Geometric description of where BC applies (NO CAD IDs!)"
    )
    
    values: dict[str, float] = Field(
        default_factory=dict,
        description="BC values (e.g., displacement magnitude, etc.)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "bc_type": "fixed",
                    "geometric_region": {
                        "region_type": "planar_face",
                        "normal_vector": [0, 0, -1],
                        "z_coordinate": 0.0,
                        "tolerance": 0.1
                    },
                    "values": {}
                }
            ]
        }
    )


class Load(BaseModel):
    """Load specified using geometric descriptions (NO CAD IDs!)."""
    
    load_type: LoadType = Field(
        ...,
        description="Type of load"
    )
    
    magnitude: float = Field(
        ...,
        description="Load magnitude (units depend on type: N for force, Pa for pressure)"
    )
    
    direction: Optional[list[float]] = Field(
        default=None,
        description="Load direction vector [x, y, z] for directional loads"
    )
    
    geometric_region: dict[str, Any] = Field(
        ...,
        description="Geometric description of where load applies (NO CAD IDs!)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "load_type": "force",
                    "magnitude": 100.0,
                    "direction": [0, 0, -1],
                    "geometric_region": {
                        "region_type": "point",
                        "coordinates": [25.0, 25.0, 50.0]
                    }
                }
            ]
        }
    )


class MaterialProperties(BaseModel):
    """Material properties for simulation."""
    
    material_name: str = Field(
        ...,
        description="Material name (e.g., '6061-T6 Aluminum')"
    )
    
    youngs_modulus_gpa: float = Field(
        ...,
        description="Young's modulus in GPa",
        gt=0
    )
    
    poissons_ratio: float = Field(
        ...,
        description="Poisson's ratio (dimensionless)",
        ge=0,
        le=0.5
    )
    
    density_kg_m3: float = Field(
        ...,
        description="Density in kg/m³",
        gt=0
    )
    
    yield_strength_mpa: Optional[float] = Field(
        default=None,
        description="Yield strength in MPa",
        gt=0
    )
    
    ultimate_strength_mpa: Optional[float] = Field(
        default=None,
        description="Ultimate tensile strength in MPa",
        gt=0
    )
    
    thermal_conductivity_w_mk: Optional[float] = Field(
        default=None,
        description="Thermal conductivity in W/(m·K)"
    )
    
    specific_heat_j_kgk: Optional[float] = Field(
        default=None,
        description="Specific heat capacity in J/(kg·K)"
    )


class MeshSettings(BaseModel):
    """Mesh settings for FEA."""
    
    element_size_mm: float = Field(
        ...,
        description="Global element size in mm",
        gt=0
    )
    
    refinement_regions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Regions requiring finer mesh (using geometric descriptions)"
    )
    
    element_order: int = Field(
        default=1,
        description="Element order (1=linear, 2=quadratic)",
        ge=1,
        le=2
    )


# ============================================================================
# Simulation Request
# ============================================================================

class SimulationRequest(BaseModel):
    """
    Complete simulation request for external FEA agent.
    
    Uses ONLY geometric descriptions (NO CAD internal IDs).
    """
    
    # Schema metadata
    schema_version: str = Field(
        default="1.0.0",
        description="Schema version"
    )
    
    # Identity
    simulation_id: str = Field(
        ...,
        description="Unique simulation identifier"
    )
    
    component_id: str = Field(
        ...,
        description="Component being simulated"
    )
    
    parent_program_id: str = Field(
        ...,
        description="Parent program context"
    )
    
    # Artifact reference
    artifact_uri: str = Field(
        ...,
        description="URI to CAD file (STEP, etc.)"
    )
    
    # Simulation type
    simulation_type: SimulationType = Field(
        ...,
        description="Type of simulation"
    )
    
    # Physics
    boundary_conditions: list[BoundaryCondition] = Field(
        default_factory=list,
        description="Boundary conditions using geometric descriptions"
    )
    
    loads: list[Load] = Field(
        default_factory=list,
        description="Loads using geometric descriptions"
    )
    
    material_properties: MaterialProperties = Field(
        ...,
        description="Material properties"
    )
    
    # Mesh
    mesh_settings: MeshSettings = Field(
        ...,
        description="Mesh generation settings"
    )
    
    # Analysis settings
    analysis_settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Solver-specific settings"
    )


# ============================================================================
# Simulation Response
# ============================================================================

class SimulationReport(BaseModel):
    """
    Simulation results from simulation-agent-calculix.
    
    Contains results summary and artifact URIs (VTK, ParaView).
    """
    
    # Schema metadata
    schema_version: str = Field(
        default="1.0.0",
        description="Schema version"
    )
    
    # Identity
    report_id: str = Field(
        ...,
        description="Unique report identifier"
    )
    
    component_id: str = Field(
        ...,
        description="Component/assembly simulated"
    )
    
    parent_program_id: str = Field(
        ...,
        description="Program context"
    )
    
    # Analysis
    simulation_type: SimulationType = Field(
        ...,
        description="Type of simulation performed"
    )
    
    status: SimulationStatus = Field(
        ...,
        description="Simulation completion status"
    )
    
    # Results summary
    max_stress_mpa: Optional[float] = Field(
        default=None,
        description="Maximum von Mises stress (MPa)"
    )
    
    max_displacement_mm: Optional[float] = Field(
        default=None,
        description="Maximum displacement (mm)"
    )
    
    safety_factor: Optional[float] = Field(
        default=None,
        description="Minimum safety factor"
    )
    
    pass_fail: Optional[bool] = Field(
        default=None,
        description="True if design passes acceptance criteria"
    )
    
    summary: str = Field(
        default="",
        description="Human-readable summary of results"
    )
    
    # Artifacts
    artifact_refs: list[str] = Field(
        default_factory=list,
        description="URIs to VTK, ParaView, mesh files"
    )
    
    # Metadata
    generated_at: datetime = Field(
        default_factory=datetime.utcnow
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "schema_version": "1.0.0",
                    "report_id": "sim_001",
                    "component_id": "comp_bracket",
                    "parent_program_id": "prog_2024_motor_mount",
                    "simulation_type": "static_stress",
                    "status": "success",
                    "max_stress_mpa": 45.2,
                    "max_displacement_mm": 0.08,
                    "safety_factor": 5.3,
                    "pass_fail": True,
                    "summary": "Stress well below yield; max displacement acceptable",
                    "artifact_refs": ["storage/artifacts/prog_2024_motor_mount/simulation/rev1/stress.vtk"],
                }
            ]
        }
    )

