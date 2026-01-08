"""
Component Agent Communication Schemas.

Request: ComponentIntent (sent to component agent)
Response: ComponentPackage (returned by component agent)
"""

from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ComponentIntent(BaseModel):
    """
    Design intent for a single component.
    
    Sent to component-agent-cadquery for CAD generation.
    Includes material, process, and design constraints.
    """
    
    # Schema metadata
    schema_version: str = Field(
        default="1.0.0",
        description="Schema version"
    )
    
    # Identity
    component_id: str = Field(
        ...,
        description="Stable unique identifier (e.g., comp_bracket)"
    )
    
    parent_program_id: str = Field(
        ...,
        description="Program this component belongs to"
    )
    
    parent_assembly_id: Optional[str] = Field(
        default=None,
        description="Parent assembly if this is a sub-component"
    )
    
    # Intent
    component_name: str = Field(
        ...,
        description="Human-readable component name",
        min_length=1
    )
    
    functional_description: str = Field(
        ...,
        description="What this component does",
        min_length=10
    )
    
    # Constraints
    interface_contracts: list[str] = Field(
        default_factory=list,
        description="List of InterfaceContract IDs this component must satisfy"
    )
    
    material_preference: Optional[str] = Field(
        default=None,
        description="Material suggestion (e.g., 'aluminum 6061', 'PLA')"
    )
    
    manufacturing_process: Optional[str] = Field(
        default=None,
        description="Intended process (e.g., 'CNC milling', '3D printing')"
    )
    
    design_constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional constraints (max mass, envelope, etc.)"
    )
    
    # References
    reference_requirements: list[str] = Field(
        default_factory=list,
        description="SystemRequirement IDs this component addresses"
    )
    
    # Lifecycle
    revision: int = Field(
        default=1,
        description="Design revision number"
    )
    
    is_frozen: bool = Field(
        default=False,
        description="True once design is locked"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.utcnow
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "schema_version": "1.0.0",
                    "component_id": "comp_bracket",
                    "parent_program_id": "prog_2024_motor_mount",
                    "component_name": "Motor Mount Bracket",
                    "functional_description": "L-shaped bracket to mount NEMA 17 motor to 20x20 extrusion",
                    "interface_contracts": ["iface_motor_holes", "iface_extrusion_slot"],
                    "material_preference": "aluminum 6061-T6",
                    "manufacturing_process": "CNC milling",
                    "design_constraints": {
                        "max_mass_kg": 0.1,
                        "min_wall_thickness_mm": 3.0
                    },
                }
            ]
        }
    )


class ComponentPackage(BaseModel):
    """
    Complete component package returned by component-agent-cadquery.
    
    Includes artifact URIs, semantic tags, and validation status.
    """
    
    # Schema metadata
    schema_version: str = Field(
        default="1.0.0",
        description="Schema version"
    )
    
    # Identity
    component_id: str = Field(
        ...,
        description="Component identifier (matches request)"
    )
    
    parent_program_id: str = Field(
        ...,
        description="Program context"
    )
    
    # Artifacts
    artifact_uris: dict[str, str] = Field(
        ...,
        description="URIs to CAD files (step, stl, glb, etc.)"
    )
    
    # Semantic tagging
    semantic_tags: dict[str, str] = Field(
        default_factory=dict,
        description="Map of semantic tag names to descriptions (for assembly reference)"
    )
    
    # Parameters
    design_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Final parameter values used in design"
    )
    
    # Validation
    validation_status: str = Field(
        ...,
        description="Validation result: 'passed', 'warnings', 'failed'"
    )
    
    validation_messages: list[str] = Field(
        default_factory=list,
        description="Validation findings (e.g., 'wall thickness 3.5mm > 3mm min')"
    )
    
    # Interface compliance
    interface_compliance: dict[str, bool] = Field(
        default_factory=dict,
        description="Map of interface_id to compliance status (True/False)"
    )
    
    # Metadata
    design_notes: Optional[str] = Field(
        default=None,
        description="Human-readable design notes or assumptions"
    )
    
    generated_at: datetime = Field(
        default_factory=datetime.utcnow
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "schema_version": "1.0.0",
                    "component_id": "comp_bracket",
                    "parent_program_id": "prog_2024_gripper",
                    "artifact_uris": {
                        "step": "storage/artifacts/prog_2024_gripper/components/comp_bracket_rev1.step",
                        "stl": "storage/artifacts/prog_2024_gripper/components/comp_bracket_rev1.stl",
                        "glb": "storage/artifacts/prog_2024_gripper/components/comp_bracket_rev1.glb"
                    },
                    "semantic_tags": {
                        "mounting_holes": "Four M3 holes at 31mm centers for motor attachment",
                        "base_surface": "Bottom mating surface, 50x50mm planar",
                        "datum_a": "Primary datum for GD&T reference"
                    },
                    "design_parameters": {
                        "base_thickness_mm": 5.0,
                        "hole_spacing_mm": 31.0,
                        "hole_diameter_mm": 3.2,
                        "overall_width_mm": 50.0,
                        "overall_length_mm": 60.0
                    },
                    "validation_status": "passed",
                    "validation_messages": [
                        "Minimum wall thickness: 4.2mm > 3mm requirement ✓",
                        "All interface features present and tagged ✓",
                        "Solid is manifold and valid ✓"
                    ],
                    "interface_compliance": {
                        "iface_motor_holes": True,
                        "iface_base_mount": True
                    },
                    "design_notes": "L-bracket design with reinforcement ribs for rigidity"
                }
            ]
        }
    )

