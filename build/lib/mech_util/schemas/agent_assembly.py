"""
Assembly Agent Communication Schemas.

Request: AssemblyIntent (sent to assembly agent)
Response: AssemblyPackage (returned by assembly agent)
"""

from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class AssemblyIntent(BaseModel):
    """
    Design intent for an assembly.
    
    Sent to assembly-agent-freecad for mating and hierarchy management.
    """
    
    # Schema metadata
    schema_version: str = Field(
        default="1.0.0",
        description="Schema version"
    )
    
    # Identity
    assembly_id: str = Field(
        ...,
        description="Stable unique identifier (e.g., asm_mount)"
    )
    
    parent_program_id: str = Field(
        ...,
        description="Program this assembly belongs to"
    )
    
    parent_assembly_id: Optional[str] = Field(
        default=None,
        description="Parent assembly for nested assemblies"
    )
    
    # Intent
    assembly_name: str = Field(
        ...,
        description="Human-readable assembly name",
        min_length=1
    )
    
    functional_description: str = Field(
        ...,
        description="What this assembly does",
        min_length=10
    )
    
    # Structure
    child_components: list[str] = Field(
        default_factory=list,
        description="List of ComponentIntent IDs in this assembly"
    )
    
    child_assemblies: list[str] = Field(
        default_factory=list,
        description="List of sub-AssemblyIntent IDs"
    )
    
    # Mating
    interface_contracts: list[str] = Field(
        default_factory=list,
        description="InterfaceContract IDs defining internal mates"
    )
    
    mating_instructions: Optional[str] = Field(
        default=None,
        description="High-level assembly instructions (e.g., 'motor bolts to bracket via 4x M3 screws')"
    )
    
    # Constraints
    assembly_constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Assembly-level constraints (total mass, envelope, etc.)"
    )
    
    # References
    reference_requirements: list[str] = Field(
        default_factory=list,
        description="SystemRequirement IDs this assembly addresses"
    )
    
    # Lifecycle
    revision: int = Field(
        default=1,
        description="Assembly revision number"
    )
    
    is_frozen: bool = Field(
        default=False,
        description="True once assembly design is locked"
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
                    "assembly_id": "asm_mount",
                    "parent_program_id": "prog_2024_motor_mount",
                    "assembly_name": "Motor Mount Assembly",
                    "functional_description": "Complete assembly for mounting NEMA 17 motor to extrusion",
                    "child_components": ["comp_bracket", "comp_spacer"],
                    "interface_contracts": ["iface_motor_holes", "iface_extrusion_slot"],
                    "mating_instructions": "Motor fastened to bracket with 4x M3x8mm screws; bracket slides into extrusion T-slot",
                }
            ]
        }
    )


class AssemblyPackage(BaseModel):
    """
    Complete assembly package returned by assembly-agent-freecad.
    
    Includes assembly artifacts, BOM, and validation status.
    """
    
    # Schema metadata
    schema_version: str = Field(
        default="1.0.0",
        description="Schema version"
    )
    
    # Identity
    assembly_id: str = Field(
        ...,
        description="Assembly identifier (matches request)"
    )
    
    parent_program_id: str = Field(
        ...,
        description="Program context"
    )
    
    # Artifacts
    artifact_uris: dict[str, str] = Field(
        ...,
        description="URIs to assembly files (fcstd, step, glb, etc.)"
    )
    
    # Structure
    bom: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Bill of materials with component IDs and quantities"
    )
    
    mating_relationships: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of component-to-component mating relationships"
    )
    
    # Validation
    validation_status: str = Field(
        ...,
        description="Validation result: 'passed', 'warnings', 'failed'"
    )
    
    interference_check: dict[str, Any] = Field(
        default_factory=dict,
        description="Interference/collision check results"
    )
    
    clearance_violations: list[str] = Field(
        default_factory=list,
        description="List of clearance violations if any"
    )
    
    envelope_compliance: Optional[bool] = Field(
        default=None,
        description="True if assembly fits within envelope constraints"
    )
    
    # Metadata
    assembly_notes: Optional[str] = Field(
        default=None,
        description="Assembly instructions or notes"
    )
    
    generated_at: datetime = Field(
        default_factory=datetime.utcnow
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "schema_version": "1.0.0",
                    "assembly_id": "asm_gripper",
                    "parent_program_id": "prog_2024_gripper",
                    "artifact_uris": {
                        "fcstd": "storage/artifacts/prog_2024_gripper/assemblies/asm_gripper_rev1.fcstd",
                        "step": "storage/artifacts/prog_2024_gripper/assemblies/asm_gripper_rev1.step",
                        "glb": "storage/artifacts/prog_2024_gripper/assemblies/asm_gripper_rev1.glb"
                    },
                    "bom": [
                        {"component_id": "comp_base", "component_name": "Base Plate", "quantity": 1},
                        {"component_id": "comp_jaw_left", "component_name": "Left Jaw", "quantity": 1},
                        {"component_id": "comp_jaw_right", "component_name": "Right Jaw", "quantity": 1},
                        {"component_id": "comp_actuator", "component_name": "Linear Actuator", "quantity": 1}
                    ],
                    "mating_relationships": [
                        {
                            "interface_id": "iface_base_jaw_left",
                            "component_a": "comp_base",
                            "component_b": "comp_jaw_left",
                            "mate_type": "planar_coincident"
                        }
                    ],
                    "validation_status": "passed",
                    "interference_check": {
                        "interferences_found": 0,
                        "checked_pairs": 6
                    },
                    "clearance_violations": [],
                    "envelope_compliance": True,
                    "assembly_notes": "All components mate correctly; actuator centered between jaws"
                }
            ]
        }
    )

