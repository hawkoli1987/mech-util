"""
InterfaceContract: frozen geometric/functional interfaces between components.

Interfaces use geometric primitives and coordinate frames, NOT CAD internal IDs.
"""

from enum import Enum
from typing import Optional, Any
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class InterfaceType(str, Enum):
    """Interface classification."""
    
    MECHANICAL_MATE = "mechanical_mate"  # Fasteners, press-fit, etc.
    BOLT_PATTERN = "bolt_pattern"  # Standard hole patterns
    ALIGNMENT_FEATURE = "alignment_feature"  # Pins, keys, datums
    CLEARANCE_ENVELOPE = "clearance_envelope"  # Keep-out zones
    ELECTRICAL_CONNECTOR = "electrical_connector"
    FLUID_PORT = "fluid_port"


class CoordinateFrame(BaseModel):
    """
    Coordinate frame definition using geometric primitives.
    
    NO CAD internal IDs allowed; only stable geometric descriptors.
    """
    
    origin: list[float] = Field(
        ...,
        description="Origin [x, y, z] in mm",
        min_length=3,
        max_length=3
    )
    
    x_axis: list[float] = Field(
        ...,
        description="X-axis direction vector (unit)",
        min_length=3,
        max_length=3
    )
    
    y_axis: list[float] = Field(
        ...,
        description="Y-axis direction vector (unit)",
        min_length=3,
        max_length=3
    )
    
    z_axis: list[float] = Field(
        ...,
        description="Z-axis direction vector (unit)",
        min_length=3,
        max_length=3
    )
    
    reference_description: str = Field(
        ...,
        description="Human-readable reference (e.g., 'Top face of base plate')"
    )


class InterfaceContract(BaseModel):
    """
    Frozen interface specification between components/assemblies.
    
    Once frozen, changes require formal ChangeRequest.
    Uses ONLY geometric primitives—no CAD face/edge IDs.
    """
    
    # Schema metadata
    schema_version: str = Field(
        default="1.0.0",
        description="Schema version"
    )
    
    # Identity
    interface_id: str = Field(
        ...,
        description="Stable unique identifier (e.g., iface_motor_holes)"
    )
    
    parent_program_id: str = Field(
        ...,
        description="Program this interface belongs to"
    )
    
    # Participants
    component_a: str = Field(
        ...,
        description="First component ID"
    )
    
    component_b: str = Field(
        ...,
        description="Second component ID"
    )
    
    # Classification
    interface_type: InterfaceType = Field(
        ...,
        description="Type of interface"
    )
    
    description: str = Field(
        ...,
        description="Human-readable interface description",
        min_length=10
    )
    
    # Geometry (using primitives, NOT CAD IDs)
    coordinate_frame: Optional[CoordinateFrame] = Field(
        default=None,
        description="Reference frame for this interface"
    )
    
    geometric_spec: dict[str, Any] = Field(
        default_factory=dict,
        description="Geometric parameters (holes, offsets, etc.)"
    )
    
    tolerance: Optional[str] = Field(
        default=None,
        description="Tolerance specification (e.g., '±0.1mm', 'H7/g6')"
    )
    
    # Lifecycle
    is_frozen: bool = Field(
        default=False,
        description="True once locked at a design milestone"
    )
    
    frozen_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when interface was frozen"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "schema_version": "1.0.0",
                    "interface_id": "iface_motor_holes",
                    "parent_program_id": "prog_2024_motor_mount",
                    "component_a": "comp_bracket",
                    "component_b": "motor_nema17",
                    "interface_type": "bolt_pattern",
                    "description": "NEMA 17 motor bolt pattern (31mm centers)",
                    "geometric_spec": {
                        "hole_diameter": 3.2,
                        "hole_count": 4,
                        "pattern": "square",
                        "center_spacing": 31.0,
                        "depth": "through"
                    },
                    "tolerance": "±0.1mm position",
                    "is_frozen": True,
                }
            ]
        }
    )

