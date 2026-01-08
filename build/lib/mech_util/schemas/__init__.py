"""
Shared schemas for inter-agent communication.

These schemas are used across all agent repositories:
- ComponentIntent, ComponentPackage
- AssemblyIntent, AssemblyPackage
- SimulationRequest, SimulationReport
- InterfaceContract
"""

from mech_util.schemas.agent_component import ComponentIntent, ComponentPackage
from mech_util.schemas.agent_assembly import AssemblyIntent, AssemblyPackage
from mech_util.schemas.agent_simulation import (
    SimulationRequest,
    SimulationReport,
    SimulationType,
    SimulationStatus,
    BoundaryCondition,
    BoundaryConditionType,
    Load,
    LoadType,
    MaterialProperties,
    MeshSettings,
)
from mech_util.schemas.interface_contract import (
    InterfaceContract,
    InterfaceType,
    CoordinateFrame,
)

__all__ = [
    # Component schemas
    "ComponentIntent",
    "ComponentPackage",
    # Assembly schemas
    "AssemblyIntent",
    "AssemblyPackage",
    # Simulation schemas
    "SimulationRequest",
    "SimulationReport",
    "SimulationType",
    "SimulationStatus",
    "BoundaryCondition",
    "BoundaryConditionType",
    "Load",
    "LoadType",
    "MaterialProperties",
    "MeshSettings",
    # Interface schemas
    "InterfaceContract",
    "InterfaceType",
    "CoordinateFrame",
]

