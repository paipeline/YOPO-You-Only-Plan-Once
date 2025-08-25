"""
Shared models for the planning module.
"""
from typing import List, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class Node(BaseModel):
    """Represents a node in the DAG."""
    idx: str
    title: str
    description: str
    operator: str


class Edge(BaseModel):
    """Represents an edge connecting nodes in the DAG."""
    from_idx: str
    to_idx: str


class DAG(BaseModel):
    """Directed Acyclic Graph structure for the plan."""
    nodes: List[Node]
    edges: List[Edge]


class Contract(BaseModel):
    """Contract defining the interface and behavior of a node."""
    id: str
    inputs_schema: Dict[str, Any]
    outputs_schema: Dict[str, Any]
    verification: str
    tool_calls: List[Dict[str, Any]] = []
    fallbacks: List[str] = []
    version: str = "1"


class CanonicalQuery(BaseModel):
    """User query with tags."""
    goal: str
    entities: List[str]  # objects or concepts that are relevant to the goal
    constraints: List[str]
    success_criteria: List[str]


class PlanBlueprint(BaseModel):
    """Complete plan record with DAG and contracts."""
    plan_id: str
    dag: DAG
    contracts: Dict[str, Contract]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PlanPayload(BaseModel):
    """Payload for executing a plan."""
    canonical_query: CanonicalQuery
    plan_blueprint: PlanBlueprint


class ExecutableView:
    """View for executing a plan with injected parameters."""
    pass