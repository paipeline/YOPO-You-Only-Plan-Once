"""
Blueprint Templates - Reusable abstract patterns for common workflows
"""
from typing import Dict, List, Optional, Tuple
from src.core.planning.models import DAG, Node, Edge, Contract
import json


class BlueprintTemplate:
    """Base class for blueprint templates."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def create_dag(self) -> DAG:
        """Create the DAG structure for this template."""
        raise NotImplementedError
    
    def create_contracts(self) -> Dict[str, Contract]:
        """Create the contracts for this template."""
        raise NotImplementedError
    
    def customize(self, params: Dict) -> Tuple[DAG, Dict[str, Contract]]:
        """Customize the template with specific parameters."""
        dag = self.create_dag()
        contracts = self.create_contracts()
        
        # Apply customizations based on params
        return self._apply_customizations(dag, contracts, params)
    
    def _apply_customizations(self, dag: DAG, contracts: Dict[str, Contract], params: Dict) -> Tuple[DAG, Dict[str, Contract]]:
        """Apply domain-specific customizations to the template."""
        # Override in subclasses for specific customization logic
        return dag, contracts


class SearchFilterActTemplate(BlueprintTemplate):
    """Template for Search -> Filter -> Act pattern."""
    
    def __init__(self):
        super().__init__(
            name="search_filter_act",
            description="Generic pattern for searching, filtering results, and taking action"
        )
    
    def create_dag(self) -> DAG:
        return DAG(
            nodes=[
                Node(
                    idx="0",
                    title="Discover Candidates",
                    description="Search for relevant items based on query context and constraints",
                    operator="retrieve"
                ),
                Node(
                    idx="1",
                    title="Apply Filters",
                    description="Filter candidates based on specified criteria and constraints",
                    operator="extract"
                ),
                Node(
                    idx="2",
                    title="Synthesize Results",
                    description="Produce final deliverable from filtered candidates",
                    operator="act"
                )
            ],
            edges=[
                Edge(from_idx="0", to_idx="1"),
                Edge(from_idx="1", to_idx="2")
            ]
        )
    
    def create_contracts(self) -> Dict[str, Contract]:
        return {
            "0": Contract(
                id="0",
                inputs_schema={
                    "type": "object",
                    "properties": {
                        "context": {"type": "object"},
                        "query": {"type": "string"},
                        "constraints": {"type": "array", "items": {"type": "string"}},
                        "search_params": {"type": "object"}
                    },
                    "required": ["context", "query"],
                    "additionalProperties": True
                },
                outputs_schema={
                    "type": "object",
                    "properties": {
                        "candidates": {"type": "array", "items": {"type": "object"}},
                        "metadata": {"type": "object"}
                    },
                    "required": ["candidates"],
                    "additionalProperties": True
                },
                verification="len(candidates) >= 1",
                tool_calls=[{"name": "search", "args": {"mode": "broad"}}],
                fallbacks=["retry_with_backoff", "alternative_search"],
                version="1"
            ),
            "1": Contract(
                id="1",
                inputs_schema={
                    "type": "object",
                    "properties": {
                        "candidates": {"type": "array", "items": {"type": "object"}},
                        "filters": {"type": "array", "items": {"type": "object"}},
                        "constraints": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["candidates"],
                    "additionalProperties": True
                },
                outputs_schema={
                    "type": "object",
                    "properties": {
                        "filtered_items": {"type": "array", "items": {"type": "object"}},
                        "filter_metadata": {"type": "object"}
                    },
                    "required": ["filtered_items"],
                    "additionalProperties": True
                },
                verification="len(filtered_items) >= 1",
                tool_calls=[],
                fallbacks=[],
                version="1"
            ),
            "2": Contract(
                id="2",
                inputs_schema={
                    "type": "object",
                    "properties": {
                        "filtered_items": {"type": "array", "items": {"type": "object"}},
                        "output_format": {"type": "string"},
                        "success_criteria": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["filtered_items"],
                    "additionalProperties": True
                },
                outputs_schema={
                    "type": "object",
                    "properties": {
                        "result": {"type": "object"},
                        "summary": {"type": "string"}
                    },
                    "required": ["result"],
                    "additionalProperties": True
                },
                verification="result != null",
                tool_calls=[],
                fallbacks=[],
                version="1"
            )
        }


class AnalyzeDecideActTemplate(BlueprintTemplate):
    """Template for Analyze -> Decide -> Act pattern."""
    
    def __init__(self):
        super().__init__(
            name="analyze_decide_act",
            description="Generic pattern for analyzing data, making decisions, and taking action"
        )
    
    def create_dag(self) -> DAG:
        return DAG(
            nodes=[
                Node(
                    idx="0",
                    title="Gather Information",
                    description="Collect relevant data from available sources",
                    operator="retrieve"
                ),
                Node(
                    idx="1",
                    title="Analyze Data",
                    description="Extract insights and patterns from gathered information",
                    operator="extract"
                ),
                Node(
                    idx="2",
                    title="Make Decision",
                    description="Evaluate options and select best course of action",
                    operator="decide"
                ),
                Node(
                    idx="3",
                    title="Execute Action",
                    description="Implement the selected decision",
                    operator="act"
                )
            ],
            edges=[
                Edge(from_idx="0", to_idx="1"),
                Edge(from_idx="1", to_idx="2"),
                Edge(from_idx="2", to_idx="3")
            ]
        )
    
    def create_contracts(self) -> Dict[str, Contract]:
        # Similar structure to SearchFilterAct but with decision node
        return {
            "0": Contract(
                id="0",
                inputs_schema={
                    "type": "object",
                    "properties": {
                        "context": {"type": "object"},
                        "data_sources": {"type": "array", "items": {"type": "string"}},
                        "requirements": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["context"],
                    "additionalProperties": True
                },
                outputs_schema={
                    "type": "object",
                    "properties": {
                        "raw_data": {"type": "array", "items": {"type": "object"}},
                        "source_metadata": {"type": "object"}
                    },
                    "required": ["raw_data"],
                    "additionalProperties": True
                },
                verification="len(raw_data) >= 1",
                tool_calls=[{"name": "data_gather", "args": {}}],
                fallbacks=["retry_with_backoff"],
                version="1"
            ),
            "1": Contract(
                id="1",
                inputs_schema={
                    "type": "object",
                    "properties": {
                        "raw_data": {"type": "array", "items": {"type": "object"}},
                        "analysis_criteria": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["raw_data"],
                    "additionalProperties": True
                },
                outputs_schema={
                    "type": "object",
                    "properties": {
                        "insights": {"type": "array", "items": {"type": "object"}},
                        "metrics": {"type": "object"}
                    },
                    "required": ["insights"],
                    "additionalProperties": True
                },
                verification="len(insights) >= 1",
                tool_calls=[],
                fallbacks=[],
                version="1"
            ),
            "2": Contract(
                id="2",
                inputs_schema={
                    "type": "object",
                    "properties": {
                        "insights": {"type": "array", "items": {"type": "object"}},
                        "decision_criteria": {"type": "array", "items": {"type": "string"}},
                        "constraints": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["insights"],
                    "additionalProperties": True
                },
                outputs_schema={
                    "type": "object",
                    "properties": {
                        "decision": {"type": "object"},
                        "rationale": {"type": "string"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["decision"],
                    "additionalProperties": True
                },
                verification="decision != null && confidence >= 0.7",
                tool_calls=[],
                fallbacks=["human_review"],
                version="1"
            ),
            "3": Contract(
                id="3",
                inputs_schema={
                    "type": "object",
                    "properties": {
                        "decision": {"type": "object"},
                        "execution_params": {"type": "object"}
                    },
                    "required": ["decision"],
                    "additionalProperties": True
                },
                outputs_schema={
                    "type": "object",
                    "properties": {
                        "result": {"type": "object"},
                        "status": {"type": "string"}
                    },
                    "required": ["result", "status"],
                    "additionalProperties": True
                },
                verification="status == 'success'",
                tool_calls=[{"name": "executor", "args": {}}],
                fallbacks=["rollback"],
                version="1"
            )
        }


class ParallelAggregateTemplate(BlueprintTemplate):
    """Template for parallel processing and aggregation pattern."""
    
    def __init__(self):
        super().__init__(
            name="parallel_aggregate",
            description="Generic pattern for parallel data processing and aggregation"
        )
    
    def create_dag(self) -> DAG:
        return DAG(
            nodes=[
                Node(
                    idx="0",
                    title="Initialize Sources",
                    description="Identify and prepare multiple data sources for parallel processing",
                    operator="retrieve"
                ),
                Node(
                    idx="1",
                    title="Process Source A",
                    description="Process first data source",
                    operator="extract"
                ),
                Node(
                    idx="2",
                    title="Process Source B",
                    description="Process second data source",
                    operator="extract"
                ),
                Node(
                    idx="3",
                    title="Aggregate Results",
                    description="Combine results from parallel processing",
                    operator="extract"
                ),
                Node(
                    idx="4",
                    title="Generate Output",
                    description="Create final deliverable from aggregated data",
                    operator="act"
                )
            ],
            edges=[
                Edge(from_idx="0", to_idx="1"),
                Edge(from_idx="0", to_idx="2"),
                Edge(from_idx="1", to_idx="3"),
                Edge(from_idx="2", to_idx="3"),
                Edge(from_idx="3", to_idx="4")
            ]
        )
    
    def create_contracts(self) -> Dict[str, Contract]:
        # Implementation similar to above templates
        return {}


class BlueprintLibrary:
    """Library of reusable blueprint templates."""
    
    def __init__(self):
        self.templates = {
            "search_filter_act": SearchFilterActTemplate(),
            "analyze_decide_act": AnalyzeDecideActTemplate(),
            "parallel_aggregate": ParallelAggregateTemplate()
        }
    
    def get_template(self, name: str) -> Optional[BlueprintTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def list_templates(self) -> List[Dict[str, str]]:
        """List all available templates."""
        return [
            {"name": name, "description": template.description}
            for name, template in self.templates.items()
        ]
    
    def create_custom_blueprint(self, template_name: str, customizations: Dict) -> Tuple[DAG, Dict[str, Contract]]:
        """Create a customized blueprint from a template."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        return template.customize(customizations)


# Blueprint composition utilities
class BlueprintComposer:
    """Compose complex blueprints from simpler ones."""
    
    @staticmethod
    def compose(blueprints: List[Tuple[DAG, Dict[str, Contract]]]) -> Tuple[DAG, Dict[str, Contract]]:
        """Compose multiple blueprints into a single blueprint."""
        composed_nodes = []
        composed_edges = []
        composed_contracts = {}
        
        node_offset = 0
        
        for dag, contracts in blueprints:
            # Adjust node indices
            for node in dag.nodes:
                new_node = Node(
                    idx=str(int(node.idx) + node_offset),
                    title=node.title,
                    description=node.description,
                    operator=node.operator
                )
                composed_nodes.append(new_node)
            
            # Adjust edge indices
            for edge in dag.edges:
                new_edge = Edge(
                    from_idx=str(int(edge.from_idx) + node_offset),
                    to_idx=str(int(edge.to_idx) + node_offset)
                )
                composed_edges.append(new_edge)
            
            # Adjust contract indices
            for idx, contract in contracts.items():
                new_idx = str(int(idx) + node_offset)
                new_contract = Contract(
                    id=new_idx,
                    inputs_schema=contract.inputs_schema,
                    outputs_schema=contract.outputs_schema,
                    verification=contract.verification,
                    tool_calls=contract.tool_calls,
                    fallbacks=contract.fallbacks,
                    version=contract.version
                )
                composed_contracts[new_idx] = new_contract
            
            node_offset += len(dag.nodes)
        
        # Connect blueprints (last node of previous to first node of next)
        for i in range(len(blueprints) - 1):
            prev_dag = blueprints[i][0]
            next_dag = blueprints[i + 1][0]
            
            # Find last node of previous blueprint
            prev_last_idx = str(len(composed_nodes) - len(next_dag.nodes) - 1)
            next_first_idx = str(len(composed_nodes) - len(next_dag.nodes))
            
            composed_edges.append(Edge(from_idx=prev_last_idx, to_idx=next_first_idx))
        
        return DAG(nodes=composed_nodes, edges=composed_edges), composed_contracts