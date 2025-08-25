"""
Plan Manager - Unified Entry Point
Purpose: Orchestrate plan lifecycle for a user request.
"""
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4
from src.core.planning.models import (
    CanonicalQuery, 
    PlanBlueprint, 
    PlanPayload, 
    ExecutableView,
    DAG,
    Contract
)


class PlanManager:
    """
    Orchestrates plan lifecycle for user requests.
    
    Key Conditions & Policies:
    - Reuse-first: get_plan_from_chroma precedes build_plan
    - Determinism: Plan chosen with deterministic tie-breaker
    - Validation gate: Before run, ensure DAG validity
    - Escalation ladder: retry tool � fallback � local replan � subgraph rebuild � full rebuild
    - Telemetry: emit counters for reuse hits, builds, replans, failures
    """
    
    def __init__(self, 
                abstractor=None,
                registry=None, 
                chroma_client=None, 
                n_determine=3, 
                n_build=3, 
                m_retry=3):
                
        """Initialize PlanManager with dependencies."""
        self.abstractor = abstractor
        self.registry = registry
        self.chroma_client = chroma_client
        self.n_determine = n_determine
        self.n_build = n_build
        self.m_retry = m_retry
        self.context = {}
        self.retry_counts = {}
    
    
    
    def get_plan_from_chroma(self, task_signature: Dict[str, Any]) -> Optional[PlanBlueprint]:
        """
        Query memory by (task signature, tags).
        
        Condition: retry up to N_determine times if LLM-based disambiguation required.
        On fail: return None to trigger build path.
        
        Args:
            task_signature: Task signature dict
            
        Returns:
            PlanBlueprint if found, None otherwise
        """

        # get plan record from chroma by task signature and tags
        # TEMP: return None

        # TODO: Implement Chroma query with retry logic
        # 1. Query Chroma by task signature and tags
        # 2. If ambiguous, use LLM for disambiguation (up to n_determine times)
        # 3. Apply deterministic tie-breaker (latest version, highest reuse score)
        # 4. Return best match or None
        _ = task_signature  # Will be used in implementation
        return None
    
    def build_plan(self, query: str, entities: List[str]) -> PlanBlueprint:
        """
        Call abstractor to produce (DAG, contracts) from trace or template.
        
        Condition: if N_build consecutive failures � escalate (surface error).
        
        Args:
            trace: Optional execution trace for abstraction
            query: User query
            entities: Entities relevant to the query
            
        Returns:
            PlanBlueprint with DAG and contracts
            
        Raises:
            Exception: If N_build consecutive failures
        """
        # TODO: Implement plan building with abstractor
        # 1. Call abstractor with trace or default template
        # 2. Track build failures
        # 3. If exceeded n_build failures, escalate error
        # 4. Return constructed PlanBlueprint
        _ = (query, entities)  # Will be used when calling abstractor
        
        # init plan
        plan = PlanBlueprint(
            plan_id=str("plan_") + str(uuid4()),
            dag=DAG(nodes=[], edges=[]),
            contracts={},
        )

        # Import here to avoid circular imports
        from src.core.planning.pipeline import build_dag_and_contracts, build_canonical_query
        dag, contracts = build_dag_and_contracts(build_canonical_query(query, entities))
        plan.dag = dag
        plan.contracts = contracts
        
        # save temp plan to tmp file
        self.save_plan_to_file(plan)

        return plan
    
    def save_plan_to_file(self, plan: PlanBlueprint):
        """
        Save plan to tmp file.
        """
        # TODO: Implement saving to tmp file
        _ = plan  # Will be used in implementation
        pass

    def save_plan_to_chroma(self, plan: PlanBlueprint):
        """
        Persist to Chroma (semantic index + tags) and file-system fallback.
        
        Args:
            plan: PlanBlueprint to save
        """
        # TODO: Implement plan persistence
        # 1. Save to Chroma with semantic indexing
        # 2. Save to file-system via registry as fallback
        _ = plan  # Will be used in implementation
        pass
    
    def run(self, plan: PlanBlueprint, params: Dict[str, Any]) -> ExecutableView:
        """
        Inject request params into node descriptions, then hand off to executor.
        
        Condition: validate plan completeness (every node has contract).
        
        Args:
            plan: PlanBlueprint to execute
            params: Parameters to inject
            
        Returns:
            ExecutableView for execution
            
        Raises:
            ValidationError: If plan incomplete
        """
        # TODO: Implement plan execution
        # 1. Validate plan completeness:
        #    - len(dag.nodes) > 0
        #    - every node.idx in contracts
        #    - schemas present
        # 2. Inject params into node descriptions
        # 3. Create and return ExecutableView
        _ = (plan, params)  # Will be used in implementation
        return ExecutableView()
    
    def replan(self, plan: PlanBlueprint, failed_idx: str, context: Dict) -> PlanBlueprint:
        """
        Local repair for failures; prefer updating only contract/tooling/fallbacks.
        Write to tmp/workflow json file.
        
        Condition: if repeated failures for same node exceed M � escalate to 
        abstractor for subgraph rebuild.
        
        Args:
            plan: Current plan
            failed_idx: Index of failed node
            context: Execution context with failure info
            
        Returns:
            Updated PlanBlueprint
        """
        # TODO: Implement replan logic as specified
        node_key = f"{plan.plan_id}:{failed_idx}"
        self.retry_counts[node_key] = self.retry_counts.get(node_key, 0) + 1
        
        if (self.retry_counts[node_key] <= self.m_retry and 
            self._has_new_fallback(plan, failed_idx)):
            # Local tactical repair
            self._mutate_contract(plan, failed_idx, context)
            plan.version = str(int(plan.version) + 1)
        else:
            # Structural repair: call abstractor for subgraph rebuild
            sub_trace = self._get_recent_trace(context, failed_idx)
            sub_dag, sub_contracts = self.abstractor.reabstract_subgraph(
                sub_trace, 
                goal=self._get_node_goal(plan, failed_idx)
            )
            plan = self._splice(plan, failed_idx, sub_dag, sub_contracts)
            plan.version = str(int(plan.version) + 1)
        
        # Write to tmp/workflow json file
        self._save_to_tmp_workflow(plan)
        
        return plan
    
    def _has_new_fallback(self, plan: PlanBlueprint, failed_idx: str) -> bool:
        """Check if node has available fallbacks."""
        # TODO: Implement fallback availability check
        _ = (plan, failed_idx)  # Will be used in implementation
        return False
    
    def _mutate_contract(self, plan: PlanBlueprint, failed_idx: str, context: Dict):
        """Mutate contract with new tools, params, or fallbacks."""
        # TODO: Implement contract mutation
        # - Add/rotate tool_calls
        # - Adjust params
        # - Add backoff
        _ = (plan, failed_idx, context)  # Will be used in implementation
        pass
    
    def _get_recent_trace(self, context: Dict, failed_idx: str, neighbors: int = 3):
        """Get recent execution trace around failed node."""
        # TODO: Implement trace extraction
        _ = (context, failed_idx, neighbors)  # Will be used in implementation
        return {}
    
    def _get_node_goal(self, plan: PlanBlueprint, node_idx: str) -> str:
        """Get goal description for a node."""
        # TODO: Extract node goal from plan
        _ = (plan, node_idx)  # Will be used in implementation
        return ""
    
    def _splice(self, plan: PlanBlueprint, failed_idx: str, 
                sub_dag: DAG, sub_contracts: Dict[str, Contract]) -> PlanBlueprint:
        """Splice subgraph into plan at failed node."""
        # TODO: Implement graph splicing
        _ = (failed_idx, sub_dag, sub_contracts)  # Will be used in implementation
        return plan
    
    def _save_to_tmp_workflow(self, plan: PlanBlueprint):
        """Save plan to tmp/workflow json file."""
        # TODO: Implement saving to tmp/workflow
        _ = plan  # Will be used in implementation
        pass


def create_plan_manager(abstractor=None, registry=None, chroma_client=None,
                       n_determine=3, n_build=3, m_retry=3) -> 'PlanManager':
    """
    Factory returning a configured PlanManager instance.
    
    Args:
        abstractor: Abstractor for plan building
        registry: Registry for file-system operations
        chroma_client: Chroma client for semantic search
        n_determine: Max retries for plan disambiguation
        n_build: Max consecutive build failures before escalation
        m_retry: Max retries per node before structural repair
        
    Returns:
        Configured PlanManager instance
    """
    return PlanManager(
        abstractor=abstractor,
        registry=registry,
        chroma_client=chroma_client,
        n_determine=n_determine,
        n_build=n_build,
        m_retry=m_retry
    )