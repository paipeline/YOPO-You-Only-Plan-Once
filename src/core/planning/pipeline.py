from typing import Dict, Tuple, List, Optional
from src.core.agents.llm.llm_provider import llm_instance
import os
from src.core.planning.models import CanonicalQuery, DAG, Contract
from src.core.planning.abstraction_layer import create_abstract_blueprint
import json
import logging

# Set up logging for pipeline
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 1. query -> canonical query
def build_canonical_query(query: str, tags: List[str]) -> CanonicalQuery:
    """Build canonical query from query and tags."""
    SYSTEM_PROMPT = """
You are the Canonicalizer. Convert a free-form user request into a structured CanonicalQuery.

INPUT (as user message JSON):
{
  "text": "raw user query",
  "tags": ["tag1", "tag2", "..."]
}

STRICT OUTPUT (valid JSON only, no markdown/comments; MUST validate this schema):
{
  "canonical_query": {
    "goal": "string",
    "entities": ["string", "..."],
    "constraints": ["string", "..."],
    "success_criteria": ["string", "..."]
  }
}

Definitions:
- goal: One concise sentence capturing the primary objective.
- entities: Salient objects/concepts/domains (3–12). Normalize and deduplicate; use stable, specific terms.
- constraints: Verifiable requirements (budgets, tools, modalities, deadlines, jurisdictions, privacy, quality thresholds).
  IMPORTANT: Every input tag MUST appear here exactly as `Include tag: <tag>`. If a tag is a domain/category, you may also include it in entities, but do not drop it.
- success_criteria: Observable, testable checks tied to the goal/constraints (e.g., "return ≥3 options", "price ≤ 30000", "accuracy ≥ 90%", "HTTP 200 and schema valid").

Rules:
- Deterministic wording: prefer specific terms; avoid synonym drift.
- Derive obvious implicit constraints when clearly implied (e.g., "EU user" ⇒ "GDPR-compliant").
- If the request is vague, keep entities minimal and add uncertainty notes under success_criteria (e.g., "clarify target audience").
- Do NOT invent external facts; only structure what is implied by the input.
- Output MUST be valid JSON and contain no extra fields.

Failure mode:
If the input is contradictory or not actionable, output:
{ "canonical_query": { "goal": "", "entities": [], "constraints": ["unactionable_input"], "success_criteria": [] } }

Few-shot examples:

Input:
{ "text": "Find me three beginner Python courses under $50.", "tags": ["education", "budget<50"] }
Output:
{
  "canonical_query": {
    "goal": "Identify affordable beginner-level Python courses.",
    "entities": ["Python", "beginner course", "pricing", "education"],
    "constraints": ["Price ≤ 50", "Include tag: education", "Return exactly 3 options"],
    "success_criteria": ["Return 3 course options", "Each option includes title, provider, price", "All prices ≤ 50"]
  }
}

Input:
{ "text": "Summarize the latest research on gut microbiome and antibiotics.", "tags": ["research", "microbiome"] }
Output:
{
  "canonical_query": {
    "goal": "Summarize recent research linking gut microbiome and antibiotics.",
    "entities": ["gut microbiome", "antibiotics", "antibiotic resistance", "dysbiosis", "systematic review"],
    "constraints": ["Include tag: research", "Include tag: microbiome", "Focus on recent literature"],
    "success_criteria": ["Produce a 5–8 bullet summary", "Cite at least 3 recent sources", "Highlight impacts on diversity and resistance"]
  }
}
"""
    model_name = os.getenv("DEV_LLM_NAME") or os.getenv("DEV_MODEL_NAME") or "gpt-3.5-turbo"
    
    # Format the input as expected by the prompt
    user_input = {
        "text": query,
        "tags": tags
    }
    
    response = llm_instance(model_name, system_prompt=SYSTEM_PROMPT).generate(
        [{"role": "user", "content": json.dumps(user_input)}]
    )
    
    try:
        result = json.loads(response)
        print(f"Canonical Query: {result}")
        return CanonicalQuery(**result["canonical_query"])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to parse canonical query: {e}")
        # Return a default canonical query on error
        return CanonicalQuery(
            goal=query,
            entities=[],
            constraints=tags,
            success_criteria=[]
        )


# 2. canonical query -> dag and contracts
def build_dag_and_contracts(canonical_query: CanonicalQuery, template_hint: Optional[str] = None) -> Tuple[DAG, Dict[str, Contract]]:
    """Build DAG and contracts from canonical query."""
    # init llm provider
    # build dag and contracts
    SYSTEM_PROMPT = """
You are the Plan Builder. Convert a CanonicalQuery into:
1) A Directed Acyclic Graph (DAG) of executable nodes.
2) A Contract for EVERY node (id == node.idx).

INPUT (as user message JSON):
{
  "canonical_query": {
    "goal": "string",
    "entities": ["string", "..."],
    "constraints": ["string", "..."],
    "success_criteria": ["string", "..."]
  }
}

STRICT OUTPUT (valid JSON only; no markdown/comments; exact shape):
{
  "dag": {
    "nodes": [
      { "idx": "0", "title": "…", "description": "…", "operator": "retrieve" }
    ],
    "edges": [
      { "from_idx": "0", "to_idx": "1" }
    ]
  },
  "contracts": {
    "0": {
      "id": "0",
      "inputs_schema": {},
      "outputs_schema": {},
      "verification": "…",
      "tool_calls": [],
      "fallbacks": [],
      "version": "1"
    }
  },
  "assumptions": ["..."],  // concise list of any necessary assumptions; may be []
  "pattern_used": "string"  // one of: search_filter_act, analyze_decide_act, parallel_aggregate, custom
}

Operators (choose exactly one per node):
- "retrieve": access external/internal data (APIs, DBs, files, web)
- "extract": parse, clean, transform, structure, feature-engineer
- "decide": rank, select, route, validate, threshold
- "act": produce final artifact or trigger side-effect (write file, call tool, post payload)
- "notify": report/summarize outcomes for user or downstream sink

Planning guidance (abstract & reusable):
- Select from these common patterns when applicable:
  * search_filter_act: For queries involving finding and filtering items
  * analyze_decide_act: For queries requiring analysis and decision-making
  * parallel_aggregate: For queries needing parallel processing
  * Custom: When no pattern fits, create a bespoke pipeline
- Prefer a compact pipeline (often retrieve → extract → act). Add "decide" if ranking/selection is required, and "notify" if a separate reporting step helps.
- Titles ≤ 8 words, action-oriented, domain-agnostic where possible ("Discover Candidates", "Structure Attributes", "Synthesize Deliverable"). Descriptions (1–2 sentences) may mention entities/constraints explicitly.
- Use indices "0","1","2",… with no gaps. Graph MUST be acyclic. Use only essential edges (maximize parallelism).
- Map CanonicalQuery fields:
  • goal → informs final "act"/"notify" node(s) and overall flow.  
  • entities → drive scope and naming in descriptions.  
  • constraints → HARD requirements; encode as verification checks and/or schema fields. Do NOT drop any.  
  • success_criteria → translate into verification on final nodes (and intermediate ones if applicable).

Contract requirements (for EVERY node; keys must be present):
- inputs_schema / outputs_schema: explicit JSON Schema fragments (types: string, number, boolean, array, object) with required keys.
  Suggested generic shapes (adapt as needed):
  inputs_schema:
  {
    "type": "object",
    "properties": {
      "context": { "type": "object" },
      "query": { "type": "string" },
      "constraints": { "type": "array", "items": { "type": "string" } },
      "required_fields": { "type": "array", "items": { "type": "string" } },
      "items": { "type": "array", "items": { "type": "object" } }
    },
    "required": ["context"],
    "additionalProperties": true
  }
  outputs_schema:
  {
    "type": "object",
    "properties": {
      "items": { "type": "array", "items": { "type": "object" } },
      "summary": { "type": "string" },
      "metadata": { "type": "object" }
    },
    "additionalProperties": true
  }

- verification: concrete, checkable postconditions tied to constraints/success_criteria. Examples:
  "len(items) ≥ 3", "every(items, has_fields([\"price\",\"provider\"]))", "every(items, includes_tags(input_tags))", "price ≤ budget_ceiling".
- tool_calls: list external tools/APIs by capability with simple args, e.g. { "name": "search", "args": {"mode": "broad"} }; else [].
- fallbacks: short strategies or alternate tools (e.g., "retry_with_backoff", "secondary_search"); else [].
- version always "1".

Determinism & abstraction:
- Avoid domain-specific nouns in titles; prefer "Candidates/Attributes/Deliverable". Descriptions can reference entities and constraints explicitly.
- Use consistent naming and stable node order (topological; break ties by idx).
- DO NOT invent external facts or constraints.

If the CanonicalQuery is contradictory or not actionable, output:
{ "dag": { "nodes": [], "edges": [] }, "contracts": {}, "assumptions": ["no_valid_plan"] }

Checklist before you output:
- [ ] All constraints represented (schema and/or verification)
- [ ] Node indices contiguous from "0"
- [ ] Acyclic graph with minimal edges (max parallelism)
- [ ] Contract per node (id == idx; includes all keys)
- [ ] Verification ties to constraints/success_criteria
- [ ] Valid JSON only
"""


    model_name = os.getenv("DEV_LLM_NAME") or os.getenv("DEV_MODEL_NAME") or "gpt-3.5-turbo"
    response = llm_instance(model_name, system_prompt=SYSTEM_PROMPT).generate(
        [{"role": "user", "content": json.dumps({"canonical_query": canonical_query.model_dump(), "template_hint": template_hint})}]
    )
    
    # Parse the JSON response
    try:
        result = json.loads(response)
        
        # Create DAG from the response
        dag_data = result.get("dag", {"nodes": [], "edges": []})
        dag = DAG(**dag_data)
        
        # Create contracts from the response
        contracts_data = result.get("contracts", {})
        contracts = {k: Contract(**v) for k, v in contracts_data.items()}
        
        return dag, contracts
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        print(f"Response: {response}")
        # Return empty DAG and contracts on error
        return DAG(nodes=[], edges=[]), {}

def build_abstract_dag_and_contracts(canonical_query: CanonicalQuery, 
                                   domain: Optional[str] = None) -> Tuple[DAG, Dict[str, Contract]]:
    """Build abstract, reusable DAG and contracts using templates and patterns.
    
    Args:
        canonical_query: The canonical query to build from
        domain: Optional domain hint (e.g., 'ecommerce', 'research', 'data_analysis')
        
    Returns:
        Tuple of (DAG, contracts) that are abstract and reusable
    """
    return create_abstract_blueprint(canonical_query, domain)


if __name__ == "__main__":
    # 1. query
    query = "I want to buy a new car"
    # 2. tags
    tags = ["car", "buy"]
    
    print("\n" + "="*50)
    print("Building Canonical Query...")
    print("="*50)
    
    # 3. canonical query
    canonical_query = build_canonical_query(query, tags)
    print(f"\nCanonical Query:")
    print(f"  Goal: {canonical_query.goal}")
    print(f"  Entities: {canonical_query.entities}")
    print(f"  Constraints: {canonical_query.constraints}")
    print(f"  Success Criteria: {canonical_query.success_criteria}")
    
    print("\n" + "="*50)
    print("Building DAG and Contracts...")
    print("="*50)
    
    # Method 1: Direct LLM-based generation
    print("\nMethod 1: Direct LLM Generation")
    dag, contracts = build_dag_and_contracts(canonical_query)
    
    # Pretty print DAG
    print("\nDAG Nodes:")
    for node in dag.nodes:
        print(f"  [{node.idx}] {node.title}")
        print(f"      Operator: {node.operator}")
        print(f"      Description: {node.description}")
    
    print("\nDAG Edges:")
    for edge in dag.edges:
        print(f"  {edge.from_idx} -> {edge.to_idx}")
    
    # Pretty print contracts summary
    print("\nContracts Summary:")
    for idx, contract in contracts.items():
        print(f"  Node {idx}:")
        print(f"    - Has input schema: {'Yes' if contract.inputs_schema else 'No'}")
        print(f"    - Has output schema: {'Yes' if contract.outputs_schema else 'No'}")
        print(f"    - Tool calls: {len(contract.tool_calls)}")
        print(f"    - Fallbacks: {len(contract.fallbacks)}")
        print(f"    - Verification: {contract.verification[:50]}..." if len(contract.verification) > 50 else f"    - Verification: {contract.verification}")
    
    # Method 2: Abstract template-based generation
    print("\n" + "="*50)
    print("Method 2: Abstract Template-Based Generation")
    print("="*50)
    
    # Detect domain from entities
    domain = "ecommerce" if "car" in canonical_query.entities else None
    abstract_dag, abstract_contracts = build_abstract_dag_and_contracts(canonical_query, domain)
    
    print("\nAbstract DAG Nodes:")
    for node in abstract_dag.nodes:
        print(f"  [{node.idx}] {node.title}")
        print(f"      Operator: {node.operator}")
        print(f"      Description: {node.description}")
    
    print("\nAbstract Contracts Summary:")
    for idx, contract in abstract_contracts.items():
        print(f"  Node {idx}:")
        print(f"    - Tool calls: {contract.tool_calls}")
        print(f"    - Verification: {contract.verification}")