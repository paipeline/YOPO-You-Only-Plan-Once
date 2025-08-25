"""
Abstraction Layer - Domain-specific customization and blueprint adaptation
"""
from typing import Dict, List, Any, Optional, Tuple
from src.core.planning.models import CanonicalQuery, DAG, Node, Edge, Contract
from src.core.planning.blueprint_templates import BlueprintLibrary
import re


class DomainAdapter:
    """Adapts abstract blueprints to specific domains."""
    
    def __init__(self, domain: str):
        self.domain = domain
        self.domain_mappings = self._load_domain_mappings()
    
    def _load_domain_mappings(self) -> Dict[str, Dict]:
        """Load domain-specific mappings for tools, verifications, etc."""
        # This could be loaded from config files or database
        mappings = {
            "ecommerce": {
                "search": "product_catalog_api",
                "filter_fields": ["price", "category", "brand", "availability"],
                "verification_patterns": {
                    "price_check": "price <= {budget}",
                    "availability": "in_stock == true"
                }
            },
            "research": {
                "search": "academic_search_api",
                "filter_fields": ["publication_date", "journal", "citations", "relevance"],
                "verification_patterns": {
                    "recency": "publication_date >= {start_date}",
                    "quality": "citations >= {min_citations}"
                }
            },
            "data_analysis": {
                "search": "data_warehouse_query",
                "filter_fields": ["timestamp", "metric", "dimension", "aggregation"],
                "verification_patterns": {
                    "completeness": "data_points >= {min_samples}",
                    "validity": "null_rate < {max_null_rate}"
                }
            }
        }
        return mappings.get(self.domain, {})
    
    def adapt_contract(self, contract: Contract, node_context: Dict) -> Contract:
        """Adapt a generic contract to domain-specific requirements."""
        adapted_contract = contract.model_copy()
        
        # Adapt tool calls
        if contract.tool_calls:
            adapted_contract.tool_calls = self._adapt_tool_calls(contract.tool_calls)
        
        # Adapt verification
        if contract.verification:
            adapted_contract.verification = self._adapt_verification(contract.verification, node_context)
        
        # Adapt schemas based on domain
        adapted_contract.inputs_schema = self._enrich_schema(contract.inputs_schema, "input")
        adapted_contract.outputs_schema = self._enrich_schema(contract.outputs_schema, "output")
        
        return adapted_contract
    
    def _adapt_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Map generic tool calls to domain-specific implementations."""
        adapted_calls = []
        for call in tool_calls:
            if call["name"] == "search" and "search" in self.domain_mappings:
                adapted_calls.append({
                    "name": self.domain_mappings["search"],
                    "args": {**call.get("args", {}), "domain": self.domain}
                })
            else:
                adapted_calls.append(call)
        return adapted_calls
    
    def _adapt_verification(self, verification: str, context: Dict) -> str:
        """Adapt generic verification to domain-specific checks."""
        # Replace generic patterns with domain-specific ones
        for pattern_name, pattern_template in self.domain_mappings.get("verification_patterns", {}).items():
            if pattern_name in context:
                verification = pattern_template.format(**context)
        return verification
    
    def _enrich_schema(self, schema: Dict, schema_type: str) -> Dict:
        """Enrich generic schema with domain-specific fields."""
        if not schema or "properties" not in schema:
            return schema
        
        enriched_schema = schema.copy()
        
        # Add domain-specific fields
        if schema_type == "input" and "filter_fields" in self.domain_mappings:
            if "filters" in enriched_schema["properties"]:
                enriched_schema["properties"]["filters"]["properties"] = {
                    field: {"type": "string"} for field in self.domain_mappings["filter_fields"]
                }
        
        return enriched_schema


class AbstractionEngine:
    """Engine for creating abstract, reusable blueprints."""
    
    def __init__(self):
        self.library = BlueprintLibrary()
        self.pattern_detector = PatternDetector()
    
    def analyze_canonical_query(self, query: CanonicalQuery) -> Dict[str, Any]:
        """Analyze a canonical query to determine the best blueprint pattern."""
        analysis = {
            "detected_pattern": None,
            "confidence": 0.0,
            "domain_hints": [],
            "customization_params": {}
        }
        
        # Detect pattern based on goal and entities
        pattern = self.pattern_detector.detect_pattern(query)
        analysis["detected_pattern"] = pattern["name"]
        analysis["confidence"] = pattern["confidence"]
        
        # Extract domain hints from entities
        analysis["domain_hints"] = self._extract_domain_hints(query.entities)
        
        # Generate customization parameters
        analysis["customization_params"] = self._generate_customization_params(query)
        
        return analysis
    
    def _extract_domain_hints(self, entities: List[str]) -> List[str]:
        """Extract domain hints from entities."""
        domain_keywords = {
            "ecommerce": ["product", "price", "cart", "order", "customer", "shipping"],
            "research": ["paper", "study", "journal", "citation", "academic", "research"],
            "data_analysis": ["data", "metric", "analysis", "report", "dashboard", "visualization"]
        }
        
        hints = []
        for domain, keywords in domain_keywords.items():
            if any(keyword in entity.lower() for entity in entities for keyword in keywords):
                hints.append(domain)
        
        return hints
    
    def _generate_customization_params(self, query: CanonicalQuery) -> Dict:
        """Generate parameters for customizing blueprints."""
        params = {
            "constraints": query.constraints,
            "success_criteria": query.success_criteria,
            "entity_count": len(query.entities),
            "complexity_score": self._calculate_complexity(query)
        }
        
        # Extract numeric constraints
        for constraint in query.constraints:
            matches = re.findall(r'(\w+)\s*[<>=]+\s*(\d+)', constraint)
            for key, value in matches:
                params[f"constraint_{key}"] = int(value)
        
        return params
    
    def _calculate_complexity(self, query: CanonicalQuery) -> float:
        """Calculate complexity score for the query."""
        # Simple heuristic based on number of entities, constraints, and criteria
        return (len(query.entities) * 0.3 + 
                len(query.constraints) * 0.4 + 
                len(query.success_criteria) * 0.3)


class PatternDetector:
    """Detects which blueprint pattern best matches a query."""
    
    def detect_pattern(self, query: CanonicalQuery) -> Dict[str, Any]:
        """Detect the most suitable pattern for a canonical query."""
        patterns = {
            "search_filter_act": self._score_search_filter_act(query),
            "analyze_decide_act": self._score_analyze_decide_act(query),
            "parallel_aggregate": self._score_parallel_aggregate(query),
            "custom": 0.2  # Base score for custom pattern
        }
        
        # Find best matching pattern
        best_pattern = max(patterns.items(), key=lambda x: x[1])
        
        return {
            "name": best_pattern[0],
            "confidence": best_pattern[1],
            "scores": patterns
        }
    
    def _score_search_filter_act(self, query: CanonicalQuery) -> float:
        """Score how well the query matches search-filter-act pattern."""
        score = 0.0
        
        # Check for search-related keywords in goal
        search_keywords = ["find", "search", "discover", "identify", "locate", "get"]
        if any(keyword in query.goal.lower() for keyword in search_keywords):
            score += 0.4
        
        # Check for filtering constraints
        filter_keywords = ["under", "above", "between", "only", "must", "should"]
        if any(keyword in constraint.lower() for constraint in query.constraints for keyword in filter_keywords):
            score += 0.3
        
        # Check for action-oriented success criteria
        if any("return" in criterion.lower() or "provide" in criterion.lower() for criterion in query.success_criteria):
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_analyze_decide_act(self, query: CanonicalQuery) -> float:
        """Score how well the query matches analyze-decide-act pattern."""
        score = 0.0
        
        # Check for analysis keywords
        analysis_keywords = ["analyze", "evaluate", "assess", "compare", "study"]
        if any(keyword in query.goal.lower() for keyword in analysis_keywords):
            score += 0.4
        
        # Check for decision-related entities
        decision_keywords = ["option", "choice", "alternative", "recommendation", "decision"]
        if any(keyword in entity.lower() for entity in query.entities for keyword in decision_keywords):
            score += 0.3
        
        # Check for decision criteria
        if len(query.success_criteria) > 2:  # Multiple criteria suggest decision-making
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_parallel_aggregate(self, query: CanonicalQuery) -> float:
        """Score how well the query matches parallel-aggregate pattern."""
        score = 0.0
        
        # Check for aggregation keywords
        aggregate_keywords = ["combine", "aggregate", "merge", "consolidate", "summarize", "multiple"]
        if any(keyword in query.goal.lower() for keyword in aggregate_keywords):
            score += 0.4
        
        # Check for multiple sources in entities
        source_keywords = ["sources", "channels", "systems", "databases"]
        if any(keyword in entity.lower() for entity in query.entities for keyword in source_keywords):
            score += 0.4
        
        # Check for parallel processing hints
        if "parallel" in query.goal.lower() or len(query.entities) > 5:
            score += 0.2
        
        return min(score, 1.0)


def create_abstract_blueprint(canonical_query: CanonicalQuery, 
                            domain: Optional[str] = None) -> Tuple[DAG, Dict[str, Contract]]:
    """Create an abstract, reusable blueprint from a canonical query."""
    engine = AbstractionEngine()
    
    # Analyze the query
    analysis = engine.analyze_canonical_query(canonical_query)
    
    # Get the appropriate template
    template = engine.library.get_template(analysis["detected_pattern"])
    if not template:
        # Fall back to building custom blueprint
        from src.core.planning.pipeline import build_dag_and_contracts
        return build_dag_and_contracts(canonical_query, template_hint="custom")
    
    # Create base blueprint from template
    dag, contracts = template.customize(analysis["customization_params"])
    
    # Apply domain-specific adaptations if domain is specified
    if domain or analysis["domain_hints"]:
        domain = domain or analysis["domain_hints"][0]
        adapter = DomainAdapter(domain)
        
        # Adapt each contract
        adapted_contracts = {}
        for idx, contract in contracts.items():
            node_context = {
                "node_idx": idx,
                "constraints": canonical_query.constraints,
                "entities": canonical_query.entities
            }
            adapted_contracts[idx] = adapter.adapt_contract(contract, node_context)
        
        contracts = adapted_contracts
    
    return dag, contracts