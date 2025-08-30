"""
Deep Research Agent for YOPO System

A specialized agent designed for conducting comprehensive research tasks using LangGraph.
This agent implements a multi-stage research workflow with query optimization, search execution,
follow-up question generation, decision routing, and final synthesis.
"""

import json
import os
import asyncio
from typing import Dict, Any, Optional, List, TypedDict, Annotated

import yaml

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from src.core.operators.llm.llm_wrapped_agent import create_wrapped_agent
from src.core.operators.llm.llm_provider import llm_instance, LLMProvider
from src.mcp.mcp_client import MCPZoo
from src.tool.tool_manager import get_tools
from src.core.operators.base_op import BaseOperator, OperatorResult
from src.core.operators.utils import json_format_parser

class ResearchState(TypedDict):
    """State object for the Deep Research Agent workflow"""
    # Input
    original_query: str
    context: Optional[Dict[str, Any]]
    
    # Query optimization
    optimized_query: str
    key_concepts: List[str]
    
    # Search results
    search_results: List[Dict[str, Any]]
    all_findings: List[str]
    
    # Follow-up research
    followup_questions: List[str]
    research_iteration: int
    max_iterations: int
    
    # Decision making
    should_continue: bool
    research_completeness: float
    
    # Final output
    final_synthesis: str
    research_metadata: Dict[str, Any]
    
    # Messages for LLM interaction
    messages: Annotated[List[BaseMessage], add_messages]


# Node Functions - Independent of the agent class

async def optimize_query_node(state: ResearchState, prompts: Dict[str, str]) -> ResearchState:
    """Node 1: Optimize the original query for better search results."""
    prompt = prompts.get('query_optimizer_prompt', '')
    llm_provider: LLMProvider = llm_instance(
        model_name=os.getenv("DEV_MODEL_NAME"),
        system_prompt=prompt
    )
    original_query = state["original_query"]
    context: Optional[Dict[str, Any]] = state["context"]
    user_content = f"Query: {original_query}" if context is None else f"Context: {json.dumps(context, indent=2)}\n\n Query: {original_query}"

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content}
    ]
    
    response: str = llm_provider.generate(messages)
    
    if "```json" in response or "```" in response:
        response: str = json_format_parser(text=response)

    try:
        # Parse JSON response
        result = json.loads(response)
        state["optimized_query"] = result.get("optimized_query", state["original_query"])
        state["key_concepts"] = result.get("key_concepts", [])
    except json.JSONDecodeError:

        # Fallback if JSON parsing fails
        state["optimized_query"] = state["original_query"]
        state["key_concepts"] = []
    
    state["messages"] = messages + [AIMessage(content=response)]
    return state

async def execute_search_node(state: ResearchState, prompts: Dict[str, str]) -> ResearchState:
    """Node 2: Execute searches using fire_crawl and gather information."""
    prompt = prompts.get('search_executor_prompt', '')
    
    # Get authorized MCP tools for searching
    try:
        # tools = await mcp_zoo.get_tools(["fire_crawl"])
        tools = get_tools(tool_base_names=["firecrawl_search"])
    except Exception as e:
        print(f"Warning: Could not get MCP tools: {e}")
        tools = []
    
    llm_provider: LLMProvider = llm_instance(model_name=os.getenv("DEV_MODEL_NAME"), system_prompt=prompt)
    
    # Create MCP agent for search execution
    agent = create_wrapped_agent(
        agent_type="react",
        llm_instance=llm_provider,
        tools=tools
    )
    
    
    async def execute_single_search(query: str) -> Dict[str, Any]:
        """Execute a single search query"""
        if not query:
            return {
                "query_used": query,
                "error": "Empty query"
            }
        
        try:
            # Use MCP agent to execute search
            search_context = {
                "task": "search_and_extract",
                "query": query,
            }
            
            result = await agent.ainvoke(
                f"Search for: {query}",
                context=search_context
            )
            
            return {
                "query_used": query,
                "result": result["messages"][-1].content,
                "sources_found": [],  # Extract from result if possible
                "key_findings": [],   # Extract from result if possible
                "source_quality": "medium"  # Default assessment
            }
            
        except Exception as e:
            print(f"Search failed for query '{query}': {e}")
            return {
                "query_used": query,
                "error": str(e)
            }
    
    # Execute all search queries in parallel
    query_list: List[str] = []
    if state["followup_questions"] == []:
        optimized_query: str = state.get("optimized_query", state["original_query"])
        query_list.append(optimized_query)
    else:
        query_list = state["followup_questions"]
    
    print(query_list)
    if query_list:        
        # Use asyncio.gather to execute all searches in parallel
        # Since optimized_query is now a single string, create a single query info dict
        search_results = await asyncio.gather(
            *[execute_single_search(query_info) for query_info in query_list],
            return_exceptions=True  # Continue execution even if some searches fail
        )
        
        # Handle potential exception results
        processed_results = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                # If it's an exception, create an error result
                processed_results.append({
                    "query_used": result["query_used"],
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        search_results = processed_results
    else:
        search_results = []
        print("No optimized queries to execute")
    
    print(processed_results)

    # Update state with search results
    current_results = state.get("search_results", [])
    current_results.extend(search_results)
    state["search_results"] = current_results
    
    # Extract all findings
    all_findings = state.get("all_findings", [])
    successful_searches = 0
    
    for result in search_results:
        if "result" in result and "error" not in result:
            all_findings.append(str(result["result"]))
            successful_searches += 1
    
    state["all_findings"] = all_findings
    
    print(f"Search completed: {successful_searches}/{len(search_results)} successful")
    
    return state

async def generate_followup_node(state: ResearchState, prompts: Dict[str, str]) -> ResearchState:
    """Node 3: Generate follow-up questions based on current findings."""
    prompt = prompts.get('followup_generator_prompt', '')
    llm_provider: LLMProvider = llm_instance(model_name=os.getenv("DEV_MODEL_NAME"), system_prompt=prompt)
    
    # Prepare context with current findings
    current_findings = "\n".join(state.get("all_findings", []))
    research_progress = f"Research Iteration: {state.get('research_iteration', 1)}"
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"""
        Current Research Progress:
        {research_progress}

        Original Query: {state['original_query']}

        Current Findings:
        {current_findings}

        Please analyze the current research state and generate follow-up questions if needed.
        """}
    ]
    
    response = llm_provider.generate(messages)
    
    if "```json" in response or "```" in response:
        response = json_format_parser(text=response)
    
    try:
        # Parse JSON response
        result = json.loads(response)
        state["followup_questions"] = result.get("followup_questions", [])
        state["should_continue"] = result.get("should_continue", False)
    except json.JSONDecodeError:
        # Fallback
        state["followup_questions"] = []
        state["should_continue"] = False
    
    return state

async def route_decision_node(state: ResearchState, prompts: Dict[str, str], max_iterations: int) -> ResearchState:
    """Node 4: Decide whether to continue research or proceed to synthesis."""
    prompt = prompts.get('decision_router_prompt', '')
    llm_provider: LLMProvider = llm_instance(model_name=os.getenv("DEV_MODEL_NAME"), system_prompt=prompt)
    
    current_iteration = state.get("research_iteration", 1)
    
    # Prepare decision context
    context = f"""
    Research Status:
    - Current iteration: {current_iteration}/{max_iterations}
    - Total findings collected: {len(state.get('all_findings', []))}
    - Follow-up questions generated: {len(state.get('followup_questions', []))}
    - Original query: {state['original_query']}

    Current Research State:
    {json.dumps(state.get('followup_questions', []), indent=2)}
    """
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context}
    ]
    
    response = llm_provider.generate(messages)
    
    if "```json" in response or "```" in response:
        response = json_format_parser(text=response)
    
    try:
        result = json.loads(response)
        state["should_continue"] = result.get("decision") == "continue_research"
        state["research_completeness"] = result.get("research_completeness", 50.0)
    except json.JSONDecodeError:
        # Default decision logic
        state["should_continue"] = (
            current_iteration < max_iterations and 
            len(state.get("followup_questions", [])) > 0
        )
        state["research_completeness"] = min(current_iteration * 30, 90)
    
    # Increment iteration counter
    state["research_iteration"] = current_iteration + 1
    
    return state

async def generate_synthesis_node(state: ResearchState, prompts: Dict[str, str]) -> ResearchState:
    """Node 5: Generate final comprehensive research synthesis."""
    prompt = prompts.get('synthesis_generator_prompt', '')
    llm_provider: LLMProvider = llm_instance(model_name=os.getenv("DEV_MODEL_NAME"), system_prompt=prompt)
    
    # Prepare all research data for synthesis
    synthesis_context = f"""
    Original Query: {state['original_query']}

    Research Summary:
    - Total iterations completed: {state.get('research_iteration', 1) - 1}
    - Total findings: {len(state.get('all_findings', []))}
    - Research completeness: {state.get('research_completeness', 0)}%

    All Research Findings:
    {chr(10).join(state.get('all_findings', []))}

    Key Concepts Explored:
    {', '.join(state.get('key_concepts', []))}
    """
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": synthesis_context}
    ]
    
    final_synthesis = llm_provider.generate(messages)
    
    state["final_synthesis"] = final_synthesis
    state["research_metadata"] = {
        "total_iterations": state.get("research_iteration", 1) - 1,
        "total_findings": len(state.get("all_findings", [])),
        "research_completeness": state.get("research_completeness", 0),
        "queries_used": len(state.get("optimized_query", [])),
        "key_concepts": state.get("key_concepts", [])
    }
    
    return state

def should_continue_research(state: ResearchState) -> str:
    """Conditional edge function to determine next step."""
    return "continue" if state.get("should_continue", False) else "synthesize"


class DeepResearchOperator(BaseOperator):
    """
    Specialized agent for conducting comprehensive research tasks using LangGraph workflow.
    
    This agent implements a systematic research process:
    1. Query Optimization - Refine and optimize the original query
    2. Search Execution - Execute searches using fire_crawl
    3. Follow-up Generation - Generate targeted follow-up questions
    4. Decision Routing - Decide whether to continue or synthesize
    5. Final Synthesis - Create comprehensive research report
    """
    
    def __init__(self, 
                 prompt_path: str,
                 max_iterations: int = 3,
                 **kwargs):
        """
        Initialize the Deep Research Agent.
        
        Args:
            prompt_path: Path to YAML config file with all node prompts
            mcps: List of MCP tools to use for research (default: ["fire_crawl"])
            user_id: User ID for MCP authorization
            max_iterations: Maximum number of research iterations
            **kwargs: Additional parameters for LLM
        """
        # Load all prompts from YAML
        with open(prompt_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            system_prompt = config.get('system_prompt', None)
            description = config.get('description', None)
        self.prompts = config
        # Initialize MCP Zoo for tool access
        # self.user_id = "default"
        # self.mcp_zoo = MCPZoo(user_id=self.user_id, mcp_names=mcps)
        self.max_iterations = max_iterations
        
        # Initialize base agent
        name = "DeepResearchOperator"
        
        super().__init__(name=name, description=description, system_prompt=system_prompt, **kwargs)

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for deep research."""
        workflow = StateGraph(ResearchState)
        
        # Create wrapper functions that pass the required parameters
        async def _optimize_query_wrapper(state: ResearchState) -> ResearchState:
            return await optimize_query_node(state, self.prompts)
        
        async def _execute_search_wrapper(state: ResearchState) -> ResearchState:
            return await execute_search_node(state, self.prompts)
        
        async def _generate_followup_wrapper(state: ResearchState) -> ResearchState:
            return await generate_followup_node(state, self.prompts)
        
        # async def _route_decision_wrapper(state: ResearchState) -> ResearchState:
        #     return await route_decision_node(state, self.prompts, self.max_iterations)
        
        async def _generate_synthesis_wrapper(state: ResearchState) -> ResearchState:
            return await generate_synthesis_node(state, self.prompts)
        
        # Add all nodes with wrapper functions
        workflow.add_node("query_optimizer", _optimize_query_wrapper)
        workflow.add_node("search_executor", _execute_search_wrapper)
        workflow.add_node("followup_generator", _generate_followup_wrapper)
        # workflow.add_node("decision_router", _route_decision_wrapper)
        workflow.add_node("synthesis_generator", _generate_synthesis_wrapper)
        
        # Define the workflow edges
        workflow.add_edge(START, "query_optimizer")
        workflow.add_edge("query_optimizer", "search_executor")
        workflow.add_edge("search_executor", "followup_generator")
        # workflow.add_edge("followup_generator", "decision_router")
        
        # Conditional routing based on decision
        workflow.add_conditional_edges(
            "followup_generator",
            should_continue_research,
            {
                "continue": "search_executor",  # Loop back for more research
                "synthesize": "synthesis_generator"  # Proceed to final synthesis
            }
        )
        
        workflow.add_edge("synthesis_generator", END)
        
        return workflow.compile()

    async def arun(self, query: str, context: Optional[Dict[str, Any]] = None) -> OperatorResult:
        """
        Run the Deep Research Agent with the given research query.
        
        This method initiates a multi-stage research workflow including query optimization,
        search execution, follow-up question generation, decision routing, and final synthesis.
        
        Args:
            query: Research query or topic to investigate
            context: Additional context information, may contain:
                - user_id: User identifier
                - max_iterations: Maximum number of research iterations
                - other configuration parameters
        
        Returns:
            OperatorResult: Research results containing:
                - result: Final research synthesis report
                - metadata: Research process metadata including:
                    - research_process: Research process details
                    - optimized_query: Optimized query
                    - key_concepts: Key concepts identified
                    - total_findings: Total number of findings
                    - iterations_completed: Number of iterations completed
        """
        workflow = self._build_workflow()
        # Initialize research state
        initial_state = ResearchState(
            original_query=query,
            context=context,
            optimized_query="",
            key_concepts=[],
            search_results=[],
            all_findings=[],
            followup_questions=[],
            research_iteration=1,
            max_iterations=self.max_iterations,
            should_continue=True,
            research_completeness=0.0,
            final_synthesis="",
            research_metadata={},
            messages=[]
        )
        
        try:
            # Execute the research workflow
            final_state = await workflow.ainvoke(initial_state)
            
            return OperatorResult(
                base_result=final_state["final_synthesis"],
                metadata={
                    **final_state["research_metadata"],
                    "research_process": {
                        "optimized_query": final_state["optimized_query"],
                        "key_concepts": final_state["key_concepts"],
                        "total_findings": len(final_state["all_findings"]),
                        "iterations_completed": final_state["research_iteration"] - 1
                    }
                }
            )
            
        except Exception as e:
            return OperatorResult(
                base_result=f"Research failed: {str(e)}",
                metadata={"error": True, "error_message": str(e), "research_process": {}}
            )

        
