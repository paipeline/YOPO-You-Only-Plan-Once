"""
General Agent for YOPO System

A general-purpose agent that can handle various types of queries and tasks.
This agent provides flexible functionality for common use cases.
"""

from typing import Dict, Any, Optional, List
import yaml

from src.core.operators.llm.llm_wrapped_agent import create_wrapped_agent
from src.core.operators.base_op import BaseOperator, OperatorResult


class MCPOperator(BaseOperator):
    """
    A general-purpose AI operator that leverages available MCP tools to efficiently complete a wide variety of tasks.
    
    This operator prioritizes practical action over theoretical analysis, using a tool-first approach to deliver
    quick and effective solutions across different domains including research, file processing, data analysis,
    and general problem-solving. It acts as a versatile interface to the MCP (Model Context Protocol) ecosystem,
    providing users with access to specialized tools and capabilities through natural language interaction.
    
    Key Features:
    - Tool-first problem solving approach
    - Efficient task completion using available MCP tools
    - Flexible handling of diverse query types
    - Direct and results-focused communication
    - Robust error handling and alternative solution finding
    """
    
    def __init__(self, 
                 prompt_path: str,
                 **kwargs):
        """
        Initialize the MCP Operator.
        
        Args:
            prompt_path: Path to YAML config file containing system prompt and description
            tools: List of tools/MCPs available to the operator
            **kwargs: Additional parameters for LLM configuration
        """
        # Load system prompt from YAML if config_path is provided
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            system_prompt = config.get('system_prompt', None)
            description = config.get('description', None)

        name = "MCPOperator"
        super().__init__(name=name, description=description, system_prompt=system_prompt, **kwargs)

    def _build_workflow(self, tools: Optional[List] = None):
        return create_wrapped_agent(
            agent_type="react",
            llm_instance=self.llm_provider,
            tools=tools if tools else []
        )

    async def arun(self, query: str, context: Optional[Dict[str, Any]] = None, tools: Optional[List] = None) -> OperatorResult:
        """
        Run the MCP Operator with the given query.
        
        Args:
            query: User query or task
            context: Additional context information
        
        Returns:
            OperatorResult: Response containing the agent's result
        """
        # Default: ReAct Agent
        workflow = self._build_workflow(tools=tools)
        result: Any = await workflow.ainvoke(query=query, context=context)
        print(result)
        if isinstance(result, str):
            return OperatorResult(base_result=result)
        if isinstance(result, dict):
            final_message = result["messages"][-1]
            return OperatorResult(base_result=final_message.content)
        

        
