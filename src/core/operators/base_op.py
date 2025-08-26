"""
Base Agent for YOPO System - Simple Interface

This module provides a simple base class that all agents inherit from.
It only provides a unified run interface for consistency.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, get_type_hints
import logging
from dotenv import load_dotenv
import inspect
import importlib
from pathlib import Path

from pydantic import BaseModel, Field

from src.core.operators.llm.llm_provider import llm_instance
from src.mcp.mcp_client import MCPZoo

load_dotenv()

class OperatorResult(BaseModel):
    base_result: str = Field(description="The main result content from the agent")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional information and metadata from the agent execution")

class BaseOperator(ABC):
    """
    Simple base class for all YOPO agents.
    
    Provides only a unified run interface. All agent-specific logic
    is implemented in the subclasses.
    """
    
    def __init__(self, 
                 name: str,
                 description: Optional[str] = None,
                 model_name: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 tools: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name identifier
            model_name: LLM model name to use for this agent
            system_prompt: Custom system prompt for the agent
            tools: List of tools for the agent
            **kwargs: Additional parameters for LLM
        """
        self.name = name
        self.description = description if description else f"A {name} operator for YOPO system"
        self.model_name = model_name if model_name else os.getenv("DEV_MODEL_NAME") or os.getenv("DEV_LLM_NAME") or "gpt-3.5-turbo"
        self.system_prompt = system_prompt
        self.mcps = []
        self.tools = tools
        # self.tools = self._load_tools()
        # TODO: Add python interpretor tool

        # Set up logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.name}")
        
        # Initialize LLM provider if model_name is provided
        self.llm_provider = llm_instance(
            model_name=self.model_name,
            system_prompt=system_prompt,
            **kwargs
        )
        self.worklfow = self._build_workflow()

        self.logger.info(f"Initialized {self.name} agent")

    async def _load_tools(self) -> List[Any]:
        """
        [ARCHIVED] Load the tools for the agent from MCP servers.
        
        This function has been archived and is no longer in use.
        Tool loading is now handled differently in the system.
        
        Integrates with the MCP system to load tools based on the agent's
        configured MCP server list.
        
        Returns:
            List[Any]: List of tool objects for the agent
        """
        mcp_zoo = MCPZoo(mcp_names=self.mcps)
        if not self.mcps:
            return []

        # Ensure we have an MCPZoo. If none provided, attempt a safe default for convenience.
        try:
            tools = await mcp_zoo.get_tools(self.mcps)
            return tools
        except Exception as e:
            return []

    @abstractmethod
    def _build_workflow(self, **kwargs):
        """
        Build a LangGraph workflow for the agent.

        Returns:
            The configured workflow (or agent) ready for execution
        """
        

    @abstractmethod
    async def arun(self, query: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Unified run interface for the agent.
        
        Args:
            query: User query/task
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Agent response with at least 'result' key
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

