"""
Base Tool for YOPO Tool System

This module provides the abstract base class for all tools in the YOPO system.
All specific tool implementations should inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class BaseToolConfig(BaseModel):
    """Configuration for tool instances."""
    name: str
    description: str
    metadata: Dict[str, Any] = {}


class YOPOBaseTool(ABC):
    """
    Abstract base class for all YOPO tools.
    
    This class provides the foundation for tool implementations that integrate
    with LangChain tools while maintaining YOPO-specific functionality.
    """
    
    def __init__(self, config: BaseToolConfig):
        """
        Initialize the base tool.
        
        Args:
            config: Tool configuration containing name, description, etc.
        """
        self.config = config
        self._tools: List[BaseTool] = []
        self._initialized = False

    @abstractmethod
    def get_langchain_tools(self) -> List[BaseTool]:
        """
        Get the LangChain tools provided by this tool implementation.
        
        Returns:
            List[BaseTool]: List of LangChain tools
        """
        pass

    def is_initialized(self) -> bool:
        """Check if the tool has been initialized."""
        return self._initialized
    
    def get_tool_info(self) -> Dict[str, Any]:
        """
        Get information about this tool.
        
        Returns:
            Dict[str, Any]: Tool information including name, description, status
        """
        return {
            "name": self.config.name,
            "description": self.config.description
        }
    