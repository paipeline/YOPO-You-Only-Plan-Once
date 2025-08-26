"""
Tool Manager for YOPO System

This module provides a unified interface for managing and accessing all tools
in the YOPO tool system. It automatically discovers available tools and provides
centralized access to their LangChain tool implementations.
"""

import os
import importlib
import inspect
from typing import Dict, List, Any, Optional, Type, Union
from pathlib import Path

from langchain_core.tools import BaseTool

from src.tool.base_tool import YOPOBaseTool, BaseToolConfig


class ToolManager:
    """
    Unified tool manager for the YOPO system.
    
    This class automatically discovers all available tools in the tool module
    and provides centralized access to their functionality.
    """
    
    def __init__(self, auto_discover: bool = True, **kwargs):
        """
        Initialize the tool manager.
        
        Args:
            auto_discover: Whether to automatically discover tools on initialization
        """
        self._tool_bases: Dict[str, YOPOBaseTool] = {}
        self._tool_bases_classes: Dict[str, Type[YOPOBaseTool]] = {}
        self._initialized_tools: Dict[str, bool] = {}
        
        if auto_discover:
            self._discover_tools()

    
    def _discover_tools(self) -> None:
        """
        Automatically discover all available tool classes in the tool module.
        """
        tool_dir = Path(__file__).parent
        
        # Get all Python files in the tool directory
        for file_path in tool_dir.glob("*.py"):
            if file_path.name.startswith("_") or file_path.name in ["base_tools.py", "tool_manager.py"]:
                continue
                
            module_name = file_path.stem
            
            try:
                # Import the module
                module = importlib.import_module(f".{module_name}", package="src.tool")
                
                # Find all classes that inherit from YOPOBaseTool
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, YOPOBaseTool) and 
                        obj is not YOPOBaseTool and 
                        obj.__module__ == module.__name__):
                        
                        # Register the tool class
                        self._register_tool(tool_class=obj)
                        
            except Exception as e:
                print(f"Failed to import tool module {module_name}: {e}")
    
    def _register_tool(self, tool_class: Type[YOPOBaseTool]) -> None:
        """
        Initialize all discovered tool classes and store instances in _tools.
        
        Args:
            **kwargs: Common initialization parameters to pass to all tools
        """
        try:
            # Create tool instance
            tool_instance: YOPOBaseTool = tool_class()
            name = tool_instance.config.name
            # Store the initialized tool
            self._tool_bases[name] = tool_instance
            
            print(f"Initialized tool: {name}")
            
        except Exception as e:
            print(f"Failed to initialize tool '{tool_class.__name__}': {e}")

    def get_tool(self, tool_base_name: str) -> Optional[YOPOBaseTool]:
        """
        Get a tool instance by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            YOPOBaseTool: Tool instance if found, None otherwise
        """
        return self._tool_bases.get(tool_base_name)
    
    def get_all_base_tool_infos(self) -> Dict[str, Any]:
        """
        Get information about all initialized tools.
        
        Returns:
            Dict[str, Any]: Dictionary mapping tool names to their information
        """
        all_base_tool_infos = dict()
        for tool_base_name in self._tool_bases.keys():
            all_base_tool_infos[tool_base_name] = self._tool_bases[tool_base_name].get_tool_info()

        return all_base_tool_infos

    def get_langchain_tools(self, tool_base_names: Optional[List[str]] = None) -> List[BaseTool]:
        """
        Get LangChain tools from specified or all initialized tools.
        
        Args:
            tool_names: List of tool names to get tools from (None for all)
            
        Returns:
            List[BaseTool]: List of LangChain tools
        """
        all_tools = []
        tools_to_process = tool_base_names if tool_base_names else list(self._tool_bases.keys())

        for tool_name in tools_to_process:
            if tool_name not in self._tool_bases:
                print(f"Warning: Tool '{tool_name}' not found")
                continue
                
            tool = self._tool_bases[tool_name]
                            
            try:
                tool_langchain_tools = tool.get_langchain_tools()
                all_tools.extend(tool_langchain_tools)
            except Exception as e:
                print(f"Error getting tools from '{tool_name}': {e}")
        
        return all_tools
    
    def cleanup(self) -> None:
        """Clean up all tools and reset the manager."""
        self._tool_bases.clear()
        self._initialized_tools.clear()
        print("Tool manager cleaned up")


# Global tool manager instance
_global_tool_manager: Optional[ToolManager] = None


def get_tool_manager() -> ToolManager:
    """
    Get the global tool manager instance.
    
    Returns:
        ToolManager: Global tool manager instance
    """
    global _global_tool_manager
    if _global_tool_manager is None:
        _global_tool_manager = ToolManager()
    return _global_tool_manager


def get_tools(tool_base_names: Optional[List[str]] = None) -> List[BaseTool]:
    """
    Convenience function to get all available LangChain tools.
    
    Args:
        **kwargs: Arguments passed to get_langchain_tools()
        
    Returns:
        List[BaseTool]: List of all available LangChain tools
    """
    return get_tool_manager().get_langchain_tools(tool_base_names=tool_base_names) 

def get_tool_infos(tool_base_names: Optional[List[str]] = None) -> List[BaseTool]:
    """
    Convenience function to get information about all available tools.
    
    Args:
        tool_base_names: Optional list of specific tool base names to get info for
        
    Returns:
        Dict[str, Dict[str, str]]: Dictionary mapping tool instances to their info
    """
    tool_infos = dict()
    all_tools = get_tool_manager().get_langchain_tools(tool_base_names=tool_base_names) 
    for tool in all_tools:
        tool_infos[tool.name] = {
            "name": tool.name,
            "description": tool.description
        }
        
    return tool_infos

def get_all_base_tool_names() -> Dict[str, Any]:
    """
    Get all available tool names and their information.
    
    Returns:
        Dict[str, Any]: Dictionary containing tool names and their metadata
    """
    return get_tool_manager().get_all_base_tool_infos()