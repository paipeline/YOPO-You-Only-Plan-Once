"""
Final Answer Tool for YOPO System

This module provides a final answer tool that agents can use to provide
their final response to the user's query.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from src.tool.base_tool import YOPOBaseTool, BaseToolConfig


class FinalAnswerResult(BaseModel):
    """
    Result model for final answer operations.
    
    This model represents the final response that will be returned to the user.
    """
    answer: str = Field(description="The final answer to the user's query")

class FinalAnswerTool(YOPOBaseTool):
    """
    Final answer tool implementation for providing responses to users.
    
    This tool allows agents to provide structured final answers to user queries,
    including confidence levels and source attribution.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the final answer tool.
        
        Args:
            **kwargs: Additional configuration parameters
        """
        config = BaseToolConfig(
            name="final_answer",
            description="Tool for providing final answers to user queries",
            **kwargs
        )
        super().__init__(config)
    
    def get_langchain_tools(self) -> List[BaseTool]:
        """Create LangChain tools for FireCrawl functionality."""
        
        @tool
        def final_answer(answer: str) -> FinalAnswerResult:
            """
            Provide a final answer to the user's query.
            
            This tool should be used when the agent has gathered sufficient information
            and is ready to provide a comprehensive final response to the user.
            
            Args:
                answer: The complete final answer to the user's query
                
            Returns:
                FinalAnswerResult: Structured final answer result
            """
            return FinalAnswerResult(answer=answer)

        
        return [final_answer]