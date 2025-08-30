"""
Code Act Tool for YOPO System

This module provides a code execution tool that agents can use to run
Python code and interact with the environment programmatically.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from src.tool.base_tool import YOPOBaseTool, BaseToolConfig
from src.tool.interpreter.local_interpreter import evaluate_python_code, BASE_BUILTIN_MODULES, BASE_PYTHON_TOOLS


class CodeExecutionResult(BaseModel):
    """
    Result model for code execution operations.
    
    This model represents the result of executing Python code.
    """
    output: Optional[str] = Field(default=None, description="The output from executing the code")
    error: Optional[str] = Field(default=None, description="Any error that occurred during execution")

class CodeActTool(YOPOBaseTool):
    """
    Code execution tool implementation for running Python code.
    
    This tool allows agents to execute Python code in a safe environment,
    with access to authorized imports and custom tools.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the code act tool.
        
        Args:
            **kwargs: Additional configuration parameters
        """
        config = BaseToolConfig(
            name="code_act",
            description="Tool for executing Python code in a safe environment",
            **kwargs
        )
        super().__init__(config)
    
    def get_langchain_tools(self) -> List[BaseTool]:
        """Create LangChain tools for FireCrawl functionality."""
        
        @tool
        def run_python_code(code: str):
            """
            Execute Python code in a safe environment.
            
            This tool allows execution of Python code with access to authorized
            imports and built-in modules. The code is executed in an isolated
            environment to ensure safety.
            
            Args:
                code: The Python code to execute
                
            Returns:
                CodeExecutionResult: Result containing output or error information
            """
            try:
                state = {}
                output = str(
                    evaluate_python_code(
                        code,
                        state=state,
                        static_tools=BASE_PYTHON_TOOLS,
                        authorized_imports=BASE_BUILTIN_MODULES,
                    )[0]  # The second element is boolean is_final_answer
                )

                output = f"Stdout:\n{str(state['_print_outputs'])}\nOutput: {output}"

                result = CodeExecutionResult(
                    output=output,
                    error = None
                )
            except Exception as e:
                result = CodeExecutionResult(
                    output = None,
                    error=str(e),
                )
            return result

        return [run_python_code]