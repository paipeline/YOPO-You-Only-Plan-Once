"""
Browser Use Operator for YOPO System

A specialized operator that enables AI agents to control web browsers using the browser-use library.
This operator allows agents to navigate websites, interact with web elements, and automate browser tasks
through natural language instructions.
"""

import os
from typing import Dict, Any, Optional, List, TypedDict, Annotated
import yaml
import logging

from browser_use import Agent as BrowserAgent
from browser_use import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from src.core.operators.base_op import BaseOperator, OperatorResult
from src.core.operators.llm.llm_provider import llm_instance, LLMProvider


class BrowserUseState(TypedDict):
    """State object for the Browser Use workflow"""
    # Input
    original_query: str
    context: Optional[Dict[str, Any]]

    # Task execution results
    task_result: str
    execution_success: bool
    execution_error: Optional[str]
    execution_steps: Optional[List]
    execution_step_num: int
    
    # Messages for LLM interaction
    messages: Annotated[List[BaseMessage], add_messages]


async def execute_browser_task_node(state: BrowserUseState, model: str) -> BrowserUseState:
    """
    Node 2: Execute the browser task using browser-use Agent.
    """
    original_query = state["original_query"]
    context: Optional[Dict[str, Any]] = state["context"]
    llm = ChatOpenAI(model=model)

    try:
        if context:
            agent = BrowserAgent(
                task=original_query,
                llm=llm,
                page_extraction_llm=llm,
                extend_system_message=f"Context: {context}"
            )
        else:
            agent = BrowserAgent(
                task=original_query,
                llm=llm,
                page_extraction_llm=llm,
            )
        
        print("🚀 Starting browser agent task...")
        
        # Execute the task
        max_steps = 50
        result = await agent.run(max_steps=max_steps)
        
        print("🎉 Browser task completed successfully!")
        
        # FIXME: task_result: final or all middle result of all actions
        state.update({
            "task_result": "\n".join(result.extracted_content()),
            "execution_success": result.is_successful(),
            "execution_error": "\n".join([str(item) for item in result.errors() if item]),
            "execution_step_num": result.number_of_steps(),
            "execution_steps": result.action_history()
        })
        
    except Exception as e:
        print(f"❌ Browser task failed: {str(e)}")
        
        state.update({
            "task_result": f"Browser task failed: {str(e)}",
            "execution_success": False,
            "execution_error": str(e),
            "execution_step_num": 0,
            "execution_steps": None
        })
    
    return state



class BrowserUseOperator(BaseOperator):
    """
    Browser Use Operator using browser-use library.
    
    This operator enables AI agents to control web browsers through natural language instructions.
    It supports both local browser automation and cloud-based browser services.
    """
    
    def __init__(self, 
                 prompt_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Browser Use Operator.
        
        Args:
            prompt_path: Path to YAML file containing prompts (optional)
            browser_config: Browser configuration options
            execution_options: Task execution options
            **kwargs: Additional configuration
        """
        
        # Load prompts if provided
        self.prompts = {}
        with open(prompt_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            system_prompt = config.get('system_prompt')
            description = config.get('description')

        # Initialize base operator
        name = "BrowserUseOperator"
        super().__init__(name=name, description=description, model_name=os.getenv("BROWSER_USER_MODEL_NAME", "gpt-4o"), system_prompt=system_prompt, **kwargs)

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for browser automation."""
        workflow = StateGraph(BrowserUseState)
        
        async def _execute_task_wrapper(state: BrowserUseState) -> BrowserUseState:
            return await execute_browser_task_node(state, self.model_name)
        
        # Add nodes
        workflow.add_node("browse_executor", _execute_task_wrapper)
        
        # Define the workflow edges
        workflow.add_edge(START, "browse_executor")
        workflow.add_edge("browse_executor", END)
        
        return workflow.compile()

    async def arun(self, query: str, context: Optional[Dict[str, Any]] = None) -> OperatorResult:
        """
        Run the browser automation workflow.
        
        Args:
            query: Natural language task description for browser automation
            context: Additional context containing configuration overrides
            
        Returns:
            AgentResult with browser automation results
        """
        try:
            # Initialize state
            initial_state = BrowserUseState(
                original_query=query,
                context=context,
                task_result="",
                execution_success=False,
                execution_error=None,
                execution_steps=[],
                execution_step_num=0,
                messages=[]
            )
            
            # Run the workflow
            final_state = await self.worklfow.ainvoke(initial_state)
            
            # Prepare result
            result = OperatorResult(
                base_result=final_state["task_result"],
                metadata={
                    "execution_success": final_state.get("execution_success", False),
                    "error": final_state.get("execution_error"),
                    "execution_steps": final_state.get("execution_steps", []),
                    "execution_step_num": final_state.get("execution_step_num", 0),
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Browser automation failed: {str(e)}")
            return OperatorResult(
                base_result=f"Browser automation failed: {str(e)}",
                metadata={
                    "error": str(e),
                    "execution_success": False,
                    "execution_steps": [],
                    "execution_step_num": 0,
                }
            )
