from __future__ import annotations

import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from mcp_use import MCPAgent as MCPUseAgent

from src.mcp.mcp_client import MCPZoo
from src.tool.tool_manager import get_tools, get_tool_infos
from src.core.operators.llm.llm_provider import LLMProvider


class WrappedAgent(ABC):
    def __init__(self, llm_instance: LLMProvider, resources: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize wrapped agent with resources.
        
        Args:
            llm_instance: LLM provider instance
            resources: Dictionary containing agent-specific resources:
                - 'tools': List of tools for ReActAgent
                - 'mcps': List of MCP names for MCPAgent
                - Any other agent-specific configuration
        """
        self.llm_instance: LLMProvider = llm_instance
        self.resources = resources
        self.workflow = self._build_agent()
    
    @abstractmethod
    def _build_agent(self, resources: Dict[str, Any]):
        """
        Build the agent workflow based on provided resources.
        
        Args:
            resources: Dictionary containing agent-specific resources
        """
        pass

    @abstractmethod
    async def ainvoke(self, query: str, context: Optional[Dict[str, Any]] = None):
        """
        Invoke the agent with query and optional context.
        """
        pass
        

class ReActAgent(WrappedAgent):
    def _build_agent(self):
        """
        Build ReAct agent workflow.
        
        Args:
            resources: Expected to contain 'tools' key with list of tools
        """
        tools = self.resources.get('tools', []) if self.resources else []
        workflow = create_react_agent(
            model=self.llm_instance.llm,
            tools=tools
        )
        return workflow

    async def ainvoke(self, query: str, context: Optional[Dict[str, Any]] = None, custom_initial_state: Optional[Dict[str, Any]] = None):
        # 1. Prepare the initial state
        initial_state: Dict[str, Any] = {
            "messages": [],
            "user_query": query,
        }
        # Add system prompt if available, then the user query
        if hasattr(self.llm_instance, 'system_prompt') and self.llm_instance.system_prompt:
            initial_state["messages"].append(SystemMessage(content=self.llm_instance.system_prompt))
        initial_state["messages"].append(HumanMessage(content=query))
        
        # 2. Add context if provided
        if context:
            context_message = f"Context: {json.dumps(context, indent=2)}\n\nQuery: {query}"
            # Replace the last message (user query) with a context-enriched one
            if initial_state.get("messages"):
                initial_state["messages"][-1] = HumanMessage(content=context_message)
            else:
                initial_state["messages"] = [HumanMessage(content=context_message)]
            initial_state["context"] = context
        
        # 3. Default config for workflow execution
        workflow_config = {"configurable": {"thread_id": "default"}}

        # 4. Execute the workflow asynchronously (let exceptions propagate)
        final_state = await self.workflow.ainvoke(initial_state, config=workflow_config)
        return final_state
    

class MCPAgent(WrappedAgent):
    def _build_agent(self, resources: Dict[str, Any]):
        """
        Build MCP agent workflow.
        
        Args:
            resources: Expected to contain 'mcps' keys
        """
        user_id = "default"
        mcp_names = resources.get('mcps', [])
        assert mcp_names, "MCPs list cannot be empty for MCPAgent"
        
        self.mcp_zoo = MCPZoo(user_id=user_id, mcp_names=mcp_names)
        
        # Check authorization status
        if not self.mcp_zoo.is_all_authorized():
            unauthorized_mcps = self.mcp_zoo.get_unauthorized_mcps()
            print(f"Warning: Some MCPs require authorization:")
            for status in unauthorized_mcps:
                print(f"  - {status.name}: {status.message}")
                if hasattr(status, 'redirect_url') and status.redirect_url:
                    print(f"    OAuth2 URL: {status.redirect_url}")
        
        try:
            # Create MCP client for authorized MCPs only
            mcp_client = self.mcp_zoo.create_client(mcp_name_list=mcp_names)
        except ValueError as e:
            print(f"Warning: {str(e)}")
            print("Available authorized MCPs:", self.mcp_zoo.get_authorized_mcps())
            # Create client with only authorized MCPs
            authorized_mcps = self.mcp_zoo.get_authorized_mcps()
            if authorized_mcps:
                mcp_client = self.mcp_zoo.create_client(mcp_name_list=authorized_mcps)
            else:
                raise ValueError("No authorized MCPs available")
                
        workflow = MCPUseAgent(
            llm=self.llm_instance.llm,
            client=mcp_client,
            max_steps=15,
            system_prompt=getattr(self.llm_instance, 'system_prompt', None),
        )

        return workflow

    async def ainvoke(self, query: str, context: Optional[Dict[str, Any]] = None):
        message_content: str = query
        if context:
            message_content: str = f"Context: {json.dumps(context, indent=2)}\n\nQuery: {query}"

        result = await self.workflow.run(message_content)
        return result


def create_wrapped_agent(
    agent_type: str, 
    llm_instance: LLMProvider, 
    **resources
) -> WrappedAgent:
    """
    Factory function to create wrapped agents.
    
    Args:
        agent_type: Type of agent ('react', 'mcp')
        llm_instance: LLM provider instance
        **resources: Agent-specific resources
    
    Returns:
        WrappedAgent instance
    
    Examples:
        # Create ReAct agent
        react_agent = create_wrapped_agent(
            'react', 
            llm_instance, 
            tools=[tool1, tool2]
        )
        
        # Create MCP agent
        mcp_agent = create_wrapped_agent(
            'mcp', 
            llm_instance, 
            mcps=['mcp1', 'mcp2']
        )
    """
    if agent_type.lower() == 'react':
        return ReActAgent(llm_instance, resources)
    elif agent_type.lower() == 'mcp':
        return MCPAgent(llm_instance, resources)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    

if __name__ == "__main__":
    import asyncio
    from src.core.operators.llm.llm_provider import llm_instance
    
    async def test_wrapped_agents():
        """Simple test for wrapped agents"""
        # Mock LLM instance for testing
        model_name = os.getenv("DEV_MODEL_NAME") or os.getenv("DEV_LLM_NAME") or "gpt-3.5-turbo"
        print(f"Using model: {model_name}")
        
        # Example 1: Without custom system prompt
        provider = llm_instance(model_name)
        
        print("Testing ReAct Agent...")
        try:
            react_agent = create_wrapped_agent(
                'react',
                provider,
                tools=get_tools()  # Empty tools for testing
            )
            print(f"✓ ReAct agent created: {type(react_agent).__name__}")
            
            # Test basic invoke
            result = await react_agent.ainvoke("Tell me the weather in Beijing!")
            print(f"✓ ReAct agent response: {result}")
            
        except Exception as e:
            print(f"✗ ReAct agent test failed: {e}")

    # Run the test
    asyncio.run(test_wrapped_agents())
