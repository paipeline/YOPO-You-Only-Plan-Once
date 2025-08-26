"""
This file is used to help the MCP server to check connection and load the tools.
"""

import os
import yaml
from typing import List, Dict, Any

from src.mcp.mcp_models import ComposioMCP

MCP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "mcp_config.yaml")

def get_all_mcps_info_from_composio() -> List[ComposioMCP]:
    """
    Load MCP definitions from the Composio config and return structured objects.
    Returns a list of ComposioMCP instances so callers can access fields like
    `.name` and `.description` without errors.
    """
    config = read_mcp_config()
    mcp_servers = config.get("mcpServers", {})

    mcps: List[ComposioMCP] = []
    for name, server_config in mcp_servers.items():
        if "url" in server_config:
            url = server_config["url"]
            mcp_id: str = url.strip().split("/")[5]
        mcps.append(
            ComposioMCP(
                name=name,
                description=server_config.get("description", ""),
                mcp_id=mcp_id,
                mcp_prompt=server_config.get("mcp_prompt", "")
            )
        )

    return mcps

def get_mcp_from_composio(mcp_name: str) -> ComposioMCP:
    """
    This function is used to get the MCP from the composio server.
    Returns the MCP server object, or raises an error if not found.
    """
    config = read_mcp_config()
    mcp_servers = config.get("mcpServers", {})
    mcp_server = mcp_servers.get(mcp_name)
    try:
        return ComposioMCP(
            name=mcp_name,
            url=mcp_server.get('url'),
            description=mcp_server.get('description')
        )
    except Exception as e:
        raise Exception(f"Error getting MCP from composio: {e}")

def read_mcp_config() -> Dict[str, Any]:
    """
    Directly read MCP configuration from JSON file.
    Returns the entire configuration dictionary.
    """
    with open(MCP_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

