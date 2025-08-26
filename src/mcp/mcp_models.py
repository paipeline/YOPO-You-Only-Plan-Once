from typing import List, Dict, Any, Optional, Set

from pydantic import BaseModel, Field


class MCPConnectionStatus(BaseModel):
    name: str = Field(..., description="Name of the MCP server")
    auth_scheme: Optional[str] = Field(default=None, description="Authentication scheme (OAUTH2, API_KEY, etc.)")
    is_authorized: bool = Field(..., description="Whether the MCP connection is authorized")
    redirect_url: Optional[str] = Field(default=None, description="OAuth2 redirect URL if needed")
    message: Optional[str] = Field(default=None, description="Status message or error description")


class ComposioMCP(BaseModel):
    name: str = Field(..., description="Name of the MCP server")
    description: str = Field(..., description="Description of the MCP server")
    url: Optional[str] = Field(default=None, description="URL of the MCP server")
    mcp_id: Optional[str] = Field(default=None, description="Composio mcp server ID")
    mcp_prompt: Optional[str] = Field(default=None, description="Custom prompt for the MCP server")
