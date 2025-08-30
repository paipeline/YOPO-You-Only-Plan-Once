import os
import json
from typing import List, Dict, Any, Optional, Set, Union, Tuple
import hashlib
import yaml
from collections import OrderedDict, defaultdict
from dotenv import load_dotenv

from mcp_use import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from composio_client import Composio

from src.mcp.mcp_utils import MCP_CONFIG_PATH, get_all_mcps_info_from_composio
from src.mcp.mcp_models import MCPConnectionStatus, ComposioMCP

load_dotenv()


class MCPZoo:
    """
    MCP Tools wrapper for Composio integration with built-in authorization checking
    """
    def __init__(self, user_id: str = "default", mcp_names: Optional[List[str]] = None):
        """Initialize MCP tools client with authorization checking
        
        Args:
            user_id: User ID for authorization checking
            mcp_names: Set of MCP names to initialize. If None, loads all available MCPs
        """
        self.user_id: str = user_id
        self.mcp_names: Optional[List[str]] = mcp_names

        self._tool_cache: OrderedDict[str, Any] = OrderedDict()
        self._max_cache_size = 100

        # Initialize Composio client for authorization checking
        self.composio = Composio(api_key=os.getenv("COMPOSIO_API_KEY"))
        
        # Initialize MCP instance mapping
        self._init_mcp_instances()
        
        # Initialize MCP configuration boxes
        self._init_mcp_boxes()
        
        # Check authorization status for all MCPs
        self.mcp_connection_status = self._check_mcp_authorization()

    def _init_mcp_instances(self):
        """Initialize MCP instance dictionary from Composio"""
        self.mcp_name2instance: Dict[str, ComposioMCP] = {}
        for mcp_instance in get_all_mcps_info_from_composio():
            self.mcp_name2instance[mcp_instance.name] = mcp_instance

    def _init_mcp_boxes(self):
        """Initialize MCP configuration boxes based on user ID and MCP names."""        
        with open(MCP_CONFIG_PATH, 'r') as file:
            mcp_config_dict: dict = yaml.safe_load(file)

        self.mcp_box: Dict[str, dict] = defaultdict(dict)
        if not self.mcp_names:
            # Default: load all MCPs
            self.mcp_box = mcp_config_dict["mcpServers"]
            return
        
        for mcp_name in self.mcp_names:
            if mcp_name in mcp_config_dict["mcpServers"]:
                item = mcp_config_dict["mcpServers"][mcp_name].copy()
                if "url" in item:
                    item["url"] += f"&user_id={self.user_id}"
                self.mcp_box[mcp_name] = item

    def _get_auth_config_id(self, mcp: str) -> Optional[str]:
        """Get auth config ID for a specific MCP"""
        if mcp not in self.mcp_name2instance:
            return None
            
        mcp_instance: ComposioMCP = self.mcp_name2instance[mcp]
        if mcp_instance.mcp_id is not None:
            try:
                mcp_server = self.composio.mcp.retrieve(mcp_instance.mcp_id)
                auth_config_ids: List[str] = mcp_server.auth_config_ids
                return auth_config_ids[0] if auth_config_ids else None
            except Exception:
                return None
        return None

    def _get_auth_scheme(self, auth_config_id: str) -> Optional[str]:
        """Get the authentication scheme for a specific MCP server"""
        try:
            connected_list = self.composio.connected_accounts.list(
                auth_config_ids=auth_config_id
            )
            if connected_list.items:
                return connected_list.items[0].auth_config.auth_scheme
        except Exception:
            pass
        return None

    def _check_single_mcp_authorization(self, mcp: str) -> MCPConnectionStatus:
        """Check authorization status for a single MCP"""
        auth_config_id = self._get_auth_config_id(mcp)
        
        if auth_config_id is None:
            # No authentication required
            return MCPConnectionStatus(
                name=mcp,
                auth_scheme=None,
                is_authorized=True,
                message="No authentication required"
            )

        auth_scheme = self._get_auth_scheme(mcp, auth_config_id)
        
        try:
            connected_list = self.composio.connected_accounts.list(
                user_ids=self.user_id,
                auth_config_ids=auth_config_id,
            )
            
            if not connected_list.items:
                # Not connected
                if auth_scheme == "OAUTH2":
                    # Create connection request for OAuth2
                    try:
                        connection_request = self.composio.connected_accounts.create(
                            auth_config={
                                "id": auth_config_id
                            },
                            connection={
                                "user_id": self.user_id
                            }
                        )
                        return MCPConnectionStatus(
                            name=mcp,
                            auth_scheme=auth_scheme,
                            is_authorized=False,
                            redirect_url=connection_request.redirect_url,
                            message="OAuth2 authorization required"
                        )
                    except Exception as e:
                        return MCPConnectionStatus(
                            name=mcp,
                            auth_scheme=auth_scheme,
                            is_authorized=False,
                            message=f"Failed to create OAuth2 connection: {str(e)}"
                        )
                elif auth_scheme == 'API_KEY':
                    return MCPConnectionStatus(
                        name=mcp,
                        auth_scheme=auth_scheme,
                        is_authorized=False,
                        message="API key required"
                    )
            else:
                # Connected and authorized
                return MCPConnectionStatus(
                    name=mcp,
                    auth_scheme=auth_scheme,
                    is_authorized=True,
                    message="Authorized"
                )
                
        except Exception as e:
            return MCPConnectionStatus(
                name=mcp,
                auth_scheme=auth_scheme,
                is_authorized=False,
                message=f"Authorization check failed: {str(e)}"
            )

    def _check_mcp_authorization(self) -> List[MCPConnectionStatus]:
        """Check authorization status for all configured MCPs"""
        connection_status = []
        
        mcp_names_to_check = self.mcp_names if self.mcp_names else list(self.mcp_box.keys())
        
        for mcp_name in mcp_names_to_check:
            status = self._check_single_mcp_authorization(mcp_name)
            connection_status.append(status)
            
        return connection_status

    def get_unauthorized_mcps(self) -> List[MCPConnectionStatus]:
        """Get list of MCPs that require authorization"""
        return [status for status in self.mcp_connection_status if not status.is_authorized]

    def get_authorized_mcps(self) -> List[str]:
        """Get list of authorized MCP names"""
        return [status.name for status in self.mcp_connection_status if status.is_authorized]

    def is_all_authorized(self) -> bool:
        """Check if all MCPs are authorized"""
        return all(status.is_authorized for status in self.mcp_connection_status)

    def create_client(self, mcp_name_list: List[str]) -> MCPClient:
        """Create and configure the MCP client for authorized MCPs only"""
        # Filter to only include authorized MCPs
        authorized_mcps = set(self.get_authorized_mcps())
        authorized_mcp_list = [mcp for mcp in mcp_name_list if mcp in authorized_mcps]
        
        if not authorized_mcp_list:
            raise ValueError("No authorized MCPs available for client creation")
            
        new_mcp_server_dict = {mcp_name: self.mcp_box[mcp_name] for mcp_name in authorized_mcp_list}
        
        return MCPClient.from_dict({
            "mcpServers": new_mcp_server_dict
        })

    def authorize_mcp_with_api_key(self, mcp: str, api_key: str) -> Tuple[bool, str]:
        """Authorize MCP with API key"""
        auth_config_id = self._get_auth_config_id(mcp)
        if auth_config_id is None:
            return True, "No authorization required"

        try:
            connection_request = self.composio.connected_accounts.create(
                auth_config={
                    "id": auth_config_id
                },
                connection={
                    "user_id": self.user_id,
                    "state": {
                        "authScheme": "API_KEY",
                        "val": {
                            "status": "ACTIVE",
                            "generic_api_key": api_key
                        }
                    }
                }
            )
            
            # Refresh authorization status
            self.mcp_connection_status = self._check_mcp_authorization()
            
            return True, "Connection created successfully"
        except Exception as e:
            return False, f"Failed to create connection: {str(e)}"

    def delete_mcp_connection(self, mcp: str) -> bool:
        """Delete MCP connection for the user"""
        auth_config_id = self._get_auth_config_id(mcp)
        if auth_config_id is None:
            return True

        try:
            connected_list = self.composio.connected_accounts.list(
                user_ids=self.user_id
            )

            for item in connected_list.items:
                if item.auth_config.id == auth_config_id:
                    dele_connected_account = self.composio.connected_accounts.delete(item.id)
                    if dele_connected_account.success:
                        # Refresh authorization status
                        self.mcp_connection_status = self._check_mcp_authorization()
                        return True
                    else:
                        return False
            return True
        except Exception:
            return False

    def delete_all_mcp_connections(self) -> bool:
        """Delete all MCP connections for the user"""
        try:
            connected_list = self.composio.connected_accounts.list(
                user_ids=self.user_id
            )
            all_success = True
            for item in connected_list.items:
                dele_connected_account = self.composio.connected_accounts.delete(item.id)
                if not dele_connected_account.success:
                    all_success = False

            if all_success:
                # Refresh authorization status
                self.mcp_connection_status = self._check_mcp_authorization()
                
            return all_success
        except Exception:
            return False

    @staticmethod
    def _create_cache_key(mcp_name: str) -> str:
        return hashlib.md5(mcp_name.encode()).hexdigest()

    async def _cache_mcp_tools(self, cache_key: str, mcp_name: str):
        """Asynchronously cache MCP tools"""
        if cache_key in self._tool_cache:
            self._tool_cache.move_to_end(cache_key)
            return self._tool_cache[cache_key]
        
        # Cache miss, create new tools
        client = self.create_client(mcp_name_list=[mcp_name])
        adapter = LangChainAdapter()
        tools = await adapter.create_tools(client)
        
        # Add to cache
        self._tool_cache[cache_key] = tools
        self._tool_cache.move_to_end(cache_key)
        
        # Check cache size limit
        while len(self._tool_cache) > self._max_cache_size:
            oldest_key = next(iter(self._tool_cache))
            del self._tool_cache[oldest_key]
            
        return tools

    async def get_tools(self, mcp_name_list: List[str]) -> list:
        """Discover available tools from authorized MCP servers"""
        # Filter to only include authorized MCPs
        authorized_mcps = set(self.get_authorized_mcps())
        authorized_mcp_list = [mcp for mcp in mcp_name_list if mcp in authorized_mcps]
        
        if not authorized_mcp_list:
            return []
            
        all_tools = []
        for mcp_name in authorized_mcp_list:
            cache_key = self._create_cache_key(mcp_name)
            tools = await self._cache_mcp_tools(cache_key, mcp_name)
            all_tools.extend(tools)
        
        return all_tools
    
    def get_authorization_summary(self) -> Dict[str, Any]:
        """Get summary of authorization status"""
        unauthorized = self.get_unauthorized_mcps()
        authorized = self.get_authorized_mcps()
        
        return {
            "total_mcps": len(self.mcp_connection_status),
            "authorized_count": len(authorized),
            "unauthorized_count": len(unauthorized),
            "authorized_mcps": authorized,
            "unauthorized_mcps": [
                {
                    "name": status.name,
                    "auth_scheme": status.auth_scheme,
                    "message": status.message,
                    "redirect_url": getattr(status, 'redirect_url', None)
                }
                for status in unauthorized
            ],
            "all_authorized": self.is_all_authorized()
        }
    
    def cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return {
            "cache_size": len(self._tool_cache),
            "max_cache_size": self._max_cache_size,
            "cached_keys": list(self._tool_cache.keys())
        }
    
    def clear_cache(self):
        """Clear cache"""
        self._tool_cache.clear()



if __name__ == "__main__":
    # Test the new integrated MCPZoo
    mcp_zoo = MCPZoo(user_id="default", mcp_names=["fire_crawl"])
    
    print("Authorization Summary:")
    print(json.dumps(mcp_zoo.get_authorization_summary(), indent=2))
    
    # Test authorization
    if not mcp_zoo.is_all_authorized():
        print("\nSome MCPs need authorization:")
        for status in mcp_zoo.get_unauthorized_mcps():
            print(f"- {status.name}: {status.message}")
            if status.auth_scheme == "API_KEY":
                # Example: authorize with API key
                # success, msg = mcp_zoo.authorize_mcp_with_api_key(status.name, "your_api_key")
                # print(f"  Authorization result: {success}, {msg}")
                pass