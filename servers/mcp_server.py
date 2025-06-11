"""
MCP Server Management
Handles Model Context Protocol server with subsurface data tools
"""

import time
import json
from typing import Dict, Any, Optional

# MCP and tool imports (from your original code)
from python_a2a.mcp import FastMCP

from .base_server import BaseServer, HealthCheckMixin
from config.settings import MCPConfig, DataConfig
from tools.las_tools import create_las_tools
from tools.segy_tools import create_segy_tools
from tools.system_tools import create_system_tools


class MCPServerManager(BaseServer, HealthCheckMixin):
    """
    Manages MCP (Model Context Protocol) server lifecycle

    Responsibilities:
    - Create MCP server with subsurface data tools
    - Register LAS, SEG-Y, and system tools
    - Manage server lifecycle
    - Tool availability monitoring
    """

    def __init__(self, config: MCPConfig, data_config: DataConfig):
        super().__init__("MCP", config.host, config.port)
        self.config = config
        self.data_config = data_config
        self.mcp_server: Optional[FastMCP] = None
        self.tools_registered = 0

    def _create_server(self):
        """Create MCP server with all subsurface data tools"""
        self.logger.info("Creating MCP server with subsurface data tools...")

        # Create MCP server instance
        self.mcp_server = FastMCP(
            name="Subsurface Data Management Tools",
            description="Advanced tools for managing, processing, and analyzing LAS files, SEG-Y seismic data, and integrated subsurface datasets"
        )

        # Register all tool categories
        self._register_las_tools()
        self._register_segy_tools()
        self._register_system_tools()

        self.logger.info(f"MCP server created with {self.tools_registered} tools")

    def _register_las_tools(self):
        """Register LAS file processing tools"""
        self.logger.info("Registering LAS tools...")

        las_tools = create_las_tools(self.mcp_server, self.data_config)
        tool_count = len(las_tools)
        self.tools_registered += tool_count

        self.logger.info(f"Registered {tool_count} LAS tools")

    def _register_segy_tools(self):
        """Register SEG-Y seismic processing tools"""
        self.logger.info("Registering SEG-Y tools...")

        segy_tools = create_segy_tools(self.mcp_server, self.data_config)
        tool_count = len(segy_tools)
        self.tools_registered += tool_count

        self.logger.info(f"Registered {tool_count} SEG-Y tools")

    def _register_system_tools(self):
        """Register system and utility tools"""
        self.logger.info("Registering system tools...")

        system_tools = create_system_tools(self.mcp_server, self.data_config)
        tool_count = len(system_tools)
        self.tools_registered += tool_count

        self.logger.info(f"Registered {tool_count} system tools")

    def _start_server(self):
        """Start the MCP server in background thread"""
        if not self.mcp_server:
            raise RuntimeError("MCP server not created")

        def run_mcp_server():
            self.mcp_server.run(host=self.host, port=self.port)

        self.run_in_thread(run_mcp_server)

        # Give server time to start
        time.sleep(2)

    def _stop_server(self):
        """Stop the MCP server"""
        # MCP server stops when thread ends
        self._running = False

    def _check_health(self) -> bool:
        """Check MCP server health"""
        if not self.is_running():
            return False

        try:
            import requests
            # Try to access the tools endpoint
            response = requests.get(f"{self.url}/tools", timeout=5)
            return response.status_code == 200
        except Exception:
            # Fallback to thread status
            return self.is_running()

    def get_tools(self) -> Dict[str, Any]:
        """Get list of available tools"""
        if not self.is_ready():
            return {"error": "Server not ready"}

        try:
            import requests
            response = requests.get(f"{self.url}/tools", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": f"Failed to get tools: {str(e)}"}

    def call_tool(self, tool_name: str, input_data: Any) -> Dict[str, Any]:
        """Call a specific tool"""
        if not self.is_ready():
            return {"error": "Server not ready"}

        try:
            import requests

            # Convert input to appropriate format
            if isinstance(input_data, dict):
                input_json = json.dumps(input_data)
            elif isinstance(input_data, str) and (input_data.startswith('{') or input_data.startswith('[')):
                try:
                    json.loads(input_data)  # Validate JSON
                    input_json = input_data
                except json.JSONDecodeError:
                    input_json = input_data
            else:
                input_json = str(input_data)

            response = requests.post(
                f"{self.url}/tools/{tool_name}",
                json={"input": input_json},
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}

        except Exception as e:
            return {"error": f"Tool call failed: {str(e)}"}

    def get_status(self) -> dict:
        """Get comprehensive MCP server status"""
        base_status = super().get_status()

        mcp_status = {
            **base_status,
            "tools_registered": self.tools_registered,
            "tools_available": self.get_tools() if self.is_ready() else "Server not ready"
        }

        return mcp_status


class MCPClient:
    """
    Simple MCP client for tool calls
    Used by agents to interact with MCP tools
    """

    def __init__(self, server_url: str):
        self.server_url = server_url

    def call_tool(self, tool_name: str, input_data: Any) -> Dict[str, Any]:
        """Make a direct call to an MCP tool"""
        try:
            import requests

            # Convert input to appropriate format
            if isinstance(input_data, dict):
                input_json = json.dumps(input_data)
            elif isinstance(input_data, str) and (input_data.startswith('{') or input_data.startswith('[')):
                try:
                    json.loads(input_data)  # Validate JSON
                    input_json = input_data
                except json.JSONDecodeError:
                    input_json = input_data
            else:
                input_json = str(input_data)

            response = requests.post(
                f"{self.server_url}/tools/{tool_name}",
                json={"input": input_json},
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP error {response.status_code}: {response.text}"}

        except Exception as e:
            return {"error": f"Error calling MCP tool: {str(e)}"}

    def get_tools(self) -> Dict[str, Any]:
        """Get list of available tools"""
        try:
            import requests
            response = requests.get(f"{self.server_url}/tools", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": f"Failed to get tools: {str(e)}"}


if __name__ == "__main__":
    # Test MCP server manager
    from config.settings import MCPConfig, DataConfig

    mcp_config = MCPConfig(port=7001)
    data_config = DataConfig()

    server = MCPServerManager(mcp_config, data_config)

    print(f"Server status: {server.get_status()}")

    try:
        server.start()
        print("Server started successfully")

        if server.wait_ready(timeout=10):
            print("Server is ready!")

            # Test tool access
            tools = server.get_tools()
            print(f"Available tools: {list(tools.keys()) if isinstance(tools, dict) and 'error' not in tools else 'Error getting tools'}")

        else:
            print("Server not ready")

        time.sleep(2)
        server.stop()
        print("Server stopped")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()