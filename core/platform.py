"""
Core Platform Class
Main orchestrator for the Subsurface Data Management Platform
"""

import logging
import time
from typing import Optional

from config.settings import Config
from servers.a2a_server import A2AServerManager
from servers.mcp_server import MCPServerManager
from agents.hybrid_agent import HybridAgentFactory
from agents.meta_agent import MetaAgentFactory
from monitoring.production_monitor import ProductionMonitor
from cli.interactive_shell import InteractiveShell
from utils.port_finder import find_available_port


class SubsurfaceDataPlatform:
    """
    Main platform class that orchestrates all components

    This class manages:
    - Server lifecycle (A2A, MCP)
    - Agent creation (Hybrid, Meta)
    - Production monitoring
    - Interactive shell
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Component managers
        self.a2a_server: Optional[A2AServerManager] = None
        self.mcp_server: Optional[MCPServerManager] = None
        self.hybrid_agent = None
        self.meta_agent = None
        self.monitor: Optional[ProductionMonitor] = None

        # State tracking
        self._initialized = False
        self._start_time = time.time()

    def initialize(self):
        """Initialize all platform components in correct order"""
        if self._initialized:
            self.logger.warning("Platform already initialized")
            return

        self.logger.info("Initializing Subsurface Data Platform...")

        try:
            # Step 1: Setup servers
            self._setup_servers()

            # Step 2: Create agents
            self._create_agents()

            # Step 3: Setup monitoring
            self._setup_monitoring()

            self._initialized = True
            self.logger.info("Platform initialization complete")

        except Exception as e:
            self.logger.error(f"Platform initialization failed: {e}")
            self.shutdown()  # Clean up on failure
            raise

    def _setup_servers(self):
        """Setup A2A and MCP servers"""
        self.logger.info("Setting up servers...")

        # Find available ports if not specified
        if self.config.a2a.port == 0:
            self.config.a2a.port = find_available_port(5000)

        if self.config.mcp.port == 0:
            self.config.mcp.port = find_available_port(7000)

        # Ensure different ports
        if self.config.a2a.port == self.config.mcp.port:
            self.config.mcp.port = find_available_port(self.config.a2a.port + 1)

        # Create and start A2A server
        self.logger.info(f"Starting A2A server on port {self.config.a2a.port}")
        self.a2a_server = A2AServerManager(self.config.a2a, self.config.data)
        self.a2a_server.start()

        # Create and start MCP server
        self.logger.info(f"Starting MCP server on port {self.config.mcp.port}")
        self.mcp_server = MCPServerManager(self.config.mcp, self.config.data)
        self.mcp_server.start()

        # Wait for servers to be ready
        self._wait_for_servers()

    def _create_agents(self):
        """Create hybrid and meta agents"""
        self.logger.info("Creating agents...")

        # Verify servers are ready
        if not self.a2a_server or not self.mcp_server:
            raise RuntimeError("Servers must be started before creating agents")

        if not self.a2a_server.is_ready() or not self.mcp_server.is_ready():
            raise RuntimeError("Servers are not ready for agent creation")

        # Create hybrid agent
        self.logger.info("Creating hybrid agent...")
        agent_factory = HybridAgentFactory(
            a2a_url=self.a2a_server.url,
            mcp_url=self.mcp_server.url,
            config=self.config.agent
        )
        self.hybrid_agent = agent_factory.create()

        if not self.hybrid_agent:
            raise RuntimeError("Failed to create hybrid agent")

        # Create meta agent
        self.logger.info("Creating meta agent...")
        meta_factory = MetaAgentFactory(
            hybrid_agent=self.hybrid_agent,
            config=self.config
        )
        self.meta_agent = meta_factory.create()

        if not self.meta_agent:
            raise RuntimeError("Failed to create meta agent")

        # Verify agent functionality
        self._verify_agents()

    def _setup_monitoring(self):
        """Setup production monitoring"""
        if not self.config.monitoring.enabled:
            self.logger.info("Monitoring disabled by configuration")
            return

        self.logger.info("Setting up production monitoring...")

        try:
            self.monitor = ProductionMonitor(
                meta_agent=self.meta_agent,
                servers={
                    'a2a': self.a2a_server,
                    'mcp': self.mcp_server
                },
                config=self.config.monitoring
            )
            self.monitor.start()
            self.logger.info("Production monitoring active")

        except Exception as e:
            self.logger.warning(f"Monitoring setup failed: {e}")
            # Continue without monitoring

    def _wait_for_servers(self):
        """Wait for all servers to be ready"""
        self.logger.info("Waiting for servers to be ready...")

        # Wait for A2A server
        if not self.a2a_server.wait_ready(timeout=30):
            raise RuntimeError("A2A server failed to start within timeout")

        # Wait for MCP server
        if not self.mcp_server.wait_ready(timeout=30):
            raise RuntimeError("MCP server failed to start within timeout")

        self.logger.info("All servers ready")

    def _verify_agents(self):
        """Verify agents are working correctly"""
        self.logger.info(" Verifying agent functionality...")

        # Test hybrid agent
        if not hasattr(self.hybrid_agent, 'run'):
            raise RuntimeError("Hybrid agent missing 'run' method")

        # Test meta agent
        if not hasattr(self.meta_agent, 'run'):
            raise RuntimeError("Meta agent missing 'run' method")

        # Quick functionality test
        try:
            test_response = self.meta_agent.run("system status")
            if not test_response:
                self.logger.warning("Meta agent test returned empty response")
        except Exception as e:
            self.logger.warning(f"Meta agent test failed: {e}")
            # Don't fail initialization for test failure

        self.logger.info("Agent verification complete")

    def run_interactive(self):
        """Run interactive shell"""
        if not self._initialized:
            raise RuntimeError("Platform must be initialized before running interactive mode")

        self.logger.info("Starting interactive shell...")

        shell = InteractiveShell(
            meta_agent=self.meta_agent,
            config=self.config,
            platform=self
        )
        shell.run()

    def get_status(self) -> dict:
        """Get comprehensive platform status"""
        uptime = time.time() - self._start_time

        status = {
            "platform": {
                "initialized": self._initialized,
                "uptime_seconds": round(uptime, 2),
                "uptime_hours": round(uptime / 3600, 2)
            },
            "servers": {
                "a2a": {
                    "running": self.a2a_server.is_running() if self.a2a_server else False,
                    "ready": self.a2a_server.is_ready() if self.a2a_server else False,
                    "url": self.a2a_server.url if self.a2a_server else None
                },
                "mcp": {
                    "running": self.mcp_server.is_running() if self.mcp_server else False,
                    "ready": self.mcp_server.is_ready() if self.mcp_server else False,
                    "url": self.mcp_server.url if self.mcp_server else None
                }
            },
            "agents": {
                "hybrid_agent": self.hybrid_agent is not None,
                "meta_agent": self.meta_agent is not None
            },
            "monitoring": {
                "enabled": self.monitor is not None,
                "running": self.monitor.is_running() if self.monitor else False
            }
        }

        return status

    def shutdown(self):
        """Clean shutdown of all components"""
        self.logger.info("Shutting down platform...")

        # Stop monitoring first
        if self.monitor:
            try:
                self.monitor.stop()
                self.logger.info("Monitoring stopped")
            except Exception as e:
                self.logger.error(f"Error stopping monitor: {e}")

        # Stop MCP server
        if self.mcp_server:
            try:
                self.mcp_server.stop()
                self.logger.info("MCP server stopped")
            except Exception as e:
                self.logger.error(f"Error stopping MCP server: {e}")

        # Stop A2A server
        if self.a2a_server:
            try:
                self.a2a_server.stop()
                self.logger.info("A2A server stopped")
            except Exception as e:
                self.logger.error(f"Error stopping A2A server: {e}")

        self._initialized = False
        self.logger.info("Platform shutdown complete")

    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# Example usage
if __name__ == "__main__":
    from config.settings import load_config

    config = load_config()

    # Using context manager
    with SubsurfaceDataPlatform(config) as platform:
        print("Platform status:", platform.get_status())