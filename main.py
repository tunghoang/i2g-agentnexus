#!/usr/bin/env python3
"""
Subsurface Data Management Platform
Complete main entry point for the platform - FIXED VERSION

Author: AI Assistant
Version: 2.0.0 (Complete Rewrite)
"""

import sys
import logging
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import Config, load_config
from utils.logging_setup import setup_logging
from utils.api_key_checker import check_api_key

# Platform components
from servers.mcp_server import MCPServerManager, MCPClient
from agents.hybrid_agent import create_adaptive_agent, HybridAgentFactory
from cli.interactive_shell import InteractiveShell

# NEW: Agent configuration imports
from config.agent_config import AGENT_TYPE

# NEW: Google ADK agent import
try:
    from agents.google_adk_hybrid_agent import create_google_adk_hybrid_agent

    GOOGLE_ADK_AVAILABLE = True
    print("Google ADK Hybrid Agent available")
except ImportError as e:
    GOOGLE_ADK_AVAILABLE = False
    print(f"Google ADK Hybrid Agent not available: {e}")


class EmergencyAgent:
    """Emergency fallback agent"""

    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.start_time = time.time()
        self.stats = {
            "total_queries": 0,
            "direct_commands": 0,
            "agent_responses": 0,
            "errors": 0,
            "system_type": "Emergency Agent"
        }
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Emergency agent created - limited functionality")

    def run(self, query):
        """Basic query processing"""
        self.stats["total_queries"] += 1

        try:
            query_lower = query.lower().strip()

            if any(word in query_lower for word in ["list", "files", "show"]):
                result = self.mcp_client.call_tool("list_files", "*")
                content = self._extract_content(result)
                self.stats["direct_commands"] += 1
                return f"Available Files:\n{content}"

            elif any(word in query_lower for word in ["status", "health", "system"]):
                result = self.mcp_client.call_tool("system_status", "")
                content = self._extract_content(result)
                self.stats["direct_commands"] += 1
                return f"System Status:\n{content}"

            elif any(word in query_lower for word in ["help", "commands"]):
                return self._show_help()

            else:
                self.stats["errors"] += 1
                return """Emergency Agent Active

Available commands:
- 'list files' - Show available data files
- 'system status' - Check system health
- 'help' - Show this message

WARNING: Limited functionality - Full analysis requires Google ADK:
```
pip install google-adk
```
Then update config/agent_config.py: AGENT_TYPE = "google_adk_hybrid"
And restart the platform."""

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Emergency agent error: {e}")
            return f"Emergency agent error: {str(e)}\n\nTry: 'list files' or 'system status'"

    def get_stats(self):
        """Return statistics"""
        self.stats["uptime_hours"] = (time.time() - self.start_time) / 3600
        return self.stats.copy()

    def _extract_content(self, result):
        """Extract content from MCP response"""
        try:
            if isinstance(result, dict):
                if 'content' in result and isinstance(result['content'], list):
                    if len(result['content']) > 0 and 'text' in result['content'][0]:
                        return result['content'][0]['text']
                if 'text' in result:
                    return result['text']
            return str(result)
        except Exception as e:
            return f"Error extracting content: {e}"

    def _show_help(self):
        """Show help message"""
        return """Emergency Agent Help

Available Commands:
- `list files` - Show available data files
- `system status` - Check system health
- `help` - Show this help message

To Get Full Google ADK HybridAgent:
1. Install: `pip install google-adk`
2. Update config/agent_config.py: `AGENT_TYPE = "google_adk_hybrid"`
3. Restart platform
4. Enjoy full subsurface analysis with better parameter handling!"""


class SubsurfaceDataPlatform:
    """
    Main platform class that orchestrates all components - FIXED VERSION
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Platform components
        self.mcp_server = None
        self.agent = None
        self.initialized = False
        self.start_time = time.time()

    def initialize(self):
        """Initialize all platform components"""
        self.logger.info("Initializing Subsurface Data Platform...")

        try:
            # Step 1: Initialize MCP Server
            self._initialize_mcp_server()

            # Step 2: Wait for MCP server to be ready
            self._wait_for_mcp_server()

            # Step 3: Create agent (NEW: configurable)
            self._create_agent()

            self.initialized = True
            self.logger.info("Platform initialization complete")

        except Exception as e:
            self.logger.error(f"Platform initialization failed: {e}")
            raise

    def _initialize_mcp_server(self):
        """Initialize the MCP server"""
        self.logger.info("Starting MCP server...")

        self.mcp_server = MCPServerManager(
            config=self.config.mcp,
            data_config=self.config.data
        )

        # Start the server
        self.mcp_server.start()
        self.logger.info("MCP server started")

    def _wait_for_mcp_server(self):
        """Wait for MCP server to be ready"""
        self.logger.info("Waiting for MCP server to be ready...")

        if not self.mcp_server.wait_ready(timeout=30):
            raise RuntimeError("MCP server failed to start within 30 seconds")

        self.logger.info("MCP server is ready")

    def _create_agent(self):
        """Create agent based on configuration"""
        self.logger.info(f"Creating agent using {AGENT_TYPE} implementation...")

        # Create MCP client for the agent
        mcp_client = MCPClient(self.mcp_server.url)

        # Test that we can get tools
        tools_response = mcp_client.get_tools()
        if "error" in tools_response:
            self.logger.warning(f"Tools not immediately available: {tools_response['error']}")

        # NEW: Agent creation with fallback support
        self.agent = self._create_agent_with_fallback(mcp_client, tools_response)

        # Wrap agent to add compatibility methods if needed
        if not hasattr(self.agent, 'get_stats'):
            self.agent = self._wrap_agent_for_compatibility(self.agent)

        self.logger.info(f"Agent created successfully using {AGENT_TYPE} implementation")

    def _create_agent_with_fallback(self, mcp_client, tools_response):
        """Create agent with Google ADK HybridAgent support"""

        # Import agent_config dynamically to get current settings
        from config.agent_config import AGENT_TYPE

        try:
            if AGENT_TYPE == "google_adk_hybrid":
                self.logger.info("Creating Google ADK-powered HybridAgent...")

                if not GOOGLE_ADK_AVAILABLE:
                    raise ImportError("Google ADK not available. Install with: pip install google-adk")

                # Create Google ADK HybridAgent using your existing pattern
                agent = create_google_adk_hybrid_agent(
                    mcp_url=self.config.mcp.url,
                    config=self.config.agent
                )

                self.logger.info("Google ADK HybridAgent created successfully")
                return agent

            elif AGENT_TYPE == "hybrid":
                self.logger.info("Creating original HybridAgent (with LangChain)...")

                # Your existing hybrid agent creation
                from agents.hybrid_agent import create_hybrid_agent
                agent = create_hybrid_agent(
                    a2a_url=self.config.a2a.url,
                    mcp_url=self.config.mcp.url,
                    config=self.config.agent
                )

                self.logger.info("Original HybridAgent created")
                return agent

            else:
                # Emergency fallback
                self.logger.warning("Creating emergency fallback agent")
                return EmergencyAgent(mcp_client)

        except Exception as e:
            self.logger.error(f"Error creating {AGENT_TYPE} agent: {e}")

            # Fallback logic
            if AGENT_TYPE == "google_adk_hybrid":
                self.logger.info("Google ADK failed, trying original HybridAgent...")
                try:
                    from agents.hybrid_agent import create_hybrid_agent
                    agent = create_hybrid_agent(
                        a2a_url=self.config.a2a.url,
                        mcp_url=self.config.mcp.url,
                        config=self.config.agent
                    )
                    self.logger.info("Fallback to original HybridAgent successful")
                    return agent
                except Exception as fallback_error:
                    self.logger.error(f"HybridAgent fallback failed: {fallback_error}")

            # Final emergency fallback
            self.logger.warning("All agents failed, using emergency agent")
            return EmergencyAgent(mcp_client)

    def _wrap_agent_for_compatibility(self, agent):
        """Wrap agent for MetaAgent compatibility - IMPROVED VERSION"""

        # Check if agent is already a HybridAgent (Google ADK or original)
        if hasattr(agent, 'run') and hasattr(agent, 'get_stats'):
            # Perfect! HybridAgent already has the right interface for MetaAgent
            self.logger.info("HybridAgent is compatible with MetaAgent - no wrapper needed")
            return agent

        # Create wrapper for other agent types
        self.logger.info("Creating compatibility wrapper for non-HybridAgent")

        class AgentWrapper:
            def __init__(self, agent, platform):
                self.agent = agent
                self.platform = platform
                self.stats = {
                    "total_queries": 0,
                    "uptime_hours": 0,
                    "system_type": self._get_agent_type()
                }

            def _get_agent_type(self):
                """Determine agent type - IMPROVED VERSION"""
                if hasattr(self.agent, '__class__'):
                    class_name = self.agent.__class__.__name__
                    module_name = getattr(self.agent.__class__, '__module__', '')

                    # Better detection logic
                    if 'google_adk' in module_name.lower() or 'GoogleADK' in class_name:
                        return "Google ADK HybridAgent"
                    elif 'Emergency' in class_name:
                        return "Emergency Agent"
                    elif 'Hybrid' in class_name:
                        return "HybridAgent"
                    else:
                        return f"{class_name} Agent"

                # Fallback based on AGENT_TYPE config
                if AGENT_TYPE == "google_adk_hybrid":
                    return "Google ADK HybridAgent"
                elif AGENT_TYPE == "hybrid":
                    return "HybridAgent"
                else:
                    return "Unknown Agent"

            def run(self, query):
                """Run method for MetaAgent compatibility"""
                self.stats["total_queries"] += 1
                self.stats["uptime_hours"] = (time.time() - self.platform.start_time) / 3600

                try:
                    if hasattr(self.agent, 'run'):
                        return self.agent.run(query)
                    elif hasattr(self.agent, 'execute_query'):
                        return self.agent.execute_query(query)
                    else:
                        return str(self.agent(query))
                except Exception as e:
                    self.platform.logger.error(f"Wrapped agent error: {e}")
                    raise e

            def get_stats(self):
                """Get comprehensive stats - IMPROVED VERSION"""
                # Update wrapper stats
                self.stats["uptime_hours"] = (time.time() - self.platform.start_time) / 3600

                if hasattr(self.agent, 'get_stats'):
                    try:
                        agent_stats = self.agent.get_stats()
                        # Merge with wrapper stats, prioritizing agent stats
                        combined_stats = self.stats.copy()
                        if isinstance(agent_stats, dict):
                            combined_stats.update(agent_stats)
                        return combined_stats
                    except Exception as e:
                        self.platform.logger.warning(f"Error getting agent stats: {e}")
                        return self.stats.copy()
                else:
                    return self.stats.copy()

        return AgentWrapper(agent, self)

    def run_interactive(self):
        """Run the interactive shell"""
        if not self.initialized:
            raise RuntimeError("Platform not initialized. Call initialize() first.")

        self.logger.info("Starting interactive mode...")

        # Create and run interactive shell
        shell = InteractiveShell(self.agent, self.config, platform=self)
        shell.run()

    def get_mcp_server(self):
        """Get the MCP server instance"""
        return self.mcp_server

    def get_status(self):
        """Get comprehensive platform status - FIXED VERSION"""
        uptime_hours = (time.time() - self.start_time) / 3600

        # Get tools count safely
        tools_count = 0
        if self.mcp_server:
            # Try different ways to get tools count
            if hasattr(self.mcp_server, 'tools_registered'):
                tools_count = self.mcp_server.tools_registered
            elif hasattr(self.mcp_server, '_tools'):
                tools_count = len(self.mcp_server._tools)
            else:
                # Fallback: try to get tools from MCP client
                try:
                    mcp_client = MCPClient(self.mcp_server.url)
                    tools_response = mcp_client.get_tools()
                    if isinstance(tools_response, list):
                        tools_count = len(tools_response)
                    elif isinstance(tools_response, dict) and "error" not in tools_response:
                        tools_count = len(tools_response)
                except:
                    tools_count = 0

        status = {
            "platform": {
                "initialized": self.initialized,
                "uptime_hours": uptime_hours,
                "version": "2.0.0",
                "agent_type": AGENT_TYPE
            },
            "servers": {
                "mcp": {
                    "ready": self.mcp_server.is_ready() if self.mcp_server else False,
                    "url": self.mcp_server.url if self.mcp_server else None,
                    "tools_count": tools_count
                }
            },
            "agents": {
                "current_agent": AGENT_TYPE,
                "agent_ready": self.agent is not None,
                "google_adk_available": GOOGLE_ADK_AVAILABLE,
                "agent_class": self.agent.__class__.__name__ if self.agent else None
            },
            "monitoring": {
                "enabled": True
            }
        }

        # Add agent-specific status if available
        if self.agent and hasattr(self.agent, 'get_stats'):
            try:
                agent_stats = self.agent.get_stats()
                if isinstance(agent_stats, dict):
                    status["agent_stats"] = agent_stats
            except Exception as e:
                self.logger.warning(f"Could not get agent stats: {e}")

        return status

    def shutdown(self):
        """Shutdown the platform cleanly"""
        self.logger.info("Shutting down platform...")

        try:
            if self.mcp_server:
                self.mcp_server.stop()
                self.logger.info("MCP server stopped")
        except Exception as e:
            self.logger.error(f"Error stopping MCP server: {e}")

        self.logger.info("Platform shutdown complete")


def main():
    """Main entry point with Google ADK HybridAgent support - IMPROVED VERSION"""
    try:
        # Check API key
        if not check_api_key():
            print("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            return 1

        # Load configuration
        config = load_config()
        setup_logging(config.logging)
        logger = logging.getLogger(__name__)

        # Import current agent config
        from config.agent_config import AGENT_TYPE

        logger.info(f"Starting Subsurface Data Management Platform v2.0 (Agent: {AGENT_TYPE})")

        # Enhanced startup banner
        print("\n" + "=" * 80)
        print("SUBSURFACE DATA MANAGEMENT PLATFORM v2.0")
        print("GOOGLE ADK HYBRID AGENT INTEGRATION")
        print("=" * 80)
        print(f"Agent Type: {AGENT_TYPE}")

        if AGENT_TYPE == "google_adk_hybrid":
            if GOOGLE_ADK_AVAILABLE:
                print("Google ADK: Available")
                print("Features: HybridAgent + Google ADK + 22 MCP Tools")
                print("Benefits: Better parameter handling, no LangChain issues")
            else:
                print("Google ADK: Not available")
                print("Will fallback to original HybridAgent")
                print("Install: pip install google-adk")
        elif AGENT_TYPE == "hybrid":
            print("Using original HybridAgent (may have LangChain issues)")
            print("Recommended: Switch to 'google_adk_hybrid' in agent_config.py")
        else:
            print("Emergency mode - limited functionality")

        print("\nInitializing platform components...")

        # Create and initialize platform
        platform = SubsurfaceDataPlatform(config)

        print("Starting MCP server...")
        platform.initialize()

        print("Creating agent...")

        # Show final agent status - IMPROVED VERSION
        if platform.agent:
            if hasattr(platform.agent, 'get_stats'):
                try:
                    agent_stats = platform.agent.get_stats()
                    agent_type = agent_stats.get('system_type', 'Unknown Agent')
                    print(f"Active Agent: {agent_type}")

                    # Additional debug info
                    if hasattr(platform.agent, '__class__'):
                        class_name = platform.agent.__class__.__name__
                        module_name = getattr(platform.agent.__class__, '__module__', '')
                        print(f"   Class: {class_name} (from {module_name})")

                except Exception as e:
                    logger.warning(f"Could not get agent stats: {e}")
                    print(f"Agent: Created successfully ({platform.agent.__class__.__name__})")
            else:
                print(f"Agent: Created successfully ({platform.agent.__class__.__name__})")

        print("\nPlatform initialization complete!")
        print("Ready for interactive analysis")
        print("=" * 80)

        # Run interactive mode
        platform.run_interactive()
        return 0

    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Interrupted by user")
        print("\nGoodbye!")
        return 0
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Platform startup failed: {e}")
        print(f"\nPlatform failed to start: {e}")

        # Show helpful error information
        if "google" in str(e).lower() and "adk" in str(e).lower():
            print("\nGoogle ADK issue detected:")
            print("   1. Install: pip install google-adk")
            print("   2. Check OpenAI API key is set")
            print("   3. Or switch to original HybridAgent in agent_config.py")

        import traceback
        traceback.print_exc()
        return 1
    finally:
        if 'platform' in locals():
            try:
                platform.shutdown()
            except Exception as e:
                print(f"Error during shutdown: {e}")


def test_platform():
    """Test function to verify platform setup"""
    try:
        print(f"Testing platform setup with {AGENT_TYPE} agent...")

        # Check API key
        if not check_api_key():
            print("OpenAI API key missing")
            return False

        # Load config
        config = load_config()
        print("Configuration loaded")

        # Test MCP server creation
        from servers.mcp_server import MCPServerManager
        mcp_server = MCPServerManager(config.mcp, config.data)
        print("MCP server manager created")

        # Test agent creation function based on type
        if AGENT_TYPE == "openai_tools":
            try:
                from agents.openai_tools_agent import OpenAIToolsExecutor
                print("OpenAI Tools Agent available")
            except ImportError as e:
                print(f"OpenAI Tools Agent not available: {e}")
                return False
        else:
            from agents.hybrid_agent import create_adaptive_agent
            print("Adaptive Agent available")

        print("Platform setup test passed!")
        return True

    except Exception as e:
        print(f"Platform setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_start():
    """Quick start mode for testing"""
    print(f"\nQUICK START MODE ({AGENT_TYPE.upper()} AGENT)")
    print("=" * 40)

    if not test_platform():
        print("Setup test failed. Please fix configuration issues.")
        return 1

    print("\nSetup test passed!")
    print("Starting full platform...")

    return main()


if __name__ == "__main__":
    # You can choose which mode to run:

    # Option 1: Full platform (default)
    sys.exit(main())

    # Option 2: Quick start with testing (uncomment to use)
    # sys.exit(quick_start())

    # Option 3: Just run setup test (uncomment to use)
    # test_platform()