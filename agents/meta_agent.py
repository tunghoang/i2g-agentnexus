"""
Meta Agent Factory
Creates meta agents that wrap hybrid agents with additional functionality
"""

import time
import logging
from typing import Dict, Any

from config.settings import Config
from .hybrid_agent import HybridAgent


class MetaAgent:
    """
    Meta agent that wraps a hybrid agent with additional production features

    Features:
    - Enhanced error handling and recovery
    - Performance monitoring
    - Query preprocessing and postprocessing
    - System integration and status reporting
    """

    def __init__(self, hybrid_agent: HybridAgent, config: Config):
        self.hybrid_agent = hybrid_agent
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Meta-agent state
        self.start_time = time.time()
        self.query_count = 0
        self.last_query_time = None

    def run(self, query: str) -> str:
        """
        Enhanced run method with meta-agent features

        Args:
            query: User query string

        Returns:
            Processed response with enhanced error handling
        """
        self.query_count += 1
        self.last_query_time = time.time()

        # Preprocess query
        processed_query = self._preprocess_query(query)

        try:
            # Call hybrid agent
            result = self.hybrid_agent.run(processed_query)

            # Postprocess result
            enhanced_result = self._postprocess_result(result, query)

            return enhanced_result

        except Exception as e:
            self.logger.error(f"Meta-agent error handling: {e}")
            return self._handle_error(e, query)

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better handling"""
        # Clean up whitespace
        query = query.strip()

        # Handle common shortcuts
        shortcuts = {
            "ls": "list files",
            "status": "system status",
            "health": "health check",
            "help": "What can you help me with? Show me available capabilities."
        }

        if query.lower() in shortcuts:
            query = shortcuts[query.lower()]
            self.logger.debug(f"Expanded shortcut to: {query}")

        return query

    def _postprocess_result(self, result: str, original_query: str) -> str:
        """Postprocess result for enhanced presentation"""
        # Add helpful context for certain types of results
        if "No files found" in result and "pattern" in result:
            result += "\n\n💡 Tip: Try 'list files' to see all available files."

        elif "Error:" in result and "file not found" in result.lower():
            result += "\n\n💡 Tip: Use 'list files' to see available files, or check the file path."

        return result

    def _handle_error(self, error: Exception, query: str) -> str:
        """Enhanced error handling with helpful suggestions"""
        error_msg = str(error)

        if "rate limit" in error_msg.lower() or "429" in error_msg:
            return "Rate limit reached. Please wait 30 seconds before trying again."

        elif "recursion" in error_msg.lower():
            return "Query too complex. Try a simpler request or break it into smaller parts."

        elif "timeout" in error_msg.lower():
            return "Request timed out. Try a smaller file or simpler analysis."

        elif "api" in error_msg.lower() and "key" in error_msg.lower():
            return "API key issue. Please check your OpenAI API key configuration."

        else:
            return f"I encountered an unexpected error: {error_msg[:200]}{'...' if len(error_msg) > 200 else ''}"

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive meta-agent stats"""
        hybrid_stats = self.hybrid_agent.get_stats()

        uptime = time.time() - self.start_time

        meta_stats = {
            "uptime_hours": round(uptime / 3600, 2),
            "uptime_seconds": round(uptime, 2),
            "total_queries": self.query_count,
            "last_query_time": self.last_query_time,
            "system_type": "Production Meta-Agent with HybridAgent",
            "config": {
                "max_iterations": self.config.agent.max_iterations,
                "max_execution_time": self.config.agent.max_execution_time,
                "verbose": self.config.agent.verbose
            }
        }

        return {**hybrid_stats, **meta_stats}

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status information"""
        return {
            "agent_type": "MetaAgent with HybridAgent",
            "stats": self.get_stats(),
            "data_directory": self.config.data.data_dir,
            "servers": {
                "a2a_url": self.config.a2a.url,
                "mcp_url": self.config.mcp.url
            },
            "monitoring_enabled": self.config.monitoring.enabled
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the meta-agent system"""
        try:
            # Test basic functionality
            test_response = self.run("system status")
            functional = "error" not in test_response.lower()

            return {
                "functional": functional,
                "response_time": time.time() - self.last_query_time if self.last_query_time else 0,
                "queries_processed": self.query_count,
                "uptime_hours": round((time.time() - self.start_time) / 3600, 2),
                "last_error": None if functional else "Test query failed"
            }

        except Exception as e:
            return {
                "functional": False,
                "error": str(e),
                "queries_processed": self.query_count,
                "uptime_hours": round((time.time() - self.start_time) / 3600, 2)
            }


class MetaAgentFactory:
    """
    Factory for creating meta agents

    Wraps hybrid agents with production-ready features:
    - Enhanced error handling
    - Performance monitoring
    - Query preprocessing
    - System integration
    """

    def __init__(self, hybrid_agent: HybridAgent, config: Config):
        self.hybrid_agent = hybrid_agent
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create(self) -> MetaAgent:
        """Create a meta agent with production features"""
        self.logger.info("Creating meta agent...")

        # Validate hybrid agent
        if not self.hybrid_agent:
            raise ValueError("Hybrid agent is required")

        if not hasattr(self.hybrid_agent, 'run'):
            raise ValueError("Hybrid agent must have 'run' method")

        # Create meta agent
        meta_agent = MetaAgent(self.hybrid_agent, self.config)

        # Verify functionality
        self._verify_meta_agent(meta_agent)

        self.logger.info("Meta agent created successfully")
        return meta_agent

    def _verify_meta_agent(self, meta_agent: MetaAgent):
        """Verify meta agent functionality"""
        try:
            # Test basic functionality
            health = meta_agent.health_check()

            if not health.get("functional", False):
                self.logger.warning("Meta agent health check failed")

            # Test stats collection
            stats = meta_agent.get_stats()
            if not isinstance(stats, dict):
                raise ValueError("Stats must return dictionary")

            self.logger.debug("Meta agent verification passed")

        except Exception as e:
            self.logger.error(f"Meta agent verification failed: {e}")
            # Don't fail creation for verification issues


if __name__ == "__main__":
    # Test meta agent factory
    from config.settings import load_config
    from .hybrid_agent import HybridAgent


    # Mock hybrid agent for testing
    class MockHybridAgent:
        def run(self, query):
            return f"Mock response to: {query}"

        def get_stats(self):
            return {"test": True}


    config = load_config()
    mock_hybrid = MockHybridAgent()

    factory = MetaAgentFactory(mock_hybrid, config)
    meta_agent = factory.create()

    print("Meta agent created successfully")
    print(f"Stats: {meta_agent.get_stats()}")
    print(f"Test response: {meta_agent.run('test query')}")