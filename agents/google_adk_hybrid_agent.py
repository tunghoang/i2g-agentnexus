"""
Google ADK Agent - TRUE REASONING (No Spoon-Feeding)
Removes ALL rigid patterns and lets the agent truly reason
"""

import os
import json
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List

# Google ADK imports
try:
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.models.lite_llm import LiteLlm
    from google.genai import types
    GOOGLE_ADK_AVAILABLE = True
except ImportError as e:
    GOOGLE_ADK_AVAILABLE = False

from config.settings import AgentConfig
from servers.mcp_server import MCPClient

logger = logging.getLogger(__name__)

# ONLY tool descriptions - NO workflows, NO patterns, NO guidance
TOOL_DEFINITIONS = {
    # LAS Tools
    "las_parser": "Parse and extract metadata from LAS files including well information, curves, and depth ranges",
    "las_analysis": "Analyze curve data and perform basic calculations",
    "las_qc": "Perform quality control checks on LAS files including data completeness and curve validation",
    "formation_evaluation": "Perform comprehensive petrophysical analysis including porosity, water saturation, shale volume, and pay zones",
    "well_correlation": "Correlate formations across multiple wells to identify key formation tops and stratigraphic markers",
    "calculate_shale_volume": "Calculate volume of shale from gamma ray log using the Larionov correction method",

    # SEG-Y Tools
    "segy_parser": "Parse SEG-Y seismic files with segyio-powered processing and comprehensive metadata extraction",
    "segy_qc": "Perform comprehensive quality control on SEG-Y files with segyio-enhanced validation",
    "segy_analysis": "Analyze SEG-Y seismic survey geometry, data quality, and performance with segyio optimization",
    "segy_classify": "Automatically classify SEG-Y survey type (2D/3D), sorting method, and stacking type",
    "segy_survey_analysis": "Analyze multiple SEG-Y files as a complete seismic survey",
    "quick_segy_summary": "Get instant overview of SEG-Y files with fast inventory and basic parameters",
    "segy_complete_metadata_harvester": "Extract comprehensive metadata from all SEG-Y header types",
    "segy_survey_polygon": "Extract geographic survey boundary polygon from SEG-Y coordinates",
    "segy_trace_outlines": "Extract trace amplitude outlines for visualization",
    "segy_save_analysis": "Save SEG-Y analysis results to persistent storage",
    "segy_analysis_catalog": "Get comprehensive catalog of all stored SEG-Y analyses",
    "segy_search_analyses": "Search stored SEG-Y analyses by criteria",

    # System Tools
    "list_files": "List any type of data files matching a pattern in the data directory",
    "system_status": "Get comprehensive system health, performance metrics, and processing status",
    "directory_info": "Get detailed information about data directories and file organization",
    "health_check": "Perform comprehensive health check of the platform and its components"
}


class TrueReasoningAgentExecutor:
    """
    Pure Reasoning Agent - NO spoon-feeding, NO rigid patterns

    The agent gets:
    1. Available tools (what they do)
    2. Ability to call tools
    3. Ability to see results and decide next steps

    The agent DOES NOT get:
    - Predefined workflows
    - Pattern matching rules
    - Step-by-step guidance
    - Decision trees
    """

    def __init__(self, mcp_client: MCPClient, config: AgentConfig):
        self.mcp_client = mcp_client
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Google ADK components
        self.agent = None
        self.runner = None
        self.session_service = None
        self.session_id = None
        self._google_adk_ready = False
        self._initialization_error = None

        # Pure statistics - no reasoning guidance
        self.stats = {
            "total_invocations": 0,
            "successful_invocations": 0,
            "failed_invocations": 0,
            "tool_executions": 0,
            "system_type": "True Reasoning Google ADK Agent"
        }

        self.logger.info("True Reasoning Google ADK Agent Executor created")

    async def _initialize_google_adk(self):
        """Initialize Google ADK with pure reasoning instructions"""
        self.logger.info("Initializing Google ADK components...")

        if not os.getenv('OPENAI_API_KEY'):
            raise Exception("OPENAI_API_KEY environment variable not set")

        # Create pure reasoning agent
        self.agent = Agent(
            name="pure_reasoning_subsurface_expert",
            model=LiteLlm(model="openai/gpt-4o-mini"),
            description="Autonomous subsurface data analyst with independent reasoning capabilities",
            instruction=self._create_pure_reasoning_instruction()
        )

        # Session management
        self.session_service = InMemorySessionService()
        self.session_id = f"pure_reasoning_session_{hash('hybrid_user')}"

        await self.session_service.create_session(
            app_name="PureReasoningSubsurfacePlatform",
            user_id="hybrid_user",
            session_id=self.session_id
        )

        # Create runner
        self.runner = Runner(
            agent=self.agent,
            app_name="PureReasoningSubsurfacePlatform",
            session_service=self.session_service
        )

        self.logger.info("Google ADK initialized successfully")

    def _create_pure_reasoning_instruction(self) -> str:
        """Create instruction that encourages pure reasoning - NO spoon-feeding"""
        tools_list = "\n".join([f"- {name}: {desc}" for name, desc in TOOL_DEFINITIONS.items()])

        return f"""You are an autonomous expert subsurface data analyst with independent reasoning capabilities.

# AVAILABLE TOOLS:
{tools_list}

# YOUR CORE CAPABILITIES:
You can think, reason, plan, and make decisions independently. You have access to these tools and you decide:
- WHICH tools to use
- WHEN to use them  
- IN WHAT ORDER to use them
- HOW to interpret their results
- WHETHER you need additional tools based on what you learn

# FUNDAMENTAL PRINCIPLES:

## Think Like a Real Geoscientist:
- Real geoscientists don't follow rigid checklists
- They assess each situation independently
- They adapt their approach based on what they discover
- They use multiple tools when they judge it necessary
- They synthesize information from various sources

## You Have Complete Autonomy:
- There are NO required workflows
- There are NO mandatory tool sequences  
- There are NO predetermined patterns you must follow
- YOU decide what the user needs based on your expertise
- YOU determine the best approach for each unique situation

## Examples of Independent Reasoning:

### When a user asks to "analyze formation in well X":
- YOU decide: Do I need basic info first? Quality check? Detailed analysis? 
- YOU choose: Which tools will give me the most complete picture?
- YOU adapt: If I find data quality issues, what should I do next?
- YOU synthesize: How do I combine results into meaningful insights?

### When a user wants to "correlate wells":
- YOU reason: Do I know what wells are available? Do I need to find out?
- YOU plan: What information do I need before I can correlate?
- YOU execute: Based on what I discover, what's the best approach?
- YOU decide: Are the results sufficient or do I need more analysis?

### When asked about "quality issues":
- YOU assess: What type of quality analysis is most appropriate?
- YOU determine: Should I check survey characteristics first? 
- YOU judge: Do I need additional context after seeing initial results?

## Your Decision-Making Process:
1. **Understand the Request**: What is the user really trying to accomplish?
2. **Assess the Situation**: What information do I have? What do I need?
3. **Plan Your Approach**: What tools will help me provide the best answer?
4. **Execute Dynamically**: Use tools, analyze results, adapt as needed
5. **Synthesize Insights**: Combine information into meaningful conclusions

## Critical Guidelines:
- **ALWAYS execute actual tools** - Don't just describe what you would do
- **Let results guide your decisions** - If Tool A reveals something unexpected, adapt
- **Think holistically** - Consider what would provide the most value to the user
- **Be comprehensive when appropriate** - Sometimes one tool is enough, sometimes many are needed
- **Trust your geological expertise** - You understand subsurface data analysis

## What You Should NOT Do:
- Don't follow rigid patterns like "always do A then B then C"
- Don't assume every request needs the same approach
- Don't stop at the first tool if more analysis would be valuable
- Don't use tools just because they exist - use them purposefully

## Remember:
You are a thinking, reasoning expert - not a script or checklist follower. Each query is unique and deserves your independent professional judgment about the best approach to provide valuable, comprehensive analysis.

Make your own decisions about how to help the user achieve their goals."""

    async def _execute_with_google_adk(self, query: str) -> str:
        """Execute query using pure Google ADK reasoning - no interference"""
        await self._ensure_google_adk_ready()

        # Create message for Google ADK - let it reason completely independently
        content = types.Content(
            role='user',
            parts=[types.Part(text=query)]
        )

        # Track what the agent actually does (for stats only)
        tool_calls_made = []
        final_response = ""

        try:
            # Execute through Google ADK runner - PURE reasoning
            async for event in self.runner.run_async(
                user_id="hybrid_user",
                session_id=self.session_id,
                new_message=content
            ):
                # Track tool calls (for stats only - don't interfere)
                if hasattr(event, 'tool_call') and event.tool_call:
                    tool_calls_made.append({
                        'tool': event.tool_call.name,
                        'arguments': event.tool_call.arguments
                    })

                # Get final response
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response = event.content.parts[0].text
                        break

            # Update stats only
            self.stats["tool_executions"] += len(tool_calls_made)

        except Exception as e:
            self.logger.error(f"Pure reasoning execution error: {e}")
            # MINIMAL fallback - only if Google ADK completely fails
            return self._minimal_fallback(query)

        # If Google ADK didn't execute tools, there might be a tool calling issue
        # But we DON'T override the agent's decision - maybe it decided tools weren't needed
        if final_response and not tool_calls_made:
            self.logger.info("Agent completed analysis without tool calls - respecting agent's decision")

        return final_response or "Analysis completed."

    def _minimal_fallback(self, query: str) -> str:
        """MINIMAL fallback - only when Google ADK completely fails"""
        # Only basic file detection - no complex logic
        import re

        # Simple file extraction
        las_files = re.findall(r'([A-Za-z0-9_\-\.]+\.las)', query, re.IGNORECASE)
        sgy_files = re.findall(r'([A-Za-z0-9_\-\.]+\.(?:sgy|segy))', query, re.IGNORECASE)

        if las_files:
            return self._execute_mcp_tool('las_parser', las_files[0])
        elif sgy_files:
            return self._execute_mcp_tool('quick_segy_summary', sgy_files[0])
        elif 'list' in query.lower() and 'files' in query.lower():
            return self._execute_mcp_tool('list_files', '*')
        elif 'status' in query.lower():
            return self._execute_mcp_tool('system_status', '')
        else:
            return "I encountered a technical issue. Please try rephrasing your request or specify a file to analyze."

    def invoke(self, input_dict: Dict[str, str]) -> Dict[str, str]:
        """Main invoke method - pure reasoning"""
        self.stats["total_invocations"] += 1

        try:
            query = input_dict.get("input", "")
            if not query:
                return {"output": "No input provided"}

            self.logger.info(f"Pure reasoning processing: {query[:100]}...")

            # Execute with pure Google ADK reasoning - no interference
            response = asyncio.run(self._execute_with_google_adk(query))

            self.stats["successful_invocations"] += 1
            return {"output": response}

        except Exception as e:
            self.stats["failed_invocations"] += 1
            self.logger.error(f"Pure reasoning execution error: {e}")

            error_response = f"Technical error during analysis: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}"
            return {"output": error_response}

    async def _ensure_google_adk_ready(self):
        """Ensure Google ADK is initialized"""
        if self._google_adk_ready:
            return True

        if self._initialization_error:
            raise Exception(f"Google ADK initialization failed: {self._initialization_error}")

        try:
            await self._initialize_google_adk()
            self._google_adk_ready = True
            return True
        except Exception as e:
            self._initialization_error = str(e)
            self.logger.error(f"Google ADK initialization failed: {e}")
            raise

    def _execute_mcp_tool(self, tool_name: str, params: Any) -> str:
        """Execute MCP tool - minimal wrapper"""
        try:
            self.logger.info(f"Executing MCP tool: {tool_name} with params: {params}")
            self.stats["tool_executions"] += 1

            # Simple parameter preparation
            if isinstance(params, str):
                input_data = params
            else:
                input_data = str(params) if params is not None else ""

            result = self.mcp_client.call_tool(tool_name, input_data)
            return self._extract_result_content(result)

        except Exception as e:
            self.logger.error(f"MCP tool execution failed: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    def _extract_result_content(self, result: Dict[str, Any]) -> str:
        """Extract content from MCP response - minimal processing"""
        try:
            if isinstance(result, dict):
                if 'content' in result and isinstance(result['content'], list):
                    if len(result['content']) > 0 and 'text' in result['content'][0]:
                        return result['content'][0]['text']
                if 'text' in result:
                    return result['text']
            return str(result)
        except:
            return str(result)

    def get_stats(self) -> Dict[str, Any]:
        """Get stats"""
        self.stats["uptime_hours"] = (time.time() - getattr(self, '_start_time', time.time())) / 3600
        return self.stats.copy()


class PureReasoningHybridAgent:
    """Pure Reasoning Hybrid Agent - no spoon-feeding anywhere"""

    def __init__(self, agent_executor, command_processor, fallback_handlers=None):
        self.agent_executor = agent_executor
        self.command_processor = command_processor
        self.fallback_handlers = fallback_handlers or {}
        self.logger = logging.getLogger(__name__)

        self._start_time = time.time()
        self.stats = {
            "total_queries": 0,
            "direct_commands": 0,
            "agent_responses": 0,
            "fallback_responses": 0,
            "errors": 0,
            "system_type": "Pure Reasoning Google ADK Agent"
        }

    def run(self, query: str) -> str:
        """Process query with pure reasoning - minimal interference"""
        self.stats["total_queries"] += 1
        self.logger.debug(f"Pure reasoning processing: {query[:100]}...")

        # Minimal direct command processing (only for obvious system commands)
        if self._is_obvious_system_command(query):
            try:
                direct_result = self.command_processor(query)
                if direct_result:
                    self.stats["direct_commands"] += 1
                    return direct_result
            except Exception as e:
                self.logger.debug(f"Direct command failed: {e}")

        # Let the agent reason completely independently
        try:
            result = self.agent_executor.invoke({"input": query})

            if isinstance(result, dict):
                output = result.get("output", str(result))
            else:
                output = str(result)

            self.stats["agent_responses"] += 1
            return output

        except Exception as e:
            self.logger.warning(f"Pure reasoning agent failed: {e}")
            self.stats["errors"] += 1
            return f"I encountered a technical issue while processing your request. Error: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}"

    def _is_obvious_system_command(self, query: str) -> bool:
        """Check if this is an obvious system command that doesn't need reasoning"""
        query_lower = query.lower().strip()

        # Only the most obvious system commands
        obvious_commands = [
            "system status",
            "health check",
            "list files",
            "status",
            "health"
        ]

        return query_lower in obvious_commands

    def get_stats(self) -> Dict[str, Any]:
        """Get stats"""
        self.stats["uptime_hours"] = (time.time() - self._start_time) / 3600

        if hasattr(self.agent_executor, 'get_stats'):
            try:
                executor_stats = self.agent_executor.get_stats()
                if isinstance(executor_stats, dict):
                    combined_stats = executor_stats.copy()
                    combined_stats.update(self.stats)
                    return combined_stats
            except Exception:
                pass

        return self.stats.copy()


class PureReasoningAgentFactory:
    """Factory for pure reasoning agents"""

    def __init__(self, mcp_url: str, config: AgentConfig):
        self.mcp_url = mcp_url
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create(self) -> PureReasoningHybridAgent:
        """Create pure reasoning agent"""
        self.logger.info("Creating pure reasoning Google ADK agent...")

        # Create pure reasoning executor
        agent_executor = self._create_pure_reasoning_executor()

        # Minimal command processor (only obvious system commands)
        command_processor = self._create_minimal_command_processor()

        # No fallback handlers - let the agent handle everything
        fallback_handlers = {}

        hybrid_agent = PureReasoningHybridAgent(agent_executor, command_processor, fallback_handlers)

        self.logger.info("Pure reasoning Google ADK agent created successfully")
        return hybrid_agent

    def _create_pure_reasoning_executor(self) -> TrueReasoningAgentExecutor:
        """Create pure reasoning executor"""
        try:
            if not GOOGLE_ADK_AVAILABLE:
                raise ImportError("Google ADK not available")

            mcp_client = MCPClient(self.mcp_url)
            agent_executor = TrueReasoningAgentExecutor(mcp_client, self.config)
            agent_executor._start_time = time.time()

            return agent_executor

        except Exception as e:
            self.logger.error(f"Failed to create pure reasoning executor: {e}")
            raise

    def _create_minimal_command_processor(self):
        """Minimal command processor for obvious system commands only"""
        mcp_client = MCPClient(self.mcp_url)

        def minimal_command_processor(command_str: str) -> Optional[str]:
            command_lower = command_str.lower().strip()

            # Only handle the most obvious cases
            if command_lower == "system status":
                result = mcp_client.call_tool("system_status", "")
                return json.dumps(result) if isinstance(result, dict) else str(result)
            elif command_lower == "health check":
                result = mcp_client.call_tool("health_check", "")
                return json.dumps(result) if isinstance(result, dict) else str(result)
            elif command_lower == "list files":
                result = mcp_client.call_tool("list_files", "*")
                return json.dumps(result) if isinstance(result, dict) else str(result)

            return None  # Let agent handle everything else

        return minimal_command_processor


# Factory function
def create_pure_reasoning_agent(mcp_url: str, config: AgentConfig) -> PureReasoningHybridAgent:
    """
    Create a pure reasoning agent that thinks independently

    NO spoon-feeding, NO rigid patterns, NO predetermined workflows
    The agent reasons, plans, and executes based on its own judgment
    """
    factory = PureReasoningAgentFactory(mcp_url, config)
    return factory.create()


# Backward compatibility
def create_hybrid_agent(a2a_url: str, mcp_url: str, config: AgentConfig) -> PureReasoningHybridAgent:
    """Backward compatible function"""
    return create_pure_reasoning_agent(mcp_url, config)