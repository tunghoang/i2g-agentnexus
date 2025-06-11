"""
Hybrid Agent Factory
Creates hybrid agents that combine LangChain and direct MCP access
"""

import logging
import json
from typing import Optional, Dict, Any, List

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain.prompts import PromptTemplate

# A2A imports
from python_a2a.langchain import to_langchain_agent, to_langchain_tool

from config.settings import AgentConfig
from servers.mcp_server import MCPClient

# GLOBAL TOOL DEFINITIONS - moved out of the class
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


class HybridAgent:
    """
    Hybrid agent that combines LangChain ReAct agent with direct command processing

    This agent:
    1. First tries direct command processing for efficiency
    2. Falls back to LangChain ReAct agent for complex queries
    3. Provides enhanced error handling and response formatting
    """

    def __init__(self, agent_executor, command_processor, fallback_handlers=None):
        self.agent_executor = agent_executor
        self.command_processor = command_processor
        self.fallback_handlers = fallback_handlers or {}
        self.logger = logging.getLogger(__name__)

        # Statistics tracking
        self.stats = {
            "total_queries": 0,
            "direct_commands": 0,
            "agent_responses": 0,
            "fallback_responses": 0,
            "errors": 0
        }

    def run(self, query: str) -> str:
        """
        Process a query using the hybrid approach

        Args:
            query: User query string

        Returns:
            Processed response string
        """
        self.stats["total_queries"] += 1
        self.logger.debug(f"Processing query: {query[:100]}...")

        # Step 1: Try direct command processing first
        try:
            direct_result = self.command_processor(query)
            if direct_result:
                self.stats["direct_commands"] += 1
                self.logger.debug("Query handled by direct command processor")
                return direct_result
        except Exception as e:
            self.logger.debug(f"Direct command processing failed: {str(e)}")

        # Step 2: Use ReAct agent
        try:
            result = self.agent_executor.invoke({"input": query})

            # Handle ReAct agent response structure
            if isinstance(result, dict):
                output = result.get("output", str(result))
            else:
                output = str(result)

            self.stats["agent_responses"] += 1
            self.logger.debug("Query handled by ReAct agent")

            # Post-process response for better formatting
            formatted_output = self._format_agent_response(output)
            return formatted_output

        except Exception as e:
            self.logger.warning(f"ReAct agent processing failed: {str(e)}")
            return self._try_fallback_handlers(query, str(e))

    def _format_agent_response(self, response: str) -> str:
        """Format agent response for better readability"""
        if self._is_json_response(response):
            return self._format_json_response(response)
        return response

    def _is_json_response(self, response: str) -> bool:
        """Check if response is raw JSON"""
        if isinstance(response, str):
            stripped = response.strip()
            return (stripped.startswith('{') and stripped.endswith('}')) or \
                (stripped.startswith('[') and stripped.endswith(']'))
        return False

    def _format_json_response(self, json_str: str) -> str:
        """Convert JSON response to human-readable format"""
        try:
            data = json.loads(json_str)

            if isinstance(data, dict):
                # Special formatting for different response types
                if 'file_processed' in data and 'survey_type' in data:
                    return self._format_segy_analysis(data)
                elif 'quality_rating' in data:
                    return self._format_quality_analysis(data)
                elif 'well_name' in data:
                    return self._format_las_analysis(data)

            return json_str

        except json.JSONDecodeError:
            return json_str

    def _format_segy_analysis(self, data: Dict[str, Any]) -> str:
        """Format SEG-Y analysis results"""
        return f"""
## SEG-Y Analysis Results

**File:** {data.get('file_processed', 'Unknown')}

**Survey Characteristics:**
- Survey Type: {data.get('survey_type', 'Unknown')}
- Stack Type: {data.get('stack_type', 'Unknown')}
- Quality Rating: {data.get('quality_rating', 'Unknown')}

**Technical Details:**
- Total Traces: {data.get('total_traces', 0):,}
- Sample Rate: {data.get('sample_rate_ms', 0)} ms
- File Size: {data.get('file_size_mb', 0)} MB
- Trace Length: {data.get('trace_length_ms', 0)} ms

**Quality Assessment:**
{self._format_quality_issues(data.get('quality_analysis', {}))}

**Processing Notes:**
{chr(10).join(data.get('processing_notes', []))}
"""

    def _format_quality_analysis(self, data: Dict[str, Any]) -> str:
        """Format quality analysis results"""
        return f"""
## Quality Analysis Results

**Overall Rating:** {data.get('quality_rating', 'Unknown')}

**Key Metrics:**
- Dynamic Range: {data.get('dynamic_range_db', 'N/A')} dB
- Signal-to-Noise: {data.get('signal_to_noise', 'N/A')}
- Zero Percentage: {data.get('zero_percentage', 'N/A')}%

**Issues and Recommendations:**
{self._format_quality_issues(data)}
"""

    def _format_las_analysis(self, data: Dict[str, Any]) -> str:
        """Format LAS analysis results"""
        return f"""
## Well Log Analysis Results

**Well:** {data.get('well_name', 'Unknown')}
**File:** {data.get('file_processed', 'Unknown')}

**Formation Evaluation:**
- Average Porosity: {data.get('average_porosity', 'N/A')}%
- Water Saturation: {data.get('water_saturation', 'N/A')}%
- Shale Volume: {data.get('shale_volume', 'N/A')}%

**Pay Zones:** {len(data.get('pay_zones', []))} identified

**Quality Assessment:** {data.get('quality_rating', 'Unknown')}
"""

    def _format_quality_issues(self, quality_data: Dict[str, Any]) -> str:
        """Format quality issues and warnings"""
        output = []

        issues = quality_data.get('issues', [])
        if issues:
            output.append("**Issues Found:**")
            for issue in issues:
                output.append(f"- {issue}")

        warnings = quality_data.get('warnings', [])
        if warnings:
            output.append("**Warnings:**")
            for warning in warnings:
                output.append(f"- {warning}")

        if not issues and not warnings:
            output.append("No quality issues detected.")

        return chr(10).join(output)

    def _try_fallback_handlers(self, query: str, original_error: str) -> str:
        """Enhanced fallback handler with better error parsing"""
        query_lower = query.lower()

        # First, try to extract useful information from parsing errors
        if "Could not parse LLM output" in str(original_error):
            extracted_content = self._extract_content_from_parsing_error(str(original_error))
            if extracted_content:
                return extracted_content

        # Try fallback handlers
        for handler_name, handler_config in self.fallback_handlers.items():
            if all(keyword in query_lower for keyword in handler_config["keywords"]):
                try:
                    result = handler_config["handler"](query)
                    if result:
                        self.stats["fallback_responses"] += 1
                        self.logger.debug(f"Query handled by fallback handler: {handler_name}")
                        return result
                except Exception as fallback_error:
                    self.logger.warning(f"Fallback handler {handler_name} failed: {fallback_error}")

        self.stats["errors"] += 1

        # Enhanced error message with more context
        if "Could not parse LLM output" in str(original_error):
            return f"I processed your request but encountered a formatting issue. The analysis was completed but couldn't be properly formatted. Raw details: {str(original_error)[:200]}..."

        return f"Sorry, I encountered an error and couldn't process your request: {original_error}"

    def _extract_content_from_parsing_error(self, error_str: str) -> Optional[str]:
        """Extract usable content from parsing errors"""
        try:
            if "`" in error_str:
                start_idx = error_str.find("`") + 1
                end_idx = error_str.rfind("`")
                if start_idx > 0 and end_idx > start_idx:
                    extracted = error_str[start_idx:end_idx]

                    if any(keyword in extracted.lower() for keyword in
                           ['analysis', 'survey', 'file', 'quality', 'results', 'geometry']):
                        return f"Analysis completed:\n\n{extracted}"

            return None
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return self.stats.copy()


class HybridAgentFactory:
    """
    Factory for creating hybrid agents

    Handles the complex setup of:
    1. LangChain ReAct agent with tools
    2. Direct command processor
    3. Fallback handlers
    4. Error handling and formatting
    """

    def __init__(self, a2a_url: str, mcp_url: str, config: AgentConfig):
        self.a2a_url = a2a_url
        self.mcp_url = mcp_url
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create(self) -> HybridAgent:
        """Create a complete hybrid agent"""
        self.logger.info("Creating hybrid agent...")

        # Step 1: Create LangChain agent from A2A
        langchain_agent = self._create_langchain_agent()

        # Step 2: Create LangChain tools from MCP
        tools = self._create_langchain_tools()

        # Step 3: Create ReAct agent executor
        agent_executor = self._create_react_agent_executor(tools)

        # Step 4: Create command processor
        command_processor = self._create_command_processor()

        # Step 5: Create fallback handlers
        fallback_handlers = self._create_fallback_handlers()

        # Step 6: Combine into hybrid agent
        hybrid_agent = HybridAgent(agent_executor, command_processor, fallback_handlers)

        self.logger.info("Hybrid agent created successfully")
        return hybrid_agent

    def _create_langchain_agent(self):
        """Create LangChain agent from A2A server"""
        try:
            langchain_agent = to_langchain_agent(self.a2a_url)
            self.logger.debug("LangChain agent created from A2A")
            return langchain_agent
        except Exception as e:
            self.logger.error(f"Failed to create LangChain agent: {e}")
            raise

    def _create_langchain_tools(self) -> List[Tool]:
        """Create LangChain tools from MCP server"""
        tools = []

        # Use the global tool definitions
        tool_definitions = TOOL_DEFINITIONS

        # Create tools using MCP client
        mcp_client = MCPClient(self.mcp_url)

        for tool_name, description in tool_definitions.items():
            try:
                def create_tool_function(tool_name):
                    def tool_function(input_text: str) -> str:
                        """Execute MCP tool and return results"""
                        try:
                            if not input_text.strip():
                                return f"Error: No input provided for {tool_name}"

                            # Call MCP tool
                            result = mcp_client.call_tool(tool_name, input_text)

                            # Handle response
                            if isinstance(result, dict):
                                if "error" in result:
                                    return f"Tool error: {result['error']}"
                                elif "text" in result:
                                    return str(result["text"])
                                else:
                                    return json.dumps(result, indent=2)
                            else:
                                return str(result)
                        except Exception as e:
                            return f"Error calling {tool_name}: {str(e)}"

                    return tool_function

                # Create LangChain Tool
                langchain_tool = Tool(
                    name=tool_name,
                    func=create_tool_function(tool_name),
                    description=description
                )
                tools.append(langchain_tool)
                self.logger.debug(f"Created tool: {tool_name}")

            except Exception as e:
                self.logger.warning(f"Failed to create tool {tool_name}: {e}")

        self.logger.info(f"Created {len(tools)} LangChain tools")
        return tools

    def _create_react_agent_executor(self, tools: List[Tool]) -> AgentExecutor:
        """Create ReAct agent executor"""
        try:
            # Create LLM
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=self.config.temperature,
                max_tokens=1500,
                request_timeout=30,
                max_retries=2,
                model_kwargs={
                    "stop": ["Observation:", "\nObservation:", "Observation:\n"]
                }
            )

            # Get ReAct prompt
            try:
                react_prompt = hub.pull("hwchase17/react")
                self.logger.debug("Pulled ReAct prompt from LangChain hub")
            except Exception as e:
                self.logger.warning(f"Failed to pull prompt from hub: {e}")
                react_prompt = self._create_fallback_react_prompt()

            # Enhance prompt with system context
            enhanced_prompt = self._enhance_react_prompt(react_prompt)

            # Create ReAct agent
            agent = create_react_agent(
                llm=llm,
                tools=tools,
                prompt=enhanced_prompt
            )

            # Create memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=self.config.verbose,
                handle_parsing_errors=True,
                max_iterations=15,
                max_execution_time=300,
                return_intermediate_steps=False
            )

            self.logger.info("ReAct agent executor created")
            return agent_executor

        except Exception as e:
            self.logger.error(f"Failed to create ReAct agent executor: {e}")
            raise

    def _create_fallback_react_prompt(self) -> PromptTemplate:
        """Fallback ReAct prompt if hub is unavailable"""
        template = """You are a specialized subsurface data analysis assistant.
Answer the following questions as best you can. You have access to the following tools:
TOOLS:
{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: I need to analyze this request
Action: [tool name]
Action Input: [input for the tool]
Observation: [result of the action]
Thought: I now have all the information needed
Final Answer: [your comprehensive response to the user]

IMPORTANT: Always end with "Final Answer:" followed by your complete response.

Previous conversation:
{chat_history}

Question: {input}
{agent_scratchpad}
"""

        return PromptTemplate.from_template(template)

    def _enhance_react_prompt(self, react_prompt) -> PromptTemplate:
        """Enhance ReAct prompt with system context"""
        system_context = """
You are an expert subsurface data analyst specializing in well logs (LAS files) and seismic data (SEG-Y files).

Key Guidelines:
1. Always provide clear, formatted responses instead of raw JSON
2. When analyzing files, provide meaningful interpretations of the technical data
3. For quality assessments, explain what the ratings mean in practical terms
4. When processing multiple files, provide comparative analysis
5. If file patterns are used (like *.segy), process all matching files systematically

Technical Expertise:
- Petrophysical analysis and formation evaluation
- Seismic data processing and interpretation
- Quality control and data validation
- Multi-well correlation and pattern recognition

Response Format:
- Provide executive summaries for complex analyses
- Use clear headings and bullet points for readability
- Include technical details but explain their significance
- Highlight any quality issues or recommendations
"""

        # Get the original template
        if hasattr(react_prompt, 'template'):
            original_template = react_prompt.template
        else:
            original_template = str(react_prompt)

        # Insert system context before the main instructions
        enhanced_template = system_context + "\n\n" + original_template

        # Create new prompt with enhanced template
        return PromptTemplate(
            template=enhanced_template,
            input_variables=react_prompt.input_variables if hasattr(react_prompt, 'input_variables')
            else ["input", "agent_scratchpad", "tools", "tool_names"]
        )

    def _create_command_processor(self):
        """Create direct command processor for efficiency"""
        mcp_client = MCPClient(self.mcp_url)

        def enhanced_direct_command_processor(command_str: str) -> Optional[str]:
            """Intent-based command processor"""
            try:
                command_lower = command_str.lower()

                # INTENT: List files
                if any(keyword in command_lower for keyword in ["list", "show", "display"]) and \
                        any(keyword in command_lower for keyword in ["files", "data", "available"]):

                    # Determine file type from context
                    if any(keyword in command_lower for keyword in ["seismic", "segy", "sgy"]):
                        pattern = "*.sgy"
                    elif any(keyword in command_lower for keyword in ["well", "las", "log"]):
                        pattern = "*.las"
                    else:
                        pattern = "*"  # All files

                    self.logger.debug(f"List intent detected - using pattern: '{pattern}'")
                    result = mcp_client.call_tool("list_files", pattern)
                    return json.dumps(result) if isinstance(result, dict) else str(result)

                # INTENT: System status
                if any(keyword in command_lower for keyword in ["status", "health", "system"]):
                    self.logger.debug("System status intent detected")
                    result = mcp_client.call_tool("system_status", "")
                    return json.dumps(result) if isinstance(result, dict) else str(result)

                # No direct intent recognized
                return None

            except Exception as e:
                self.logger.warning(f"Error in command processor: {str(e)}")
                return None

        return enhanced_direct_command_processor

    def _create_fallback_handlers(self) -> Dict[str, Dict[str, Any]]:
        """Create fallback handlers for error recovery"""
        return {
            "las_metadata": {
                "keywords": ["metadata", ".las", "file"],
                "handler": lambda query: self._extract_and_process_file(query, ".las", "las_parser")
            },
            "segy_classify": {
                "keywords": ["classify", ".sgy", "seismic"],
                "handler": lambda query: self._extract_and_process_file(query, [".sgy", ".segy"], "segy_classify")
            }
        }

    def _extract_and_process_file(self, query: str, file_extensions, tool_name: str):
        """Extract filename from query and process with specified tool"""
        if isinstance(file_extensions, str):
            file_extensions = [file_extensions]

        mcp_client = MCPClient(self.mcp_url)

        for word in query.split():
            for ext in file_extensions:
                if word.lower().endswith(ext):
                    result = mcp_client.call_tool(tool_name, word)
                    return json.dumps(result) if isinstance(result, dict) else str(result)
        return None


# FIXED: Now this function can access TOOL_DEFINITIONS
def create_adaptive_agent(mcp_server):
    """
    Create adaptive agent that replaces your ReAct agent
    Uses the global TOOL_DEFINITIONS
    """
    from .adaptive_tool_executor import create_flexible_agent
    return create_flexible_agent(mcp_server, TOOL_DEFINITIONS)


# Alternative function that creates a full hybrid agent
def create_hybrid_agent(a2a_url: str, mcp_url: str, config: AgentConfig) -> HybridAgent:
    """
    Create a complete hybrid agent

    Args:
        a2a_url: A2A server URL
        mcp_url: MCP server URL
        config: Agent configuration

    Returns:
        HybridAgent instance
    """
    factory = HybridAgentFactory(a2a_url, mcp_url, config)
    return factory.create()


if __name__ == "__main__":
    # Test hybrid agent factory
    from config.settings import AgentConfig

    config = AgentConfig(max_iterations=2, verbose=True)

    factory = HybridAgentFactory(
        a2a_url="http://localhost:5000",
        mcp_url="http://localhost:7000",
        config=config
    )

    print("Hybrid agent factory created")
    print(f"Configuration: {config}")