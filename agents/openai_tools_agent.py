"""
OpenAI Tools Agent for Subsurface Data Management Platform
Replaces hardcoded intent detection with LLM-based tool selection
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)


@dataclass
class ToolInfo:
    """Tool information structure"""
    name: str
    description: str
    category: str
    parameters: Dict[str, Any]


class OpenAIToolsExecutor:
    """
    LLM-powered tool executor using OpenAI Tools Agent
    Maintains compatibility with existing MCP server integration
    """

    def __init__(self, mcp_client, tools_config: dict):
        """
        Initialize OpenAI Tools Agent

        Args:
            mcp_client: Existing MCP client instance
            tools_config: Tools configuration dictionary
        """
        self.mcp_client = mcp_client
        self.tools_config = tools_config

        # Load tools from MCP (existing logic)
        self.available_tools = self._load_tools_from_mcp()
        logger.info(f"Loaded {len(self.available_tools)} tools from MCP")

        # Convert to LangChain tools
        self.langchain_tools = self._convert_to_langchain_tools()
        logger.info(f"Converted {len(self.langchain_tools)} tools for LangChain")

        # Setup OpenAI agent
        self.agent_executor = self._setup_openai_agent()
        logger.info("OpenAI Tools Agent initialized successfully")

    def _load_tools_from_mcp(self) -> Dict[str, ToolInfo]:
        """
        Load tools from MCP server (existing logic preserved)
        """
        tools = {}

        try:
            # Get tools from MCP client (your existing implementation)
            if hasattr(self.mcp_client, 'list_tools'):
                mcp_tools = self.mcp_client.list_tools()
            else:
                # Fallback to tools_config if MCP client doesn't have list_tools
                mcp_tools = self.tools_config.get('tools', {})

            for tool_name, tool_data in mcp_tools.items():
                # Extract tool information
                description = tool_data.get('description', f'Tool for {tool_name}')
                category = self._categorize_tool(tool_name)
                parameters = tool_data.get('inputSchema', {}).get('properties', {})

                tools[tool_name] = ToolInfo(
                    name=tool_name,
                    description=description,
                    category=category,
                    parameters=parameters
                )

        except Exception as e:
            logger.error(f"Error loading tools from MCP: {e}")
            # Fallback to hardcoded tool list if MCP fails
            tools = self._get_fallback_tools()

        return tools

    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize tool based on name (existing logic)"""
        if 'las' in tool_name.lower():
            return 'LAS Analysis'
        elif 'segy' in tool_name.lower():
            return 'SEG-Y Analysis'
        elif tool_name in ['list_files', 'system_status', 'directory_info', 'health_check']:
            return 'System Operations'
        else:
            return 'General Tools'

    def _get_fallback_tools(self) -> Dict[str, ToolInfo]:
        """Fallback tools if MCP connection fails"""
        return {
            'list_files': ToolInfo('list_files', 'List available data files', 'System Operations', {}),
            'system_status': ToolInfo('system_status', 'Check system status', 'System Operations', {}),
            'segy_parser': ToolInfo('segy_parser', 'Parse SEG-Y seismic files', 'SEG-Y Analysis',
                                    {'filename': {'type': 'string'}}),
            'las_parser': ToolInfo('las_parser', 'Parse LAS well log files', 'LAS Analysis',
                                   {'filename': {'type': 'string'}}),
        }

    def _convert_to_langchain_tools(self) -> List[Tool]:
        """
        Convert MCP tools to LangChain Tool format with enhanced descriptions
        """
        langchain_tools = []

        for tool_name, tool_info in self.available_tools.items():
            # Create enhanced description for better LLM understanding
            enhanced_description = self._create_enhanced_description(tool_info)

            # Create LangChain Tool with better function handling
            def create_tool_func(tn: str):
                def tool_func(query: str) -> str:
                    return self._execute_mcp_tool(tn, query)
                return tool_func

            tool = Tool(
                name=tool_name,
                description=enhanced_description,
                func=create_tool_func(tool_name)  # Fixed closure issue
            )

            langchain_tools.append(tool)

        return langchain_tools

    def _create_enhanced_description(self, tool_info: ToolInfo) -> str:
        """
        Create enhanced tool descriptions for better LLM understanding
        """
        tool_name = tool_info.name
        base_desc = tool_info.description
        category = tool_info.category

        # Enhanced descriptions with specific use cases and parameter guidance
        enhanced_descriptions = {
            # System Tools
            'list_files': "List files in the data directory. Use 'pattern' parameter for specific patterns (e.g., '*.sgy', 'F3_*', 'shots*'). If no pattern specified, lists all files.",
            'system_status': "Get comprehensive system performance metrics including CPU, memory, and disk usage. No parameters needed.",
            'directory_info': "Get detailed information about the data directory structure and contents. No parameters needed.",
            'health_check': "Perform system health diagnostics. No parameters needed.",

            # SEG-Y Tools
            'segy_parser': "Parse individual SEG-Y seismic files. Requires 'filename' parameter with exact filename (e.g., 'filename': 'data.sgy').",
            'segy_qc': "Perform quality control analysis on SEG-Y files. Requires 'filename' parameter.",
            'segy_analysis': "Comprehensive analysis of SEG-Y file structure and content. Requires 'filename' parameter.",
            'segy_classify': "Classify SEG-Y survey type (2D/3D, marine/land). Requires 'filename' parameter.",
            'segy_survey_analysis': "Analyze survey geometry and acquisition parameters. Works with individual files or patterns.",
            'quick_segy_summary': "Generate quick summary of SEG-Y file characteristics. Requires 'filename' parameter.",
            'segy_complete_metadata_harvester': "Extract comprehensive metadata from SEG-Y headers. Requires 'filename' parameter.",
            'segy_survey_polygon': "Extract survey boundary polygon coordinates. Requires 'filename' parameter.",
            'segy_trace_outlines': "Analyze trace distribution and geometry. Requires 'filename' parameter.",

            # LAS Tools
            'las_parser': "Parse LAS well log files and extract curve data. Requires 'filename' parameter.",
            'las_analysis': "Comprehensive analysis of LAS file content and well data. Requires 'filename' parameter.",
            'las_qc': "Quality control analysis for LAS well logs. Requires 'filename' parameter.",
            'formation_evaluation': "Evaluate formation properties from well logs. Requires 'filename' parameter.",
            'well_correlation': "Correlate multiple wells for stratigraphic analysis. Can handle multiple files.",
            'calculate_shale_volume': "Calculate shale volume from gamma ray logs. Requires 'filename' parameter.",
        }

        if tool_name in enhanced_descriptions:
            return enhanced_descriptions[tool_name]

        # Fallback enhanced description
        context = ""
        param_hint = ""

        if 'SEG-Y' in category or 'segy' in tool_name.lower():
            context = "For seismic data analysis, parsing, classification, and quality control."
            param_hint = " Always use 'filename' parameter with exact file name."
        elif 'LAS' in category or 'las' in tool_name.lower():
            context = "For well log analysis, formation evaluation, and petrophysical calculations."
            param_hint = " Always use 'filename' parameter with exact file name."
        elif 'System' in category:
            context = "For file management and system health monitoring."
            param_hint = " Usually requires no parameters."

        return f"{base_desc} {context}{param_hint}"

    def _execute_mcp_tool(self, tool_name: str, query: str) -> str:
        """Execute MCP tool with proper parameter handling and validation"""
        try:
            logger.info(f"Executing {tool_name} with query: {query[:100]}...")

            # Extract parameters using the fixed method
            params = self._extract_parameters(query, tool_name)

            # Validate parameters
            validation_error = self._validate_tool_parameters(tool_name, params)
            if validation_error:
                logger.warning(f"Parameter validation failed: {validation_error}")
                return validation_error

            # Log what we're actually sending (for debugging)
            logger.debug(f"Extracted parameters for {tool_name}: {params}")

            # Call MCP tool
            raw_result = self._call_mcp_tool(tool_name, params)

            # Check for empty results
            if self._is_empty_result(raw_result):
                return self._handle_empty_result(tool_name, params)

            # Format and return result
            formatted_result = self._format_response(raw_result, tool_name)
            return formatted_result

        except Exception as e:
            logger.error(f"Error executing {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}\n\nPlease check the filename and try again."


    def _validate_tool_parameters(self, tool_name: str, params: dict) -> str:
        """
        Validate parameters for specific tools
        Returns error message if validation fails, None if OK
        """
        # Tools that require filename parameter
        filename_required_tools = [
            'segy_parser', 'segy_qc', 'segy_analysis', 'segy_classify',
            'quick_segy_summary', 'segy_complete_metadata_harvester',
            'segy_survey_polygon', 'segy_trace_outlines',
            'las_parser', 'las_analysis', 'las_qc', 'formation_evaluation',
            'calculate_shale_volume'
        ]

        if tool_name in filename_required_tools:
            if 'filename' not in params and 'query' not in params:
                return f"{tool_name} requires a filename parameter. Please specify a file to analyze."

            # Extract filename from query if needed
            if 'filename' not in params and 'query' in params:
                filename = self._extract_filename_from_query(params['query'])
                if not filename:
                    return f"Could not extract filename from query for {tool_name}. Please specify a valid filename."
                params['filename'] = filename

        # Tools that work without parameters
        no_param_tools = ['list_files', 'system_status', 'directory_info', 'health_check']

        # Validate file extensions for specific tools
        if tool_name.startswith('segy_') and 'filename' in params:
            filename = params['filename']
            if not any(filename.lower().endswith(ext) for ext in ['.sgy', '.segy']):
                return f"SEG-Y tool {tool_name} requires a .sgy or .segy file. Got: {filename}"

        if tool_name.startswith('las_') and 'filename' in params:
            filename = params['filename']
            if not filename.lower().endswith('.las'):
                return f"LAS tool {tool_name} requires a .las file. Got: {filename}"

        return None  # No validation errors

    def _extract_filename_from_query(self, query: str) -> str:
        """Extract filename from query string"""
        import re

        patterns = [
            r'([A-Za-z0-9_\-\.]+\.(?:las|sgy|segy|LAS|SGY|SEGY))',
            r'([A-Za-z0-9_\-]+[A-Za-z0-9_\-\.\s]+\.(?:sgy|segy|las))',
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _is_empty_result(self, result: dict) -> bool:
        """Check if MCP tool returned empty or minimal results"""
        if not result:
            return True

        # Check for common empty result patterns
        if isinstance(result, dict):
            if 'error' in result:
                return False  # Errors are not empty, they're meaningful

            # Check if content is essentially empty
            content = self._extract_mcp_content(result)
            if not content or content.strip() in ['', '{}', '[]', 'null']:
                return True

            # Check for minimal JSON responses
            if content.strip() in ['{"text": ""}', '{"text": null}']:
                return True

        return False

    def _handle_empty_result(self, tool_name: str, params: dict) -> str:
        """Handle empty results with helpful messages"""
        if 'filename' in params:
            filename = params['filename']
            return f"""**{tool_name} Analysis**

    The analysis of '{filename}' completed but returned no detailed results. This could indicate:

    1. **File not found**: Please verify the filename is correct
    2. **File format issue**: The file may be corrupted or in an unexpected format  
    3. **Tool configuration**: The tool may need additional parameters

    **Suggestions:**
    - Check if the file exists: Use 'list files' to see available files
    - Verify file format: Ensure SEG-Y files end in .sgy/.segy and LAS files end in .las
    - Try a different analysis tool for this file type

    Please try again with a verified filename or contact support if the issue persists."""

        else:
            return f"""**{tool_name} Analysis**

    The operation completed but returned no results. Please try:

    1. Providing more specific parameters
    2. Using 'list files' to see what data is available
    3. Checking system status to ensure all services are running

    **Analysis Status**: Completed with no data returned"""

    def _extract_parameters(self, query: str, tool_name: str) -> dict:
        """
        Extract parameters from natural language query
        Enhanced implementation with JSON handling
        """
        import re
        import json

        params = {}

        # NEW: Handle JSON string inputs from OpenAI agent
        if query.strip().startswith('{') and query.strip().endswith('}'):
            try:
                json_params = json.loads(query)
                if isinstance(json_params, dict):
                    return json_params
            except json.JSONDecodeError:
                pass

        # Clean the query of common LLM artifacts
        cleaned_query = query.replace('"', '').replace("'", "").strip()

        # Extract filenames with extensions (improved patterns)
        filename_patterns = [
            r'([A-Za-z0-9_\-\.]+\.las)',  # LAS files
            r'([A-Za-z0-9_\-\.]+\.segy?)',  # SEG-Y files
            r'([A-Za-z0-9_\-\.]+\.sgy)',  # SGY files
            r'([A-Za-z0-9_\-\.]+\.SGY)',  # Uppercase SGY
            r'([A-Za-z0-9_\-\.]+\.SEGY)',  # Uppercase SEGY
            # Complex filenames with special characters
            r'([A-Za-z0-9_\-]+[A-Za-z0-9_\-\.\s]+\.(?:sgy|segy|las|SGY|SEGY|LAS))',
        ]

        for pattern in filename_patterns:
            matches = re.findall(pattern, cleaned_query, re.IGNORECASE)
            if matches:
                # Take the longest match (most specific)
                filename = max(matches, key=len)
                params['filename'] = filename
                break

        # Extract file patterns for multi-file operations
        pattern_matches = re.findall(r'(\*\w*\.?\w*)', cleaned_query)
        if pattern_matches:
            params['pattern'] = pattern_matches[0]

        # Extract specific patterns like "F3_*"
        specific_patterns = re.findall(r'([A-Za-z0-9_]+\*[A-Za-z0-9_]*)', cleaned_query)
        if specific_patterns:
            params['file_pattern'] = specific_patterns[0]

        # Tool-specific parameter extraction
        if tool_name in ['list_files']:
            # For list_files, extract any pattern mentioned
            if 'files matching' in cleaned_query.lower():
                pattern_part = cleaned_query.lower().split('files matching')[-1].strip()
                pattern_match = re.search(r'([A-Za-z0-9_\*\.]+)', pattern_part)
                if pattern_match:
                    params['pattern'] = pattern_match.group(1)
            elif not params:
                params = {}  # list_files needs no params for "list files"

        elif tool_name in ['system_status', 'health_check', 'directory_info']:
            params = {}  # These tools need no parameters

        elif tool_name.startswith('segy_') or tool_name.startswith('las_'):
            # For analysis tools, ensure we have a filename
            if not params.get('filename'):
                # Extract any file-like string
                file_match = re.search(r'([A-Za-z0-9_\-\.]+\.[A-Za-z]{2,4})', cleaned_query)
                if file_match:
                    params['filename'] = file_match.group(1)
                else:
                    # Default parameter for query-based tools
                    params['query'] = cleaned_query

        # Fallback for unmatched tools
        if not params and tool_name not in ['list_files', 'system_status', 'health_check', 'directory_info']:
            params = {'query': cleaned_query}

        return params

    def _call_mcp_tool(self, tool_name: str, params: dict) -> dict:
        """
        Call MCP tool (preserve existing logic)
        """
        try:
            # Use your existing MCP client call method
            if hasattr(self.mcp_client, 'call_tool'):
                result = self.mcp_client.call_tool(tool_name, params)
            else:
                # Fallback implementation
                result = {'error': 'MCP client not properly configured'}

            return result

        except Exception as e:
            logger.error(f"MCP tool call failed for {tool_name}: {e}")
            return {'error': str(e)}

    def _format_response(self, result: dict, tool_name: str) -> str:
        """
        Format tool response (preserve existing formatting logic)
        This preserves all your existing formatting improvements
        """
        confidence_indicator = "HIGH CONFIDENCE"

        try:
            # Extract content using existing logic
            extracted_content = self._extract_mcp_content(result)

            # Use existing formatting
            return self._format_text_content(extracted_content, tool_name, confidence_indicator)

        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return f"**Analysis Complete** (using {tool_name})\n\nProcessing completed.\n\n**Analysis Complete**"

    def _extract_mcp_content(self, result: dict) -> str:
        """
        Extract actual content from MCP response structure
        (Copy your existing implementation)
        """
        try:
            # Handle the nested MCP response structure
            if 'content' in result and isinstance(result['content'], list):
                if len(result['content']) > 0 and 'text' in result['content'][0]:
                    inner_text = result['content'][0]['text']

                    # Check if inner_text is JSON with nested text
                    if inner_text.strip().startswith('{"text":'):
                        try:
                            inner_parsed = json.loads(inner_text)
                            if 'text' in inner_parsed:
                                return inner_parsed['text']
                        except json.JSONDecodeError:
                            pass

                    return inner_text

            # Fallback to original result
            return str(result)

        except Exception as e:
            logger.error(f"Error extracting MCP content: {e}")
            return str(result)

    def _format_text_content(self, text_content: str, tool_name: str, indicator: str) -> str:
        """
        Format text content (preserve existing formatting)
        """
        try:
            # Try to parse as JSON first
            if text_content.strip().startswith('{') or text_content.strip().startswith('['):
                try:
                    parsed_data = json.loads(text_content)
                    return self._format_structured_response(tool_name, parsed_data, indicator)
                except json.JSONDecodeError:
                    pass

            # If not JSON, format as text response
            return self._format_text_response(tool_name, text_content, indicator)

        except Exception as e:
            logger.error(f"Error formatting text content: {e}")
            return f"{indicator} **Analysis Complete** ({tool_name})\n\n{text_content}\n\n**Processing Complete**"

    def _format_structured_response(self, tool_name: str, data: dict, indicator: str) -> str:
        """Format structured JSON response (preserve existing logic)"""
        if tool_name == "system_status":
            return self._format_system_status(data, indicator)

        # SEG-Y specific formatting
        if 'segy' in tool_name.lower() and isinstance(data, dict):
            return self._format_segy_response(data, indicator, tool_name)

        # Generic formatting
        return self._format_generic_response(data, indicator, tool_name)

    def _format_system_status(self, data: dict, indicator: str) -> str:
        """Format system status (preserve existing implementation)"""
        response = f"{indicator} **System Status Report**\n\n"

        if 'timestamp' in data:
            response += f"**Report Time:** {data['timestamp']}\n\n"

        if 'system_metrics' in data:
            metrics = data['system_metrics']
            response += "### System Performance\n"
            if 'cpu_percent' in metrics:
                response += f"- **CPU Usage:** {metrics['cpu_percent']}%\n"
            if 'memory_percent' in metrics:
                response += f"- **Memory Usage:** {metrics['memory_percent']}%\n"
            if 'disk_percent' in metrics:
                response += f"- **Disk Usage:** {metrics['disk_percent']}%\n"

        response += "\n**Status Report Complete**"
        return response

    def _format_segy_response(self, data: dict, indicator: str, tool_name: str) -> str:
        """Format SEG-Y response (preserve existing implementation)"""
        response = f"{indicator} **SEG-Y Analysis Complete** (using {tool_name})\n\n"

        if 'file_processed' in data:
            response += f"**File:** {data['file_processed']}\n"
        if 'survey_type' in data:
            response += f"**Survey Type:** {data['survey_type']}\n"
        if 'total_traces' in data:
            response += f"**Total Traces:** {data['total_traces']:,}\n"

        response += "\n**Analysis Complete**"
        return response

    def _format_generic_response(self, data: dict, indicator: str, tool_name: str) -> str:
        """Format generic response"""
        response = f"{indicator} **Analysis Complete** (using {tool_name})\n\n"

        # Display key-value pairs
        for key, value in list(data.items())[:10]:
            if isinstance(value, (str, int, float)):
                response += f"**{key.replace('_', ' ').title()}:** {value}\n"

        return response + "\n**Analysis Complete**"

    def _format_text_response(self, tool_name: str, text_content: str, indicator: str) -> str:
        """Format plain text response"""
        return f"{indicator} **Analysis Complete** (using {tool_name})\n\n{text_content}\n\n**Analysis Complete**"

    def _setup_openai_agent(self) -> AgentExecutor:
        """
        Setup OpenAI Tools Agent with subsurface domain customization
        """
        try:
            # Create custom prompt for subsurface domain
            prompt = self._create_subsurface_prompt()

            # Setup OpenAI model
            llm = ChatOpenAI(
                model="gpt-3.5-turbo-1106",
                temperature=0.1,  # Low temperature for consistent technical responses
                max_tokens=2000,
                # OpenAI API key should be in environment variables
            )

            # Create the agent
            agent = create_openai_tools_agent(llm, self.langchain_tools, prompt)

            # Create executor with error handling
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.langchain_tools,
                verbose=True,  # Show reasoning for debugging
                handle_parsing_errors=True,
                max_iterations=5,  # Prevent infinite loops
                # early_stopping_method="generate"
            )

            return agent_executor

        except Exception as e:
            logger.error(f"Error setting up OpenAI agent: {e}")
            raise

    def _create_subsurface_prompt(self) -> ChatPromptTemplate:
        """
        Create specialized prompt for subsurface data analysis
        """
        system_message = """You are an expert subsurface data analyst and geophysicist with specialized knowledge in:

    **SEISMIC DATA (SEG-Y FILES):**
    - 2D/3D seismic survey analysis and interpretation
    - Survey geometry, acquisition parameters, and trace analysis
    - Quality control, metadata extraction, and file validation
    - Common formats: .sgy, .segy, .SGY, .SEGY

    **WELL LOG DATA (LAS FILES):**
    - Wireline log analysis and formation evaluation
    - Petrophysical calculations and curve analysis
    - Well correlation and stratigraphic interpretation
    - Format: .las, .LAS

    **CRITICAL PARAMETER HANDLING:**
    - Always use exact filenames as provided by the user
    - For file analysis tools, use the 'filename' parameter with the complete filename
    - When listing files, use patterns like 'F3_*' or '*.sgy' for specific searches
    - If a filename contains complex characters or spaces, use it exactly as shown

    **TOOL SELECTION GUIDELINES:**
    1. **File Listing**: Use 'list_files' for finding available data
    2. **SEG-Y Analysis**: 
       - Use 'segy_parser' for basic file parsing
       - Use 'segy_qc' for quality control
       - Use 'segy_survey_analysis' for geometry analysis
       - Use 'segy_complete_metadata_harvester' for detailed metadata
    3. **LAS Analysis**:
       - Use 'las_parser' for basic parsing
       - Use 'formation_evaluation' for petrophysical analysis
       - Use 'calculate_shale_volume' for shale content
    4. **System Operations**: Use 'system_status' for health checks

    **RESPONSE REQUIREMENTS:**
    - Always provide technical context and interpretation
    - Explain what the analysis reveals about the subsurface data
    - Suggest follow-up analyses when appropriate
    - If results are minimal, explain possible reasons and next steps

    **PARAMETER FORMAT:**
    When calling tools that need filenames, use the exact filename string:
    - Correct: filename = "F3_Similarity_FEF_subvolume_IL230-430_XL475-675_T1600-1800.sgy"
    - Incorrect: filename = {"filename": "shortened_name.sgy"}

    Remember: Your goal is to provide actionable geological and geophysical insights from the data analysis."""

        try:
            base_prompt = hub.pull("hwchase17/openai-tools-agent")
            messages = base_prompt.messages.copy()
            messages[0] = SystemMessage(content=system_message)
            return ChatPromptTemplate.from_messages(messages)

        except Exception as e:
            logger.warning(f"Could not load prompt from hub: {e}, using fallback")

            return ChatPromptTemplate.from_messages([
                SystemMessage(content=system_message),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])

    def execute_query(self, query: str) -> str:
        """
        Execute user query using OpenAI Tools Agent

        Args:
            query: User's natural language query

        Returns:
            Formatted response string
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")

            # Execute using OpenAI agent
            result = self.agent_executor.invoke({
                "input": query
            })

            # Extract the final output
            output = result.get("output", "Processing completed.")

            logger.info("Query processed successfully")
            return output

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return f"I encountered an error processing your request: {str(e)}\n\nPlease try rephrasing your question or check if the requested file exists."

    def get_available_tools(self) -> Dict[str, str]:
        """
        Get list of available tools for debugging
        """
        return {name: info.description for name, info in self.available_tools.items()}