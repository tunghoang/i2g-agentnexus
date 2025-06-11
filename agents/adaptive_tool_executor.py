# agents/adaptive_tool_executor.py
"""
Complete adaptive tool executor that works with your tool definitions
FIXED VERSION - Better MCP response handling
"""

import re
import logging
from typing import Dict, Any, Optional, List
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolIntent:
    """Represents a detected user intent"""
    tool_name: str
    confidence: float
    extracted_params: Dict[str, Any]
    reasoning: str


class ToolRegistry:
    """Discovers and categorizes your tools automatically"""

    def __init__(self, mcp_tools: Dict[str, Any], tool_descriptions: Dict[str, str]):
        self.tools = mcp_tools
        self.descriptions = tool_descriptions
        self.categorized_tools = self._categorize_tools()
        logger.info(f"Initialized with {len(self.tools)} tools in {len(self.categorized_tools)} categories")

    def _categorize_tools(self) -> Dict[str, List[str]]:
        """Automatically categorize tools based on naming patterns"""
        categories = {}

        for tool_name in self.tools.keys():
            if tool_name.startswith('segy_'):
                category = 'segy'
            elif tool_name.startswith('las_'):
                category = 'las'
            elif tool_name in ['list_files', 'system_status', 'directory_info', 'health_check']:
                category = 'system'
            else:
                category = 'other'

            if category not in categories:
                categories[category] = []
            categories[category].append(tool_name)

        return categories

    def get_tools_by_category(self, category: str) -> List[str]:
        return self.categorized_tools.get(category, [])

    def get_tool_description(self, tool_name: str) -> str:
        return self.descriptions.get(tool_name, f"Execute {tool_name}")


class IntentDetector:
    """Detects user intent and maps to appropriate tools"""

    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry
        self.intent_patterns = self._build_intent_patterns()

    def _build_intent_patterns(self) -> Dict[str, Dict]:
        """Build intent patterns dynamically based on available tools"""
        patterns = {}

        segy_tools = self.registry.get_tools_by_category('segy')
        if segy_tools:
            patterns['segy_analysis'] = {
                'keywords': ['segy', '.sgy', 'seismic', 'metadata'],
                'file_extensions': ['.sgy'],
                'tools': segy_tools,
                'default_tool': 'segy_parser'
            }

        las_tools = self.registry.get_tools_by_category('las')
        if las_tools:
            patterns['las_analysis'] = {
                'keywords': ['las', '.las', 'well', 'log'],
                'file_extensions': ['.las'],
                'tools': las_tools,
                'default_tool': 'las_parser'
            }

        system_tools = self.registry.get_tools_by_category('system')
        if system_tools:
            patterns['system_operations'] = {
                'keywords': ['list', 'files', 'status', 'health', 'directory'],
                'tools': system_tools,
                'default_tool': 'list_files'
            }

        return patterns

    def detect_intent(self, query: str) -> Optional[ToolIntent]:
        """Detect user intent from query"""
        query_lower = query.lower()
        best_intent = None
        highest_confidence = 0.0

        filename = self._extract_filename(query)
        file_type = self._get_file_type(filename) if filename else None

        # Check for system commands first
        if any(keyword in query_lower for keyword in ["list", "show", "display"]) and \
                any(keyword in query_lower for keyword in ["files", "data", "available"]):
            return ToolIntent(
                tool_name="list_files",
                confidence=0.9,
                extracted_params={},
                reasoning="Detected file listing request"
            )

        if any(keyword in query_lower for keyword in ["status", "health", "system"]):
            return ToolIntent(
                tool_name="system_status",
                confidence=0.9,
                extracted_params={},
                reasoning="Detected system status request"
            )

        # Check for specific SEG-Y tool requests
        if "classify" in query_lower and any(ext in query_lower for ext in ['.sgy', '.segy', 'segy']):
            return ToolIntent(
                tool_name="segy_classify",
                confidence=0.9,
                extracted_params={'filename': filename},
                reasoning="Detected SEG-Y classification request"
            )

        if "quick" in query_lower and "summary" in query_lower and any(
                ext in query_lower for ext in ['.sgy', '.segy', 'segy']):
            return ToolIntent(
                tool_name="quick_segy_summary",
                confidence=0.9,
                extracted_params={'filename': filename},
                reasoning="Detected quick SEG-Y summary request"
            )

        # Check for LAS analysis (use parser instead of analysis to avoid the error)
        if "formation" in query_lower and "analyze" in query_lower and filename and filename.endswith('.las'):
            return ToolIntent(
                tool_name="las_parser",  # Use parser instead of analysis
                confidence=0.9,
                extracted_params={'filename': filename},
                reasoning="Detected LAS formation analysis request"
            )

        for intent_name, pattern in self.intent_patterns.items():
            confidence = self._calculate_confidence(query_lower, pattern, file_type)

            if confidence > highest_confidence:
                tool_name = self._select_specific_tool(query_lower, pattern, filename)

                best_intent = ToolIntent(
                    tool_name=tool_name,
                    confidence=confidence,
                    extracted_params={'filename': filename} if filename else {},
                    reasoning=f"Matched {intent_name} with {confidence:.2f} confidence"
                )
                highest_confidence = confidence

        return best_intent

    def _calculate_confidence(self, query: str, pattern: Dict, file_type: Optional[str]) -> float:
        """Calculate confidence score for an intent pattern"""
        confidence = 0.0

        if file_type and 'file_extensions' in pattern:
            if file_type in pattern['file_extensions']:
                confidence += 0.6

        keywords = pattern.get('keywords', [])
        matched_keywords = sum(1 for keyword in keywords if keyword in query)
        if keywords:
            confidence += (matched_keywords / len(keywords)) * 0.4

        return min(confidence, 1.0)

    def _select_specific_tool(self, query: str, pattern: Dict, filename: Optional[str]) -> str:
        """Select the most appropriate tool from a category"""
        available_tools = pattern['tools']

        # Enhanced tool selection based on query intent
        if 'complete' in query and 'metadata' in query:
            for tool in available_tools:
                if 'complete_metadata_harvester' in tool:
                    return tool

        if 'harvest' in query or 'complete' in query:
            for tool in available_tools:
                if 'harvester' in tool or 'complete' in tool:
                    return tool

        if 'metadata' in query or 'parse' in query or 'information' in query:
            for tool in available_tools:
                if 'parser' in tool:
                    return tool

        if 'quality' in query or 'qc' in query:
            for tool in available_tools:
                if 'qc' in tool:
                    return tool

        if 'analysis' in query or 'analyze' in query:
            for tool in available_tools:
                if 'analysis' in tool and 'parser' not in tool:
                    return tool

        if 'classify' in query:
            for tool in available_tools:
                if 'classify' in tool:
                    return tool

        if 'survey' in query:
            for tool in available_tools:
                if 'survey' in tool:
                    return tool

        if 'quick' in query or 'summary' in query:
            for tool in available_tools:
                if 'quick' in tool or 'summary' in tool:
                    return tool

        return pattern.get('default_tool', available_tools[0])

    def _extract_filename(self, query: str) -> Optional[str]:
        """Extract filename from query"""
        patterns = [
            r'([A-Za-z0-9_\-\.]+\.sgy)',
            r'([A-Za-z0-9_\-\.]+\.segy)',
            r'([A-Za-z0-9_\-\.]+\.las)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)

        return None

    def _get_file_type(self, filename: str) -> Optional[str]:
        """Get file extension"""
        if not filename:
            return None
        return '.' + filename.split('.')[-1].lower() if '.' in filename else None


class ResponseFormatter:
    """Formats responses based on tool output"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def format_response(self, tool_name: str, result: Any, query: str, confidence: float) -> str:
        """Format tool response for end user"""
        try:
            confidence_indicator = "HIGH CONFIDENCE" if confidence >= 0.8 else "CONFIRMED" if confidence >= 0.6 else "WARNING"

            # Enhanced debugging
            logger.debug(f"Formatting response for {tool_name}")
            logger.debug(f"Result type: {type(result)}")
            logger.debug(f"Result content preview: {str(result)[:200]}...")

            # Better handling of MCP responses
            if isinstance(result, dict):
                # Check if it's an MCP error response
                if "error" in result:
                    return f"**Tool Error** ({tool_name})\n\n{result['error']}"

                # Check for MCP content structure
                if "content" in result:
                    extracted_content = self._extract_mcp_content(result)
                    return self._format_text_content(extracted_content, tool_name, confidence_indicator)

                # Check if it's an MCP success response with text content
                if "text" in result:
                    return self._format_text_content(result["text"], tool_name, confidence_indicator)

                # Direct dictionary response
                return self._format_structured_response(tool_name, result, confidence_indicator)

            elif isinstance(result, str):
                return self._format_text_content(result, tool_name, confidence_indicator)

            else:
                return self._format_text_content(str(result), tool_name, confidence_indicator)

        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return f"**Formatting Error** ({tool_name})\n\nAnalysis completed but formatting failed: {str(e)}"

    def _format_text_content(self, text_content: str, tool_name: str, indicator: str) -> str:
        """Format text content from MCP response"""
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

    def _extract_mcp_content(self, result: dict) -> str:
        """Extract actual content from MCP response structure"""
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

    def _format_structured_response(self, tool_name: str, data: dict, indicator: str) -> str:
        """Format structured data response"""
        if tool_name.startswith('segy_'):
            return self._format_segy_response(data, indicator, tool_name)
        elif tool_name.startswith('las_'):
            return self._format_las_response(data, indicator, tool_name)
        else:
            return self._format_generic_response(data, indicator, tool_name)

    def _format_segy_response(self, data: dict, indicator: str, tool_name: str) -> str:
        """Format SEG-Y analysis response"""
        response = f"{indicator} **SEG-Y Analysis Complete** (using {tool_name})\n\n"

        # Key metrics upfront
        if 'file_processed' in data:
            response += f"**File:** {data['file_processed']}\n"
        if 'survey_type' in data:
            response += f"**Survey Type:** {data['survey_type']}\n"
        if 'total_traces' in data:
            response += f"**Total Traces:** {data['total_traces']:,}\n"
        if 'quality_rating' in data:
            response += f"**Quality:** {data['quality_rating'].upper()}\n\n"

        # Geometry Information
        if any(k in data for k in ['min_inline', 'max_inline', 'min_xline', 'max_xline']):
            response += "### Geometry\n"
            if 'min_inline' in data and 'max_inline' in data:
                response += f"- **Inlines:** {data['min_inline']} to {data['max_inline']}\n"
            if 'min_xline' in data and 'max_xline' in data:
                response += f"- **Crosslines:** {data['min_xline']} to {data['max_xline']}\n"

        # Technical Details
        if any(k in data for k in ['sample_rate_ms', 'trace_length_ms', 'file_size_mb']):
            response += "\n### Technical Details\n"
            if 'sample_rate_ms' in data:
                response += f"- **Sample Rate:** {data['sample_rate_ms']} ms\n"
            if 'trace_length_ms' in data:
                response += f"- **Trace Length:** {data['trace_length_ms']} ms\n"
            if 'file_size_mb' in data:
                response += f"- **File Size:** {data['file_size_mb']} MB\n"

        # Survey details
        if any(k in data for k in ['detected_survey_type', 'primary_sorting', 'stack_type']):
            response += "\n### Survey Details\n"
            if 'detected_survey_type' in data:
                response += f"- **Survey Type:** {data['detected_survey_type']}\n"
            if 'primary_sorting' in data:
                response += f"- **Primary Sorting:** {data['primary_sorting']}\n"
            if 'stack_type' in data:
                response += f"- **Stack Type:** {data['stack_type']}\n"

        # Quality assessment
        if 'quality_analysis' in data:
            quality = data['quality_analysis']
            response += "\n### Quality Assessment\n"
            if isinstance(quality, dict):
                if 'signal_metrics' in quality:
                    metrics = quality['signal_metrics']
                    if isinstance(metrics, dict):
                        if 'dynamic_range_db' in metrics:
                            response += f"- **Dynamic Range:** {metrics['dynamic_range_db']:.1f} dB\n"
                        if 'signal_to_noise' in metrics:
                            response += f"- **Signal/Noise:** {metrics['signal_to_noise']:.1f}\n"

                if 'issues' in quality and quality['issues']:
                    response += f"- **Issues:** {', '.join(quality['issues'])}\n"

        # Add any other keys that weren't specifically formatted
        other_keys = [k for k in data.keys() if k not in [
            'file_processed', 'survey_type', 'total_traces', 'quality_rating',
            'min_inline', 'max_inline', 'min_xline', 'max_xline',
            'sample_rate_ms', 'trace_length_ms', 'file_size_mb',
            'detected_survey_type', 'primary_sorting', 'stack_type', 'quality_analysis'
        ]]

        if other_keys:
            response += "\n### Additional Information\n"
            for key in other_keys[:5]:  # Limit to first 5 additional keys
                if isinstance(data[key], (str, int, float)):
                    response += f"- **{key.replace('_', ' ').title()}:** {data[key]}\n"

        return response + "\n**Analysis Complete**"

    def _format_las_response(self, data: dict, indicator: str, tool_name: str) -> str:
        """Format LAS analysis response"""
        response = f"{indicator} **LAS Analysis Complete** (using {tool_name})\n\n"

        if 'file_processed' in data:
            response += f"**File:** {data.get('file_processed', 'N/A')}\n"
        if 'well_name' in data:
            response += f"**Well:** {data.get('well_name', 'N/A')}\n"

        # Add any other LAS-specific formatting here
        response += "\n**Analysis Complete**"
        return response

    def _format_generic_response(self, data: dict, indicator: str, tool_name: str) -> str:
        """Format generic response"""
        response = f"{indicator} **Analysis Complete** (using {tool_name})\n\n"

        # Special formatting for system status
        if tool_name == "system_status":
            return self._format_system_status(data, indicator)

        # Display key-value pairs
        for key, value in list(data.items())[:10]:  # Limit to first 10 items
            if isinstance(value, (str, int, float)):
                response += f"**{key.replace('_', ' ').title()}:** {value}\n"

        return response + "\n**Analysis Complete**"

    def _format_system_status(self, data: dict, indicator: str) -> str:
        """Format system status response"""
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
            if 'active_processes' in metrics:
                response += f"- **Active Processes:** {metrics['active_processes']}\n"

        if 'data_directory_info' in data:
            dir_info = data['data_directory_info']
            response += "\n### Data Directory\n"
            if 'data_directory' in dir_info:
                response += f"- **Location:** {dir_info['data_directory']}\n"
            if 'las_files_count' in dir_info:
                response += f"- **LAS Files:** {dir_info['las_files_count']}\n"
            if 'segy_files_count' in dir_info:
                response += f"- **SEG-Y Files:** {dir_info['segy_files_count']}\n"

        if 'environment_info' in data:
            env_info = data['environment_info']
            response += "\n### Environment\n"
            if 'openai_api_key_set' in env_info:
                api_status = "✓ Configured" if env_info['openai_api_key_set'] else "✗ Missing"
                response += f"- **OpenAI API Key:** {api_status}\n"
            if 'data_dir_writable' in env_info:
                write_status = "✓ Writable" if env_info['data_dir_writable'] else "✗ Read-only"
                response += f"- **Data Directory:** {write_status}\n"

        if 'tool_count' in data:
            response += f"\n### Tools Available\n"
            response += f"- **Total Tools:** {data['tool_count']}\n"

        if 'overall_status' in data:
            response += f"\n**Overall Status:** {data['overall_status']}\n"

        if 'recommendations' in data and data['recommendations']:
            response += f"\n### Recommendations\n"
            for rec in data['recommendations']:
                response += f"- {rec}\n"

        return response + "\n**Status Report Complete**"

    def _format_text_response(self, tool_name: str, text: str, indicator: str) -> str:
        """Format text response"""
        return f"{indicator} **Analysis Complete** (using {tool_name})\n\n{text}\n\n**Analysis Complete**"


class AdaptiveToolExecutor:
    """Main executor that adapts to your tool configuration"""

    def __init__(self, mcp_tools: Dict[str, Any], tool_descriptions: Dict[str, str]):
        self.registry = ToolRegistry(mcp_tools, tool_descriptions)
        self.intent_detector = IntentDetector(self.registry)
        self.response_formatter = ResponseFormatter(self.registry)

    def execute_query(self, query: str) -> str:
        """Execute user query and return formatted response"""
        try:
            intent = self.intent_detector.detect_intent(query)

            if not intent or intent.confidence < 0.3:
                return self._handle_unknown_query(query)

            logger.info(f"Executing {intent.tool_name} - {intent.reasoning}")
            result = self._execute_tool(intent.tool_name, intent.extracted_params, query)

            return self.response_formatter.format_response(
                intent.tool_name, result, query, intent.confidence
            )

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return f"**Processing Error**\n\nI encountered an issue: {str(e)}\n\nPlease try rephrasing your question."

    def _execute_tool(self, tool_name: str, params: Dict[str, Any], original_query: str) -> Any:
        """Execute tool with appropriate parameters"""
        if tool_name not in self.registry.tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        tool = self.registry.tools[tool_name]

        # Extract filename if present
        filename = params.get('filename') or self._extract_filename_from_query(original_query)

        # Prepare input - try simple approach first
        if filename and tool_name.startswith(('segy_', 'las_')):
            # For file-based tools, try just the filename first
            tool_input = filename
        elif tool_name == 'list_files':
            # Extract pattern from query
            pattern = "*"
            if "segy" in original_query.lower() or "sgy" in original_query.lower():
                pattern = "*.sgy"
            elif "las" in original_query.lower():
                pattern = "*.las"
            tool_input = pattern
        elif tool_name in ['system_status', 'health_check', 'directory_info']:
            # For system tools, try empty string or minimal input
            tool_input = ""
        else:
            # Fallback to original query
            tool_input = original_query

        # Enhanced logging
        logger.debug(f"Executing tool {tool_name} with simple input: {tool_input}")

        # Try different call patterns
        try:
            if hasattr(tool, 'run'):
                result = tool.run(tool_input)
            elif hasattr(tool, 'invoke'):
                result = tool.invoke(tool_input)
            elif callable(tool):
                result = tool(tool_input)
            else:
                raise ValueError(f"Don't know how to call tool {tool_name}")

            # Enhanced debugging - show raw result
            print(f"DEBUG RAW RESULT from {tool_name}: {result}")
            logger.warning(f"RAW RESULT from {tool_name}: {result}")
            logger.debug(f"Tool {tool_name} returned: {type(result)} - {str(result)[:200]}...")
            return result

        except Exception as e:
            logger.error(f"Error calling {tool_name} with simple input: {e}")

            # If simple approach fails, try structured approach
            logger.debug(f"Trying structured input for {tool_name}")

            try:
                if tool_name.startswith('segy_'):
                    structured_input = {
                        "args": filename or original_query,
                        "kwargs": "{}"
                    }
                elif tool_name.startswith('las_'):
                    structured_input = {
                        "file_path": filename or original_query,
                        "kwargs": "{}"
                    }
                elif tool_name == 'list_files':
                    pattern = "*"
                    if "segy" in original_query.lower() or "sgy" in original_query.lower():
                        pattern = "*.sgy"
                    elif "las" in original_query.lower():
                        pattern = "*.las"
                    structured_input = {
                        "pattern": pattern,
                        "kwargs": "{}"
                    }
                elif tool_name in ['system_status', 'health_check']:
                    structured_input = {
                        "kwargs": "{}"
                    }
                elif tool_name == 'directory_info':
                    structured_input = {
                        "directory_path": "./data",
                        "kwargs": "{}"
                    }
                else:
                    raise e  # Re-raise original error

                logger.debug(f"Trying structured input: {structured_input}")

                if hasattr(tool, 'run'):
                    result = tool.run(structured_input)
                elif hasattr(tool, 'invoke'):
                    result = tool.invoke(structured_input)
                elif callable(tool):
                    result = tool(structured_input)
                else:
                    raise ValueError(f"Don't know how to call tool {tool_name}")

                logger.debug(f"Tool {tool_name} succeeded with structured input")
                return result

            except Exception as e2:
                logger.error(f"Both simple and structured approaches failed for {tool_name}: {e2}")
                raise e2

    def _extract_filename_from_query(self, query: str) -> Optional[str]:
        """Extract filename from query"""
        patterns = [
            r'([A-Za-z0-9_\-\.]+\.sgy)',
            r'([A-Za-z0-9_\-\.]+\.segy)',
            r'([A-Za-z0-9_\-\.]+\.las)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)

        return None

    def _handle_unknown_query(self, query: str) -> str:
        """Handle queries with no clear intent"""
        return f"""I'm not sure how to help with that request.

**Available capabilities:**
**SEG-Y Analysis**: Parse, analyze, QC seismic data
**Well Log Analysis**: Parse, analyze, evaluate LAS files
**System Operations**: File management, health checks

**Suggestions:**
- For SEG-Y files: "What metadata is in filename.sgy?"
- For LAS files: "Analyze filename.las" 
- For system info: "List files" or "System status"

Please try rephrasing your question or specify a filename."""


class FlexibleAgentReplacement:
    """Drop-in replacement for your ReAct agent"""

    def __init__(self, mcp_server_or_tools, tool_descriptions: Dict[str, str]):
        if hasattr(mcp_server_or_tools, 'get_tools'):
            self.tools = mcp_server_or_tools.get_tools()
        else:
            self.tools = mcp_server_or_tools

        self.executor = AdaptiveToolExecutor(self.tools, tool_descriptions)
        logger.info(f"Flexible agent replacement ready with {len(self.tools)} tools")

    def run(self, query: str) -> str:
        """Drop-in replacement for agent.run() - same interface"""
        return self.executor.execute_query(query)

    def invoke(self, query: str) -> str:
        """Alternative interface"""
        return self.executor.execute_query(query)


# THIS IS THE FUNCTION YOU NEED TO IMPORT
def create_flexible_agent(mcp_server, tool_descriptions: Dict[str, str]):
    """
    Create flexible agent that replaces your ReAct agent
    Args:
        mcp_server: Your MCP server instance or tools dict
        tool_descriptions: Dictionary of tool descriptions
    Returns:
        FlexibleAgentReplacement instance with .run() method
    """
    return FlexibleAgentReplacement(mcp_server, tool_descriptions)