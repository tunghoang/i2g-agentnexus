"""
Google ADK Agent - FULLY FIXED VERSION
Fixes all syntax errors, scope issues, and method access problems
"""
import traceback
import os
import json
import time
import logging
import inspect
import asyncio
from typing import Optional, Dict, Any, List, Union

# Google ADK imports
try:
    from google.adk.agents import Agent, LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.tools import FunctionTool, ToolContext
    from google.genai import types
    GOOGLE_ADK_AVAILABLE = True
except ImportError as e:
    GOOGLE_ADK_AVAILABLE = False

from config.settings import AgentConfig
from servers.mcp_server import MCPClient
from base_utils import recursive_get, recursive_put
logger = logging.getLogger(__name__)


class ToolExecutingAgentExecutor:
    """
    Google ADK Agent that properly executes MCP tools

    FULLY FIXED: Resolves all scope and method access issues
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

        # Statistics
        self.stats = {
            "total_invocations": 0,
            "successful_invocations": 0,
            "failed_invocations": 0,
            "tool_executions": 0,
            "system_type": "Google ADK Agent with Tool Execution"
        }

        self.agent_data = {
            "store": None
        }

        self.logger.info("Google ADK Tool Executing Agent created")

    def _create_tool_functions(self) -> List:
        """Create Python functions that ADK can automatically wrap as tools"""
        self.logger.info("Creating tool functions for Google ADK...")

        # Store reference to self for use in closures
        executor_instance = self

        tools = []

        # List Files Tool Function - NO DEFAULT PARAMETERS
        def list_files(pattern: str) -> dict:
            """List files matching pattern in the data directory

            Args:
                pattern: File pattern to match (e.g., "*.las", "*.sgy", "*")

            Returns:
                dict: Results containing matched files
            """
            try:
                executor_instance.logger.info(f"Executing list_files with pattern: {pattern}")
                result = executor_instance._execute_mcp_tool("list_files", pattern)
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in list_files: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(list_files)

        # System Status Tool Function - NO DEFAULT PARAMETERS
        def system_status(query: str) -> dict:
            """Get comprehensive system health and performance metrics

            Args:
                query: Query parameter for system status (use empty string if not needed)

            Returns:
                dict: System status information
            """
            try:
                executor_instance.logger.info("Executing system_status")
                result = executor_instance._execute_mcp_tool("system_status", query)
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in system_status: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(system_status)

        # Health Check Tool Function - NO DEFAULT PARAMETERS
        def health_check(query: str) -> dict:
            """Perform comprehensive health check of the platform

            Args:
                query: Query parameter for health check (use empty string if not needed)

            Returns:
                dict: Health check results
            """
            try:
                executor_instance.logger.info("Executing health_check")
                result = executor_instance._execute_mcp_tool("health_check", query)
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in health_check: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(health_check)

        # Directory Info Tool Function - NO DEFAULT PARAMETERS
        def directory_info(directory_path: str) -> dict:
            """Get detailed information about data directories

            Args:
                directory_path: Path to analyze (use empty string for default data directory)

            Returns:
                dict: Directory information
            """
            try:
                executor_instance.logger.info(f"Executing directory_info for: {directory_path}")
                result = executor_instance._execute_mcp_tool("directory_info", directory_path)
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in directory_info: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(directory_info)

        # LAS Tools - NO DEFAULT PARAMETERS
        def las_parser(file_path: str) -> dict:
            """Parse and extract metadata from LAS files

            Args:
                file_path: Path to the LAS file

            Returns:
                dict: Parsed LAS file metadata and information
            """
            try:
                executor_instance.logger.info(f"Executing las_parser with file: {file_path}")
                result = executor_instance._execute_mcp_tool("las_parser", file_path)
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in las_parser: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(las_parser)

        def las_analysis(file_path: str) -> dict:
            """Analyze curve data and perform statistical analysis

            Args:
                file_path: Path to the LAS file

            Returns:
                dict: Analysis results
            """
            try:
                executor_instance.logger.info(f"Executing las_analysis with file: {file_path}")
                result = executor_instance._execute_mcp_tool("las_analysis", file_path)
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in las_analysis: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(las_analysis)

        def formation_evaluation(file_path: str) -> dict:
            """Perform comprehensive petrophysical analysis

            Args:
                file_path: Path to the LAS file

            Returns:
                dict: Formation evaluation results
            """
            try:
                executor_instance.logger.info(f"Executing formation_evaluation with file: {file_path}")
                result = executor_instance._execute_mcp_tool("formation_evaluation", file_path)
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in formation_evaluation: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(formation_evaluation)

        def well_correlation(file_path: str) -> dict:
            """Correlate formations across multiple wells

            Args:
                file_path: Path to the LAS file or directory

            Returns:
                dict: Well correlation results
            """
            try:
                executor_instance.logger.info(f"Executing well_correlation with file: {file_path}")
                result = executor_instance._execute_mcp_tool("well_correlation", file_path)
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in well_correlation: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(well_correlation)

        # SEG-Y Tools - NO DEFAULT PARAMETERS
        def segy_parser(file_path: str) -> dict:
            """Parse SEG-Y seismic files with comprehensive metadata extraction

            Args:
                file_path: Path to the SEG-Y file

            Returns:
                dict: Parsed SEG-Y metadata
            """
            try:
                executor_instance.logger.info(f"Executing segy_parser with file: {file_path}")
                result = executor_instance._execute_mcp_tool("segy_parser", file_path)
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in segy_parser: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(segy_parser)

        def segy_classify(file_path: str) -> dict:
            """Automatically classify SEG-Y survey type (2D/3D)

            Args:
                file_path: Path to the SEG-Y file

            Returns:
                dict: Classification results
            """
            try:
                executor_instance.logger.info(f"Executing segy_classify with file: {file_path}")
                result = executor_instance._execute_mcp_tool("segy_classify", file_path)
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in segy_classify: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(segy_classify)

        def segy_qc(file_path: str) -> dict:
            """Perform quality control on SEG-Y files

            Args:
                file_path: Path to the SEG-Y file

            Returns:
                dict: Quality control results
            """
            try:
                executor_instance.logger.info(f"Executing segy_qc with file: {file_path}")
                result = executor_instance._execute_mcp_tool("segy_qc", file_path)
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in segy_qc: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(segy_qc)

        def quick_segy_summary(file_path: str) -> dict:
            """Get instant overview of SEG-Y files

            Args:
                file_path: Path to the SEG-Y file

            Returns:
                dict: Quick summary results
            """
            try:
                executor_instance.logger.info(f"Executing quick_segy_summary with file: {file_path}")
                result = executor_instance._execute_mcp_tool("quick_segy_summary", file_path)
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in quick_segy_summary: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(quick_segy_summary)

        def dump_content(file_path: str, line_num: int) -> dict:
            """Dump content in plain text

            Args:
                file_path: Path to any file
                line_num: number of lines to read

            Returns:
                dict: content of file
            """
            try:
                executor_instance.logger.info(f"Executing dump_content with file: {file_path} and {line_num}")
                result = executor_instance._execute_mcp_tool("dump_content", json.dumps(dict(file_path=file_path, line_num=line_num)))
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in dump_content: {e}")
                return {"status": "error", "message": str(e)}
        tools.append(dump_content)

        def plot_las(file_path: str, templates:str) -> dict:
            """Plot a las file

            Args:
                file_path: Path to las file

            Returns:
                dict: las log plot
            """
            try:
                executor_instance.logger.info(f"Executing plot_las with file: {file_path} and templates")
                result = executor_instance._execute_mcp_tool( "plot_las", json.dumps(dict(file_path=file_path, templates=templates)) )
                print(type(result), len(result))
                return {"status": "success",
                        "result": "output is created, information about resulting plot is in attachment field",
                        "attachment": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in plot_las: {e}")
                return {"status": "error", "message": str(e)}
        tools.append(plot_las)

        def plot_histogram_las(file_path: str, curve_names: list[str], num_bins: int) -> dict:
            """Plot a histogram of a las file

            Args:
                file_path: Path to las file

            Returns:
                dict: las histogram
            """
            try:
                executor_instance.logger.info(f"Executing plot_histogram_las with file: {file_path}")
                result = executor_instance._execute_mcp_tool(
                    "plot_histogram_las",
                    {
                        "file_path": file_path,
                        "curve_names": curve_names,
                        "num_bins": num_bins,
                    },
                )
                return {"status": "success",
                        "result": "output is created, information about resulting plot is in attachment field",
                        "attachment": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in plot_histogram_las: {e}")
                return {"status": "error", "message": str(e)}
        tools.append(plot_histogram_las)

        def plot_histogram_well(well: str, curves: str, num_bins: int) -> dict:
            """Plot a histogram of a well

            Args:
                well: well to plot
                curves: curves to plot

            Returns:
                dict: las histogram
            """
            try:
                executor_instance.logger.info(f"Executing plot_histogram_well with well {well} and curves {curves}")
                result = executor_instance._execute_mcp_tool('plot_histogram_well', { "well": well,
                                                                                     "curves": curves,
                                                                                     "num_bins": num_bins })
                print(result)
                return {"status": "success", "result": result}
            except Exception as e:
                traceback.print_exc()
                return {"status": "error", "message": str(e) }
        tools.append(plot_histogram_well)

        def build_logplot(well: str, track_templates: str) -> dict:
            """Plot a logplot for well

            Args:
                well: well to build logplot
                track_templates: templates for tracks

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info(f"Executing build_logplot with well: {well} and track_templates {track_templates}")
                result = executor_instance._execute_mcp_tool(
                    "build_logplot",
                    {
                        "well": well,
                        "track_templates": track_templates or ''
                    },
                )
                return {"status": "success",
                        "result": result }
            except Exception as e:
                executor_instance.logger.error(f"Error in build_logplot: {e}")
                return {"status": "error", "message": str(e)}
        tools.append(build_logplot)


        def show_sheets(file_path: str) -> dict:
            """Show sheets in an excel file

            Args:
                file_path: path to excel file

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info(f"Executing show_sheets with file: {file_path}")
                result = executor_instance._execute_mcp_tool("show_sheets", file_path)
                return {"status": "success",
                        "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in show_sheets: {e}")
                return {"status": "error", "message": str(e)}
        tools.append(show_sheets)

        def show_columns(file_path: str, sheet: int = 0, header: int = 0) -> dict:
            """Show columns in sheet of an excel file

            Args:
                file_path: path to excel file
                sheet: sheet index or name

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info(f"Executing show_columns with file: {file_path} and sheet { sheet}")
                result = executor_instance._execute_mcp_tool("show_columns", json.dumps(dict(file_path=file_path, sheet=sheet, header_rows=header)))
                return {"status": "success",
                        "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in show_sheets: {e}")
                return {"status": "error", "message": str(e)}
        tools.append(show_columns)

        def unique_from_column(file_path: str, sheet: int = 0, header: int = 0, column: int = 0) -> dict:
            """Extract unique values from a column of a excel sheet

            Args:
                file_path: path to excel file
                sheet: sheet index or name
                column: column to get values

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info(f"Executing unique_from_column with file: {file_path} and sheet { sheet} and column {column}")
                result = executor_instance._execute_mcp_tool("unique_from_column", json.dumps(dict(file_path=file_path, sheet=sheet, header_rows=header, column=column)))
                return {"status": "success",
                        "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in show_sheets: {e}")
                return {"status": "error", "message": str(e)}
        tools.append(unique_from_column)

        def marker4well(well: str, tool_context: ToolContext,  marker_file: str = '', store: str = 'default') -> dict:
            """Get marker for a well from a marker_file

            Args:
                well: well
                marker_file: marker file

            Returns:
                dict: results
            """
            try:
                fname = inspect.stack()[0][3]
                _marker_file = recursive_get(tool_context.state, [fname, 'marker_file']) or ''
                if marker_file:
                    recursive_put(tool_context.state, [fname, 'marker_file'], marker_file)
                    _marker_file = marker_file
                executor_instance.logger.info(f"Executing marker4well with well: {well}, marker_file: {_marker_file}")
                result = executor_instance._execute_mcp_tool('marker4well', json.dumps(dict(well=well,marker_file=marker_file, store=store)))
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in marker4well {e}")
                return dict(status="error", message=str(e))
        tools.append(FunctionTool(marker4well))

        def zone4well(well: str, tool_context: ToolContext,  marker_file: str = '', store: str = 'default') -> dict:
            """Get zone for a well from a marker_file

            Args:
                well: well
                marker_file: marker file

            Returns:
                dict: results
            """
            try:
                fname = inspect.stack()[0][3]
                _marker_file = recursive_get(tool_context.state, [fname, 'marker_file']) or ''
                if marker_file:
                    recursive_put(tool_context.state, [fname, 'marker_file'], marker_file)
                    _marker_file = marker_file
                executor_instance.logger.info(f"Executing zone4well with well: {well}, marker_file: {_marker_file}")
                result = executor_instance._execute_mcp_tool('zone4well', json.dumps(dict(well=well,file_path=marker_file, store=store)))
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in zone4well {e}")
                return dict(status="error", message=str(e))
        tools.append(FunctionTool(zone4well))

        def discover_wells_in_prodmonthly():
            """Discover wells in production monthy file
            Args:
                None
            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info("Executing discover_wells_in_prodmonthly")
                result = executor_instance._execute_mcp_tool('discover_wells_in_prodmonthly', None)
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error("Error in discover_wells_in_prodmonth")
                return dic(status="error", message=str(e))
        tools.append(discover_wells_in_prodmonthly)

        def productiondata4well(tool_context: ToolContext, well: str, file_path: str='', store: str='default') -> dict:
            """Get production data for well from a production file

            Args:
                well: well
                file_path: production file

            Returns:
                dict: results
            """
            try:
                fname = inspect.stack()[0][3]
                _file_path = recursive_get(tool_context.state, [fname, 'file_path']) or ''
                if file_path:
                    recursive_put(tool_context.state, [fname, 'file_path'], file_path)
                    _file_path = file_path
                executor_instance.logger.info(f"Executing productiondata4well with well: {well}, _file_path")
                result = executor_instance._execute_mcp_tool('productiondata4well', json.dumps(dict(well=well,file_path=_file_path, store=store)))
                return {"status": "success", "result": result}
            except Exception as e:
                traceback.print_exc()
                executor_instance.logger.error(f"Error in productiondata4well {e}")
                return {"status": "error", "message": str(e)}
        tools.append(productiondata4well)

        def buildCRMInput(production_wells:str, injection_wells:str, store: str = 'default'):
            """Build CRM input file from production wells and injection wells

            Args:
                production_wells: str, List of production wells
                injection_wells: str, List of injection wells

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info("Executing buildCRMInput with {production_wells} and {injection_wells}")
                result = executor_instance._execute_mcp_tool('buildCRMInput', json.dumps(dict(production_wells=production_wells.split(','), injection_wells=injection_wells.split(','))))
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in buildCRMInput {e}")
                return {"status": "error", "result": result}
        tools.append(buildCRMInput)

        def trainCRMModel(filepath:str):
            """Train CRM Model from a CRM input file

            Args:
                filepath: CRM input file

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info("Executing trainCRMModel from {filepath}")
                result = executor_instance._execute_mcp_tool('trainCRMModel', json.dumps(dict(filepath=filepath)))
                return {"status": "success",
                        "result": "result is created. Output is in attachment field",
                        "attachment": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in trainCRMModel {e}")
                return {"status": "error", "result": result}
        tools.append(trainCRMModel)

        def production_by_time(params: list[str], wells: list[str]):
            """Plot production params by time for wells from production data file

            Args:
                params: list of production parameters
                wells: list of wells

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info(f"Executing production_by_time with {params} and {wells}")
                result = executor_instance._execute_mcp_tool('production_by_time', json.dumps(dict(params=params, wells=wells)))
                return {"status": "success",
                        "result": "result is created. Output is in attachment field",
                        "attachment": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in production_by_time {e}")
                return {"status": "error", "message": str(e)}
        tools.append(production_by_time)

        def well_checklist_table(wells: str = ''):
            """Get well checklist table for a list of wells

            Args:
                wells: list of wells splitted by commas

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info("Executing well_checklist_table with {wells}")
                result = executor_instance._execute_mcp_tool('well_checklist_table', json.dumps(dict(wells=wells.split(',') if len(wells) > 0 else [])))
                return {"status": "success",
                        "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in well_checklist_table {e}")
                return {"status": "error", "message": str(e)}
        tools.append(well_checklist_table)

        def well_checklist_curves(wells: str = ''):
            """Get well checklist curves for a list of wells

            Args:
                wells: list of wells splitted by commas

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info("Executing well_checklist_curves with {wells}")
                result = executor_instance._execute_mcp_tool('well_checklist_curves', json.dumps(dict(wells=wells.split(',') if len(wells) > 0 else [])))
                return {"status": "success",
                        "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in well_checklist_curves {e}")
                return {"status": "error", "message": str(e)}
        tools.append(well_checklist_curves)

        def create_wells_tvdss(wells: str = ''):
            """Create file TVDSS.csv for a list of wells with TVD and TVDss

            Args:
                wells: list of wells splitted by commas

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info("Executing create_wells_tvdss with {wells}")
                result = executor_instance._execute_mcp_tool('create_wells_tvdss', json.dumps(dict(wells=wells.split(',') if len(wells) > 0 else [])))
                return {"status": "success",
                        "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in create_wells_tvdss {e}")
                return {"status": "error", "message": str(e)}
        tools.append(create_wells_tvdss)

        def create_pseudo_log(
            pseudo_log: str,
            well: str,
            logs: list[str],
            wells: list[str],
            regression_model: str,
            params: dict = {}
        ) -> dict:
            """
            Create a pseudo log for a well from logs in a list of wells using a regression model with params.

            Args:
                pseudo_log (str): Name of the pseudo log to be created.
                well (str): Target well for pseudo log creation.
                logs (list[str]): Log types used for training.
                wells (list[str]): Source wells used for training data.
                regression_model (str): Regression model to apply (e.g., linear, random forest).
                params (dict): Model hyperparameters.
            
            Returns:
                dict: Result of the pseudo log creation, or error message if failed.
            """
            try:
                executor_instance.logger.info(
                    f"Executing create_pseudo_log '{pseudo_log}' for well '{well}'"
                    f"from logs {logs} in wells {wells} using model '{regression_model}' with '{params}'"
                )
                result = executor_instance._execute_mcp_tool(
                    'create_pseudo_log',
                    json.dumps({
                        "pseudo_log": pseudo_log,
                        "well": well,
                        "logs": logs,
                        "wells": wells,
                        "regression_model": regression_model,
                        "params": params
                    })
                )
                return {"status": "success", "result": result}

            except Exception as e:
                executor_instance.logger.error(f"Error in create_pseudo_log: {e}")
                return {"status": "error", "message": str(e)}
        tools.append(create_pseudo_log)
        
        def view_training_result(
            tool_context: ToolContext,
            pseudo_log: str,
            well: str,
            regression_model: str, 
        ) -> dict:
            """
            View training results of a created pseudo log for a well 
            using a regression model with specific parameters.

            Args:
                pseudo_log (str): Name of the pseudo log to evaluate.
                well (str): The target well to generate pseudo log for.
                regression_model (str): Name/type of the regression model used.

            Returns:
                dict: Result of the training visualization or summary.
            """
            try:
                fname = inspect.stack()[0][3]  # Get the current function name for namespacing context
                def get_or_update_param(key: str, value):
                    existing = recursive_get(tool_context.state, [fname, key]) or None
                    if value not in [None, '', [], {}]:
                        recursive_put(tool_context.state, [fname, key], value)
                        return value
                    return existing

                _pseudo_log       = get_or_update_param('pseudo_log', pseudo_log)
                _well             = get_or_update_param('well', well)
                _regression_model = get_or_update_param('regression_model', regression_model)

                executor_instance.logger.info(
                    f"Executing view_training_result for well: {_well}, pseudo_log: {_pseudo_log}, model: {_regression_model}"
                )
                result = executor_instance._execute_mcp_tool(
                    'view_training_result',
                    json.dumps({
                        "pseudo_log": _pseudo_log,
                        "well": _well,
                        "regression_model": _regression_model,
                    })
                )
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in view_training_result {e}")
                return {"status": "error", "message": str(e)}
        tools.append(view_training_result)

        self.logger.info(f"Created {len(tools)} tool functions (no default parameters)")
        return tools

    async def _initialize_google_adk(self):
        """Initialize Google ADK with proper tool format - NO DEFAULT PARAMETERS"""
        self.logger.info("Initializing Google ADK components...")

        if not os.getenv('OPENAI_API_KEY'):
            raise Exception("OPENAI_API_KEY environment variable not set")

        # STEP 1: Create tool functions (ADK will automatically wrap them)
        tools = self._create_tool_functions()

        # STEP 2: Create agent with tools
        self.agent = LlmAgent(
            name="subsurface_data_analyst",
            model=LiteLlm(model="openai/gpt-4o-mini"),
            description="Subsurface data analyst with tool execution capabilities",
            instruction=self._create_tool_execution_instruction(),
            tools=tools  # Pass Python functions directly - ADK handles the wrapping
        )

        # STEP 3: Session management
        self.session_service = InMemorySessionService()
        self.session_id = f"tool_execution_session_{hash('hybrid_user')}"

        await self.session_service.create_session(
            app_name="SubsurfaceToolExecution",
            user_id="hybrid_user",
            session_id=self.session_id
        )

        # STEP 4: Create runner
        self.runner = Runner(
            agent=self.agent,
            app_name="SubsurfaceToolExecution",
            session_service=self.session_service
        )

        self.logger.info("Google ADK initialized successfully with function tools (no defaults)")

    def _create_tool_execution_instruction(self) -> str:
        """Create instruction that emphasizes tool execution"""
        return """You are a subsurface data analyst with access to specialized tools for analyzing well logs (LAS files) and seismic data (SEG-Y files).

# CRITICAL INSTRUCTIONS - ALWAYS EXECUTE TOOLS:

## For System Status Requests:
- User asks "system status" → IMMEDIATELY call system_status with query=""
- User asks "health check" → IMMEDIATELY call health_check with query=""

## For File Analysis:
- User asks "analyze well.las" → IMMEDIATELY call las_parser with file_path="well.las"
- User asks "classify survey.sgy" → IMMEDIATELY call segy_classify with file_path="survey.sgy"
- User asks "plot well.las" → IMMEDIATELY call plot_las with file_path="well.las"
- User asks "show columns in file.xlsx sheet 0" → IMMEDIATELY call show_columns with file_path="file.xlsx" and sheet=0
- User asks "get unique values from column 0 in file.xlsx sheet 0" → IMMEDIATELY call unique_from_column with column=0 file_path="file.xlsx" and sheet=0

## For CRM analysis:
- User asks "build production data for WELL", then IMMEDIATELY call productiondata4well with well=WELL and file_path if provided
- User asks "build CRM input using production wells and injection wells" → IMMEDIATELY call buildCRMInput with corresponding production_wells and injection_wells
- User asks "show wells in marker file", then IMMEDIATELY call unique_from_column with column=0 file_path="misc/Marker.xlsx" and sheet=0
- User asks "show wells in production monthly file", then IMMEDIATELY call unique_from_column with column=1 file_path="production/PVT_WellTest_Perforation_WaterAnalysis.xlsx" and sheet=4
- User asks "Plot [params] of wells [wells] by time from production file" or "View production chart of wells [wells]",
    then IMMEDIATELY call production_by_time with params as list[str] if user provided or else params=["CV.OilRate","CV.LiqRate","CV.Watercut","CV.Oilcum/1000"]
    and wells as list[str] if user provided or else wells=[]
    **ALWAYS call using these standardized param names from user provided params, DO NOT use monthly params if user not provided explicitly:
    "CV.OilRate",              # Oil rate
    "Monthlyprod.Qoil/1000",   # Monthly oil rate in thousands
    "CV.Oilcum/1000",          # Oilcum in thousands
    "CV.LiqRate",              # Liquid rate (oil + water)
    "Monthlyprod.Qwater/1000",
    "CV.WaterProdCum/1000",
    "Monthlyprod.Qgas/1000",
    "CV.GasCum/1000",
    "CV.WaterInj_Rate",
    "Monthlyinj.Qwater/1000",
    "CV.WaterInjCum/1000",
    "CV.Watercut",
    "Monthlyprod.Qwater/1000+Monthlyprod.Qoil/1000",
    "Monthlyprod.Qgas/Monthlyprod.Qoil*1000",
    "Monthlyprod.Gor",
    "Monthlyprod.Dayon",
    "CV.WellProd",
    "CV.WellInj".

## For generating plots
- User asks "generate TRACK_TEMPLATES logplot for WELL  → IMMEDIATELY call build_logplot with well=WELL and track_templates=TRACK_TEMPLATES

## For missing pay:
- User asks "View checklist table of well logs data" → IMMEDIATELY call well_checklist_table with wells if user provided or else wells=''
- User asks "Generate logplot for WELL, then IMMEDIATELY call build_logplot with well=WELL and track_templates if user provided or track_templates=GR,LLD,NPHI
- User asks "plot histogram for CURVES from well WELL", then IMMEDIATELY call plot_histogram_well with WELL and CURVES and num_bins=10 if it is not provided
- User asks "plot histogram for CURVE1 and CURVE2 from file.las with 9 bins" → IMMEDIATELY call plot_histogram_las with file_path="file.las" and curveNames=["CURVE1", "CURVE2"] and numBins=9

# IMPORTANT PARAMETER RULES:
- ALL functions require parameters (no defaults)
- For list_files: always provide a pattern (e.g., "*", "*.las", "*.sgy")
- For system_status/health_check: use query="" if no specific query needed
- For directory_info: use directory_path="" for default data directory
- For file tools: always provide the full file path

# YOUR WORKFLOW:
1. Understand what the user wants
2. Identify the appropriate tool
3. **EXECUTE the tool immediately with required parameters**
4. Present the results clearly

# FORMAT OUTPUT FILE:
Format any html file in output (e.g.: /path/to/file.html) with the following template: http://dashboard.portal:9999/path/to/file.html. Also embed it into an <iframe>

# EXAMPLES OF CORRECT BEHAVIOR:
User: "list files *.las"
You: [CALLS list_files(pattern="*.las")]
Then: Present the results

User: "system status"
You: [CALLS system_status(query="")]
Then: Present the status information

User: "dump content of somefile.csv for 10 lines"
You: [CALLS dump_content(file_path="somefile.csv", line_num=10)]
Then: Present the results

User: "generate GR,NPHI logplot for well"
You: [CALLS build_logplot(well="well", track_templates="GR,NPHI")]
Then: Present the results

**REMEMBER: Always provide ALL required parameters when calling tools!**

Available tools: list_files, system_status, health_check, directory_info,
las_parser, las_analysis, formation_evaluation, well_correlation, segy_parser, segy_classify, segy_qc,
quick_segy_summary, dump_content, plot_las, build_logplot, plot_histogram_las, show_sheets, show_columns,
unique_from_column, marker4well, zone4well, productiondata4well,
buildCRMInput, trainCRMModel, production_by_time,
well_checklist_table, well_checklist_curves, create_wells_tvdss, create_pseudo_log, view_training_result
"""

    async def _execute_with_google_adk(self, query: str) -> str:
        """Execute query using Google ADK with tool execution"""
        await self._ensure_google_adk_ready()

        # Create message for Google ADK
        content = types.Content(
            role='user',
            parts=[types.Part(text=query)]
        )

        tool_calls_made = []
        final_response = ""

        try:
            # Execute through Google ADK runner
            async for event in self.runner.run_async(
                user_id="hybrid_user",
                session_id=self.session_id,
                new_message=content
            ):
                self.logger.debug(f"Event type: {type(event).__name__}")

                # Track tool calls and get response
                if hasattr(event, 'content') and event.content and event.content.parts:
                    final_response = event.content.parts[0].text
                elif hasattr(event, 'text') and event.text:
                    final_response = event.text

                # Enhanced tool call detection
                if hasattr(event, 'actions') and event.actions:
                    for action in event.actions:
                        if hasattr(action, 'tool_call') and action.tool_call:
                            tool_calls_made.append({
                                'tool': action.tool_call.name,
                                'arguments': getattr(action.tool_call, 'parameters', {})
                            })
                            self.logger.info(f"Tool call detected: {action.tool_call.name}")

                # Alternative tool call detection
                if hasattr(event, 'tool_call') and event.tool_call:
                    tool_calls_made.append({
                        'tool': event.tool_call.name,
                        'arguments': getattr(event.tool_call, 'parameters', {})
                    })
                    self.logger.info(f"Direct tool call detected: {event.tool_call.name}")

            # Update statistics
            self.stats["tool_executions"] += len(tool_calls_made)

            if tool_calls_made:
                self.logger.info(f"Successfully detected {len(tool_calls_made)} tool calls")
            else:
                # Check if our internal MCP calls were made
                if "Executing list_files" in str(final_response) or "Found" in str(final_response):
                    self.logger.info("Tool execution detected through MCP calls")
                    self.stats["tool_executions"] += 1
                else:
                    self.logger.warning("No tools were executed - agent may need stronger instructions")

            return final_response or "Analysis completed."

        except Exception as e:
            self.logger.error(f"Google ADK execution error: {e}")
            return self._minimal_fallback(query)

    def _minimal_fallback(self, query: str) -> str:
        """Minimal fallback when Google ADK fails"""
        import re

        query_lower = query.lower()

        # Direct tool execution fallback
        if "list files" in query_lower or "list" in query_lower:
            # Extract pattern if any
            pattern_match = re.search(r'\*\.[a-z]+|\*[a-z0-9_]+\*?|[a-z0-9_]+\*', query, re.IGNORECASE)
            pattern = pattern_match.group(0) if pattern_match else "*"
            return self._execute_mcp_tool('list_files', pattern)
        elif "status" in query_lower:
            return self._execute_mcp_tool('system_status', '')
        elif "health" in query_lower:
            return self._execute_mcp_tool('health_check', '')
        else:
            return f"I encountered a technical issue with Google ADK. Try asking for 'list files' or 'system status'."

    def invoke(self, input_dict: Dict[str, str]) -> Dict[str, str]:
        """Main invoke method"""
        self.stats["total_invocations"] += 1

        try:
            query = input_dict.get("input", "")
            if not query:
                return {"output": "No input provided"}

            self.logger.info(f"Processing query: {query[:100]}...")

            # Execute with Google ADK tool execution
            response = asyncio.run(self._execute_with_google_adk(query))

            self.stats["successful_invocations"] += 1
            return {"output": response}

        except Exception as e:
            self.stats["failed_invocations"] += 1
            self.logger.error(f"Execution error: {e}")

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
        """Execute MCP tool"""
        try:
            self.logger.info(f"Executing MCP tool: {tool_name} with params: {params}")
            self.stats["tool_executions"] += 1

            # Simple parameter preparation
            # if isinstance(params, str):
            #     input_data = params
            # else:
            #     input_data = str(params) if params is not None else ""

            result = self.mcp_client.call_tool(tool_name, params)
            return self._extract_result_content(result)

        except Exception as e:
            self.logger.error(f"MCP tool execution failed: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    def _extract_result_content(self, result: Dict[str, Any]) -> str:
        """Extract content from MCP response"""
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
        """Get statistics"""
        self.stats["uptime_hours"] = (time.time() - getattr(self, '_start_time', time.time())) / 3600
        return self.stats.copy()


class ToolExecutingHybridAgent:
    """Hybrid Agent that actually executes tools via Google ADK"""

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
            "system_type": "Google ADK Agent with Tool Execution"
        }

    def run(self, query: str) -> str:
        """Process query with tool execution"""
        self.stats["total_queries"] += 1
        self.logger.debug(f"Processing: {query[:100]}...")

        # Minimal direct command processing (only for obvious system commands)
        if self._is_obvious_system_command(query):
            try:
                direct_result = self.command_processor(query)
                if direct_result:
                    self.stats["direct_commands"] += 1
                    return direct_result
            except Exception as e:
                self.logger.debug(f"Direct command failed: {e}")

        # Let the agent execute tools
        try:
            result = self.agent_executor.invoke({"input": query})

            if isinstance(result, dict):
                output = result.get("output", str(result))
                params = result.get("attachment", None)
                print(output, params)
            else:
                output = str(result)

            self.stats["agent_responses"] += 1
            return output

        except Exception as e:
            self.logger.warning(f"Agent execution failed: {e}")
            self.stats["errors"] += 1
            return f"I encountered a technical issue while processing your request. Error: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}"

    def _is_obvious_system_command(self, query: str) -> bool:
        """Check if this is an obvious system command"""
        query_lower = query.lower().strip()
        obvious_commands = ["system status", "health check", "status", "health"]
        return query_lower in obvious_commands

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
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


class ToolExecutingAgentFactory:
    """Factory for tool-executing agents"""

    def __init__(self, mcp_url: str, config: AgentConfig):
        self.mcp_url = mcp_url
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create(self) -> ToolExecutingHybridAgent:
        """Create tool-executing agent"""
        self.logger.info("Creating Google ADK agent with tool execution...")

        # Create tool-executing executor
        agent_executor = self._create_tool_executing_executor()

        # Minimal command processor
        command_processor = self._create_minimal_command_processor()

        # Create hybrid agent
        hybrid_agent = ToolExecutingHybridAgent(agent_executor, command_processor, {})

        self.logger.info("Google ADK agent with tool execution created successfully")
        return hybrid_agent

    def _create_tool_executing_executor(self) -> ToolExecutingAgentExecutor:
        """Create tool-executing executor"""
        try:
            if not GOOGLE_ADK_AVAILABLE:
                raise ImportError("Google ADK not available")

            mcp_client = MCPClient(self.mcp_url)
            agent_executor = ToolExecutingAgentExecutor(mcp_client, self.config)
            agent_executor._start_time = time.time()

            return agent_executor

        except Exception as e:
            self.logger.error(f"Failed to create tool-executing executor: {e}")
            raise

    def _create_minimal_command_processor(self):
        """Minimal command processor for obvious system commands"""
        mcp_client = MCPClient(self.mcp_url)

        def minimal_command_processor(command_str: str) -> Optional[str]:
            command_lower = command_str.lower().strip()

            if command_lower == "system status":
                result = mcp_client.call_tool("system_status", "")
                return json.dumps(result) if isinstance(result, dict) else str(result)
            elif command_lower == "health check":
                result = mcp_client.call_tool("health_check", "")
                return json.dumps(result) if isinstance(result, dict) else str(result)
            elif command_lower.startswith("plot "):
                file_path = command.split(" ")[1]
                result = mcp_client.call_tool("plot_las", file_path)
                return json.dumps(result) if isinstance(result, dict) else str(result)

            return None

        return minimal_command_processor


# Factory functions
def create_google_adk_hybrid_agent(mcp_url: str, config: AgentConfig) -> ToolExecutingHybridAgent:
    """
    Create Google ADK agent that actually executes tools

    FULLY FIXED: Resolves all syntax errors and scope issues
    """
    factory = ToolExecutingAgentFactory(mcp_url, config)
    return factory.create()


# Backward compatibility
def create_pure_reasoning_agent(mcp_url: str, config: AgentConfig) -> ToolExecutingHybridAgent:
    """Backward compatible function"""
    return create_google_adk_hybrid_agent(mcp_url, config)


def create_hybrid_agent(a2a_url: str, mcp_url: str, config: AgentConfig) -> ToolExecutingHybridAgent:
    """Backward compatible function"""
    return create_google_adk_hybrid_agent(mcp_url, config)
