"""
Google ADK Agent - FULLY FIXED VERSION
Fixes all syntax errors, scope issues, and method access problems
"""
import traceback
import re
import os
import json
import time
import logging
import inspect
import asyncio
import numpy as np
from typing import Optional, Dict, Any, List, Union
from glob import iglob

import urllib
import urllib.parse

# Google ADK imports
try:
    from google.adk.agents import Agent, LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.tools import FunctionTool, BaseTool, ToolContext
    from google.adk.events import Event
    from google.genai import types
    GOOGLE_ADK_AVAILABLE = True
except ImportError as e:
    GOOGLE_ADK_AVAILABLE = False

from config.settings import AgentConfig
from servers.mcp_server import MCPClient
from naming import Naming
from base_utils import recursive_get, recursive_put, iframe, link, PUBLISH_BASE
from context import Context
from xlsx_utils import XLSX
logger = logging.getLogger(__name__)

AGENT_URL = os.getenv("AGENT_URL") or "http://localhost:8990"

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

        def calc_distance_between_wells(tool_context: ToolContext, well1:str, well2:str) -> dict:
            """Calculate distance between well1 and well2

            Args:
                well1: first well
                well2: second well

            Returns:
                dict: results
            """
            try:
                wellpos_df = XLSX.extract_wellpos([well1, well2])
                if len(wellpos_df.index) == 1:
                    result = f"Only {wellpos_df['Well'].values[0]} found."
                elif len(wellpos_df.index) == 0:
                    result = "None of the wells found"
                else:
                    wells = wellpos_df[['X', 'Y']].values
                    distance = np.sqrt(np.sum(np.square(wells[0] - wells[1])))
                    result = f"Distance = {distance:.2f} meters"

                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in list_files: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(calc_distance_between_wells)

        def list_wells(tool_context: ToolContext, pattern: str) -> dict:
            """List wells matching pattern in the data directory

            Args:
                pattern: pattern to match 

            Returns:
                dict: Results containing matched wells
            """
            try:
                basepath = Naming.well_path()
                wells = [ os.path.basename(d) for d in iglob(f"{basepath}/{pattern or '*'}") ]
                tool_context.actions.skip_summarization = True
                result = "No wells found"
                if len(wells):
                    result = "Here is the list of wells:\n" + ( "\n- ".join(wells) )
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in list_files: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(list_wells)
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

        def plot_las(tool_context:ToolContext, file_path: str, templates:str = '') -> dict:
            """Plot a las file

            Args:
                file_path: Path to las file
                templates: templates for tracks

            Returns:
                dict: las log plot
            """
            try:
                executor_instance.logger.info(f"Executing plot_las with file: {file_path} and templates")
                result = executor_instance._execute_mcp_tool( "plot_las", json.dumps(dict(file_path=file_path, templates=templates)) )
                print("MMM", type(result), len(result), result)
                tool_context.actions.skip_summarization = True
                return {"status": "success",
                        "skip_summarization": True,
                        "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in plot_las: {e}")
                return {"status": "error", "message": str(e)}
        tools.append(plot_las)

        def well_logplot(tool_context:ToolContext, well: str, curves: list[str] = []) -> dict:
            """Plot a logplot for curves in a single well.
            
            Args:
                well (str): input well
                curves (list[str]): list of curves to plot. If curves is empty, plot all curves in well

            Return:
                dict: result
            """
            try:
                executor_instance.logger.info(f"Executing well_logplot with well: {well} and curves {curves}")
                result = executor_instance._execute_mcp_tool( "well_logplot", json.dumps(dict(well=well, curves=curves)) )
                tool_context.actions.skip_summarization = True
                return {"status": "success",
                        "skip_summarization": True,
                        "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in well_logplot: {e}")
                return {"status": "error", "message": str(e)}
    
        tools.append(well_logplot)

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
                        "skip_summarization": True,
                        "result": result}
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
                return {"status": "success", "skip_summarization": True, "result": result}
            except Exception as e:
                traceback.print_exc()
                return {"status": "error", "message": str(e) }
        tools.append(plot_histogram_well)

        def build_logplot(well: str, track_templates: str = 'GR,LLD,NPHI') -> dict:
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

                print("MMM", result, type(result))
                return {"status": "success",
                        "skip_summarization": True,
                        "result": f"{result}" }
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
                if not file_path:
                    file_path = Naming.default_production_monthly_file(category='raw')
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
                return {"status": "success", "skip_summarization": True, "result": result}
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

        def discover_wells_in_prodmonthly(tool_context: ToolContext):
            """Discover wells in production monthy file
            Args:
                None
            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info("Executing discover_wells_in_prodmonthly")
                result = executor_instance._execute_mcp_tool('discover_wells_in_prodmonthly', None)
                tool_context.actions.skip_summarization = True
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error("Error in discover_wells_in_prodmonth")
                return dic(status="error", message=str(e))
        tools.append(discover_wells_in_prodmonthly)

#        def productiondata4well(tool_context: ToolContext, well: str, file_path: str='', store: str='default') -> dict:
#            """Get production data for well from a production file
#
#            Args:
#                well: well
#                file_path: production file
#
#            Returns:
#                dict: results
#            """
#            try:
#                fname = inspect.stack()[0][3]
#                _file_path = recursive_get(tool_context.state, [fname, 'file_path']) or ''
#                if file_path:
#                    recursive_put(tool_context.state, [fname, 'file_path'], file_path)
#                    _file_path = file_path
#                executor_instance.logger.info(f"Executing productiondata4well with well: {well}, _file_path")
#                result = executor_instance._execute_mcp_tool('productiondata4well', json.dumps(dict(well=well,file_path=_file_path, store=store)))
#                return {"status": "success", "result": result}
#            except Exception as e:
#                traceback.print_exc()
#                executor_instance.logger.error(f"Error in productiondata4well {e}")
#                return {"status": "error", "message": str(e)}
#        tools.append(productiondata4well)
#

        def describe_production_data(wells: list[str]):
            """Describe production data for input wells

            Args:
                wells: input wells to describe production data

            Returns:
                dict: result
            """
            try:
                executor_instance.logger.info(f"Executing describe_production_data with wells: {wells}")
                result = executor_instance._execute_mcp_tool('describe_production_data', json.dumps(wells))
                return {"status": "success", "skip_summarization": True,
                        "result": result}
            except Exception as e:
                traceback.print_exc()
                executor_instance.logger.error(f"Error in describe_production_data {e}")
                return {"status": "error", "message": str(e)}
        tools.append(describe_production_data)

        def production_monthly_data_table(wells:str = ""):
            """Build and show production monthly data table for wells

            Args:
                wells: wells to include in the production monthly data table

            Returns:
                dict: result
            """
            try:
                executor_instance.logger.info(f"Executing production_monthly_data_table with wells: {wells}")
                result = executor_instance._execute_mcp_tool('production_monthly_data_table', wells)
                print("RESULT:", result)
                return {"status": "success", "skip_summarization": True, "result": result}
            except Exception as e:
                traceback.print_exc()
                executor_instance.logger.error(f"Error in production_monthly_data_table {e}")
                return {"status": "error", "message": str(e)}

        tools.append(production_monthly_data_table)

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

        def train_crm_model(tool_context: ToolContext, i_wells:list[str], o_wells:list[str]) -> dict:
            """Train CRM Model from injection wells i_wells and production wells o_wells
            
            Args:
                i_wells (list[str]): injection wells
                o_wells (list[str]): production wells

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info("Executing train_crm_model from injection wells {i_wells} and production wells {o_wells}")
                result = executor_instance._execute_mcp_tool('train_crm_model', json.dumps(dict(i_wells=i_wells, o_wells=o_wells)))
                tool_context.actions.skip_summarization = True
                return {"status": "success",
                        "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in train_crm_model: {e}")
                return {"status": "error", "message": str(e)}
        tools.append(train_crm_model)

        def production_crossplot(params: list[str], wells: list[str], xparam: str):
            """Plot production params by oilcum for wells from production data file

            Args:
                params: list of production parameters
                wells: list of wells

            Returns:
                dict: results
            """
            try:
                fname = inspect.stack()[0][3]
                print(fname)
                modes = Context.get_instance().get([fname, 'modes'])
                if modes is None or len(modes) == 0:
                    modes = "markers"
                print(modes)
                
                modes = modes.split(",")
                modes = [m.strip() for m in modes]
                executor_instance.logger.info(f"Executing production_crossplot with {fname} {params} and {wells} {xparam}")
                result = executor_instance._execute_mcp_tool('production_crossplot', json.dumps(dict(params=params, 
                                                                                                     wells=wells, 
                                                                                                     xparam=xparam,  modes=modes)))
                return {"status": "success",
                        "skip_summarization": True,
                        "result": result }
            except Exception as e:
                traceback.print_exc()
                executor_instance.logger.error(f"Error in production_crossplot {e}")
                return {"status": "error", "message": str(e)}
        tools.append(production_crossplot)
            
        def production_by_oilcum(params: list[str], wells: list[str]):
            """Plot production params by oilcum for wells from production data file

            Args:
                params: list of production parameters
                wells: list of wells

            Returns:
                dict: results
            """
            try:
                fname = inspect.stack()[0][3]
                print(fname)
                modes = Context.get_instance().get([fname, 'modes'])
                if modes is None or len(modes) == 0:
                    modes = "markers"
                print(modes)
                
                modes = modes.split(",")
                modes = [m.strip() for m in modes]
                executor_instance.logger.info(f"Executing production_by_oilcum with {fname} {params} and {wells}")
                executor_instance.logger.info(f"Executing production_by_oilcum with {params} and {wells}")
                result = executor_instance._execute_mcp_tool('production_by_oilcum', json.dumps(dict(params=params, wells=wells, modes=modes)))
                return {"status": "success",
                        "skip_summarization": True,
                        "result": result }
            except Exception as e:
                executor_instance.logger.error(f"Error in production_by_oilcum {e}")
                return {"status": "error", "message": str(e)}
        tools.append(production_by_oilcum)

        def production_by_time(params: list[str], wells: list[str]):
            """Plot production params by time for wells from production data file

            Args:
                params: list of production parameters
                wells: list of wells

            Returns:
                dict: results
            """
            try:
                fname = inspect.stack()[0][3]
                modes = Context.get_instance().get( [fname, 'modes'] )
                if not modes:
                    modes = "markers"
                print(modes)
                modes = modes.split(",")
                modes = [m.strip() for m in modes]
                executor_instance.logger.info(f"Executing production_by_time with {params} and {wells}")
                result = executor_instance._execute_mcp_tool('production_by_time', json.dumps(dict(params=params, wells=wells, modes=modes)))
                return {"status": "success",
                        "skip_summarization": True,
                        "result": result}
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

        def plt_table(tool_context: ToolContext) -> dict:
            """View PLT table
            Args:
                None

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info("Executing plt_table")
                result = executor_instance._execute_mcp_tool('plt_table', 'notused')
                tool_context.actions.skip_summarization = True
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in plt_table {e}")
                return {"status": "error", "message": str(e)}
        tools.append(plt_table)

        def suggest_log_creation(tool_context: ToolContext, target_curve:str, target_well: str, wells: list[str]):
            """
            Suggest a model for creating a target_curve in target_well using input wells. If input wells is empty, find suitable input wells

            Args:
                taget_well (str): Target well for curve creation.
                wells (list[str]): Source wells used for training data. wells is empty means find suitable input wells
            
            Returns:
                dict: Result of the pseudo log creation, or error message if failed.
            """
            try:
                executor_instance.logger.info(f"Executing suggest_log_creation for curve {target_curve} in well '{target_well}' from '{wells}'")
                result = executor_instance._execute_mcp_tool(
                    'suggest_log_creation',
                    json.dumps({
                        "target_curve": target_curve,
                        "target_well": target_well,
                        "wells": wells
                    })
                )
                tool_context.actions.skip_summarization = True
                return {"status": "success", "result": result}

            except Exception as e:
                executor_instance.logger.error(f"Error in create_pseudo_log: {e}")
                return {"status": "error", "message": str(e)}

        tools.append(suggest_log_creation)

        def create_pseudo_log(
            tool_context: ToolContext,
            target_curve: str,
            target_well: str,
            curves: list[str],
            wells: list[str],
            model_type: str,
            model_params: dict = {},
        ) -> dict:
            """
            Generate a curve for a well from curves in a list of wells using a machine learning model with model parameters.

            Args:
                target_curve (str): Name of the curve to be created.
                taget_well (str): Target well for curve creation.
                curves (list[str]): Curve types used for training.
                wells (list[str]): Source wells used for training data.
                model_type (str): Type of machine learning model to use. Supported:
                    - 'linear' for Linear Regression
                    - 'random_forest' for Random Forest Regressor
                    - 'neural_network' for Multi-layer Perceptron Regressor
                model_params (dict, optional): Dictionary of model parameters. Example:
                    - For 'neural_network': {"hidden_layer_sizes": [50, 20], "max_iter": 200}
                    - For 'random_forest': {"n_estimators": 100, "max_depth": 10}
            
            Returns:
                dict: Result of the pseudo log creation, or error message if failed.
            """
            try:
                executor_instance.logger.info(
                    f"Executing create_pseudo_log of curve '{target_curve}' for well '{target_well}'"
                    f"from curves {curves} in wells {wells} using model '{model_type}' with params '{model_params}'"
                )
                result = executor_instance._execute_mcp_tool(
                    'create_pseudo_log',
                    json.dumps({
                        "target_curve": target_curve,
                        "target_well": target_well,
                        "curves": curves,
                        "wells": wells,
                        "model_type": model_type,
                        "model_params": model_params
                    })
                )
                tool_context.actions.skip_summarization = True
                return {"status": "success", "result": result}

            except Exception as e:
                executor_instance.logger.error(f"Error in create_pseudo_log: {e}")
                return {"status": "error", "message": str(e)}
        tools.append(create_pseudo_log)
        
        def view_training_experiment(
            tool_context: ToolContext,
            target_curve: str = '',
            target_well: str = '',
            model_type: str = '', 
            seconds: int = 0, 
            filter_expr: str = ''
        ) -> dict:
            """
            View training experiments of a curve for a well using a machine learning model 
            with a time range specified in miliseconds ago and a filter expression.

            Args:
                target_curve (str, optional): Name of the curve to evaluate.
                target_well (str, optional): The target well to generate the curve for.
                model_type (str, optional): Type of machine learning model to use. 
                    Supported values:
                        - 'linear' for Linear Regression
                        - 'random_forest' for Random Forest Regressor
                        - 'neural_network' for Multi-layer Perceptron Regressor
                
                seconds (int, optional): Time since experiment creation, in seconds.  
                    Used to normalize relative time expressions (e.g., "last minutes", "last days", "last years") to a common unit: seconds.

                filter_expr (str, optional): A filter expression to filter experiments.
                    Supported fields:
                        - r2 (accuracy): R-squared
                        - mape, rmse (loss): MAPE, RMSE 

                     Examples:
                        - "r2 > 0.85 and loss < 10"
                        - "mape <= 5"
                        - "accuracy >= 0.9 or loss < 7"

            Returns:
                dict: Result of the training visualization or summary.
            """
            try:
                executor_instance.logger.info(
                    f"Executing view_training_experiment of curve {target_curve} for well {target_well} "
                    f"using model {model_type}, filtering last {seconds} seconds with expression {filter_expr}"
                )
                result = executor_instance._execute_mcp_tool(
                    'view_training_experiment',
                    json.dumps({
                        "target_curve": target_curve,
                        "target_well": target_well,
                        "model_type": model_type,
                        "seconds": seconds,
                        "filter_expr": filter_expr,
                    })
                )
                tool_context.actions.skip_summarization = True
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in view_training_experiment {e}")
                return {"status": "error", "message": str(e)}
        tools.append(view_training_experiment)
        
        def view_wf_experiment(
            tool_context: ToolContext,
            iwells: list[str] = [],
            owells: list[str] = [],
            model_type: str = 'CRM',
            seconds: int = 0, 
            filter_expr: str = ''
        ) -> dict:
            """
            View water flooding (or wf) experiments of injection wells (iwells) and production wells (owells) for a well using a model type and
            a time range specified in miliseconds ago and a filter expression.

            Args:
                iwells (str, optional): List of injection wells.
                owells (str, optional): List of production wells.
                model_type (str, optional): Type of machine learning model to use. 
                    Supported values:
                        - 'CRM' for Capacitance Registance Model
                        - 'LSTM' for Long Short Term Memory
                
                seconds (int, optional): Time since experiment creation, in seconds.  
                    Used to normalize relative time expressions (e.g., "last minutes", "last days", "last years") to a common unit: seconds.

                filter_expr (str, optional): A filter expression to filter experiments.

            Returns:
                dict: Result.
            """
            try:
                executor_instance.logger.info(
                    f"Executing view_wf_experiment of injection {iwells} and production {owells} "
                    f"using model {model_type}, filtering last {seconds} seconds with expression {filter_expr}"
                )
                result = executor_instance._execute_mcp_tool(
                    'view_wf_experiment',
                    json.dumps({
                        "iwells": iwells,
                        "owells": owells,
                        "model_type": model_type,
                        "seconds": seconds,
                        "filter_expr": filter_expr,
                    })
                )
                tool_context.actions.skip_summarization = True
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in view_wf_experiment {e}")
                return {"status": "error", "message": str(e)}

        tools.append(view_wf_experiment)
        def delete_training_experiment(
            experiment_id: str,
        ) -> dict:
            """
            Delete a training experiment with experiment_id

            Args:
                experiment_id (str): The id of an experiment

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info(
                    f"Executing delete a training experiment with experiment_id {experiment_id}"
                )
                result = executor_instance._execute_mcp_tool(
                    'delete_training_experiment',
                    json.dumps({
                        "experiment_id": experiment_id,
                    })
                )
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in view_training_experiment {e}")
                return {"status": "error", "message": str(e)}
        tools.append(delete_training_experiment)

        def suggest_log_creation(tool_context: ToolContext, target_curve: str, target_well: str, wells: list[str] = []):
            """
            Suggest how to calculate a target_curve for a target_well from input wells. If input wells is empty. Find suitable input wells from existing data

            Args:
                target_curve (str): the target curve to create
                target_well (str): the target_well in which need to create the target_curve
                wells (list[str]): input wells. If it is empty, find suitable wells in existing data store

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info(
                    f"Executing suggest_log_creation for {target_curve} in {target_well} from wells {wells}"
                )
                result = executor_instance._execute_mcp_tool(
                    'suggest_log_creation',
                    json.dumps({
                        "target_curve": target_curve,
                        "target_well": target_well,
                        "wells": wells
                    })
                )
                tool_context.actions.skip_summarization = True
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in view_training_experiment {e}")
                return {"status": "error", "message": str(e)}
        tools.append(delete_training_experiment)

        def accept_experiment_las_file(tool_context: ToolContext, experiment_id:str):
            """Accept las file in experiment identified by experiment_id

            Args:
                experiment_id (str): the experiment id

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info( f"Executing accept_experiment_las_file with experiment_id {experiment_id}")
                result = executor_instance._execute_mcp_tool( 'accept_experiment_las_file', experiment_id )
                tool_context.actions.skip_summarization = True
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in view_training_experiment {e}")
                return {"status": "error", "message": str(e)}

        tools.append(accept_experiment_las_file)
        def summarize_marker_data() -> dict:
            """Summarize marker data from marker file

            Args:
                None

            Returns:
                dict: results
            """
            try:
                executor_instance.logger.info(f"Executing summarize_marker_data for all wells")
                result = executor_instance._execute_mcp_tool('summarize_marker_data', { })
                return {"status": "success", "result": result}
            except Exception as e:
                executor_instance.logger.error(f"Error in create_psuedo_log {e}")
                return {"status": "error", "message": str(e)}
        tools.append(summarize_marker_data)

        def set_context_for_tool_function(tool_function:str, context_param:str, context_value: str) -> dict:
            """ Set context parameter value for tool function

            Args:
                tool_function: the tool function name
                context_param: context parameter name
                context_value: the value to set to context_param

            Returns:
                this function save context_value to context_param for the tool_function and return result as a dictionary.
            
            Show "result" field in return value of this function AS IS to user
            """
            try:
                executor_instance.logger.info(f"Executing set_context_for_tool_function: {tool_function} {context_param}={context_value}")
                Context.get_instance().put([tool_function, context_param], context_value)
                return {"status": "success", "result": f"*{context_param}* of *{tool_function}* is set to *{context_value}*"}
            except Exception as e:
                traceback.print_exc()
                return {"status": "error", "message": str(e)}
        tools.append(set_context_for_tool_function)

        def welltest_table(tool_context: ToolContext, wells: list[str], test: str) -> dict:
            """ Show welltest table for input wells
        
            Args:
                wells: input wells
                test: is a string and can be either "production" or "injection" or "all"

            Returns:
                dict: result
            """
            try:
                executor_instance.logger.info(f"Executing welltest_table for wells {wells}")
                result = executor_instance._execute_mcp_tool("welltest_table", json.dumps(dict(wells=wells, test=test)))
                tool_context.actions.skip_summarization = True
                return {"status": "success", "result": result }
            except Exception as e:
                traceback.print_exc()
                return {"status": "error", "message": str(e)}
        tools.append(welltest_table)

        def welltest_chart(tool_context: ToolContext, wells: list[str], test: str) -> dict:
            """ Show welltest chart for input wells
        
            Args:
                wells: input wells
                test: is a string and can be either "production" or "injection" or "all"

            Returns:
                dict: result
            """
            try:
                executor_instance.logger.info(f"Executing welltest_chart for wells {wells}")
                result = executor_instance._execute_mcp_tool("welltest_chart", json.dumps(dict(wells=wells, test=test)))
                tool_context.actions.skip_summarization = True
                return {"status": "success", "result": result }
            except Exception as e:
                traceback.print_exc()
                return {"status": "error", "message": str(e)}
        tools.append(welltest_chart)

        #def water_io_ratio_map(tool_context: ToolContext, wells: list[str]) -> dict:
        #    """Show cum water / cum water injection ratio map
        #    
        #    Args:
        #        wells: input wells
        #
        #    Returns:
        #        dict: result
        #    """
        #    try:
        #        executor_instance.logger.info(f"Executing water_io_ratio_map for wells {wells}")
        #        result = executor_instance._execute_mcp_tool("water_io_ratio_map", json.dumps(dict(wells=wells)))
        #        tool_context.actions.skip_summarization = True
        #        return {"status": "success", "result": result }
        #    except Exception as e:
        #        traceback.print_exc()
        #        return {"status": "error", "message": str(e)}
        #tools.append(water_io_ratio_map)

        def water_analysis_map(tool_context: ToolContext, wells: list[str] = [], year: int = 0) -> dict:
            """Show water analysis map for a year. If a list of wells is specified, show the map of those wells. If the well list is empty, show the map for all existing wells
            
            Args:
                wells: input wells
                year: year

            Returns:
                dict: result
            """
            try:
                executor_instance.logger.info(f"Executing water_analysis_map for wells {wells}")
                result = executor_instance._execute_mcp_tool("water_analysis_map", json.dumps(dict(wells=wells, year=year)))
                tool_context.actions.skip_summarization = True
                return {"status": "success", "result": result }
            except Exception as e:
                traceback.print_exc()
                return {"status": "error", "message": str(e)}
        tools.append(water_analysis_map)

        def production_map(tool_context: ToolContext, wells: list[str] = [], year: int = 0, data: str = 'monthly_rate') -> dict:
            """View production map for input wells in a year. If a list of wells is specified, show the map of those wells. If the well list is empty, show the map for all existing wells
            
            Args:
                wells: input wells
                year: year
                data (str): data to view on map. Supported:
                    - 'cv' for "cumulated volume"
                    - 'monthly_rate' for "monthly rate"

            Returns:
                dict: result
            """
            try:
                executor_instance.logger.info(f"Executing production_map for wells {wells}")
                result = executor_instance._execute_mcp_tool("production_map", json.dumps(dict(wells=wells, year=year, data=data)))
                tool_context.actions.skip_summarization = True
                return {"status": "success", "result": result }
            except Exception as e:
                traceback.print_exc()
                return {"status": "error", "message": str(e)}
        tools.append(production_map)

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
- "bubble map" means "production map"
- User asks "build CRM input using production wells and injection wells" → IMMEDIATELY call buildCRMInput with corresponding production_wells and injection_wells
- User asks "show wells in marker file", then IMMEDIATELY call unique_from_column with column=0 file_path="misc/Marker.xlsx" and sheet=0
- User asks "show wells in production monthly file", then IMMEDIATELY call unique_from_column with column=1 file_path="" and sheet=0
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

- User asks "Plot [params] of wells [wells] by [xparam] from production file" or "View production crossplot of wells [wells]",
    then IMMEDIATELY call production_crossplot with params as list[str] if user provided or else params=["CV.WaterProdCum/1000","CV.WaterInjCum/1000","CV.WaterRate", "CV.WaterInj_Rate","CV.Watercut"]
    and wells as list[str] if user provided or else wells=[]
    and xparam as str if user provided or else xparam="CV.Oilcum/1000"
    **ALWAYS call using these standardized param names from user provided params, DO NOT use monthly params if user not provided explicitly:
    "CV.OilRate",              # Oil rate
    "Monthlyprod.Qoil/1000",   # Monthly oil rate in thousands
    "CV.Oilcum/1000",          # Oilcum in thousands
    "CV.LiqRate",              # Liquid rate (oil + water)
    "CV.WaterRate",            # Water rate, calculated by CV.LiqRate - CV.OilRate
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
- User asks "Generate well_logplot for WELL, then IMMEDIATELY call well_logplot with well=WELL and curves if user provided or curves=[]
- "planset" is equivalent with "logplot"
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

**REMEMBER:
- Always provide ALL required parameters when calling tools!**
- Always place any URL in result into an <iframe>

Available tools: list_files, system_status, health_check, directory_info,
las_parser, las_analysis, formation_evaluation, well_correlation, segy_parser, segy_classify, segy_qc,
quick_segy_summary, dump_content, plot_las, well_logplot, build_logplot, plot_histogram_las, show_sheets, show_columns,
unique_from_column, marker4well, zone4well, production_monthly_data_table, describe_production_data, summarize_marker_data,
train_crm_model, view_wf_experiment, production_by_time, production_crossplot,welltest_chart, welltest_table,
water_analysis_map, production_map, calc_distance_between_wells, train
plt_table, well_checklist_table, well_checklist_curves, create_wells_tvdss, create_pseudo_log, 
view_training_experiment, delete_training_experiment, accept_experiment_las_file, suggest_log_creation, set_context_for_tool_function

Available context_params: modes, marker_file, file_path
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
        tool_response = ""

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
                    if hasattr(event.content.parts[0], 'function_response') and event.content.parts[0].function_response:
                        print(event.content.parts[0].function_response.response)
                        tool_response = event.content.parts[0].function_response.response['result']
                        try:
                            tool_response = json.loads(tool_response)
                            tool_response = tool_response['text']
                        except:
                            pass
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
            return final_response or tool_response # "Analysis completed."

        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Google ADK execution error: {e}")
            return self._minimal_fallback(query)

    def _minimal_fallback(self, query: str) -> str:
        """Minimal fallback when Google ADK fails"""
        #import re

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
                #print(output, params)
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
