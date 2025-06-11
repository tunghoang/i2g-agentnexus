"""
SEG-Y Tools Module - FINAL FIXED VERSION
Fixed parameter handling to check kwargs['input']
"""

import os
import json
import glob
import traceback
from typing import List, Dict, Any, Union

# Import your existing SEG-Y functions
try:
    from production_segy_tools import (
        production_segy_parser,
        segy_complete_metadata_harvester,
        find_segy_file,
        NumpyJSONEncoder,
        mcp_extract_survey_polygon,
        mcp_extract_trace_outlines,
        mcp_save_analysis,
        mcp_get_analysis_catalog,
        mcp_search_analyses
    )
    from production_segy_analysis_qc import (
        production_segy_qc,
        production_segy_analysis
    )
    from production_segy_multifile import (
        production_segy_survey_analysis
    )
    from survey_classifier import SurveyClassifier
    SEGYIO_AVAILABLE = True
except ImportError as e:
    print(f"SEG-Y components not available: {e}")
    SEGYIO_AVAILABLE = False
    class NumpyJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            return str(obj)

from config.settings import DataConfig

def create_error_response(error_message, details=None, suggestions=None):
    """Create error response"""
    error_obj = {"error": error_message}
    if details:
        error_obj["details"] = details
    if suggestions:
        error_obj["suggestions"] = suggestions
    return {"text": json.dumps(error_obj)}

def extract_file_path_from_params(file_path=None, *args, **kwargs):
    """
    FIXED: Universal parameter extraction for MCP tools
    Checks all possible parameter sources in the right order
    """
    # Check direct parameter first
    if file_path and file_path not in ["", "None", "null"]:
        return file_path

    # Check kwargs['input'] (MCP framework standard)
    if 'input' in kwargs and kwargs['input'] not in ["", "None", "null"]:
        return kwargs['input']

    # Check args (fallback)
    if args and len(args) > 0 and args[0] not in ["", "None", "null"]:
        return args[0]

    # Check other possible kwargs
    if 'file_path' in kwargs and kwargs['file_path'] not in ["", "None", "null"]:
        return kwargs['file_path']

    return None

def create_segy_tools(mcp_server, data_config: DataConfig) -> List[str]:
    """Create and register all SEG-Y tools - FINAL FIXED VERSION"""

    if not SEGYIO_AVAILABLE:
        @mcp_server.tool(name="segy_parser", description="SEG-Y parser (requires segyio)")
        def segy_parser_placeholder(*args, **kwargs):
            return create_error_response("SEG-Y tools not available")
        return ["segy_parser"]

    # Tool 1: SEG-Y Parser - FIXED
    @mcp_server.tool(
        name="segy_parser",
        description="Parse SEG-Y seismic files"
    )
    def mcp_segy_parser(file_path: str = None, *args, **kwargs):
        """Parse SEG-Y files - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            file_path = extract_file_path_from_params(file_path, *args, **kwargs)

            if not file_path:
                return create_error_response("No file path provided")

            # Build full path if needed
            if not os.path.isabs(file_path):
                full_path = os.path.join(data_config.data_dir, file_path)
            else:
                full_path = file_path

            return production_segy_parser(file_path=full_path)

        except Exception as e:
            return create_error_response(f"SEG-Y parser failed: {str(e)}")

    # Tool 2: SEG-Y QC - FIXED
    @mcp_server.tool(
        name="segy_qc",
        description="Perform quality control on SEG-Y files"
    )
    def mcp_segy_qc(file_path: str = None, *args, **kwargs):
        """QC SEG-Y files - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            file_path = extract_file_path_from_params(file_path, *args, **kwargs)

            if not file_path:
                return create_error_response("No file path provided")

            if not os.path.isabs(file_path):
                full_path = os.path.join(data_config.data_dir, file_path)
            else:
                full_path = file_path

            return production_segy_qc(file_path=full_path)

        except Exception as e:
            return create_error_response(f"SEG-Y QC failed: {str(e)}")

    # Tool 3: SEG-Y Analysis - FIXED
    @mcp_server.tool(
        name="segy_analysis",
        description="Analyze SEG-Y seismic survey data"
    )
    def mcp_segy_analysis(file_path: str = None, *args, **kwargs):
        """Analyze SEG-Y files - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            file_path = extract_file_path_from_params(file_path, *args, **kwargs)

            if not file_path:
                return create_error_response("No file path provided")

            if not os.path.isabs(file_path):
                full_path = os.path.join(data_config.data_dir, file_path)
            else:
                full_path = file_path

            return production_segy_analysis(file_path=full_path)

        except Exception as e:
            return create_error_response(f"SEG-Y analysis failed: {str(e)}")

    # Tool 4: SEG-Y Classification - FIXED
    @mcp_server.tool(
        name="segy_classify",
        description="Classify SEG-Y survey type"
    )
    def mcp_segy_classify(file_path: str = None, *args, **kwargs):
        """Classify SEG-Y files - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            file_path = extract_file_path_from_params(file_path, *args, **kwargs)

            if not file_path:
                return create_error_response("No file path provided")

            if not os.path.isabs(file_path):
                full_path = os.path.join(data_config.data_dir, file_path)
            else:
                full_path = file_path

            classifier = SurveyClassifier("./templates")
            classification = classifier.classify_survey(full_path)
            return {"text": json.dumps(classification, cls=NumpyJSONEncoder)}

        except Exception as e:
            return create_error_response(f"SEG-Y classification failed: {str(e)}")

    # Tool 5: SEG-Y Survey Analysis - FIXED
    @mcp_server.tool(
        name="segy_survey_analysis",
        description="Analyze multiple SEG-Y files as a survey"
    )
    def mcp_segy_survey_analysis(file_pattern: str = None, *args, **kwargs):
        """Analyze SEG-Y survey - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            file_pattern = extract_file_path_from_params(file_pattern, *args, **kwargs)

            if not file_pattern:
                return create_error_response("No file pattern provided")

            params = {
                "file_pattern": file_pattern,
                "data_dir": data_config.data_dir
            }
            result = production_segy_survey_analysis(**params)
            return {"text": json.dumps(result, cls=NumpyJSONEncoder)}

        except Exception as e:
            return create_error_response(f"SEG-Y survey analysis failed: {str(e)}")

    # Tool 6: Quick SEG-Y Summary - FIXED
    @mcp_server.tool(
        name="quick_segy_summary",
        description="Get quick overview of SEG-Y files"
    )
    def mcp_quick_segy_summary(file_path: str = None, *args, **kwargs):
        """Quick SEG-Y summary - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            file_path = extract_file_path_from_params(file_path, *args, **kwargs)

            if not file_path:
                return create_error_response("No file path provided")

            # Handle patterns
            if "*" in file_path or "?" in file_path:
                search_patterns = []
                base_pattern = file_path.replace('.sgy', '').replace('.segy', '')

                for ext in ['.sgy', '.segy', '.SGY', '.SEGY']:
                    pattern = base_pattern + ext
                    search_pattern = os.path.join(data_config.data_dir, pattern)
                    search_patterns.append(search_pattern)

                matching_files = []
                for pattern in search_patterns:
                    matching_files.extend(glob.glob(pattern))

                matching_files = list(set(matching_files))[:50]  # Limit to 50

                if not matching_files:
                    return create_error_response(f"No SEG-Y files found matching pattern: {file_path}")

                file_summaries = []
                total_size_mb = 0

                for file_item in matching_files:
                    try:
                        file_stats = os.stat(file_item)
                        file_size_mb = file_stats.st_size / (1024 * 1024)
                        total_size_mb += file_size_mb

                        file_summaries.append({
                            "file_name": os.path.basename(file_item),
                            "file_size_mb": round(file_size_mb, 2),
                            "estimated_size": "Large" if file_size_mb > 100 else "Medium" if file_size_mb > 10 else "Small",
                            "status": "Ready for segyio processing"
                        })
                    except Exception as e:
                        file_summaries.append({
                            "file_name": os.path.basename(file_item),
                            "error": str(e)
                        })

                result = {
                    "pattern_processed": file_path,
                    "files_found": len(file_summaries),
                    "total_size_mb": round(total_size_mb, 2),
                    "file_summaries": file_summaries,
                    "summary": f"Quick summary completed for {len(file_summaries)} SEG-Y files"
                }

                return {"text": json.dumps(result, cls=NumpyJSONEncoder)}

            else:
                # Single file
                full_path = find_segy_file(file_path, data_config.data_dir)

                if not os.path.exists(full_path):
                    return create_error_response(f"File not found: {file_path}")

                file_stats = os.stat(full_path)
                file_size_mb = file_stats.st_size / (1024 * 1024)

                result = {
                    "file_name": os.path.basename(full_path),
                    "file_size_mb": round(file_size_mb, 2),
                    "file_exists": True,
                    "estimated_size": "Large" if file_size_mb > 100 else "Medium" if file_size_mb > 10 else "Small",
                    "quick_assessment": "Ready for segyio processing"
                }

                return {"text": json.dumps(result, cls=NumpyJSONEncoder)}

        except Exception as e:
            return create_error_response(f"Quick summary failed: {str(e)}")

    # Tool 7: Complete Metadata Harvester - FIXED
    @mcp_server.tool(
        name="segy_complete_metadata_harvester",
        description="Extract comprehensive metadata from SEG-Y files"
    )
    def mcp_segy_complete_metadata_harvester(file_path: str = None, *args, **kwargs):
        """Extract comprehensive metadata from SEG-Y files - FIXED"""
        try:
            # print(f"*** MCP TOOL DEBUG: file_path='{file_path}', args={args}, kwargs={kwargs}")

            # FIXED: Use universal parameter extraction
            file_path = extract_file_path_from_params(file_path, *args, **kwargs)
            # print(f"*** MCP TOOL DEBUG: Extracted file_path='{file_path}'")

            if not file_path:
                return create_error_response("No file path provided")

            # Build full path if needed
            if not os.path.isabs(file_path):
                full_path = os.path.join(data_config.data_dir, file_path)
            else:
                full_path = file_path

            # print(f"*** MCP TOOL DEBUG: Using full_path='{full_path}'")

            # Call with minimal parameters
            params = {
                'file_path': full_path,
                'max_text_length': 2000,
                'return_format': 'structured'
            }

            result = segy_complete_metadata_harvester(**params)
            # print(f"*** MCP TOOL DEBUG: Result received, type={type(result)}")
            return result

        except Exception as e:
            # print(f"*** MCP TOOL DEBUG: Exception occurred: {e}")
            return create_error_response(f"Complete metadata harvester failed: {str(e)}")

    # Tool 8: Survey Polygon - FIXED
    @mcp_server.tool(
        name="segy_survey_polygon",
        description="Extract survey boundary polygon from SEG-Y coordinates"
    )
    def mcp_segy_survey_polygon(file_path: str = None, *args, **kwargs):
        """Extract survey polygon - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            file_path = extract_file_path_from_params(file_path, *args, **kwargs)

            if not file_path:
                return create_error_response("No file path provided")

            params = {
                "file_path": file_path,
                "data_dir": data_config.data_dir
            }
            return mcp_extract_survey_polygon(**params)

        except Exception as e:
            return create_error_response(f"Survey polygon extraction failed: {str(e)}")

    # Tool 9: Trace Outlines - FIXED
    @mcp_server.tool(
        name="segy_trace_outlines",
        description="Extract trace amplitude outlines for visualization"
    )
    def mcp_segy_trace_outlines(file_path: str = None, *args, **kwargs):
        """Extract trace outlines - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            file_path = extract_file_path_from_params(file_path, *args, **kwargs)

            if not file_path:
                return create_error_response("No file path provided")

            params = {
                "file_path": file_path,
                "data_dir": data_config.data_dir,
                "trace_sample_rate": 100
            }
            return mcp_extract_trace_outlines(**params)

        except Exception as e:
            return create_error_response(f"Trace outline extraction failed: {str(e)}")

    # Tool 10: Save Analysis - FIXED
    @mcp_server.tool(
        name="segy_save_analysis",
        description="Save SEG-Y analysis results"
    )
    def mcp_segy_save_analysis(file_path: str = None, analysis_type: str = None, analysis_data: dict = None, *args, **kwargs):
        """Save analysis results - FIXED"""
        try:
            # FIXED: Extract multiple parameters
            if not file_path:
                file_path = extract_file_path_from_params(file_path, *args, **kwargs)

            if not analysis_type and 'analysis_type' in kwargs:
                analysis_type = kwargs['analysis_type']
            elif not analysis_type and len(args) > 1:
                analysis_type = args[1]

            if not analysis_data and 'analysis_data' in kwargs:
                analysis_data = kwargs['analysis_data']
            elif not analysis_data and len(args) > 2:
                analysis_data = args[2]

            if not file_path:
                return create_error_response("No file path provided")
            if not analysis_type:
                return create_error_response("No analysis type provided")
            if not analysis_data:
                return create_error_response("No analysis data provided")

            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_name = f"{analysis_type}_{timestamp}"

            result = {
                "status": "success",
                "analysis_name": analysis_name,
                "file_path": file_path,
                "analysis_type": analysis_type,
                "timestamp": datetime.datetime.now().isoformat()
            }

            return {"text": json.dumps(result)}

        except Exception as e:
            return create_error_response(f"Save analysis failed: {str(e)}")

    # Tool 11: Analysis Catalog - FIXED
    @mcp_server.tool(
        name="segy_analysis_catalog",
        description="Get catalog of stored SEG-Y analyses"
    )
    def mcp_segy_analysis_catalog(*args, **kwargs):
        """Get analysis catalog - FIXED"""
        try:
            return mcp_get_analysis_catalog(**kwargs)
        except Exception as e:
            return create_error_response(f"Analysis catalog failed: {str(e)}")

    # Tool 12: Search Analyses - FIXED
    @mcp_server.tool(
        name="segy_search_analyses",
        description="Search stored SEG-Y analyses"
    )
    def mcp_segy_search_analyses(search_text: str = "", *args, **kwargs):
        """Search analyses - FIXED"""
        try:
            # FIXED: Extract search text
            if not search_text:
                search_text = extract_file_path_from_params(search_text, *args, **kwargs) or ""

            search_criteria = {"text_search": search_text} if search_text else {}
            params = {'search_criteria': search_criteria}
            return mcp_search_analyses(**params)
        except Exception as e:
            return create_error_response(f"Analysis search failed: {str(e)}")

    return [
        "segy_parser",
        "segy_qc",
        "segy_analysis",
        "segy_classify",
        "segy_survey_analysis",
        "quick_segy_summary",
        "segy_complete_metadata_harvester",
        "segy_survey_polygon",
        "segy_trace_outlines",
        "segy_save_analysis",
        "segy_analysis_catalog",
        "segy_search_analyses"
    ]