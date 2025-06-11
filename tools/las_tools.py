"""
LAS Tools Module - FINAL FIXED VERSION
Fixed parameter handling to check kwargs['input'] - same as SEG-Y tools
"""

import os
import json
import glob
import traceback
from typing import List, Dict, Any

# Import your existing LAS functions
from enhanced_mcp_tools import (
    enhanced_las_parser,
    enhanced_las_analysis,
    enhanced_las_qc,
    enhanced_formation_evaluation,
    enhanced_well_correlation_with_qc,
    find_las_file,
    load_las_file,
    NumpyJSONEncoder
)
from formation_evaluation import estimate_vshale
from config.settings import DataConfig


def create_error_response(error_message: str, details: str = None, suggestions: List[str] = None) -> Dict[str, Any]:
    """Create standardized error response"""
    error_obj = {"error": error_message}
    if details:
        error_obj["details"] = details
    if suggestions:
        error_obj["suggestions"] = suggestions
    return {"text": json.dumps(error_obj, cls=NumpyJSONEncoder)}


def extract_file_path_from_params(file_path=None, *args, **kwargs):
    """
    FIXED: Universal parameter extraction for MCP tools
    Same as SEG-Y tools - checks all possible parameter sources
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


def validate_las_file(file_path: str) -> bool:
    """Validate that file is a LAS file"""
    if not file_path:
        return False
    return file_path.lower().endswith(('.las', '.LAS'))


def find_las_files_by_pattern(pattern: str, data_dir: str) -> List[str]:
    """Find LAS files matching pattern"""
    if not pattern:
        return []

    # Handle different pattern types
    search_patterns = []

    if "*" in pattern or "?" in pattern:
        # Pattern contains wildcards
        base_pattern = pattern.replace('.las', '').replace('.LAS', '')

        for ext in ['.las', '.LAS']:
            full_pattern = base_pattern + ext
            if not os.path.isabs(full_pattern):
                search_pattern = os.path.join(data_dir, full_pattern)
            else:
                search_pattern = full_pattern
            search_patterns.append(search_pattern)
    else:
        # Single file or exact pattern
        if not os.path.isabs(pattern):
            search_pattern = os.path.join(data_dir, pattern)
        else:
            search_pattern = pattern
        search_patterns.append(search_pattern)

    # Collect all matching files
    matching_files = []
    for pattern in search_patterns:
        matching_files.extend(glob.glob(pattern))

    # Remove duplicates and filter LAS files only
    las_files = []
    seen = set()
    for file_path in matching_files:
        if file_path not in seen and validate_las_file(file_path):
            las_files.append(file_path)
            seen.add(file_path)

    return las_files


def create_las_tools(mcp_server, data_config: DataConfig) -> List[str]:
    """
    Create and register all LAS tools - FINAL FIXED VERSION
    """

    # Tool 1: LAS Parser - FIXED
    @mcp_server.tool(
        name="las_parser",
        description="Parse and extract metadata from LAS files including well information, curves, and depth ranges"
    )
    def las_parser(file_path: str = None, *args, **kwargs):
        """Parse LAS file with pattern support - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            file_path = extract_file_path_from_params(file_path, *args, **kwargs)

            if not file_path:
                return create_error_response("No file path provided")

            # Validate LAS file extension
            if not validate_las_file(file_path):
                return create_error_response(
                    f"LAS tool las_parser requires a .las file. Got: {file_path}",
                    suggestions=["Please provide a .las file"]
                )

            # Handle pattern matching
            if "*" in file_path or "?" in file_path:
                matching_files = find_las_files_by_pattern(file_path, data_config.data_dir)

                if not matching_files:
                    return create_error_response(f"No LAS files found matching pattern: {file_path}")

                # Process multiple files
                results = []
                for file in matching_files:
                    try:
                        result = enhanced_las_parser(file)
                        results.append({
                            "file": os.path.basename(file),
                            "result": result
                        })
                    except Exception as e:
                        results.append({
                            "file": os.path.basename(file),
                            "error": str(e)
                        })

                summary = {
                    "pattern_processed": file_path,
                    "files_processed": len(results),
                    "results": results,
                    "summary": f"Parsed {len(results)} LAS files matching '{file_path}'"
                }

                return {"text": json.dumps(summary, cls=NumpyJSONEncoder)}

            else:
                # Single file processing
                if not os.path.isabs(file_path):
                    full_path = os.path.join(data_config.data_dir, file_path)
                else:
                    full_path = file_path

                return enhanced_las_parser(full_path)

        except Exception as e:
            return create_error_response(f"LAS parser failed: {str(e)}")

    # Tool 2: LAS Analysis - FIXED
    @mcp_server.tool(
        name="las_analysis",
        description="Analyze curve data and perform statistical analysis on LAS files"
    )
    def las_analysis(file_path: str = None, curves: str = None, *args, **kwargs):
        """Analyze LAS curves with pattern support - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            file_path = extract_file_path_from_params(file_path, *args, **kwargs)

            # Extract curves parameter
            if not curves:
                if 'curves' in kwargs:
                    curves = kwargs['curves']
                elif len(args) > 1:
                    curves = args[1]

            if not file_path:
                return create_error_response("No file path provided")

            # Validate LAS file extension
            if not validate_las_file(file_path):
                return create_error_response(
                    f"LAS tool las_analysis requires a .las file. Got: {file_path}",
                    suggestions=["Please provide a .las file"]
                )

            # Handle pattern matching
            if "*" in file_path or "?" in file_path:
                matching_files = find_las_files_by_pattern(file_path, data_config.data_dir)

                if not matching_files:
                    return create_error_response(f"No LAS files found matching pattern: {file_path}")

                # Process multiple files
                results = []
                for file in matching_files:
                    try:
                        result = enhanced_las_analysis(file, curves)
                        results.append({
                            "file": os.path.basename(file),
                            "result": result
                        })
                    except Exception as e:
                        results.append({
                            "file": os.path.basename(file),
                            "error": str(e)
                        })

                summary = {
                    "pattern_processed": file_path,
                    "files_processed": len(results),
                    "results": results,
                    "summary": f"Analyzed {len(results)} LAS files matching '{file_path}'"
                }

                return {"text": json.dumps(summary, cls=NumpyJSONEncoder)}

            else:
                # Single file processing
                if not os.path.isabs(file_path):
                    full_path = os.path.join(data_config.data_dir, file_path)
                else:
                    full_path = file_path

                return enhanced_las_analysis(full_path, curves)

        except Exception as e:
            return create_error_response(f"LAS analysis failed: {str(e)}")

    # Tool 3: LAS Quality Control - FIXED
    @mcp_server.tool(
        name="las_qc",
        description="Perform quality control checks on LAS files including data completeness and curve validation"
    )
    def las_qc(file_path: str = None, *args, **kwargs):
        """Quality control for LAS files with pattern support - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            file_path = extract_file_path_from_params(file_path, *args, **kwargs)

            if not file_path:
                return create_error_response("No file path provided")

            # Validate LAS file extension
            if not validate_las_file(file_path):
                return create_error_response(
                    f"LAS tool las_qc requires a .las file. Got: {file_path}",
                    suggestions=["Please provide a .las file"]
                )

            # Handle pattern matching
            if "*" in file_path or "?" in file_path:
                matching_files = find_las_files_by_pattern(file_path, data_config.data_dir)

                if not matching_files:
                    return create_error_response(f"No LAS files found matching pattern: {file_path}")

                # Process multiple files
                results = []
                for file in matching_files:
                    try:
                        result = enhanced_las_qc(file)
                        results.append({
                            "file": os.path.basename(file),
                            "result": result
                        })
                    except Exception as e:
                        results.append({
                            "file": os.path.basename(file),
                            "error": str(e)
                        })

                summary = {
                    "pattern_processed": file_path,
                    "files_processed": len(results),
                    "results": results,
                    "summary": f"QC checked {len(results)} LAS files matching '{file_path}'"
                }

                return {"text": json.dumps(summary, cls=NumpyJSONEncoder)}

            else:
                # Single file processing
                if not os.path.isabs(file_path):
                    full_path = os.path.join(data_config.data_dir, file_path)
                else:
                    full_path = file_path

                return enhanced_las_qc(full_path)

        except Exception as e:
            return create_error_response(f"LAS QC failed: {str(e)}")

    # Tool 4: Formation Evaluation - FIXED
    @mcp_server.tool(
        name="formation_evaluation",
        description="Perform comprehensive petrophysical analysis including porosity, water saturation, shale volume, and pay zones"
    )
    def formation_evaluation(file_path: str = None, *args, **kwargs):
        """Formation evaluation with pattern support - FIXED"""
        try:
            # print(f"*** LAS TOOL DEBUG: file_path='{file_path}', args={args}, kwargs={kwargs}")

            # FIXED: Use universal parameter extraction
            file_path = extract_file_path_from_params(file_path, *args, **kwargs)
            # print(f"*** LAS TOOL DEBUG: Extracted file_path='{file_path}'")

            if not file_path:
                return create_error_response("No file path provided")

            # Validate LAS file extension
            if not validate_las_file(file_path):
                return create_error_response(
                    f"LAS tool formation_evaluation requires a .las file. Got: {file_path}",
                    suggestions=["Please provide a .las file"]
                )

            # Handle pattern matching
            if "*" in file_path or "?" in file_path:
                matching_files = find_las_files_by_pattern(file_path, data_config.data_dir)

                if not matching_files:
                    return create_error_response(f"No LAS files found matching pattern: {file_path}")

                # Process multiple files
                results = []
                for file in matching_files:
                    try:
                        result = enhanced_formation_evaluation(file)
                        results.append({
                            "file": os.path.basename(file),
                            "result": result
                        })
                    except Exception as e:
                        results.append({
                            "file": os.path.basename(file),
                            "error": str(e)
                        })

                summary = {
                    "pattern_processed": file_path,
                    "files_processed": len(results),
                    "results": results,
                    "summary": f"Formation evaluation completed for {len(results)} LAS files matching '{file_path}'"
                }

                return {"text": json.dumps(summary, cls=NumpyJSONEncoder)}

            else:
                # Single file processing
                if not os.path.isabs(file_path):
                    full_path = os.path.join(data_config.data_dir, file_path)
                else:
                    full_path = file_path

                # print(f"*** LAS TOOL DEBUG: Using full_path='{full_path}'")
                result = enhanced_formation_evaluation(full_path)
                # print(f"*** LAS TOOL DEBUG: Result received, type={type(result)}")
                return result

        except Exception as e:
            # print(f"*** LAS TOOL DEBUG: Exception occurred: {e}")
            return create_error_response(f"Formation evaluation failed: {str(e)}")

    # Tool 5: Well Correlation - FIXED
    @mcp_server.tool(
        name="well_correlation",
        description="Correlate formations across multiple wells to identify key formation tops and stratigraphic markers"
    )
    def well_correlation(well_list: str = None, marker_curve: str = "GR", *args, **kwargs):
        """Well correlation with pattern support - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            well_list = extract_file_path_from_params(well_list, *args, **kwargs)

            # Extract marker_curve
            if not marker_curve:
                if 'marker_curve' in kwargs:
                    marker_curve = kwargs['marker_curve']
                elif len(args) > 1:
                    marker_curve = args[1]
                else:
                    marker_curve = "GR"  # Default

            if not well_list:
                return create_error_response("No well list provided")

            # Handle file patterns for correlation
            if isinstance(well_list, str):
                if "*" in well_list or "?" in well_list:
                    # Pattern matching
                    matching_files = find_las_files_by_pattern(well_list, data_config.data_dir)

                    if not matching_files:
                        return create_error_response(f"No LAS files found matching pattern: {well_list}")

                    well_paths = matching_files
                else:
                    # Single file
                    if not validate_las_file(well_list):
                        return create_error_response(
                            f"LAS tool well_correlation requires .las files. Got: {well_list}",
                            suggestions=["Please provide .las files or patterns"]
                        )

                    if not os.path.isabs(well_list):
                        full_path = os.path.join(data_config.data_dir, well_list)
                    else:
                        full_path = well_list

                    if not os.path.exists(full_path):
                        return create_error_response(f"File not found: {well_list}")

                    well_paths = [full_path]

            elif isinstance(well_list, list):
                # List of files - validate and convert to full paths
                well_paths = []
                for file_name in well_list:
                    if not validate_las_file(file_name):
                        continue  # Skip non-LAS files

                    if os.path.isabs(file_name):
                        if os.path.exists(file_name):
                            well_paths.append(file_name)
                    else:
                        full_path = os.path.join(data_config.data_dir, file_name)
                        if os.path.exists(full_path):
                            well_paths.append(full_path)

                if not well_paths:
                    return create_error_response("No valid LAS files found in the provided list")

            else:
                return create_error_response("Invalid well_list format. Expected string or list.")

            if not well_paths:
                return create_error_response("No valid wells found for correlation")

            # Call correlation function
            result = enhanced_well_correlation_with_qc(well_paths, marker_curve)
            return result

        except Exception as e:
            return create_error_response(f"Well correlation failed: {str(e)}")

    # Tool 6: Calculate Shale Volume - FIXED
    @mcp_server.tool(
        name="calculate_shale_volume",
        description="Calculate volume of shale from gamma ray log using the Larionov correction method"
    )
    def calculate_shale_volume(file_path: str = None, curve: str = "GR", *args, **kwargs):
        """Calculate shale volume with pattern support - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            file_path = extract_file_path_from_params(file_path, *args, **kwargs)

            # Extract curve parameter
            if not curve:
                if 'curve' in kwargs:
                    curve = kwargs['curve']
                elif len(args) > 1:
                    curve = args[1]
                else:
                    curve = "GR"  # Default

            if not file_path:
                return create_error_response("No file path provided")

            # Validate LAS file extension
            if not validate_las_file(file_path):
                return create_error_response(
                    f"LAS tool calculate_shale_volume requires a .las file. Got: {file_path}",
                    suggestions=["Please provide a .las file"]
                )

            # Handle pattern matching
            if "*" in file_path or "?" in file_path:
                matching_files = find_las_files_by_pattern(file_path, data_config.data_dir)

                if not matching_files:
                    return create_error_response(f"No LAS files found matching pattern: {file_path}")

                # Process multiple files
                shale_results = []
                total_wells = 0
                avg_shale_overall = 0

                for file in matching_files:
                    try:
                        las, error = load_las_file(file)
                        if error:
                            shale_results.append({
                                "file": os.path.basename(file),
                                "error": f"Error loading LAS file: {error}"
                            })
                            continue

                        if not las.curve_exists(curve):
                            available_curves = las.get_curve_names()
                            shale_results.append({
                                "file": os.path.basename(file),
                                "error": f"Gamma ray curve '{curve}' not found",
                                "available_curves": available_curves
                            })
                            continue

                        # Calculate shale volume
                        import numpy as np
                        gr_data = las.get_curve_data(curve)
                        vshale = estimate_vshale(gr_data)
                        valid_vshale = vshale[~np.isnan(vshale)]

                        if len(valid_vshale) == 0:
                            shale_results.append({
                                "file": os.path.basename(file),
                                "error": "No valid shale volume data calculated"
                            })
                            continue

                        # Calculate statistics
                        avg_vshale_pct = float(np.mean(valid_vshale)) * 100
                        min_vshale_pct = float(np.min(valid_vshale)) * 100
                        max_vshale_pct = float(np.max(valid_vshale)) * 100

                        well_result = {
                            "file": os.path.basename(file),
                            "well_name": las.well_info.get("WELL", "Unknown"),
                            "gamma_ray_curve_used": curve,
                            "average_shale_volume_percent": round(avg_vshale_pct, 2),
                            "min_shale_volume_percent": round(min_vshale_pct, 2),
                            "max_shale_volume_percent": round(max_vshale_pct, 2),
                            "depth_range": [float(las.index[0]), float(las.index[-1])],
                            "data_points_analyzed": len(valid_vshale)
                        }

                        shale_results.append(well_result)
                        total_wells += 1
                        avg_shale_overall += avg_vshale_pct

                    except Exception as e:
                        shale_results.append({
                            "file": os.path.basename(file),
                            "error": str(e)
                        })

                # Create summary
                successful_calculations = [r for r in shale_results if "average_shale_volume_percent" in r]
                field_avg_shale = avg_shale_overall / len(successful_calculations) if successful_calculations else 0

                summary_result = {
                    "pattern_processed": file_path,
                    "files_processed": len(shale_results),
                    "successful_calculations": len(successful_calculations),
                    "method": "Larionov correction for Tertiary rocks",
                    "gamma_ray_curve_used": curve,
                    "field_average_shale_percent": round(field_avg_shale, 2),
                    "individual_results": shale_results,
                    "summary": f"Calculated shale volume for {len(successful_calculations)}/{len(shale_results)} wells, field average: {round(field_avg_shale, 1)}%"
                }

                return {"text": json.dumps(summary_result, cls=NumpyJSONEncoder)}

            else:
                # Single file processing
                if not os.path.isabs(file_path):
                    full_path = os.path.join(data_config.data_dir, file_path)
                else:
                    full_path = file_path

                if not os.path.isfile(full_path):
                    return create_error_response(f"File not found: {file_path}")

                las, error = load_las_file(full_path)
                if error:
                    return create_error_response(f"Error loading LAS file: {error}")

                if not las.curve_exists(curve):
                    return create_error_response(
                        f"Gamma ray curve '{curve}' not found",
                        suggestions=[f"Available curves: {las.get_curve_names()}"]
                    )

                # Calculate shale volume
                import numpy as np
                gr_data = las.get_curve_data(curve)
                vshale = estimate_vshale(gr_data)
                valid_vshale = vshale[~np.isnan(vshale)]

                if len(valid_vshale) == 0:
                    return create_error_response("No valid shale volume data calculated")

                # Calculate statistics
                avg_vshale_pct = float(np.mean(valid_vshale)) * 100
                min_vshale_pct = float(np.min(valid_vshale)) * 100
                max_vshale_pct = float(np.max(valid_vshale)) * 100

                result = {
                    "well_name": las.well_info.get("WELL", "Unknown"),
                    "file_processed": os.path.basename(full_path),
                    "gamma_ray_curve_used": curve,
                    "average_shale_volume_percent": round(avg_vshale_pct, 2),
                    "min_shale_volume_percent": round(min_vshale_pct, 2),
                    "max_shale_volume_percent": round(max_vshale_pct, 2),
                    "method": "Larionov correction for Tertiary rocks",
                    "depth_range": [float(las.index[0]), float(las.index[-1])],
                    "data_points_analyzed": len(valid_vshale),
                    "summary": f"Average shale volume: {round(avg_vshale_pct, 1)}%, range: {round(min_vshale_pct, 1)}%-{round(max_vshale_pct, 1)}%"
                }

                return {"text": json.dumps(result, cls=NumpyJSONEncoder)}

        except Exception as e:
            return create_error_response(f"Shale volume calculation failed: {str(e)}")

    # Return list of registered tool names
    tool_names = [
        "las_parser",
        "las_analysis",
        "las_qc",
        "formation_evaluation",
        "well_correlation",
        "calculate_shale_volume"
    ]

    return tool_names


if __name__ == "__main__":
    # Test LAS tools creation
    from python_a2a.mcp import FastMCP
    from config.settings import DataConfig

    # Create test MCP server
    mcp_server = FastMCP("Test LAS Tools")
    data_config = DataConfig()

    # Create tools
    tools = create_las_tools(mcp_server, data_config)
    print(f"Created {len(tools)} LAS tools: {tools}")