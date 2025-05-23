"""
Enhanced MCP tool implementations for the A2A+MCP+LangChain architecture.
These tools use the robust_las_parser module to handle LAS files reliably.
"""

import os
import json
import traceback
import numpy as np
from typing import Dict, Any, Optional

# Import the robust LAS parser module
from robust_las_parser import (
    load_las_file,
    extract_metadata,
    analyze_curve,
    perform_quality_check
)


# Define custom JSON encoder for NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types"""

    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Helper function to find LAS file in data directory
def find_las_file(file_path: str, data_dir: str = "./data") -> str:
    """
    Find a LAS file in the data directory

    Args:
        file_path: File path or name
        data_dir: Data directory

    Returns:
        str: Full path to the LAS file
    """
    # Check if it's already a full path
    if os.path.isfile(file_path):
        return file_path

    # Check in data directory
    potential_path = os.path.join(data_dir, file_path)
    if os.path.isfile(potential_path):
        return potential_path

    # Try adding .las extension
    if not file_path.lower().endswith('.las'):
        potential_path = os.path.join(data_dir, file_path + '.las')
        if os.path.isfile(potential_path):
            return potential_path

    # Return original path if not found (will be handled by the caller)
    return file_path

def find_las_files_by_pattern(pattern, data_dir="./data"):
    """
    Find all LAS files matching a pattern

    Args:
        pattern: File pattern (can include glob wildcards like * and ?)
        data_dir: Data directory

    Returns:
        List[str]: List of matching file paths
    """
    import glob
    import os

    # If pattern already includes a directory, use it as is
    if os.path.dirname(pattern):
        # Pattern includes a directory path
        file_paths = glob.glob(pattern)
        if not file_paths:
            # Try adding .las extension if no matches found
            file_paths = glob.glob(pattern + ".las")
    else:
        # Pattern is just a filename, prepend data_dir
        file_paths = glob.glob(os.path.join(data_dir, pattern))
        if not file_paths:
            # Try adding .las extension if no matches found
            file_paths = glob.glob(os.path.join(data_dir, pattern + ".las"))

    # Filter to make sure we only include .las files
    las_files = [path for path in file_paths if path.lower().endswith('.las')]

    return las_files


def select_files_from_matches(matches, selection_mode="ask", max_files=5):
    """
    Handle file selection from multiple matches

    Args:
        matches: List of matching file paths
        selection_mode: How to handle multiple matches:
                       "first" - use first file only
                       "all" - use all files (up to max_files)
                       "ask" - return info about matches for user to select
        max_files: Maximum number of files to process when selection_mode="all"

    Returns:
        dict: Selection result with files and status
    """
    if not matches:
        return {
            "status": "error",
            "message": "No matching files found",
            "files": []
        }

    if len(matches) == 1:
        return {
            "status": "selected",
            "message": "Single file matched",
            "files": matches
        }

    # Multiple matches - handle according to selection_mode
    if selection_mode == "first":
        return {
            "status": "selected",
            "message": f"Selected first of {len(matches)} matching files",
            "files": [matches[0]],
            "all_matches": matches
        }

    elif selection_mode == "all":
        # Limit to max_files
        selected = matches[:max_files]
        return {
            "status": "selected",
            "message": f"Selected {len(selected)} of {len(matches)} matching files (limited to {max_files})",
            "files": selected,
            "all_matches": matches
        }

    else:  # "ask" or any other value
        # Return info for user to choose
        return {
            "status": "choose",
            "message": f"Multiple files ({len(matches)}) matched the pattern. Please select specific file(s):",
            "files": [],  # No files selected yet
            "all_matches": matches
        }


def format_file_selection_response(selection_result):
    """Format file selection result as JSON response"""
    if selection_result["status"] == "error":
        return {"text": json.dumps({
            "error": selection_result["message"]
        })}

    elif selection_result["status"] == "choose":
        # Prepare a list of files with info
        file_options = []
        for i, file_path in enumerate(selection_result["all_matches"]):
            file_name = os.path.basename(file_path)
            file_options.append({
                "index": i + 1,
                "file_name": file_name,
                "file_path": file_path
            })

        return {"text": json.dumps({
            "status": "multiple_matches",
            "message": selection_result["message"],
            "file_count": len(selection_result["all_matches"]),
            "files": file_options,
            "usage": "To select specific files, use: {'file_paths': ['path1', 'path2']} or {'file_indices': [1, 3, 5]}"
        })}

    # Status is "selected" - files were selected
    return None  # Return None to continue processing

# Enhanced LAS Parser Tool
def enhanced_las_parser(file_path=None, data_dir="./data", **kwargs):
    """
    Enhanced LAS parser tool using robust parsing with improved file selection

    Args:
        file_path: Path to the LAS file (can include glob patterns)
        data_dir: Data directory
        **kwargs: Additional keyword arguments including:
            - selection_mode: How to handle multiple matches ("first", "all", or "ask")
            - max_files: Maximum number of files to process when selection_mode="all"
            - file_paths: List of specific file paths to process
            - file_indices: List of indices to select from matched files
            - list_only: If True, just list matching files without parsing them

    Returns:
        dict: Tool response with JSON text
    """
    try:
        # Extract parameters
        selection_mode = kwargs.pop('selection_mode', "first")  # Default to first file
        max_files = int(kwargs.pop('max_files', 5))
        file_paths = kwargs.pop('file_paths', None)  # Specific paths to use
        file_indices = kwargs.pop('file_indices', None)  # Indices to select from matches
        list_only = kwargs.pop('list_only', False)

        # Handle keyword arguments
        if 'input' in kwargs and kwargs['input'] is not None:
            try:
                # Check if input is JSON
                input_data = json.loads(kwargs['input'])
                if isinstance(input_data, dict):
                    if 'file_path' in input_data:
                        file_path = input_data['file_path']
                    if 'selection_mode' in input_data:
                        selection_mode = input_data['selection_mode']
                    if 'max_files' in input_data:
                        max_files = int(input_data['max_files'])
                    if 'file_paths' in input_data:
                        file_paths = input_data['file_paths']
                    if 'file_indices' in input_data:
                        file_indices = input_data['file_indices']
                    if 'list_only' in input_data:  # Check for list_only parameter
                        list_only = input_data['list_only']
            except json.JSONDecodeError:
                # Not JSON, treat as file path
                file_path = kwargs['input']

        # Make sure we have a path (unless specific file_paths were provided)
        if file_path is None and file_paths is None:
            return {"text": json.dumps({"error": "No file path provided"})}

        # If list_only is True, just find and return matching files
        if list_only:
            matching_files = []

            # If we have specific file_paths, use those directly
            if file_paths:
                for path in file_paths:
                    full_path = find_las_file(path, data_dir)
                    if os.path.isfile(full_path):
                        matching_files.append(full_path)

            # If we have file_path with a pattern, find matching files
            elif file_path and ('*' in file_path or '?' in file_path):
                matching_files = find_las_files_by_pattern(file_path, data_dir)

            # Regular single file path
            else:
                file_path = find_las_file(file_path, data_dir)
                if os.path.isfile(file_path):
                    matching_files.append(file_path)

            # Return the list of matching files
            return {"text": json.dumps({
                "pattern": file_path if file_path else "Multiple patterns",
                "matching_files": [os.path.basename(f) for f in matching_files],
                "full_paths": matching_files,
                "count": len(matching_files)
            }, cls=NumpyJSONEncoder)}

        # Find files to process
        selected_files = []

        # If we have specific file_paths, use those directly
        if file_paths:
            for path in file_paths:
                full_path = find_las_file(path, data_dir)
                if os.path.isfile(full_path):
                    selected_files.append(full_path)

            if not selected_files:
                return {"text": json.dumps({"error": "None of the specified file paths were found"})}

        # If we have file_path with a pattern, find matching files
        elif file_path and ('*' in file_path or '?' in file_path):
            # Find all matching files
            matching_files = find_las_files_by_pattern(file_path, data_dir)

            # If we have file_indices, select those specific files
            if file_indices:
                try:
                    # Convert 1-based indices to 0-based
                    indices = [int(idx) - 1 for idx in file_indices]
                    selected_files = [matching_files[i] for i in indices if 0 <= i < len(matching_files)]

                    if not selected_files:
                        return {"text": json.dumps({"error": "None of the specified file indices were valid"})}
                except (ValueError, TypeError):
                    return {"text": json.dumps({"error": "Invalid file indices provided"})}
            else:
                # Use the selection helper function
                selection_result = select_files_from_matches(matching_files, selection_mode, max_files)

                # If we need to ask the user to choose (and haven't been provided with choices)
                if selection_result["status"] == "choose":
                    return format_file_selection_response(selection_result)

                selected_files = selection_result["files"]

        # Regular single file path
        else:
            file_path = find_las_file(file_path, data_dir)
            if not os.path.isfile(file_path):
                return {"text": json.dumps({"error": f"File '{file_path}' not found"})}
            selected_files = [file_path]

        # Now we have our final list of files to process
        if len(selected_files) == 1:
            # Single file - process normally
            las, error = load_las_file(selected_files[0])

            if error:
                return {"text": json.dumps({"error": error})}

            # Extract metadata
            metadata = extract_metadata(las)

            # Add file info
            metadata["file_processed"] = os.path.basename(selected_files[0])

            # Return the metadata as JSON
            return {"text": json.dumps(metadata, cls=NumpyJSONEncoder)}

        else:
            # Multiple files - create a summary of all metadata
            multi_file_results = {
                "file_count": len(selected_files),
                "files_processed": [os.path.basename(f) for f in selected_files],
                "metadata_summary": []
            }

            # Process each file
            for file_path in selected_files:
                try:
                    las, error = load_las_file(file_path)

                    if error:
                        multi_file_results["metadata_summary"].append({
                            "file": os.path.basename(file_path),
                            "error": error
                        })
                        continue

                    # Extract metadata (simplified version)
                    metadata = extract_metadata(las)

                    # Add key metadata for summary
                    file_result = {
                        "file": os.path.basename(file_path),
                        "well_name": metadata.get("well_name", "Unknown"),
                        "depth_range": metadata.get("depth_info", {}).get("start", 0),
                        "depth_end": metadata.get("depth_info", {}).get("end", 0),
                        "curve_count": len(metadata.get("curves", [])),
                        "curves": [curve["mnemonic"] for curve in metadata.get("curves", [])]
                    }

                    multi_file_results["metadata_summary"].append(file_result)

                except Exception as e:
                    multi_file_results["metadata_summary"].append({
                        "file": os.path.basename(file_path),
                        "error": str(e)
                    })

            # Return the multi-file results
            return {"text": json.dumps(multi_file_results, cls=NumpyJSONEncoder)}

    except Exception as e:
        error_details = traceback.format_exc()
        return {"text": json.dumps({
            "error": f"Error parsing LAS file: {str(e)}",
            "details": error_details
        }, cls=NumpyJSONEncoder)}


# Enhanced LAS Analysis Tool
def enhanced_las_analysis(file_path=None, curves=None, data_dir="./data", **kwargs):
    """
    Enhanced LAS analysis tool using robust parsing with improved file selection

    Args:
        file_path: Path to the LAS file (can include glob patterns)
        curves: Curve name(s) to analyze
        data_dir: Data directory
        **kwargs: Additional keyword arguments including:
            - selection_mode: How to handle multiple matches ("first", "all", or "ask")
            - max_files: Maximum number of files to process when selection_mode="all"
            - file_paths: List of specific file paths to process
            - file_indices: List of indices to select from matched files

    Returns:
        dict: Tool response with JSON text
    """
    try:
        # Extract parameters
        selection_mode = kwargs.pop('selection_mode', "first")  # Default to first file
        max_files = int(kwargs.pop('max_files', 5))
        file_paths = kwargs.pop('file_paths', None)  # Specific paths to use
        file_indices = kwargs.pop('file_indices', None)  # Indices to select from matches

        # Handle keyword arguments
        if 'input' in kwargs:
            # Check if input is a JSON string
            try:
                input_data = json.loads(kwargs['input'])
                if isinstance(input_data, dict):
                    if 'file_path' in input_data:
                        file_path = input_data['file_path']
                    if 'curves' in input_data:
                        curves = input_data['curves']
                    if 'selection_mode' in input_data:
                        selection_mode = input_data['selection_mode']
                    if 'max_files' in input_data:
                        max_files = int(input_data['max_files'])
                    if 'file_paths' in input_data:
                        file_paths = input_data['file_paths']
                    if 'file_indices' in input_data:
                        file_indices = input_data['file_indices']
            except json.JSONDecodeError:
                # Not JSON, treat as file path
                file_path = kwargs['input']

        # Make sure we have a path (unless specific file_paths were provided)
        if file_path is None and file_paths is None:
            return {"text": json.dumps({"error": "No file path provided"})}

        # Parse curves parameter
        if isinstance(curves, str):
            curves = [c.strip() for c in curves.split(",")]
        elif curves is None:
            curves = []

        # Find files to process
        selected_files = []

        # If we have specific file_paths, use those directly
        if file_paths:
            for path in file_paths:
                full_path = find_las_file(path, data_dir)
                if os.path.isfile(full_path):
                    selected_files.append(full_path)

            if not selected_files:
                return {"text": json.dumps({"error": "None of the specified file paths were found"})}

        # If we have file_path with a pattern, find matching files
        elif file_path and ('*' in file_path or '?' in file_path):
            # Find all matching files
            matching_files = find_las_files_by_pattern(file_path, data_dir)

            # If we have file_indices, select those specific files
            if file_indices:
                try:
                    # Convert 1-based indices to 0-based
                    indices = [int(idx) - 1 for idx in file_indices]
                    selected_files = [matching_files[i] for i in indices if 0 <= i < len(matching_files)]

                    if not selected_files:
                        return {"text": json.dumps({"error": "None of the specified file indices were valid"})}
                except (ValueError, TypeError):
                    return {"text": json.dumps({"error": "Invalid file indices provided"})}
            else:
                # Use the selection helper function
                selection_result = select_files_from_matches(matching_files, selection_mode, max_files)

                # If we need to ask the user to choose (and haven't been provided with choices)
                if selection_result["status"] == "choose":
                    return format_file_selection_response(selection_result)

                selected_files = selection_result["files"]

        # Regular single file path
        else:
            file_path = find_las_file(file_path, data_dir)
            if not os.path.isfile(file_path):
                return {"text": json.dumps({"error": f"File '{file_path}' not found"})}
            selected_files = [file_path]

        # Now we have our final list of files to process
        if len(selected_files) == 1:
            # Single file - process normally
            las, error = load_las_file(selected_files[0])

            if error:
                return {"text": json.dumps({"error": error})}

            # If no curves specified, use all available curves
            if not curves:
                curves = las.get_curve_names()
                if curves and len(curves) > 0:
                    # Skip the first curve (usually depth)
                    curves = curves[1:]

            # Analyze each curve
            results = {
                "well": las.well_info.get("WELL", "Unknown"),
                "file_processed": os.path.basename(selected_files[0]),
                "curves_analyzed": len(curves),
                "depth_range": list(las.get_depth_range()),
                "curve_data": {}
            }

            for curve in curves:
                analysis = analyze_curve(las, curve)
                if "error" in analysis:
                    results["curve_data"][curve] = {"status": "error", "message": analysis["error"]}
                else:
                    results["curve_data"][curve] = analysis

            # Return the analysis as JSON
            return {"text": json.dumps(results, cls=NumpyJSONEncoder)}

        else:
            # Multiple files - create a summary of all analyses
            multi_file_results = {
                "file_count": len(selected_files),
                "files_processed": [os.path.basename(f) for f in selected_files],
                "analyses": []
            }

            # Process each file
            for file_path in selected_files:
                try:
                    las, error = load_las_file(file_path)

                    if error:
                        multi_file_results["analyses"].append({
                            "file": os.path.basename(file_path),
                            "error": error
                        })
                        continue

                    # Determine curves to analyze for this file
                    file_curves = curves.copy() if curves else []
                    if not file_curves:
                        available_curves = las.get_curve_names()
                        if available_curves and len(available_curves) > 0:
                            # Skip the first curve (usually depth)
                            file_curves = available_curves[1:]

                    # Analyze each curve
                    file_result = {
                        "file": os.path.basename(file_path),
                        "well": las.well_info.get("WELL", "Unknown"),
                        "depth_range": list(las.get_depth_range()),
                        "curves_analyzed": len(file_curves),
                        "curve_data": {}
                    }

                    for curve in file_curves:
                        analysis = analyze_curve(las, curve)
                        if "error" in analysis:
                            file_result["curve_data"][curve] = {"status": "error", "message": analysis["error"]}
                        else:
                            # Include only key statistics for multi-file summary
                            file_result["curve_data"][curve] = {
                                "min": analysis.get("min"),
                                "max": analysis.get("max"),
                                "mean": analysis.get("mean"),
                                "std_dev": analysis.get("std_dev"),
                                "null_percentage": analysis.get("null_percentage")
                            }

                    multi_file_results["analyses"].append(file_result)

                except Exception as e:
                    multi_file_results["analyses"].append({
                        "file": os.path.basename(file_path),
                        "error": str(e)
                    })

            # Return the multi-file results
            return {"text": json.dumps(multi_file_results, cls=NumpyJSONEncoder)}

    except Exception as e:
        error_details = traceback.format_exc()
        return {"text": json.dumps({
            "error": f"Error analyzing LAS file: {str(e)}",
            "details": error_details
        }, cls=NumpyJSONEncoder)}


# Enhanced LAS QC Tool
def enhanced_las_qc(file_path=None, data_dir="./data", **kwargs):
    """
    Enhanced LAS quality control tool using robust parsing with improved file selection

    Args:
        file_path: Path to the LAS file (can include glob patterns)
        data_dir: Data directory
        **kwargs: Additional keyword arguments including:
            - selection_mode: How to handle multiple matches ("first", "all", or "ask")
            - max_files: Maximum number of files to process when selection_mode="all"
            - file_paths: List of specific file paths to process
            - file_indices: List of indices to select from matched files

    Returns:
        dict: Tool response with JSON text
    """
    try:
        # Extract parameters
        selection_mode = kwargs.pop('selection_mode', "first")  # Default to first file
        max_files = int(kwargs.pop('max_files', 5))
        file_paths = kwargs.pop('file_paths', None)  # Specific paths to use
        file_indices = kwargs.pop('file_indices', None)  # Indices to select from matches

        # Handle keyword arguments
        if 'input' in kwargs and kwargs['input'] is not None:
            try:
                # Check if input is JSON
                input_data = json.loads(kwargs['input'])
                if isinstance(input_data, dict):
                    if 'file_path' in input_data:
                        file_path = input_data['file_path']
                    if 'selection_mode' in input_data:
                        selection_mode = input_data['selection_mode']
                    if 'max_files' in input_data:
                        max_files = int(input_data['max_files'])
                    if 'file_paths' in input_data:
                        file_paths = input_data['file_paths']
                    if 'file_indices' in input_data:
                        file_indices = input_data['file_indices']
            except json.JSONDecodeError:
                # Not JSON, treat as file path
                file_path = kwargs['input']

        # Make sure we have a path (unless specific file_paths were provided)
        if file_path is None and file_paths is None:
            return {"text": json.dumps({"error": "No file path provided"})}

        # Find files to process
        selected_files = []

        # If we have specific file_paths, use those directly
        if file_paths:
            for path in file_paths:
                full_path = find_las_file(path, data_dir)
                if os.path.isfile(full_path):
                    selected_files.append(full_path)

            if not selected_files:
                return {"text": json.dumps({"error": "None of the specified file paths were found"})}

        # If we have file_path with a pattern, find matching files
        elif file_path and ('*' in file_path or '?' in file_path):
            # Find all matching files
            matching_files = find_las_files_by_pattern(file_path, data_dir)

            # If we have file_indices, select those specific files
            if file_indices:
                try:
                    # Convert 1-based indices to 0-based
                    indices = [int(idx) - 1 for idx in file_indices]
                    selected_files = [matching_files[i] for i in indices if 0 <= i < len(matching_files)]

                    if not selected_files:
                        return {"text": json.dumps({"error": "None of the specified file indices were valid"})}
                except (ValueError, TypeError):
                    return {"text": json.dumps({"error": "Invalid file indices provided"})}
            else:
                # Use the selection helper function
                selection_result = select_files_from_matches(matching_files, selection_mode, max_files)

                # If we need to ask the user to choose (and haven't been provided with choices)
                if selection_result["status"] == "choose":
                    return format_file_selection_response(selection_result)

                selected_files = selection_result["files"]

        # Regular single file path
        else:
            file_path = find_las_file(file_path, data_dir)
            if not os.path.isfile(file_path):
                return {"text": json.dumps({"error": f"File '{file_path}' not found"})}
            selected_files = [file_path]

        # Now we have our final list of files to process
        if len(selected_files) == 1:
            # Single file - process normally
            las, error = load_las_file(selected_files[0])

            if error:
                return {"text": json.dumps({"error": error})}

            # Perform quality check
            qc_results = perform_quality_check(las)

            # Add file info
            qc_results["file_processed"] = os.path.basename(selected_files[0])

            # Return the QC results as JSON
            return {"text": json.dumps(qc_results, cls=NumpyJSONEncoder)}

        else:
            # Multiple files - create a summary of all QC results
            multi_file_results = {
                "file_count": len(selected_files),
                "files_processed": [os.path.basename(f) for f in selected_files],
                "qc_results": []
            }

            # Process each file
            quality_ratings = {
                "Excellent": 0,
                "Good": 0,
                "Fair": 0,
                "Poor": 0
            }

            for file_path in selected_files:
                try:
                    las, error = load_las_file(file_path)

                    if error:
                        multi_file_results["qc_results"].append({
                            "file": os.path.basename(file_path),
                            "error": error
                        })
                        continue

                    # Perform quality check
                    qc_result = perform_quality_check(las)

                    # Count the quality rating
                    rating = qc_result.get("quality_rating", "Unknown")
                    if rating in quality_ratings:
                        quality_ratings[rating] += 1

                    # Add summary of issues
                    file_result = {
                        "file": os.path.basename(file_path),
                        "well": las.well_info.get("WELL", "Unknown"),
                        "quality_rating": rating,
                        "issue_count": len(qc_result.get("issues", [])),
                        "curve_issue_count": sum(len(issues) for issues in qc_result.get("curve_issues", {}).values()),
                        "curves": list(qc_result.get("curve_issues", {}).keys())
                    }

                    multi_file_results["qc_results"].append(file_result)

                except Exception as e:
                    multi_file_results["qc_results"].append({
                        "file": os.path.basename(file_path),
                        "error": str(e)
                    })

            # Add quality summary
            multi_file_results["quality_summary"] = quality_ratings

            # Identify wells with the most issues for prioritization
            if multi_file_results["qc_results"]:
                multi_file_results["qc_results"].sort(
                    key=lambda x: x.get("issue_count", 0) + x.get("curve_issue_count", 0), reverse=True)
                multi_file_results["priority_wells"] = [
                    {
                        "file": result["file"],
                        "well": result.get("well", "Unknown"),
                        "quality_rating": result.get("quality_rating", "Unknown"),
                        "total_issues": result.get("issue_count", 0) + result.get("curve_issue_count", 0)
                    }
                    for result in multi_file_results["qc_results"][:3] if "error" not in result
                ]

            # Return the multi-file results
            return {"text": json.dumps(multi_file_results, cls=NumpyJSONEncoder)}

    except Exception as e:
        error_details = traceback.format_exc()
        return {"text": json.dumps({
            "error": f"Error performing QC on LAS file: {str(e)}",
            "details": error_details
        }, cls=NumpyJSONEncoder)}


def enhanced_formation_evaluation(file_path=None, data_dir="./data", llm_agent=None, **kwargs):
    """
    Enhanced formation evaluation tool using robust parsing, with optional LLM interpretation

    Args:
        file_path: Path to the LAS file (can include glob patterns)
        data_dir: Data directory
        llm_agent: Optional LangChain agent for LLM enhancement
        **kwargs: Additional keyword arguments including:
            - selection_mode: How to handle multiple matches ("first", "all", or "ask")
            - max_files: Maximum number of files to process when selection_mode="all"
            - file_paths: List of specific file paths to process
            - file_indices: List of indices to select from matched files

    Returns:
        dict: Tool response with JSON text
    """
    from formation_evaluation import evaluate_formation, create_formation_evaluation_summary

    try:
        # Extract parameters
        llm_enhance = kwargs.pop('llm_enhance', False)
        selection_mode = kwargs.pop('selection_mode', "ask")  # Default to asking user
        max_files = int(kwargs.pop('max_files', 5))
        file_paths = kwargs.pop('file_paths', None)  # Specific paths to use
        file_indices = kwargs.pop('file_indices', None)  # Indices to select from matches

        # Handle keyword arguments
        if 'input' in kwargs:
            # Check if input is a JSON string
            try:
                input_data = json.loads(kwargs['input'])
                if isinstance(input_data, dict):
                    # Extract file selection parameters
                    if 'file_path' in input_data:
                        file_path = input_data['file_path']
                    if 'selection_mode' in input_data:
                        selection_mode = input_data['selection_mode']
                    if 'max_files' in input_data:
                        max_files = int(input_data['max_files'])
                    if 'file_paths' in input_data:
                        file_paths = input_data['file_paths']
                    if 'file_indices' in input_data:
                        file_indices = input_data['file_indices']

                    # Extract other parameters
                    for param in ['gr_curve', 'density_curve', 'resistivity_curve',
                                  'neutron_curve', 'rw', 'vsh_cutoff',
                                  'porosity_cutoff', 'sw_cutoff', 'summary_only', 'llm_enhance']:
                        if param in input_data:
                            if param == 'llm_enhance':
                                llm_enhance = input_data[param]
                            else:
                                kwargs[param] = input_data[param]
            except json.JSONDecodeError:
                # Not JSON, treat as file path
                file_path = kwargs['input']

        # Make sure we have a path (unless specific file_paths were provided)
        if file_path is None and file_paths is None:
            return {"text": json.dumps({"error": "No file path provided"})}

        # If we have specific file_paths, use those directly
        if file_paths:
            selected_files = []
            for path in file_paths:
                full_path = find_las_file(path, data_dir)
                if os.path.isfile(full_path):
                    selected_files.append(full_path)

            if not selected_files:
                return {"text": json.dumps({"error": "None of the specified file paths were found"})}

        # If we have file_path with a pattern, find matching files
        elif '*' in file_path or '?' in file_path:
            # Find all matching files
            matching_files = find_las_files_by_pattern(file_path, data_dir)

            # If we have file_indices, select those specific files
            if file_indices:
                try:
                    # Convert 1-based indices to 0-based
                    indices = [int(idx) - 1 for idx in file_indices]
                    selected_files = [matching_files[i] for i in indices if 0 <= i < len(matching_files)]

                    if not selected_files:
                        return {"text": json.dumps({"error": "None of the specified file indices were valid"})}
                except (ValueError, TypeError):
                    return {"text": json.dumps({"error": "Invalid file indices provided"})}
            else:
                # Use the selection helper function
                selection_result = select_files_from_matches(matching_files, selection_mode, max_files)

                # If we need to ask the user to choose (and haven't been provided with choices)
                if selection_result["status"] == "choose":
                    return format_file_selection_response(selection_result)

                selected_files = selection_result["files"]

        # Regular single file path
        else:
            file_path = find_las_file(file_path, data_dir)
            if not os.path.isfile(file_path):
                return {"text": json.dumps({"error": f"File '{file_path}' not found"})}
            selected_files = [file_path]

        # Now we have our final list of files to process
        if len(selected_files) == 1:
            # Single file - process normally
            las, error = load_las_file(selected_files[0])

            if error:
                return {"text": json.dumps({"error": error})}

            # Extract parameters from kwargs
            gr_curve = kwargs.get('gr_curve', "GR")
            density_curve = kwargs.get('density_curve', "RHOB")
            resistivity_curve = kwargs.get('resistivity_curve', "RT")
            neutron_curve = kwargs.get('neutron_curve', "NPHI")
            rw = float(kwargs.get('rw', 0.1))
            vsh_cutoff = float(kwargs.get('vsh_cutoff', 0.5))
            porosity_cutoff = float(kwargs.get('porosity_cutoff', 0.1))
            sw_cutoff = float(kwargs.get('sw_cutoff', 0.7))
            summary_only = kwargs.get('summary_only', False)

            # Perform formation evaluation
            result = evaluate_formation(
                las,
                gr_curve=gr_curve,
                density_curve=density_curve,
                resistivity_curve=resistivity_curve,
                neutron_curve=neutron_curve,
                rw=rw,
                vsh_cutoff=vsh_cutoff,
                porosity_cutoff=porosity_cutoff,
                sw_cutoff=sw_cutoff
            )

            # Create human-readable summary
            summary = create_formation_evaluation_summary(result)

            # Add file info to result
            result["file_processed"] = os.path.basename(selected_files[0])

            # If LLM enhancement is requested and an LLM agent is available
            if llm_enhance and llm_agent:
                try:
                    llm_query = f"""As a petrophysical expert, analyze these formation evaluation results and provide geological insights:

                    WELL: {result.get('well_name', 'Unknown')}
                    DEPTH RANGE: {result.get('depth_range', [0, 0])}

                    SUMMARY:
                    {summary}

                    Please provide:
                    1. Depositional environment interpretation based on these results
                    2. Assessment of reservoir quality and heterogeneity
                    3. Recommendations for completion strategy
                    4. Any other geological insights from the petrophysical data

                    Format your response as JSON with these keys: "depositional_environment", "reservoir_quality", "completion_recommendations", "additional_insights"
                    """

                    # Get LLM insights
                    llm_response = llm_agent.invoke(llm_query)
                    llm_output = llm_response.get('output', '')

                    # Try to parse JSON from the response
                    try:
                        # Look for JSON in the response (might be surrounded by text)
                        import re
                        json_match = re.search(r'({[\s\S]*})', llm_output)
                        if json_match:
                            llm_json_str = json_match.group(1)
                            llm_insights = json.loads(llm_json_str)
                        else:
                            # Create structured insights manually
                            llm_insights = {
                                "llm_analysis": llm_output
                            }
                    except:
                        llm_insights = {
                            "llm_analysis": llm_output
                        }

                    # Add LLM insights to result
                    result["llm_insights"] = llm_insights
                except Exception as e:
                    # If LLM enhancement fails, add error message but continue
                    result["llm_insights"] = {
                        "error": f"LLM enhancement failed: {str(e)}",
                        "note": "Computational results are still valid."
                    }

            # If summary only, return just the text summary and optional LLM insights
            if summary_only:
                response = {"summary": summary}
                if llm_enhance and llm_agent and "llm_insights" in result:
                    response["llm_insights"] = result["llm_insights"]
                return {"text": json.dumps(response)}

            # Add summary to result
            result["text_summary"] = summary

            # Return the full evaluation results as JSON
            return {"text": json.dumps(result, cls=NumpyJSONEncoder)}

        else:
            # Multiple files - create a summary of all evaluations
            multi_file_results = {
                "file_count": len(selected_files),
                "files_processed": [os.path.basename(f) for f in selected_files],
                "evaluations": []
            }

            # Process each file
            for file_path in selected_files:
                try:
                    las, error = load_las_file(file_path)

                    if error:
                        multi_file_results["evaluations"].append({
                            "file": os.path.basename(file_path),
                            "error": error
                        })
                        continue

                    # Extract parameters from kwargs
                    gr_curve = kwargs.get('gr_curve', "GR")
                    density_curve = kwargs.get('density_curve', "RHOB")
                    resistivity_curve = kwargs.get('resistivity_curve', "RT")
                    neutron_curve = kwargs.get('neutron_curve', "NPHI")
                    rw = float(kwargs.get('rw', 0.1))
                    vsh_cutoff = float(kwargs.get('vsh_cutoff', 0.5))
                    porosity_cutoff = float(kwargs.get('porosity_cutoff', 0.1))
                    sw_cutoff = float(kwargs.get('sw_cutoff', 0.7))

                    # Perform formation evaluation
                    result = evaluate_formation(
                        las,
                        gr_curve=gr_curve,
                        density_curve=density_curve,
                        resistivity_curve=resistivity_curve,
                        neutron_curve=neutron_curve,
                        rw=rw,
                        vsh_cutoff=vsh_cutoff,
                        porosity_cutoff=porosity_cutoff,
                        sw_cutoff=sw_cutoff
                    )

                    # Create summary
                    summary = create_formation_evaluation_summary(result)

                    # Add essential info to results
                    file_result = {
                        "file": os.path.basename(file_path),
                        "well_name": result.get("well_name", "Unknown"),
                        "depth_range": result.get("depth_range", [0, 0]),
                        "formation_properties": result.get("formation_properties", {}),
                        "pay_summary": {
                            "net_pay": result.get("pay_summary", {}).get("net_pay", 0),
                            "num_zones": result.get("pay_summary", {}).get("num_zones", 0)
                        },
                        "text_summary": summary
                    }

                    multi_file_results["evaluations"].append(file_result)

                except Exception as e:
                    multi_file_results["evaluations"].append({
                        "file": os.path.basename(file_path),
                        "error": str(e)
                    })

            # Add aggregated statistics
            valid_evaluations = [e for e in multi_file_results["evaluations"] if "error" not in e]

            if valid_evaluations:
                multi_file_results["aggregated_stats"] = {
                    "avg_net_pay": sum(e["pay_summary"]["net_pay"] for e in valid_evaluations) / len(valid_evaluations),
                    "total_net_pay": sum(e["pay_summary"]["net_pay"] for e in valid_evaluations),
                    "total_pay_zones": sum(e["pay_summary"]["num_zones"] for e in valid_evaluations),
                    "successful_evaluations": len(valid_evaluations),
                    "failed_evaluations": len(multi_file_results["evaluations"]) - len(valid_evaluations)
                }

                # Add overall summary
                multi_file_results["overall_summary"] = f"Evaluated {len(valid_evaluations)} wells successfully. " \
                                                        f"Total net pay: {multi_file_results['aggregated_stats']['total_net_pay']:.2f} units. " \
                                                        f"Average net pay per well: {multi_file_results['aggregated_stats']['avg_net_pay']:.2f} units. " \
                                                        f"Total pay zones identified: {multi_file_results['aggregated_stats']['total_pay_zones']}."

            # If LLM enhancement is requested and an LLM agent is available
            if llm_enhance and llm_agent and valid_evaluations:
                try:
                    llm_query = f"""As a petrophysical expert, analyze these formation evaluation results from multiple wells and provide geological insights:

                    WELLS EVALUATED: {len(valid_evaluations)}
                    {multi_file_results.get('overall_summary', '')}

                    INDIVIDUAL WELL SUMMARIES:
                    {chr(10).join([f"WELL: {e.get('well_name', 'Unknown')} - Net Pay: {e['pay_summary']['net_pay']}" for e in valid_evaluations])}

                    Please provide:
                    1. Field-wide depositional environment interpretation
                    2. Assessment of reservoir quality and heterogeneity across the field
                    3. Recommendations for field development strategy
                    4. Insights on variations between different wells

                    Format your response as JSON with these keys: "field_interpretation", "reservoir_heterogeneity", "development_strategy", "well_variations"
                    """

                    # Get LLM insights
                    llm_response = llm_agent.invoke(llm_query)
                    llm_output = llm_response.get('output', '')

                    # Try to parse JSON from the response
                    try:
                        # Look for JSON in the response (might be surrounded by text)
                        import re
                        json_match = re.search(r'({[\s\S]*})', llm_output)
                        if json_match:
                            llm_json_str = json_match.group(1)
                            llm_insights = json.loads(llm_json_str)
                        else:
                            # Create structured insights manually
                            llm_insights = {
                                "llm_analysis": llm_output
                            }
                    except:
                        llm_insights = {
                            "llm_analysis": llm_output
                        }

                    # Add LLM insights to result
                    multi_file_results["llm_insights"] = llm_insights
                except Exception as e:
                    # If LLM enhancement fails, add error message but continue
                    multi_file_results["llm_insights"] = {
                        "error": f"LLM enhancement failed: {str(e)}",
                        "note": "Computational results are still valid."
                    }

            # Return the multi-file results
            return {"text": json.dumps(multi_file_results, cls=NumpyJSONEncoder)}

    except Exception as e:
        error_details = traceback.format_exc()
        return {"text": json.dumps({
            "error": f"Error performing formation evaluation: {str(e)}",
            "details": error_details
        }, cls=NumpyJSONEncoder)}


# Well Correlation Tool with optional LLM enhancement
def enhanced_well_correlation(well_list=None, marker_curve="GR", data_dir="./data", llm_agent=None, **kwargs):
    """
    Enhanced well correlation tool for identifying formation tops across multiple wells
    """
    from well_correlation import correlate_wells, create_correlation_summary

    try:
        # PREPROCESSING: Handle malformed agent input
        if isinstance(well_list, str):
            processed_input = preprocess_agent_input(well_list)
            if isinstance(processed_input, dict):
                # Extract parameters from the processed JSON
                if 'files' in processed_input:
                    file_paths = processed_input['files']
                    kwargs['file_paths'] = file_paths
                    well_list = None  # Clear well_list since we're using file_paths
                if 'curve' in processed_input:
                    marker_curve = processed_input['curve']
                # Add any other parameters
                for key, value in processed_input.items():
                    if key not in ['files', 'curve']:
                        kwargs[key] = value

        # Extract parameters
        llm_enhance = kwargs.pop('llm_enhance', False)
        selection_mode = kwargs.pop('selection_mode', "all")  # Default to process all for correlation
        max_files = int(kwargs.pop('max_files', 10))
        file_paths = kwargs.pop('file_paths', None)  # Specific paths to use
        file_indices = kwargs.pop('file_indices', None)  # Indices to select from matches

        # Handle keyword arguments
        if 'input' in kwargs:
            # Check if input is a JSON string
            input_processed = preprocess_agent_input(kwargs['input'])
            if isinstance(input_processed, dict):
                # Extract file selection parameters
                if 'well_list' in input_processed:
                    well_list = input_processed['well_list']
                if 'files' in input_processed:
                    file_paths = input_processed['files']
                if 'marker_curve' in input_processed:
                    marker_curve = input_processed['marker_curve']
                if 'curve' in input_processed:
                    marker_curve = input_processed['curve']
                if 'selection_mode' in input_processed:
                    selection_mode = input_processed['selection_mode']
                if 'max_files' in input_processed:
                    max_files = int(input_processed['max_files'])
                if 'file_paths' in input_processed:
                    file_paths = input_processed['file_paths']
                if 'file_indices' in input_processed:
                    file_indices = input_processed['file_indices']

                # Extract other parameters
                for param in ['depth_tolerance', 'prominence', 'summary_only', 'llm_enhance']:
                    if param in input_processed:
                        if param == 'llm_enhance':
                            llm_enhance = input_processed[param]
                        else:
                            kwargs[param] = input_processed[param]
            elif isinstance(input_processed, str):
                # Not JSON, treat as well list or pattern
                well_list = input_processed

        # Debug logging
        print(f"DEBUG: After preprocessing - well_list={well_list}")
        print(f"DEBUG: After preprocessing - file_paths={file_paths}")
        print(f"DEBUG: After preprocessing - marker_curve={marker_curve}")
        print(f"DEBUG: data_dir={data_dir}")

        # Find well files to process
        well_files = []

        # If we have specific file_paths, use those directly
        if file_paths:
            print(f"DEBUG: Processing file_paths: {file_paths}")
            for path in file_paths:
                # Handle both relative and absolute paths
                if os.path.isabs(path):
                    full_path = path
                else:
                    # Try path as-is first
                    if os.path.isfile(path):
                        full_path = path
                    else:
                        # Try with data_dir
                        full_path = os.path.join(data_dir, path)
                        if not os.path.isfile(full_path):
                            # Try just the filename with data_dir
                            basename = os.path.basename(path)
                            full_path = os.path.join(data_dir, basename)

                print(f"DEBUG: Checking path: {path} -> {full_path}")
                if os.path.isfile(full_path):
                    well_files.append(full_path)
                    print(f"DEBUG: Added file: {full_path}")
                else:
                    print(f"DEBUG: File not found: {full_path}")

            if len(well_files) < 2:
                return {"text": json.dumps({
                    "error": f"Found only {len(well_files)} valid LAS files. At least 2 required for correlation.",
                    "files_found": well_files,
                    "files_requested": file_paths,
                    "data_dir": data_dir
                })}
        # Handle different input types for well_list
        else:
            matching_files = []

            # Make sure we have wells (unless specific file_paths were provided)
            if well_list is None:
                return {"text": json.dumps({"error": "No wells provided"})}

            print(f"DEBUG: Processing well_list: {well_list}")

            # Case 1: well_list is a single string that might be a glob pattern
            if isinstance(well_list, str) and ('*' in well_list or '?' in well_list):
                print(f"DEBUG: Pattern detected: {well_list}")
                # It's a pattern, find matching files
                matching_files = find_las_files_by_pattern(well_list, data_dir)
                print(f"DEBUG: Pattern matches: {matching_files}")

            # Case 2: well_list is a comma-separated list
            elif isinstance(well_list, str):
                print(f"DEBUG: Processing as comma-separated list: {well_list}")
                well_items = [w.strip() for w in well_list.split(",")]
                for well in well_items:
                    # Check if individual item is a pattern
                    if '*' in well or '?' in well:
                        print(f"DEBUG: Processing pattern item: {well}")
                        pattern_matches = find_las_files_by_pattern(well, data_dir)
                        print(f"DEBUG: Pattern item matches: {pattern_matches}")
                        if pattern_matches:
                            matching_files.extend(pattern_matches)
                    else:
                        print(f"DEBUG: Processing individual file: {well}")
                        file_path = find_las_file(well, data_dir)
                        print(f"DEBUG: Found file path: {file_path}")
                        if os.path.isfile(file_path):
                            matching_files.append(file_path)

            # Case 3: well_list is already a list
            elif isinstance(well_list, list):
                print(f"DEBUG: Processing as list: {well_list}")
                for well in well_list:
                    # Check if individual item is a pattern
                    if isinstance(well, str) and ('*' in well or '?' in well):
                        print(f"DEBUG: Processing list pattern item: {well}")
                        pattern_matches = find_las_files_by_pattern(well, data_dir)
                        print(f"DEBUG: List pattern item matches: {pattern_matches}")
                        if pattern_matches:
                            matching_files.extend(pattern_matches)
                    else:
                        print(f"DEBUG: Processing list individual file: {well}")
                        file_path = find_las_file(well, data_dir)
                        print(f"DEBUG: Found list file path: {file_path}")
                        if os.path.isfile(file_path):
                            matching_files.append(file_path)

            print(f"DEBUG: Total matching files found: {len(matching_files)}")
            print(f"DEBUG: Matching files: {matching_files}")

            # Remove duplicates while preserving order
            well_files = []
            seen = set()
            for f in matching_files:
                if f not in seen:
                    well_files.append(f)
                    seen.add(f)

            print(f"DEBUG: Unique well files: {len(well_files)}")

            # If we have file_indices, select those specific files
            if well_files and file_indices:
                try:
                    # Convert 1-based indices to 0-based
                    indices = [int(idx) - 1 for idx in file_indices]
                    selected_files = [well_files[i] for i in indices if 0 <= i < len(well_files)]
                    well_files = selected_files

                    if len(well_files) < 2:
                        return {"text": json.dumps({
                            "error": f"Selected only {len(well_files)} valid LAS files. At least 2 required for correlation.",
                            "files_selected": well_files
                        })}
                except (ValueError, TypeError):
                    return {"text": json.dumps({"error": "Invalid file indices provided"})}
            else:
                # Use the selection helper function only if we have matches
                if not well_files:
                    return {"text": json.dumps({
                        "error": "No matching LAS files found",
                        "pattern": well_list,
                        "data_dir": data_dir
                    })}

                selection_result = select_files_from_matches(well_files, selection_mode, max_files)

                # If we need to ask the user to choose (and haven't been provided with choices)
                if selection_result["status"] == "choose":
                    return format_file_selection_response(selection_result)

                well_files = selection_result["files"]

        # Verify we have enough wells for correlation
        if len(well_files) < 2:
            return {"text": json.dumps({
                "error": f"Found only {len(well_files)} valid LAS files. At least 2 required for correlation.",
                "files_found": well_files
            })}

        print(f"DEBUG: Final well_files list: {well_files}")

        # Extract parameters from kwargs
        depth_tolerance = float(kwargs.get('depth_tolerance', 5.0))
        prominence = float(kwargs.get('prominence', 0.3))
        summary_only = kwargs.get('summary_only', False)

        # Perform well correlation
        print(f"DEBUG: Calling correlate_wells with {len(well_files)} files")
        result = correlate_wells(
            well_files,
            marker_curve=marker_curve,
            depth_tolerance=depth_tolerance,
            prominence=prominence
        )

        # Check for errors in correlation result
        if "error" in result:
            return {"text": json.dumps(result, cls=NumpyJSONEncoder)}

        # Create human-readable summary
        summary = create_correlation_summary(result)

        # Add file info to result
        result["files_processed"] = [os.path.basename(f) for f in well_files]

        # If LLM enhancement is requested and an LLM agent is available
        if llm_enhance and llm_agent:
            try:
                llm_query = f"""As a stratigraphy and well correlation expert, analyze these well correlation results and provide geological insights:

                WELLS: {', '.join(result.get('wells', []))}
                MARKER CURVE: {result.get('marker_curve', 'Unknown')}

                SUMMARY:
                {summary}

                Please provide:
                1. Interpretation of the stratigraphic sequence based on these correlations
                2. Potential geological structures or depositional trends across these wells
                3. Confidence assessment of the correlations and suggestions for improvement
                4. Possible names for the marker horizons based on regional geology

                Format your response as JSON with these keys: "stratigraphic_interpretation", "structural_trends", "correlation_assessment", "marker_naming"
                """

                # Get LLM insights
                llm_response = llm_agent.invoke(llm_query)
                llm_output = llm_response.get('output', '')

                # Try to parse JSON from the response
                try:
                    # Look for JSON in the response (might be surrounded by text)
                    import re
                    json_match = re.search(r'({[\s\S]*})', llm_output)
                    if json_match:
                        llm_json_str = json_match.group(1)
                        llm_insights = json.loads(llm_json_str)
                    else:
                        # Create structured insights manually
                        llm_insights = {
                            "llm_analysis": llm_output
                        }
                except:
                    llm_insights = {
                        "llm_analysis": llm_output
                    }

                # Add LLM insights to result
                result["llm_insights"] = llm_insights
            except Exception as e:
                # If LLM enhancement fails, add error message but continue
                result["llm_insights"] = {
                    "error": f"LLM enhancement failed: {str(e)}",
                    "note": "Computational results are still valid."
                }

        # If summary only, return just the text summary and optional LLM insights
        if summary_only:
            response = {"summary": summary}
            if llm_enhance and llm_agent and "llm_insights" in result:
                response["llm_insights"] = result["llm_insights"]
            return {"text": json.dumps(response)}

        # Add summary to result
        result["text_summary"] = summary

        # Return the full correlation results as JSON
        return {"text": json.dumps(result, cls=NumpyJSONEncoder)}

    except Exception as e:
        error_details = traceback.format_exc()
        return {"text": json.dumps({
            "error": f"Error performing well correlation: {str(e)}",
            "details": error_details
        }, cls=NumpyJSONEncoder)}


# Add this to enhanced_mcp_tools.py - Enhanced correlation function

def enhanced_well_correlation_with_qc(well_list=None, marker_curve="GR", data_dir="./data",
                                      llm_agent=None, **kwargs):
    """
    Enhanced well correlation with quality control pre-filtering
    """
    try:
        # First, get the list of files using existing logic
        result = enhanced_well_correlation(well_list, marker_curve, data_dir, llm_agent, **kwargs)

        # If we got a successful file list but no correlations, try enhanced methods
        if isinstance(result, dict) and "text" in result:
            result_data = json.loads(result["text"])

            if "error" in result_data and "Found only" in result_data["error"]:
                # The file finding worked, try the enhanced correlation
                return result
            elif "formation_count" in result_data and result_data["formation_count"] == 0:
                # We have files but no correlations found - try enhanced methods
                files_processed = result_data.get("files_processed", [])
                if len(files_processed) >= 2:
                    # Try with multiple curves and adaptive parameters
                    well_files = []
                    for filename in files_processed:
                        full_path = os.path.join(data_dir, filename)
                        if os.path.isfile(full_path):
                            well_files.append(full_path)

                    if len(well_files) >= 2:
                        # Try enhanced correlation
                        enhanced_result = try_multiple_curves_and_params(well_files)

                        if enhanced_result and "error" not in enhanced_result:
                            # Create summary
                            from well_correlation import create_correlation_summary
                            summary = create_correlation_summary(enhanced_result)
                            enhanced_result["text_summary"] = summary
                            enhanced_result["enhancement_note"] = "Used adaptive parameters and multiple curves"
                            return {"text": json.dumps(enhanced_result, cls=NumpyJSONEncoder)}
                        else:
                            # Still no good correlation, provide helpful guidance
                            guidance = create_correlation_guidance(well_files, data_dir)
                            result_data["enhancement_guidance"] = guidance
                            return {"text": json.dumps(result_data, cls=NumpyJSONEncoder)}

        return result

    except Exception as e:
        error_details = traceback.format_exc()
        return {"text": json.dumps({
            "error": f"Error in enhanced correlation: {str(e)}",
            "details": error_details
        })}


def try_multiple_curves_and_params(well_files):
    """Try correlation with multiple curves and parameters"""
    from robust_las_parser import load_las_file
    from well_correlation import correlate_wells

    # Find common curves
    common_curves = None
    for file_path in well_files:
        las, error = load_las_file(file_path)
        if error:
            continue

        file_curves = set(las.get_curve_names())
        if common_curves is None:
            common_curves = file_curves
        else:
            common_curves = common_curves.intersection(file_curves)

    if not common_curves:
        return {"error": "No common curves found across wells"}

    # Priority curves for correlation
    priority_curves = ["GR", "SP", "RHOB", "NPHI", "RT", "RILD", "RILM", "DT"]
    test_curves = [curve for curve in priority_curves if curve in common_curves]

    # Add other curves
    for curve in common_curves:
        if curve not in test_curves and not curve.upper().startswith(('DEPT', 'DEPTH')):
            test_curves.append(curve)

    # Parameter sets to try
    param_sets = [
        {"depth_tolerance": 5.0, "prominence": 0.2, "min_distance": 5},
        {"depth_tolerance": 10.0, "prominence": 0.15, "min_distance": 8},
        {"depth_tolerance": 15.0, "prominence": 0.1, "min_distance": 10},
        {"depth_tolerance": 20.0, "prominence": 0.25, "min_distance": 12},
    ]

    best_result = None
    best_score = 0

    for curve in test_curves:
        for params in param_sets:
            try:
                result = correlate_wells(
                    well_files,
                    marker_curve=curve,
                    **params
                )

                if "error" not in result and result.get("formation_count", 0) > 0:
                    # Calculate score
                    formation_count = result["formation_count"]
                    avg_confidence = sum(
                        top.get("confidence", 0) for top in result.get("formation_tops", [])
                    ) / formation_count
                    score = formation_count * avg_confidence

                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_result["best_curve"] = curve
                        best_result["best_params"] = params

            except Exception as e:
                continue

    return best_result


def preprocess_agent_input(input_str):
    """
    Preprocess input from the agent to handle malformed JSON

    Args:
        input_str: Raw input string from the agent

    Returns:
        Cleaned input ready for processing
    """
    if isinstance(input_str, str):
        # Handle "JSON with parameters" format
        if input_str.startswith("JSON with parameters"):
            # Extract the actual JSON part
            json_start = input_str.find("{")
            if json_start != -1:
                json_part = input_str[json_start:]
                try:
                    # Try to parse the JSON
                    parsed = json.loads(json_part)
                    return parsed
                except json.JSONDecodeError:
                    # If parsing fails, return original
                    return input_str

        # Handle other JSON-like strings
        elif input_str.startswith("{") and input_str.endswith("}"):
            try:
                parsed = json.loads(input_str)
                return parsed
            except json.JSONDecodeError:
                return input_str

    return input_str


def create_correlation_guidance(well_files, data_dir):
    """Create guidance for improving correlation results"""
    from robust_las_parser import load_las_file, perform_quality_check

    guidance = {
        "wells_analyzed": len(well_files),
        "quality_summary": [],
        "recommendations": []
    }

    # Check quality of each well
    poor_quality_wells = []
    good_curves = set()

    for file_path in well_files:
        las, error = load_las_file(file_path)
        if error:
            continue

        qc = perform_quality_check(las)
        well_name = las.well_info.get("WELL", os.path.basename(file_path))

        guidance["quality_summary"].append({
            "well": well_name,
            "quality": qc.get("quality_rating", "Unknown"),
            "curves": len(las.get_curve_names())
        })

        if qc.get("quality_rating") in ["Poor", "Fair"]:
            poor_quality_wells.append(well_name)
        else:
            # Track good curves from high-quality wells
            good_curves.update(las.get_curve_names())

    # Generate recommendations
    if poor_quality_wells:
        guidance["recommendations"].append(
            f"Consider excluding poor quality wells: {', '.join(poor_quality_wells)}"
        )

    if good_curves:
        priority_curves = ["GR", "SP", "RHOB", "NPHI", "RT"]
        available_priority = [c for c in priority_curves if c in good_curves]
        if available_priority:
            guidance["recommendations"].append(
                f"Try these alternative marker curves: {', '.join(available_priority)}"
            )

    guidance["recommendations"].extend([
        "Consider increasing depth tolerance (try 10-20 meters)",
        "Reduce prominence threshold (try 0.1-0.2)",
        "Check if wells are from the same geological area",
        "Verify depth units are consistent across wells"
    ])

    return guidance