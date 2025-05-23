"""
well_correlation.py - Module for correlating formations across multiple wells

This module provides functions for identifying and correlating key formation tops
across multiple wells using log curve pattern matching.
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, peak_prominences
from scipy.interpolate import interp1d
from typing import List, Dict, Any, Tuple, Optional
import json
import traceback

# Import the robust LAS parser
from robust_las_parser import load_las_file, RobustLASFile

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

def preprocess_curve(curve_data, window_size=5):
    """
    Preprocess a log curve by smoothing and normalizing

    Args:
        curve_data: NumPy array of curve values
        window_size: Window size for smoothing filter

    Returns:
        NumPy array: Preprocessed curve data
    """
    # Remove NaN values
    nan_indices = np.isnan(curve_data)
    clean_data = curve_data.copy()

    if np.any(nan_indices):
        # Fill NaN values with nearby valid values via interpolation
        valid_indices = np.where(~nan_indices)[0]
        if len(valid_indices) == 0:
            return np.zeros_like(curve_data)  # All NaNs, return zeros

        # Create interpolation function based on valid data
        valid_values = curve_data[valid_indices]
        interp_func = interp1d(valid_indices, valid_values,
                               kind='linear', bounds_error=False,
                               fill_value=(valid_values[0], valid_values[-1]))

        # Generate full index array and interpolate
        all_indices = np.arange(len(curve_data))
        clean_data = interp_func(all_indices)

    # Apply Savitzky-Golay filter for smoothing
    smoothed_data = signal.savgol_filter(clean_data, window_size, 2)

    # Normalize to range [0, 1]
    min_val = np.min(smoothed_data)
    max_val = np.max(smoothed_data)

    if max_val > min_val:
        normalized_data = (smoothed_data - min_val) / (max_val - min_val)
        return normalized_data
    else:
        return np.zeros_like(smoothed_data)  # Avoid division by zero


# Add this enhanced correlation function to well_correlation.py

def correlate_wells_adaptive(well_files, marker_curve="GR", adaptive_params=True):
    """
    Enhanced well correlation with adaptive parameter tuning

    Args:
        well_files: List of LAS file paths
        marker_curve: Curve to use for correlation
        adaptive_params: Whether to try multiple parameter sets

    Returns:
        Best correlation result found
    """
    if adaptive_params:
        # Try multiple parameter combinations
        param_sets = [
            {"depth_tolerance": 10.0, "prominence": 0.2, "min_distance": 5},
            {"depth_tolerance": 15.0, "prominence": 0.15, "min_distance": 8},
            {"depth_tolerance": 5.0, "prominence": 0.3, "min_distance": 10},
            {"depth_tolerance": 20.0, "prominence": 0.1, "min_distance": 15},
        ]

        best_result = None
        best_score = 0

        for params in param_sets:
            try:
                result = correlate_wells(
                    well_files,
                    marker_curve=marker_curve,
                    **params
                )

                if "error" not in result:
                    # Score based on number of formation tops and average confidence
                    formation_count = result.get("formation_count", 0)
                    if formation_count > 0:
                        avg_confidence = sum(
                            top.get("confidence", 0) for top in result.get("formation_tops", [])
                        ) / formation_count
                        score = formation_count * avg_confidence

                        if score > best_score:
                            best_score = score
                            best_result = result
                            best_result["correlation_params_used"] = params

            except Exception as e:
                print(f"Parameter set {params} failed: {str(e)}")
                continue

        if best_result:
            return best_result

    # Fall back to default parameters
    return correlate_wells(
        well_files,
        marker_curve=marker_curve,
        depth_tolerance=5.0,
        prominence=0.3,
        min_distance=10
    )


# Add this alternative curve correlation function
def try_multiple_curves(well_files, curves=None):
    """
    Try correlation with multiple curves to find the best result

    Args:
        well_files: List of LAS file paths
        curves: List of curves to try (if None, will auto-detect)

    Returns:
        Best correlation result across all curves
    """
    from robust_las_parser import load_las_file

    if curves is None:
        # Auto-detect common curves across all wells
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

        # Prioritize common correlation curves
        priority_curves = ["GR", "SP", "RHOB", "NPHI", "RT", "RILD", "RILM"]
        curves = [curve for curve in priority_curves if curve in common_curves]

        # Add any other common curves
        for curve in common_curves:
            if curve not in curves and curve not in ["DEPT", "DEPTH"]:
                curves.append(curve)

    best_result = None
    best_score = 0

    for curve in curves:
        try:
            result = correlate_wells_adaptive(well_files, marker_curve=curve)

            if "error" not in result:
                formation_count = result.get("formation_count", 0)
                if formation_count > 0:
                    avg_confidence = sum(
                        top.get("confidence", 0) for top in result.get("formation_tops", [])
                    ) / formation_count
                    score = formation_count * avg_confidence

                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_result["best_marker_curve"] = curve

        except Exception as e:
            print(f"Curve {curve} failed: {str(e)}")
            continue

    return best_result if best_result else {"error": "No successful correlations found with any curve"}

def simple_dtw(x, y):
    """
    Simple implementation of Dynamic Time Warping

    Args:
        x: First sequence
        y: Second sequence

    Returns:
        float: DTW distance between sequences
    """
    n, m = len(x), len(y)

    # Initialize cost matrix
    dtw_matrix = np.zeros((n+1, m+1))

    # Fill first row and column with infinity
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf
    dtw_matrix[0, 0] = 0

    # Compute DTW matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]   # match
            )

    # Return DTW distance
    return dtw_matrix[n, m]

def identify_inflection_points(depth, curve_data, window_size=5, prominence=0.3, min_distance=10):
    """
    Identify significant inflection points in a log curve

    Args:
        depth: Depth array
        curve_data: Log curve values array
        window_size: Window size for preprocessing
        prominence: Minimum prominence for peak detection (0-1 range)
        min_distance: Minimum distance between peaks

    Returns:
        Dict: Inflection points with depths and characteristics
    """
    # Preprocess the curve
    preprocessed_data = preprocess_curve(curve_data, window_size)

    # First derivative (gradient)
    gradient = np.gradient(preprocessed_data)

    # Find inflection points using peak finding on gradient
    # Positive peaks = transitions from decreasing to increasing (minima in original curve)
    pos_peaks, pos_properties = find_peaks(gradient, prominence=prominence, distance=min_distance)

    # Negative peaks = transitions from increasing to decreasing (maxima in original curve)
    neg_peaks, neg_properties = find_peaks(-gradient, prominence=prominence, distance=min_distance)

    # Calculate additional properties for each inflection point
    inflection_points = []

    # Process positive inflection points (local minima in original curve)
    for i, peak_idx in enumerate(pos_peaks):
        if peak_idx < len(depth):
            point = {
                "depth": float(depth[peak_idx]),
                "value": float(curve_data[peak_idx]),
                "type": "minimum",
                "prominence": float(pos_properties["prominences"][i]),
                "gradient": float(gradient[peak_idx])
            }
            inflection_points.append(point)

    # Process negative inflection points (local maxima in original curve)
    for i, peak_idx in enumerate(neg_peaks):
        if peak_idx < len(depth):
            point = {
                "depth": float(depth[peak_idx]),
                "value": float(curve_data[peak_idx]),
                "type": "maximum",
                "prominence": float(neg_properties["prominences"][i]),
                "gradient": float(gradient[peak_idx])
            }
            inflection_points.append(point)

    # Sort by depth
    inflection_points.sort(key=lambda x: x["depth"])

    return {
        "count": len(inflection_points),
        "points": inflection_points
    }

def extract_curve_segments(depth, curve_data, inflection_points, padding=5):
    """
    Extract characteristic curve segments around inflection points

    Args:
        depth: Depth array
        curve_data: Log curve values array
        inflection_points: Dict of inflection points from identify_inflection_points
        padding: Number of samples to include before and after inflection point

    Returns:
        List[Dict]: List of curve segments with characteristics
    """
    segments = []

    for point in inflection_points["points"]:
        # Find index of the inflection point depth
        depth_idx = np.argmin(np.abs(depth - point["depth"]))

        # Extract segment around inflection point
        start_idx = max(0, depth_idx - padding)
        end_idx = min(len(depth) - 1, depth_idx + padding)

        segment_depth = depth[start_idx:end_idx+1]
        segment_data = curve_data[start_idx:end_idx+1]

        # Calculate segment characteristics
        segment = {
            "center_depth": float(point["depth"]),
            "start_depth": float(segment_depth[0]),
            "end_depth": float(segment_depth[-1]),
            "type": point["type"],
            "prominence": point["prominence"],
            "mean": float(np.mean(segment_data)),
            "std": float(np.std(segment_data)),
            "data": list(map(float, segment_data)),
            "depths": list(map(float, segment_depth))
        }

        segments.append(segment)

    return segments

def correlate_segments(reference_segments, target_segments, depth_tolerance=5.0):
    """
    Correlate curve segments between reference and target wells

    Args:
        reference_segments: List of segments from reference well
        target_segments: List of segments from target well
        depth_tolerance: Maximum depth difference to consider (in depth units)

    Returns:
        List[Dict]: Correlation results with confidence scores
    """
    correlations = []

    # For each reference segment, find best match in target segments
    for ref_idx, ref_segment in enumerate(reference_segments):
        best_match = None
        best_confidence = 0.0
        best_target_idx = None

        # Find potential matches within depth tolerance
        potential_matches = []
        for tgt_idx, tgt_segment in enumerate(target_segments):
            depth_diff = abs(ref_segment["center_depth"] - tgt_segment["center_depth"])

            # Only consider segments within depth tolerance and of same type
            if depth_diff <= depth_tolerance and ref_segment["type"] == tgt_segment["type"]:
                potential_matches.append((tgt_idx, tgt_segment, depth_diff))

        # Sort potential matches by depth difference
        potential_matches.sort(key=lambda x: x[2])

        # Evaluate similarity of potential matches
        for tgt_idx, tgt_segment, depth_diff in potential_matches:
            # Use dynamic time warping to compare segments
            ref_data = np.array(ref_segment["data"])
            tgt_data = np.array(tgt_segment["data"])

            # Ensure data arrays are same length
            min_len = min(len(ref_data), len(tgt_data))
            ref_data = ref_data[:min_len]
            tgt_data = tgt_data[:min_len]

            # Calculate similarity using DTW distance
            dtw_distance = simple_dtw(ref_data, tgt_data)

            # Normalize DTW distance to 0-1 range (lower is better)
            max_possible_dist = min_len * 1.0  # Approximate max distance
            similarity = 1.0 - (dtw_distance / max_possible_dist)
            similarity = max(0.0, similarity)  # Ensure non-negative

            # Calculate depth match score (1.0 at 0 difference, 0.0 at tolerance)
            depth_score = 1.0 - (depth_diff / depth_tolerance)

            # Calculate overall confidence as weighted combination
            confidence = 0.7 * similarity + 0.3 * depth_score

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = tgt_segment
                best_target_idx = tgt_idx

        # Add correlation if match found
        if best_match is not None:
            correlation = {
                "reference": {
                    "depth": ref_segment["center_depth"],
                    "type": ref_segment["type"],
                    "idx": ref_idx
                },
                "target": {
                    "depth": best_match["center_depth"],
                    "type": best_match["type"],
                    "idx": best_target_idx
                },
                "depth_difference": abs(ref_segment["center_depth"] - best_match["center_depth"]),
                "confidence": best_confidence
            }
            correlations.append(correlation)

    # Sort correlations by reference depth
    correlations.sort(key=lambda x: x["reference"]["depth"])

    return correlations

def normalize_depths(well_data, base_well_idx=0):
    """
    Normalize depth arrays across wells to a common reference

    Args:
        well_data: List of dicts with depth and curve data for each well
        base_well_idx: Index of well to use as base for normalization

    Returns:
        List[Dict]: Updated well data with normalized depths
    """
    # Use first well as reference by default
    base_well = well_data[base_well_idx]
    base_depth_min = np.min(base_well["depth"])
    base_depth_max = np.max(base_well["depth"])

    normalized_wells = []

    for well in well_data:
        well_depth_min = np.min(well["depth"])
        well_depth_max = np.max(well["depth"])

        # Skip normalization for base well
        if well is base_well:
            normalized_wells.append(well.copy())
            continue

        # Calculate normalization factors
        # Scale = ratio of depth ranges
        scale = (base_depth_max - base_depth_min) / (well_depth_max - well_depth_min)
        # Offset = difference in minimum depths
        offset = base_depth_min - (well_depth_min * scale)

        # Create normalized well data
        normalized_well = well.copy()
        normalized_well["normalized_depth"] = well["depth"] * scale + offset
        normalized_well["depth_scale"] = scale
        normalized_well["depth_offset"] = offset

        normalized_wells.append(normalized_well)

    return normalized_wells

def correlate_wells(well_files, marker_curve="GR", depth_tolerance=5.0, prominence=0.3, min_distance=10):
    """
    Correlate formations across multiple wells

    Args:
        well_files: List of paths to LAS files
        marker_curve: Name of the curve to use for correlation (default: "GR")
        depth_tolerance: Maximum depth difference for correlation (in meters/feet)
        prominence: Minimum prominence for peak detection (0-1 range)
        min_distance: Minimum distance between markers (in samples)

    Returns:
        Dict: Correlation results with markers and confidence levels
    """
    if len(well_files) < 2:
        return {"error": "At least two wells are required for correlation"}

    # Load wells and extract curve data
    wells_data = []
    well_names = []

    for file_path in well_files:
        las, error = load_las_file(file_path)
        if error:
            return {"error": f"Error loading {file_path}: {error}"}

        # Check if marker curve exists
        if not las.curve_exists(marker_curve):
            available_curves = las.get_curve_names()
            return {
                "error": f"Marker curve '{marker_curve}' not found in {file_path}",
                "available_curves": available_curves
            }

        # Extract curve data
        depth = las.index
        curve_data = las.get_curve_data(marker_curve)

        # Store well data
        well_name = las.well_info.get("WELL", os.path.basename(file_path))
        well_names.append(well_name)

        well_data = {
            "name": well_name,
            "file": file_path,
            "depth": depth,
            "curve_data": curve_data
        }

        wells_data.append(well_data)

    # Use the first well as reference
    reference_well = wells_data[0]

    # Identify inflection points (potential markers) in each well
    print(f"Identifying inflection points in {len(well_files)} wells...")

    for well in wells_data:
        well["inflection_points"] = identify_inflection_points(
            well["depth"],
            well["curve_data"],
            prominence=prominence,
            min_distance=min_distance
        )
        well["segments"] = extract_curve_segments(
            well["depth"],
            well["curve_data"],
            well["inflection_points"]
        )

    # Normalize depths if needed
    # wells_data = normalize_depths(wells_data)

    # Correlate each well with reference well
    correlations = []

    for i, target_well in enumerate(wells_data[1:], 1):
        well_correlation = correlate_segments(
            reference_well["segments"],
            target_well["segments"],
            depth_tolerance=depth_tolerance
        )

        correlation_result = {
            "reference_well": reference_well["name"],
            "target_well": target_well["name"],
            "marker_curve": marker_curve,
            "correlations": well_correlation,
            "correlation_count": len(well_correlation)
        }

        correlations.append(correlation_result)

    # Create formation tops based on correlations
    formation_tops = []
    high_confidence_threshold = 0.75

    reference_markers = reference_well["inflection_points"]["points"]

    # Group correlations by reference marker
    for i, marker in enumerate(reference_markers):
        # Skip if not enough info to make a formation top
        if marker["prominence"] < prominence:
            continue

        # Find correlations for this marker
        marker_correlations = []
        for corr_set in correlations:
            matching_corrs = [c for c in corr_set["correlations"]
                            if c["reference"]["idx"] == i]
            if matching_corrs:
                best_match = max(matching_corrs, key=lambda x: x["confidence"])
                marker_correlations.append({
                    "well": corr_set["target_well"],
                    "depth": best_match["target"]["depth"],
                    "confidence": best_match["confidence"]
                })

        # Calculate average confidence
        if marker_correlations:
            avg_confidence = sum(c["confidence"] for c in marker_correlations) / len(marker_correlations)

            # Only create formation top if confidence is high enough and correlates in multiple wells
            if avg_confidence >= high_confidence_threshold and len(marker_correlations) >= len(well_files) // 2:
                formation_top = {
                    "name": f"Marker_{i+1}",
                    "reference_depth": marker["depth"],
                    "reference_well": reference_well["name"],
                    "type": marker["type"],
                    "prominence": marker["prominence"],
                    "confidence": avg_confidence,
                    "well_depths": [
                        {"well": reference_well["name"], "depth": marker["depth"], "confidence": 1.0}
                    ] + marker_correlations
                }
                formation_tops.append(formation_top)

    # Final result
    result = {
        "wells": well_names,
        "reference_well": reference_well["name"],
        "marker_curve": marker_curve,
        "well_correlations": correlations,
        "formation_tops": formation_tops,
        "formation_count": len(formation_tops),
        "correlation_parameters": {
            "depth_tolerance": depth_tolerance,
            "prominence": prominence,
            "min_distance": min_distance
        }
    }

    return result

def create_correlation_summary(result):
    """
    Create a human-readable summary of correlation results

    Args:
        result: Result dictionary from correlate_wells function

    Returns:
        str: Human-readable summary
    """
    if "error" in result:
        return f"Error in well correlation: {result['error']}"

    # Extract key information
    wells = result["wells"]
    reference_well = result["reference_well"]
    marker_curve = result["marker_curve"]
    formation_tops = result["formation_tops"]

    # Create summary text
    summary = f"Well Correlation Summary using {marker_curve}\n"
    summary += f"=========================================\n\n"

    summary += f"Reference well: {reference_well}\n"
    summary += f"Wells correlated: {', '.join(wells)}\n\n"

    summary += f"Formation Tops Identified: {len(formation_tops)}\n\n"

    if formation_tops:
        summary += "Key Formation Tops:\n"
        for i, top in enumerate(formation_tops, 1):
            confidence_percent = top["confidence"] * 100
            summary += f"{i}. {top['name']} (Confidence: {confidence_percent:.1f}%)\n"

            # Well depths for this formation top
            summary += f"   Reference: {reference_well} at depth {top['reference_depth']:.1f}\n"

            for well_depth in top["well_depths"]:
                if well_depth["well"] != reference_well:
                    well_conf = well_depth["confidence"] * 100
                    summary += f"   {well_depth['well']}: depth {well_depth['depth']:.1f} "
                    summary += f"(Confidence: {well_conf:.1f}%)\n"

            summary += "\n"
    else:
        summary += "No reliable formation tops identified across wells.\n"
        summary += "Consider adjusting correlation parameters or using a different marker curve.\n"

    summary += "\nRecommendations:\n"
    if len(formation_tops) > 0:
        summary += "- The identified formation tops can be used for structural mapping.\n"
        summary += "- Consider using these markers for well-to-well correlation in geological modeling.\n"
    else:
        summary += "- Try decreasing the prominence threshold to capture more potential markers.\n"
        summary += "- Consider using alternative marker curves if available (e.g., Resistivity).\n"
        summary += "- Normalize depths if wells are in different structural positions.\n"

    return summary

# Test code if run directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python well_correlation.py <las_file1> <las_file2> [las_file3 ...]")
        sys.exit(1)

    well_files = sys.argv[1:]
    print(f"Correlating wells: {', '.join(well_files)}")

    # Correlate wells
    result = correlate_wells(well_files)

    # Print JSON result
    print("\nDetailed Results:")
    print(json.dumps(result, indent=2, cls=NumpyJSONEncoder))

    # Print human-readable summary
    print("\nSummary Report:")
    print(create_correlation_summary(result))