"""
production_segy_analysis_qc.py - Production-quality SEG-Y analysis and QC tools

This module provides robust analysis and quality control capabilities for SEG-Y files
with comprehensive error handling, validation, and progress reporting.

UPDATED: Now uses segyio as the core engine with calibrated quality thresholds
for accurate assessment of real-world seismic data.
"""

import os
import sys
import json
import traceback
import numpy as np
import math
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
import segyio
from enum import Enum

# Import compatibility components from the updated production_segy_tools
from production_segy_tools import (
    NumpyJSONEncoder, ProgressReporter, MemoryMonitor,
    find_segy_file, find_template_file
)

logger = logging.getLogger(__name__)

class QualityRating(Enum):
    """Quality rating enumeration"""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"
    INVALID = "Invalid"

class SurveyType(Enum):
    """Survey type for quality assessment"""
    SHOT_GATHER = "shot_gather"
    CDP_STACK = "cdp_stack"
    MIGRATED_2D = "migrated_2d"
    MIGRATED_3D = "migrated_3d"
    UNKNOWN = "unknown"

class SegyioQualityAnalyzer:
    """Production-quality SEG-Y analyzer using segyio with calibrated thresholds"""

    def __init__(self, max_memory_gb: float = 4.0):
        self.memory_monitor = MemoryMonitor(max_memory_gb)

        # CALIBRATED quality thresholds for real-world data
        self.quality_thresholds = {
            SurveyType.SHOT_GATHER: {
                'min_dynamic_range_db': 20,     # Lower for raw acquisition data
                'max_zero_percent': 60,         # Higher for muted zones
                'min_traces': 50,               # Lower minimum for shot data
                'max_amplitude_range': 1e6,     # Higher for field data
                'min_useful_samples': 100,      # Minimum samples for analysis
                'description': 'Raw acquisition shot gathers'
            },
            SurveyType.CDP_STACK: {
                'min_dynamic_range_db': 25,     # Slightly higher for stacked data
                'max_zero_percent': 40,         # Moderate for processed data
                'min_traces': 100,
                'max_amplitude_range': 1e5,
                'min_useful_samples': 200,
                'description': 'CDP stacked data'
            },
            SurveyType.MIGRATED_3D: {
                'min_dynamic_range_db': 30,     # Higher for final processed data
                'max_zero_percent': 20,         # Lower for migrated volumes
                'min_traces': 1000,
                'max_amplitude_range': 1e4,
                'min_useful_samples': 500,
                'description': 'Migrated 3D volume'
            },
            SurveyType.MIGRATED_2D: {
                'min_dynamic_range_db': 28,
                'max_zero_percent': 25,
                'min_traces': 500,
                'max_amplitude_range': 1e4,
                'min_useful_samples': 300,
                'description': 'Migrated 2D line'
            },
            SurveyType.UNKNOWN: {
                'min_dynamic_range_db': 20,     # Conservative thresholds
                'max_zero_percent': 50,
                'min_traces': 50,
                'max_amplitude_range': 1e6,
                'min_useful_samples': 100,
                'description': 'Unknown data type'
            }
        }

    def detect_survey_type_for_qc(self, file_path: str) -> SurveyType:
        """Detect survey type for appropriate QC thresholds"""
        try:
            with segyio.open(file_path, ignore_geometry=True) as f:
                # Sample headers for classification
                sample_size = min(100, f.tracecount)
                step = max(1, f.tracecount // sample_size)

                cdps = set()
                inlines = set()
                crosslines = set()
                field_records = set()

                for i in range(0, f.tracecount, step):
                    if len(cdps) > 500:  # Prevent memory issues
                        break

                    header = f.header[i]
                    cdps.add(header[segyio.TraceField.CDP])
                    inlines.add(header[segyio.TraceField.INLINE_3D])
                    crosslines.add(header[segyio.TraceField.CROSSLINE_3D])
                    field_records.add(header[segyio.TraceField.FieldRecord])

                # Remove zero values
                unique_cdps = len(cdps - {0})
                unique_inlines = len(inlines - {0})
                unique_crosslines = len(crosslines - {0})
                unique_field_records = len(field_records - {0})

                # Classification logic
                if unique_field_records > unique_cdps and unique_cdps <= 10:
                    return SurveyType.SHOT_GATHER
                elif unique_inlines > 5 and unique_crosslines > 5:
                    return SurveyType.MIGRATED_3D
                elif unique_cdps > 100 and unique_inlines <= 2:
                    return SurveyType.MIGRATED_2D
                elif unique_cdps > 20:
                    return SurveyType.CDP_STACK
                else:
                    return SurveyType.UNKNOWN

        except Exception as e:
            logger.warning(f"Survey type detection failed: {e}")
            return SurveyType.UNKNOWN

    def analyze_amplitude_distribution_segyio(self, file_path: str, max_traces: int = 50) -> Dict[str, Any]:
        """Analyze amplitude distribution using segyio with memory efficiency"""
        try:
            with segyio.open(file_path, ignore_geometry=True) as f:
                # Sample traces efficiently
                total_traces = f.tracecount
                sample_size = min(max_traces, total_traces)
                step = max(1, total_traces // sample_size)

                amplitudes = []
                zero_counts = []

                for i in range(0, total_traces, step):
                    if len(amplitudes) > 100000:  # Prevent memory explosion
                        break

                    try:
                        trace_data = f.trace[i]
                        amplitudes.extend(trace_data[::4])  # Subsample trace data
                        zero_counts.append(np.sum(trace_data == 0))
                    except Exception as e:
                        logger.debug(f"Error reading trace {i}: {e}")
                        continue

                if not amplitudes:
                    return {"error": "No amplitude data could be extracted"}

                # Convert to numpy array for analysis
                amplitude_array = np.array(amplitudes)

                # Remove NaN and infinite values
                clean_data = amplitude_array[np.isfinite(amplitude_array)]

                if len(clean_data) == 0:
                    return {"error": "No valid amplitude data found"}

                # Basic statistics
                stats_dict = {
                    "count": len(clean_data),
                    "min": float(np.min(clean_data)),
                    "max": float(np.max(clean_data)),
                    "mean": float(np.mean(clean_data)),
                    "median": float(np.median(clean_data)),
                    "std_dev": float(np.std(clean_data)),
                    "rms": float(np.sqrt(np.mean(clean_data**2))),
                    "traces_sampled": sample_size,
                    "total_traces": total_traces
                }

                # Percentiles
                percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
                stats_dict["percentiles"] = {
                    f"p{p}": float(np.percentile(clean_data, p)) for p in percentiles
                }

                # Distribution characteristics
                if len(clean_data) > 10:
                    try:
                        stats_dict["skewness"] = float(stats.skew(clean_data))
                        stats_dict["kurtosis"] = float(stats.kurtosis(clean_data))
                    except:
                        logger.warning("Could not calculate skewness/kurtosis")

                # Dynamic range (improved calculation)
                max_abs = np.max(np.abs(clean_data))
                noise_estimate = np.std(clean_data) + 1e-10  # Avoid division by zero

                if max_abs > 0 and noise_estimate > 0:
                    stats_dict["dynamic_range_db"] = float(20 * np.log10(max_abs / noise_estimate))
                    stats_dict["signal_to_noise"] = float(np.mean(np.abs(clean_data)) / noise_estimate)
                else:
                    stats_dict["dynamic_range_db"] = 0.0
                    stats_dict["signal_to_noise"] = 0.0

                # Zero percentage calculation
                total_samples = len(amplitude_array)
                zero_percentage = float(np.sum(amplitude_array == 0) / total_samples * 100) if total_samples > 0 else 0
                stats_dict["zero_percentage"] = zero_percentage

                # Data quality indicators
                stats_dict["data_quality"] = {
                    "zero_percentage": zero_percentage,
                    "constant_value_detected": len(np.unique(clean_data)) < 10,
                    "reasonable_range": abs(stats_dict["max"]) < 1e6 and abs(stats_dict["min"]) < 1e6,
                    "has_variation": stats_dict["std_dev"] > 1e-10,
                    "nan_count": int(np.sum(np.isnan(amplitude_array))),
                    "inf_count": int(np.sum(np.isinf(amplitude_array)))
                }

                return stats_dict

        except segyio.exceptions.InvalidError:
            return {"error": "Invalid SEG-Y format"}
        except Exception as e:
            return {"error": f"Amplitude analysis failed: {str(e)}"}

    def analyze_survey_geometry_segyio(self, file_path: str, max_traces: int = 1000) -> Dict[str, Any]:
        """Analyze survey geometry using segyio"""
        try:
            with segyio.open(file_path, ignore_geometry=True) as f:
                bin_header = f.bin

                geometry_info = {
                    "total_traces": f.tracecount,
                    "samples_per_trace": bin_header[segyio.BinField.Samples],
                    "sample_interval_us": bin_header[segyio.BinField.Interval],
                    "sample_interval_ms": bin_header[segyio.BinField.Interval] / 1000,
                    "format_code": bin_header[segyio.BinField.Format],
                    "analysis_traces": min(max_traces, f.tracecount)
                }

                # Sample trace headers for geometry analysis
                sample_size = min(max_traces, f.tracecount)
                step = max(1, f.tracecount // sample_size)

                coordinates = []
                inlines = []
                crosslines = []
                cdps = []
                elevations = []

                for i in range(0, f.tracecount, step):
                    if len(coordinates) > max_traces:
                        break

                    try:
                        header = f.header[i]

                        # Extract coordinates
                        x = header[segyio.TraceField.CDP_X]
                        y = header[segyio.TraceField.CDP_Y]
                        if x != 0 or y != 0:
                            coordinates.append((x, y))

                        # Extract positioning info
                        inline = header[segyio.TraceField.INLINE_3D]
                        xline = header[segyio.TraceField.CROSSLINE_3D]
                        cdp = header[segyio.TraceField.CDP]
                        elev = header[segyio.TraceField.ReceiverGroupElevation]

                        if inline != 0:
                            inlines.append(inline)
                        if xline != 0:
                            crosslines.append(xline)
                        if cdp != 0:
                            cdps.append(cdp)
                        if elev != 0:
                            elevations.append(elev)

                    except Exception as e:
                        logger.debug(f"Error reading header {i}: {e}")
                        continue

                # Analyze collected data
                if coordinates:
                    x_coords = [c[0] for c in coordinates]
                    y_coords = [c[1] for c in coordinates]

                    geometry_info["coordinate_analysis"] = {
                        "coordinate_count": len(coordinates),
                        "x_range": [min(x_coords), max(x_coords)],
                        "y_range": [min(y_coords), max(y_coords)],
                        "x_span": max(x_coords) - min(x_coords),
                        "y_span": max(y_coords) - min(y_coords)
                    }

                    # PCA analysis for survey type
                    if len(coordinates) > 2:
                        coords_array = np.array(coordinates)
                        coords_centered = coords_array - np.mean(coords_array, axis=0)

                        try:
                            cov_matrix = np.cov(coords_centered.T)
                            eigenvalues = np.linalg.eigvals(cov_matrix)
                            eigenvalues = np.sort(eigenvalues)[::-1]

                            total_variance = np.sum(eigenvalues)
                            if total_variance > 0:
                                major_var = eigenvalues[0] / total_variance
                                minor_var = eigenvalues[1] / total_variance if len(eigenvalues) > 1 else 0

                                geometry_info["coordinate_analysis"]["pca"] = {
                                    "major_variance": float(major_var),
                                    "minor_variance": float(minor_var),
                                    "geometry_type": "2D_line" if major_var > 0.999 else "3D_areal"
                                }
                        except Exception as e:
                            logger.debug(f"PCA analysis failed: {e}")

                # Grid analysis
                if inlines and crosslines:
                    geometry_info["grid_analysis"] = {
                        "inline_count": len(set(inlines)),
                        "crossline_count": len(set(crosslines)),
                        "inline_range": [min(inlines), max(inlines)],
                        "crossline_range": [min(crosslines), max(crosslines)]
                    }

                if cdps:
                    geometry_info["cdp_analysis"] = {
                        "cdp_count": len(set(cdps)),
                        "cdp_range": [min(cdps), max(cdps)]
                    }

                if elevations:
                    geometry_info["elevation_analysis"] = {
                        "elevation_range": [min(elevations), max(elevations)],
                        "elevation_variation": max(elevations) - min(elevations)
                    }

                return geometry_info

        except segyio.exceptions.InvalidError:
            return {"error": "Invalid SEG-Y format"}
        except Exception as e:
            return {"error": f"Geometry analysis failed: {str(e)}"}

    def assess_data_quality_segyio(self, file_path: str, survey_type: SurveyType,
                                  amplitude_stats: Dict, geometry_info: Dict) -> Dict[str, Any]:
        """Assess data quality using calibrated thresholds for survey type"""

        # Get appropriate thresholds for survey type
        thresholds = self.quality_thresholds[survey_type]

        quality_checks = {
            "survey_type_detected": survey_type.value,
            "thresholds_applied": thresholds['description'],
            "checks": [],
            "rating_factors": {},
            "overall_rating": QualityRating.GOOD.value,
            "confidence": "medium"
        }

        try:
            # 1. Dynamic Range Assessment
            dynamic_range = amplitude_stats.get("dynamic_range_db", 0)
            if dynamic_range >= thresholds['min_dynamic_range_db'] + 10:
                quality_checks["checks"].append({
                    "category": "amplitude",
                    "severity": "info",
                    "message": f"Excellent dynamic range: {dynamic_range:.1f}dB (>{thresholds['min_dynamic_range_db']}dB expected for {thresholds['description']})"
                })
                quality_checks["rating_factors"]["dynamic_range"] = "excellent"
            elif dynamic_range >= thresholds['min_dynamic_range_db']:
                quality_checks["checks"].append({
                    "category": "amplitude",
                    "severity": "info",
                    "message": f"Good dynamic range: {dynamic_range:.1f}dB (meets {thresholds['min_dynamic_range_db']}dB threshold for {thresholds['description']})"
                })
                quality_checks["rating_factors"]["dynamic_range"] = "good"
            elif dynamic_range >= thresholds['min_dynamic_range_db'] - 5:
                quality_checks["checks"].append({
                    "category": "amplitude",
                    "severity": "warning",
                    "message": f"Marginal dynamic range: {dynamic_range:.1f}dB (below {thresholds['min_dynamic_range_db']}dB expected for {thresholds['description']})"
                })
                quality_checks["rating_factors"]["dynamic_range"] = "marginal"
            else:
                quality_checks["checks"].append({
                    "category": "amplitude",
                    "severity": "error",
                    "message": f"Poor dynamic range: {dynamic_range:.1f}dB (well below {thresholds['min_dynamic_range_db']}dB expected for {thresholds['description']})"
                })
                quality_checks["rating_factors"]["dynamic_range"] = "poor"

            # 2. Zero Percentage Assessment (calibrated for survey type)
            zero_percent = amplitude_stats.get("zero_percentage", 0)
            if zero_percent <= thresholds['max_zero_percent'] / 2:
                quality_checks["checks"].append({
                    "category": "amplitude",
                    "severity": "info",
                    "message": f"Low zero percentage: {zero_percent:.1f}% (good for {thresholds['description']})"
                })
                quality_checks["rating_factors"]["zero_percentage"] = "excellent"
            elif zero_percent <= thresholds['max_zero_percent']:
                if survey_type in [SurveyType.SHOT_GATHER, SurveyType.CDP_STACK]:
                    quality_checks["checks"].append({
                        "category": "amplitude",
                        "severity": "info",
                        "message": f"Zero percentage: {zero_percent:.1f}% (normal for {thresholds['description']} due to muting)"
                    })
                    quality_checks["rating_factors"]["zero_percentage"] = "normal"
                else:
                    quality_checks["checks"].append({
                        "category": "amplitude",
                        "severity": "warning",
                        "message": f"Moderate zero percentage: {zero_percent:.1f}% (acceptable for {thresholds['description']})"
                    })
                    quality_checks["rating_factors"]["zero_percentage"] = "acceptable"
            else:
                quality_checks["checks"].append({
                    "category": "amplitude",
                    "severity": "warning" if survey_type == SurveyType.SHOT_GATHER else "error",
                    "message": f"High zero percentage: {zero_percent:.1f}% (above {thresholds['max_zero_percent']}% expected for {thresholds['description']})"
                })
                quality_checks["rating_factors"]["zero_percentage"] = "high"

            # 3. Amplitude Range Assessment
            amp_range = abs(amplitude_stats.get("max", 0)) + abs(amplitude_stats.get("min", 0))
            if amp_range > thresholds['max_amplitude_range']:
                quality_checks["checks"].append({
                    "category": "amplitude",
                    "severity": "warning",
                    "message": f"Large amplitude range: {amp_range:.2e} (may indicate scaling issues for {thresholds['description']})"
                })
                quality_checks["rating_factors"]["amplitude_range"] = "large"
            elif amp_range < 1.0:
                quality_checks["checks"].append({
                    "category": "amplitude",
                    "severity": "warning",
                    "message": f"Very small amplitude range: {amp_range:.3f} (may indicate over-normalization)"
                })
                quality_checks["rating_factors"]["amplitude_range"] = "small"
            else:
                quality_checks["checks"].append({
                    "category": "amplitude",
                    "severity": "info",
                    "message": f"Reasonable amplitude range: {amp_range:.3f}"
                })
                quality_checks["rating_factors"]["amplitude_range"] = "reasonable"

            # 4. Data Integrity Checks
            data_quality = amplitude_stats.get("data_quality", {})

            nan_count = data_quality.get("nan_count", 0)
            inf_count = data_quality.get("inf_count", 0)

            if nan_count > 0 or inf_count > 0:
                quality_checks["checks"].append({
                    "category": "data_integrity",
                    "severity": "error",
                    "message": f"Data corruption detected: {nan_count} NaN values, {inf_count} infinite values"
                })
                quality_checks["rating_factors"]["data_integrity"] = "corrupted"
            else:
                quality_checks["checks"].append({
                    "category": "data_integrity",
                    "severity": "info",
                    "message": "No data corruption detected (no NaN or infinite values)"
                })
                quality_checks["rating_factors"]["data_integrity"] = "clean"

            # 5. Trace Count Assessment
            total_traces = geometry_info.get("total_traces", 0)
            if total_traces >= thresholds['min_traces'] * 10:
                quality_checks["checks"].append({
                    "category": "geometry",
                    "severity": "info",
                    "message": f"Large dataset: {total_traces} traces (excellent for {thresholds['description']})"
                })
                quality_checks["rating_factors"]["trace_count"] = "large"
            elif total_traces >= thresholds['min_traces']:
                quality_checks["checks"].append({
                    "category": "geometry",
                    "severity": "info",
                    "message": f"Adequate trace count: {total_traces} traces (sufficient for {thresholds['description']})"
                })
                quality_checks["rating_factors"]["trace_count"] = "adequate"
            else:
                quality_checks["checks"].append({
                    "category": "geometry",
                    "severity": "warning",
                    "message": f"Low trace count: {total_traces} traces (below {thresholds['min_traces']} recommended for {thresholds['description']})"
                })
                quality_checks["rating_factors"]["trace_count"] = "low"

            # 6. Calculate Overall Rating
            rating_factors = quality_checks["rating_factors"]

            # Count positive and negative factors
            excellent_factors = sum(1 for v in rating_factors.values() if v in ["excellent", "large"])
            good_factors = sum(1 for v in rating_factors.values() if v in ["good", "reasonable", "adequate", "clean", "normal"])
            poor_factors = sum(1 for v in rating_factors.values() if v in ["poor", "corrupted", "high"])

            # Determine overall rating
            if poor_factors > 0:
                if rating_factors.get("data_integrity") == "corrupted":
                    quality_checks["overall_rating"] = QualityRating.INVALID.value
                else:
                    quality_checks["overall_rating"] = QualityRating.POOR.value
            elif excellent_factors >= 2:
                quality_checks["overall_rating"] = QualityRating.EXCELLENT.value
            elif good_factors >= len(rating_factors) * 0.7:
                quality_checks["overall_rating"] = QualityRating.GOOD.value
            else:
                quality_checks["overall_rating"] = QualityRating.FAIR.value

            # Calculate confidence based on data coverage
            traces_analyzed = amplitude_stats.get("traces_sampled", 0)
            total_traces = geometry_info.get("total_traces", 1)
            coverage = traces_analyzed / total_traces if total_traces > 0 else 0

            if coverage > 0.1:
                quality_checks["confidence"] = "high"
            elif coverage > 0.01:
                quality_checks["confidence"] = "medium"
            else:
                quality_checks["confidence"] = "low"

            quality_checks["analysis_coverage"] = {
                "traces_analyzed": traces_analyzed,
                "total_traces": total_traces,
                "coverage_percentage": round(coverage * 100, 2)
            }

            return quality_checks

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                "survey_type_detected": survey_type.value,
                "error": f"Quality assessment failed: {str(e)}",
                "overall_rating": QualityRating.INVALID.value,
                "confidence": "none"
            }

def production_segy_analysis(file_path=None, template_path=None, analysis_type="full",
                            data_dir="./data", template_dir="./templates", **kwargs):
    """
    Production-quality SEG-Y analysis tool using segyio with comprehensive validation
    """
    operation_start = time.time()
    analyzer = SegyioQualityAnalyzer()

    try:
        # Handle JSON input with validation
        if 'input' in kwargs and kwargs['input'] is not None:
            try:
                if isinstance(kwargs['input'], str) and kwargs['input'].startswith('{'):
                    input_data = json.loads(kwargs['input'])
                    if isinstance(input_data, dict):
                        file_path = input_data.get('file_path', file_path)
                        template_path = input_data.get('template_path', template_path)
                        analysis_type = input_data.get('analysis_type', analysis_type)
                        for key, value in input_data.items():
                            if key not in ['file_path', 'template_path', 'analysis_type']:
                                kwargs[key] = value
                else:
                    file_path = kwargs['input']
            except json.JSONDecodeError as e:
                return {"text": json.dumps({
                    "error": f"Invalid JSON input: {str(e)}"
                })}

        # Validate inputs
        if file_path is None:
            return {"text": json.dumps({
                "error": "No file path provided"
            })}

        # Validate analysis type
        valid_analysis_types = ["full", "geometry", "amplitudes", "headers", "quick"]
        if analysis_type not in valid_analysis_types:
            return {"text": json.dumps({
                "error": f"Invalid analysis_type: {analysis_type}",
                "valid_options": valid_analysis_types
            })}

        # Find files
        full_file_path = find_segy_file(file_path, data_dir)
        if not os.path.isfile(full_file_path):
            return {"text": json.dumps({
                "error": f"SEG-Y file not found: {file_path}",
                "searched_path": full_file_path
            })}

        # Extract parameters
        max_analysis_traces = int(kwargs.get('max_analysis_traces', 1000))

        # Set up progress reporting
        file_size_mb = os.path.getsize(full_file_path) / 1024**2
        operations = 3 if analysis_type == "quick" else 6 if analysis_type == "full" else 4
        progress = ProgressReporter(operations, f"Analyzing {os.path.basename(full_file_path)}")

        logger.info(f"Starting segyio-based SEG-Y analysis: {full_file_path} (type: {analysis_type})")

        # Initialize result
        result = {
            "file_processed": os.path.basename(full_file_path),
            "analysis_type": analysis_type,
            "analysis_engine": "segyio-based",
            "file_size_mb": round(file_size_mb, 2),
            "processing_time_seconds": 0,
            "memory_usage_mb": round(analyzer.memory_monitor.get_memory_usage_mb(), 1)
        }

        # 1. Detect survey type for appropriate analysis
        progress.update(1, "Detecting survey type...")
        survey_type = analyzer.detect_survey_type_for_qc(full_file_path)
        result["survey_type_detected"] = survey_type.value

        # 2. Basic file information using segyio
        progress.update(1, "Extracting file information...")
        try:
            with segyio.open(full_file_path, ignore_geometry=True) as f:
                bin_header = f.bin

                result["file_info"] = {
                    "total_traces": f.tracecount,
                    "samples_per_trace": bin_header[segyio.BinField.Samples],
                    "sample_rate_ms": bin_header[segyio.BinField.Interval] / 1000,
                    "trace_length_ms": (bin_header[segyio.BinField.Interval] * bin_header[segyio.BinField.Samples]) / 1000,
                    "format_code": bin_header[segyio.BinField.Format],
                    "format_description": _get_format_description(bin_header[segyio.BinField.Format]),
                    "file_revision": "SEG-Y Rev 1"  # segyio standard
                }
        except Exception as e:
            result["file_info_error"] = str(e)

        # 3. Geometry analysis
        if analysis_type in ["full", "geometry", "quick"]:
            progress.update(1, "Analyzing survey geometry...")
            try:
                geometry_analysis = analyzer.analyze_survey_geometry_segyio(full_file_path, max_analysis_traces)
                result["geometry_analysis"] = geometry_analysis
            except Exception as e:
                result["geometry_analysis_error"] = str(e)

        # 4. Amplitude analysis
        if analysis_type in ["full", "amplitudes"]:
            progress.update(1, "Analyzing amplitude statistics...")
            try:
                amplitude_analysis = analyzer.analyze_amplitude_distribution_segyio(full_file_path, max_analysis_traces)
                result["amplitude_analysis"] = amplitude_analysis
            except Exception as e:
                result["amplitude_analysis_error"] = str(e)

        # 5. Data quality assessment with calibrated thresholds
        if analysis_type in ["full", "quick"]:
            progress.update(1, "Assessing data quality...")
            try:
                # Use amplitude and geometry data for quality assessment
                amplitude_data = result.get("amplitude_analysis", {})
                geometry_data = result.get("geometry_analysis", {})

                quality_assessment = analyzer.assess_data_quality_segyio(
                    full_file_path, survey_type, amplitude_data, geometry_data
                )
                result["quality_assessment"] = quality_assessment

            except Exception as e:
                result["quality_assessment_error"] = str(e)

        # Performance metrics
        processing_time = time.time() - operation_start
        result.update({
            "processing_time_seconds": round(processing_time, 2),
            "final_memory_usage_mb": round(analyzer.memory_monitor.get_memory_usage_mb(), 1),
            "analysis_efficiency": {
                "traces_per_second": round(result.get("file_info", {}).get("total_traces", 0) / processing_time, 0) if processing_time > 0 else 0,
                "mb_per_second": round(file_size_mb / processing_time, 1) if processing_time > 0 else 0
            }
        })

        progress.finish()
        logger.info(f"segyio-based SEG-Y analysis completed in {processing_time:.1f}s")

        return {"text": json.dumps(result, cls=NumpyJSONEncoder)}

    except Exception as e:
        processing_time = time.time() - operation_start
        error_details = traceback.format_exc()
        logger.error(f"SEG-Y analysis failed: {str(e)}")

        return {"text": json.dumps({
            "error": f"Production SEG-Y analysis error: {str(e)}",
            "processing_time_seconds": round(processing_time, 2),
            "details": error_details
        })}

def production_segy_qc(file_path=None, template_path=None, data_dir="./data",
                      template_dir="./templates", **kwargs):
    """
    Production-quality SEG-Y quality control using segyio with calibrated thresholds
    """
    operation_start = time.time()
    analyzer = SegyioQualityAnalyzer()

    try:
        # Handle JSON input
        if 'input' in kwargs and kwargs['input'] is not None:
            try:
                if isinstance(kwargs['input'], str) and kwargs['input'].startswith('{'):
                    input_data = json.loads(kwargs['input'])
                    if isinstance(input_data, dict):
                        file_path = input_data.get('file_path', file_path)
                        template_path = input_data.get('template_path', template_path)
                        for key, value in input_data.items():
                            if key not in ['file_path', 'template_path']:
                                kwargs[key] = value
                else:
                    file_path = kwargs['input']
            except json.JSONDecodeError as e:
                return {"text": json.dumps({
                    "error": f"Invalid JSON input: {str(e)}"
                })}

        if file_path is None:
            return {"text": json.dumps({
                "error": "No file path provided"
            })}

        # Find files
        full_file_path = find_segy_file(file_path, data_dir)
        if not os.path.isfile(full_file_path):
            return {"text": json.dumps({
                "error": f"SEG-Y file not found: {file_path}"
            })}

        # Set up progress reporting
        file_size_mb = os.path.getsize(full_file_path) / 1024**2
        progress = ProgressReporter(5, f"QC {os.path.basename(full_file_path)}")

        logger.info(f"Starting segyio-based SEG-Y QC: {full_file_path}")

        # Initialize QC results
        qc_results = {
            "file_info": {
                "filename": os.path.basename(full_file_path),
                "file_size_mb": round(file_size_mb, 2),
                "file_accessible": True,
                "qc_engine": "segyio-based",
                "total_traces": 0
            },
            "validation_results": {},
            "format_checks": [],
            "geometry_checks": [],
            "data_checks": [],
            "template_checks": [],
            "overall_assessment": {}
        }

        # 1. File structure validation using segyio
        progress.update(1, "Validating file structure...")
        try:
            with segyio.open(full_file_path, ignore_geometry=True) as f:
                bin_header = f.bin

                qc_results["validation_results"]["file_structure"] = {
                    "valid_segy_format": True,
                    "total_traces": f.tracecount,
                    "samples_per_trace": bin_header[segyio.BinField.Samples],
                    "sample_interval": bin_header[segyio.BinField.Interval],
                    "format_code": bin_header[segyio.BinField.Format]
                }

                # Basic validation checks
                if f.tracecount <= 0:
                    qc_results["format_checks"].append({
                        "severity": "error",
                        "category": "file_structure",
                        "message": "Invalid trace count: file appears to have no traces"
                    })
                elif f.tracecount < 10:
                    qc_results["format_checks"].append({
                        "severity": "warning",
                        "category": "file_structure",
                        "message": f"Very low trace count: {f.tracecount} traces"
                    })
                else:
                    qc_results["format_checks"].append({
                        "severity": "info",
                        "category": "file_structure",
                        "message": f"Valid trace count: {f.tracecount} traces"
                    })

                # Sample rate validation
                sample_rate_ms = bin_header[segyio.BinField.Interval] / 1000
                if sample_rate_ms < 0.5 or sample_rate_ms > 20:
                    qc_results["format_checks"].append({
                        "severity": "warning",
                        "category": "file_structure",
                        "message": f"Unusual sample rate: {sample_rate_ms}ms"
                    })
                else:
                    qc_results["format_checks"].append({
                        "severity": "info",
                        "category": "file_structure",
                        "message": f"Normal sample rate: {sample_rate_ms}ms"
                    })

                # Format code validation
                format_code = bin_header[segyio.BinField.Format]
                valid_formats = [1, 2, 3, 5, 8]
                if format_code not in valid_formats:
                    qc_results["format_checks"].append({
                        "severity": "warning",
                        "category": "file_structure",
                        "message": f"Non-standard format code: {format_code}"
                    })
                else:
                    qc_results["format_checks"].append({
                        "severity": "info",
                        "category": "file_structure",
                        "message": f"Standard format code: {format_code} ({_get_format_description(format_code)})"
                    })

        except segyio.exceptions.InvalidError:
            qc_results["format_checks"].append({
                "severity": "error",
                "category": "file_structure",
                "message": "Invalid SEG-Y format - file structure corrupted"
            })
        except Exception as e:
            qc_results["format_checks"].append({
                "severity": "error",
                "category": "file_structure",
                "message": f"File structure validation failed: {str(e)}"
            })

        # 2. Survey type detection and geometry validation
        progress.update(1, "Analyzing survey geometry...")
        try:
            survey_type = analyzer.detect_survey_type_for_qc(full_file_path)
            qc_results["validation_results"]["survey_type"] = survey_type.value

            geometry_analysis = analyzer.analyze_survey_geometry_segyio(full_file_path, 1000)
            qc_results["validation_results"]["geometry"] = geometry_analysis

            # Geometry-specific checks
            if "error" in geometry_analysis:
                qc_results["geometry_checks"].append({
                    "severity": "error",
                    "category": "geometry",
                    "message": geometry_analysis["error"]
                })
            else:
                # Check coordinate availability
                if geometry_analysis.get("coordinate_analysis"):
                    coord_analysis = geometry_analysis["coordinate_analysis"]
                    if coord_analysis["coordinate_count"] > 0:
                        qc_results["geometry_checks"].append({
                            "severity": "info",
                            "category": "geometry",
                            "message": f"Coordinates found: {coord_analysis['coordinate_count']} valid coordinate pairs"
                        })

                        # Check coordinate ranges
                        x_span = coord_analysis.get("x_span", 0)
                        y_span = coord_analysis.get("y_span", 0)

                        if x_span > 1000000 or y_span > 1000000:
                            qc_results["geometry_checks"].append({
                                "severity": "info",
                                "category": "geometry",
                                "message": f"Large coordinate system detected (UTM-like): X span={x_span:.0f}m, Y span={y_span:.0f}m"
                            })
                        elif x_span < 1000 and y_span < 1000:
                            qc_results["geometry_checks"].append({
                                "severity": "info",
                                "category": "geometry",
                                "message": f"Local coordinate system detected: X span={x_span:.0f}m, Y span={y_span:.0f}m"
                            })
                    else:
                        qc_results["geometry_checks"].append({
                            "severity": "warning",
                            "category": "geometry",
                            "message": "No valid coordinates found in headers"
                        })

                # Grid analysis for 3D surveys
                if geometry_analysis.get("grid_analysis"):
                    grid = geometry_analysis["grid_analysis"]
                    qc_results["geometry_checks"].append({
                        "severity": "info",
                        "category": "geometry",
                        "message": f"3D grid detected: {grid['inline_count']} inlines, {grid['crossline_count']} crosslines"
                    })

        except Exception as e:
            qc_results["geometry_checks"].append({
                "severity": "warning",
                "category": "geometry",
                "message": f"Geometry analysis failed: {str(e)}"
            })

        # 3. Amplitude analysis and data quality
        progress.update(1, "Analyzing amplitude data...")
        try:
            amplitude_analysis = analyzer.analyze_amplitude_distribution_segyio(full_file_path, 100)
            qc_results["validation_results"]["amplitude_analysis"] = amplitude_analysis

            if "error" in amplitude_analysis:
                qc_results["data_checks"].append({
                    "severity": "error",
                    "category": "amplitudes",
                    "message": amplitude_analysis["error"]
                })
            else:
                # Data integrity checks
                data_quality = amplitude_analysis.get("data_quality", {})

                if data_quality.get("nan_count", 0) > 0 or data_quality.get("inf_count", 0) > 0:
                    qc_results["data_checks"].append({
                        "severity": "error",
                        "category": "amplitudes",
                        "message": f"Data corruption: {data_quality.get('nan_count', 0)} NaN, {data_quality.get('inf_count', 0)} infinite values"
                    })
                else:
                    qc_results["data_checks"].append({
                        "severity": "info",
                        "category": "amplitudes",
                        "message": "No data corruption detected (no NaN or infinite values)"
                    })

                # Amplitude range assessment
                amp_min = amplitude_analysis.get("min", 0)
                amp_max = amplitude_analysis.get("max", 0)
                qc_results["data_checks"].append({
                    "severity": "info",
                    "category": "amplitudes",
                    "message": f"Amplitude range: {amp_min:.3f} to {amp_max:.3f}"
                })

                # Dynamic range assessment
                dynamic_range = amplitude_analysis.get("dynamic_range_db", 0)
                if dynamic_range > 40:
                    qc_results["data_checks"].append({
                        "severity": "info",
                        "category": "amplitudes",
                        "message": f"Excellent dynamic range: {dynamic_range:.1f}dB"
                    })
                elif dynamic_range > 25:
                    qc_results["data_checks"].append({
                        "severity": "info",
                        "category": "amplitudes",
                        "message": f"Good dynamic range: {dynamic_range:.1f}dB"
                    })
                elif dynamic_range > 15:
                    qc_results["data_checks"].append({
                        "severity": "warning",
                        "category": "amplitudes",
                        "message": f"Moderate dynamic range: {dynamic_range:.1f}dB"
                    })
                else:
                    qc_results["data_checks"].append({
                        "severity": "warning",
                        "category": "amplitudes",
                        "message": f"Low dynamic range: {dynamic_range:.1f}dB"
                    })

        except Exception as e:
            qc_results["data_checks"].append({
                "severity": "warning",
                "category": "amplitudes",
                "message": f"Amplitude analysis failed: {str(e)}"
            })

        # 4. Overall quality assessment using calibrated thresholds
        progress.update(1, "Performing quality assessment...")
        try:
            # Get survey type and data for quality assessment
            survey_type = SurveyType(qc_results["validation_results"].get("survey_type", "unknown"))
            amplitude_data = qc_results["validation_results"].get("amplitude_analysis", {})
            geometry_data = qc_results["validation_results"].get("geometry", {})

            quality_assessment = analyzer.assess_data_quality_segyio(
                full_file_path, survey_type, amplitude_data, geometry_data
            )
            qc_results["validation_results"]["quality_assessment"] = quality_assessment

            # Extract overall rating
            overall_rating = quality_assessment.get("overall_rating", QualityRating.POOR.value)
            confidence = quality_assessment.get("confidence", "low")

        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            overall_rating = QualityRating.FAIR.value
            confidence = "low"

        # 5. Template checks (simplified for segyio)
        progress.update(1, "Finalizing assessment...")
        qc_results["template_checks"].append({
            "severity": "info",
            "category": "template",
            "message": "Using segyio native header reading (no template files required)"
        })

        # Overall assessment summary
        all_checks = (qc_results["format_checks"] + qc_results["geometry_checks"] +
                     qc_results["data_checks"] + qc_results["template_checks"])

        error_count = sum(1 for check in all_checks if check["severity"] == "error")
        warning_count = sum(1 for check in all_checks if check["severity"] == "warning")
        info_count = sum(1 for check in all_checks if check["severity"] == "info")

        # Generate recommendations based on calibrated assessment
        recommendations = []
        if overall_rating == QualityRating.EXCELLENT.value:
            recommendations.append("Excellent quality - ready for processing")
        elif overall_rating == QualityRating.GOOD.value:
            recommendations.append("Good quality - proceed with processing")
        elif overall_rating == QualityRating.FAIR.value:
            recommendations.append("Fair quality - review warnings before processing")
        elif overall_rating == QualityRating.POOR.value:
            recommendations.append("Poor quality - address issues before processing")
        else:
            recommendations.append("Invalid data - file cannot be processed")

        if error_count > 0:
            recommendations.append("Fix critical errors before processing")
        if warning_count > 2:
            recommendations.append("Review multiple warnings")

        # Processing time and performance
        processing_time = time.time() - operation_start

        qc_results["overall_assessment"] = {
            "quality_rating": overall_rating,
            "confidence": confidence,
            "recommendation": recommendations[0] if recommendations else "Manual review required",
            "additional_recommendations": recommendations[1:] if len(recommendations) > 1 else [],
            "total_checks": len(all_checks),
            "errors": error_count,
            "warnings": warning_count,
            "info_messages": info_count,
            "processing_time_seconds": round(processing_time, 2),
            "memory_usage_mb": round(analyzer.memory_monitor.get_memory_usage_mb(), 1),
            "qc_engine": "segyio-based with calibrated thresholds"
        }

        progress.finish()
        logger.info(f"segyio-based SEG-Y QC completed in {processing_time:.1f}s - Rating: {overall_rating}")

        return {"text": json.dumps(qc_results, cls=NumpyJSONEncoder)}

    except Exception as e:
        processing_time = time.time() - operation_start
        error_details = traceback.format_exc()
        logger.error(f"SEG-Y QC failed: {str(e)}")

        return {"text": json.dumps({
            "error": f"Production SEG-Y QC error: {str(e)}",
            "processing_time_seconds": round(processing_time, 2),
            "details": error_details
        })}

def _get_format_description(format_code: int) -> str:
    """Get format code description"""
    format_descriptions = {
        1: "32-bit IBM floating point",
        2: "32-bit two's complement integer",
        3: "16-bit two's complement integer",
        5: "32-bit IEEE floating point",
        8: "8-bit two's complement integer"
    }
    return format_descriptions.get(format_code, f"Format code {format_code}")

# Backward compatibility function
def FC_to_text(format_code: int) -> str:
    """Backward compatibility function for format code conversion"""
    return _get_format_description(format_code)