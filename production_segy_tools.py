"""
production_segy_tools.py - Production-quality SEG-Y analysis tools

This module provides robust, production-ready SEG-Y file processing capabilities
with comprehensive error handling, validation, and progress reporting.

UPDATED: Now uses segyio as the core engine for maximum reliability and accuracy.
"""

import os
import sys
import json
import traceback
import math
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
import gc
import psutil
import segyio
import numpy as np
from enum import Enum
from datetime import datetime
import re
import hashlib
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums for better type safety
class SurveyType(Enum):
    SHOT_GATHER = "shot_gather"
    CDP_STACK = "cdp_stack"
    MIGRATED_2D = "migrated_2d"
    MIGRATED_3D = "migrated_3d"
    UNKNOWN = "unknown"

class QualityRating(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class SEGYValidationError(Exception):
    """Custom exception for SEG-Y validation errors"""
    pass

class SEGYProcessingError(Exception):
    """Custom exception for SEG-Y processing errors"""
    pass

class ProgressReporter:
    """Progress reporting utility for long-running operations"""

    def __init__(self, total_operations: int, operation_name: str = "Processing"):
        self.total_operations = total_operations
        self.current_operation = 0
        self.operation_name = operation_name
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.report_interval = 5.0  # Report every 5 seconds

    def update(self, increment: int = 1, message: str = None):
        """Update progress and optionally report"""
        self.current_operation += increment
        current_time = time.time()

        if (current_time - self.last_report_time >= self.report_interval or
            self.current_operation >= self.total_operations):

            percentage = (self.current_operation / self.total_operations) * 100
            elapsed = current_time - self.start_time

            if self.current_operation > 0:
                eta = (elapsed / self.current_operation) * (self.total_operations - self.current_operation)
                eta_str = f", ETA: {eta:.1f}s"
            else:
                eta_str = ""

            status_msg = f"{self.operation_name}: {percentage:.1f}% ({self.current_operation}/{self.total_operations}){eta_str}"
            if message:
                status_msg += f" - {message}"

            logger.info(status_msg)
            self.last_report_time = current_time

    def finish(self):
        """Mark operation as complete"""
        elapsed = time.time() - self.start_time
        logger.info(f"{self.operation_name} completed in {elapsed:.1f}s")

class MemoryMonitor:
    """Memory usage monitoring and management"""

    def __init__(self, max_memory_gb: float = 4.0):
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.initial_memory = self.get_current_memory()

    def get_current_memory(self) -> int:
        """Get current memory usage in bytes"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except:
            return 0

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.get_current_memory() / 1024**2

    def check_memory_limit(self) -> bool:
        """Check if memory usage is approaching limit"""
        current_memory = self.get_current_memory()
        return current_memory < self.max_memory_bytes

    def suggest_gc(self) -> bool:
        """Suggest garbage collection if memory usage is high"""
        current_memory = self.get_current_memory()
        if current_memory > (self.max_memory_bytes * 0.8):
            gc.collect()
            return True
        return False

    def get_available_memory_gb(self) -> float:
        """Get available system memory in GB"""
        try:
            return psutil.virtual_memory().available / 1024**3
        except:
            return 4.0  # Conservative default

class SegyioValidator:
    """Comprehensive SEG-Y file validation using segyio"""

    # Standard SEG-Y format codes
    VALID_FORMAT_CODES = {
        1: "32-bit IBM floating point",
        2: "32-bit two's complement integer",
        3: "16-bit two's complement integer",
        5: "32-bit IEEE floating point",
        8: "8-bit two's complement integer"
    }

    def __init__(self):
        self.validation_results = {}

    def validate_file_structure(self, file_path: str) -> Dict[str, Any]:
        """Validate basic SEG-Y file structure using segyio"""
        results = {
            "file_accessible": False,
            "file_size_reasonable": False,
            "has_segy_headers": False,
            "binary_header_valid": False,
            "issues": [],
            "warnings": []
        }

        try:
            # Check file accessibility
            if not os.path.isfile(file_path):
                results["issues"].append(f"File not found: {file_path}")
                return results

            results["file_accessible"] = True

            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size < 3600:  # Minimum for headers
                results["issues"].append(f"File too small ({file_size} bytes) - missing headers")
                return results
            elif file_size > 50 * 1024**3:  # 50GB warning
                results["warnings"].append(f"Very large file ({file_size/1024**3:.1f}GB) - processing may be slow")

            results["file_size_reasonable"] = True

            # Use segyio to validate SEG-Y structure
            try:
                with segyio.open(file_path, ignore_geometry=True) as f:
                    results["has_segy_headers"] = True
                    results["binary_header_valid"] = True

                    # Extract header information
                    bin_header = f.bin
                    sample_rate = bin_header[segyio.BinField.Interval]
                    num_samples = bin_header[segyio.BinField.Samples]
                    format_code = bin_header[segyio.BinField.Format]

                    # Validate sample rate (stored as microseconds)
                    if sample_rate <= 0 or sample_rate > 100000:
                        results["issues"].append(f"Invalid sample rate: {sample_rate} microseconds")
                    elif sample_rate < 1000 or sample_rate > 10000:
                        results["warnings"].append(f"Unusual sample rate: {sample_rate/1000:.1f}ms")

                    # Validate number of samples
                    if num_samples <= 0 or num_samples > 50000:
                        results["issues"].append(f"Invalid number of samples: {num_samples}")
                    elif num_samples > 10000:
                        results["warnings"].append(f"High sample count: {num_samples}")

                    # Validate format code
                    if format_code not in self.VALID_FORMAT_CODES:
                        results["warnings"].append(f"Non-standard format code: {format_code}")
                    else:
                        results["format_description"] = self.VALID_FORMAT_CODES[format_code]

                    results["header_info"] = {
                        "sample_rate_us": sample_rate,
                        "sample_rate_ms": sample_rate / 1000,
                        "num_samples": num_samples,
                        "format_code": format_code,
                        "trace_length_ms": (sample_rate * num_samples) / 1000,
                        "trace_count": f.tracecount
                    }

            except segyio.exceptions.InvalidError:
                results["issues"].append("Invalid SEG-Y format - file structure corrupted")
            except Exception as e:
                results["issues"].append(f"Error reading SEG-Y headers: {str(e)}")

        except Exception as e:
            results["issues"].append(f"File validation error: {str(e)}")

        return results

class SegyioSurveyClassifier:
    """Intelligent survey type classification using segyio"""

    def __init__(self):
        self.classification_cache = {}

    def classify_survey_type(self, file_path: str) -> Tuple[SurveyType, Dict[str, Any]]:
        """Classify survey type using segyio header analysis"""

        if file_path in self.classification_cache:
            return self.classification_cache[file_path]

        classification_info = {
            "confidence": "low",
            "reasoning": "",
            "geometry_stats": {},
            "issues": []
        }

        try:
            with segyio.open(file_path, ignore_geometry=True) as f:
                # Sample headers for analysis (limit to reasonable amount)
                sample_size = min(100, f.tracecount)
                step = max(1, f.tracecount // sample_size)

                cdps = set()
                inlines = set()
                crosslines = set()
                shot_points = set()
                field_records = set()

                for i in range(0, f.tracecount, step):
                    if len(cdps) > 1000:  # Prevent memory issues
                        break

                    header = f.header[i]
                    cdps.add(header[segyio.TraceField.CDP])
                    inlines.add(header[segyio.TraceField.INLINE_3D])
                    crosslines.add(header[segyio.TraceField.CROSSLINE_3D])
                    shot_points.add(header[segyio.TraceField.TRACE_SEQUENCE_FILE])
                    field_records.add(header[segyio.TraceField.FieldRecord])

                # Remove zero values for meaningful analysis
                unique_cdps = len(cdps - {0})
                unique_inlines = len(inlines - {0})
                unique_crosslines = len(crosslines - {0})
                unique_shots = len(shot_points - {0})
                unique_field_records = len(field_records - {0})

                classification_info["geometry_stats"] = {
                    "unique_cdps": unique_cdps,
                    "unique_inlines": unique_inlines,
                    "unique_crosslines": unique_crosslines,
                    "unique_shots": unique_shots,
                    "unique_field_records": unique_field_records,
                    "total_traces": f.tracecount,
                    "traces_sampled": min(sample_size, f.tracecount)
                }

                # Classification logic based on header analysis
                if unique_field_records > unique_cdps and unique_cdps <= 10:
                    survey_type = SurveyType.SHOT_GATHER
                    classification_info["confidence"] = "high" if unique_field_records > 5 else "medium"
                    classification_info["reasoning"] = f"High field record variation ({unique_field_records}) with low CDP variation ({unique_cdps}) suggests shot gather data"

                elif unique_inlines > 5 and unique_crosslines > 5:
                    survey_type = SurveyType.MIGRATED_3D
                    classification_info["confidence"] = "high"
                    classification_info["reasoning"] = f"Grid organization with {unique_inlines} inlines and {unique_crosslines} crosslines indicates 3D migrated volume"

                elif unique_cdps > unique_shots and unique_inlines <= 2:
                    if unique_cdps > 100:
                        survey_type = SurveyType.MIGRATED_2D
                        classification_info["confidence"] = "medium"
                        classification_info["reasoning"] = f"High CDP count ({unique_cdps}) with linear organization suggests 2D migrated line"
                    else:
                        survey_type = SurveyType.CDP_STACK
                        classification_info["confidence"] = "medium"
                        classification_info["reasoning"] = f"Moderate CDP organization ({unique_cdps}) suggests CDP stack"

                elif unique_cdps > 20:
                    survey_type = SurveyType.CDP_STACK
                    classification_info["confidence"] = "low"
                    classification_info["reasoning"] = f"CDP organization detected ({unique_cdps}) but unclear geometry"

                else:
                    survey_type = SurveyType.UNKNOWN
                    classification_info["confidence"] = "low"
                    classification_info["reasoning"] = f"Unclear organization pattern - CDPs:{unique_cdps}, Shots:{unique_shots}, IL:{unique_inlines}, XL:{unique_crosslines}"

                # Cache result
                result = (survey_type, classification_info)
                self.classification_cache[file_path] = result
                return result

        except Exception as e:
            classification_info["issues"].append(f"Classification error: {str(e)}")
            return SurveyType.UNKNOWN, classification_info

class SegyioQualityAnalyzer:
    """Quality analysis using segyio with calibrated thresholds"""

    def __init__(self):
        # Realistic quality thresholds based on survey type
        self.quality_thresholds = {
            SurveyType.SHOT_GATHER: {
                'min_dynamic_range': 20,  # Lower for raw data
                'max_zero_percent': 50,   # Higher for muted zones
                'min_traces': 100,
                'description': 'Raw acquisition data'
            },
            SurveyType.CDP_STACK: {
                'min_dynamic_range': 25,
                'max_zero_percent': 30,
                'min_traces': 500,
                'description': 'Processed CDP stacks'
            },
            SurveyType.MIGRATED_3D: {
                'min_dynamic_range': 30,  # Higher for processed data
                'max_zero_percent': 20,
                'min_traces': 1000,
                'description': 'Migrated 3D volume'
            },
            SurveyType.MIGRATED_2D: {
                'min_dynamic_range': 28,
                'max_zero_percent': 25,
                'min_traces': 500,
                'description': 'Migrated 2D line'
            },
            SurveyType.UNKNOWN: {
                'min_dynamic_range': 20,  # Conservative
                'max_zero_percent': 40,
                'min_traces': 100,
                'description': 'Unknown data type'
            }
        }

    def analyze_quality(self, file_path: str, survey_type: SurveyType) -> Tuple[QualityRating, Dict[str, Any]]:
        """Analyze data quality using segyio"""

        quality_metrics = {
            "survey_type": survey_type.value,
            "thresholds_used": self.quality_thresholds[survey_type]["description"],
            "amplitude_stats": {},
            "signal_metrics": {},
            "issues": [],
            "warnings": []
        }

        try:
            with segyio.open(file_path, ignore_geometry=True) as f:
                # Sample traces for quality analysis (avoid memory issues)
                max_traces_to_sample = min(50, f.tracecount)
                trace_step = max(1, f.tracecount // max_traces_to_sample)

                amplitudes = []
                zero_counts = []

                for i in range(0, f.tracecount, trace_step):
                    if len(amplitudes) > 100000:  # Prevent memory explosion
                        break

                    trace_data = f.trace[i]
                    amplitudes.extend(trace_data)
                    zero_counts.append(np.sum(trace_data == 0))

                amplitudes = np.array(amplitudes)

                # Calculate comprehensive metrics
                quality_metrics["amplitude_stats"] = {
                    'min': float(np.min(amplitudes)),
                    'max': float(np.max(amplitudes)),
                    'mean': float(np.mean(amplitudes)),
                    'std': float(np.std(amplitudes)),
                    'zero_percentage': float(np.mean(zero_counts) / len(f.samples) * 100)
                }

                # Data integrity checks
                nan_count = int(np.sum(np.isnan(amplitudes)))
                inf_count = int(np.sum(np.isinf(amplitudes)))

                quality_metrics["signal_metrics"] = {
                    'nan_count': nan_count,
                    'inf_count': inf_count,
                    'traces_sampled': max_traces_to_sample,
                    'total_traces': f.tracecount,
                    'sample_percentage': round((max_traces_to_sample / f.tracecount) * 100, 1)
                }

                # Dynamic range calculation (more robust)
                max_amplitude = np.max(np.abs(amplitudes))
                noise_estimate = np.std(amplitudes) + 1e-10  # Avoid division by zero
                dynamic_range = 20 * np.log10(max_amplitude / noise_estimate)

                quality_metrics["signal_metrics"]["dynamic_range_db"] = float(dynamic_range)
                quality_metrics["signal_metrics"]["signal_to_noise"] = float(
                    np.mean(np.abs(amplitudes)) / noise_estimate
                )

                # Apply survey-specific quality assessment
                thresholds = self.quality_thresholds[survey_type]
                issues = []
                warnings = []

                # Check data integrity
                if nan_count > 0:
                    issues.append(f"Contains {nan_count} NaN values")
                if inf_count > 0:
                    issues.append(f"Contains {inf_count} infinite values")

                # Apply calibrated thresholds
                if dynamic_range < thresholds['min_dynamic_range']:
                    if survey_type == SurveyType.SHOT_GATHER:
                        # More lenient for shot gathers
                        warnings.append(f"Dynamic range {dynamic_range:.1f}dB below typical {thresholds['min_dynamic_range']}dB for {thresholds['description']}")
                    else:
                        issues.append(f"Low dynamic range: {dynamic_range:.1f}dB (expected >{thresholds['min_dynamic_range']}dB for {thresholds['description']})")

                zero_percentage = quality_metrics["amplitude_stats"]["zero_percentage"]
                if zero_percentage > thresholds['max_zero_percent']:
                    if survey_type in [SurveyType.SHOT_GATHER, SurveyType.CDP_STACK]:
                        # Expected for raw/early processing data
                        warnings.append(f"High zero percentage: {zero_percentage:.1f}% (normal for {thresholds['description']} due to muting)")
                    else:
                        issues.append(f"High zero percentage: {zero_percentage:.1f}% (expected <{thresholds['max_zero_percent']}% for {thresholds['description']})")

                if f.tracecount < thresholds['min_traces']:
                    warnings.append(f"Low trace count: {f.tracecount} (expected >{thresholds['min_traces']} for {thresholds['description']})")

                quality_metrics["issues"] = issues
                quality_metrics["warnings"] = warnings

                # Determine overall quality rating
                critical_issues = len([i for i in issues if "NaN" in i or "infinite" in i])

                if critical_issues > 0:
                    quality_rating = QualityRating.INVALID
                elif len(issues) == 0 and dynamic_range > (thresholds['min_dynamic_range'] + 10):
                    quality_rating = QualityRating.EXCELLENT
                elif len(issues) <= 1 and dynamic_range >= thresholds['min_dynamic_range']:
                    quality_rating = QualityRating.GOOD
                elif len(issues) <= 2 or dynamic_range >= (thresholds['min_dynamic_range'] - 5):
                    quality_rating = QualityRating.FAIR
                else:
                    quality_rating = QualityRating.POOR

                return quality_rating, quality_metrics

        except Exception as e:
            quality_metrics["issues"].append(f"Quality analysis error: {str(e)}")
            return QualityRating.INVALID, quality_metrics

def get_segyio_version():
    """Safely get segyio version with fallback"""
    try:
        import segyio
        # Try different ways to get version
        if hasattr(segyio, '__version__'):
            return segyio_version
        elif hasattr(segyio, 'version'):
            return segyio.version
        elif hasattr(segyio, '__version_info__'):
            return str(segyio.__version_info__)
        else:
            # Fallback: Try to get version from package metadata
            try:
                import pkg_resources
                return pkg_resources.get_distribution('segyio').version
            except:
                return "unknown"
    except Exception:
        return "unknown"

# Use this function instead of segyio.__version__
segyio_version = get_segyio_version()

def create_intelligent_template(file_path: str) -> Dict[str, int]:
    """Create intelligent template using segyio header analysis"""
    try:
        with segyio.open(file_path, ignore_geometry=True) as f:
            # Sample headers to detect field positions
            sample_size = min(20, f.tracecount)
            detected_fields = {}

            for i in range(sample_size):
                header = f.header[i]

                # Check standard positions for non-zero values
                if header[segyio.TraceField.CDP] != 0:
                    detected_fields["CDP"] = 21  # Standard position
                if header[segyio.TraceField.INLINE_3D] != 0:
                    detected_fields["ILINE"] = 189  # Standard position
                if header[segyio.TraceField.CROSSLINE_3D] != 0:
                    detected_fields["XLINE"] = 193  # Standard position
                if header[segyio.TraceField.FieldRecord] != 0:
                    detected_fields["SP"] = 9   # Alternative position
                if header[segyio.TraceField.SourceX] != 0:
                    detected_fields["SX"] = 73  # Standard position
                if header[segyio.TraceField.SourceY] != 0:
                    detected_fields["SY"] = 77  # Standard position

            logger.info(f"Detected fields: {', '.join(detected_fields.keys())}")
            return detected_fields

    except Exception as e:
        logger.warning(f"Template detection failed: {e}, using standard positions")
        return {
            "CDP": 21, "ILINE": 189, "XLINE": 193, "SP": 17,
            "SX": 73, "SY": 77, "XY_SCALAR": 71
        }

def production_segy_parser(file_path=None, template_path=None, data_dir="./data",
                          template_dir="./templates", **kwargs):
    """
    Production-quality SEG-Y parser using segyio with comprehensive validation and error handling
    """
    operation_start = time.time()
    memory_monitor = MemoryMonitor()

    try:
        # Handle JSON input with validation
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
                    "error": f"Invalid JSON input: {str(e)}",
                    "suggestion": "Provide valid JSON or plain file path"
                })}

        # Validate required parameters
        if file_path is None:
            return {"text": json.dumps({
                "error": "No file path provided",
                "usage": "Provide file_path parameter or JSON with file_path field"
            })}

        # Find and validate files
        full_file_path = find_segy_file(file_path, data_dir)
        if not os.path.isfile(full_file_path):
            return {"text": json.dumps({
                "error": f"SEG-Y file not found: {file_path}",
                "searched_path": full_file_path,
                "suggestion": f"Check file exists in {data_dir} directory"
            })}

        # Extract and validate parameters
        sort_key = kwargs.get('sort_key', kwargs.get('primary_sorting', 'INLINE'))
        dimension = kwargs.get('dimension', kwargs.get('survey_type', '3D'))
        stack_type = kwargs.get('stack_type', 'PostStack')
        traces_to_read = int(kwargs.get('traces_to_read', kwargs.get('max_traces', -1)))

        # Validate parameter values
        valid_sort_keys = ['INLINE', 'XLINE', 'SP', 'CDP']
        if sort_key.upper() not in valid_sort_keys:
            return {"text": json.dumps({
                "error": f"Invalid sort_key: {sort_key}",
                "valid_options": valid_sort_keys
            })}

        valid_dimensions = ['2D', '3D']
        if dimension.upper() not in valid_dimensions:
            return {"text": json.dumps({
                "error": f"Invalid dimension: {dimension}",
                "valid_options": valid_dimensions
            })}

        # Create progress reporter
        file_size_mb = os.path.getsize(full_file_path) / 1024**2
        estimated_operations = 5  # Fixed operations for segyio
        progress = ProgressReporter(estimated_operations, f"Parsing {os.path.basename(full_file_path)}")

        logger.info(f"Starting SEG-Y parsing: {full_file_path}")
        logger.info(f"File size: {file_size_mb:.1f} MB, Memory available: {memory_monitor.get_available_memory_gb():.1f} GB")

        progress.update(1, "Validating SEG-Y file structure...")

        # Validate file structure using segyio
        validator = SegyioValidator()
        file_validation = validator.validate_file_structure(full_file_path)

        if file_validation["issues"]:
            return {"text": json.dumps({
                "error": f"File validation failed: {'; '.join(file_validation['issues'])}",
                "file_path": full_file_path,
                "validation_details": file_validation
            })}

        progress.update(1, "Creating intelligent template...")

        # Create intelligent template using segyio
        intelligent_template = create_intelligent_template(full_file_path)
        logger.info("✓ Intelligent template created successfully")

        # Basic file analysis using segyio
        progress.update(1, "Creating seismic file object...")

        with segyio.open(full_file_path, ignore_geometry=True) as f:
            progress.update(1, "Extracting basic file information...")

            # Extract basic information using segyio
            bin_header = f.bin

            result = {
                "file_processed": os.path.basename(full_file_path),
                "file_path": full_file_path,
                "file_size_mb": round(file_size_mb, 2),
                "template_used": "intelligent_segyio_detection",
                "parsing_method": "segyio_native",
                "survey_type": dimension.upper(),
                "primary_sorting": sort_key.upper(),
                "stack_type": stack_type,
                "processing_time_seconds": 0,  # Will be updated at end
                "memory_usage_mb": round(memory_monitor.get_memory_usage_mb(), 1)
            }

            # Add file format information using segyio
            try:
                sample_interval = bin_header[segyio.BinField.Interval]
                samples_per_trace = bin_header[segyio.BinField.Samples]
                format_code = bin_header[segyio.BinField.Format]

                # Format code descriptions
                format_descriptions = {
                    1: "32 BIT IBM FORMAT",
                    2: "32 BIT INTEGER",
                    3: "16 BIT INTEGER",
                    5: "32 BIT IEEE FORMAT",
                    8: "8 BIT INTEGER"
                }

                result.update({
                    "file_revision": 1,  # Standard assumption
                    "number_of_samples": samples_per_trace,
                    "sample_rate_ms": sample_interval / 1000,
                    "trace_length_ms": (sample_interval * samples_per_trace) / 1000,
                    "sample_format": format_descriptions.get(format_code, f"FORMAT CODE {format_code}"),
                    "bytes_per_trace": 240 + (samples_per_trace * (4 if format_code in [1,2,5] else 2)),
                    "total_traces": f.tracecount,
                    "xy_scalar": 1  # Default, would need trace analysis for actual value
                })

            except Exception as e:
                logger.warning(f"Error extracting format information: {str(e)}")
                result["format_warning"] = str(e)

            progress.update(1, f"Analyzing geometry ({f.tracecount} traces)...")

            # Survey classification using segyio
            classifier = SegyioSurveyClassifier()
            detected_survey_type, classification_info = classifier.classify_survey_type(full_file_path)

            result["detected_survey_type"] = detected_survey_type.value
            result["classification_confidence"] = classification_info["confidence"]
            result["classification_reasoning"] = classification_info["reasoning"]
            result["geometry_stats"] = classification_info["geometry_stats"]

            # Quality analysis using segyio
            quality_analyzer = SegyioQualityAnalyzer()
            quality_rating, quality_metrics = quality_analyzer.analyze_quality(full_file_path, detected_survey_type)

            result["quality_rating"] = quality_rating.value
            result["quality_analysis"] = quality_metrics

            # Geometry analysis based on detected survey type
            try:
                if detected_survey_type == SurveyType.MIGRATED_3D:
                    # 3D geometry analysis
                    inline_values = []
                    xline_values = []
                    x_coords = []
                    y_coords = []

                    # Sample geometry (limit for performance)
                    sample_size = min(1000, f.tracecount)
                    step = max(1, f.tracecount // sample_size)

                    for i in range(0, f.tracecount, step):
                        header = f.header[i]
                        inline = header[segyio.TraceField.INLINE_3D]
                        xline = header[segyio.TraceField.CROSSLINE_3D]
                        x = header[segyio.TraceField.CDP_X]
                        y = header[segyio.TraceField.CDP_Y]

                        if inline != 0:
                            inline_values.append(inline)
                        if xline != 0:
                            xline_values.append(xline)
                        if x != 0:
                            x_coords.append(x)
                        if y != 0:
                            y_coords.append(y)

                    if inline_values and xline_values:
                        result.update({
                            "geometry_type": "3D_grid",
                            "min_inline": min(inline_values),
                            "max_inline": max(inline_values),
                            "inline_count": len(set(inline_values)),
                            "min_xline": min(xline_values),
                            "max_xline": max(xline_values),
                            "xline_count": len(set(xline_values))
                        })

                        if x_coords and y_coords:
                            result["coordinate_range"] = {
                                "min_x": min(x_coords),
                                "max_x": max(x_coords),
                                "min_y": min(y_coords),
                                "max_y": max(y_coords)
                            }

                elif detected_survey_type in [SurveyType.MIGRATED_2D, SurveyType.CDP_STACK]:
                    # 2D geometry analysis
                    cdp_values = []
                    shot_values = []
                    x_coords = []
                    y_coords = []

                    sample_size = min(1000, f.tracecount)
                    step = max(1, f.tracecount // sample_size)

                    for i in range(0, f.tracecount, step):
                        header = f.header[i]
                        cdp = header[segyio.TraceField.CDP]
                        shot = header[segyio.TraceField.TRACE_SEQUENCE_FILE]
                        x = header[segyio.TraceField.CDP_X]
                        y = header[segyio.TraceField.CDP_Y]

                        if cdp != 0:
                            cdp_values.append(cdp)
                        if shot != 0:
                            shot_values.append(shot)
                        if x != 0:
                            x_coords.append(x)
                        if y != 0:
                            y_coords.append(y)

                    if cdp_values:
                        result.update({
                            "geometry_type": "2D_line",
                            "min_cdp": min(cdp_values),
                            "max_cdp": max(cdp_values),
                            "cdp_count": len(set(cdp_values))
                        })

                    if shot_values:
                        result.update({
                            "min_shot_point": min(shot_values),
                            "max_shot_point": max(shot_values),
                            "shot_count": len(set(shot_values))
                        })

                    if x_coords and y_coords:
                        result["coordinate_range"] = {
                            "min_x": min(x_coords),
                            "max_x": max(x_coords),
                            "min_y": min(y_coords),
                            "max_y": max(y_coords)
                        }

                        # Calculate line length
                        line_length_m = math.sqrt((max(x_coords) - min(x_coords))**2 +
                                                (max(y_coords) - min(y_coords))**2)
                        result["line_length_km"] = round(line_length_m / 1000, 2)

                elif detected_survey_type == SurveyType.SHOT_GATHER:
                    # Shot gather analysis
                    field_records = []
                    offsets = []

                    sample_size = min(500, f.tracecount)
                    step = max(1, f.tracecount // sample_size)

                    for i in range(0, f.tracecount, step):
                        header = f.header[i]
                        field_record = header[segyio.TraceField.FieldRecord]
                        offset = header[segyio.TraceField.offset]

                        if field_record != 0:
                            field_records.append(field_record)
                        if offset != 0:
                            offsets.append(offset)

                    result.update({
                        "geometry_type": "shot_gather",
                        "field_record_count": len(set(field_records)),
                        "offset_range": [min(offsets), max(offsets)] if offsets else [0, 0]
                    })

            except Exception as e:
                logger.warning(f"Geometry analysis failed: {str(e)}")
                result["geometry_warning"] = f"Geometry analysis failed: {str(e)}"
                result["geometry_type"] = "unknown"

        # Add processing metadata
        result.update({
            "intelligent_template_fields": intelligent_template,
            "segyio_version": get_segyio_version(),
            "processing_notes": [
                "Processed using segyio for maximum reliability",
                "Quality thresholds calibrated for survey type",
                "Intelligent template detection applied"
            ]
        })

        # Final memory and timing information
        processing_time = time.time() - operation_start
        result["processing_time_seconds"] = round(processing_time, 2)
        result["final_memory_usage_mb"] = round(memory_monitor.get_memory_usage_mb(), 1)

        progress.finish()
        logger.info(f"SEG-Y parsing completed successfully in {processing_time:.1f}s")

        return {"text": json.dumps(result, cls=NumpyJSONEncoder)}

    except Exception as e:
        processing_time = time.time() - operation_start
        error_details = traceback.format_exc()
        logger.error(f"SEG-Y parsing failed after {processing_time:.1f}s: {str(e)}")
        logger.debug(error_details)

        return {"text": json.dumps({
            "error": f"Production SEG-Y parser error: {str(e)}",
            "processing_time_seconds": round(processing_time, 2),
            "memory_usage_mb": round(memory_monitor.get_memory_usage_mb(), 1),
            "details": error_details,
            "suggestions": [
                "Check file format compliance",
                "Verify file is accessible and not corrupted",
                "Ensure sufficient memory is available",
                "Try with smaller sample size for large files"
            ]
        })}

def find_segy_file(file_path: str, data_dir: str = "./data") -> str:
    """Find a SEG-Y file with enhanced path resolution"""
    # Check if it's already a full path
    if os.path.isfile(file_path):
        return file_path

    # Check in data directory
    potential_path = os.path.join(data_dir, file_path)
    if os.path.isfile(potential_path):
        return potential_path

    # Try adding extensions
    for ext in ['.sgy', '.segy', '.SGY', '.SEGY']:
        if not file_path.lower().endswith(ext.lower()):
            potential_path = os.path.join(data_dir, file_path + ext)
            if os.path.isfile(potential_path):
                return potential_path

    return file_path

def find_template_file(template_path: str, template_dir: str = "./templates") -> str:
    """Find a template file with enhanced path resolution"""
    if os.path.isfile(template_path):
        return template_path

    potential_path = os.path.join(template_dir, template_path)
    if os.path.isfile(potential_path):
        return potential_path

    # Try adding .sgyfmt extension
    if not template_path.lower().endswith('.sgyfmt'):
        potential_path = os.path.join(template_dir, template_path + '.sgyfmt')
        if os.path.isfile(potential_path):
            return potential_path

    return template_path

# Backward compatibility functions for existing MCP integrations
def FC_to_text(format_code: int) -> str:
    """Convert format code to text description"""
    format_descriptions = {
        1: "32 BIT IBM FORMAT",
        2: "32 BIT INTEGER",
        3: "16 BIT INTEGER",
        5: "32 BIT IEEE FORMAT",
        8: "8 BIT INTEGER"
    }
    return format_descriptions.get(format_code, f"FORMAT CODE {format_code}")

# Template validation for backward compatibility
class TemplateValidator:
    """Enhanced template validation and management"""

    def create_default_template(self, template_dir: str = "./templates") -> str:
        """Create a comprehensive default template"""
        os.makedirs(template_dir, exist_ok=True)

        default_template_path = os.path.join(template_dir, "DEFAULT_TRACE.sgyfmt")

        template_content = '''# Default SEG-Y Trace Header Template
# Based on SEG-Y Rev 1 Standard
# Format: "FIELD_NAME"  BYTE_POSITION

# Essential positioning fields
"SX"  73
"SY"  77
"XY SCALER"  71

# 2D Survey fields
"SHOT POINT"  17
"CDP"  21
"SP_SCALAR"  69

# 3D Survey fields  
"IN LINE"  189
"CROSS LINE"  193

# Additional useful fields
"ELEVATION"  41
"OFFSET"  37
"TRACE_NUMBER"  1
'''

        with open(default_template_path, 'w') as f:
            f.write(template_content)

        logger.info(f"Created default template: {default_template_path}")
        return default_template_path


def segy_complete_metadata_harvester(**params):
    """Complete metadata harvester using segyio for all header types"""

    # Parameter extraction and validation
    file_path = params.get("file_path")
    data_dir = params.get("data_dir", "./data")
    include_trace_sampling = params.get("include_trace_sampling", True)
    trace_sample_size = params.get("trace_sample_size", 100)
    include_statistics = params.get("include_statistics", True)

    if not file_path:
        return create_error_response("file_path parameter is required")

    # Construct full file path - handle relative paths properly
    if not os.path.isabs(file_path):
        if file_path.startswith('./') or file_path.startswith('.\\'):
            # Path already includes directory, just make absolute
            full_path = os.path.abspath(file_path)
        else:
            # Simple filename, add data_dir
            full_path = os.path.abspath(os.path.join(data_dir, file_path))
    else:
        full_path = file_path

    if not os.path.exists(full_path):
        return create_error_response(f"File not found: {full_path}")

    try:
        # Main processing with segyio
        metadata = extract_complete_metadata(
            full_path,
            include_trace_sampling,
            trace_sample_size,
            include_statistics
        )

        return {"text": json.dumps(metadata, cls=NumpyJSONEncoder)}

    except Exception as e:
        return create_error_response(f"Error processing file {file_path}: {str(e)}")


def extract_complete_metadata(filepath: str, include_trace_sampling: bool,
                              trace_sample_size: int, include_statistics: bool) -> Dict[str, Any]:
    """Extract comprehensive metadata from all header types"""

    start_time = datetime.now()

    with segyio.open(filepath, 'r', strict=False) as f:
        metadata = {
            "file_info": extract_file_info(filepath, f),
            "ebcdic_header": extract_ebcdic_metadata(f),
            "binary_header": extract_binary_metadata(f),
            "trace_headers_analysis": None,
            "processing_summary": None,
            "extraction_metadata": {
                "extraction_time": datetime.now().isoformat(),
                "processing_duration_seconds": None,
                "segyio_version": segyio_version,
                "parameters_used": {
                    "include_trace_sampling": include_trace_sampling,
                    "trace_sample_size": trace_sample_size,
                    "include_statistics": include_statistics
                }
            }
        }

        # Trace header analysis (optional, can be performance intensive)
        if include_trace_sampling and f.tracecount > 0:
            metadata["trace_headers_analysis"] = extract_trace_metadata(
                f, trace_sample_size, include_statistics
            )

        # Processing summary combining all headers
        metadata["processing_summary"] = generate_processing_summary(metadata)

        # Calculate processing time
        end_time = datetime.now()
        metadata["extraction_metadata"]["processing_duration_seconds"] = (
                end_time - start_time
        ).total_seconds()

    return metadata


def extract_file_info(filepath: str, segy_file) -> Dict[str, Any]:
    """Extract basic file information and accessibility metrics"""

    file_stats = os.stat(filepath)

    return {
        "file_path": filepath,
        "filename": os.path.basename(filepath),
        "file_size_bytes": file_stats.st_size,
        "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
        "file_modification_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
        "file_accessible": True,
        "segyio_readable": True,
        "total_traces": segy_file.tracecount,
        "samples_per_trace": segy_file.samples.size,
        "estimated_memory_requirement_mb": estimate_memory_usage(segy_file),
        "file_format_detected": detect_segy_format(segy_file)
    }


def estimate_memory_usage(segy_file) -> float:
    """Estimate memory requirement for full file loading"""
    bytes_per_sample = 4  # Assuming 32-bit format
    total_samples = segy_file.tracecount * segy_file.samples.size
    estimated_bytes = total_samples * bytes_per_sample
    return round(estimated_bytes / (1024 * 1024), 2)


def detect_segy_format(segy_file) -> str:
    """Detect SEG-Y format and revision"""
    try:
        format_code = segy_file.format
        format_map = {
            1: "32-bit IBM floating point",
            2: "32-bit two's complement integer",
            3: "16-bit two's complement integer",
            5: "32-bit IEEE floating point",
            8: "8-bit two's complement integer"
        }
        return format_map.get(format_code, f"Unknown format code: {format_code}")
    except:
        return "Format detection failed"


def extract_ebcdic_metadata(segy_file) -> Dict[str, Any]:
    """Extract and process EBCDIC header information"""

    ebcdic_data = {
        "text_headers_count": len(segy_file.text),
        "headers": [],
        "extracted_information": {
            "client_info": None,
            "contractor_info": None,
            "survey_area": None,
            "acquisition_dates": [],
            "processing_dates": [],
            "software_mentions": [],
            "coordinate_system_info": None,
            "processing_keywords": []
        },
        "text_quality": {
            "encoding_successful": True,
            "readable_lines": 0,
            "empty_lines": 0,
            "non_ascii_characters": 0,
            "documentation_score": 0
        }
    }

    for header_idx, header in enumerate(segy_file.text):
        try:
            # Convert EBCDIC to ASCII
            ascii_text = header.decode('cp037', errors='replace')
            lines = ascii_text.split('\n')[:40]  # SEG-Y has 40 lines max

            header_info = {
                "header_number": header_idx,
                "raw_text_lines": lines,
                "extracted_metadata": parse_ebcdic_content(lines),
                "quality_metrics": assess_text_quality(lines)
            }

            ebcdic_data["headers"].append(header_info)

            # Aggregate extracted information
            aggregate_ebcdic_information(
                ebcdic_data["extracted_information"],
                header_info["extracted_metadata"]
            )

            # Update quality metrics
            update_quality_metrics(
                ebcdic_data["text_quality"],
                header_info["quality_metrics"]
            )

        except UnicodeDecodeError as e:
            ebcdic_data["headers"].append({
                "header_number": header_idx,
                "encoding_error": str(e),
                "raw_bytes_sample": header[:100].hex()
            })
            ebcdic_data["text_quality"]["encoding_successful"] = False

    # Calculate overall documentation score
    ebcdic_data["text_quality"]["documentation_score"] = calculate_documentation_score(
        ebcdic_data["extracted_information"]
    )

    return ebcdic_data


def parse_ebcdic_content(lines: List[str]) -> Dict[str, Any]:
    """Parse structured information from EBCDIC text lines"""

    extracted = {
        "client_mentions": [],
        "contractor_mentions": [],
        "area_mentions": [],
        "date_mentions": [],
        "software_mentions": [],
        "processing_steps": [],
        "coordinate_references": [],
        "equipment_mentions": []
    }

    # Define regex patterns for common information
    patterns = {
        "client": [
            r'CLIENT[:\s]+([A-Z\s&]+)',
            r'COMPANY[:\s]+([A-Z\s&]+)',
            r'OPERATOR[:\s]+([A-Z\s&]+)'
        ],
        "contractor": [
            r'CONTRACTOR[:\s]+([A-Z\s&]+)',
            r'ACQUIRED BY[:\s]+([A-Z\s&]+)',
            r'PROCESSED BY[:\s]+([A-Z\s&]+)'
        ],
        "dates": [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*\d{1,2}[,\s]*\d{4}'
        ],
        "software": [
            r'(PROMAX|SEISSPACE|GEOCLUSTER|KINGDOM|PETREL|FOCUS|MESA)',
            r'SOFTWARE[:\s]+([A-Z\s]+)',
            r'SYSTEM[:\s]+([A-Z\s]+)'
        ],
        "processing": [
            r'(MIGRAT\w+|STACK\w+|FILTER\w+|DECON\w+|NMO|VELOCITY|AGC|GAIN)',
            r'PROCESS\w*[:\s]+([A-Z\s,]+)'
        ]
    }

    for line in lines:
        line_upper = line.upper().strip()
        if not line_upper:
            continue

        # Extract different types of information
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, line_upper)
                if matches:
                    if category == "client":
                        extracted["client_mentions"].extend(matches)
                    elif category == "contractor":
                        extracted["contractor_mentions"].extend(matches)
                    elif category == "dates":
                        extracted["date_mentions"].extend(matches)
                    elif category == "software":
                        extracted["software_mentions"].extend(matches)
                    elif category == "processing":
                        extracted["processing_steps"].extend(matches)

    return extracted


def assess_text_quality(lines: List[str]) -> Dict[str, Any]:
    """Assess quality of EBCDIC text content"""

    total_lines = len(lines)
    readable_lines = 0
    empty_lines = 0
    non_ascii_chars = 0

    for line in lines:
        if not line.strip():
            empty_lines += 1
        else:
            readable_lines += 1

        # Count non-ASCII characters
        non_ascii_chars += sum(1 for c in line if ord(c) > 127)

    return {
        "total_lines": total_lines,
        "readable_lines": readable_lines,
        "empty_lines": empty_lines,
        "non_ascii_characters": non_ascii_chars,
        "readability_percentage": (readable_lines / max(total_lines, 1)) * 100
    }


def extract_binary_metadata(segy_file) -> Dict[str, Any]:
    """Extract comprehensive binary header information"""

    binary_data = {
        "technical_specifications": extract_technical_specs(segy_file),
        "survey_parameters": extract_survey_parameters(segy_file),
        "processing_indicators": extract_processing_indicators(segy_file),
        "coordinate_information": extract_coordinate_info(segy_file),
        "quality_indicators": assess_binary_quality(segy_file)
    }

    return binary_data


def extract_technical_specs(segy_file) -> Dict[str, Any]:
    """Extract technical specifications from binary header - CORRECTED VERSION"""

    return {
        "job_identification_number": safe_bin_read(segy_file, segyio.BinField.JobID),
        "line_number": safe_bin_read(segy_file, segyio.BinField.LineNumber),
        "reel_number": safe_bin_read(segy_file, segyio.BinField.ReelNumber),
        # CORRECTED: Use available field names
        "traces_per_ensemble": safe_bin_read(segy_file, segyio.BinField.Traces),
        "auxiliary_traces_per_ensemble": safe_bin_read(segy_file, segyio.BinField.AuxTraces),
        "sample_interval_microseconds": safe_bin_read(segy_file, segyio.BinField.Interval),
        "sample_interval_original": safe_bin_read(segy_file, segyio.BinField.IntervalOriginal),
        "samples_per_trace": safe_bin_read(segy_file, segyio.BinField.Samples),
        "samples_per_trace_original": safe_bin_read(segy_file, segyio.BinField.SamplesOriginal),
        "data_sample_format_code": safe_bin_read(segy_file, segyio.BinField.Format),
        "measurement_system": safe_bin_read(segy_file, segyio.BinField.MeasurementSystem),
        "segy_format_revision_number": safe_bin_read(segy_file, segyio.BinField.SEGYRevision),
        "trace_sorting_code": safe_bin_read(segy_file, segyio.BinField.SortingCode)
    }

def extract_survey_parameters(segy_file) -> Dict[str, Any]:
    """Extract survey and acquisition parameters - CORRECTED VERSION"""

    return {
        # CORRECTED: Use EnsembleFold instead of CDPFold
        "cdp_fold": safe_bin_read(segy_file, segyio.BinField.EnsembleFold),
        "impulse_signal_polarity": safe_bin_read(segy_file, segyio.BinField.ImpulseSignalPolarity),
        "vibratory_polarity_code": safe_bin_read(segy_file, segyio.BinField.VibratoryPolarity),
        "correlated_data_traces": safe_bin_read(segy_file, segyio.BinField.CorrelatedTraces),
        "binary_gain_recovered": safe_bin_read(segy_file, segyio.BinField.BinaryGainRecovery),
        "amplitude_recovery_method": safe_bin_read(segy_file, segyio.BinField.AmplitudeRecovery),
        # Removed FixedLengthTraceFlag as it doesn't exist in segyio
    }


def extract_coordinate_info(segy_file) -> Dict[str, Any]:
    """Extract coordinate system information - CORRECTED VERSION"""

    return {
        "coordinate_units": safe_bin_read(segy_file, segyio.BinField.MeasurementSystem),
        # Removed EnsembleCoordinateScalar as it doesn't exist in segyio
        "coordinate_scalar_interpretation": "Coordinate scaling information available in trace headers"
    }


def safe_bin_read(segy_file, field) -> Any:
    """Safely read binary header field with error handling - ENHANCED VERSION"""
    try:
        return segy_file.bin[field]
    except (KeyError, IndexError, AttributeError) as e:
        # Log the specific error for debugging
        import logging
        logging.debug(f"Failed to read binary field {field}: {e}")
        return None

# DEBUGGING FUNCTION - Remove after testing
def debug_available_binfields():
    """Debug function to show all available BinField attributes"""
    print("=== Available segyio.BinField attributes ===")
    for attr in dir(segyio.BinField):
        if not attr.startswith('_'):
            try:
                value = getattr(segyio.BinField, attr)
                print(f"{attr:30} = {value}")
            except Exception as e:
                print(f"{attr:30} = ERROR: {e}")
    print("=" * 50)


# VALIDATION FUNCTION
def validate_binfield_access(segy_file):
    """Validate that we can access all the BinField attributes we use"""
    required_fields = {
        'JobID': segyio.BinField.JobID,
        'LineNumber': segyio.BinField.LineNumber,
        'ReelNumber': segyio.BinField.ReelNumber,
        'Traces': segyio.BinField.Traces,
        'AuxTraces': segyio.BinField.AuxTraces,
        'Interval': segyio.BinField.Interval,
        'IntervalOriginal': segyio.BinField.IntervalOriginal,
        'Samples': segyio.BinField.Samples,
        'SamplesOriginal': segyio.BinField.SamplesOriginal,
        'Format': segyio.BinField.Format,
        'MeasurementSystem': segyio.BinField.MeasurementSystem,
        'SEGYRevision': segyio.BinField.SEGYRevision,
        'SortingCode': segyio.BinField.SortingCode,
        'EnsembleFold': segyio.BinField.EnsembleFold,
        'ImpulseSignalPolarity': segyio.BinField.ImpulseSignalPolarity,
        'VibratoryPolarity': segyio.BinField.VibratoryPolarity,
        'CorrelatedTraces': segyio.BinField.CorrelatedTraces,
        'BinaryGainRecovery': segyio.BinField.BinaryGainRecovery,
        'AmplitudeRecovery': segyio.BinField.AmplitudeRecovery,
    }

    print("=== Validating BinField access ===")
    for name, field in required_fields.items():
        try:
            value = safe_bin_read(segy_file, field)
            print(f"✓ {name:25} = {value}")
        except Exception as e:
            print(f"✗ {name:25} = ERROR: {e}")
    print("=" * 40)


def interpret_coordinate_scalar(scalar_value) -> str:
    """Interpret coordinate scalar value"""
    if scalar_value is None:
        return "Unknown"
    elif scalar_value > 0:
        return f"Multiply coordinates by {scalar_value}"
    elif scalar_value < 0:
        return f"Divide coordinates by {abs(scalar_value)}"
    else:
        return "No scaling applied"


def extract_trace_metadata(segy_file, trace_sample_size: int,
                           include_statistics: bool) -> Dict[str, Any]:
    """Extract trace header information with sampling"""

    total_traces = segy_file.tracecount

    # FIX: Prevent division by zero error
    if trace_sample_size <= 0:
        trace_sample_size = 1  # Minimum sample size

    sample_size = min(trace_sample_size, total_traces)

    # Generate sampling indices
    if total_traces <= sample_size:
        sample_indices = list(range(total_traces))
    else:
        step = total_traces // sample_size
        sample_indices = list(range(0, total_traces, step))[:sample_size]

    trace_data = {
        "sampling_info": {
            "total_traces": total_traces,
            "traces_sampled": len(sample_indices),
            "sampling_percentage": (len(sample_indices) / total_traces) * 100,
            "sampling_method": "systematic" if total_traces > sample_size else "complete"
        },
        "spatial_analysis": analyze_spatial_distribution(segy_file, sample_indices),
        "trace_characteristics": analyze_trace_characteristics(segy_file, sample_indices),
        "coordinate_analysis": None,
        "statistics": None
    }

    # Coordinate analysis
    trace_data["coordinate_analysis"] = analyze_coordinates(segy_file, sample_indices)

    # Statistical analysis (optional)
    if include_statistics:
        trace_data["statistics"] = calculate_trace_statistics(segy_file, sample_indices)

    return trace_data


def analyze_spatial_distribution(segy_file, sample_indices: List[int]) -> Dict[str, Any]:
    """Analyze spatial distribution of traces"""

    spatial_info = {
        "unique_cdps": set(),
        "unique_inlines": set(),
        "unique_crosslines": set(),
        "unique_shots": set(),
        "unique_receivers": set(),
        "coordinate_ranges": {
            "source_x": {"min": None, "max": None},
            "source_y": {"min": None, "max": None},
            "group_x": {"min": None, "max": None},
            "group_y": {"min": None, "max": None}
        }
    }

    source_x_values = []
    source_y_values = []
    group_x_values = []
    group_y_values = []

    for idx in sample_indices:
        try:
            header = segy_file.header[idx]

            # Collect spatial identifiers
            cdp = header.get(segyio.TraceField.CDP)
            if cdp:
                spatial_info["unique_cdps"].add(cdp)

            inline = header.get(segyio.TraceField.INLINE_3D)
            if inline:
                spatial_info["unique_inlines"].add(inline)

            crossline = header.get(segyio.TraceField.CROSSLINE_3D)
            if crossline:
                spatial_info["unique_crosslines"].add(crossline)

            shot = header.get(segyio.TraceField.FieldRecord)
            if shot:
                spatial_info["unique_shots"].add(shot)

            # Collect coordinates
            source_x = header.get(segyio.TraceField.SourceX)
            if source_x:
                source_x_values.append(source_x)

            source_y = header.get(segyio.TraceField.SourceY)
            if source_y:
                source_y_values.append(source_y)

            group_x = header.get(segyio.TraceField.GroupX)
            if group_x:
                group_x_values.append(group_x)

            group_y = header.get(segyio.TraceField.GroupY)
            if group_y:
                group_y_values.append(group_y)

        except Exception:
            continue

    # Calculate coordinate ranges
    if source_x_values:
        spatial_info["coordinate_ranges"]["source_x"] = {
            "min": min(source_x_values),
            "max": max(source_x_values)
        }

    if source_y_values:
        spatial_info["coordinate_ranges"]["source_y"] = {
            "min": min(source_y_values),
            "max": max(source_y_values)
        }

    if group_x_values:
        spatial_info["coordinate_ranges"]["group_x"] = {
            "min": min(group_x_values),
            "max": max(group_x_values)
        }

    if group_y_values:
        spatial_info["coordinate_ranges"]["group_y"] = {
            "min": min(group_y_values),
            "max": max(group_y_values)
        }

    # Convert sets to counts for JSON serialization
    return {
        "unique_cdp_count": len(spatial_info["unique_cdps"]),
        "unique_inline_count": len(spatial_info["unique_inlines"]),
        "unique_crossline_count": len(spatial_info["unique_crosslines"]),
        "unique_shot_count": len(spatial_info["unique_shots"]),
        "coordinate_ranges": spatial_info["coordinate_ranges"],
        "survey_type_indicators": classify_survey_from_spatial(spatial_info)
    }


def classify_survey_from_spatial(spatial_info) -> Dict[str, Any]:
    """Classify survey type based on spatial distribution"""

    has_inlines = len(spatial_info["unique_inlines"]) > 1
    has_crosslines = len(spatial_info["unique_crosslines"]) > 1
    has_cdps = len(spatial_info["unique_cdps"]) > 1

    if has_inlines and has_crosslines:
        survey_type = "3D"
    elif has_cdps and not (has_inlines and has_crosslines):
        survey_type = "2D"
    else:
        survey_type = "Unknown"

    return {
        "probable_survey_type": survey_type,
        "evidence": {
            "has_inline_organization": has_inlines,
            "has_crossline_organization": has_crosslines,
            "has_cdp_organization": has_cdps
        }
    }


def generate_processing_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate high-level summary combining all header information"""

    summary = {
        "file_assessment": {
            "overall_quality": "Unknown",
            "readiness_for_processing": "Unknown",
            "data_integrity_status": "Unknown",
            "documentation_quality": "Unknown"
        },
        "survey_characterization": {
            "survey_type": "Unknown",
            "processing_level": "Unknown",
            "vintage_assessment": "Unknown",
            "business_context": "Unknown"
        },
        "technical_summary": {
            "data_volume": calculate_data_volume(metadata),
            "coordinate_system_status": assess_coordinate_system(metadata),
            "header_consistency": assess_header_consistency(metadata),
            "processing_requirements": estimate_processing_requirements(metadata)
        },
        "recommendations": generate_recommendations(metadata)
    }

    return summary


def calculate_data_volume(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate data volume metrics"""

    file_info = metadata.get("file_info", {})

    return {
        "file_size_mb": file_info.get("file_size_mb", 0),
        "total_traces": file_info.get("total_traces", 0),
        "samples_per_trace": file_info.get("samples_per_trace", 0),
        "estimated_memory_mb": file_info.get("estimated_memory_requirement_mb", 0),
        "size_category": categorize_file_size(file_info.get("file_size_mb", 0))
    }


def categorize_file_size(size_mb: float) -> str:
    """Categorize file size for processing planning"""
    if size_mb < 50:
        return "Small"
    elif size_mb < 500:
        return "Medium"
    elif size_mb < 2000:
        return "Large"
    else:
        return "Very Large"


def generate_recommendations(metadata: Dict[str, Any]) -> List[str]:
    """Generate processing and usage recommendations"""

    recommendations = []

    # File size recommendations
    file_info = metadata.get("file_info", {})
    size_mb = file_info.get("file_size_mb", 0)

    if size_mb > 1000:
        recommendations.append("Consider memory-efficient processing due to large file size")

    # Documentation recommendations
    ebcdic_info = metadata.get("ebcdic_header", {})
    doc_score = ebcdic_info.get("text_quality", {}).get("documentation_score", 0)

    if doc_score < 50:
        recommendations.append("Documentation quality is poor - consider header improvement")

    # Coordinate system recommendations
    binary_info = metadata.get("binary_header", {})
    coord_scalar = binary_info.get("coordinate_information", {}).get("ensemble_coordinate_scalar")

    if coord_scalar is None:
        recommendations.append("Coordinate scalar not defined - verify coordinate system")

    return recommendations

def create_error_response(message: str) -> Dict[str, str]:
    """Create standardized error response"""
    return {"error": message, "status": "failed"}


def aggregate_ebcdic_information(target: Dict, source: Dict) -> None:
    """Aggregate EBCDIC information from multiple headers"""
    for key, value in source.items():
        if isinstance(value, list):
            target[key] = target.get(key, []) + value
        elif value and not target.get(key):
            target[key] = value


def update_quality_metrics(target: Dict, source: Dict) -> None:
    """Update quality metrics aggregation"""
    target["readable_lines"] += source.get("readable_lines", 0)
    target["empty_lines"] += source.get("empty_lines", 0)
    target["non_ascii_characters"] += source.get("non_ascii_characters", 0)


def calculate_documentation_score(extracted_info: Dict[str, Any]) -> int:
    """Calculate documentation quality score 0-100"""

    score = 0
    max_score = 100

    # Client information (20 points)
    if extracted_info.get("client_info"):
        score += 20

    # Contractor information (20 points)
    if extracted_info.get("contractor_info"):
        score += 20

    # Date information (15 points)
    if extracted_info.get("acquisition_dates") or extracted_info.get("processing_dates"):
        score += 15

    # Processing information (15 points)
    if extracted_info.get("processing_keywords"):
        score += 15

    # Software information (10 points)
    if extracted_info.get("software_mentions"):
        score += 10

    # Coordinate system (10 points)
    if extracted_info.get("coordinate_system_info"):
        score += 10

    # Survey area (10 points)
    if extracted_info.get("survey_area"):
        score += 10

    return min(score, max_score)


def extract_processing_indicators(segy_file) -> dict:
    """Extract processing indicators from binary header - CORRECTED VERSION"""
    return {
        "binary_gain_recovered": safe_bin_read(segy_file, segyio.BinField.BinaryGainRecovery),
        "amplitude_recovery_method": safe_bin_read(segy_file, segyio.BinField.AmplitudeRecovery),
        "impulse_signal_polarity": safe_bin_read(segy_file, segyio.BinField.ImpulseSignalPolarity),
        "vibratory_polarity_code": safe_bin_read(segy_file, segyio.BinField.VibratoryPolarity),
        "correlated_data_traces": safe_bin_read(segy_file, segyio.BinField.CorrelatedTraces),
        # Removed FixedLengthTraceFlag as it doesn't exist in segyio
    }


def assess_binary_quality(segy_file) -> dict:
    """Assess binary header quality and completeness"""
    required_fields = [
        segyio.BinField.Interval,
        segyio.BinField.Samples,
        segyio.BinField.Format
    ]

    valid_fields = 0
    for field in required_fields:
        if safe_bin_read(segy_file, field) is not None:
            valid_fields += 1

    return {
        "completeness_score": (valid_fields / len(required_fields)) * 100,
        "critical_fields_present": valid_fields == len(required_fields),
        "format_code_valid": safe_bin_read(segy_file, segyio.BinField.Format) in [1, 2, 3, 5, 8]
    }


def analyze_trace_characteristics(segy_file, sample_indices: list) -> dict:
    """Analyze trace characteristics from sampled traces"""
    trace_stats = {
        "trace_length_consistency": True,
        "sample_format_consistency": True,
        "header_completeness": 0
    }

    expected_samples = segy_file.samples.size
    complete_headers = 0

    for idx in sample_indices[:min(50, len(sample_indices))]:  # Limit to 50 for performance
        try:
            trace = segy_file.trace[idx]
            header = segy_file.header[idx]

            # Check trace length consistency
            if len(trace) != expected_samples:
                trace_stats["trace_length_consistency"] = False

            # Check header completeness
            required_fields = [
                segyio.TraceField.TRACE_SEQUENCE_FILE,
                segyio.TraceField.CDP,
                segyio.TraceField.SourceX,
                segyio.TraceField.SourceY
            ]

            field_count = 0
            for field in required_fields:
                if header.get(field) is not None:
                    field_count += 1

            if field_count == len(required_fields):
                complete_headers += 1

        except Exception:
            continue

    trace_stats["header_completeness"] = (complete_headers / len(sample_indices)) * 100 if sample_indices else 0

    return trace_stats


def analyze_coordinates(segy_file, sample_indices: list) -> dict:
    """Analyze coordinate information from trace headers"""
    coordinate_analysis = {
        "coordinate_system_detected": False,
        "coordinate_ranges": {},
        "coordinate_consistency": True,
        "coordinate_scaling_detected": False
    }

    coordinates = {
        "source_x": [],
        "source_y": [],
        "group_x": [],
        "group_y": [],
        "cdp_x": [],
        "cdp_y": []
    }

    for idx in sample_indices[:min(100, len(sample_indices))]:  # Sample for performance
        try:
            header = segy_file.header[idx]

            coord_fields = [
                (segyio.TraceField.SourceX, "source_x"),
                (segyio.TraceField.SourceY, "source_y"),
                (segyio.TraceField.GroupX, "group_x"),
                (segyio.TraceField.GroupY, "group_y"),
                (segyio.TraceField.CDP_X, "cdp_x"),
                (segyio.TraceField.CDP_Y, "cdp_y")
            ]

            for field, coord_type in coord_fields:
                value = header.get(field)
                if value is not None and value != 0:
                    coordinates[coord_type].append(value)
                    coordinate_analysis["coordinate_system_detected"] = True

        except Exception:
            continue

    # Calculate ranges for non-empty coordinate lists
    for coord_type, values in coordinates.items():
        if values:
            coordinate_analysis["coordinate_ranges"][coord_type] = {
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }

    return coordinate_analysis


def calculate_trace_statistics(segy_file, sample_indices: list) -> dict:
    """Calculate statistical information from trace data"""
    statistics = {
        "amplitude_statistics": {},
        "data_quality_metrics": {},
        "trace_characteristics": {}
    }

    amplitude_values = []
    zero_counts = []

    for idx in sample_indices[:min(20, len(sample_indices))]:  # Limit for performance
        try:
            trace = segy_file.trace[idx]

            # Collect amplitude statistics
            amplitude_values.extend(trace[::10])  # Sample every 10th value

            # Count zeros
            zero_count = np.sum(trace == 0)
            zero_counts.append(zero_count / len(trace) * 100)

        except Exception:
            continue

    if amplitude_values:
        amplitude_values = np.array(amplitude_values)
        statistics["amplitude_statistics"] = {
            "min": float(np.min(amplitude_values)),
            "max": float(np.max(amplitude_values)),
            "mean": float(np.mean(amplitude_values)),
            "std": float(np.std(amplitude_values)),
            "dynamic_range_db": float(
                20 * np.log10(np.max(np.abs(amplitude_values)) / (np.std(amplitude_values) + 1e-10)))
        }

    if zero_counts:
        statistics["data_quality_metrics"] = {
            "average_zero_percentage": float(np.mean(zero_counts)),
            "max_zero_percentage": float(np.max(zero_counts)),
            "traces_analyzed": len(zero_counts)
        }

    return statistics


def assess_coordinate_system(metadata: dict) -> str:
    """Assess coordinate system status from metadata"""
    binary_header = metadata.get("binary_header", {})
    coord_info = binary_header.get("coordinate_information", {})

    if coord_info.get("coordinate_scalar"):
        return "Coordinate scaling defined"

    trace_analysis = metadata.get("trace_headers_analysis", {})
    if trace_analysis and trace_analysis.get("coordinate_analysis", {}).get("coordinate_system_detected"):
        return "Coordinates present, scaling unclear"

    return "Coordinate system unclear"


def assess_header_consistency(metadata: dict) -> str:
    """Assess consistency between different header types"""
    issues = []

    # Check binary vs file info consistency
    file_info = metadata.get("file_info", {})
    binary_header = metadata.get("binary_header", {})

    file_samples = file_info.get("samples_per_trace")
    binary_samples = binary_header.get("technical_specifications", {}).get("samples_per_trace")

    if file_samples and binary_samples and file_samples != binary_samples:
        issues.append("Sample count mismatch between file and binary header")

    if not issues:
        return "Headers consistent"
    else:
        return f"Issues found: {'; '.join(issues)}"


def estimate_processing_requirements(metadata: dict) -> dict:
    """Estimate processing requirements based on file characteristics"""
    file_info = metadata.get("file_info", {})
    size_mb = file_info.get("file_size_mb", 0)

    requirements = {
        "memory_category": "Low",
        "processing_complexity": "Standard",
        "estimated_processing_time": "Short"
    }

    if size_mb > 1000:
        requirements["memory_category"] = "High"
        requirements["estimated_processing_time"] = "Long"
    elif size_mb > 200:
        requirements["memory_category"] = "Medium"
        requirements["estimated_processing_time"] = "Medium"

    return requirements


def interpret_measurement_system(system_code) -> str:
    """Interpret measurement system code"""
    if system_code == 1:
        return "Meters"
    elif system_code == 2:
        return "Feet"
    else:
        return f"Unknown system code: {system_code}"


def classify_survey_from_traces(spatial_data: dict) -> dict:
    """Classify survey type from spatial trace data"""
    inline_count = spatial_data.get("unique_inline_count", 0)
    crossline_count = spatial_data.get("unique_crossline_count", 0)
    cdp_count = spatial_data.get("unique_cdp_count", 0)

    if inline_count > 1 and crossline_count > 1:
        survey_type = "3D"
        confidence = "High"
    elif cdp_count > 1 and (inline_count <= 1 or crossline_count <= 1):
        survey_type = "2D"
        confidence = "Medium"
    else:
        survey_type = "Unknown"
        confidence = "Low"

    return {
        "survey_type": survey_type,
        "confidence": confidence,
        "evidence": {
            "inline_count": inline_count,
            "crossline_count": crossline_count,
            "cdp_count": cdp_count
        }
    }


def calculate_coordinate_ranges(coordinate_data: dict) -> dict:
    """Calculate coordinate ranges from coordinate data"""
    ranges = {}

    for coord_type, values in coordinate_data.items():
        if values:
            ranges[coord_type] = {
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values),
                "count": len(values)
            }

    return ranges


def determine_organization_type(cdp_count: int, inline_count: int, crossline_count: int) -> str:
    """Determine data organization type"""
    if inline_count > 1 and crossline_count > 1:
        return "3D Grid Organization"
    elif cdp_count > 1:
        return "2D Line Organization"
    elif inline_count > 1 or crossline_count > 1:
        return "Partial 3D Organization"
    else:
        return "Unknown Organization"


def assess_file_complexity(metadata: dict) -> str:
    """Assess overall file complexity"""
    file_info = metadata.get("file_info", {})
    traces = file_info.get("total_traces", 0)
    size_mb = file_info.get("file_size_mb", 0)

    if traces > 100000 or size_mb > 1000:
        return "High"
    elif traces > 10000 or size_mb > 100:
        return "Medium"
    else:
        return "Low"