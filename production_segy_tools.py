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
            "segyio_version": segyio.__version__ if hasattr(segyio, '__version__') else "unknown",
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