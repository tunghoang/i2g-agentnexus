"""
production_segy_multifile.py - Production-quality multi-file SEG-Y processing

This module provides robust multi-file processing capabilities for SEG-Y surveys
with advanced configuration management, batch processing, and comprehensive reporting.
"""

import os
import sys
import json
import traceback
import time
import logging
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import yaml
from dataclasses import dataclass, asdict
import hashlib

from production_segy_tools import (
    NumpyJSONEncoder, ProgressReporter, MemoryMonitor, SegyioValidator,
    TemplateValidator, production_segy_parser, find_segy_file, find_template_file
)
from production_segy_analysis_qc import (
    production_segy_analysis, production_segy_qc
)

logger = logging.getLogger(__name__)

@dataclass
class SEGYProcessingConfig:
    """Configuration class for SEG-Y processing operations"""
    
    # File processing settings
    max_concurrent_files: int = 4
    max_memory_gb: float = 8.0
    timeout_seconds: int = 3600  # 1 hour default timeout
    
    # Analysis settings
    default_analysis_type: str = "quick"
    max_traces_per_file: int = 50000
    enable_qc: bool = True
    enable_geometry_analysis: bool = True
    
    # Template settings
    default_template: str = "DEFAULT_TRACE.sgyfmt"
    template_auto_detect: bool = True
    template_validation_strict: bool = False
    
    # Output settings
    save_individual_results: bool = True
    save_summary_report: bool = True
    output_format: str = "json"  # json, yaml, csv
    
    # Error handling
    continue_on_error: bool = True
    max_retry_attempts: int = 2
    error_threshold: float = 0.5  # Stop if >50% of files fail
    
    # Performance settings
    enable_progress_reporting: bool = True
    progress_update_interval: float = 5.0
    memory_check_interval: float = 30.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SEGYProcessingConfig':
        """Create from dictionary"""
        return cls(**config_dict)
        
    @classmethod
    def from_file(cls, config_path: str) -> 'SEGYProcessingConfig':
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return cls()  # Return default config
            
    def save_to_file(self, config_path: str):
        """Save configuration to file"""
        config_dict = self.to_dict()
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_dict, f, indent=2)

class SEGYSurveyProcessor:
    """Production-quality multi-file SEG-Y processor"""
    
    def __init__(self, config: Optional[SEGYProcessingConfig] = None):
        self.config = config or SEGYProcessingConfig()
        self.memory_monitor = MemoryMonitor(self.config.max_memory_gb)
        self.validator = SegyioValidator()
        self.template_validator = TemplateValidator()
        
        # Processing state
        self.results = {}
        self.errors = {}
        self.processing_stats = {
            "files_processed": 0,
            "files_failed": 0,
            "total_processing_time": 0,
            "total_file_size_mb": 0
        }
        
    def find_survey_files(self, pattern: str, data_dir: str = "./data") -> List[str]:
        """Find all SEG-Y files matching the pattern"""
        try:
            # Handle different pattern types
            if os.path.dirname(pattern):
                # Pattern includes directory
                search_paths = [pattern]
            else:
                # Pattern is just filename, prepend data_dir
                search_paths = [os.path.join(data_dir, pattern)]
                
            # Add extension variants if not specified
            if not any(pattern.lower().endswith(ext) for ext in ['.sgy', '.segy']):
                search_paths.extend([
                    path + '.sgy' for path in search_paths
                ] + [
                    path + '.segy' for path in search_paths
                ])
                
            # Find all matching files
            all_files = []
            for search_path in search_paths:
                matching_files = glob.glob(search_path)
                all_files.extend(matching_files)
                
            # Filter to ensure they're SEG-Y files and remove duplicates
            segy_files = []
            seen_files = set()
            
            for file_path in all_files:
                if file_path not in seen_files and file_path.lower().endswith(('.sgy', '.segy')):
                    if os.path.isfile(file_path):
                        segy_files.append(file_path)
                        seen_files.add(file_path)
                        
            return sorted(segy_files)
            
        except Exception as e:
            logger.error(f"Error finding survey files: {str(e)}")
            return []
            
    def validate_survey_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Pre-validate all files before processing"""
        validation_results = {
            "total_files": len(file_paths),
            "valid_files": [],
            "invalid_files": [],
            "warnings": [],
            "total_size_mb": 0
        }
        
        for file_path in file_paths:
            try:
                # Basic file checks
                if not os.path.isfile(file_path):
                    validation_results["invalid_files"].append({
                        "file": file_path,
                        "error": "File not found"
                    })
                    continue
                    
                file_size_mb = os.path.getsize(file_path) / 1024**2
                validation_results["total_size_mb"] += file_size_mb
                
                # Quick SEG-Y validation
                file_validation = self.validator.validate_file_structure(file_path)
                
                if file_validation["issues"]:
                    validation_results["invalid_files"].append({
                        "file": os.path.basename(file_path),
                        "size_mb": round(file_size_mb, 1),
                        "issues": file_validation["issues"]
                    })
                else:
                    validation_results["valid_files"].append({
                        "file": os.path.basename(file_path),
                        "full_path": file_path,
                        "size_mb": round(file_size_mb, 1)
                    })
                    
                # Collect warnings
                if file_validation["warnings"]:
                    validation_results["warnings"].extend([
                        f"{os.path.basename(file_path)}: {warning}"
                        for warning in file_validation["warnings"]
                    ])
                    
            except Exception as e:
                validation_results["invalid_files"].append({
                    "file": os.path.basename(file_path),
                    "error": f"Validation error: {str(e)}"
                })
                
        return validation_results
        
    def process_single_file(self, file_path: str, template_path: str, 
                           operation: str = "parse") -> Dict[str, Any]:
        """Process a single SEG-Y file with error handling"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing {operation} for {os.path.basename(file_path)}")
            
            # Choose processing function based on operation
            if operation == "parse":
                result = production_segy_parser(
                    file_path=file_path,
                    template_path=template_path,
                    traces_to_read=self.config.max_traces_per_file
                )
            elif operation == "analyze":
                result = production_segy_analysis(
                    file_path=file_path,
                    template_path=template_path,
                    analysis_type=self.config.default_analysis_type,
                    max_analysis_traces=self.config.max_traces_per_file
                )
            elif operation == "qc":
                result = production_segy_qc(
                    file_path=file_path,
                    template_path=template_path
                )
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
            processing_time = time.time() - start_time
            
            # Parse result
            if "text" in result:
                parsed_result = json.loads(result["text"])
                
                if "error" not in parsed_result:
                    # Success
                    return {
                        "status": "success",
                        "file": os.path.basename(file_path),
                        "operation": operation,
                        "processing_time": round(processing_time, 2),
                        "result": parsed_result
                    }
                else:
                    # Processing error
                    return {
                        "status": "error",
                        "file": os.path.basename(file_path),
                        "operation": operation,
                        "processing_time": round(processing_time, 2),
                        "error": parsed_result["error"]
                    }
            else:
                return {
                    "status": "error",
                    "file": os.path.basename(file_path),
                    "operation": operation,
                    "processing_time": round(processing_time, 2),
                    "error": "No result returned from processing function"
                }
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Exception during {operation}: {str(e)}"
            logger.error(f"Error processing {file_path}: {error_msg}")
            
            return {
                "status": "error",
                "file": os.path.basename(file_path),
                "operation": operation,
                "processing_time": round(processing_time, 2),
                "error": error_msg,
                "details": traceback.format_exc()
            }
            
    def process_survey_parallel(self, file_paths: List[str], template_path: str,
                               operation: str = "parse") -> Dict[str, Any]:
        """Process multiple files in parallel with progress reporting"""
        
        if not file_paths:
            return {
                "status": "error",
                "message": "No files to process",
                "results": []
            }
            
        logger.info(f"Starting parallel {operation} of {len(file_paths)} files")
        
        # Progress reporting
        progress = ProgressReporter(len(file_paths), f"Processing {operation}")
        
        # Results storage
        results = []
        completed_files = 0
        failed_files = 0
        
        # Memory monitoring
        memory_check_time = time.time()
        
        def process_with_progress(file_path):
            """Wrapper function for processing with progress updates"""
            result = self.process_single_file(file_path, template_path, operation)
            
            nonlocal completed_files, failed_files, memory_check_time
            
            if result["status"] == "success":
                completed_files += 1
            else:
                failed_files += 1
                
            # Update progress
            progress.update(1, f"Completed: {completed_files}, Failed: {failed_files}")
            
            # Periodic memory check
            current_time = time.time()
            if current_time - memory_check_time > self.config.memory_check_interval:
                if not self.memory_monitor.check_memory_limit():
                    logger.warning("Memory usage high, suggesting garbage collection")
                    self.memory_monitor.suggest_gc()
                memory_check_time = current_time
                
            return result
            
        # Process files in parallel
        max_workers = min(self.config.max_concurrent_files, len(file_paths))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_with_progress, file_path): file_path
                for file_path in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    results.append(result)
                    
                    # Check error threshold
                    if (failed_files / len(file_paths)) > self.config.error_threshold:
                        logger.error(f"Error threshold exceeded ({failed_files}/{len(file_paths)})")
                        if not self.config.continue_on_error:
                            # Cancel remaining tasks
                            for remaining_future in future_to_file:
                                remaining_future.cancel()
                            break
                            
                except Exception as e:
                    logger.error(f"Task failed for {file_path}: {str(e)}")
                    results.append({
                        "status": "error",
                        "file": os.path.basename(file_path),
                        "operation": operation,
                        "error": f"Task execution failed: {str(e)}"
                    })
                    failed_files += 1
                    
        progress.finish()
        
        return {
            "status": "completed",
            "operation": operation,
            "total_files": len(file_paths),
            "successful_files": completed_files,
            "failed_files": failed_files,
            "success_rate": round((completed_files / len(file_paths)) * 100, 1),
            "results": results
        }
        
    def create_survey_summary(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive survey summary from processing results"""
        
        summary = {
            "survey_overview": {
                "total_files_processed": processing_results["total_files"],
                "successful_files": processing_results["successful_files"],
                "failed_files": processing_results["failed_files"],
                "success_rate_percent": processing_results["success_rate"]
            },
            "file_statistics": {
                "total_size_gb": 0,
                "average_file_size_mb": 0,
                "largest_file_mb": 0,
                "smallest_file_mb": float('inf')
            },
            "survey_characteristics": {
                "survey_types": set(),
                "format_codes": set(),
                "sample_rates": set(),
                "total_traces": 0
            },
            "quality_summary": {
                "excellent": 0,
                "good": 0, 
                "fair": 0,
                "poor": 0
            },
            "processing_performance": {
                "total_processing_time_minutes": 0,
                "average_processing_time_seconds": 0,
                "throughput_mb_per_second": 0
            }
        }
        
        # Analyze successful results
        successful_results = [r for r in processing_results["results"] if r["status"] == "success"]
        
        total_processing_time = 0
        total_file_size = 0
        
        for result in successful_results:
            if "result" in result:
                data = result["result"]
                
                # File size statistics
                file_size_mb = data.get("file_size_mb", 0)
                total_file_size += file_size_mb
                summary["file_statistics"]["largest_file_mb"] = max(
                    summary["file_statistics"]["largest_file_mb"], file_size_mb
                )
                summary["file_statistics"]["smallest_file_mb"] = min(
                    summary["file_statistics"]["smallest_file_mb"], file_size_mb
                )
                
                # Survey characteristics
                summary["survey_characteristics"]["survey_types"].add(
                    data.get("survey_type", "Unknown")
                )
                summary["survey_characteristics"]["format_codes"].add(
                    data.get("sample_format", "Unknown")
                )
                summary["survey_characteristics"]["sample_rates"].add(
                    data.get("sample_rate_ms", 0)
                )
                summary["survey_characteristics"]["total_traces"] += data.get("total_traces", 0)
                
                # Quality assessment (if available)
                if "quality_assessment" in data:
                    quality = data["quality_assessment"].get("overall_rating", "Unknown").lower()
                    if quality in summary["quality_summary"]:
                        summary["quality_summary"][quality] += 1
                        
                # Processing time
                total_processing_time += result.get("processing_time", 0)
                
        # Calculate derived statistics
        if successful_results:
            summary["file_statistics"]["total_size_gb"] = round(total_file_size / 1024, 2)
            summary["file_statistics"]["average_file_size_mb"] = round(
                total_file_size / len(successful_results), 1
            )
            summary["processing_performance"]["total_processing_time_minutes"] = round(
                total_processing_time / 60, 1
            )
            summary["processing_performance"]["average_processing_time_seconds"] = round(
                total_processing_time / len(successful_results), 1
            )
            
            if total_processing_time > 0:
                summary["processing_performance"]["throughput_mb_per_second"] = round(
                    total_file_size / total_processing_time, 1
                )
                
        # Convert sets to lists for JSON serialization
        for key in summary["survey_characteristics"]:
            if isinstance(summary["survey_characteristics"][key], set):
                summary["survey_characteristics"][key] = list(summary["survey_characteristics"][key])
                
        # Handle edge case for smallest file
        if summary["file_statistics"]["smallest_file_mb"] == float('inf'):
            summary["file_statistics"]["smallest_file_mb"] = 0
            
        return summary

def production_segy_survey_analysis(file_pattern=None, data_dir="./data", 
                                   template_dir="./templates", 
                                   config_path=None, **kwargs):
    """
    Production-quality multi-file SEG-Y survey analysis
    """
    operation_start = time.time()
    
    try:
        # Load configuration
        if config_path and os.path.isfile(config_path):
            config = SEGYProcessingConfig.from_file(config_path)
        else:
            config = SEGYProcessingConfig()
            
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        # Handle JSON input
        if 'input' in kwargs and kwargs['input'] is not None:
            try:
                if isinstance(kwargs['input'], str) and kwargs['input'].startswith('{'):
                    input_data = json.loads(kwargs['input'])
                    if isinstance(input_data, dict):
                        file_pattern = input_data.get('file_pattern', file_pattern)
                        data_dir = input_data.get('data_dir', data_dir)
                        template_dir = input_data.get('template_dir', template_dir)
                        
                        # Update config from input
                        for key, value in input_data.items():
                            if hasattr(config, key):
                                setattr(config, key, value)
                else:
                    file_pattern = kwargs['input']
            except json.JSONDecodeError as e:
                return {"text": json.dumps({
                    "error": f"Invalid JSON input: {str(e)}"
                })}
                
        if file_pattern is None:
            file_pattern = "*.sgy"
            
        logger.info(f"Starting survey analysis with pattern: {file_pattern}")
        
        # Initialize processor
        processor = SEGYSurveyProcessor(config)
        
        # Find survey files
        survey_files = processor.find_survey_files(file_pattern, data_dir)
        
        if not survey_files:
            return {"text": json.dumps({
                "error": f"No SEG-Y files found matching pattern: {file_pattern}",
                "search_directory": data_dir,
                "suggestions": [
                    "Check file pattern syntax",
                    "Verify files exist in data directory", 
                    "Try pattern like '*.sgy' or 'survey_*.segy'"
                ]
            })}
            
        logger.info(f"Found {len(survey_files)} SEG-Y files")
        
        # Pre-validate files
        validation_results = processor.validate_survey_files(survey_files)
        
        if not validation_results["valid_files"]:
            return {"text": json.dumps({
                "error": "No valid SEG-Y files found",
                "validation_results": validation_results,
                "suggestions": [
                    "Check file format compliance",
                    "Verify files are not corrupted",
                    "Review validation errors for specific issues"
                ]
            })}
            
        # Get or create template
        template_validator = TemplateValidator()
        template_path = kwargs.get('template_path')
        
        if template_path is None:
            template_path = template_validator.create_default_template(template_dir)
        else:
            full_template_path = find_template_file(template_path, template_dir)
            if not os.path.isfile(full_template_path):
                logger.warning(f"Template not found: {template_path}, using default")
                template_path = template_validator.create_default_template(template_dir)
            else:
                template_path = full_template_path
                
        # Determine operation type
        operation = kwargs.get('operation', 'parse')
        if operation not in ['parse', 'analyze', 'qc']:
            operation = 'parse'
            
        # Process files
        valid_file_paths = [f["full_path"] for f in validation_results["valid_files"]]
        
        processing_results = processor.process_survey_parallel(
            valid_file_paths, template_path, operation
        )
        
        # Create comprehensive summary
        survey_summary = processor.create_survey_summary(processing_results)
        
        # Build final result
        total_processing_time = time.time() - operation_start
        
        final_result = {
            "survey_analysis_summary": survey_summary,
            "file_validation": validation_results,
            "processing_results": processing_results,
            "configuration_used": config.to_dict(),
            "performance_metrics": {
                "total_processing_time_minutes": round(total_processing_time / 60, 1),
                "files_per_minute": round(len(valid_file_paths) / (total_processing_time / 60), 1),
                "average_memory_usage_mb": round(processor.memory_monitor.get_memory_usage_mb(), 1)
            },
            "recommendations": []
        }
        
        # Add recommendations based on results
        if processing_results["failed_files"] > 0:
            failure_rate = processing_results["failed_files"] / processing_results["total_files"]
            if failure_rate > 0.2:
                final_result["recommendations"].append(
                    f"High failure rate ({failure_rate*100:.1f}%) - review file quality and templates"
                )
                
        if survey_summary["quality_summary"]["poor"] > 0:
            final_result["recommendations"].append(
                f"{survey_summary['quality_summary']['poor']} files have poor quality - consider reprocessing"
            )
            
        if survey_summary["file_statistics"]["total_size_gb"] > 50:
            final_result["recommendations"].append(
                "Large survey detected - consider processing in smaller batches for better performance"
            )
            
        logger.info(f"Survey analysis completed in {total_processing_time/60:.1f} minutes")
        
        return {"text": json.dumps(final_result, cls=NumpyJSONEncoder)}
        
    except Exception as e:
        processing_time = time.time() - operation_start
        error_details = traceback.format_exc()
        logger.error(f"Survey analysis failed: {str(e)}")
        
        return {"text": json.dumps({
            "error": f"Production SEG-Y survey analysis error: {str(e)}",
            "processing_time_minutes": round(processing_time / 60, 1),
            "details": error_details,
            "suggestions": [
                "Check file patterns and accessibility",
                "Verify sufficient system resources",
                "Review configuration parameters",
                "Check logs for detailed error information"
            ]
        })}

# Configuration management utilities
def create_default_config(config_path: str = "./config/segy_config.yaml"):
    """Create default configuration file"""
    config = SEGYProcessingConfig()
    config.save_to_file(config_path)
    logger.info(f"Created default configuration: {config_path}")
    return config_path

def validate_config(config_path: str) -> Dict[str, Any]:
    """Validate configuration file"""
    try:
        config = SEGYProcessingConfig.from_file(config_path)
        return {
            "valid": True,
            "config": config.to_dict(),
            "issues": []
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "issues": [f"Configuration validation failed: {str(e)}"]
        }
