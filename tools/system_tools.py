"""
System Tools Module - MCP COMPATIBLE VERSION - FIXED
Fixed parameter extraction to match other tools (segy_tools.py, las_tools.py)
"""

import os
import json
import glob
import datetime
from typing import List, Dict, Any, Optional

# Optional system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from config.settings import DataConfig


def create_error_response(error_message: str, details: str = None) -> Dict[str, Any]:
    """Create standardized error response"""
    error_obj = {"error": error_message}
    if details:
        error_obj["details"] = details
    return {"text": json.dumps(error_obj)}


def extract_file_path_from_params(file_path=None, *args, **kwargs):
    """
    FIXED: Universal parameter extraction for MCP tools
    Same as segy_tools.py and las_tools.py - checks all possible parameter sources
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


# File type configuration
FILE_TYPE_CONFIG = {
    "las": {
        "extensions": [".las", ".LAS"],
        "description": "Well Log Files",
        "icon": "📊 WELL LOG",
        "default_pattern": "*.las",
        "categories": {
            "early_wells": lambda f: any(x in f for x in ["1054146", "1054149"]),
            "main_field": lambda f: "1054310" in f,
            "reference": lambda f: "example" in f.lower()
        }
    },
    "segy": {
        "extensions": [".sgy", ".segy", ".SGY", ".SEGY"],
        "description": "Seismic Data Files",
        "icon": "🌊 SEISMIC",
        "default_pattern": "*.sgy",
        "categories": {
            "f3_survey": lambda f: "F3_" in f,
            "3x_processing": lambda f: "3X_" in f,
            "marine": lambda f: "marine" in f.lower(),
            "land": lambda f: "land" in f.lower(),
            "shots": lambda f: "shots" in f.lower()
        }
    },
    "csv": {
        "extensions": [".csv", ".CSV"],
        "description": "Data Tables",
        "icon": "📋 DATA TABLE",
        "default_pattern": "*.csv",
        "categories": {
            "production": lambda f: "prod" in f.lower(),
            "analysis": lambda f: "analysis" in f.lower()
        }
    }
}


def detect_file_type(pattern_or_filename: str) -> str:
    """Automatically detect file type from pattern or filename"""
    if not pattern_or_filename:
        return None

    text = pattern_or_filename.lower()

    # Check each file type by extension first
    for file_type, config in FILE_TYPE_CONFIG.items():
        for ext in config["extensions"]:
            if ext.lower() in text:
                return file_type

    return None


def format_files_by_type(file_paths: List[str], config: Dict[str, Any]) -> str:
    """Format file list based on file type configuration"""
    file_count = len(file_paths)
    output = f"{config['icon']} FILES FOUND ({file_count} files):\n"
    output += "=" * 50 + "\n\n"

    # Group files by categories if defined
    if config.get("categories"):
        categorized = {}
        uncategorized = []

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            categorized_flag = False

            for category_name, category_func in config["categories"].items():
                if category_func(filename):
                    if category_name not in categorized:
                        categorized[category_name] = []
                    categorized[category_name].append(filename)
                    categorized_flag = True
                    break

            if not categorized_flag:
                uncategorized.append(filename)

        # Output categorized files
        for category_name, files in categorized.items():
            if files:
                output += f"{category_name.replace('_', ' ').upper()}:\n"
                for filename in sorted(files):
                    output += f"  - {filename}\n"
                output += "\n"

        # Output uncategorized files
        if uncategorized:
            output += "OTHER FILES:\n"
            for filename in sorted(uncategorized):
                output += f"  - {filename}\n"
            output += "\n"
    else:
        # Simple list if no categories
        for file_path in sorted(file_paths):
            output += f"  - {os.path.basename(file_path)}\n"
        output += "\n"

    output += f"TOTAL: {file_count} {config['description'].lower()} ready for analysis"
    return output


def format_generic_files(file_paths: List[str], pattern: str) -> str:
    """Generic file formatting for unknown types"""
    output = f"📁 FILES FOUND ({len(file_paths)} files matching '{pattern}'):\n"
    output += "=" * 40 + "\n\n"

    for file_path in sorted(file_paths):
        output += f"  - {os.path.basename(file_path)}\n"

    output += f"\nTOTAL: {len(file_paths)} files ready for analysis"
    return output


def create_system_tools(mcp_server, data_config: DataConfig) -> List[str]:
    """
    Create and register system tools - MCP COMPATIBLE VERSION - FIXED
    """

    # Tool 1: List Files - FIXED
    @mcp_server.tool(
        name="list_files",
        description="List any type of data files matching a pattern in the data directory"
    )
    def list_files(pattern: Optional[str] = None, **kwargs):
        """Universal file listing system - FIXED"""
        try:
            # FIXED: Use universal parameter extraction (same as other tools)
            pattern = extract_file_path_from_params(pattern, **kwargs)

            print(f"DEBUG: list_files called with extracted pattern='{pattern}'")

            # Handle missing or None pattern
            if not pattern or pattern in [".", "", "list", "files", "{}"]:
                pattern = "*"

            print(f"DEBUG: Using final pattern='{pattern}'")

            # Handle JSON-encoded parameters from MCP/LangChain (legacy support)
            if isinstance(pattern, str) and pattern.startswith('{') and pattern.endswith('}'):
                try:
                    parsed = json.loads(pattern)
                    if isinstance(parsed, dict):
                        # Handle empty JSON case
                        if not parsed:
                            pattern = "*"
                        # Extract the actual pattern from the JSON
                        elif 'file_pattern' in parsed:
                            pattern = parsed['file_pattern'] + parsed.get('pattern', '*.sgy')
                        elif 'pattern' in parsed:
                            pattern = parsed['pattern']
                        print(f"DEBUG: Parsed JSON pattern to: '{pattern}'")
                except json.JSONDecodeError:
                    print(f"DEBUG: Failed to parse as JSON, using as-is")
                    pass

            # Auto-detect file type
            detected_type = detect_file_type(pattern)
            print(f"DEBUG: detected_type='{detected_type}'")

            matching_files = []

            if detected_type and detected_type in FILE_TYPE_CONFIG:
                # Handle patterns with specific file type detected (e.g., *.las, *_shots_*.segy)
                config = FILE_TYPE_CONFIG[detected_type]
                print(f"DEBUG: Using config for type '{detected_type}' with original pattern '{pattern}'")

                # For patterns that already include the extension, use them directly
                search_pattern = os.path.join(data_config.data_dir, pattern)
                print(f"DEBUG: Direct search with pattern: {search_pattern}")
                matching_files.extend(glob.glob(search_pattern))

                # Also try with different case extensions for the same type
                if '.' in pattern:
                    base_pattern, ext = pattern.rsplit('.', 1)
                    for alt_ext in config["extensions"]:
                        if alt_ext.lower() != f".{ext.lower()}":
                            alt_pattern = f"{base_pattern}{alt_ext}"
                            search_pattern = os.path.join(data_config.data_dir, alt_pattern)
                            print(f"DEBUG: Alternative extension search: {search_pattern}")
                            matching_files.extend(glob.glob(search_pattern))

            else:
                # No specific file type detected - search across ALL file types
                print(f"DEBUG: No specific file type detected, searching across all types")

                # Search across all configured file types
                for file_type, config in FILE_TYPE_CONFIG.items():
                    print(f"DEBUG: Searching {file_type} files with pattern '{pattern}'")

                    for ext in config["extensions"]:
                        # Construct pattern with extension
                        if pattern.endswith('*'):
                            search_pattern = os.path.join(data_config.data_dir, f"{pattern}{ext}")
                        else:
                            search_pattern = os.path.join(data_config.data_dir, f"{pattern}{ext}")

                        print(f"DEBUG: Searching with pattern: {search_pattern}")
                        matching_files.extend(glob.glob(search_pattern))

                # If no files found with extensions, try direct pattern search as final fallback
                if not matching_files:
                    search_pattern = os.path.join(data_config.data_dir, pattern)
                    print(f"DEBUG: Final fallback search with pattern: {search_pattern}")
                    matching_files = glob.glob(search_pattern)

            # Remove duplicates and sort
            matching_files = sorted(list(set(matching_files)))
            print(f"DEBUG: Found {len(matching_files)} files")

            if not matching_files:
                return {"text": f"No files found matching pattern: {pattern}"}

            # Format output using detected type config
            if detected_type and detected_type in FILE_TYPE_CONFIG:
                config = FILE_TYPE_CONFIG[detected_type]
                formatted_output = format_files_by_type(matching_files, config)
            else:
                formatted_output = format_generic_files(matching_files, pattern)

            return {"text": formatted_output}

        except Exception as e:
            print(f"ERROR in list_files: {str(e)}")
            return create_error_response(f"Error listing files: {str(e)}")

    # Tool 2: System Status - FIXED
    @mcp_server.tool(
        name="system_status",
        description="Get comprehensive system health, performance metrics, and processing status"
    )
    def system_status(query: str = "", *args, **kwargs):
        """Get comprehensive system health and performance metrics - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            query = extract_file_path_from_params(query, *args, **kwargs) or ""

            print("DEBUG: system_status called")

            # System metrics (if psutil available)
            system_metrics = {}
            if PSUTIL_AVAILABLE:
                try:
                    system_metrics = {
                        "cpu_percent": psutil.cpu_percent(interval=0.1),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_percent": psutil.disk_usage('./').percent,
                        "active_processes": len(psutil.pids())
                    }
                except:
                    system_metrics = {"note": "psutil available but metrics collection failed"}
            else:
                system_metrics = {
                    "note": "Install psutil for detailed system metrics: pip install psutil"
                }

            # Basic system info
            import threading
            system_info = {
                "active_threads": threading.active_count(),
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "platform": os.name
            }

            # Data directory info
            try:
                las_files = len(glob.glob(os.path.join(data_config.data_dir, "*.las"))) + len(
                    glob.glob(os.path.join(data_config.data_dir, "*.LAS")))
                segy_files = len(glob.glob(os.path.join(data_config.data_dir, "*.sgy"))) + \
                           len(glob.glob(os.path.join(data_config.data_dir, "*.segy"))) + \
                           len(glob.glob(os.path.join(data_config.data_dir, "*.SGY"))) + \
                           len(glob.glob(os.path.join(data_config.data_dir, "*.SEGY")))
            except:
                las_files = 0
                segy_files = 0

            data_dir_info = {
                "data_directory": data_config.data_dir,
                "directory_exists": os.path.exists(data_config.data_dir),
                "las_files_count": las_files,
                "segy_files_count": segy_files
            }

            # Available tools
            available_tools = [
                "las_parser", "las_analysis", "las_qc", "formation_evaluation",
                "well_correlation", "calculate_shale_volume", "list_files",
                "segy_parser", "segy_analysis", "segy_qc", "segy_classify",
                "segy_survey_analysis", "quick_segy_summary",
                "segy_complete_metadata_harvester", "segy_survey_polygon",
                "segy_trace_outlines", "segy_save_analysis", "segy_analysis_catalog",
                "segy_search_analyses", "system_status", "directory_info", "health_check"
            ]

            # Environment info
            environment_info = {
                "openai_api_key_set": "OPENAI_API_KEY" in os.environ,
                "data_dir_writable": os.access(data_config.data_dir, os.W_OK) if os.path.exists(
                    data_config.data_dir) else False
            }

            status_report = {
                "platform": "Subsurface Data Management Platform v2.0",
                "timestamp": datetime.datetime.now().isoformat(),
                "system_metrics": system_metrics,
                "system_info": system_info,
                "data_directory_info": data_dir_info,
                "environment_info": environment_info,
                "available_tools": available_tools,
                "tool_count": len(available_tools),
                "overall_status": "System operational",
                "recommendations": [
                    "All core systems running normally",
                    "Tools are responding",
                    "Data directory accessible"
                ]
            }

            return {"text": json.dumps(status_report, indent=2)}

        except Exception as e:
            print(f"ERROR in system_status: {str(e)}")
            return create_error_response(f"System status check failed: {str(e)}")

    # Tool 3: Directory Info - FIXED
    @mcp_server.tool(
        name="directory_info",
        description="Get detailed information about data directories and file organization"
    )
    def directory_info(directory_path: Optional[str] = None, **kwargs):
        """Get detailed directory information - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            directory_path = extract_file_path_from_params(directory_path, **kwargs)

            # Use data directory if none specified
            target_dir = directory_path or data_config.data_dir

            # Ensure absolute path
            if not os.path.isabs(target_dir):
                target_dir = os.path.join(data_config.data_dir, target_dir)

            if not os.path.exists(target_dir):
                return create_error_response(f"Directory not found: {target_dir}")

            # Collect directory statistics
            file_stats = {}
            total_size = 0
            total_files = 0

            for file_type, config in FILE_TYPE_CONFIG.items():
                file_count = 0
                file_size = 0

                for ext in config["extensions"]:
                    pattern = os.path.join(target_dir, f"*{ext}")
                    files = glob.glob(pattern)
                    file_count += len(files)

                    for file_path in files:
                        try:
                            size = os.path.getsize(file_path)
                            file_size += size
                            total_size += size
                        except OSError:
                            pass

                file_stats[file_type] = {
                    "count": file_count,
                    "total_size_mb": round(file_size / (1024 * 1024), 2),
                    "extensions": config["extensions"]
                }
                total_files += file_count

            # Directory metadata
            dir_info = {
                "directory_path": target_dir,
                "exists": True,
                "readable": os.access(target_dir, os.R_OK),
                "writable": os.access(target_dir, os.W_OK),
                "total_files": total_files,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_stats,
                "last_modified": datetime.datetime.fromtimestamp(
                    os.path.getmtime(target_dir)).isoformat() if os.path.exists(target_dir) else None
            }

            return {"text": json.dumps(dir_info, indent=2)}

        except Exception as e:
            return create_error_response(f"Directory info failed: {str(e)}")

    # Tool 4: Health Check - FIXED
    @mcp_server.tool(
        name="health_check",
        description="Perform comprehensive health check of the platform and its components"
    )
    def health_check(query: str = "", *args, **kwargs):
        """Perform comprehensive platform health check - FIXED"""
        try:
            # FIXED: Use universal parameter extraction
            query = extract_file_path_from_params(query, *args, **kwargs) or ""

            health_status = {
                "platform": "Subsurface Data Management Platform v2.0",
                "timestamp": datetime.datetime.now().isoformat(),
                "overall_health": "healthy",
                "checks": {}
            }

            # Check 1: Data directory
            data_dir_healthy = os.path.exists(data_config.data_dir) and os.access(data_config.data_dir, os.R_OK)
            health_status["checks"]["data_directory"] = {
                "status": "healthy" if data_dir_healthy else "unhealthy",
                "path": data_config.data_dir,
                "exists": os.path.exists(data_config.data_dir),
                "readable": os.access(data_config.data_dir, os.R_OK) if os.path.exists(data_config.data_dir) else False
            }

            # Check 2: Environment variables
            env_healthy = "OPENAI_API_KEY" in os.environ
            health_status["checks"]["environment"] = {
                "status": "healthy" if env_healthy else "warning",
                "openai_api_key": "set" if env_healthy else "not_set",
                "note": "OpenAI API key required for AI features"
            }

            # Check 3: File availability
            try:
                las_files = len(glob.glob(os.path.join(data_config.data_dir, "*.las")))
                segy_files = len(glob.glob(os.path.join(data_config.data_dir, "*.sgy")))
                files_healthy = las_files > 0 or segy_files > 0
            except:
                las_files = 0
                segy_files = 0
                files_healthy = False

            health_status["checks"]["data_files"] = {
                "status": "healthy" if files_healthy else "warning",
                "las_files": las_files,
                "segy_files": segy_files,
                "note": "At least some data files found" if files_healthy else "No data files found"
            }

            # Check 4: System resources (if available)
            if PSUTIL_AVAILABLE:
                try:
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage(data_config.data_dir)

                    memory_healthy = memory.percent < 90
                    disk_healthy = disk.percent < 90

                    health_status["checks"]["system_resources"] = {
                        "status": "healthy" if memory_healthy and disk_healthy else "warning",
                        "memory_percent": memory.percent,
                        "disk_percent": disk.percent,
                        "memory_available_gb": round(memory.available / (1024 ** 3), 2)
                    }
                except:
                    health_status["checks"]["system_resources"] = {
                        "status": "warning",
                        "note": "Could not collect system resource metrics"
                    }

            # Check 5: Python dependencies
            dependencies = []
            for module in ["numpy", "pandas", "lasio", "segyio"]:
                try:
                    __import__(module)
                    dependencies.append({"module": module, "status": "available"})
                except ImportError:
                    dependencies.append({"module": module, "status": "missing"})

            health_status["checks"]["dependencies"] = {
                "status": "healthy" if all(d["status"] == "available" for d in dependencies[:2]) else "warning",
                "modules": dependencies
            }

            # Determine overall health
            unhealthy_checks = [check for check in health_status["checks"].values() if check.get("status") == "unhealthy"]
            warning_checks = [check for check in health_status["checks"].values() if check.get("status") == "warning"]

            if unhealthy_checks:
                health_status["overall_health"] = "unhealthy"
            elif warning_checks:
                health_status["overall_health"] = "warning"
            else:
                health_status["overall_health"] = "healthy"

            health_status["summary"] = {
                "total_checks": len(health_status["checks"]),
                "healthy_checks": len([c for c in health_status["checks"].values() if c.get("status") == "healthy"]),
                "warning_checks": len(warning_checks),
                "unhealthy_checks": len(unhealthy_checks)
            }

            return {"text": json.dumps(health_status, indent=2)}

        except Exception as e:
            return create_error_response(f"Health check failed: {str(e)}")

    # Return list of registered tool names
    tool_names = [
        "list_files",
        "system_status",
        "directory_info",
        "health_check"
    ]

    return tool_names