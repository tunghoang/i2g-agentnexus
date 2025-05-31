"""
MIVAA AgentNexus - Intelligent SEG-Y Processing Module

This module provides production-ready SEG-Y file analysis capabilities including:
- segyio-based robust file parsing and analysis
- Intelligent survey classification (2D/3D, Shot Gathers, Processed Data)
- Calibrated quality control with realistic thresholds
- Real-world data processing optimized for field conditions
- Template-free processing using industry-standard segyio library

Architecture:
- Core Engine: segyio (industry standard)
- Quality Control: Calibrated for survey-specific thresholds
- Classification: AI-powered with confidence scoring
- Performance: Memory-efficient, high-throughput processing
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Version and metadata
__version__ = "2.0.0"
__author__ = "MIVAA AgentNexus"
__description__ = "Production SEG-Y Processing with segyio"

# Check for required dependencies
_DEPENDENCIES_STATUS = {
    "segyio": {"available": False, "error": None, "required": True},
    "numpy": {"available": False, "error": None, "required": True},
    "scipy": {"available": False, "error": None, "required": False},
    "sklearn": {"available": False, "error": None, "required": False}
}

def _check_dependency(module_name: str, import_name: str = None):
    """Check if a dependency is available"""
    if import_name is None:
        import_name = module_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        _DEPENDENCIES_STATUS[module_name]["available"] = True
        _DEPENDENCIES_STATUS[module_name]["version"] = version
        logger.info(f"{module_name} {version} available")
        return True
    except ImportError as e:
        _DEPENDENCIES_STATUS[module_name]["error"] = str(e)
        logger.warning(f"{module_name} not available: {e}")
        return False

# Check all dependencies
_SEGYIO_AVAILABLE = _check_dependency("segyio")
_NUMPY_AVAILABLE = _check_dependency("numpy")
_SCIPY_AVAILABLE = _check_dependency("scipy")
_SKLEARN_AVAILABLE = _check_dependency("sklearn", "sklearn")

# Critical dependency check
_CORE_DEPENDENCIES_OK = _SEGYIO_AVAILABLE and _NUMPY_AVAILABLE

if not _CORE_DEPENDENCIES_OK:
    logger.error("Critical dependencies missing!")
    if not _SEGYIO_AVAILABLE:
        logger.error("Install segyio with: pip install segyio")
    if not _NUMPY_AVAILABLE:
        logger.error("Install numpy with: pip install numpy")
else:
    logger.info("✓ Core dependencies satisfied (segyio + numpy)")

# Import core modules only if dependencies are satisfied
if _CORE_DEPENDENCIES_OK:
    try:
        # Import updated segyio-based modules
        from production_segy_tools import (
            production_segy_parser,
            SegyioValidator,
            MemoryMonitor,
            ProgressReporter,
            NumpyJSONEncoder
        )

        from survey_classifier import (
            SegyioSurveyClassifier as SurveyClassifier,
            quick_classify_survey,
            batch_classify_directory
        )

        from production_segy_analysis_qc import (
            production_segy_qc,
            production_segy_analysis,
            SegyioQualityAnalyzer
        )

        from result_classes import (
            ClassificationResult,
            QualityMetrics,
            GeometryInfo,
            SurveyType,
            QualityRating,
            ProcessingEngine,
            ensure_result_compatibility,
            create_error_result
        )

        # Export main classes and functions
        __all__ = [
            # Core processing functions
            'production_segy_parser',
            'production_segy_qc',
            'production_segy_analysis',

            # Classification
            'SurveyClassifier',
            'quick_classify_survey',
            'batch_classify_directory',

            # Quality analysis
            'SegyioQualityAnalyzer',

            # Data structures
            'ClassificationResult',
            'QualityMetrics',
            'GeometryInfo',

            # Enums
            'SurveyType',
            'QualityRating',
            'ProcessingEngine',

            # Utilities
            'SegyioValidator',
            'MemoryMonitor',
            'ProgressReporter',
            'NumpyJSONEncoder',
            'ensure_result_compatibility',
            'create_error_result',

            # Dependency checking
            'check_dependencies',
            'get_dependency_status',
            'get_system_info'
        ]

        logger.info("✓ Intelligent SEG-Y processing enabled with segyio")

    except ImportError as e:
        logger.error(f"Failed to import core modules: {e}")
        logger.error("Some modules may not be updated to segyio architecture")

        # Fallback exports for partial functionality
        __all__ = ['check_dependencies', 'get_dependency_status', 'get_system_info']

else:
    # Graceful fallback when core dependencies are missing
    logger.warning("Intelligent SEG-Y processing disabled due to missing core dependencies")
    logger.warning("Install required packages: pip install segyio numpy")

    __all__ = ['check_dependencies', 'get_dependency_status', 'get_system_info']

def check_dependencies() -> bool:
    """
    Check if all required dependencies are available

    Returns:
        bool: True if core dependencies (segyio, numpy) are available
    """
    return _CORE_DEPENDENCIES_OK

def get_dependency_status() -> dict:
    """
    Get detailed dependency status information

    Returns:
        dict: Comprehensive dependency status and system information
    """
    return {
        "version": __version__,
        "dependencies": _DEPENDENCIES_STATUS.copy(),
        "core_dependencies_ok": _CORE_DEPENDENCIES_OK,
        "segyio_available": _SEGYIO_AVAILABLE,
        "numpy_available": _NUMPY_AVAILABLE,
        "scipy_available": _SCIPY_AVAILABLE,
        "sklearn_available": _SKLEARN_AVAILABLE,
        "architecture": "segyio-based" if _CORE_DEPENDENCIES_OK else "dependencies-missing",
        "recommended_install": [
            "pip install segyio numpy scipy scikit-learn"
        ] if not _CORE_DEPENDENCIES_OK else []
    }

def get_system_info() -> dict:
    """
    Get system information for debugging and support

    Returns:
        dict: System and package information
    """
    system_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "version": __version__,
        "architecture": "segyio-based",
        "dependencies": get_dependency_status()["dependencies"]
    }

    # Add package locations if available
    if _SEGYIO_AVAILABLE:
        try:
            import segyio
            system_info["segyio_location"] = str(Path(segyio.__file__).parent)
        except:
            pass

    return system_info

# Convenience functions for common operations
def quick_analysis(file_path: str, **kwargs) -> dict:
    """
    Perform quick analysis of a SEG-Y file

    Args:
        file_path: Path to SEG-Y file
        **kwargs: Additional parameters for analysis

    Returns:
        dict: Analysis results including classification and quality
    """
    if not _CORE_DEPENDENCIES_OK:
        return {
            "error": "Core dependencies not available",
            "install_command": "pip install segyio numpy"
        }

    try:
        # Quick classification
        classification = quick_classify_survey(file_path, **kwargs)

        # Quick QC if successful
        if classification.get("success", False):
            qc_result = production_segy_qc(file_path, **kwargs)
            if qc_result and "text" in qc_result:
                import json
                qc_data = json.loads(qc_result["text"])
                classification["quality_assessment"] = qc_data.get("overall_assessment", {})

        return classification

    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "file_path": file_path
        }

def batch_analysis(directory_path: str, **kwargs) -> dict:
    """
    Perform batch analysis of all SEG-Y files in a directory

    Args:
        directory_path: Path to directory containing SEG-Y files
        **kwargs: Additional parameters for analysis

    Returns:
        dict: Batch analysis results
    """
    if not _CORE_DEPENDENCIES_OK:
        return {
            "error": "Core dependencies not available",
            "install_command": "pip install segyio numpy"
        }

    try:
        return batch_classify_directory(directory_path, **kwargs)
    except Exception as e:
        return {
            "error": f"Batch analysis failed: {str(e)}",
            "directory": directory_path
        }

# Module-level diagnostics
def run_diagnostics() -> dict:
    """
    Run comprehensive module diagnostics

    Returns:
        dict: Diagnostic results including dependency status and test results
    """
    diagnostics = {
        "module_version": __version__,
        "dependencies": get_dependency_status(),
        "system_info": get_system_info(),
        "tests": {}
    }

    if _CORE_DEPENDENCIES_OK:
        # Test core functionality
        try:
            # Test segyio import
            import segyio
            diagnostics["tests"]["segyio_import"] = "✓ Success"

            # Test module imports
            from production_segy_tools import production_segy_parser
            diagnostics["tests"]["parser_import"] = "✓ Success"

            from survey_classifier import SurveyClassifier
            diagnostics["tests"]["classifier_import"] = "✓ Success"

            from production_segy_analysis_qc import production_segy_qc
            diagnostics["tests"]["qc_import"] = "✓ Success"

            diagnostics["tests"]["overall"] = "✓ All core modules available"

        except Exception as e:
            diagnostics["tests"]["overall"] = f"✗ Module import failed: {str(e)}"
    else:
        diagnostics["tests"]["overall"] = "✗ Core dependencies missing"

    return diagnostics

# Print startup status
if __name__ != "__main__":  # Only when imported, not when run directly
    if _CORE_DEPENDENCIES_OK:
        logger.info(f"MIVAA AgentNexus SEG-Y Module v{__version__} ready (segyio-based)")
    else:
        logger.warning(f"MIVAA AgentNexus SEG-Y Module v{__version__} - dependencies missing")

# Command-line interface for diagnostics
if __name__ == "__main__":
    print(f"MIVAA AgentNexus SEG-Y Processing Module v{__version__}")
    print("=" * 60)

    # Run diagnostics
    diagnostics = run_diagnostics()

    print("Dependencies:")
    for name, status in diagnostics["dependencies"]["dependencies"].items():
        if status["available"]:
            version = status.get("version", "unknown")
            print(f"  ✓ {name}: {version}")
        else:
            print(f"  ✗ {name}: {status['error']}")

    print(f"\nCore Dependencies: {'✓ OK' if _CORE_DEPENDENCIES_OK else '✗ Missing'}")
    print(f"Architecture: {diagnostics['dependencies']['architecture']}")

    if not _CORE_DEPENDENCIES_OK:
        print("\nTo install missing dependencies:")
        print("  pip install segyio numpy scipy scikit-learn")

    print(f"\nModule Tests:")
    for test_name, result in diagnostics["tests"].items():
        print(f"  {result} - {test_name}")

    print(f"\nFor more information:")
    print(f"  python -c \"import {__name__.split('.')[0]}; print({__name__.split('.')[0]}.get_system_info())\"")