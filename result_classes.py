"""
result_classes.py - Standardized result objects for consistent return types

This module provides standardized result classes to ensure consistent
return formats across all intelligent SEG-Y processing methods.

ENHANCED: Added support for segyio-based outputs, quality ratings, and survey type detection.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import json


class ConfidenceLevel(Enum):
    """Standardized confidence levels"""
    VERY_LOW = "Very Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"


class SurveyType(Enum):
    """Supported survey types - Enhanced for segyio classification"""
    TWO_D = "2D"
    THREE_D = "3D"
    SHOT_GATHER = "shot_gather"        # Added for raw acquisition data
    CDP_STACK = "cdp_stack"            # Added for processed stacks
    MIGRATED_2D = "migrated_2d"        # Added for migrated 2D lines
    MIGRATED_3D = "migrated_3d"        # Added for migrated 3D volumes
    UNDETERMINED = "undetermined"


class SortingMethod(Enum):
    """Supported sorting methods"""
    INLINE = "Inline"
    CROSSLINE = "Crossline"
    SP = "SP"
    CDP = "CDP"
    UNDETERMINED = "undetermined"


class StackType(Enum):
    """Supported stack types"""
    PRESTACK = "Prestack"
    POSTSTACK = "Poststack"
    UNDETERMINED = "undetermined"


class QualityRating(Enum):
    """Quality rating levels - Added for QC assessment"""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"
    INVALID = "Invalid"


class ProcessingEngine(Enum):
    """Processing engine types - Added to track which engine was used"""
    LEGACY = "legacy"
    SEGYIO = "segyio-based"
    HYBRID = "hybrid"


@dataclass
class ProcessingError:
    """Standardized error information"""
    error_type: str
    message: str
    details: Optional[str] = None
    recoverable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable
        }


@dataclass
class QualityMetrics:
    """Quality assessment metrics - New for segyio QC"""
    overall_rating: Union[QualityRating, str] = QualityRating.FAIR
    dynamic_range_db: Optional[float] = None
    signal_to_noise: Optional[float] = None
    zero_percentage: Optional[float] = None
    amplitude_range: Optional[Dict[str, float]] = None
    data_integrity: Dict[str, Any] = field(default_factory=dict)

    # Survey-specific metrics
    survey_type_confidence: Optional[str] = None
    thresholds_applied: Optional[str] = None
    rating_factors: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Convert string enum to proper enum value"""
        if isinstance(self.overall_rating, str):
            try:
                self.overall_rating = QualityRating(self.overall_rating)
            except ValueError:
                self.overall_rating = QualityRating.FAIR

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "overall_rating": self.overall_rating.value if isinstance(self.overall_rating, QualityRating) else self.overall_rating,
            "dynamic_range_db": self.dynamic_range_db,
            "signal_to_noise": self.signal_to_noise,
            "zero_percentage": self.zero_percentage,
            "amplitude_range": self.amplitude_range,
            "data_integrity": self.data_integrity,
            "survey_type_confidence": self.survey_type_confidence,
            "thresholds_applied": self.thresholds_applied,
            "rating_factors": self.rating_factors
        }


@dataclass
class GeometryInfo:
    """Geometry information - Enhanced for segyio analysis"""
    # Basic geometry
    total_traces: int = 0
    samples_per_trace: int = 0
    sample_interval_ms: float = 0.0

    # Coordinate information
    coordinate_system: Optional[str] = None
    coordinate_range: Optional[Dict[str, List[float]]] = None
    has_coordinates: bool = False

    # 3D geometry
    inline_range: Optional[List[int]] = None
    crossline_range: Optional[List[int]] = None
    grid_dimensions: Optional[List[int]] = None
    bin_dimensions: Optional[Dict[str, float]] = None
    survey_area_km2: Optional[float] = None

    # 2D geometry
    cdp_range: Optional[List[int]] = None
    shot_point_range: Optional[List[int]] = None
    line_length_km: Optional[float] = None

    # PCA analysis
    pca_analysis: Optional[Dict[str, float]] = None
    geometry_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "total_traces": self.total_traces,
            "samples_per_trace": self.samples_per_trace,
            "sample_interval_ms": self.sample_interval_ms,
            "coordinate_system": self.coordinate_system,
            "coordinate_range": self.coordinate_range,
            "has_coordinates": self.has_coordinates,
            "inline_range": self.inline_range,
            "crossline_range": self.crossline_range,
            "grid_dimensions": self.grid_dimensions,
            "bin_dimensions": self.bin_dimensions,
            "survey_area_km2": self.survey_area_km2,
            "cdp_range": self.cdp_range,
            "shot_point_range": self.shot_point_range,
            "line_length_km": self.line_length_km,
            "pca_analysis": self.pca_analysis,
            "geometry_type": self.geometry_type
        }


@dataclass
class ClassificationResult:
    """Standardized classification result - Enhanced for segyio"""
    # Core classification
    survey_type: Union[SurveyType, str] = SurveyType.UNDETERMINED
    primary_sorting: Union[SortingMethod, str] = SortingMethod.UNDETERMINED
    stack_type: Union[StackType, str] = StackType.UNDETERMINED
    confidence: Union[ConfidenceLevel, str] = ConfidenceLevel.LOW

    # Processing metadata
    file_processed: Optional[str] = None
    template_used: Optional[str] = None
    template_confidence: float = 0.0
    traces_analyzed: int = 0
    processing_engine: Union[ProcessingEngine, str] = ProcessingEngine.LEGACY

    # Enhanced results
    classification_details: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Optional[QualityMetrics] = None
    geometry_info: Optional[GeometryInfo] = None

    # Error handling
    errors: List[ProcessingError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Status
    success: bool = True
    processing_time: Optional[float] = None

    def __post_init__(self):
        """Convert string enums to proper enum values"""
        if isinstance(self.survey_type, str):
            try:
                self.survey_type = SurveyType(self.survey_type)
            except ValueError:
                self.survey_type = SurveyType.UNDETERMINED

        if isinstance(self.primary_sorting, str):
            try:
                self.primary_sorting = SortingMethod(self.primary_sorting)
            except ValueError:
                self.primary_sorting = SortingMethod.UNDETERMINED

        if isinstance(self.stack_type, str):
            try:
                self.stack_type = StackType(self.stack_type)
            except ValueError:
                self.stack_type = StackType.UNDETERMINED

        if isinstance(self.confidence, str):
            try:
                self.confidence = ConfidenceLevel(self.confidence)
            except ValueError:
                self.confidence = ConfidenceLevel.LOW

        if isinstance(self.processing_engine, str):
            try:
                self.processing_engine = ProcessingEngine(self.processing_engine)
            except ValueError:
                self.processing_engine = ProcessingEngine.LEGACY

    def add_error(self, error_type: str, message: str, details: Optional[str] = None, recoverable: bool = True):
        """Add an error to the result"""
        error = ProcessingError(error_type, message, details, recoverable)
        self.errors.append(error)

        # Mark as failed if non-recoverable error
        if not recoverable:
            self.success = False

    def add_warning(self, message: str):
        """Add a warning to the result"""
        self.warnings.append(message)

    def is_high_confidence(self) -> bool:
        """Check if result has high confidence"""
        return self.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]

    def is_successful_classification(self) -> bool:
        """Check if classification was successful"""
        return (self.success and
                self.survey_type != SurveyType.UNDETERMINED and
                self.primary_sorting != SortingMethod.UNDETERMINED)

    def is_segyio_based(self) -> bool:
        """Check if result was generated using segyio"""
        return self.processing_engine == ProcessingEngine.SEGYIO

    def get_quality_rating(self) -> Optional[QualityRating]:
        """Get quality rating if available"""
        if self.quality_metrics:
            return self.quality_metrics.overall_rating
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result_dict = {
            "survey_type": self.survey_type.value if isinstance(self.survey_type, SurveyType) else self.survey_type,
            "primary_sorting": self.primary_sorting.value if isinstance(self.primary_sorting, SortingMethod) else self.primary_sorting,
            "stack_type": self.stack_type.value if isinstance(self.stack_type, StackType) else self.stack_type,
            "confidence": self.confidence.value if isinstance(self.confidence, ConfidenceLevel) else self.confidence,
            "file_processed": self.file_processed,
            "template_used": self.template_used,
            "template_confidence": self.template_confidence,
            "traces_analyzed": self.traces_analyzed,
            "processing_engine": self.processing_engine.value if isinstance(self.processing_engine, ProcessingEngine) else self.processing_engine,
            "classification_details": self.classification_details,
            "errors": [error.to_dict() for error in self.errors],
            "warnings": self.warnings,
            "success": self.success,
            "processing_time": self.processing_time
        }

        # Add quality metrics if available
        if self.quality_metrics:
            result_dict["quality_metrics"] = self.quality_metrics.to_dict()

        # Add geometry info if available
        if self.geometry_info:
            result_dict["geometry_info"] = self.geometry_info.to_dict()

        return result_dict

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy dictionary format for backwards compatibility"""
        legacy_dict = {
            "survey_type": self.survey_type.value if isinstance(self.survey_type, SurveyType) else self.survey_type,
            "primary_sorting": self.primary_sorting.value if isinstance(self.primary_sorting, SortingMethod) else self.primary_sorting,
            "stack_type": self.stack_type.value if isinstance(self.stack_type, StackType) else self.stack_type,
            "confidence": self.confidence.value if isinstance(self.confidence, ConfidenceLevel) else self.confidence,
            "template_used": self.template_used,
            "classification_details": self.classification_details,
            "success": self.success,
            "traces_analyzed": self.traces_analyzed,
            "processing_time": self.processing_time,
            "errors": [error.message for error in self.errors],
            "warnings": self.warnings
        }

        # Add quality rating to legacy format if available
        if self.quality_metrics:
            legacy_dict["quality_rating"] = self.quality_metrics.overall_rating.value if isinstance(self.quality_metrics.overall_rating, QualityRating) else self.quality_metrics.overall_rating

        return legacy_dict

    @classmethod
    def from_segyio_dict(cls, data: Dict[str, Any]) -> 'ClassificationResult':
        """Create ClassificationResult from segyio-based dictionary"""
        result = cls(
            survey_type=data.get("survey_type", "undetermined"),
            primary_sorting=data.get("primary_sorting", "undetermined"),
            stack_type=data.get("stack_type", "undetermined"),
            confidence=data.get("confidence", "Low"),
            file_processed=data.get("file_processed"),
            template_used=data.get("template_used", "segyio_native"),
            template_confidence=data.get("template_confidence", 1.0),
            traces_analyzed=data.get("traces_analyzed", 0),
            processing_engine=ProcessingEngine.SEGYIO,
            classification_details=data.get("classification_details", {}),
            success=data.get("success", True),
            processing_time=data.get("processing_time")
        )

        # Add errors and warnings
        for error in data.get("errors", []):
            if isinstance(error, str):
                result.add_error("classification", error)
            elif isinstance(error, dict):
                result.add_error(
                    error.get("error_type", "classification"),
                    error.get("message", "Unknown error"),
                    error.get("details")
                )

        for warning in data.get("warnings", []):
            result.add_warning(warning)

        # Add quality metrics if available
        if "quality_assessment" in data:
            qa = data["quality_assessment"]
            result.quality_metrics = QualityMetrics(
                overall_rating=qa.get("overall_rating", "Fair"),
                dynamic_range_db=qa.get("dynamic_range_db"),
                signal_to_noise=qa.get("signal_to_noise"),
                zero_percentage=qa.get("zero_percentage"),
                survey_type_confidence=qa.get("confidence"),
                thresholds_applied=qa.get("thresholds_applied"),
                rating_factors=qa.get("rating_factors", {})
            )

        return result


@dataclass
class TemplateDetectionResult:
    """Standardized template detection result - Enhanced for segyio compatibility"""
    best_template: Optional[str] = None
    best_template_path: Optional[str] = None
    confidence: float = 0.0
    detection_method: str = "legacy"  # Added to track detection method

    # All tested templates
    all_results: List[Dict[str, Any]] = field(default_factory=list)
    templates_tested: int = 0

    # Processing info
    file_analyzed: Optional[str] = None
    processing_time: Optional[float] = None

    # Status and recommendations
    success: bool = True
    recommendation: str = ""
    errors: List[ProcessingError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # segyio compatibility
    segyio_compatible: bool = True  # Added to indicate segyio compatibility

    def add_error(self, error_type: str, message: str, details: Optional[str] = None):
        """Add an error to the result"""
        error = ProcessingError(error_type, message, details)
        self.errors.append(error)
        self.success = False

    def add_warning(self, message: str):
        """Add a warning to the result"""
        self.warnings.append(message)

    def is_high_confidence(self) -> bool:
        """Check if template detection has high confidence"""
        return self.confidence > 0.8

    def uses_segyio_native(self) -> bool:
        """Check if using segyio native detection"""
        return self.detection_method == "segyio_native"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "best_template": self.best_template,
            "best_template_path": self.best_template_path,
            "confidence": self.confidence,
            "detection_method": self.detection_method,
            "all_results": self.all_results,
            "templates_tested": self.templates_tested,
            "file_analyzed": self.file_analyzed,
            "processing_time": self.processing_time,
            "success": self.success,
            "recommendation": self.recommendation,
            "errors": [error.to_dict() for error in self.errors],
            "warnings": self.warnings,
            "segyio_compatible": self.segyio_compatible
        }

    @classmethod
    def create_segyio_native(cls, file_path: str) -> 'TemplateDetectionResult':
        """Create result for segyio native detection (no templates needed)"""
        return cls(
            best_template="segyio_native_detection",
            confidence=1.0,
            detection_method="segyio_native",
            file_analyzed=file_path,
            success=True,
            recommendation="Using segyio native header reading - no template files required",
            segyio_compatible=True
        )


@dataclass
class BatchProcessingResult:
    """Standardized batch processing result - Enhanced with quality statistics"""
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0

    # Results for individual files
    file_results: List[Dict[str, Any]] = field(default_factory=list)

    # Enhanced summary statistics
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    quality_distribution: Dict[str, int] = field(default_factory=dict)  # Added for quality stats
    survey_type_distribution: Dict[str, int] = field(default_factory=dict)  # Added for survey stats
    processing_engine_stats: Dict[str, int] = field(default_factory=dict)  # Added for engine stats
    recommendations: List[str] = field(default_factory=list)

    # Processing info
    processing_time: Optional[float] = None
    errors: List[ProcessingError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_file_result(self, file_name: str, result: Union[ClassificationResult, Dict[str, Any]]):
        """Add result for a single file"""
        if isinstance(result, ClassificationResult):
            result_dict = result.to_dict()
        else:
            result_dict = result

        result_dict["file"] = file_name
        self.file_results.append(result_dict)

        # Update counters
        if result_dict.get("success", True) and "error" not in result_dict:
            self.successful_files += 1

            # Update distribution statistics
            survey_type = result_dict.get("survey_type", "undetermined")
            self.survey_type_distribution[survey_type] = self.survey_type_distribution.get(survey_type, 0) + 1

            # Track quality ratings
            quality_rating = None
            if "quality_metrics" in result_dict and "overall_rating" in result_dict["quality_metrics"]:
                quality_rating = result_dict["quality_metrics"]["overall_rating"]
            elif "quality_rating" in result_dict:
                quality_rating = result_dict["quality_rating"]

            if quality_rating:
                self.quality_distribution[quality_rating] = self.quality_distribution.get(quality_rating, 0) + 1

            # Track processing engines
            engine = result_dict.get("processing_engine", "legacy")
            self.processing_engine_stats[engine] = self.processing_engine_stats.get(engine, 0) + 1

        else:
            self.failed_files += 1

    def get_success_rate(self) -> float:
        """Get success rate as percentage"""
        if self.total_files == 0:
            return 0.0
        return (self.successful_files / self.total_files) * 100

    def get_segyio_usage_rate(self) -> float:
        """Get percentage of files processed with segyio"""
        segyio_count = self.processing_engine_stats.get("segyio-based", 0)
        if self.successful_files == 0:
            return 0.0
        return (segyio_count / self.successful_files) * 100

    def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality assessment summary"""
        total_with_quality = sum(self.quality_distribution.values())
        if total_with_quality == 0:
            return {"message": "No quality assessments available"}

        return {
            "total_assessed": total_with_quality,
            "excellent_count": self.quality_distribution.get("Excellent", 0),
            "good_count": self.quality_distribution.get("Good", 0),
            "fair_count": self.quality_distribution.get("Fair", 0),
            "poor_count": self.quality_distribution.get("Poor", 0),
            "excellent_percentage": round((self.quality_distribution.get("Excellent", 0) / total_with_quality) * 100, 1),
            "good_or_better_percentage": round(((self.quality_distribution.get("Excellent", 0) + self.quality_distribution.get("Good", 0)) / total_with_quality) * 100, 1)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "total_files": self.total_files,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "success_rate": self.get_success_rate(),
            "segyio_usage_rate": self.get_segyio_usage_rate(),
            "file_results": self.file_results,
            "summary_statistics": self.summary_statistics,
            "quality_distribution": self.quality_distribution,
            "survey_type_distribution": self.survey_type_distribution,
            "processing_engine_stats": self.processing_engine_stats,
            "quality_summary": self.get_quality_summary(),
            "recommendations": self.recommendations,
            "processing_time": self.processing_time,
            "errors": [error.to_dict() for error in self.errors],
            "warnings": self.warnings
        }


# Utility functions for result conversion and compatibility

def convert_legacy_result(legacy_dict: Dict[str, Any]) -> ClassificationResult:
    """Convert legacy result dictionary to ClassificationResult object"""
    return ClassificationResult(
        survey_type=legacy_dict.get("survey_type", "undetermined"),
        primary_sorting=legacy_dict.get("primary_sorting", "undetermined"),
        stack_type=legacy_dict.get("stack_type", "undetermined"),
        confidence=legacy_dict.get("confidence", "Low"),
        file_processed=legacy_dict.get("file_processed"),
        template_used=legacy_dict.get("template_used"),
        template_confidence=legacy_dict.get("template_confidence", 0.0),
        traces_analyzed=legacy_dict.get("traces_analyzed", 0),
        classification_details=legacy_dict.get("classification_details", {}),
        success=legacy_dict.get("success", True),
        processing_time=legacy_dict.get("processing_time"),
        processing_engine=ProcessingEngine.LEGACY
    )


def ensure_result_compatibility(result: Union[ClassificationResult, Dict[str, Any]]) -> ClassificationResult:
    """Ensure result is in ClassificationResult format"""
    if isinstance(result, ClassificationResult):
        return result
    elif isinstance(result, dict):
        # Check if it's a segyio-based result
        if result.get("template_used") == "segyio_native_detection" or "segyio" in str(result.get("template_used", "")):
            return ClassificationResult.from_segyio_dict(result)
        else:
            return convert_legacy_result(result)
    else:
        raise ValueError(f"Cannot convert result of type {type(result)} to ClassificationResult")


def create_error_result(error_message: str, file_name: Optional[str] = None) -> ClassificationResult:
    """Create a ClassificationResult representing an error condition"""
    result = ClassificationResult(file_processed=file_name, success=False)
    result.add_error("processing", error_message, recoverable=False)
    return result