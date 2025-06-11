"""
survey_classifier.py - Main classification engine for SEG-Y survey characterization

This module provides the main classification engine that combines all algorithms
to automatically determine survey type, sorting method, and stacking type.

UPDATED: Now uses segyio as the core engine for maximum reliability and accuracy.
Maintains 100% API compatibility with existing MCP tools.

FINAL VERSION - All methods properly implemented and tested.
"""

import os
import time
import math
import logging
from typing import Dict, Any, Optional, List, Tuple
import segyio
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class SurveyType(Enum):
    """Survey type enumeration"""
    SURVEY_2D = "2D"
    SURVEY_3D = "3D"
    UNDETERMINED = "undetermined"

class SortingMethod(Enum):
    """Primary sorting method enumeration"""
    INLINE = "Inline"
    CROSSLINE = "Crossline"
    CDP = "CDP"
    SP = "SP"
    UNDETERMINED = "undetermined"

class StackType(Enum):
    """Stack type enumeration"""
    PRESTACK = "Prestack"
    POSTSTACK = "Poststack"
    UNDETERMINED = "undetermined"

class ConfidenceLevel(Enum):
    """Confidence level enumeration"""
    VERY_LOW = "Very Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"

class SegyioSurveyClassifier:
    """
    Main classification engine using segyio for SEG-Y survey characterization
    Maintains backward compatibility with original SurveyClassifier API
    """

    def __init__(self, template_directory: str = "./templates"):
        """
        Initialize survey classifier

        Args:
            template_directory: Directory containing template files (maintained for compatibility)
        """
        self.template_directory = template_directory

        # Classification thresholds - calibrated for real-world data
        self.thresholds = {
            "pca_2d_threshold": 0.9998,  # major_var > 0.9998 = 2D line
            "pca_3d_threshold": 0.9990,  # major_var < 0.9990 = 3D areal (slightly more lenient)
            "min_traces_for_analysis": 50,  # Reduced for faster processing
            "high_confidence_threshold": 0.8,
            "medium_confidence_threshold": 0.6,
            "template_confidence_weight": 0.2,  # Reduced since we don't use templates
            "gather_consistency_threshold": 0.8,  # For prestack detection
            "coordinate_variation_threshold": 1000,  # Minimum coordinate variation
            "min_unique_values": 3  # Minimum unique values to consider a field valid
        }

    def classify_survey(self, segy_file_path: str,
                       template_path: Optional[str] = None,
                       max_traces: int = 5000) -> Dict[str, Any]:
        """
        Main classification method - determines survey characteristics

        Args:
            segy_file_path: Path to SEG-Y file
            template_path: Optional specific template to use (ignored, maintained for compatibility)
            max_traces: Maximum number of traces to analyze

        Returns:
            Dict: Standardized classification results (same format as original)
        """
        start_time = time.time()
        result = {
            "file_processed": os.path.basename(segy_file_path),
            "success": False,
            "survey_type": "undetermined",
            "primary_sorting": "undetermined",
            "stack_type": "undetermined",
            "confidence": "Very Low",
            "template_used": "segyio_native_detection",
            "template_confidence": 1.0,  # Always high since we use segyio
            "traces_analyzed": 0,
            "processing_time": 0.0,
            "classification_details": {},
            "errors": [],
            "warnings": []
        }

        try:
            logger.info(f"Classifying survey: {result['file_processed']}")

            # Step 1: Validate file and extract header data
            header_data = self._extract_header_data(segy_file_path, max_traces, result)
            if not header_data:
                result["processing_time"] = time.time() - start_time
                return result

            result["traces_analyzed"] = header_data["traces_analyzed"]
            result["template_used"] = "segyio_native_detection"
            result["template_confidence"] = 1.0

            # Step 2: Perform classification analysis
            self._perform_segyio_classification(header_data, result)

            # Step 3: Calculate final confidence
            self._calculate_confidence_segyio(result)

            result["success"] = True
            result["processing_time"] = time.time() - start_time

            logger.info(f"Classification complete: {result['survey_type']}/{result['primary_sorting']}/{result['stack_type']} (confidence: {result['confidence']})")
            return result

        except Exception as e:
            result["errors"].append(f"Classification failed: {str(e)}")
            result["processing_time"] = time.time() - start_time
            logger.error(f"Error classifying survey {segy_file_path}: {str(e)}")
            return result

    def _extract_header_data(self, file_path: str, max_traces: int, result: Dict) -> Optional[Dict]:
        """Extract header data using segyio"""
        try:
            if not os.path.isfile(file_path):
                result["errors"].append(f"File not found: {file_path}")
                return None

            with segyio.open(file_path, ignore_geometry=True) as f:
                total_traces = f.tracecount

                # Determine sampling strategy
                if total_traces <= max_traces:
                    # Use all traces
                    trace_indices = list(range(total_traces))
                else:
                    # Sample traces evenly
                    step = max(1, total_traces // max_traces)
                    trace_indices = list(range(0, total_traces, step))[:max_traces]

                # Extract header information
                header_data = {
                    "traces_analyzed": len(trace_indices),
                    "total_traces": total_traces,
                    "inlines": [],
                    "crosslines": [],
                    "cdp_values": [],
                    "sp_values": [],
                    "field_records": [],
                    "x_coords": [],
                    "y_coords": [],
                    "trace_numbers": [],
                    "offsets": [],
                    "file_info": {
                        "sample_rate": f.bin[segyio.BinField.Interval] / 1000,  # Convert to ms
                        "samples_per_trace": f.bin[segyio.BinField.Samples],
                        "format_code": f.bin[segyio.BinField.Format]
                    }
                }

                # Extract headers for sampled traces
                for i in trace_indices:
                    try:
                        header = f.header[i]

                        header_data["inlines"].append(header[segyio.TraceField.INLINE_3D])
                        header_data["crosslines"].append(header[segyio.TraceField.CROSSLINE_3D])
                        header_data["cdp_values"].append(header[segyio.TraceField.CDP])
                        header_data["sp_values"].append(header[segyio.TraceField.TRACE_SEQUENCE_FILE])
                        header_data["field_records"].append(header[segyio.TraceField.FieldRecord])
                        header_data["x_coords"].append(header[segyio.TraceField.CDP_X])
                        header_data["y_coords"].append(header[segyio.TraceField.CDP_Y])
                        header_data["trace_numbers"].append(header[segyio.TraceField.TraceNumber])
                        header_data["offsets"].append(header[segyio.TraceField.offset])

                    except Exception as e:
                        logger.debug(f"Error reading header for trace {i}: {e}")
                        continue

                # Calculate statistics for each field
                self._calculate_header_statistics(header_data)

                return header_data

        except segyio.exceptions.InvalidError:
            result["errors"].append("Invalid SEG-Y format")
            return None
        except Exception as e:
            result["errors"].append(f"Error reading file: {str(e)}")
            return None

    def _calculate_header_statistics(self, header_data: Dict):
        """Calculate statistics for header fields"""
        for field in ["inlines", "crosslines", "cdp_values", "sp_values", "field_records"]:
            values = header_data[field]
            non_zero_values = [v for v in values if v != 0]

            header_data[f"{field}_stats"] = {
                "total_count": len(values),
                "non_zero_count": len(non_zero_values),
                "unique_count": len(set(non_zero_values)) if non_zero_values else 0,
                "min_value": min(non_zero_values) if non_zero_values else 0,
                "max_value": max(non_zero_values) if non_zero_values else 0,
                "has_variation": len(set(non_zero_values)) > 1 if non_zero_values else False
            }

        # Coordinate statistics
        x_coords = [v for v in header_data["x_coords"] if v != 0]
        y_coords = [v for v in header_data["y_coords"] if v != 0]

        if x_coords and y_coords:
            header_data["coordinate_stats"] = {
                "has_coordinates": True,
                "x_range": max(x_coords) - min(x_coords),
                "y_range": max(y_coords) - min(y_coords),
                "coordinate_pairs": len(x_coords)
            }
        else:
            header_data["coordinate_stats"] = {"has_coordinates": False}

    def _perform_segyio_classification(self, header_data: Dict, result: Dict):
        """Perform classification analysis using segyio-extracted data"""

        # Step 1: Try 3D survey detection
        if self._classify_3d_survey_segyio(header_data, result):
            return

        # Step 2: Try shot gather detection
        if self._classify_shot_gathers_segyio(header_data, result):
            return

        # Step 3: Try 2D survey detection
        if self._classify_2d_survey_segyio(header_data, result):
            return

        # Step 4: Try prestack gather detection
        if self._classify_prestack_gathers_segyio(header_data, result):
            return

        # Step 5: Fallback to educated guess
        self._make_educated_guess_segyio(header_data, result)

    def _classify_3d_survey_segyio(self, header_data: Dict, result: Dict) -> bool:
        """Classify 3D surveys using segyio data"""
        try:
            inline_stats = header_data["inlines_stats"]
            xline_stats = header_data["crosslines_stats"]
            coord_stats = header_data["coordinate_stats"]

            # Check for 3D inline/crossline pattern
            has_inlines = inline_stats["unique_count"] >= self.thresholds["min_unique_values"]
            has_crosslines = xline_stats["unique_count"] >= self.thresholds["min_unique_values"]

            if has_inlines and has_crosslines:
                logger.info("3D inline/crossline pattern detected")

                # Determine sorting method
                if inline_stats["unique_count"] > xline_stats["unique_count"]:
                    primary_sorting = SortingMethod.INLINE.value
                elif xline_stats["unique_count"] > inline_stats["unique_count"]:
                    primary_sorting = SortingMethod.CROSSLINE.value
                else:
                    primary_sorting = SortingMethod.INLINE.value  # Default

                # Verify with coordinate analysis if available
                coordinate_confidence = "medium"
                pca_analysis = {}

                if coord_stats["has_coordinates"]:
                    x_coords = [v for v in header_data["x_coords"] if v != 0]
                    y_coords = [v for v in header_data["y_coords"] if v != 0]

                    if len(x_coords) > 2 and len(y_coords) > 2:
                        major_var, minor_var = self._calculate_pca_variance(x_coords, y_coords)
                        pca_analysis = {
                            "major_variance": major_var,
                            "minor_variance": minor_var
                        }

                        if major_var < self.thresholds["pca_3d_threshold"]:
                            coordinate_confidence = "high"
                        else:
                            coordinate_confidence = "low"

                # Determine stack type based on trace organization
                stack_type = self._determine_stack_type_from_organization(header_data)

                result.update({
                    "survey_type": SurveyType.SURVEY_3D.value,
                    "primary_sorting": primary_sorting,
                    "stack_type": stack_type,
                    "confidence": ConfidenceLevel.HIGH.value if coordinate_confidence == "high" else ConfidenceLevel.MEDIUM.value
                })

                result["classification_details"] = {
                    "method": "3D_inline_crossline_segyio",
                    "inline_count": inline_stats["unique_count"],
                    "crossline_count": xline_stats["unique_count"],
                    "inline_range": [inline_stats["min_value"], inline_stats["max_value"]],
                    "crossline_range": [xline_stats["min_value"], xline_stats["max_value"]],
                    "coordinate_confidence": coordinate_confidence,
                    "pca_analysis": pca_analysis
                }

                return True

            return False

        except Exception as e:
            logger.error(f"Error in 3D classification: {str(e)}")
            return False

    def _classify_shot_gathers_segyio(self, header_data: Dict, result: Dict) -> bool:
        """Classify shot gathers using segyio data"""
        try:
            field_record_stats = header_data["field_records_stats"]
            cdp_stats = header_data["cdp_values_stats"]
            coord_stats = header_data["coordinate_stats"]

            # Check for shot gather pattern
            has_field_records = field_record_stats["unique_count"] >= self.thresholds["min_unique_values"]
            low_cdp_variation = cdp_stats["unique_count"] <= 10

            if has_field_records and low_cdp_variation:
                logger.info("Shot gather pattern detected")

                # Determine if 2D or 3D based on coordinate spread
                survey_dimension = SurveyType.UNDETERMINED.value

                if coord_stats["has_coordinates"]:
                    x_coords = [v for v in header_data["x_coords"] if v != 0]
                    y_coords = [v for v in header_data["y_coords"] if v != 0]

                    if len(x_coords) > 2 and len(y_coords) > 2:
                        major_var, minor_var = self._calculate_pca_variance(x_coords, y_coords)

                        if major_var > self.thresholds["pca_2d_threshold"]:
                            survey_dimension = SurveyType.SURVEY_2D.value
                        elif major_var < self.thresholds["pca_3d_threshold"]:
                            survey_dimension = SurveyType.SURVEY_3D.value

                result.update({
                    "survey_type": survey_dimension,
                    "primary_sorting": SortingMethod.SP.value,
                    "stack_type": StackType.PRESTACK.value,
                    "confidence": ConfidenceLevel.HIGH.value if survey_dimension != "undetermined" else ConfidenceLevel.MEDIUM.value
                })

                result["classification_details"] = {
                    "method": "shot_gather_segyio",
                    "field_record_count": field_record_stats["unique_count"],
                    "field_record_range": [field_record_stats["min_value"], field_record_stats["max_value"]],
                    "cdp_variation": cdp_stats["unique_count"],
                    "dimension_confidence": "high" if survey_dimension != "undetermined" else "low"
                }

                return True

            return False

        except Exception as e:
            logger.error(f"Error in shot gather classification: {e}")
            return False

    def _classify_2d_survey_segyio(self, header_data: Dict, result: Dict) -> bool:
        """Classify 2D surveys using segyio data"""
        try:
            cdp_stats = header_data["cdp_values_stats"]
            sp_stats = header_data["sp_values_stats"]
            coord_stats = header_data["coordinate_stats"]

            # Check for 2D CDP pattern
            has_cdps = cdp_stats["unique_count"] >= self.thresholds["min_unique_values"]
            has_sps = sp_stats["unique_count"] >= self.thresholds["min_unique_values"]

            if has_cdps or has_sps:
                logger.info("2D pattern detected")

                # Verify with coordinate analysis
                coordinate_verification = False
                pca_analysis = {}

                if coord_stats["has_coordinates"]:
                    x_coords = [v for v in header_data["x_coords"] if v != 0]
                    y_coords = [v for v in header_data["y_coords"] if v != 0]

                    if len(x_coords) > 2 and len(y_coords) > 2:
                        major_var, minor_var = self._calculate_pca_variance(x_coords, y_coords)
                        pca_analysis = {
                            "major_variance": major_var,
                            "minor_variance": minor_var
                        }

                        if major_var > self.thresholds["pca_2d_threshold"]:
                            coordinate_verification = True

                # Determine sorting preference
                if has_cdps and cdp_stats["unique_count"] > sp_stats["unique_count"]:
                    primary_sorting = SortingMethod.CDP.value
                elif has_sps:
                    primary_sorting = SortingMethod.SP.value
                else:
                    primary_sorting = SortingMethod.CDP.value

                # Determine stack type
                stack_type = self._determine_stack_type_from_organization(header_data)

                confidence = ConfidenceLevel.HIGH.value if coordinate_verification else ConfidenceLevel.MEDIUM.value

                result.update({
                    "survey_type": SurveyType.SURVEY_2D.value,
                    "primary_sorting": primary_sorting,
                    "stack_type": stack_type,
                    "confidence": confidence
                })

                result["classification_details"] = {
                    "method": "2D_CDP_SP_segyio",
                    "cdp_count": cdp_stats["unique_count"],
                    "sp_count": sp_stats["unique_count"],
                    "cdp_range": [cdp_stats["min_value"], cdp_stats["max_value"]] if has_cdps else None,
                    "sp_range": [sp_stats["min_value"], sp_stats["max_value"]] if has_sps else None,
                    "coordinate_verification": coordinate_verification,
                    "pca_analysis": pca_analysis
                }

                return True

            return False

        except Exception as e:
            logger.error(f"Error in 2D classification: {e}")
            return False

    def _classify_prestack_gathers_segyio(self, header_data: Dict, result: Dict) -> bool:
        """Classify prestack gathers using segyio data"""
        try:
            # Check for gather patterns by looking at trace organization
            cdp_values = header_data["cdp_values"]
            offsets = header_data["offsets"]

            # Check for CDP gathers (same CDP, varying offsets)
            if len(set(cdp_values)) < len(cdp_values) * 0.5:  # Significant repetition
                offset_variation = len(set([o for o in offsets if o != 0]))

                if offset_variation > self.thresholds["min_unique_values"]:
                    logger.info("CDP prestack gathers detected")

                    result.update({
                        "survey_type": SurveyType.UNDETERMINED.value,  # Would need coordinate analysis
                        "primary_sorting": SortingMethod.CDP.value,
                        "stack_type": StackType.PRESTACK.value,
                        "confidence": ConfidenceLevel.MEDIUM.value
                    })

                    result["classification_details"] = {
                        "method": "prestack_cdp_gathers_segyio",
                        "gather_organization": "CDP gathers with varying offsets",
                        "unique_cdps": len(set(cdp_values)),
                        "unique_offsets": offset_variation
                    }

                    return True

            return False

        except Exception as e:
            logger.error(f"Error in prestack classification: {e}")
            return False

    def _make_educated_guess_segyio(self, header_data: Dict, result: Dict):
        """Make educated guess using segyio data when no clear pattern is found"""
        try:
            # Analyze available header information
            inline_stats = header_data["inlines_stats"]
            xline_stats = header_data["crosslines_stats"]
            cdp_stats = header_data["cdp_values_stats"]
            sp_stats = header_data["sp_values_stats"]
            field_stats = header_data["field_records_stats"]

            classification_evidence = []

            # Check what headers have meaningful data
            if inline_stats["has_variation"] or xline_stats["has_variation"]:
                survey_type = SurveyType.SURVEY_3D.value
                primary_sorting = SortingMethod.INLINE.value if inline_stats["unique_count"] > xline_stats["unique_count"] else SortingMethod.CROSSLINE.value
                classification_evidence.append("3D headers present")

            elif cdp_stats["has_variation"] or sp_stats["has_variation"]:
                survey_type = SurveyType.SURVEY_2D.value
                primary_sorting = SortingMethod.CDP.value if cdp_stats["unique_count"] > sp_stats["unique_count"] else SortingMethod.SP.value
                classification_evidence.append("2D headers present")

            elif field_stats["has_variation"]:
                survey_type = SurveyType.UNDETERMINED.value
                primary_sorting = SortingMethod.SP.value
                classification_evidence.append("Field record variation suggests shot data")

            else:
                survey_type = SurveyType.UNDETERMINED.value
                primary_sorting = SortingMethod.UNDETERMINED.value
                classification_evidence.append("Insufficient header variation")

            # Default stack type guess
            stack_type = StackType.POSTSTACK.value

            result.update({
                "survey_type": survey_type,
                "primary_sorting": primary_sorting,
                "stack_type": stack_type,
                "confidence": ConfidenceLevel.LOW.value
            })

            result["classification_details"] = {
                "method": "educated_guess_segyio",
                "evidence": classification_evidence,
                "header_analysis": {
                    "inlines_unique": inline_stats["unique_count"],
                    "crosslines_unique": xline_stats["unique_count"],
                    "cdps_unique": cdp_stats["unique_count"],
                    "sps_unique": sp_stats["unique_count"],
                    "field_records_unique": field_stats["unique_count"]
                },
                "note": "Classification based on available header information with low confidence"
            }

        except Exception as e:
            logger.error(f"Error in educated guess: {e}")
            result.update({
                "survey_type": SurveyType.UNDETERMINED.value,
                "primary_sorting": SortingMethod.UNDETERMINED.value,
                "stack_type": StackType.UNDETERMINED.value,
                "confidence": ConfidenceLevel.VERY_LOW.value
            })

    def _determine_stack_type_from_organization(self, header_data: Dict) -> str:
        """Determine stack type based on trace organization"""
        try:
            # Simple heuristic: if offsets vary significantly, likely prestack
            offsets = [o for o in header_data["offsets"] if o != 0]

            if len(set(offsets)) > len(offsets) * 0.1:  # More than 10% unique offsets
                return StackType.PRESTACK.value
            else:
                return StackType.POSTSTACK.value

        except Exception:
            return StackType.POSTSTACK.value  # Default assumption

    def _calculate_pca_variance(self, x_coords: List[float], y_coords: List[float]) -> Tuple[float, float]:
        """Calculate PCA variance for coordinate analysis"""
        try:
            # Create coordinate matrix
            coords = np.column_stack((x_coords, y_coords))

            # Center the data
            coords_centered = coords - np.mean(coords, axis=0)

            # Calculate covariance matrix
            cov_matrix = np.cov(coords_centered.T)

            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

            # Calculate variance ratios
            total_variance = np.sum(eigenvalues)
            if total_variance > 0:
                major_var = eigenvalues[0] / total_variance
                minor_var = eigenvalues[1] / total_variance if len(eigenvalues) > 1 else 0
            else:
                major_var = 1.0
                minor_var = 0.0

            return float(major_var), float(minor_var)

        except Exception as e:
            logger.debug(f"PCA calculation failed: {e}")
            return 1.0, 0.0  # Default to 2D-like

    def _calculate_confidence_segyio(self, result: Dict):
        """Calculate final confidence based on classification evidence"""
        try:
            base_confidence = result.get("confidence", ConfidenceLevel.VERY_LOW.value)

            # Factors that increase confidence
            details = result.get("classification_details", {})
            confidence_factors = []

            # Strong method indicators
            method = details.get("method", "")
            if "3D_inline_crossline" in method:
                confidence_factors.append("Strong 3D pattern")
            elif "shot_gather" in method:
                confidence_factors.append("Clear shot gather pattern")
            elif "2D_CDP" in method:
                confidence_factors.append("Clear 2D pattern")

            # Coordinate verification
            if details.get("coordinate_verification") or details.get("coordinate_confidence") == "high":
                confidence_factors.append("Coordinate verification")

            # Sufficient data
            traces_analyzed = result.get("traces_analyzed", 0)
            if traces_analyzed > 1000:
                confidence_factors.append("Large sample size")
            elif traces_analyzed > 100:
                confidence_factors.append("Adequate sample size")

            # Adjust confidence based on factors
            if len(confidence_factors) >= 3:
                final_confidence = ConfidenceLevel.VERY_HIGH.value
            elif len(confidence_factors) == 2:
                final_confidence = ConfidenceLevel.HIGH.value
            elif len(confidence_factors) == 1:
                final_confidence = ConfidenceLevel.MEDIUM.value if base_confidence != ConfidenceLevel.LOW.value else ConfidenceLevel.LOW.value
            else:
                final_confidence = base_confidence

            result["confidence"] = final_confidence
            result["classification_details"]["confidence_factors"] = confidence_factors

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            result["confidence"] = ConfidenceLevel.VERY_LOW.value

    def classify_multiple_files(self, file_paths: List[str],
                               max_traces_per_file: int = 2000) -> Dict[str, Any]:
        """
        Classify multiple SEG-Y files in batch
        Maintains compatibility with original SurveyClassifier API
        """
        try:
            logger.info(f"Starting batch classification of {len(file_paths)} files")

            results = []
            survey_types = {}
            sorting_methods = {}
            stack_types = {}

            for file_path in file_paths:
                try:
                    classification = self.classify_survey(file_path, max_traces=max_traces_per_file)

                    # Track statistics
                    survey_type = classification.get("survey_type", "undetermined")
                    primary_sorting = classification.get("primary_sorting", "undetermined")
                    stack_type = classification.get("stack_type", "undetermined")

                    survey_types[survey_type] = survey_types.get(survey_type, 0) + 1
                    sorting_methods[primary_sorting] = sorting_methods.get(primary_sorting, 0) + 1
                    stack_types[stack_type] = stack_types.get(stack_type, 0) + 1

                    results.append({
                        "file": os.path.basename(file_path),
                        "classification": classification
                    })

                except Exception as e:
                    results.append({
                        "file": os.path.basename(file_path),
                        "error": str(e)
                    })

            # Generate batch analysis
            successful_classifications = [r for r in results if "error" not in r]

            batch_results = {
                "total_files": len(file_paths),
                "successful_classifications": len(successful_classifications),
                "failed_classifications": len(file_paths) - len(successful_classifications),
                "survey_type_distribution": survey_types,
                "sorting_method_distribution": sorting_methods,
                "stack_type_distribution": stack_types,
                "file_results": results,
                "batch_recommendations": self._generate_batch_recommendations(
                    survey_types, sorting_methods, stack_types, len(successful_classifications)
                )
            }

            logger.info(f"Batch classification completed: {len(successful_classifications)}/{len(file_paths)} successful")
            return batch_results

        except Exception as e:
            logger.error(f"Batch classification failed: {str(e)}")
            return {"error": f"Batch classification failed: {str(e)}"}

    def _generate_batch_recommendations(self, survey_types: Dict[str, int],
                                      sorting_methods: Dict[str, int],
                                      stack_types: Dict[str, int],
                                      total_successful: int) -> List[str]:
        """Generate recommendations for batch classification results"""
        recommendations = []

        try:
            if total_successful == 0:
                recommendations.append("No files could be classified - check file formats and accessibility")
                return recommendations

            # Analyze survey type consistency
            if len(survey_types) == 1:
                survey_type = list(survey_types.keys())[0]
                if survey_type != "undetermined":
                    recommendations.append(f"All files are {survey_type} surveys - consistent survey type")
                else:
                    recommendations.append("All files have undetermined survey type - review file quality and header completeness")
            else:
                mixed_types = [k for k, v in survey_types.items() if k != "undetermined"]
                if len(mixed_types) > 1:
                    recommendations.append(f"Mixed survey types detected: {', '.join(mixed_types)} - verify survey consistency")

            # Analyze sorting consistency
            if len(sorting_methods) == 1:
                sorting = list(sorting_methods.keys())[0]
                if sorting != "undetermined":
                    recommendations.append(f"Consistent sorting method: {sorting}")
            else:
                recommendations.append("Multiple sorting methods detected - may need different processing parameters")

            # Analyze stack type consistency
            if len(stack_types) == 1:
                stack = list(stack_types.keys())[0]
                if stack != "undetermined":
                    recommendations.append(f"Consistent stack type: {stack}")
            else:
                recommendations.append("Mixed stack types detected - separate prestack and poststack processing recommended")

            # Confidence analysis
            undetermined_count = survey_types.get("undetermined", 0)
            if undetermined_count > total_successful * 0.3:
                recommendations.append(f"{undetermined_count} files could not be clearly classified - manual review recommended")

            return recommendations

        except Exception as e:
            return [f"Error generating recommendations: {str(e)}"]

    def generate_classification_report(self, classification_result: Dict[str, Any]) -> str:
        """
        Generate human-readable classification report
        Maintains compatibility with original SurveyClassifier API
        """
        try:
            if "error" in classification_result or not classification_result.get("success", False):
                errors = classification_result.get("errors", ["Unknown error"])
                return f"Classification Error: {'; '.join(errors)}"

            report = []
            report.append("SEG-Y Survey Classification Report")
            report.append("=" * 50)

            # Basic information
            report.append(f"File: {classification_result.get('file_processed', 'Unknown')}")
            report.append(f"Classification Engine: segyio-based (v2.0)")
            report.append(f"Template Used: {classification_result.get('template_used', 'Unknown')}")
            report.append(f"Template Confidence: {classification_result.get('template_confidence', 0):.3f}")
            report.append(f"Traces Analyzed: {classification_result.get('traces_analyzed', 0)}")
            report.append(f"Processing Time: {classification_result.get('processing_time', 0):.2f}s")
            report.append("")

            # Classification results
            report.append("CLASSIFICATION RESULTS:")
            report.append("-" * 30)
            report.append(f"Survey Type: {classification_result.get('survey_type', 'Unknown')}")
            report.append(f"Primary Sorting: {classification_result.get('primary_sorting', 'Unknown')}")
            report.append(f"Stack Type: {classification_result.get('stack_type', 'Unknown')}")
            report.append(f"Confidence: {classification_result.get('confidence', 'Unknown')}")
            report.append("")

            # Classification details
            details = classification_result.get('classification_details', {})
            if details:
                report.append("CLASSIFICATION DETAILS:")
                report.append("-" * 30)

                method = details.get('method', 'Unknown')
                report.append(f"Classification Method: {method}")

                # Add specific details based on method
                if 'inline_count' in details:
                    report.append(f"Inline Count: {details['inline_count']}")
                    report.append(f"Crossline Count: {details['crossline_count']}")

                if 'inline_range' in details and details['inline_range']:
                    report.append(f"Inline Range: {details['inline_range'][0]} to {details['inline_range'][1]}")

                if 'crossline_range' in details and details['crossline_range']:
                    report.append(f"Crossline Range: {details['crossline_range'][0]} to {details['crossline_range'][1]}")

                if 'cdp_range' in details and details['cdp_range']:
                    report.append(f"CDP Range: {details['cdp_range'][0]} to {details['cdp_range'][1]}")

                if 'field_record_count' in details:
                    report.append(f"Field Record Count: {details['field_record_count']}")

                if 'pca_analysis' in details and details['pca_analysis']:
                    pca = details['pca_analysis']
                    if 'major_variance' in pca:
                        report.append(f"PCA Major Variance: {pca['major_variance']:.6f}")
                        report.append(f"PCA Minor Variance: {pca['minor_variance']:.6f}")

                if 'coordinate_verification' in details:
                    report.append(f"Coordinate Verification: {details['coordinate_verification']}")

                if 'confidence_factors' in details:
                    factors = details['confidence_factors']
                    if factors:
                        report.append(f"Confidence Factors: {', '.join(factors)}")

                if 'note' in details:
                    report.append(f"Note: {details['note']}")

                report.append("")

            # Warnings and errors
            warnings = classification_result.get('warnings', [])
            if warnings:
                report.append("WARNINGS:")
                report.append("-" * 15)
                for warning in warnings:
                    report.append(f"⚠ {warning}")
                report.append("")

            # Recommendations
            confidence = classification_result.get('confidence', 'Unknown')
            report.append("RECOMMENDATIONS:")
            report.append("-" * 20)

            if confidence == "Very High":
                report.append("Very high confidence classification - proceed with automatic processing")
                self._add_processing_recommendations(report, classification_result)

            elif confidence == "High":
                report.append("High confidence classification - proceed with automatic processing")
                self._add_processing_recommendations(report, classification_result)

            elif confidence == "Medium":
                report.append("⚠ Medium confidence - review results before processing")
                report.append("⚠ Consider manual verification of survey parameters")
                self._add_processing_recommendations(report, classification_result)

            elif confidence == "Low":
                report.append("⚠ Low confidence - manual review recommended")
                report.append("⚠ Verify file format and header completeness")
                report.append("⚠ Consider analyzing more traces for better accuracy")

            else:
                report.append("Very low confidence - manual intervention required")
                report.append("Check file format compliance")
                report.append("Review header field completeness")
                report.append("Consider file-specific analysis")

            # segyio-specific advantages
            report.append("")
            report.append("SEGYIO ADVANTAGES:")
            report.append("-" * 20)
            report.append("Industry-standard SEG-Y reading")
            report.append("No template file dependencies")
            report.append("Robust error handling")
            report.append("Memory-efficient processing")

            return "\n".join(report)

        except Exception as e:
            return f"Error generating classification report: {str(e)}"

    def _add_processing_recommendations(self, report: List[str], result: Dict):
        """Add processing recommendations to report"""
        survey_type = result.get('survey_type', '')
        sorting = result.get('primary_sorting', '')
        stack = result.get('stack_type', '')

        if survey_type == "3D":
            report.append(f"Use 3D processing parameters with {sorting} sorting")
            report.append("Consider 3D migration and attribute analysis")
        elif survey_type == "2D":
            report.append(f"Use 2D processing parameters with {sorting} sorting")
            report.append("Consider 2D migration and structural interpretation")

        if stack == "Prestack":
            report.append("Configure for prestack processing workflow")
            report.append("Consider gather analysis and velocity analysis")
            report.append("Plan for CMP stacking or prestack migration")
        elif stack == "Poststack":
            report.append("Configure for poststack processing workflow")
            report.append("Ready for interpretation and attribute analysis")
            report.append("Consider poststack migration if not already migrated")

    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get statistics about classification capabilities and thresholds"""
        return {
            "classifier_version": "2.0.0_segyio",
            "engine": "segyio-based",
            "supported_survey_types": [e.value for e in SurveyType],
            "supported_sorting_methods": [e.value for e in SortingMethod],
            "supported_stack_types": [e.value for e in StackType],
            "confidence_levels": [e.value for e in ConfidenceLevel],
            "classification_thresholds": self.thresholds,
            "template_directory": self.template_directory,
            "template_dependency": "None - uses segyio native detection",
            "advantages": [
                "No template files required",
                "Industry-standard segyio engine",
                "Robust error handling",
                "Memory-efficient processing",
                "Calibrated for real-world data"
            ]
        }

    def validate_classification_result(self, result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a classification result for consistency and completeness
        Maintains compatibility with original SurveyClassifier API
        """
        issues = []

        try:
            # Check required fields
            required_fields = ["survey_type", "primary_sorting", "stack_type", "confidence"]
            for field in required_fields:
                if field not in result:
                    issues.append(f"Missing required field: {field}")
                elif result[field] is None:
                    issues.append(f"Field {field} is None")

            # Validate field values using enums
            valid_survey_types = [e.value for e in SurveyType]
            if result.get("survey_type") not in valid_survey_types:
                issues.append(f"Invalid survey_type: {result.get('survey_type')}")

            valid_sorting = [e.value for e in SortingMethod]
            if result.get("primary_sorting") not in valid_sorting:
                issues.append(f"Invalid primary_sorting: {result.get('primary_sorting')}")

            valid_stack = [e.value for e in StackType]
            if result.get("stack_type") not in valid_stack:
                issues.append(f"Invalid stack_type: {result.get('stack_type')}")

            valid_confidence = [e.value for e in ConfidenceLevel]
            if result.get("confidence") not in valid_confidence:
                issues.append(f"Invalid confidence: {result.get('confidence')}")

            # Logical consistency checks
            survey_type = result.get("survey_type")
            sorting = result.get("primary_sorting")

            if survey_type == "2D" and sorting in ["Inline", "Crossline"]:
                issues.append("Inconsistent: 2D survey cannot have Inline/Crossline sorting")

            if survey_type == "3D" and sorting == "SP":
                issues.append("Warning: 3D survey with SP sorting is unusual")

            # Check for reasonable traces analyzed
            traces_analyzed = result.get("traces_analyzed", 0)
            if traces_analyzed < 10:
                issues.append(f"Very few traces analyzed: {traces_analyzed}")

            # segyio-specific validation
            template_used = result.get("template_used", "")
            if "segyio" not in template_used:
                issues.append("Warning: Expected segyio-based classification")

            return len(issues) == 0, issues

        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            return False, issues


# Backward compatibility - create an alias to maintain existing imports
SurveyClassifier = SegyioSurveyClassifier


# Additional utility functions for compatibility
def quick_classify_survey(file_path: str, max_traces: int = 1000) -> Dict[str, str]:
    """
    Quick survey classification returning just the essential results
    Useful for rapid file assessment
    """
    try:
        classifier = SegyioSurveyClassifier()
        result = classifier.classify_survey(file_path, max_traces=max_traces)

        return {
            "file": os.path.basename(file_path),
            "survey_type": result.get("survey_type", "undetermined"),
            "primary_sorting": result.get("primary_sorting", "undetermined"),
            "stack_type": result.get("stack_type", "undetermined"),
            "confidence": result.get("confidence", "Very Low"),
            "success": result.get("success", False)
        }
    except Exception as e:
        return {
            "file": os.path.basename(file_path),
            "survey_type": "undetermined",
            "primary_sorting": "undetermined",
            "stack_type": "undetermined",
            "confidence": "Very Low",
            "success": False,
            "error": str(e)
        }


def batch_classify_directory(directory_path: str, pattern: str = "*.segy") -> Dict[str, Any]:
    """
    Classify all SEG-Y files in a directory
    Convenience function for batch operations
    """
    try:
        from pathlib import Path

        # Find all matching files
        directory = Path(directory_path)
        files = list(directory.glob(pattern))
        files.extend(directory.glob("*.sgy"))
        files.extend(directory.glob("*.SGY"))
        files.extend(directory.glob("*.SEGY"))

        # Remove duplicates
        files = list(set(files))

        if not files:
            return {"error": f"No SEG-Y files found in {directory_path}"}

        # Classify all files
        classifier = SegyioSurveyClassifier()
        return classifier.classify_multiple_files([str(f) for f in files])

    except Exception as e:
        return {"error": f"Batch classification failed: {str(e)}"}


if __name__ == "__main__":
    # Example usage and testing
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]

        print(f"Classifying: {file_path}")
        print("-" * 50)

        # Quick classification
        quick_result = quick_classify_survey(file_path)
        print("Quick Classification:")
        for key, value in quick_result.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 50)

        # Detailed classification
        classifier = SegyioSurveyClassifier()
        detailed_result = classifier.classify_survey(file_path)

        # Generate report
        report = classifier.generate_classification_report(detailed_result)
        print(report)

    else:
        print("Usage: python survey_classifier.py <segy_file_path>")
        print("Example: python survey_classifier.py ./data/survey_3d.segy")