#!/usr/bin/env python3
"""
comprehensive_test_script.py - Enhanced with Rate Limiting Validation

This script tests all major components including:
1. segyio Transformation validation
2. Metadata Harvester functionality and rate limiting
3. MCP tools rate limiting validation
4. Complete workflow integration testing
5. Performance and quality assessments

Run this after deploying all updates including rate limiting fixes.
"""

import sys
import json
import time
import traceback
from pathlib import Path


# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")


def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")


def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")


def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}")
    print(f"{msg}")
    print(f"{'=' * 60}{Colors.END}")


class SEGYTransformationTester:
    """Comprehensive tester for segyio transformation + metadata harvester + rate limiting"""

    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)

        # Initialize results dictionary FIRST
        self.results = {
            "import_tests": {},
            "parser_tests": {},
            "classifier_tests": {},
            "qc_tests": {},
            "metadata_harvester_tests": {},
            "rate_limiting_tests": {},  # NEW: Rate limiting validation
            "mcp_tools_tests": {},  # NEW: MCP tools validation
            "performance_tests": {},
            "integration_tests": {},
            "file_discovery": {},
            "summary": {}
        }

        # Auto-discover SEG-Y files after results is initialized
        self.segy_extensions = [
            "*.segy", "*.sgy", "*.SGY", "*.SEGY",
            "*.seg", "*.SEG", "*.segY", "*.Segy"
        ]

        self.test_files = self._discover_segy_files()

    def _discover_segy_files(self):
        """Auto-discover all SEG-Y files in the data directory"""
        print_info(f"Scanning for SEG-Y files in: {self.data_dir}")

        discovered_files = []

        if not self.data_dir.exists():
            print_error(f"Data directory does not exist: {self.data_dir}")
            return []

        # Search for all SEG-Y file extensions
        for extension in self.segy_extensions:
            files = list(self.data_dir.glob(extension))
            for file_path in files:
                if file_path.is_file():
                    discovered_files.append(file_path.name)

        # Remove duplicates (in case of case-insensitive filesystem)
        unique_files = list(set(discovered_files))

        # Sort by file size (largest first) for better testing
        file_info = []
        for filename in unique_files:
            filepath = self.data_dir / filename
            size_mb = filepath.stat().st_size / (1024 * 1024)
            file_info.append((filename, size_mb))

        # Sort by size (largest first) and extract filenames
        file_info.sort(key=lambda x: x[1], reverse=True)
        sorted_files = [f[0] for f in file_info]

        # Store discovery results
        self.results["file_discovery"] = {
            "directory_scanned": str(self.data_dir),
            "extensions_searched": self.segy_extensions,
            "total_files_found": len(sorted_files),
            "files_by_size": file_info
        }

        print_info(f"Found {len(sorted_files)} SEG-Y files:")
        for filename, size_mb in file_info:
            print_info(f"  {filename} ({size_mb:.1f}MB)")

        return sorted_files

    def run_all_tests(self):
        """Run comprehensive test suite"""
        print_header("COMPLETE SEG-Y WORKFLOW + RATE LIMITING VALIDATION")
        print(f"Data directory: {self.data_dir}")

        if not self.test_files:
            print_error("No SEG-Y files found in the specified directory!")
            print_info("Searched for extensions: " + ", ".join(self.segy_extensions))
            return False

        print(f"Discovered files: {len(self.test_files)} SEG-Y files")

        # Validate discovered files
        available_files = self._check_test_files()
        print(f"Valid files: {len(available_files)}")

        if len(available_files) == 0:
            print_error("No valid SEG-Y files found for testing!")
            return False

        # Enhanced test suite with rate limiting
        tests = [
            ("Import Tests", self._test_imports),
            ("Parser Tests", self._test_parser),
            ("Classifier Tests", self._test_classifier),
            ("Quality Control Tests", self._test_qc),
            ("Metadata Harvester Tests", self._test_metadata_harvester),
            ("Rate Limiting Tests", self._test_rate_limiting),  # NEW
            ("MCP Tools Tests", self._test_mcp_tools),  # NEW
            ("Performance Tests", self._test_performance),
            ("Integration Tests", self._test_integration)
        ]

        overall_success = True

        for test_name, test_func in tests:
            print_header(test_name)
            try:
                success = test_func(available_files)
                if not success:
                    overall_success = False
            except Exception as e:
                print_error(f"{test_name} failed with exception: {e}")
                traceback.print_exc()
                overall_success = False

        # Generate summary
        self._generate_summary()

        return overall_success

    def _check_test_files(self):
        """Check which test files are available"""
        print_info("Validating discovered SEG-Y files...")

        if not self.test_files:
            print_warning("No SEG-Y files found in the directory!")
            return []

        valid_files = []
        corrupted_files = []

        for filename in self.test_files:
            filepath = self.data_dir / filename
            try:
                if filepath.exists() and filepath.is_file():
                    size_mb = filepath.stat().st_size / (1024 * 1024)

                    # Basic file validation
                    if size_mb < 0.001:  # Less than 1KB
                        print_warning(f"Very small file: {filename} ({size_mb * 1000:.1f}KB)")
                        corrupted_files.append(filename)
                    elif size_mb > 10000:  # Greater than 10GB
                        print_warning(f"Very large file: {filename} ({size_mb:.1f}MB) - may be slow")
                        valid_files.append(filename)
                    else:
                        print_success(f"Valid: {filename} ({size_mb:.1f}MB)")
                        valid_files.append(filename)
                else:
                    print_error(f"File issue: {filename}")
                    corrupted_files.append(filename)

            except Exception as e:
                print_error(f"Error checking {filename}: {e}")
                corrupted_files.append(filename)

        # Update results with file validation info
        self.results["file_discovery"]["valid_files"] = len(valid_files)
        self.results["file_discovery"]["corrupted_files"] = len(corrupted_files)
        self.results["file_discovery"]["validation_issues"] = corrupted_files

        return valid_files

    def _test_imports(self, available_files):
        """Test that all updated modules import correctly"""
        print_info("Testing module imports...")

        import_tests = {
            "production_segy_tools": "from production_segy_tools import production_segy_parser, SegyioValidator, segy_complete_metadata_harvester, mcp_extract_survey_polygon, mcp_extract_trace_outlines",
            "survey_classifier": "from survey_classifier import SurveyClassifier, SegyioSurveyClassifier",
            "production_segy_analysis_qc": "from production_segy_analysis_qc import production_segy_qc, SegyioQualityAnalyzer",
            "result_classes": "from result_classes import ClassificationResult, QualityMetrics, SurveyType"
        }

        success = True
        for module_name, import_statement in import_tests.items():
            try:
                exec(import_statement)
                print_success(f"Import: {module_name}")
                self.results["import_tests"][module_name] = True
            except Exception as e:
                print_error(f"Import failed: {module_name} - {e}")
                self.results["import_tests"][module_name] = False
                success = False

        return success

    def _test_parser(self, available_files):
        """Test the updated parser with segyio"""
        print_info("Testing production_segy_parser...")

        try:
            from production_segy_tools import production_segy_parser

            success = True
            for filename in available_files[:3]:  # Test first 3 files
                print_info(f"Testing parser with: {filename}")

                start_time = time.time()
                result = production_segy_parser(filename)
                processing_time = time.time() - start_time

                # Parse result
                try:
                    data = json.loads(result["text"])

                    # Check for essential fields
                    required_fields = ["file_processed", "total_traces", "sample_rate_ms", "parsing_method"]
                    missing_fields = [f for f in required_fields if f not in data]

                    if missing_fields:
                        print_error(f"Missing fields in {filename}: {missing_fields}")
                        success = False
                    else:
                        print_success(
                            f"Parser success: {filename} ({data['total_traces']} traces, {processing_time:.2f}s)")

                        # Check for segyio indicators
                        if "segyio" in data.get("parsing_method", "").lower():
                            print_success(f"  Using segyio engine")
                        else:
                            print_warning(f"  May not be using segyio engine")

                        self.results["parser_tests"][filename] = {
                            "success": True,
                            "traces": data.get("total_traces", 0),
                            "processing_time": processing_time,
                            "uses_segyio": "segyio" in data.get("parsing_method", "").lower()
                        }

                except json.JSONDecodeError as e:
                    print_error(f"JSON decode error for {filename}: {e}")
                    success = False

            return success

        except ImportError as e:
            print_error(f"Cannot import production_segy_parser: {e}")
            return False

    def _test_classifier(self, available_files):
        """Test the updated survey classifier"""
        print_info("Testing survey classifier...")

        try:
            from survey_classifier import SurveyClassifier

            classifier = SurveyClassifier()
            success = True

            for filename in available_files[:3]:  # Test first 3 files
                print_info(f"Testing classifier with: {filename}")

                start_time = time.time()
                result = classifier.classify_survey(str(self.data_dir / filename))
                processing_time = time.time() - start_time

                # Check result structure
                required_fields = ["survey_type", "primary_sorting", "confidence", "success"]
                missing_fields = [f for f in required_fields if f not in result]

                if missing_fields:
                    print_error(f"Missing fields in classification: {missing_fields}")
                    success = False
                else:
                    survey_type = result["survey_type"]
                    confidence = result["confidence"]

                    print_success(f"Classification: {filename}")
                    print_info(f"  Survey: {survey_type}, Confidence: {confidence}")
                    print_info(f"  Processing time: {processing_time:.2f}s")

                    # Check for shot gather detection (expected for your files)
                    if "shot" in filename.lower():
                        if survey_type in ["2D", "3D"] and "shot_gather" in str(
                                result.get("classification_details", {})):
                            print_success(f"  Correctly detected shot gather characteristics")
                        elif survey_type == "undetermined":
                            print_warning(f"  Shot gather not clearly classified")

                    self.results["classifier_tests"][filename] = {
                        "success": result["success"],
                        "survey_type": survey_type,
                        "confidence": confidence,
                        "processing_time": processing_time
                    }

            return success

        except ImportError as e:
            print_error(f"Cannot import SurveyClassifier: {e}")
            return False

    def _test_qc(self, available_files):
        """Test the updated QC system with calibrated thresholds"""
        print_info("Testing quality control system...")

        try:
            from production_segy_analysis_qc import production_segy_qc

            success = True
            quality_ratings = []

            for filename in available_files[:3]:  # Test first 3 files
                print_info(f"Testing QC with: {filename}")

                start_time = time.time()
                result = production_segy_qc(filename)
                processing_time = time.time() - start_time

                try:
                    data = json.loads(result["text"])

                    overall_assessment = data.get("overall_assessment", {})
                    quality_rating = overall_assessment.get("quality_rating", "Unknown")
                    recommendation = overall_assessment.get("recommendation", "No recommendation")

                    print_success(f"QC complete: {filename}")
                    print_info(f"  Quality: {quality_rating}")
                    print_info(f"  Recommendation: {recommendation}")
                    print_info(f"  Processing time: {processing_time:.2f}s")

                    # Check for improvement (should not be "Poor" for valid files)
                    if quality_rating == "Poor":
                        print_warning(f"  Still rated as Poor - check calibration")
                    elif quality_rating in ["Good", "Excellent", "Fair"]:
                        print_success(f"  Improved quality rating!")

                    # Check for segyio indicators
                    qc_engine = data.get("file_info", {}).get("qc_engine", "")
                    if "segyio" in qc_engine.lower():
                        print_success(f"  Using segyio-based QC engine")

                    quality_ratings.append(quality_rating)

                    self.results["qc_tests"][filename] = {
                        "quality_rating": quality_rating,
                        "recommendation": recommendation,
                        "processing_time": processing_time,
                        "uses_segyio": "segyio" in qc_engine.lower()
                    }

                except json.JSONDecodeError as e:
                    print_error(f"QC JSON decode error for {filename}: {e}")
                    success = False

            # Summary of quality ratings
            if quality_ratings:
                quality_distribution = {rating: quality_ratings.count(rating) for rating in set(quality_ratings)}
                print_info(f"Quality distribution: {quality_distribution}")
            else:
                print_warning("No quality ratings collected")

            return success

        except ImportError as e:
            print_error(f"Cannot import production_segy_qc: {e}")
            return False

    def _test_metadata_harvester(self, available_files):
        """Test the complete metadata harvester functionality"""
        print_info("Testing segy_complete_metadata_harvester...")

        try:
            from production_segy_tools import segy_complete_metadata_harvester

            success = True
            harvester_results = []

            # Test scenarios for metadata harvester including rate limiting
            test_scenarios = [
                {
                    "name": "Default Parameters",
                    "params": {"file_path": None}
                },
                {
                    "name": "Rate Limit Safe",
                    "params": {
                        "file_path": None,
                        "include_trace_sampling": True,
                        "trace_sample_size": 5,
                        "include_statistics": False,
                        "max_text_length": 2000  # NEW: Test the fix
                    }
                },
                {
                    "name": "Large Sample with Truncation",
                    "params": {
                        "file_path": None,
                        "include_trace_sampling": True,
                        "trace_sample_size": 500,
                        "include_statistics": True,
                        "max_text_length": 3000
                    }
                },
                {
                    "name": "Summary Mode",
                    "params": {
                        "file_path": None,
                        "return_format": "summary",
                        "max_text_length": 2000
                    }
                },
                {
                    "name": "Full Mode",
                    "params": {
                        "file_path": None,
                        "return_format": "full",
                        "max_text_length": 10000
                    }
                }
            ]

            for filename in available_files[:2]:  # Test first 2 files
                print_info(f"Testing metadata harvester with: {filename}")
                file_results = {"filename": filename, "scenarios": {}}

                for scenario in test_scenarios:
                    scenario_name = scenario["name"]
                    print_info(f"  Scenario: {scenario_name}")

                    # Set file path for this scenario
                    test_params = scenario["params"].copy()
                    test_params["file_path"] = filename

                    start_time = time.time()
                    try:
                        result = segy_complete_metadata_harvester(**test_params)
                        processing_time = time.time() - start_time

                        # Validate result structure
                        if "error" in result:
                            print_error(f"    Error: {result['error']}")
                            file_results["scenarios"][scenario_name] = {
                                "success": False,
                                "error": result["error"],
                                "processing_time": processing_time
                            }
                            success = False
                            continue

                        # Parse JSON result and validate
                        text_content = result.get("text", "")
                        tokens = len(text_content) // 4

                        try:
                            metadata = json.loads(text_content)
                            json_valid = True
                            print_success(f"    ✓ JSON valid ({len(text_content)} chars, {tokens} tokens)")

                            # Check for rate limiting info
                            if "metadata" in result and "rate_limit_warning" in result["metadata"]:
                                warning = result["metadata"]["rate_limit_warning"]
                                print_info(f"    Rate limiting applied: {warning}")

                        except json.JSONDecodeError as e:
                            json_valid = False
                            print_error(f"    JSON decode error: {e}")
                            success = False

                        # Store results
                        file_results["scenarios"][scenario_name] = {
                            "success": json_valid,
                            "processing_time": processing_time,
                            "text_length": len(text_content),
                            "estimated_tokens": tokens,
                            "json_valid": json_valid,
                            "rate_limit_applied": "metadata" in result and "rate_limit_warning" in result.get(
                                "metadata", {}),
                            "extraction_parameters": result.get("extraction_parameters", {})
                        }

                    except Exception as e:
                        processing_time = time.time() - start_time
                        print_error(f"    Exception: {e}")
                        file_results["scenarios"][scenario_name] = {
                            "success": False,
                            "error": str(e),
                            "processing_time": processing_time
                        }
                        success = False

                harvester_results.append(file_results)

            # Store overall results
            self.results["metadata_harvester_tests"] = {
                "overall_success": success,
                "files_tested": len(available_files[:2]),
                "scenarios_tested": len(test_scenarios),
                "detailed_results": harvester_results
            }

            return success

        except ImportError as e:
            print_error(f"Cannot import segy_complete_metadata_harvester: {e}")
            return False

    def _test_rate_limiting(self, available_files):
        """NEW: Test rate limiting across all MCP tools"""
        print_info("Testing rate limiting implementation...")

        if not available_files:
            print_warning("No files available for rate limiting testing")
            return True

        try:
            # Import all MCP tools
            from production_segy_tools import (
                segy_complete_metadata_harvester,
                mcp_extract_survey_polygon,
                mcp_extract_trace_outlines
            )

            success = True
            rate_limit_results = {}
            test_file = available_files[0]

            print_info(f"Rate limiting test with: {test_file}")

            # Test 1: Individual tool rate limiting
            print_info("  Testing individual tools...")

            # Metadata harvester rate limiting
            metadata_result = segy_complete_metadata_harvester(
                file_path=test_file,
                trace_sample_size=5,
                include_statistics=False,
                max_text_length=2000
            )
            metadata_tokens = len(json.dumps(metadata_result)) // 4
            metadata_safe = metadata_tokens < 5000

            print_info(f"    Metadata: {metadata_tokens} tokens ({'✓' if metadata_safe else '✗'})")

            # Survey polygon rate limiting
            polygon_result = mcp_extract_survey_polygon(
                file_path=test_file,
                return_format="summary"
            )
            polygon_tokens = len(json.dumps(polygon_result)) // 4
            polygon_safe = polygon_tokens < 1000

            print_info(f"    Polygon: {polygon_tokens} tokens ({'✓' if polygon_safe else '✗'})")

            # Trace outlines rate limiting
            trace_result = mcp_extract_trace_outlines(
                file_path=test_file,
                return_format="summary",
                max_traces=5
            )
            trace_tokens = len(json.dumps(trace_result)) // 4
            trace_safe = trace_tokens < 1000

            print_info(f"    Traces: {trace_tokens} tokens ({'✓' if trace_safe else '✗'})")

            # Test 2: Complete workflow rate limiting
            print_info("  Testing complete workflow...")

            total_tokens = metadata_tokens + polygon_tokens + trace_tokens
            workflow_safe = total_tokens < 5000

            print_info(f"    Total workflow: {total_tokens} tokens ({'✓' if workflow_safe else '✗'})")

            # Test 3: Different return formats
            print_info("  Testing different return formats...")

            format_tests = [
                ("summary", {"return_format": "summary"}),
                ("simplified", {"return_format": "simplified", "max_coordinates": 20}),
                ("limited", {"return_format": "limited", "max_traces": 3})
            ]

            format_results = {}
            for format_name, params in format_tests:
                if format_name in ["summary", "simplified"]:
                    test_result = mcp_extract_survey_polygon(file_path=test_file, **params)
                else:
                    test_result = mcp_extract_trace_outlines(file_path=test_file, **params)

                format_tokens = len(json.dumps(test_result)) // 4
                format_safe = format_tokens < 1000
                format_results[format_name] = {"tokens": format_tokens, "safe": format_safe}

                print_info(f"    {format_name}: {format_tokens} tokens ({'✓' if format_safe else '✗'})")

            # Test 4: JSON validity under rate limiting
            print_info("  Testing JSON validity...")

            json_tests = [metadata_result, polygon_result, trace_result]
            json_valid_count = 0

            for i, result in enumerate(json_tests):
                try:
                    json.loads(json.dumps(result))
                    json_valid_count += 1
                except:
                    success = False
                    print_error(f"    Tool {i + 1}: Invalid JSON")

            print_info(f"    JSON validity: {json_valid_count}/3 tools ({'✓' if json_valid_count == 3 else '✗'})")

            # Store results
            rate_limit_results = {
                "individual_tools": {
                    "metadata": {"tokens": metadata_tokens, "safe": metadata_safe},
                    "polygon": {"tokens": polygon_tokens, "safe": polygon_safe},
                    "traces": {"tokens": trace_tokens, "safe": trace_safe}
                },
                "workflow": {"total_tokens": total_tokens, "safe": workflow_safe},
                "format_tests": format_results,
                "json_validity": {"valid_count": json_valid_count, "total": 3},
                "overall_success": (metadata_safe and polygon_safe and trace_safe and
                                    workflow_safe and json_valid_count == 3)
            }

            self.results["rate_limiting_tests"] = rate_limit_results

            if rate_limit_results["overall_success"]:
                print_success("✅ Rate limiting working correctly across all tools")
            else:
                print_error("❌ Rate limiting issues detected")
                success = False

            return success

        except Exception as e:
            print_error(f"Rate limiting test failed: {e}")
            return False

    def _test_mcp_tools(self, available_files):
        """NEW: Test MCP-specific tools functionality"""
        print_info("Testing MCP tools functionality...")

        if not available_files:
            print_warning("No files available for MCP tools testing")
            return True

        try:
            from production_segy_tools import (
                mcp_extract_survey_polygon,
                mcp_extract_trace_outlines
            )

            success = True
            mcp_results = {}
            test_file = available_files[0]

            print_info(f"MCP tools test with: {test_file}")

            # Test 1: Survey Polygon Extraction
            print_info("  Testing survey polygon extraction...")

            polygon_tests = [
                ("default", {}),
                ("summary", {"return_format": "summary"}),
                ("simplified", {"return_format": "simplified", "max_coordinates": 10}),
                ("full", {"return_format": "full", "coordinate_sample_rate": 20})
            ]

            polygon_results = {}
            for test_name, params in polygon_tests:
                try:
                    result = mcp_extract_survey_polygon(file_path=test_file, **params)

                    if "error" in result:
                        print_error(f"    {test_name}: {result['error']}")
                        success = False
                        polygon_results[test_name] = {"success": False, "error": result["error"]}
                    else:
                        coords = len(result.get("survey_polygon", []))
                        area = result.get("polygon_area_km2", 0)
                        tokens = len(json.dumps(result)) // 4

                        print_success(f"    {test_name}: {coords} coords, {area:.2f} km², {tokens} tokens")
                        polygon_results[test_name] = {
                            "success": True,
                            "coordinates": coords,
                            "area": area,
                            "tokens": tokens
                        }

                except Exception as e:
                    print_error(f"    {test_name}: Exception - {e}")
                    success = False
                    polygon_results[test_name] = {"success": False, "error": str(e)}

            # Test 2: Trace Outlines Extraction
            print_info("  Testing trace outlines extraction...")

            trace_tests = [
                ("default", {}),
                ("summary", {"return_format": "summary"}),
                ("limited", {"return_format": "limited", "max_traces": 3}),
                ("full", {"return_format": "full", "max_traces": 5})
            ]

            trace_results = {}
            for test_name, params in trace_tests:
                try:
                    result = mcp_extract_trace_outlines(file_path=test_file, **params)

                    if "error" in result:
                        print_error(f"    {test_name}: {result['error']}")
                        success = False
                        trace_results[test_name] = {"success": False, "error": result["error"]}
                    else:
                        traces = len(result.get("trace_outlines", []))
                        tokens = len(json.dumps(result)) // 4
                        vis_ready = result.get("status", {}).get("visualization_ready", False)

                        print_success(f"    {test_name}: {traces} traces, {tokens} tokens, vis: {vis_ready}")
                        trace_results[test_name] = {
                            "success": True,
                            "traces": traces,
                            "tokens": tokens,
                            "visualization_ready": vis_ready
                        }

                except Exception as e:
                    print_error(f"    {test_name}: Exception - {e}")
                    success = False
                    trace_results[test_name] = {"success": False, "error": str(e)}

            # Test 3: Data directory handling
            print_info("  Testing data directory handling...")

            data_dir_tests = [
                ("explicit_path", {"data_dir": str(self.data_dir)}),
                ("relative_path", {"data_dir": "./data"}),
                ("default", {})
            ]

            data_dir_success = 0
            for test_name, params in data_dir_tests:
                try:
                    result = mcp_extract_survey_polygon(file_path=test_file, **params)
                    if "error" not in result:
                        data_dir_success += 1
                        print_success(f"    {test_name}: ✓")
                    else:
                        print_warning(f"    {test_name}: {result['error']}")
                except Exception as e:
                    print_warning(f"    {test_name}: {e}")

            print_info(f"    Data directory handling: {data_dir_success}/3 tests passed")

            # Store MCP results
            mcp_results = {
                "polygon_extraction": polygon_results,
                "trace_extraction": trace_results,
                "data_directory_handling": {"success_count": data_dir_success, "total": 3},
                "overall_success": success
            }

            self.results["mcp_tools_tests"] = mcp_results

            if success:
                print_success("✅ MCP tools working correctly")
            else:
                print_error("❌ MCP tools issues detected")

            return success

        except Exception as e:
            print_error(f"MCP tools test failed: {e}")
            return False

    def _test_performance(self, available_files):
        """Test performance improvements"""
        print_info("Testing performance improvements...")

        if not available_files:
            print_warning("No files available for performance testing")
            return True

        try:
            from production_segy_tools import production_segy_parser, segy_complete_metadata_harvester
            from survey_classifier import SurveyClassifier

            # Test with largest available file
            test_file = available_files[0]
            file_size = (self.data_dir / test_file).stat().st_size / (1024 * 1024)

            print_info(f"Performance test with: {test_file} ({file_size:.1f}MB)")

            # Parser performance
            start_time = time.time()
            parser_result = production_segy_parser(test_file)
            parser_time = time.time() - start_time

            # Classifier performance
            classifier = SurveyClassifier()
            start_time = time.time()
            classifier_result = classifier.classify_survey(str(self.data_dir / test_file))
            classifier_time = time.time() - start_time

            # Metadata harvester performance
            start_time = time.time()
            metadata_result = segy_complete_metadata_harvester(
                file_path=test_file,
                trace_sample_size=5,
                include_statistics=False,
                max_text_length=2000
            )
            metadata_time = time.time() - start_time

            # Calculate performance metrics
            parser_data = json.loads(parser_result["text"])
            total_traces = parser_data.get("total_traces", 1)

            traces_per_sec_parser = total_traces / parser_time if parser_time > 0 else 0
            traces_per_sec_classifier = total_traces / classifier_time if classifier_time > 0 else 0
            traces_per_sec_metadata = total_traces / metadata_time if metadata_time > 0 else 0
            mb_per_sec = file_size / parser_time if parser_time > 0 else 0

            print_success(f"Parser performance:")
            print_info(f"  Time: {parser_time:.2f}s")
            print_info(f"  Traces/sec: {traces_per_sec_parser:.0f}")
            print_info(f"  MB/sec: {mb_per_sec:.1f}")

            print_success(f"Classifier performance:")
            print_info(f"  Time: {classifier_time:.2f}s")
            print_info(f"  Traces/sec: {traces_per_sec_classifier:.0f}")

            print_success(f"Metadata harvester performance:")
            print_info(f"  Time: {metadata_time:.2f}s")
            print_info(f"  Traces/sec: {traces_per_sec_metadata:.0f}")

            self.results["performance_tests"] = {
                "test_file": test_file,
                "file_size_mb": file_size,
                "parser_time": parser_time,
                "classifier_time": classifier_time,
                "metadata_time": metadata_time,
                "traces_per_sec_parser": traces_per_sec_parser,
                "traces_per_sec_classifier": traces_per_sec_classifier,
                "traces_per_sec_metadata": traces_per_sec_metadata,
                "mb_per_sec": mb_per_sec
            }

            # Performance thresholds (adjust based on your system)
            if traces_per_sec_parser > 1000:
                print_success("Parser performance: Excellent")
            elif traces_per_sec_parser > 500:
                print_success("Parser performance: Good")
            else:
                print_warning("Parser performance: Consider optimization")

            return True

        except Exception as e:
            print_error(f"Performance test failed: {e}")
            return False

    def _test_integration(self, available_files):
        """Test integration between components including rate limiting"""
        print_info("Testing component integration...")

        if not available_files:
            print_warning("No files available for integration testing")
            return True

        try:
            # Test full pipeline: Parser → Classifier → QC → Metadata Harvester → MCP Tools
            test_file = available_files[0]
            print_info(f"Integration test with: {test_file}")

            # Step 1: Parse
            from production_segy_tools import production_segy_parser
            parser_result = production_segy_parser(test_file)
            parser_data = json.loads(parser_result["text"])

            # Step 2: Classify
            from survey_classifier import SurveyClassifier
            classifier = SurveyClassifier()
            classifier_result = classifier.classify_survey(str(self.data_dir / test_file))

            # Step 3: QC
            from production_segy_analysis_qc import production_segy_qc
            qc_result = production_segy_qc(test_file)
            qc_data = json.loads(qc_result["text"])

            # Step 4: Metadata Harvester (with rate limiting)
            from production_segy_tools import segy_complete_metadata_harvester
            metadata_result = segy_complete_metadata_harvester(
                file_path=test_file,
                trace_sample_size=5,
                include_statistics=False,
                max_text_length=2000
            )
            metadata_data = json.loads(metadata_result["text"]) if "error" not in metadata_result else {}

            # Step 5: MCP Tools (with rate limiting)
            from production_segy_tools import mcp_extract_survey_polygon, mcp_extract_trace_outlines

            polygon_result = mcp_extract_survey_polygon(
                file_path=test_file,
                return_format="summary"
            )

            trace_result = mcp_extract_trace_outlines(
                file_path=test_file,
                return_format="summary",
                max_traces=5
            )

            # ROBUST TRACE COUNT EXTRACTION
            parser_traces = parser_data.get("total_traces", 0)
            classifier_traces = classifier_result.get("traces_analyzed", 0)
            metadata_traces = metadata_data.get("file_info", {}).get("total_traces", 0)

            # Try multiple paths to find QC trace count
            qc_traces = 0
            possible_qc_paths = [
                ("file_info", "total_traces"),
                ("validation_results", "file_structure", "total_traces"),
                ("validation_results", "geometry", "total_traces"),
                ("overall_assessment", "total_traces"),
                ("file_info", "trace_count"),
            ]

            for path in possible_qc_paths:
                temp_data = qc_data
                try:
                    for key in path:
                        temp_data = temp_data.get(key, {})
                    if isinstance(temp_data, int) and temp_data > 0:
                        qc_traces = temp_data
                        print_info(f"Found QC traces at: {' -> '.join(path)} = {qc_traces}")
                        break
                except (AttributeError, TypeError):
                    continue

            print_info(
                f"Trace count comparison: Parser={parser_traces}, Classifier={classifier_traces}, QC={qc_traces}, Metadata={metadata_traces}")

            # NEW: Rate limiting validation in integration
            print_info("Validating rate limiting in integration...")

            # Calculate total token usage
            total_tokens = 0
            component_tokens = {}

            for name, result in [
                ("parser", parser_result),
                ("classifier", {"text": json.dumps(classifier_result)}),
                ("qc", qc_result),
                ("metadata", metadata_result),
                ("polygon", {"text": json.dumps(polygon_result)}),
                ("traces", {"text": json.dumps(trace_result)})
            ]:
                tokens = len(json.dumps(result)) // 4
                component_tokens[name] = tokens
                total_tokens += tokens
                print_info(f"  {name}: {tokens} tokens")

            print_info(f"  Total integration tokens: {total_tokens}")
            integration_rate_safe = total_tokens < 15000  # Allow higher limit for full integration

            print_info(f"  Integration rate limit safe: {'✓' if integration_rate_safe else '✗'}")

            # Check consistency between components
            issues = []

            # Trace count consistency
            if qc_traces == 0:
                issues.append(f"QC trace count not accessible: Parser={parser_traces}, QC={qc_traces}")
            elif parser_traces != qc_traces:
                issues.append(f"Trace count mismatch: Parser={parser_traces}, QC={qc_traces}")

            # Check metadata harvester consistency
            if metadata_traces > 0 and parser_traces != metadata_traces:
                issues.append(f"Metadata trace count mismatch: Parser={parser_traces}, Metadata={metadata_traces}")

            # Survey type consistency
            classifier_survey = classifier_result.get("survey_type", "")
            qc_survey = qc_data.get("validation_results", {}).get("survey_type", "")

            survey_type_compatible = self._check_survey_type_compatibility(classifier_survey, qc_survey)

            if qc_survey and classifier_survey and not survey_type_compatible:
                issues.append(f"Survey type mismatch: Classifier={classifier_survey}, QC={qc_survey}")

            # NEW: MCP tools validation
            polygon_coords = len(polygon_result.get("survey_polygon", []))
            trace_outlines = len(trace_result.get("trace_outlines", []))

            if polygon_coords == 0:
                issues.append("MCP polygon extraction returned no coordinates")

            if trace_outlines == 0:
                issues.append("MCP trace extraction returned no outlines")

            # DETERMINE SUCCESS
            real_errors = [i for i in issues if "not accessible" not in i and "mismatch" in i]

            if len(real_errors) == 0 and integration_rate_safe:
                success = True
                print_success("✅ All components integrated successfully with rate limiting")
                print_info(f"  Traces: {parser_traces}")
                print_info(f"  Survey type: {classifier_survey}")
                print_info(f"  Quality: {qc_data.get('overall_assessment', {}).get('quality_rating', 'Unknown')}")
                print_info(f"  Metadata extraction: {'✓' if metadata_data else '✗'}")
                print_info(f"  MCP polygon coords: {polygon_coords}")
                print_info(f"  MCP trace outlines: {trace_outlines}")
                print_info(f"  Total tokens: {total_tokens} (rate safe: {integration_rate_safe})")
            else:
                success = False
                for issue in issues:
                    if "mismatch" in issue:
                        print_error(f"Integration error: {issue}")
                    else:
                        print_warning(f"Integration note: {issue}")

                if not integration_rate_safe:
                    print_error(f"Integration exceeds rate limit: {total_tokens} tokens")

            self.results["integration_tests"] = {
                "success": success,
                "issues": issues,
                "parser_traces": parser_traces,
                "classifier_traces": classifier_traces,
                "qc_traces": qc_traces,
                "metadata_traces": metadata_traces,
                "survey_type_compatibility": survey_type_compatible,
                "classifier_survey_type": classifier_survey,
                "qc_survey_type": qc_survey,
                "metadata_extraction_success": bool(metadata_data),
                "mcp_polygon_coords": polygon_coords,
                "mcp_trace_outlines": trace_outlines,
                "rate_limiting": {
                    "component_tokens": component_tokens,
                    "total_tokens": total_tokens,
                    "rate_safe": integration_rate_safe
                }
            }

            return success

        except Exception as e:
            print_error(f"Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _check_survey_type_compatibility(self, classifier_survey, qc_survey):
        """Check if survey types from different components are compatible"""
        if not qc_survey or not classifier_survey:
            return True  # If one is missing, don't consider it an error

        # Direct match
        if classifier_survey == qc_survey:
            return True

        # Compatible combinations
        compatible_pairs = [
            ("undetermined", "shot_gather"),
            ("2D", "shot_gather"),
            ("3D", "shot_gather"),
            ("2D", "migrated_2d"),
            ("3D", "migrated_3d"),
            ("2D", "cdp_stack"),
            ("3D", "cdp_stack")
        ]

        return (classifier_survey, qc_survey) in compatible_pairs or (qc_survey, classifier_survey) in compatible_pairs

    def _generate_summary(self):
        """Generate enhanced test summary"""
        print_header("COMPREHENSIVE TEST SUMMARY")

        # Enhanced summary statistics
        import_success = sum(self.results["import_tests"].values())
        import_total = len(self.results["import_tests"])

        parser_success = sum(1 for r in self.results["parser_tests"].values() if r.get("success", False))
        parser_total = len(self.results["parser_tests"])

        classifier_success = sum(1 for r in self.results["classifier_tests"].values() if r.get("success", False))
        classifier_total = len(self.results["classifier_tests"])

        qc_tests = len(self.results["qc_tests"])

        # Metadata harvester summary
        metadata_harvester_success = self.results["metadata_harvester_tests"].get("overall_success", False)
        metadata_files_tested = self.results["metadata_harvester_tests"].get("files_tested", 0)
        metadata_scenarios_tested = self.results["metadata_harvester_tests"].get("scenarios_tested", 0)

        # NEW: Rate limiting summary
        rate_limiting_success = self.results["rate_limiting_tests"].get("overall_success", False)

        # NEW: MCP tools summary
        mcp_tools_success = self.results["mcp_tools_tests"].get("overall_success", False)

        # File discovery summary
        discovery = self.results["file_discovery"]
        print_info(
            f"📁 File Discovery: {discovery['total_files_found']} files found, {discovery.get('valid_files', 0)} valid")

        print_info(f"📦 Import Tests: {import_success}/{import_total}")
        print_info(f"🔍 Parser Tests: {parser_success}/{parser_total}")
        print_info(f"🏷️  Classifier Tests: {classifier_success}/{classifier_total}")
        print_info(f"✅ QC Tests: {qc_tests} files tested")
        print_info(
            f"📊 Metadata Harvester: {'✓' if metadata_harvester_success else '✗'} ({metadata_files_tested} files, {metadata_scenarios_tested} scenarios)")
        print_info(f"🚦 Rate Limiting: {'✓' if rate_limiting_success else '✗'}")
        print_info(f"🔧 MCP Tools: {'✓' if mcp_tools_success else '✗'}")
        print_info(f"⚡ Performance Tests: {'✓' if self.results['performance_tests'] else '✗'}")
        print_info(f"🔗 Integration Tests: {'✓' if self.results['integration_tests'].get('success') else '✗'}")

        # Quality improvement summary
        qc_ratings = [r.get("quality_rating") for r in self.results["qc_tests"].values()]
        improved_ratings = sum(1 for r in qc_ratings if r in ["Good", "Excellent", "Fair"])
        poor_ratings = sum(1 for r in qc_ratings if r == "Poor")

        print_header("🎯 TRANSFORMATION RESULTS")

        if poor_ratings == 0 and improved_ratings > 0:
            print_success(f"✨ Quality ratings improved! {improved_ratings} files now rated Fair or better")
        elif poor_ratings < len(qc_ratings) / 2:
            print_success(f"📈 Significant improvement: {improved_ratings} good ratings vs {poor_ratings} poor")
        else:
            print_warning(f"⚠️  Some files still rated as Poor: {poor_ratings}/{len(qc_ratings)}")

        # segyio adoption
        segyio_parser = sum(1 for r in self.results["parser_tests"].values() if r.get("uses_segyio", False))
        segyio_qc = sum(1 for r in self.results["qc_tests"].values() if r.get("uses_segyio", False))

        if segyio_parser > 0 or segyio_qc > 0:
            print_success(f"🔧 segyio adoption successful: Parser={segyio_parser}, QC={segyio_qc}")
        else:
            print_warning("⚠️  segyio adoption may not be complete")

        # NEW: Rate limiting assessment
        print_header("🚦 RATE LIMITING ASSESSMENT")

        if rate_limiting_success:
            rate_results = self.results["rate_limiting_tests"]
            total_tokens = rate_results.get("workflow", {}).get("total_tokens", 0)
            print_success(f"✅ Rate limiting working correctly")
            print_info(f"   Complete workflow: {total_tokens} tokens (well under 5000 limit)")

            # Show individual tool performance
            individual = rate_results.get("individual_tools", {})
            for tool, data in individual.items():
                tokens = data.get("tokens", 0)
                safe = data.get("safe", False)
                print_info(f"   {tool.capitalize()}: {tokens} tokens ({'✓' if safe else '✗'})")
        else:
            print_error("❌ Rate limiting issues detected - review above")

        # NEW: MCP tools assessment
        if mcp_tools_success:
            print_success("✅ MCP tools working correctly")
            mcp_results = self.results["mcp_tools_tests"]
            polygon_tests = sum(
                1 for r in mcp_results.get("polygon_extraction", {}).values() if r.get("success", False))
            trace_tests = sum(1 for r in mcp_results.get("trace_extraction", {}).values() if r.get("success", False))
            print_info(f"   Polygon extraction: {polygon_tests} test scenarios passed")
            print_info(f"   Trace extraction: {trace_tests} test scenarios passed")
        else:
            print_error("❌ MCP tools issues detected - review above")

        # Overall assessment
        all_core_tests_passed = (
                metadata_harvester_success and
                rate_limiting_success and
                mcp_tools_success and
                self.results["integration_tests"].get("success", False)
        )

        print_header("🎉 OVERALL ASSESSMENT")

        if all_core_tests_passed:
            print_success("✅ ALL SYSTEMS VALIDATED SUCCESSFULLY!")
            print_info("✨ Your segyio-based transformation is working correctly")
            print_info("📊 Metadata harvester is functioning properly")
            print_info("🚦 Rate limiting is protecting against token overruns")
            print_info("🔧 MCP tools are ready for production use")
            print_info("🏆 Quality ratings should now be more realistic")
            print_info("🎯 Shot gather detection should be improved")

            # Performance summary
            perf = self.results.get("performance_tests", {})
            if perf:
                parser_speed = perf.get("traces_per_sec_parser", 0)
                mb_speed = perf.get("mb_per_sec", 0)
                print_info(f"⚡ Performance: {parser_speed:.0f} traces/sec, {mb_speed:.1f} MB/sec")

        else:
            print_error("❌ VALIDATION FAILED - Issues detected:")
            if not metadata_harvester_success:
                print_error("   📊 Metadata harvester problems")
            if not rate_limiting_success:
                print_error("   🚦 Rate limiting problems")
            if not mcp_tools_success:
                print_error("   🔧 MCP tools problems")
            if not self.results["integration_tests"].get("success", False):
                print_error("   🔗 Integration problems")

            print_info("👀 Review the detailed test results above to identify specific issues")

        # Store final summary
        self.results["summary"] = {
            "overall_success": all_core_tests_passed,
            "files_tested": len(self.test_files),
            "components_validated": {
                "parser": parser_success > 0,
                "classifier": classifier_success > 0,
                "qc": qc_tests > 0,
                "metadata_harvester": metadata_harvester_success,
                "rate_limiting": rate_limiting_success,
                "mcp_tools": mcp_tools_success,
                "integration": self.results["integration_tests"].get("success", False)
            }
        }


def main():
    """Main test execution with enhanced options"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive SEG-Y workflow + rate limiting validation")
    parser.add_argument("--data-dir", default="./data", help="Data directory path")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to test")
    parser.add_argument("--file-size-limit", type=float, default=None, help="Maximum file size in MB to test")
    parser.add_argument("--extensions", nargs="+", default=None, help="Custom SEG-Y extensions to search")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--list-only", action="store_true", help="Only list discovered files, don't run tests")
    parser.add_argument("--test-metadata-only", action="store_true", help="Only test metadata harvester functionality")
    parser.add_argument("--test-rate-limiting-only", action="store_true", help="Only test rate limiting functionality")
    parser.add_argument("--test-mcp-only", action="store_true", help="Only test MCP tools functionality")
    parser.add_argument("--skip-integration", action="store_true", help="Skip integration tests")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with minimal scenarios")

    args = parser.parse_args()

    # Create tester
    tester = SEGYTransformationTester(args.data_dir)

    # Override extensions if provided
    if args.extensions:
        tester.segy_extensions = args.extensions
        tester.test_files = tester._discover_segy_files()

    # Apply file limits
    if args.max_files:
        tester.test_files = tester.test_files[:args.max_files]
        print_info(f"Limited to first {args.max_files} files")

    if args.file_size_limit:
        filtered_files = []
        for filename in tester.test_files:
            filepath = tester.data_dir / filename
            size_mb = filepath.stat().st_size / (1024 * 1024)
            if size_mb <= args.file_size_limit:
                filtered_files.append(filename)
        tester.test_files = filtered_files
        print_info(f"Limited to files <= {args.file_size_limit}MB")

    # List only mode
    if args.list_only:
        print_header("DISCOVERED SEG-Y FILES")
        for filename in tester.test_files:
            filepath = tester.data_dir / filename
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print_info(f"{filename} ({size_mb:.1f}MB)")
        print_info(f"Total: {len(tester.test_files)} files")
        return

    # Specialized testing modes
    available_files = tester._check_test_files()

    if args.test_metadata_only:
        print_header("METADATA HARVESTER ONLY TESTING")
        success = tester._test_metadata_harvester(available_files)
        print_info(f"Result: {'✓ PASSED' if success else '✗ FAILED'}")
        sys.exit(0 if success else 1)

    if args.test_rate_limiting_only:
        print_header("RATE LIMITING ONLY TESTING")
        success = tester._test_rate_limiting(available_files)
        print_info(f"Result: {'✓ PASSED' if success else '✗ FAILED'}")
        sys.exit(0 if success else 1)

    if args.test_mcp_only:
        print_header("MCP TOOLS ONLY TESTING")
        success = tester._test_mcp_tools(available_files)
        print_info(f"Result: {'✓ PASSED' if success else '✗ FAILED'}")
        sys.exit(0 if success else 1)

    # Quick test mode
    if args.quick_test:
        print_info("Running quick test mode - minimal scenarios")
        tester.test_files = tester.test_files[:1]  # Only test first file

    # Run full test suite
    success = tester.run_all_tests()

    # Save results
    results_file = "segyio_transformation_test_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(tester.results, f, indent=2)
        print_info(f"📄 Detailed results saved to: {results_file}")
    except Exception as e:
        print_warning(f"Could not save results: {e}")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()