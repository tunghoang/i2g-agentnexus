#!/usr/bin/env python3
"""
comprehensive_test_script.py - Updated for Current Codebase

This script tests all major components including:
1. Import validation for both LAS and SEG-Y tools
2. MCP server tool registration and functionality
3. File discovery and validation
4. LAS tools testing
5. SEG-Y tools testing (including metadata harvester)
6. Rate limiting validation
7. Integration testing

Run this to validate the entire platform.
"""

import sys
import json
import time
import traceback
import os
import requests
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
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")


def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")


def print_info(msg):
    print(f"{Colors.BLUE}  {msg}{Colors.END}")


def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}")
    print(f"{msg}")
    print(f"{'=' * 60}{Colors.END}")


class SubsurfaceDataTester:
    """Comprehensive tester for the entire subsurface data platform"""

    def __init__(self, data_dir="./data", mcp_url="http://localhost:7000"):
        self.data_dir = Path(data_dir)
        self.mcp_url = mcp_url

        # Initialize results dictionary
        self.results = {
            "environment_tests": {},
            "import_tests": {},
            "file_discovery": {},
            "mcp_server_tests": {},
            "las_tools_tests": {},
            "segy_tools_tests": {},
            "metadata_harvester_tests": {},
            "rate_limiting_tests": {},
            "integration_tests": {},
            "performance_tests": {},
            "summary": {}
        }

        # File discovery
        self.las_extensions = ["*.las", "*.LAS", "*.Las"]
        self.segy_extensions = ["*.segy", "*.sgy", "*.SGY", "*.SEGY", "*.seg", "*.SEG"]

        self.las_files = []
        self.segy_files = []

    def run_all_tests(self):
        """Run comprehensive test suite"""
        print_header("SUBSURFACE DATA MANAGEMENT PLATFORM - COMPREHENSIVE TESTING")
        print(f"Data directory: {self.data_dir}")
        print(f"MCP server: {self.mcp_url}")

        # Test sequence
        tests = [
            ("Environment Validation", self._test_environment),
            ("Import Tests", self._test_imports),
            ("File Discovery", self._test_file_discovery),
            ("MCP Server Tests", self._test_mcp_server),
            ("LAS Tools Tests", self._test_las_tools),
            ("SEG-Y Tools Tests", self._test_segy_tools),
            ("Metadata Harvester Tests", self._test_metadata_harvester),
            ("Rate Limiting Tests", self._test_rate_limiting),
            ("Integration Tests", self._test_integration),
            ("Performance Tests", self._test_performance)
        ]

        overall_success = True

        for test_name, test_func in tests:
            print_header(test_name)
            try:
                success = test_func()
                if not success:
                    overall_success = False
                    print_error(f"{test_name} failed")
                else:
                    print_success(f"{test_name} passed")
            except Exception as e:
                print_error(f"{test_name} failed with exception: {e}")
                traceback.print_exc()
                overall_success = False

        # Generate summary
        self._generate_summary()

        return overall_success

    def _test_environment(self):
        """Test environment setup"""
        print_info("Testing environment setup...")

        success = True
        env_results = {}

        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print_info(f"Python version: {python_version}")
        env_results["python_version"] = python_version

        # Check required directories
        directories = ["./data", "./logs", "./templates"]
        for directory in directories:
            exists = os.path.exists(directory)
            env_results[f"directory_{directory.replace('./', '')}"] = exists
            if exists:
                print_success(f"Directory exists: {directory}")
            else:
                print_warning(f"Directory missing: {directory}")

        # Check current working directory
        cwd = os.getcwd()
        print_info(f"Current working directory: {cwd}")
        env_results["working_directory"] = cwd

        # Check if we're in the right project structure
        expected_files = ["main.py", "production_segy_tools.py", "tools/las_tools.py"]
        for file in expected_files:
            exists = os.path.exists(file)
            env_results[f"file_{file.replace('/', '_').replace('.', '_')}"] = exists
            if exists:
                print_success(f"Found: {file}")
            else:
                print_error(f"Missing: {file}")
                success = False

        self.results["environment_tests"] = env_results
        return success

    def _test_imports(self):
        """Test that all modules can be imported"""
        print_info("Testing module imports...")

        import_tests = {
            # Core modules
            "production_segy_tools": "import production_segy_tools",
            "tools.las_tools": "from tools import las_tools",
            "survey_classifier": "import survey_classifier",
            "production_segy_analysis_qc": "import production_segy_analysis_qc",

            # Specific functions - FIXED NAMES
            "segy_parser": "from production_segy_tools import production_segy_parser",
            "segy_metadata_harvester": "from production_segy_tools import segy_complete_metadata_harvester",
            "enhanced_las_parser": "from tools.las_tools import enhanced_las_parser",
            "enhanced_las_analysis": "from tools.las_tools import enhanced_las_analysis",

            # Required libraries
            "segyio": "import segyio",
            "lasio": "import lasio",
            "numpy": "import numpy as np",
            "pandas": "import pandas as pd",
            "scipy": "import scipy",
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

    def _test_file_discovery(self):
        """Discover and validate data files"""
        print_info("Discovering data files...")

        if not self.data_dir.exists():
            print_error(f"Data directory does not exist: {self.data_dir}")
            return False

        # Discover LAS files
        las_files = []
        for extension in self.las_extensions:
            las_files.extend(list(self.data_dir.glob(extension)))

        # Discover SEG-Y files
        segy_files = []
        for extension in self.segy_extensions:
            segy_files.extend(list(self.data_dir.glob(extension)))

        # Remove duplicates and get names
        self.las_files = list(set([f.name for f in las_files if f.is_file()]))
        self.segy_files = list(set([f.name for f in segy_files if f.is_file()]))

        print_info(f"Found {len(self.las_files)} LAS files:")
        for filename in self.las_files[:5]:  # Show first 5
            filepath = self.data_dir / filename
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print_info(f"  {filename} ({size_mb:.1f}MB)")
        if len(self.las_files) > 5:
            print_info(f"  ... and {len(self.las_files) - 5} more")

        print_info(f"Found {len(self.segy_files)} SEG-Y files:")
        for filename in self.segy_files[:5]:  # Show first 5
            filepath = self.data_dir / filename
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print_info(f"  {filename} ({size_mb:.1f}MB)")
        if len(self.segy_files) > 5:
            print_info(f"  ... and {len(self.segy_files) - 5} more")

        # Store discovery results
        self.results["file_discovery"] = {
            "las_files_found": len(self.las_files),
            "segy_files_found": len(self.segy_files),
            "las_files": self.las_files[:10],  # Store first 10
            "segy_files": self.segy_files[:10]  # Store first 10
        }

        return len(self.las_files) > 0 or len(self.segy_files) > 0

    def _test_mcp_server(self):
        """Test MCP server functionality"""
        print_info("Testing MCP server...")

        mcp_results = {}
        success = True

        # Test server connectivity
        try:
            response = requests.get(f"{self.mcp_url}/tools", timeout=5)
            if response.status_code == 200:
                print_success("MCP server is responding")
                mcp_results["server_responding"] = True

                # Parse tools list
                try:
                    tools = response.json()
                    tool_count = len(tools)
                    print_info(f"Found {tool_count} registered tools")

                    # Check for specific tools
                    tool_names = [tool.get("name", "") for tool in tools]

                    expected_tools = [
                        "las_parser", "las_analysis", "las_qc",
                        "segy_parser", "segy_complete_metadata_harvester",
                        "segy_survey_polygon", "segy_trace_outlines",
                        "list_files", "system_status"
                    ]

                    found_tools = {}
                    for tool in expected_tools:
                        found = tool in tool_names
                        found_tools[tool] = found
                        if found:
                            print_success(f"  Tool registered: {tool}")
                        else:
                            print_warning(f"  Tool missing: {tool}")

                    mcp_results["tools_registered"] = tool_count
                    mcp_results["expected_tools"] = found_tools
                    mcp_results["all_expected_tools_found"] = all(found_tools.values())

                except Exception as e:
                    print_error(f"Failed to parse tools response: {e}")
                    success = False

            else:
                print_error(f"MCP server returned status {response.status_code}")
                mcp_results["server_responding"] = False
                success = False

        except requests.exceptions.RequestException as e:
            print_error(f"Cannot connect to MCP server: {e}")
            mcp_results["server_responding"] = False
            success = False

        # Test a simple tool call if server is responding
        if mcp_results.get("server_responding", False):
            try:
                print_info("Testing system_status tool...")
                response = requests.post(
                    f"{self.mcp_url}/tools/system_status",
                    json={},
                    timeout=10
                )
                if response.status_code == 200:
                    print_success("system_status tool working")
                    mcp_results["test_tool_call"] = True
                else:
                    print_warning(f"system_status returned {response.status_code}")
                    mcp_results["test_tool_call"] = False
            except Exception as e:
                print_warning(f"Test tool call failed: {e}")
                mcp_results["test_tool_call"] = False

        self.results["mcp_server_tests"] = mcp_results
        return success

    def _test_las_tools(self):
        """Test LAS tools functionality"""
        print_info("Testing LAS tools...")

        if not self.las_files:
            print_warning("No LAS files found, skipping LAS tools tests")
            return True

        las_results = {}
        success = True

        # Test LAS parser
        test_file = self.las_files[0]
        print_info(f"Testing LAS parser with: {test_file}")

        try:
            # Test via MCP server
            response = requests.post(
                f"{self.mcp_url}/tools/las_parser",
                json={"file_path": test_file},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if "error" not in result:
                    print_success("LAS parser working via MCP")
                    las_results["parser_mcp"] = True

                    # Parse the result
                    if "text" in result:
                        try:
                            data = json.loads(result["text"])
                            curves_found = data.get("curves_found", 0)
                            print_info(f"  Found {curves_found} curves")
                            las_results["curves_found"] = curves_found
                        except:
                            print_warning("  Could not parse LAS result JSON")
                else:
                    print_error(f"LAS parser error: {result.get('error')}")
                    success = False
            else:
                print_error(f"LAS parser MCP call failed: {response.status_code}")
                success = False

        except Exception as e:
            print_error(f"LAS parser test failed: {e}")
            success = False

        # Test LAS analysis if parser worked
        if las_results.get("parser_mcp", False):
            try:
                print_info("Testing LAS analysis...")
                response = requests.post(
                    f"{self.mcp_url}/tools/las_analysis",
                    json={"file_path": test_file},
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    if "error" not in result:
                        print_success("LAS analysis working")
                        las_results["analysis_mcp"] = True
                    else:
                        print_warning(f"LAS analysis error: {result.get('error')}")
                else:
                    print_warning(f"LAS analysis failed: {response.status_code}")

            except Exception as e:
                print_warning(f"LAS analysis test failed: {e}")

        self.results["las_tools_tests"] = las_results
        return success

    def _test_segy_tools(self):
        """Test SEG-Y tools functionality"""
        print_info("Testing SEG-Y tools...")

        if not self.segy_files:
            print_warning("No SEG-Y files found, skipping SEG-Y tools tests")
            return True

        segy_results = {}
        success = True

        # Test SEG-Y parser
        test_file = self.segy_files[0]
        print_info(f"Testing SEG-Y parser with: {test_file}")

        try:
            # Test via MCP server
            response = requests.post(
                f"{self.mcp_url}/tools/segy_parser",
                json={"file_path": test_file},
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                if "error" not in result:
                    print_success("SEG-Y parser working via MCP")
                    segy_results["parser_mcp"] = True

                    # Parse the result
                    if "text" in result:
                        try:
                            data = json.loads(result["text"])
                            total_traces = data.get("total_traces", 0)
                            print_info(f"  Found {total_traces} traces")
                            segy_results["total_traces"] = total_traces
                        except Exception as e:
                            print_warning(f"  Could not parse SEG-Y result JSON: {e}")
                else:
                    print_error(f"SEG-Y parser error: {result.get('error')}")
                    success = False
            else:
                print_error(f"SEG-Y parser MCP call failed: {response.status_code}")
                success = False

        except Exception as e:
            print_error(f"SEG-Y parser test failed: {e}")
            success = False

        # Test SEG-Y QC
        if segy_results.get("parser_mcp", False):
            try:
                print_info("Testing SEG-Y QC...")
                response = requests.post(
                    f"{self.mcp_url}/tools/segy_qc",
                    json={"file_path": test_file},
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    if "error" not in result:
                        print_success("SEG-Y QC working")
                        segy_results["qc_mcp"] = True
                    else:
                        print_warning(f"SEG-Y QC error: {result.get('error')}")
                else:
                    print_warning(f"SEG-Y QC failed: {response.status_code}")

            except Exception as e:
                print_warning(f"SEG-Y QC test failed: {e}")

        self.results["segy_tools_tests"] = segy_results
        return success

    def _test_metadata_harvester(self):
        """Test the SEG-Y metadata harvester specifically"""
        print_info("Testing SEG-Y metadata harvester...")

        if not self.segy_files:
            print_warning("No SEG-Y files found, skipping metadata harvester tests")
            return True

        harvester_results = {}
        success = True

        test_file = self.segy_files[0]
        print_info(f"Testing metadata harvester with: {test_file}")

        # Test different scenarios
        scenarios = [
            {
                "name": "default",
                "params": {"file_path": test_file}
            },
            {
                "name": "rate_limited",
                "params": {
                    "file_path": test_file,
                    "trace_sample_size": 5,
                    "include_statistics": False,
                    "max_text_length": 2000,
                    "return_format": "summary"
                }
            },
            {
                "name": "simplified",
                "params": {
                    "file_path": test_file,
                    "return_format": "simplified",
                    "max_text_length": 3000
                }
            }
        ]

        for scenario in scenarios:
            scenario_name = scenario["name"]
            params = scenario["params"]

            print_info(f"  Testing scenario: {scenario_name}")

            try:
                response = requests.post(
                    f"{self.mcp_url}/tools/segy_complete_metadata_harvester",
                    json=params,
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()

                    if "error" not in result:
                        # Handle both response formats
                        if "text" in result:
                            # Direct format: {'text': '...', 'metadata': {...}}
                            text_content = result["text"]
                        elif "content" in result and not result.get("isError", True):
                            # MCP wrapped format: {'content': [...], 'isError': False}
                            content = result["content"]
                            if isinstance(content, list) and len(content) > 0:
                                # Extract text from MCP response
                                if isinstance(content[0], dict) and "text" in content[0]:
                                    text_content = content[0]["text"]
                                elif isinstance(content[0], str):
                                    text_content = content[0]
                                else:
                                    text_content = str(content[0])
                            else:
                                text_content = ""
                        else:
                            print_error(f"    {scenario_name}: Unexpected response format")
                            success = False
                            continue

                        tokens = len(text_content) // 4

                        # Validate JSON if we have content
                        if text_content:
                            try:
                                json.loads(text_content)
                                json_valid = True
                            except:
                                json_valid = False
                        else:
                            json_valid = False

                        print_success(
                            f"    {scenario_name}: {len(text_content)} chars, {tokens} tokens, JSON: {json_valid}")

                        harvester_results[scenario_name] = {
                            "success": True,
                            "text_length": len(text_content),
                            "tokens": tokens,
                            "json_valid": json_valid
                        }

                        if not json_valid and len(text_content) > 0:
                            print_warning(f"    {scenario_name}: Non-empty response but invalid JSON")
                            success = False

                    else:
                        print_error(f"    {scenario_name}: {result.get('error')}")
                        harvester_results[scenario_name] = {
                            "success": False,
                            "error": result.get("error")
                        }
                        success = False

                else:
                    print_error(f"    {scenario_name}: HTTP {response.status_code}")
                    success = False

            except Exception as e:
                print_error(f"    {scenario_name}: Exception - {e}")
                success = False

        self.results["metadata_harvester_tests"] = harvester_results
        return success

    def _test_rate_limiting(self):
        """Test rate limiting across all tools"""
        print_info("Testing rate limiting...")

        if not self.segy_files:
            print_warning("No SEG-Y files found, skipping rate limiting tests")
            return True

        rate_results = {}
        success = True

        test_file = self.segy_files[0]
        print_info(f"Testing rate limiting with: {test_file}")

        # Test individual tool rate limits
        tools_to_test = [
            ("metadata_harvester", "segy_complete_metadata_harvester", {
                "file_path": test_file,
                "return_format": "summary",
                "max_text_length": 2000
            }),
            ("survey_polygon", "segy_survey_polygon", {
                "file_path": test_file,
                "return_format": "summary"
            }),
            ("trace_outlines", "segy_trace_outlines", {
                "file_path": test_file,
                "return_format": "summary",
                "max_traces": 5
            })
        ]

        total_tokens = 0
        for tool_name, endpoint, params in tools_to_test:
            try:
                print_info(f"  Testing {tool_name}...")

                response = requests.post(
                    f"{self.mcp_url}/tools/{endpoint}",
                    json=params,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    if "error" not in result:
                        tokens = len(json.dumps(result)) // 4
                        total_tokens += tokens

                        rate_safe = tokens < 1000  # Individual tool limit

                        print_info(f"    {tool_name}: {tokens} tokens ({'✓' if rate_safe else '✗'})")

                        rate_results[tool_name] = {
                            "tokens": tokens,
                            "rate_safe": rate_safe
                        }

                        if not rate_safe:
                            success = False
                    else:
                        print_warning(f"    {tool_name}: {result.get('error')}")
                        success = False

            except Exception as e:
                print_warning(f"    {tool_name}: {e}")
                success = False

        # Check total workflow rate limit
        workflow_safe = total_tokens < 5000
        print_info(f"  Total workflow: {total_tokens} tokens ({'✓' if workflow_safe else '✗'})")

        rate_results["workflow"] = {
            "total_tokens": total_tokens,
            "rate_safe": workflow_safe
        }

        if not workflow_safe:
            success = False

        self.results["rate_limiting_tests"] = rate_results
        return success

    def _test_integration(self):
        """Test integration between tools"""
        print_info("Testing tool integration...")

        if not self.las_files and not self.segy_files:
            print_warning("No data files found, skipping integration tests")
            return True

        integration_results = {}
        success = True

        # Test file listing integration
        try:
            print_info("Testing file listing...")
            response = requests.post(
                f"{self.mcp_url}/tools/list_files",
                json={},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if "error" not in result:
                    print_success("File listing working")
                    integration_results["file_listing"] = True

                    # Check if it found our files
                    text = result.get("text", "")
                    las_mentioned = any(las_file in text for las_file in self.las_files[:3])
                    segy_mentioned = any(segy_file in text for segy_file in self.segy_files[:3])

                    print_info(f"  LAS files mentioned: {las_mentioned}")
                    print_info(f"  SEG-Y files mentioned: {segy_mentioned}")

                else:
                    print_warning(f"File listing error: {result.get('error')}")
            else:
                print_warning(f"File listing failed: {response.status_code}")

        except Exception as e:
            print_warning(f"File listing test failed: {e}")

        # Test system status
        try:
            print_info("Testing system status...")
            response = requests.post(
                f"{self.mcp_url}/tools/system_status",
                json={},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if "error" not in result:
                    print_success("System status working")
                    integration_results["system_status"] = True
                else:
                    print_warning(f"System status error: {result.get('error')}")
            else:
                print_warning(f"System status failed: {response.status_code}")

        except Exception as e:
            print_warning(f"System status test failed: {e}")

        # Test workflow: Parse -> Analyze
        if self.las_files:
            test_las_file = self.las_files[0]
            try:
                print_info(f"Testing LAS workflow with {test_las_file}...")

                # Step 1: Parse
                parse_response = requests.post(
                    f"{self.mcp_url}/tools/las_parser",
                    json={"file_path": test_las_file},
                    timeout=30
                )

                # Step 2: Analyze
                analysis_response = requests.post(
                    f"{self.mcp_url}/tools/las_analysis",
                    json={"file_path": test_las_file},
                    timeout=30
                )

                if parse_response.status_code == 200 and analysis_response.status_code == 200:
                    parse_result = parse_response.json()
                    analysis_result = analysis_response.json()

                    if "error" not in parse_result and "error" not in analysis_result:
                        print_success("LAS workflow integration working")
                        integration_results["las_workflow"] = True
                    else:
                        print_warning("LAS workflow has errors")

            except Exception as e:
                print_warning(f"LAS workflow test failed: {e}")

        self.results["integration_tests"] = integration_results
        return success

    def _test_performance(self):
        """Test performance of tools"""
        print_info("Testing performance...")

        performance_results = {}
        success = True

        # Test file listing performance
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.mcp_url}/tools/list_files",
                json={},
                timeout=30
            )
            list_time = time.time() - start_time

            print_info(f"File listing time: {list_time:.2f}s")
            performance_results["file_listing_time"] = list_time

        except Exception as e:
            print_warning(f"File listing performance test failed: {e}")

        # Test parser performance if we have files
        if self.las_files:
            test_file = self.las_files[0]
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.mcp_url}/tools/las_parser",
                    json={"file_path": test_file},
                    timeout=60
                )
                parse_time = time.time() - start_time

                print_info(f"LAS parser time: {parse_time:.2f}s")
                performance_results["las_parser_time"] = parse_time

            except Exception as e:
                print_warning(f"LAS parser performance test failed: {e}")

        if self.segy_files:
            test_file = self.segy_files[0]
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.mcp_url}/tools/segy_parser",
                    json={"file_path": test_file},
                    timeout=120
                )
                parse_time = time.time() - start_time

                print_info(f"SEG-Y parser time: {parse_time:.2f}s")
                performance_results["segy_parser_time"] = parse_time

            except Exception as e:
                print_warning(f"SEG-Y parser performance test failed: {e}")

        self.results["performance_tests"] = performance_results
        return success

    def _generate_summary(self):
        """Generate comprehensive test summary"""
        print_header("COMPREHENSIVE TEST SUMMARY")

        # Environment
        env_success = all(self.results["environment_tests"].values())
        print_info(f"🌍 Environment: {'✓' if env_success else '✗'}")

        # Imports
        import_success = sum(self.results["import_tests"].values())
        import_total = len(self.results["import_tests"])
        print_info(f"📦 Imports: {import_success}/{import_total}")

        # File discovery
        las_files = self.results["file_discovery"].get("las_files_found", 0)
        segy_files = self.results["file_discovery"].get("segy_files_found", 0)
        print_info(f"📁 Files: {las_files} LAS, {segy_files} SEG-Y")

        # MCP server
        mcp_success = self.results["mcp_server_tests"].get("server_responding", False)
        tools_registered = self.results["mcp_server_tests"].get("tools_registered", 0)
        print_info(f"🔧 MCP Server: {'✓' if mcp_success else '✗'} ({tools_registered} tools)")

        # LAS tools
        las_parser_success = self.results["las_tools_tests"].get("parser_mcp", False)
        las_analysis_success = self.results["las_tools_tests"].get("analysis_mcp", False)
        print_info(
            f"📊 LAS Tools: Parser {'✓' if las_parser_success else '✗'}, Analysis {'✓' if las_analysis_success else '✗'}")

        # SEG-Y tools
        segy_parser_success = self.results["segy_tools_tests"].get("parser_mcp", False)
        segy_qc_success = self.results["segy_tools_tests"].get("qc_mcp", False)
        print_info(f"🌊 SEG-Y Tools: Parser {'✓' if segy_parser_success else '✗'}, QC {'✓' if segy_qc_success else '✗'}")

        # Metadata harvester
        metadata_scenarios = self.results["metadata_harvester_tests"]
        metadata_success = all(
            scenario.get("success", False) and scenario.get("json_valid", False)
            for scenario in metadata_scenarios.values()
            if isinstance(scenario, dict)
        )
        print_info(f"📋 Metadata Harvester: {'✓' if metadata_success else '✗'}")

        # Rate limiting
        rate_success = self.results["rate_limiting_tests"].get("workflow", {}).get("rate_safe", False)
        total_tokens = self.results["rate_limiting_tests"].get("workflow", {}).get("total_tokens", 0)
        print_info(f"🚦 Rate Limiting: {'✓' if rate_success else '✗'} ({total_tokens} tokens)")

        # Integration
        integration_success = any(self.results["integration_tests"].values())
        print_info(f"🔗 Integration: {'✓' if integration_success else '✗'}")

        # Performance
        performance_data = self.results["performance_tests"]
        avg_time = sum(t for t in performance_data.values() if isinstance(t, (int, float))) / max(len(performance_data),
                                                                                                  1)
        print_info(f"⚡ Performance: Avg {avg_time:.2f}s per operation")

        print_header("DETAILED DIAGNOSTICS")

        # Show specific issues
        if not mcp_success:
            print_error("MCP Server not responding - check if server is running on localhost:7000")

        if not metadata_success:
            print_error("Metadata harvester issues detected:")
            for scenario, result in metadata_scenarios.items():
                if isinstance(result, dict) and not result.get("success", False):
                    print_error(f"  {scenario}: {result.get('error', 'Failed')}")
                elif isinstance(result, dict) and not result.get("json_valid", True):
                    print_error(f"  {scenario}: Invalid JSON output")

        if not rate_success:
            print_error(f"Rate limiting issues - workflow uses {total_tokens} tokens (limit: 5000)")

        # Show working components
        working_components = []
        if env_success:
            working_components.append("Environment")
        if import_success == import_total:
            working_components.append("All imports")
        if mcp_success:
            working_components.append("MCP server")
        if las_parser_success:
            working_components.append("LAS tools")
        if segy_parser_success:
            working_components.append("SEG-Y tools")
        if metadata_success:
            working_components.append("Metadata harvester")
        if rate_success:
            working_components.append("Rate limiting")

        if working_components:
            print_success("Working components: " + ", ".join(working_components))

        # Overall assessment
        all_critical_working = (
                env_success and
                mcp_success and
                (las_parser_success or segy_parser_success) and
                metadata_success and
                rate_success
        )

        print_header("OVERALL ASSESSMENT")

        if all_critical_working:
            print_success("🎉 ALL CRITICAL SYSTEMS WORKING!")
            print_info("✨ Platform is ready for production use")
            print_info("📋 Metadata harvester is functioning correctly")
            print_info("🚦 Rate limiting is protecting against overruns")
            print_info("🔧 MCP tools are properly registered and responding")

            if las_files > 0 and segy_files > 0:
                print_info("📊 Both LAS and SEG-Y processing capabilities confirmed")
            elif las_files > 0:
                print_info("📊 LAS processing capabilities confirmed")
            elif segy_files > 0:
                print_info("🌊 SEG-Y processing capabilities confirmed")
        else:
            print_error("❌ CRITICAL ISSUES DETECTED")
            print_info("Review the detailed diagnostics above")
            print_info("Most common issues:")
            print_info("  1. MCP server not running (start with python main.py)")
            print_info("  2. Missing data files in ./data directory")
            print_info("  3. Import errors due to missing dependencies")
            print_info("  4. Rate limiting not properly implemented")

        # Store final summary
        self.results["summary"] = {
            "overall_success": all_critical_working,
            "critical_components": {
                "environment": env_success,
                "imports": import_success == import_total,
                "mcp_server": mcp_success,
                "las_tools": las_parser_success,
                "segy_tools": segy_parser_success,
                "metadata_harvester": metadata_success,
                "rate_limiting": rate_success,
                "integration": integration_success
            },
            "files_available": {
                "las_files": las_files,
                "segy_files": segy_files
            },
            "performance": {
                "average_operation_time": avg_time,
                "total_tokens_workflow": total_tokens
            }
        }


def main():
    """Main test execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Subsurface Data Platform Testing")
    parser.add_argument("--data-dir", default="./data", help="Data directory path")
    parser.add_argument("--mcp-url", default="http://localhost:7000", help="MCP server URL")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--test-metadata-only", action="store_true", help="Only test metadata harvester")
    parser.add_argument("--test-las-only", action="store_true", help="Only test LAS tools")
    parser.add_argument("--test-segy-only", action="store_true", help="Only test SEG-Y tools")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Create tester
    tester = SubsurfaceDataTester(args.data_dir, args.mcp_url)

    # Specific test modes
    if args.test_metadata_only:
        print_header("METADATA HARVESTER ONLY TESTING")
        success = tester._test_file_discovery() and tester._test_metadata_harvester()
        print_info(f"Result: {'✓ PASSED' if success else '✗ FAILED'}")
        sys.exit(0 if success else 1)

    if args.test_las_only:
        print_header("LAS TOOLS ONLY TESTING")
        success = (tester._test_environment() and
                   tester._test_imports() and
                   tester._test_file_discovery() and
                   tester._test_mcp_server() and
                   tester._test_las_tools())
        print_info(f"Result: {'✓ PASSED' if success else '✗ FAILED'}")
        sys.exit(0 if success else 1)

    if args.test_segy_only:
        print_header("SEG-Y TOOLS ONLY TESTING")
        success = (tester._test_environment() and
                   tester._test_imports() and
                   tester._test_file_discovery() and
                   tester._test_mcp_server() and
                   tester._test_segy_tools())
        print_info(f"Result: {'✓ PASSED' if success else '✗ FAILED'}")
        sys.exit(0 if success else 1)

    # Full test suite
    if args.quick:
        print_info("Running quick test mode")
        # Run essential tests only
        success = (tester._test_environment() and
                   tester._test_imports() and
                   tester._test_file_discovery() and
                   tester._test_mcp_server())

        if success:
            print_success("Quick tests passed - basic functionality working")
        else:
            print_error("Quick tests failed - check basic setup")

        sys.exit(0 if success else 1)

    # Run full comprehensive test suite
    success = tester.run_all_tests()

    # Save detailed results
    results_file = "comprehensive_test_results.json"
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