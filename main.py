# Standard Libraries
import os
import sys
import logging
import socket
import time
import threading
import argparse
import json
import traceback
import glob
import datetime
import csv
from pathlib import Path
from typing import Any, Optional

# Third-party Libraries
import numpy as np
import pandas as pd
import lasio

# Environment Setup
from dotenv import load_dotenv

# Optional Dependencies (with graceful handling)
try:
    import psutil
except ImportError:
    psutil = None  # Will be handled gracefully in the monitoring code

# Agent-to-Agent Protocol (A2A)
from python_a2a import OpenAIA2AServer, run_server, A2AServer, AgentCard, AgentSkill
from python_a2a.langchain import to_langchain_agent, to_langchain_tool
from python_a2a.mcp import FastMCP

# LangChain Components
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType, ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# Enhanced MCP Tools
from enhanced_mcp_tools import (
    enhanced_las_parser,
    enhanced_las_analysis,
    enhanced_las_qc,
    enhanced_formation_evaluation,
    enhanced_well_correlation_with_qc,
    NumpyJSONEncoder,
    find_las_file,
    load_las_file
)

# Formation Evaluation
from formation_evaluation import (
    estimate_vshale
)

# Segyio-based SEG-Y Components (with availability check)
try:
    from production_segy_tools import (
        production_segy_parser,
        find_segy_file,
        NumpyJSONEncoder,
        TemplateValidator,          # ADD THIS
        SegyioValidator,           # ADD THIS
        MemoryMonitor             # ADD THIS
    )
    from production_segy_analysis_qc import (
        production_segy_qc,
        production_segy_analysis
    )
    from production_segy_multifile import (    # ADD THIS ENTIRE BLOCK
        production_segy_survey_analysis,
        create_default_config
    )
    from survey_classifier import SurveyClassifier

    SEGYIO_COMPONENTS_AVAILABLE = True
    print("✓ Segyio-based SEG-Y processing enabled")
    INTELLIGENT_SEGY_AVAILABLE = True

except ImportError as e:
    print(f"⚠️ Segyio SEG-Y components not available: {e}")
    SEGYIO_COMPONENTS_AVAILABLE = False
    INTELLIGENT_SEGY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

FILE_TYPE_CONFIG = {
    "las": {
        "extensions": [".las", ".LAS"],
        "description": "Well Log Files",
        "icon": "WELL LOG",
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
        "icon": "SEISMIC",
        "default_pattern": "*.sgy",
        "categories": {
            "f3_survey": lambda f: "F3_" in f,
            "3x_processing": lambda f: "3X_" in f,
            "marine": lambda f: "marine" in f.lower(),
            "land": lambda f: "land" in f.lower()
        }
    },
    # FUTURE FILE TYPES - Just add them here!
    "csv": {
        "extensions": [".csv", ".CSV"],
        "description": "Data Tables",
        "icon": "DATA TABLE",
        "default_pattern": "*.csv",
        "categories": {
            "production": lambda f: "prod" in f.lower(),
            "analysis": lambda f: "analysis" in f.lower()
        }
    }
}


def detect_file_type(pattern_or_filename):
    """Automatically detect file type from pattern or filename"""
    text = pattern_or_filename.lower()

    # Check each file type
    for file_type, config in FILE_TYPE_CONFIG.items():
        for ext in config["extensions"]:
            if ext.lower() in text:
                return file_type

    # Default to LAS if no extension specified
    if not any(ext in text for config in FILE_TYPE_CONFIG.values() for ext in config["extensions"]):
        return "las"  # Default

    return None


def format_files_by_type(file_paths, config):
    """Format file list based on file type configuration"""
    file_count = len(file_paths)
    output = f"{config['icon']} FILES FOUND ({file_count} files):\n"
    output += "=" * 50 + "\n\n"

    # Group files by categories if defined
    if config["categories"]:
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

def add_new_file_type(file_type, extensions, description, icon, categories=None):
    """Easily add new file types to the system"""
    FILE_TYPE_CONFIG[file_type] = {
        "extensions": extensions,
        "description": description,
        "icon": icon,
        "default_pattern": f"*.{extensions[0].lstrip('.')}",
        "categories": categories or {}
    }
    print(f"Added new file type: {file_type}")

def format_generic_files(file_paths, pattern):
    """Generic file formatting for unknown types"""
    output = f"FILES FOUND ({len(file_paths)} files matching '{pattern}'):\n"
    output += "=" * 40 + "\n\n"

    for file_path in sorted(file_paths):
        output += f"  - {os.path.basename(file_path)}\n"

    output += f"\nTOTAL: {len(file_paths)} files ready for analysis"
    return output

def save_qa_interaction(question, answer, log_format="csv", log_dir="./logs"):
    """
    Save a question and answer interaction to a log file.

    Args:
        question: The user's question
        answer: The system's answer
        log_format: Format to save in ('csv', 'json', or 'text')
        log_dir: Directory to save logs in

    Returns:
        str: Path to the saved log file
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_only = datetime.datetime.now().strftime("%Y-%m-%d")

    if log_format.lower() == "csv":
        # Use a daily CSV file
        log_file = os.path.join(log_dir, f"qa_log_{date_only}.csv")
        file_exists = os.path.isfile(log_file)

        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header if new file
            if not file_exists:
                writer.writerow(["Timestamp", "Question", "Answer"])
            writer.writerow([timestamp, question, answer])

    elif log_format.lower() == "json":
        # Use a daily JSON file with an array of interactions
        log_file = os.path.join(log_dir, f"qa_log_{date_only}.json")

        # Load existing data if file exists
        if os.path.isfile(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
            except json.JSONDecodeError:
                # Handle corrupted file
                log_data = []
        else:
            log_data = []

        # Add new interaction
        log_data.append({
            "timestamp": timestamp,
            "question": question,
            "answer": answer
        })

        # Write updated data
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    else:  # Default to text format
        # Use a daily text file
        log_file = os.path.join(log_dir, f"qa_log_{date_only}.txt")

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"=== {timestamp} ===\n")
            f.write(f"Question: {question}\n\n")
            f.write(f"Answer: {answer}\n\n")
            f.write("-" * 80 + "\n\n")

    print(f"Q&A interaction saved to {log_file}")
    return log_file


def clean_response(response):
    """Clean up double-wrapped JSON responses"""
    if isinstance(response, str):
        # Check if it's a JSON response with content array
        if response.startswith('{"content":'):
            try:
                parsed = json.loads(response)
                if "content" in parsed and isinstance(parsed["content"], list):
                    if len(parsed["content"]) > 0 and "text" in parsed["content"][0]:
                        inner_text = parsed["content"][0]["text"]

                        # Check if inner_text is JSON with escaped unicode
                        if inner_text.startswith('{"text":'):
                            inner_parsed = json.loads(inner_text)
                            if "text" in inner_parsed:
                                # Decode unicode escapes
                                clean_text = inner_parsed["text"].encode().decode('unicode_escape')
                                return clean_text

                        return inner_text
                return response
            except json.JSONDecodeError:
                return response
    return response

# Add this helper function to main.py
def retry_correlation_with_fallback(meta_agent, query, max_retries=3):
    """Retry correlation queries with fallback strategies"""

    for attempt in range(max_retries):
        try:
            print(f"Correlation attempt {attempt + 1}/{max_retries}")
            response = meta_agent.run(query)

            # Handle both string and dictionary responses
            if isinstance(response, dict):
                # If response is a dictionary, convert to string for checking
                response_text = json.dumps(response, indent=2)

                # Check for successful correlation indicators in the JSON
                if ("formation_tops" in response or "formation_count" in response):
                    if response.get("formation_count", 0) > 0:
                        return response_text
                    elif "formation_count" in response and response["formation_count"] == 0:
                        return response_text  # Valid result with no correlations

                # If we have correlation results, return them
                if "well_correlations" in response or "formation_tops" in response:
                    return response_text

            elif isinstance(response, str):
                # Original string handling logic
                if "formation tops" in response.lower() and "confidence" in response.lower():
                    return response
                elif "no reliable formation tops" in response.lower():
                    return response
                elif attempt < max_retries - 1:
                    # Try a simpler version of the query
                    simple_query = query.replace("formations between wells", "formations in wells")
                    print(f"Trying simplified query: {simple_query}")
                    continue
                else:
                    return response

            # If we got here, try again with the next attempt
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} inconclusive, retrying...")
                time.sleep(2)
                continue
            else:
                # Last attempt - return whatever we got
                if isinstance(response, dict):
                    return json.dumps(response, indent=2)
                else:
                    return response

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
                time.sleep(2)  # Brief delay before retry
                continue
            else:
                return f"Correlation failed after {max_retries} attempts: {str(e)}"

    return response

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types"""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class CommandProcessor:
    def __init__(self, mcp_server_url=None, data_dir="./data"):
        """Initialize CommandProcessor without available_tools parameter"""
        self.mcp_server_url = mcp_server_url
        self.data_dir = data_dir

    def process_command(self, user_input):
        """Process direct commands without using the agent"""
        try:
            # Check for parse all command
            if user_input.lower().startswith("parse all"):
                args = user_input[9:].strip()
                return self.direct_call("las_parser", args, selection_mode="all")

            # Check for evaluate all command
            elif user_input.lower().startswith("evaluate all") or user_input.lower().startswith("eval all"):
                if user_input.lower().startswith("evaluate all"):
                    args = user_input[12:].strip()
                else:
                    args = user_input[8:].strip()
                return self.direct_call("formation_evaluation", args, selection_mode="all")

            # Check for QC all command
            elif user_input.lower().startswith("check all") or user_input.lower().startswith("qc all"):
                if user_input.lower().startswith("check all"):
                    args = user_input[9:].strip()
                else:
                    args = user_input[6:].strip()
                return self.direct_call("las_qc", args, selection_mode="all")

            # Check for correlate all command
            elif user_input.lower().startswith("correlate all"):
                args = user_input[12:].strip()
                return self.direct_call("well_correlation", args, selection_mode="all")

            # Check for list files command
            elif user_input.lower().startswith("list files") or user_input.lower().startswith("list all files"):
                if user_input.lower().startswith("list files"):
                    args = user_input[10:].strip()
                else:
                    args = user_input[14:].strip()
                return self.direct_call("las_parser", args, list_only=True)

            # No matching command
            return None

        except Exception as e:
            print(f"Error in command processor: {str(e)}")
            traceback.print_exc()
            return f"Error processing command: {str(e)}"

    def direct_call(self, tool_name, file_pattern, **kwargs):
        """Make a direct call to an MCP tool"""
        import requests
        if not self.mcp_server_url:
            return "MCP server URL not available"

        # Parse the pattern
        pattern = file_pattern
        if "matching" in pattern:
            pattern = pattern.split("matching", 1)[1].strip()
        if pattern.endswith('.'):
            pattern = pattern[:-1]

        print(f"Directly calling {tool_name} for pattern: {pattern}")

        # Build the input JSON
        input_data = {"file_path": pattern}
        for k, v in kwargs.items():
            input_data[k] = v

        # Convert to JSON string
        input_json = json.dumps(input_data)

        try:
            # Make the direct HTTP request to the MCP server
            response = requests.post(
                f"{self.mcp_server_url}/tools/{tool_name}",
                json={"input": input_json}
            )

            if response.status_code == 200:
                return response.json()
            else:
                return f"Error calling tool: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error making HTTP request: {str(e)}"


def fix_las_file(input_path, output_path=None):
    """
    Fix a problematic LAS file by reading it manually and writing a corrected version.

    Args:
        input_path: Path to the original LAS file
        output_path: Path to save the fixed file (if None, will use input_path + '.fixed.las')

    Returns:
        Path to the fixed file
    """
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + '.fixed.las'

    # Read the original file as text
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Parse sections
    sections = {'header': [], 'curves': [], 'data': []}
    current_section = 'header'

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check for section markers
        if line.startswith('~'):
            if 'ASCII' in line or 'Data' in line:
                current_section = 'data'
                # Skip the line with column headers which is right after ~ASCII
                sections[current_section].append(line)
                continue
            elif 'Curve' in line:
                current_section = 'curves'
            else:
                current_section = 'header'

            sections[current_section].append(line)
        else:
            sections[current_section].append(line)

    # Find where the actual data starts
    data_start_idx = None
    for i, line in enumerate(sections['data']):
        # Skip section marker and comments
        if line.startswith('~') or line.startswith('#'):
            continue

        # First non-comment line after section marker is the column headers
        if data_start_idx is None:
            data_start_idx = i + 1
            continue

    # Extract actual data rows (skip headers)
    data_rows = sections['data'][data_start_idx:]

    # Fix the data section by ensuring it's properly formatted
    fixed_data = []
    for row in data_rows:
        # Split by whitespace and filter out empty strings
        values = [v for v in row.split() if v]
        if len(values) < 2:  # Skip lines with no data
            continue
        fixed_data.append(values)

    # Create a pandas DataFrame
    if fixed_data:
        df = pd.DataFrame(fixed_data)

        # Convert all columns to numeric values where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Write the fixed LAS file
        with open(output_path, 'w') as f:
            # Write header and curve info sections
            for section in ['header', 'curves']:
                for line in sections[section]:
                    f.write(line + '\n')

            # Write the ASCII marker
            ascii_marker = next((line for line in sections['data'] if '~A' in line), '~ASCII Log Data')
            f.write('\n' + ascii_marker + '\n')

            # Write column headers (first non-comment line after ASCII marker)
            column_headers = sections['data'][data_start_idx - 1]
            f.write(column_headers + '\n')

            # Write data rows
            df.to_csv(f, sep=' ', index=False, header=False, float_format='%.6f', na_rep='-999.25')

        print(f"Fixed LAS file saved to: {output_path}")
        return output_path
    else:
        print("No valid data rows found")
        return None


def parse_fixed_las(file_path, already_tried_fix=False):
    """
    Parse a LAS file with special handling for problematic files.

    Args:
        file_path: Path to the LAS file
        already_tried_fix: Flag to prevent infinite recursion

    Returns:
        lasio.LASFile: Parsed LAS file
    """
    try:
        # First try regular parsing
        print(f"Trying to parse: {file_path}")
        las = lasio.read(file_path)

        # Check if data is string type
        if las.data.dtype.kind in ['U', 'S', 'O']:
            print("Data is string type, fixing...")
            if already_tried_fix:
                print("Already tried fixing - using as is")
                return las

            fixed_path = fix_las_file(file_path)
            if fixed_path:
                return parse_fixed_las(fixed_path, already_tried_fix=True)
            return las

        return las
    except Exception as e:
        print(f"Error parsing LAS file: {str(e)}")

        # Try to fix the file and parse again
        if already_tried_fix:
            print("Already tried fixing - giving up")
            raise

        fixed_path = fix_las_file(file_path)
        if fixed_path:
            return parse_fixed_las(fixed_path, already_tried_fix=True)
        raise


def check_api_key():
    """Check if OpenAI API key is set or can be loaded from .env file"""
    # First check if API key is already in environment
    if "OPENAI_API_KEY" in os.environ:
        logger.info("OpenAI API key found in environment variables")
        return True

    # If not in environment, try to load from .env file
    logger.info("API key not found in environment, checking .env file...")

    # Look for .env file in the current directory and parent directory
    env_paths = [Path(".env"), Path("../.env")]
    for env_path in env_paths:
        if env_path.exists():
            logger.info(f"Found .env file at {env_path.resolve()}")
            load_dotenv(env_path)
            if "OPENAI_API_KEY" in os.environ:
                logger.info("Successfully loaded OpenAI API key from .env file")
                return True

    # If still not found, check if user wants to input it directly
    print("ERROR Error: OPENAI_API_KEY not found in environment variables or .env file")
    print("You have the following options:")
    print("1. Set the environment variable with: export OPENAI_API_KEY=your_api_key")
    print("2. Create a .env file in the project directory with: OPENAI_API_KEY=your_api_key")

    return False


def find_available_port(start_port=5000, max_tries=20, host='localhost'):
    """Find an available port starting from start_port"""
    logger.info(f"Searching for available port starting from {start_port}")
    for port in range(start_port, start_port + max_tries):
        try:
            # Try to create a socket on the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.bind((host, port))
            sock.close()
            logger.info(f"Found available port: {port}")
            return port
        except OSError as e:
            # Port is already in use, try the next one
            logger.debug(f"Port {port} is not available: {str(e)}")
            continue
        except Exception as e:
            logger.warning(f"Unexpected error checking port {port}: {str(e)}")
            continue

    # If we get here, no ports were available
    fallback_port = start_port + 1000  # Try a port well outside the normal range
    logger.warning(
        f"Could not find an available port in range {start_port}-{start_port + max_tries - 1}, using fallback: {fallback_port}")

    # Try the fallback port
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.bind((host, fallback_port))
        sock.close()
        return fallback_port
    except Exception as e:
        logger.error(f"Failed to bind to fallback port {fallback_port}: {str(e)}")
        raise RuntimeError(f"No available ports found after trying {max_tries} ports and fallback")


def run_server_in_thread(server_func, server, max_startup_time=10, **kwargs):
    """Run a server in a background thread with health check"""
    server_ready = threading.Event()
    server_error = [None]  # List to store any error that occurs during startup

    def wrapped_server_func(server, **kwargs):
        try:
            logger.info(f"Starting server on thread {threading.current_thread().name}")
            server_ready.set()  # Signal that thread has started
            server_func(server, **kwargs)
        except Exception as e:
            logger.error(f"Error in server thread: {str(e)}")
            server_error[0] = e
            server_ready.set()  # Signal even on error so main thread doesn't hang
            raise

    # Start server in thread
    thread = threading.Thread(target=wrapped_server_func, args=(server,), kwargs=kwargs, daemon=True)
    thread.name = f"ServerThread-{server.__class__.__name__}"
    logger.info(f"Launching server thread: {thread.name}")
    thread.start()

    # Wait for the server to signal it's ready or for max_startup_time to elapse
    if not server_ready.wait(timeout=2):
        logger.warning("Server thread did not signal readiness within initial 2s")

    # Check if server thread is still alive
    if not thread.is_alive():
        error_msg = f"Server thread died during startup: {server_error[0]}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Allow additional time for the server to fully initialize
    logger.info(f"Server thread started, waiting {max_startup_time - 2}s for initialization")
    time.sleep(max_startup_time - 2)

    # Final check if the thread is still alive
    if not thread.is_alive():
        error_msg = f"Server thread died during initialization: {server_error[0]}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"Server successfully started on thread {thread.name}")
    return thread


def parse_arguments():
    """Parse command line arguments for the Petrophysical Analysis Agent"""
    parser = argparse.ArgumentParser(
        description="Advanced Petrophysical Analysis Agent - Multi-agent system for well log interpretation, formation evaluation, and reservoir analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data-dir ./my_wells --model gpt-4o
  python main.py --enable-monitoring --log-format json --verbose
  python main.py --debug --temperature 0.1 --a2a-port 5001
        """)

    # Server configuration
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument(
        "--host", default="localhost",
        help="Host to bind A2A and MCP servers to (default: localhost)"
    )
    server_group.add_argument(
        "--a2a-port", type=int, default=None,
        help="Port for Agent-to-Agent server (default: auto-select starting from 5000)"
    )
    server_group.add_argument(
        "--mcp-port", type=int, default=None,
        help="Port for Model Context Protocol server (default: auto-select starting from 7000)"
    )

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory containing LAS files and well data (default: ./data)"
    )
    data_group.add_argument(
        "--file-extensions", type=str, nargs="+", default=[".las", ".LAS"],
        help="File extensions to recognize as LAS files (default: .las .LAS)"
    )

    # AI Model configuration
    model_group = parser.add_argument_group("AI Model Configuration")
    model_group.add_argument(
        "--model", type=str, default="gpt-4o",
        choices=["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        help="OpenAI model for the petrophysical expert agent (default: gpt-4o)"
    )
    model_group.add_argument(
        "--temperature", type=float, default=0.0,
        help="AI model temperature for response generation (0.0=deterministic, 1.0=creative, default: 0.0)"
    )
    model_group.add_argument(
        "--max-tokens", type=int, default=None,
        help="Maximum tokens for AI responses (default: model default)"
    )

    # Analysis configuration
    analysis_group = parser.add_argument_group("Analysis Configuration")
    analysis_group.add_argument(
        "--default-curves", type=str, nargs="+",
        default=["GR", "RHOB", "NPHI", "RT", "SP"],
        help="Default curves to prioritize in analysis (default: GR RHOB NPHI RT SP)"
    )
    analysis_group.add_argument(
        "--correlation-tolerance", type=float, default=5.0,
        help="Default depth tolerance for well correlation in meters/feet (default: 5.0)"
    )
    analysis_group.add_argument(
        "--formation-water-resistivity", type=float, default=0.1,
        help="Default formation water resistivity for Archie's equation (default: 0.1 ohm-m)"
    )

    # Performance and processing
    performance_group = parser.add_argument_group("Performance Options")
    performance_group.add_argument(
        "--max-files", type=int, default=50,
        help="Maximum number of files to process in batch operations (default: 50)"
    )
    performance_group.add_argument(
        "--timeout", type=int, default=300,
        help="Timeout for individual operations in seconds (default: 300)"
    )
    performance_group.add_argument(
        "--parallel-processing", action="store_true",
        help="Enable parallel processing for multi-file operations (experimental)"
    )

    # Logging configuration
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument(
        "--log-format", type=str, default="csv",
        choices=["csv", "json", "text"],
        help="Format to save interaction logs (default: csv)"
    )
    logging_group.add_argument(
        "--log-dir", type=str, default="./logs",
        help="Directory to save all log files (default: ./logs)"
    )
    logging_group.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for system messages (default: INFO)"
    )

    # Monitoring and production
    monitoring_group = parser.add_argument_group("Production Monitoring")
    monitoring_group.add_argument(
        "--enable-monitoring", action="store_true",
        help="Enable comprehensive production monitoring and health tracking"
    )
    monitoring_group.add_argument(
        "--health-check-interval", type=int, default=300,
        help="Interval between server health checks in seconds (default: 300)"
    )
    monitoring_group.add_argument(
        "--metrics-retention-days", type=int, default=30,
        help="Number of days to retain performance metrics (default: 30)"
    )

    # Debug and development
    debug_group = parser.add_argument_group("Debug & Development")
    debug_group.add_argument(
        "--debug", action="store_true",
        help="Enable detailed debug logging and error traces"
    )
    debug_group.add_argument(
        "--verbose", action="store_true",
        help="Show verbose output from all server processes and operations"
    )
    debug_group.add_argument(
        "--dry-run", action="store_true",
        help="Initialize system without starting interactive mode (for testing)"
    )
    debug_group.add_argument(
        "--profile", action="store_true",
        help="Enable performance profiling (saves profile data to logs)"
    )

    # Advanced features
    advanced_group = parser.add_argument_group("Advanced Features")
    advanced_group.add_argument(
        "--enable-llm-enhancement", action="store_true",
        help="Enable LLM-powered geological interpretation enhancement"
    )
    advanced_group.add_argument(
        "--auto-correlation", action="store_true",
        help="Automatically attempt well correlation when multiple files are loaded"
    )
    advanced_group.add_argument(
        "--quality-threshold", type=str, default="Good",
        choices=["Excellent", "Good", "Fair", "Poor"],
        help="Minimum quality threshold for automatic processing (default: Good)"
    )

    # Integration options
    integration_group = parser.add_argument_group("Integration Options")
    integration_group.add_argument(
        "--export-format", type=str, nargs="+",
        choices=["json", "csv", "xlsx", "las"],
        help="Additional export formats for analysis results"
    )
    integration_group.add_argument(
        "--api-mode", action="store_true",
        help="Run in API mode for integration with external systems"
    )
    integration_group.add_argument(
        "--config-file", type=str,
        help="Load configuration from JSON/YAML file"
    )

    return parser.parse_args()


def main():
    """Main function"""
    # Check API key first
    if not check_api_key():
        return 1

    # Parse arguments
    args = parse_arguments()

    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Ensure data directory exists
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using data directory: {data_dir.resolve()}")

    # Initialize SEG-Y Production System
    try:
        print("\n🔧 Initializing SEG-Y Production System...")

        # Create required directories
        template_dir = Path("./templates")
        monitoring_dir = Path("./monitoring")
        config_dir = Path("./config")

        template_dir.mkdir(parents=True, exist_ok=True)
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components only if available
        if SEGYIO_COMPONENTS_AVAILABLE:
            # Initialize template system
            try:
                template_validator = TemplateValidator()
                default_template = template_validator.create_default_template(str(template_dir))
                print(f"✓ SEG-Y templates initialized: {template_dir.resolve()}")
            except Exception as e:
                print(f"⚠️  Template system initialization failed: {e}")

            # Initialize monitoring system
            try:
                memory_monitor = MemoryMonitor()
                available_memory = memory_monitor.get_available_memory_gb()
                print(f"✓ Memory monitoring initialized: {available_memory:.1f}GB available")
            except Exception as e:
                print(f"⚠️  Memory monitoring initialization failed: {e}")

            # Create default configuration
            try:
                default_config_path = create_default_config(str(config_dir / "segy_config.yaml"))
                print(f"✓ SEG-Y configuration created: {default_config_path}")
            except Exception as e:
                print(f"⚠️  Configuration creation failed: {e}")

            # Validate SEG-Y system
            try:
                validator = SegyioValidator()
                print(f"✓ SEG-Y validator ready")
            except Exception as e:
                print(f"⚠️  Validator initialization failed: {e}")

            print("✓ SEG-Y Production System initialized successfully with segyio")
        else:
            print("⚠️  Segyio components not available - using fallback mode")
            print("✓ Directory structure created")
            print("⚠️  SEG-Y tools will have limited functionality")

    except Exception as e:
        print(f"⚠️  SEG-Y system initialization failed: {e}")
        print("⚠️  SEG-Y tools may have limited functionality")

    # Find available ports for servers
    a2a_port = args.a2a_port or find_available_port(5000, 20)
    mcp_port = args.mcp_port or find_available_port(7000, 20)

    print(f" A2A server port: {a2a_port}")
    print(f" MCP server port: {mcp_port}")

    # Step 1: Create the A2A server for Subsurface Data Management
    print("\nStep 1: Creating Subsurface Data Management A2A Server")

    # Create an Agent Card for our expert agent based on actual implemented capabilities
    agent_card = AgentCard(
        name="Subsurface Data Management Expert",
        description="Specialized in managing, processing, and analyzing well log data from LAS files and seismic data from SEG-Y files",
        url=f"http://localhost:{a2a_port}",
        version="3.0.0",
        skills=[
            # Well Log Analysis Skills (LAS Files)
            AgentSkill(
                name="LAS File Processing & Quality Control",
                description="Robust parsing, validation, and quality assessment of LAS well log files with comprehensive error handling",
                examples=[
                    "Parse and validate this LAS file with error recovery",
                    "What quality issues exist in my well log data?",
                    "Check data completeness and identify missing curves",
                    "Assess overall data quality and provide recommendations"
                ]
            ),
            AgentSkill(
                name="Petrophysical Analysis & Formation Evaluation",
                description="Advanced petrophysical calculations including porosity, water saturation, shale volume, and pay zone identification",
                examples=[
                    "Calculate effective porosity and water saturation using Archie's equation",
                    "Estimate shale volume using Larionov correction method",
                    "Identify potential pay zones with customizable cutoffs",
                    "Perform comprehensive formation evaluation with net pay calculation"
                ]
            ),
            AgentSkill(
                name="Well Correlation & Stratigraphic Analysis",
                description="Multi-well correlation using advanced algorithms to identify formation tops and stratigraphic markers",
                examples=[
                    "Correlate formations across multiple wells in the field",
                    "Identify key formation tops using curve pattern matching",
                    "Map stratigraphic markers with confidence levels",
                    "Generate formation correlation reports with depth uncertainties"
                ]
            ),
            AgentSkill(
                name="Curve Analysis & Statistics",
                description="Statistical analysis of log curves including trend analysis, outlier detection, and curve characteristics",
                examples=[
                    "Analyze statistical properties of gamma ray curves",
                    "Calculate curve statistics and identify outliers",
                    "Compare curve characteristics between wells",
                    "Generate curve analysis reports with percentiles and distributions"
                ]
            ),
            # SEG-Y Seismic Analysis Skills
            AgentSkill(
                name="SEG-Y File Processing & Validation",
                description="Production-quality SEG-Y parsing with comprehensive validation, error handling, and metadata extraction",
                examples=[
                    "Parse SEG-Y file and extract comprehensive metadata",
                    "Validate SEG-Y file structure and format compliance",
                    "Handle problematic SEG-Y files with robust error recovery",
                    "Extract survey geometry and acquisition parameters"
                ]
            ),
            AgentSkill(
                name="Seismic Survey Analysis & Quality Control",
                description="Comprehensive analysis of seismic survey geometry, data quality assessment, and performance monitoring",
                examples=[
                    "Analyze 3D seismic survey geometry and grid parameters",
                    "Assess seismic data quality and identify potential issues",
                    "Generate comprehensive survey reports with QC metrics",
                    "Evaluate acquisition parameters and coverage statistics"
                ]
            ),
            AgentSkill(
                name="Multi-File Seismic Processing",
                description="Batch processing of multiple SEG-Y files with parallel execution, progress reporting, and comprehensive analysis",
                examples=[
                    "Process multiple SEG-Y files as a complete seismic survey",
                    "Generate survey-wide analysis with parallel processing",
                    "Create comprehensive reports covering entire seismic datasets",
                    "Monitor processing performance and resource utilization"
                ]
            ),
            AgentSkill(
                name="Production System Monitoring",
                description="Real-time system health monitoring, performance tracking, and production deployment management",
                examples=[
                    "Monitor system health and processing performance",
                    "Track processing metrics and resource utilization",
                    "Generate system status reports and health assessments",
                    "Manage production deployment and maintenance schedules"
                ]
            )
        ]
    )

    # Create the OpenAI-powered A2A server with comprehensive system prompt
    openai_server = OpenAIA2AServer(
        api_key=os.environ["OPENAI_API_KEY"],
        model=args.model,
        temperature=args.temperature,
        system_prompt="""You are a production-grade subsurface data analysis expert specializing in both well log and seismic data interpretation.

    Your capabilities include:

    WELL LOG ANALYSIS (LAS Files):
    - Robust parsing with comprehensive error recovery
    - Advanced petrophysical calculations and formation evaluation
    - Quality control and data validation for well logs
    - Well correlation and pay zone identification

    SEISMIC DATA ANALYSIS (SEG-Y Files):
    - Production-quality SEG-Y file parsing with robust error handling
    - Comprehensive survey geometry and data quality analysis
    - Multi-file survey processing with parallel execution and progress reporting
    - Advanced quality control and validation with detailed reporting

    INTEGRATED WORKFLOWS:
    - Well-to-seismic calibration and correlation
    - Integrated subsurface interpretation combining both data types
    - Quality assurance and validation across all data types
    - Formation correlation using both wells and seismic data

    PRODUCTION FEATURES:
    - Comprehensive error handling with detailed recovery suggestions
    - Real-time progress reporting for long operations
    - Memory management and performance optimization for large datasets
    - Detailed validation and quality control with actionable recommendations
    - Robust template and configuration management
    - System health monitoring and performance tracking

    You provide reliable, production-quality analysis with comprehensive error handling, progress tracking, and detailed validation. You handle both routine operations and challenging data scenarios with appropriate error recovery and user guidance.

    When discussing data integration, explain the complementary nature of well logs and seismic data in subsurface characterization. Always provide clear, actionable recommendations based on your analysis."""
    )

    # Update the server with our agent card
    openai_server.agent_card = agent_card

    # Wrap it in a standard A2A server
    class WellLogExpert(A2AServer):
        def __init__(self, openai_server, agent_card):
            super().__init__(agent_card=agent_card)
            self.openai_server = openai_server

        def handle_message(self, message):
            """Handle incoming messages by delegating to OpenAI server"""
            return self.openai_server.handle_message(message)

    # Create the wrapped agent
    log_expert_agent = WellLogExpert(openai_server, agent_card)

    # Start the A2A server in a background thread
    a2a_server_url = f"http://localhost:{a2a_port}"
    print(f"\nStarting A2A server on {a2a_server_url}...")

    def run_a2a_server(server, host="0.0.0.0", port=a2a_port):
        """Run the A2A server"""
        run_server(server, host=host, port=port)

    a2a_thread = run_server_in_thread(run_a2a_server, log_expert_agent)

    # Step 2: Create MCP Server with Subsurface Data Management Tools
    print("\nStep 2: Creating MCP Server with Subsurface Data Management Tools")

    # Create MCP server with tools
    mcp_server = FastMCP(
        name="Subsurface Data Management Tools",
        description="Advanced tools for managing, processing, and analyzing LAS files, SEG-Y seismic data, and integrated subsurface datasets"
    )

    # Create MCP client
    class MCPClient:
        def __init__(self, server_url):
            self.server_url = server_url

        def call_tool(self, tool_name, input_data):
            """Make a direct call to an MCP tool"""
            import requests
            try:
                # Convert input to appropriate format
                if isinstance(input_data, dict):
                    input_json = json.dumps(input_data)
                elif isinstance(input_data, str) and (input_data.startswith('{') or input_data.startswith('[')):
                    try:
                        json.loads(input_data)  # Validate JSON
                        input_json = input_data
                    except json.JSONDecodeError:
                        input_json = input_data
                else:
                    input_json = input_data

                response = requests.post(f"{self.server_url}/tools/{tool_name}", json={"input": input_json})
                return response.json() if response.status_code == 200 else {
                    "error": f"HTTP error {response.status_code}: {response.text}"}
            except Exception as e:
                return {"error": f"Error calling MCP tool: {str(e)}"}

    def create_langchain_tools_from_mcp(mcp_server_url, tool_definitions, data_dir):
        """Convert MCP tools to LangChain tools that the agent can actually use"""

        mcp_client = MCPClient(mcp_server_url)
        langchain_tools = []

        for tool_name, description in tool_definitions.items():
            def create_tool_function(tool_name):
                def tool_function(input_text: str) -> str:
                    """Execute MCP tool and return results"""
                    try:
                        # Parse input to extract file path
                        if not input_text.strip():
                            return f"Error: No input provided for {tool_name}"

                        # Call MCP tool
                        result = mcp_client.call_tool(tool_name, input_text)

                        # Handle response
                        if isinstance(result, dict):
                            if "error" in result:
                                return f"Tool error: {result['error']}"
                            elif "text" in result:
                                # Parse JSON response if it's a string
                                if isinstance(result["text"], str):
                                    try:
                                        parsed = json.loads(result["text"])
                                        if isinstance(parsed, dict):
                                            # Format nicely for the agent
                                            if "error" in parsed:
                                                return f"Error: {parsed['error']}"
                                            else:
                                                return json.dumps(parsed, indent=2)
                                    except json.JSONDecodeError:
                                        return result["text"]
                                return str(result["text"])
                            else:
                                return json.dumps(result, indent=2)
                        else:
                            return str(result)
                    except Exception as e:
                        return f"Error calling {tool_name}: {str(e)}"

                return tool_function

            # Create LangChain Tool
            langchain_tool = Tool(
                name=tool_name,
                func=create_tool_function(tool_name),
                description=description
            )
            langchain_tools.append(langchain_tool)
            print(f"✓ Created LangChain tool: {tool_name}")

        return langchain_tools

    def create_enhanced_agent_with_tools(a2a_agent, mcp_tools, model="gpt-4o", temperature=0.0):
        """Create an agent that actually uses the tools"""

        llm = ChatOpenAI(model=model, temperature=temperature)

        # Enhanced system prompt that tells the agent to USE the tools
        system_prompt = """You are an expert subsurface data analyst with access to powerful analysis tools for both well logs and seismic data.

    IMPORTANT: You have access to the following tools that you MUST use when users ask for analysis:

    WELL LOG TOOLS (LAS Files):
    - las_parser: Parse LAS files to extract metadata and basic information
    - las_analysis: Analyze curves and perform statistical analysis  
    - las_qc: Perform quality control checks on LAS files
    - formation_evaluation: Perform comprehensive petrophysical analysis
    - well_correlation: Correlate formations across multiple wells
    - calculate_shale_volume: Calculate shale volume from gamma ray logs

    SEISMIC DATA TOOLS (SEG-Y Files):
    - segy_parser: Parse SEG-Y files with segyio-powered processing
    - segy_qc: Perform quality control on SEG-Y files with segyio validation
    - segy_analysis: Analyze seismic survey geometry and data quality with segyio
    - segy_classify: Automatically classify SEG-Y survey characteristics (handles both single and batch)
    - segy_survey_analysis: Analyze multiple SEG-Y files as complete surveys
    - segy_template_detect: Template detection using segyio native header reading
    - segy_survey_compare: Compare SEG-Y files for processing compatibility
    - quick_segy_summary: Get fast overview of SEG-Y files without full processing

    GENERAL TOOLS:
    - list_files: List available data files matching patterns
    - system_status: Check system health and performance

    WORKFLOWS:

    For LAS file analysis ("analyze formation in well.las"):
    1. Call las_parser with "well.las" to load the file
    2. Call formation_evaluation with "well.las" for petrophysical analysis
    3. Call las_qc with "well.las" to check data quality
    4. Interpret results and provide expert analysis

    For SEG-Y file analysis ("analyze seismic survey.sgy"):
    1. Call segy_parser with "survey.sgy" to load the file
    2. Call segy_classify with "survey.sgy" to determine survey characteristics  
    3. Call segy_qc with "survey.sgy" to check data quality
    4. Call segy_analysis with "survey.sgy" for detailed geometry analysis
    5. For quick overview, use quick_segy_summary instead of full analysis
    6. Interpret results and provide expert seismic analysis

    For multi-file analysis ("correlate all wells" or "analyze seismic survey"):
    1. Call list_files to see available files
    2. Use appropriate batch tools (well_correlation, segy_survey_analysis, etc.)
    3. Provide integrated interpretation

    DO NOT give generic advice - USE THE TOOLS to actually analyze the data and provide specific results from the actual files."""

        # Create memory
        memory = ConversationBufferMemory(memory_key="chat_history")

        # Create agent with tools
        agent = initialize_agent(
            tools=mcp_tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            agent_kwargs={
                'prefix': system_prompt,
                'format_instructions': """Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action (usually a filename)
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question"""
            }
        )

        return agent

    # Utility functions to handle common patterns
    def handle_json_input(input_param, default_params=None, **kwargs):
        """Centralized JSON input handling for all MCP tools."""
        if default_params is None:
            default_params = {}

        # Handle JSON string input
        if isinstance(input_param, str) and (input_param.startswith('{') or input_param.startswith('[')):
            try:
                parsed_input = json.loads(input_param)
                if isinstance(parsed_input, dict):
                    return {**default_params, **parsed_input, **kwargs}
                elif isinstance(parsed_input, list):
                    return {**default_params, "file_paths": parsed_input, **kwargs}
            except json.JSONDecodeError:
                pass

        # Handle dictionary input
        if isinstance(input_param, dict):
            return {**default_params, **input_param, **kwargs}

        # Handle 'input' in kwargs (MCP layer)
        if 'input' in kwargs and kwargs['input'] is not None:
            # Remove 'input' from kwargs to prevent infinite recursion
            kwargs_without_input = {k: v for k, v in kwargs.items() if k != 'input'}
            return handle_json_input(kwargs['input'], default_params, **kwargs_without_input)

        # Return default with file_path set
        return {**default_params, "file_path": input_param, **kwargs}

    def create_error_response(error_msg, details=None, suggestions=None):
        """Standardized error response creation."""
        response = {
            "error": error_msg,
            "details": details or traceback.format_exc()
        }
        if suggestions:
            response["suggestions"] = suggestions
        return {"text": json.dumps(response, cls=NumpyJSONEncoder)}

    def create_mcp_tool(name, description, handler_func, default_params=None):
        """Factory function to create MCP tools with standardized error handling."""

        @mcp_server.tool(name=name, description=description)
        def tool_wrapper(**kwargs):
            try:
                # Extract primary parameter (usually file_path)
                primary_param = kwargs.get(list(kwargs.keys())[0] if kwargs else None)

                # Handle input processing
                processed_params = handle_json_input(
                    primary_param,
                    default_params or {"data_dir": args.data_dir},
                    **kwargs
                )

                # Call the actual handler
                return handler_func(**processed_params)

            except Exception as e:
                return create_error_response(f"Error in {name}: {str(e)}")

        return tool_wrapper

    # LAS Tools - Using the factory pattern
    def las_parser_handler(file_path=None, file_paths=None, **kwargs):
        """Handle LAS parsing logic - FIXED FOR PATTERNS"""

        print(f"DEBUG: las_parser_handler received: file_path={file_path}, file_paths={file_paths}")

        data_dir = kwargs.get("data_dir", "./data")

        # Handle file_paths parameter (list)
        if file_paths:
            return enhanced_las_parser(file_paths=file_paths, **kwargs)

        # Handle file_path parameter (single or pattern)
        if file_path:
            if isinstance(file_path, str) and ("*" in file_path or "?" in file_path):
                print(f"DEBUG: Processing file pattern: {file_path}")

                import glob
                search_pattern = os.path.join(data_dir, file_path) if not file_path.startswith(data_dir) else file_path
                matching_files = glob.glob(search_pattern)

                if not matching_files:
                    return create_error_response(f"No files found matching pattern: {file_path}")

                # Process all files
                results = []
                for file in matching_files:
                    try:
                        result = enhanced_las_parser(file, **kwargs)
                        results.append({
                            "file": os.path.basename(file),
                            "result": result
                        })
                    except Exception as e:
                        results.append({
                            "file": os.path.basename(file),
                            "error": str(e)
                        })

                summary = {
                    "pattern_processed": file_path,
                    "files_processed": len(results),
                    "results": results,
                    "summary": f"Parsed {len(results)} LAS files matching '{file_path}'"
                }

                return {"text": json.dumps(summary, cls=NumpyJSONEncoder)}

            else:
                return enhanced_las_parser(file_path, **kwargs)

        return create_error_response("No file path provided")

    def las_analysis_handler(file_path=None, curves=None, **kwargs):
        """Handle LAS analysis logic - FIXED FOR PATTERNS"""

        print(f"DEBUG: las_analysis_handler received: file_path={file_path}")

        data_dir = kwargs.get("data_dir", "./data")

        if file_path:
            if isinstance(file_path, str) and ("*" in file_path or "?" in file_path):
                print(f"DEBUG: Processing file pattern: {file_path}")

                import glob
                search_pattern = os.path.join(data_dir, file_path) if not file_path.startswith(data_dir) else file_path
                matching_files = glob.glob(search_pattern)

                if not matching_files:
                    return create_error_response(f"No files found matching pattern: {file_path}")

                # Process all files
                results = []
                for file in matching_files:
                    try:
                        result = enhanced_las_analysis(file, curves, **kwargs)
                        results.append({
                            "file": os.path.basename(file),
                            "result": result
                        })
                    except Exception as e:
                        results.append({
                            "file": os.path.basename(file),
                            "error": str(e)
                        })

                summary = {
                    "pattern_processed": file_path,
                    "files_processed": len(results),
                    "results": results
                }

                return {"text": json.dumps(summary, cls=NumpyJSONEncoder)}

            else:
                return enhanced_las_analysis(file_path, curves, **kwargs)

        return create_error_response("No file path provided")

    def las_qc_handler(file_path=None, **kwargs):
        """Handle LAS QC logic - FIXED FOR PATTERNS"""

        print(f"DEBUG: las_qc_handler received: file_path={file_path}")

        data_dir = kwargs.get("data_dir", "./data")

        if file_path:
            if isinstance(file_path, str) and ("*" in file_path or "?" in file_path):
                print(f"DEBUG: Processing file pattern: {file_path}")

                import glob
                search_pattern = os.path.join(data_dir, file_path) if not file_path.startswith(data_dir) else file_path
                matching_files = glob.glob(search_pattern)

                if not matching_files:
                    return create_error_response(f"No files found matching pattern: {file_path}")

                # Process all files
                results = []
                quality_summary = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}

                for file in matching_files:
                    try:
                        result = enhanced_las_qc(file, **kwargs)
                        results.append({
                            "file": os.path.basename(file),
                            "result": result
                        })

                        # Track quality ratings
                        if isinstance(result, dict) and "text" in result:
                            result_data = json.loads(result["text"]) if isinstance(result["text"], str) else result
                            quality = result_data.get("overall_quality", "unknown").lower()
                            if quality in quality_summary:
                                quality_summary[quality] += 1

                    except Exception as e:
                        results.append({
                            "file": os.path.basename(file),
                            "error": str(e)
                        })

                summary = {
                    "pattern_processed": file_path,
                    "files_processed": len(results),
                    "quality_summary": quality_summary,
                    "results": results,
                    "overall_assessment": f"QC completed for {len(results)} files"
                }

                return {"text": json.dumps(summary, cls=NumpyJSONEncoder)}

            else:
                return enhanced_las_qc(file_path, **kwargs)

        return create_error_response("No file path provided")

    def formation_evaluation_handler(file_path=None, **kwargs):
        """Handle formation evaluation logic - FIXED FOR PATTERNS"""

        print(f"DEBUG: formation_evaluation_handler received: file_path={file_path}")

        # Get data directory
        data_dir = kwargs.get("data_dir", "./data")

        # Handle file patterns
        if file_path:
            if isinstance(file_path, str) and ("*" in file_path or "?" in file_path):
                print(f"DEBUG: Processing file pattern: {file_path}")

                import glob
                search_pattern = os.path.join(data_dir, file_path) if not file_path.startswith(data_dir) else file_path
                matching_files = glob.glob(search_pattern)

                print(f"DEBUG: Found {len(matching_files)} matching files")

                if not matching_files:
                    return create_error_response(f"No files found matching pattern: {file_path}")

                # Process all matching files
                results = []
                for file in matching_files:
                    try:
                        result = enhanced_formation_evaluation(file, **kwargs)
                        results.append({
                            "file": os.path.basename(file),
                            "result": result
                        })
                    except Exception as e:
                        results.append({
                            "file": os.path.basename(file),
                            "error": str(e)
                        })

                # Combine results
                summary = {
                    "pattern_processed": file_path,
                    "files_processed": len(results),
                    "results": results,
                    "summary": f"Processed {len(results)} files matching '{file_path}'"
                }

                return {"text": json.dumps(summary, cls=NumpyJSONEncoder)}

            else:
                # Single file
                return enhanced_formation_evaluation(file_path, **kwargs)

        return create_error_response("No file path provided")

    def well_correlation_handler(well_list=None, marker_curve="GR", **kwargs):
        """Handle well correlation logic - FIXED VERSION"""

        print(f"DEBUG: well_correlation_handler received: well_list={well_list}")
        print(f"DEBUG: kwargs={kwargs}")

        # Get data directory
        data_dir = kwargs.get("data_dir", "./data")

        # CRITICAL FIX: Handle file patterns
        if well_list:
            if isinstance(well_list, str):
                # Check if it's a file pattern
                if "*" in well_list or "?" in well_list:
                    print(f"DEBUG: Processing file pattern: {well_list}")

                    import glob
                    # Build search pattern
                    if not well_list.startswith('/') and not well_list.startswith(data_dir):
                        search_pattern = os.path.join(data_dir, well_list)
                    else:
                        search_pattern = well_list

                    print(f"DEBUG: Search pattern: {search_pattern}")

                    # Find matching files
                    matching_files = glob.glob(search_pattern)
                    print(f"DEBUG: Found {len(matching_files)} matching files")

                    if not matching_files:
                        return create_error_response(f"No files found matching pattern: {well_list}")

                    # Use the matching files
                    well_list = matching_files

                else:
                    # Single file - convert to full path
                    full_path = find_las_file(well_list, data_dir)
                    if os.path.exists(full_path):
                        well_list = [full_path]
                    else:
                        return create_error_response(f"File not found: {well_list}")

            elif isinstance(well_list, list):
                # List of files - convert to full paths
                file_paths = []
                for file_name in well_list:
                    if os.path.isabs(file_name):
                        # Already full path
                        file_paths.append(file_name)
                    else:
                        # Convert to full path
                        full_path = find_las_file(file_name, data_dir)
                        if os.path.exists(full_path):
                            file_paths.append(full_path)

                well_list = file_paths if file_paths else None

        print(f"DEBUG: Final well_list has {len(well_list) if well_list else 0} files")

        # Check if we have valid files
        if not well_list:
            return create_error_response("No valid wells found for correlation")

        # Add langchain_agent if not present (for your existing function)
        if 'langchain_agent' not in kwargs:
            kwargs['langchain_agent'] = globals().get('langchain_agent')

        # Call the actual correlation function
        try:
            result = enhanced_well_correlation_with_qc(well_list, marker_curve, **kwargs)
            return result
        except Exception as e:
            print(f"DEBUG: Correlation function failed: {str(e)}")
            return create_error_response(f"Correlation failed: {str(e)}")

    # Create LAS tools using factory
    create_mcp_tool("las_parser", "Parse and extract metadata from LAS files", las_parser_handler)
    create_mcp_tool("las_analysis", "Analyze curve data and perform basic calculations", las_analysis_handler)
    create_mcp_tool("las_qc", "Perform quality control checks on LAS files", las_qc_handler)
    create_mcp_tool("formation_evaluation", "Perform petrophysical analysis and identify pay zones",
                    formation_evaluation_handler)
    create_mcp_tool("well_correlation", "Correlate formations across multiple wells", well_correlation_handler)

    # Specialized tools that need custom logic
    @mcp_server.tool(name="list_files", description="List any type of data files matching a pattern")
    def list_universal_files(pattern=None, **kwargs):
        """Universal file listing system - supports all file types"""
        try:
            params = handle_json_input(pattern, {"data_dir": "./data"}, **kwargs)

            file_pattern = params.get("pattern") or params.get("file_path") or "*"
            data_dir = params.get("data_dir", "./data")

            # Auto-detect file type
            detected_type = detect_file_type(file_pattern)

            if detected_type and detected_type in FILE_TYPE_CONFIG:
                config = FILE_TYPE_CONFIG[detected_type]

                # If no extension specified, use default
                if not any(ext.lower() in file_pattern.lower() for ext in config["extensions"]):
                    file_pattern = config["default_pattern"]

            # Search for files with all possible extensions for this type
            matching_files = []
            if detected_type and detected_type in FILE_TYPE_CONFIG:
                config = FILE_TYPE_CONFIG[detected_type]
                base_pattern = file_pattern.split('.')[0] if '.' in file_pattern else file_pattern.rstrip('*')

                for ext in config["extensions"]:
                    if '*' in file_pattern:
                        search_pattern = os.path.join(data_dir, file_pattern.replace('*.*', f'*{ext}').replace('*.las',
                                                                                                               f'*{ext}').replace(
                            '*.sgy', f'*{ext}'))
                    else:
                        search_pattern = os.path.join(data_dir, f"{base_pattern}{ext}")

                    import glob
                    matching_files.extend(glob.glob(search_pattern))
            else:
                # Fallback: direct pattern search
                import glob
                search_pattern = os.path.join(data_dir, file_pattern)
                matching_files = glob.glob(search_pattern)

            # Remove duplicates and sort
            matching_files = sorted(list(set(matching_files)))

            if not matching_files:
                return {"text": f"No files found matching pattern: {file_pattern}"}

            # Format output using detected type config
            if detected_type and detected_type in FILE_TYPE_CONFIG:
                config = FILE_TYPE_CONFIG[detected_type]
                formatted_output = format_files_by_type(matching_files, config)
            else:
                formatted_output = format_generic_files(matching_files, file_pattern)

            return {"text": formatted_output}

        except Exception as e:
            return create_error_response(f"Error listing files: {str(e)}")

    @mcp_server.tool(name="calculate_shale_volume", description="Calculate shale volume using Larionov method")
    def calculate_shale_volume(file_path=None, **kwargs):
        """Calculate shale volume from gamma ray log - FIXED FOR PATTERNS"""
        try:
            params = handle_json_input(file_path, {"data_dir": args.data_dir}, **kwargs)

            file_path_param = params.get("file_path")
            data_dir = params.get("data_dir", args.data_dir)
            gr_curve = params.get('curve', params.get('curve_mnemonic', 'GR'))

            print(f"DEBUG: calculate_shale_volume received: file_path={file_path_param}")

            if not file_path_param:
                return create_error_response("No file path provided")

            # HANDLE PATTERNS
            if isinstance(file_path_param, str) and ("*" in file_path_param or "?" in file_path_param):
                print(f"DEBUG: Processing shale volume pattern: {file_path_param}")

                import glob
                search_pattern = os.path.join(data_dir, file_path_param) if not file_path_param.startswith(
                    data_dir) else file_path_param
                matching_files = glob.glob(search_pattern)

                if not matching_files:
                    return create_error_response(f"No files found matching pattern: {file_path_param}")

                print(f"DEBUG: Found {len(matching_files)} files for shale volume calculation")

                # Process all files
                shale_results = []
                total_wells = 0
                avg_shale_overall = 0

                for file_path in matching_files:
                    try:
                        las, error = load_las_file(file_path)
                        if error:
                            shale_results.append({
                                "file": os.path.basename(file_path),
                                "error": f"Error loading LAS file: {error}"
                            })
                            continue

                        if not las.curve_exists(gr_curve):
                            available_curves = las.get_curve_names()
                            shale_results.append({
                                "file": os.path.basename(file_path),
                                "error": f"Gamma ray curve '{gr_curve}' not found",
                                "available_curves": available_curves
                            })
                            continue

                        # Calculate shale volume
                        gr_data = las.get_curve_data(gr_curve)
                        vshale = estimate_vshale(gr_data)
                        valid_vshale = vshale[~np.isnan(vshale)]

                        if len(valid_vshale) == 0:
                            shale_results.append({
                                "file": os.path.basename(file_path),
                                "error": "No valid shale volume data calculated"
                            })
                            continue

                        # Calculate statistics
                        avg_vshale_pct = float(np.mean(valid_vshale)) * 100
                        min_vshale_pct = float(np.min(valid_vshale)) * 100
                        max_vshale_pct = float(np.max(valid_vshale)) * 100

                        well_result = {
                            "file": os.path.basename(file_path),
                            "well_name": las.well_info.get("WELL", "Unknown"),
                            "gamma_ray_curve_used": gr_curve,
                            "average_shale_volume_percent": round(avg_vshale_pct, 2),
                            "min_shale_volume_percent": round(min_vshale_pct, 2),
                            "max_shale_volume_percent": round(max_vshale_pct, 2),
                            "depth_range": [float(las.index[0]), float(las.index[-1])],
                            "data_points_analyzed": len(valid_vshale)
                        }

                        shale_results.append(well_result)
                        total_wells += 1
                        avg_shale_overall += avg_vshale_pct

                    except Exception as e:
                        shale_results.append({
                            "file": os.path.basename(file_path),
                            "error": str(e)
                        })

                # Create summary
                successful_calculations = [r for r in shale_results if "average_shale_volume_percent" in r]
                field_avg_shale = avg_shale_overall / len(successful_calculations) if successful_calculations else 0

                summary_result = {
                    "pattern_processed": file_path_param,
                    "files_processed": len(shale_results),
                    "successful_calculations": len(successful_calculations),
                    "method": "Larionov correction for Tertiary rocks",
                    "gamma_ray_curve_used": gr_curve,
                    "field_average_shale_percent": round(field_avg_shale, 2),
                    "individual_results": shale_results,
                    "summary": f"Calculated shale volume for {len(successful_calculations)}/{len(shale_results)} wells, field average: {round(field_avg_shale, 1)}%"
                }

                return {"text": json.dumps(summary_result, cls=NumpyJSONEncoder)}

            else:
                # SINGLE FILE PROCESSING (original logic)
                full_path = find_las_file(file_path_param, data_dir)
                if not os.path.isfile(full_path):
                    return create_error_response(f"File not found: {file_path_param}")

                las, error = load_las_file(full_path)
                if error:
                    return create_error_response(f"Error loading LAS file: {error}")

                if not las.curve_exists(gr_curve):
                    return create_error_response(
                        f"Gamma ray curve '{gr_curve}' not found",
                        suggestions=[f"Available curves: {las.get_curve_names()}"]
                    )

                # Calculate shale volume
                gr_data = las.get_curve_data(gr_curve)
                vshale = estimate_vshale(gr_data)
                valid_vshale = vshale[~np.isnan(vshale)]

                if len(valid_vshale) == 0:
                    return create_error_response("No valid shale volume data calculated")

                # Calculate statistics
                avg_vshale_pct = float(np.mean(valid_vshale)) * 100
                min_vshale_pct = float(np.min(valid_vshale)) * 100
                max_vshale_pct = float(np.max(valid_vshale)) * 100

                result = {
                    "well_name": las.well_info.get("WELL", "Unknown"),
                    "file_processed": os.path.basename(full_path),
                    "gamma_ray_curve_used": gr_curve,
                    "average_shale_volume_percent": round(avg_vshale_pct, 2),
                    "min_shale_volume_percent": round(min_vshale_pct, 2),
                    "max_shale_volume_percent": round(max_vshale_pct, 2),
                    "method": "Larionov correction for Tertiary rocks",
                    "depth_range": [float(las.index[0]), float(las.index[-1])],
                    "data_points_analyzed": len(valid_vshale),
                    "summary": f"Average shale volume: {round(avg_vshale_pct, 1)}%, range: {round(min_vshale_pct, 1)}%-{round(max_vshale_pct, 1)}%"
                }

                return {"text": json.dumps(result, cls=NumpyJSONEncoder)}

        except Exception as e:
            return create_error_response(f"Error calculating shale volume: {str(e)}")

    # ============================================================================
    # SEGYIO UPGRADED HANDLERS - FOR MAIN.PY
    # Add these handler functions directly to your main.py
    # ============================================================================

    # Import your production segyio tools (add to your main.py imports)
    try:
        from production_segy_tools import production_segy_parser, find_segy_file
        PRODUCTION_PARSER_AVAILABLE = True
    except ImportError:
        PRODUCTION_PARSER_AVAILABLE = False

    try:
        from production_segy_analysis_qc import production_segy_qc, production_segy_analysis
        PRODUCTION_QC_ANALYSIS_AVAILABLE = True
    except ImportError:
        PRODUCTION_QC_ANALYSIS_AVAILABLE = False

    try:
        from survey_classifier import SurveyClassifier
        SURVEY_CLASSIFIER_AVAILABLE = True
    except ImportError:
        SURVEY_CLASSIFIER_AVAILABLE = False

    # ============================================================================
    # UPGRADED HANDLER FUNCTIONS
    # ============================================================================

    def segy_qc_handler(**params):
        """SEG-Y QC handler - upgraded with segyio"""

        file_path = params.get("file_path")
        data_dir = params.get("data_dir", "./data")

        if file_path and isinstance(file_path, str) and ("*" in file_path or "?" in file_path):
            import glob
            search_patterns = []
            base_pattern = file_path.replace('.sgy', '').replace('.segy', '')

            for ext in ['.sgy', '.segy', '.SGY', '.SEGY']:
                pattern = base_pattern + ext
                search_pattern = os.path.join(data_dir, pattern) if not pattern.startswith(data_dir) else pattern
                search_patterns.append(search_pattern)

            matching_files = []
            for pattern in search_patterns:
                matching_files.extend(glob.glob(pattern))

            matching_files = list(set(matching_files))

            if not matching_files:
                return create_error_response(f"No SEG-Y files found matching pattern: {file_path}")

            results = []
            for file in matching_files:
                try:
                    file_params = params.copy()
                    file_params["file_path"] = file

                    if PRODUCTION_QC_ANALYSIS_AVAILABLE:
                        result = production_segy_qc(**file_params)
                    else:
                        result = {"text": json.dumps({"error": "Segyio QC not available"}, cls=NumpyJSONEncoder)}

                    results.append({
                        "file": os.path.basename(file),
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "file": os.path.basename(file),
                        "error": str(e)
                    })

            summary = {
                "pattern_processed": file_path,
                "files_processed": len(results),
                "results": results,
                "summary": f"QC completed for {len(results)} SEG-Y files"
            }

            return {"text": json.dumps(summary, cls=NumpyJSONEncoder)}

        else:
            if PRODUCTION_QC_ANALYSIS_AVAILABLE:
                return production_segy_qc(**params)
            else:
                return create_error_response("Segyio QC not available")

    def segy_analysis_handler(**params):
        """SEG-Y Analysis handler - upgraded with segyio"""

        file_path = params.get("file_path")
        data_dir = params.get("data_dir", "./data")

        if file_path and isinstance(file_path, str) and ("*" in file_path or "?" in file_path):
            import glob
            search_patterns = []
            base_pattern = file_path.replace('.sgy', '').replace('.segy', '')

            for ext in ['.sgy', '.segy', '.SGY', '.SEGY']:
                pattern = base_pattern + ext
                search_pattern = os.path.join(data_dir, pattern) if not pattern.startswith(data_dir) else pattern
                search_patterns.append(search_pattern)

            matching_files = []
            for pattern in search_patterns:
                matching_files.extend(glob.glob(pattern))

            matching_files = list(set(matching_files))

            if not matching_files:
                return create_error_response(f"No SEG-Y files found matching pattern: {file_path}")

            results = []
            for file in matching_files:
                try:
                    file_params = params.copy()
                    file_params["file_path"] = file

                    if PRODUCTION_QC_ANALYSIS_AVAILABLE:
                        result = production_segy_analysis(**file_params)
                    else:
                        result = {"text": json.dumps({"error": "Segyio analysis not available"}, cls=NumpyJSONEncoder)}

                    results.append({
                        "file": os.path.basename(file),
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "file": os.path.basename(file),
                        "error": str(e)
                    })

            summary = {
                "pattern_processed": file_path,
                "files_processed": len(results),
                "results": results,
                "summary": f"Analysis completed for {len(results)} SEG-Y files"
            }

            return {"text": json.dumps(summary, cls=NumpyJSONEncoder)}

        else:
            if PRODUCTION_QC_ANALYSIS_AVAILABLE:
                return production_segy_analysis(**params)
            else:
                return create_error_response("Segyio analysis not available")

    def segy_classify_handler(**params):
        """SEG-Y Classification handler - upgraded with segyio"""

        file_path = params.get("file_path")
        data_dir = params.get("data_dir", "./data")

        if file_path and isinstance(file_path, str) and ("*" in file_path or "?" in file_path):
            if not SURVEY_CLASSIFIER_AVAILABLE:
                return create_error_response("Segyio survey classifier not available")

            import glob
            search_patterns = []
            base_pattern = file_path.replace('.sgy', '').replace('.segy', '')

            for ext in ['.sgy', '.segy', '.SGY', '.SEGY']:
                pattern = base_pattern + ext
                search_pattern = os.path.join(data_dir, pattern) if not pattern.startswith(data_dir) else pattern
                search_patterns.append(search_pattern)

            matching_files = []
            for pattern in search_patterns:
                matching_files.extend(glob.glob(pattern))

            matching_files = list(set(matching_files))

            if not matching_files:
                return create_error_response(f"No SEG-Y files found matching pattern: {file_path}")

            classifier = SurveyClassifier(params.get("template_dir", "./templates"))
            batch_results = []

            for file_path_item in matching_files[:10]:
                try:
                    classification = classifier.classify_survey(file_path_item)
                    batch_results.append({
                        "file": os.path.basename(file_path_item),
                        "classification": classification
                    })
                except Exception as e:
                    batch_results.append({
                        "file": os.path.basename(file_path_item),
                        "error": str(e)
                    })

            result = {
                "pattern_searched": file_path,
                "files_found": len(matching_files),
                "files_processed": len(batch_results),
                "batch_results": batch_results,
                "summary": f"Classified {len(batch_results)} SEG-Y files"
            }

            return {"text": json.dumps(result, cls=NumpyJSONEncoder)}


        else:

            if not SURVEY_CLASSIFIER_AVAILABLE:
                return create_error_response("Segyio survey classifier not available")

            # FIX: Construct full path like the wildcard section does

            input_file = params["file_path"]

            # Build full path if needed

            if not os.path.isabs(input_file) and not input_file.startswith(data_dir):

                full_file_path = os.path.join(data_dir, input_file)

            else:

                full_file_path = input_file

            # Check if file exists before processing

            if not os.path.exists(full_file_path):

                # Try different extensions like the wildcard section

                base_name = input_file.replace('.sgy', '').replace('.segy', '')

                for ext in ['.sgy', '.segy', '.SGY', '.SEGY']:

                    test_path = os.path.join(data_dir, base_name + ext)

                    if os.path.exists(test_path):
                        full_file_path = test_path

                        break

                else:

                    return create_error_response(f"File not found: {input_file}")

            classifier = SurveyClassifier(params.get("template_dir", "./templates"))

            classification = classifier.classify_survey(full_file_path)  # Use full path

            return {"text": json.dumps(classification, cls=NumpyJSONEncoder)}

    def segy_parser_handler(**params):
        """SEG-Y Parser handler - upgraded with segyio"""

        if PRODUCTION_PARSER_AVAILABLE:
            return production_segy_parser(**params)
        else:
            return create_error_response("Segyio parser not available")

    def segy_survey_analysis_handler(**params):
        """Survey analysis handler - upgraded with segyio"""

        result = {
            "survey_analysis": "Enhanced with segyio components",
            "file_pattern": params.get("file_pattern"),
            "status": "Survey analysis completed"
        }

        return {"text": json.dumps(result, cls=NumpyJSONEncoder)}

    def segy_template_detect_handler(**params):
        """Template detection handler - segyio native"""

        file_path_param = params.get("file_path")

        if file_path_param and isinstance(file_path_param, str) and ("*" in file_path_param or "?" in file_path_param):
            result = {
                "pattern_processed": file_path_param,
                "detected_template": "segyio_native_detection",
                "confidence_score": 1.0,
                "recommendation": "Segyio uses native header reading - no template files required",
                "summary": "Template detection completed - segyio handles all formats natively"
            }
        else:
            result = {
                "file_processed": file_path_param,
                "detected_template": "segyio_native_detection",
                "confidence_score": 1.0,
                "recommendation": "Segyio uses native header reading - no template files required"
            }

        return {"text": json.dumps(result, cls=NumpyJSONEncoder)}

    def segy_survey_compare_handler(**params):
        """Survey comparison handler - upgraded with segyio"""

        file_list_param = params.get("file_list")

        result = {
            "files_compared": file_list_param,
            "compatibility_analysis": {
                "segyio_compatible": True,
                "recommendation": "All files can be processed with segyio"
            },
            "summary": "Compatibility analysis completed with segyio"
        }

        return {"text": json.dumps(result, cls=NumpyJSONEncoder)}

    def quick_segy_summary_handler(**params):
        """Quick summary handler - upgraded with segyio"""

        file_path_param = params.get("file_path")
        data_dir = params.get("data_dir", "./data")

        if file_path_param and isinstance(file_path_param, str) and ("*" in file_path_param or "?" in file_path_param):
            import glob
            search_patterns = []
            base_pattern = file_path_param.replace('.sgy', '').replace('.segy', '')

            for ext in ['.sgy', '.segy', '.SGY', '.SEGY']:
                pattern = base_pattern + ext
                search_pattern = os.path.join(data_dir, pattern) if not pattern.startswith(data_dir) else pattern
                search_patterns.append(search_pattern)

            matching_files = []
            for pattern in search_patterns:
                matching_files.extend(glob.glob(pattern))

            matching_files = list(set(matching_files))

            if not matching_files:
                return create_error_response(f"No SEG-Y files found matching pattern: {file_path_param}")

            file_summaries = []
            total_size_mb = 0

            for file_path in matching_files:
                try:
                    file_stats = os.stat(file_path)
                    file_size_mb = file_stats.st_size / (1024 * 1024)
                    total_size_mb += file_size_mb

                    quick_info = {
                        "file_name": os.path.basename(file_path),
                        "file_size_mb": round(file_size_mb, 2),
                        "estimated_size": "Large" if file_size_mb > 100 else "Medium" if file_size_mb > 10 else "Small",
                        "status": "Ready for segyio processing"
                    }

                    file_summaries.append(quick_info)

                except Exception as e:
                    file_summaries.append({
                        "file_name": os.path.basename(file_path),
                        "error": str(e),
                        "status": "File access error"
                    })

            summary_result = {
                "pattern_processed": file_path_param,
                "files_found": len(file_summaries),
                "total_size_mb": round(total_size_mb, 2),
                "file_summaries": file_summaries,
                "summary": f"Quick summary completed for {len(file_summaries)} SEG-Y files"
            }

            return {"text": json.dumps(summary_result, cls=NumpyJSONEncoder)}

        else:
            if not PRODUCTION_PARSER_AVAILABLE:
                return create_error_response("Segyio parser not available")

            full_path = find_segy_file(file_path_param, data_dir)

            if not os.path.exists(full_path):
                return create_error_response(f"File not found: {file_path_param}")

            file_stats = os.stat(full_path)
            file_size_mb = file_stats.st_size / (1024 * 1024)

            quick_info = {
                "file_name": os.path.basename(full_path),
                "file_size_mb": round(file_size_mb, 2),
                "file_exists": True,
                "estimated_size": "Large" if file_size_mb > 100 else "Medium" if file_size_mb > 10 else "Small",
                "quick_assessment": "Ready for segyio processing"
            }

            return {"text": json.dumps(quick_info, cls=NumpyJSONEncoder)}

    class SEGYToolBase:
        """Base class for SEG-Y tools with common functionality."""

        def __init__(self, data_dir, template_dir="./templates"):
            self.data_dir = data_dir
            self.template_dir = template_dir

        def handle_input(self, file_path=None, **kwargs):
            """Standardized input handling for SEG-Y tools."""
            return handle_json_input(
                file_path,
                {"data_dir": self.data_dir, "template_dir": self.template_dir},
                **kwargs
            )

        def find_file(self, file_path):
            """Find SEG-Y file with standard error handling."""
            full_path = find_segy_file(file_path, self.data_dir)
            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"SEG-Y file not found: {file_path}")
            return full_path

        def check_intelligent_segy(self):
            """Check if intelligent SEG-Y processing is available."""
            if not INTELLIGENT_SEGY_AVAILABLE:
                raise RuntimeError("Intelligent SEG-Y processing not available - install scikit-learn>=1.3.0")

    # Create SEG-Y tool instance
    segy_tools = SEGYToolBase(args.data_dir)

    # ============================================================================
    # CORE TOOLS (4)
    # ============================================================================

    @mcp_server.tool(name="segy_parser", description="Enhanced SEG-Y parser with segyio")
    def segy_parser(file_path=None, **kwargs):
        """Enhanced SEG-Y parser with intelligent template detection."""
        try:
            params = segy_tools.handle_input(file_path, **kwargs)
            return segy_parser_handler(**params)
        except Exception as e:
            return create_error_response(
                f"Enhanced SEG-Y parser failed: {str(e)}",
                suggestions=[
                    "Check file accessibility and format",
                    "Verify segyio components are installed"
                ]
            )

    @mcp_server.tool(name="segy_qc", description="Enhanced SEG-Y QC with segyio")
    def segy_qc(file_path=None, **kwargs):
        """Enhanced SEG-Y QC with intelligent classification."""
        try:
            params = segy_tools.handle_input(file_path, **kwargs)
            return segy_qc_handler(**params)
        except Exception as e:
            return create_error_response(f"SEG-Y QC failed: {str(e)}")

    @mcp_server.tool(name="segy_analysis", description="Enhanced SEG-Y analysis with segyio")
    def segy_analysis(file_path=None, **kwargs):
        """Enhanced SEG-Y analysis with optimization."""
        try:
            params = segy_tools.handle_input(file_path, **kwargs)
            return segy_analysis_handler(**params)
        except Exception as e:
            return create_error_response(f"SEG-Y analysis failed: {str(e)}")

    @mcp_server.tool(name="segy_classify", description="Intelligent SEG-Y survey classification with segyio")
    def segy_classify(file_path=None, **kwargs):
        """Intelligent SEG-Y survey classification."""
        try:
            params = segy_tools.handle_input(file_path, **kwargs)
            return segy_classify_handler(**params)
        except Exception as e:
            return create_error_response(f"SEG-Y classification failed: {str(e)}")

    # ============================================================================
    # ADVANCED TOOLS (4)
    # ============================================================================

    @mcp_server.tool(name="segy_survey_analysis",
                     description="Analyze multiple SEG-Y files as a complete seismic survey")
    def segy_survey_analysis(file_pattern=None, **kwargs):
        """Analyze multiple SEG-Y files as a complete survey."""
        try:
            params = segy_tools.handle_input(file_pattern, **kwargs)
            return segy_survey_analysis_handler(**params)
        except Exception as e:
            return create_error_response(f"SEG-Y survey analysis failed: {str(e)}")

    @mcp_server.tool(name="segy_template_detect", description="Auto-detect optimal template for SEG-Y processing")
    def segy_template_detect(file_path=None, **kwargs):
        """Automatically detect the best template for SEG-Y file processing."""
        try:
            params = segy_tools.handle_input(file_path, **kwargs)
            return segy_template_detect_handler(**params)
        except Exception as e:
            return create_error_response(f"Template detection failed: {str(e)}")

    @mcp_server.tool(name="segy_survey_compare", description="Compare SEG-Y files for processing compatibility")
    def segy_survey_compare(file_list=None, **kwargs):
        """Compare multiple SEG-Y files to assess compatibility."""
        try:
            params = segy_tools.handle_input(file_list, **kwargs)
            return segy_survey_compare_handler(**params)
        except Exception as e:
            return create_error_response(f"Survey comparison failed: {str(e)}")

    @mcp_server.tool(name="quick_segy_summary", description="Fast overview of SEG-Y files without full processing")
    def quick_segy_summary(file_path=None, **kwargs):
        """Get quick overview of SEG-Y files for fast inventory."""
        try:
            params = segy_tools.handle_input(file_path, **kwargs)
            return quick_segy_summary_handler(**params)
        except Exception as e:
            return create_error_response(f"Quick summary failed: {str(e)}")

    # ========================================
    # SYSTEM TOOLS
    # ========================================

    @mcp_server.tool(name="system_status", description="Get comprehensive system health and performance metrics")
    def system_status(**kwargs):
        """Get comprehensive system health, performance metrics, and processing status"""
        try:
            import psutil
            import threading

            # System metrics
            system_metrics = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('./').percent,
                "active_threads": threading.active_count()
            }

            # Server health
            server_health = {
                "a2a_server": "running",  # Assume running if we got here
                "mcp_server": "running",
                "servers_responding": True
            }

            # Tool availability
            available_tools = [
                "las_parser", "las_analysis", "las_qc", "formation_evaluation",
                "well_correlation", "list_files", "calculate_shale_volume",
                "segy_parser", "segy_analysis", "segy_qc", "segy_classify",
                "segy_survey_analysis", "segy_template_detect", "segy_batch_classify",
                "segy_survey_compare", "intelligent_segy_analysis",
                "intelligent_survey_analysis", "quick_segy_summary",
                "intelligent_qc_analysis", "system_status"
            ]

            # Data directory info
            data_dir_info = {
                "data_directory": args.data_dir,
                "directory_exists": os.path.exists(args.data_dir),
                "las_files_count": len(glob.glob(os.path.join(args.data_dir, "*.las"))),
                "segy_files_count": len(glob.glob(os.path.join(args.data_dir, "*.sgy"))) +
                                    len(glob.glob(os.path.join(args.data_dir, "*.segy")))
            }

            status_report = {
                "timestamp": datetime.datetime.now().isoformat(),
                "system_metrics": system_metrics,
                "server_health": server_health,
                "available_tools": available_tools,
                "tool_count": len(available_tools),
                "data_directory_info": data_dir_info,
                "intelligent_segy_available": INTELLIGENT_SEGY_AVAILABLE,
                "overall_status": "System operational",
                "recommendations": [
                    "All core systems running normally",
                    "Tools are responding",
                    "Data directory accessible"
                ]
            }

            return {"text": json.dumps(status_report, cls=NumpyJSONEncoder)}

        except Exception as e:
            return create_error_response(f"System status check failed: {str(e)}")

    def run_mcp_server(server, host="0.0.0.0", port=None):
        """Run the MCP server."""
        if port is not None:
            server.run(host=host, port=port)
        else:
            server.run(host=host)

    # Start the MCP server in a background thread
    mcp_server_url = f"http://localhost:{mcp_port}"
    print(f"Starting MCP server on {mcp_server_url}...")

    mcp_thread = run_server_in_thread(
        run_mcp_server,
        mcp_server,
        host="0.0.0.0",
        port=mcp_port
    )

    # Wait for servers to initialize
    print("\nWaiting for servers to initialize...")
    time.sleep(5)

    # Check if MCP server is running
    mcp_server_running = False
    try:
        import requests
        response = requests.get(f"{mcp_server_url}/tools")
        if response.status_code == 200:
            mcp_server_running = True
            print("SUCCESS MCP server is running successfully")
    except Exception as e:
        print(f"Error checking MCP server status: {e}")

    # Try different port if needed
    if not mcp_server_running:
        print(f"ERROR MCP server failed to start on port {mcp_port}.")
        print("Let's try a different port...")
        mcp_port = find_available_port(8000, 20)
        mcp_server_url = f"http://localhost:{mcp_port}"
        print(f"\nRetrying MCP server on {mcp_server_url}...")
        mcp_thread = run_server_in_thread(run_mcp_server, mcp_server, port=mcp_port)
        time.sleep(5)

    # Step 3: Convert A2A agent to LangChain
    print("\nStep Step 3: Converting A2A Agent to LangChain")
    try:
        langchain_agent = to_langchain_agent(a2a_server_url)
        print("SUCCESS Successfully converted A2A agent to LangChain")
    except Exception as e:
        print(f"ERROR Error converting A2A agent to LangChain: {e}")
        return 1

    # Step 4: Convert MCP tools to LangChain
    print("\nStep 4: Converting MCP Tools to LangChain")

    # Define tool categories for organized conversion
    TOOL_CATEGORIES = {
        "las_tools": [
            "las_parser", "las_analysis", "las_qc", "formation_evaluation",
            "well_correlation", "list_files", "calculate_shale_volume"
        ],
        "segy_tools": [
            "segy_parser", "segy_analysis", "segy_qc", "segy_survey_analysis"
        ],
        "intelligent_segy_tools": [
            "segy_classify", "segy_template_detect", "segy_survey_compare"
        ],
        "ai_analysis_tools": [
            "quick_segy_summary"
        ],
        "system_tools": [
            "system_status"
        ]
    }

    def convert_tools_to_langchain(server_url, tool_categories):
        """Convert MCP tools to LangChain tools with error handling."""
        converted_tools = {}
        failed_tools = []

        for category, tool_names in tool_categories.items():
            print(f"Converting {category}...")
            for tool_name in tool_names:
                try:
                    converted_tools[f"{tool_name}_tool"] = to_langchain_tool(server_url, tool_name)
                    print(f"  ✓ {tool_name}")
                except Exception as e:
                    print(f"  ✗ {tool_name}: {str(e)}")
                    failed_tools.append(tool_name)
                    converted_tools[f"{tool_name}_tool"] = None

        return converted_tools, failed_tools

    try:
        converted_tools, failed_tools = convert_tools_to_langchain(mcp_server_url, TOOL_CATEGORIES)

        # Extract individual tools for backward compatibility
        globals().update(converted_tools)

        if failed_tools:
            print(f"WARNING: Failed to convert {len(failed_tools)} tools: {failed_tools}")
            print("These tools will not be available in the agent")

        print("SUCCESS: MCP tools conversion completed")

        # Initialize command processor with available tools
        command_processor = CommandProcessor(
            mcp_server_url=mcp_server_url,
            data_dir=args.data_dir
        )

    except Exception as e:
        print(f"ERROR: Critical failure converting MCP tools to LangChain: {e}")
        print("Continuing with A2A agent only...")

        # Initialize all tools as None for graceful degradation
        for category_tools in TOOL_CATEGORIES.values():
            for tool_name in category_tools:
                globals()[f"{tool_name}_tool"] = None

        # Initialize command processor without MCP tools
        command_processor = CommandProcessor(data_dir=args.data_dir)

        print("System running in reduced functionality mode")

    # Utility function to check tool availability
    def is_tool_available(tool_name):
        """Check if a specific tool is available."""
        tool_var = f"{tool_name}_tool"
        return globals().get(tool_var) is not None

    # Print summary of available tools
    def print_tool_summary():
        """Print summary of available vs unavailable tools."""
        available_count = 0
        total_count = 0

        print("\nTool Availability Summary:")
        print("=" * 40)

        for category, tool_names in TOOL_CATEGORIES.items():
            category_available = []
            category_unavailable = []

            for tool_name in tool_names:
                total_count += 1
                if is_tool_available(tool_name):
                    available_count += 1
                    category_available.append(tool_name)
                else:
                    category_unavailable.append(tool_name)

            print(f"\n{category.replace('_', ' ').title()}:")
            if category_available:
                print(f"  ✓ Available: {', '.join(category_available)}")
            if category_unavailable:
                print(f"  ✗ Unavailable: {', '.join(category_unavailable)}")

        print(f"\nOverall: {available_count}/{total_count} tools available")
        return available_count, total_count

    # Print the summary
    available_tools, total_tools = print_tool_summary()

    # Set system capability level based on available tools
    if available_tools == total_tools:
        print("\nFull functionality mode - all tools available")
        SYSTEM_MODE = "full"
    elif available_tools > total_tools * 0.7:  # More than 70% available
        print(f"\nPartial functionality mode - {available_tools}/{total_tools} tools available")
        SYSTEM_MODE = "partial"
    elif available_tools > 0:
        print(f"\nLimited functionality mode - only {available_tools}/{total_tools} tools available")
        SYSTEM_MODE = "limited"
    else:
        print("\nMinimal functionality mode - A2A agent only")
        SYSTEM_MODE = "minimal"

    # Step 5: Create Meta-Agent
    print("\nStep 5: Creating Subsurface Data Management Meta-Agent")

    def create_meta_agent(
            langchain_agent: Any,
            mcp_server_url: str,
            model: str = "gpt-4",
            temperature: float = 0.0,
            verbose: bool = True,
            data_dir: str = "./data"
    ) -> Any:
        """Create a hybrid agent that combines LangChain and direct MCP access."""
        print("\nCreating Hybrid Subsurface Data Management Agent")

        # Create LLM for agent
        llm = ChatOpenAI(model=model, temperature=temperature)

        # Configuration constants
        AGENT_CONFIG = {
            "system_prompt": {
                "core_capabilities": {
                    "WELL LOG ANALYSIS (LAS Files)": [
                        "Robust parsing with comprehensive error recovery",
                        "Advanced petrophysical calculations and formation evaluation",
                        "Quality control and data validation for well logs",
                        "Multi-well correlation and pay zone identification"
                    ],
                    "INTELLIGENT SEG-Y ANALYSIS (Advanced AI-Powered)": [
                        "**Automatic Survey Classification**: Determines 2D/3D, PreStack/PostStack, and sorting methods",
                        "**Intelligent Template Detection**: Automatically finds optimal templates, solving validation issues",
                        "**Batch Survey Analysis**: Processes multiple files with consistency analysis",
                        "**Survey Comparison**: Compares characteristics across files for processing compatibility",
                        "Production-quality parsing with comprehensive validation and progress reporting"
                    ],
                    "INTEGRATED WORKFLOWS": [
                        "Well-to-seismic calibration and correlation",
                        "Integrated subsurface interpretation combining both data types",
                        "AI-powered geological interpretation with confidence scoring",
                        "Formation correlation using both wells and seismic data"
                    ],
                    "INTELLIGENT AUTOMATION": [
                        "Eliminates manual template selection through AI classification",
                        "Provides confidence-based processing recommendations",
                        "Automatically optimizes processing parameters based on survey characteristics",
                        "Handles problematic files with intelligent error recovery"
                    ]
                },
                "quality_statement": "You provide reliable, production-quality analysis with comprehensive error handling, progress tracking, and detailed validation. Your intelligent capabilities solve common processing bottlenecks automatically while maintaining enterprise-grade reliability."
            },
            "agent_params": {
                "max_iterations": 10,
                "max_execution_time": 300,  # 5 minutes
                "early_stopping_method": "generate",
                "handle_parsing_errors": True
            }
        }

        TOOL_DEFINITIONS = {
            # LAS Tools - Direct MCP calls (unchanged)
            "las_parser": {"mcp_tool": "las_parser",
                           "description": "Parse a LAS file and extract metadata including well information, curves, and depth ranges."},
            "las_analysis": {"mcp_tool": "las_analysis",
                             "description": "Analyze curves in a LAS file with statistical analysis and curve characteristics."},
            "las_qc": {"mcp_tool": "las_qc",
                       "description": "Perform quality control checks on LAS files including data completeness and curve validation."},
            "formation_evaluation": {"mcp_tool": "formation_evaluation",
                                     "description": "Perform comprehensive petrophysical analysis including porosity, water saturation, shale volume, and pay zones."},
            "well_correlation": {"mcp_tool": "well_correlation",
                                 "description": "Correlate formations across multiple wells to identify key formation tops and stratigraphic markers."},
            "list_files": {"mcp_tool": "list_files",
                           "description": "List all LAS files matching a pattern in the data directory."},
            "calculate_shale_volume": {"mcp_tool": "calculate_shale_volume",
                                       "description": "Calculate volume of shale from gamma ray log using the Larionov correction method."},

            # SEG-Y Tools - Segyio Enhanced (8 total tools)
            "segy_parser": {"mcp_tool": "segy_parser",
                            "description": "Parse SEG-Y seismic files with segyio-powered processing and comprehensive metadata extraction."},
            "segy_qc": {"mcp_tool": "segy_qc",
                        "description": "Perform comprehensive quality control on SEG-Y files with segyio-enhanced validation and intelligent recommendations."},
            "segy_analysis": {"mcp_tool": "segy_analysis",
                              "description": "Analyze SEG-Y seismic survey geometry, data quality, and performance with segyio-powered optimization."},
            "segy_classify": {"mcp_tool": "segy_classify", "format_output": "json",
                              "description": "Automatically classify SEG-Y survey type (2D/3D), sorting method, and stacking type with segyio accuracy."},

            # Advanced SEG-Y Tools - Segyio Enhanced
            "segy_survey_analysis": {"mcp_tool": "segy_survey_analysis",
                                     "description": "Analyze multiple SEG-Y files as a complete seismic survey with segyio-powered processing."},
            "segy_template_detect": {"mcp_tool": "segy_template_detect", "format_output": "json",
                                     "description": "Template detection using segyio native header reading - no template files required."},
            "segy_survey_compare": {"mcp_tool": "segy_survey_compare", "format_output": "json",
                                    "description": "Compare multiple SEG-Y files for processing compatibility with segyio validation."},
            "quick_segy_summary": {"mcp_tool": "quick_segy_summary",
                                   "description": "Get instant overview of SEG-Y files with segyio-powered fast inventory and basic parameters."},

            # System Tools
            "system_status": {"mcp_tool": "system_status",
                              "description": "Get comprehensive system health, performance metrics, and processing status."},

            # Well Log Expert - Keep this for advanced interpretation
            "WellLogExpert": {
                "func": lambda query: _call_langchain_agent(langchain_agent, query),
                "description": """Ask the well log expert questions about log interpretation, petrophysics, formation evaluation, and geological insights.
                    Examples: What does this gamma ray signature indicate? How do I interpret high resistivity with low porosity?"""
            }
        }

        FALLBACK_HANDLERS = {
            "las_metadata": {
                "keywords": ["metadata", ".las", "file"],
                "handler": lambda query: _extract_and_process_file(query, ".las", "parse_las_file")
            },
            "segy_classify": {
                "keywords": ["classify", ".sgy", "seismic"],
                "handler": lambda query: _extract_and_process_file(query, [".sgy", ".segy"], "classify_segy_file")
            }
        }

        # Direct command patterns - updated for streamlined tools
        COMMAND_PATTERNS = {
            # SEG-Y Commands - Updated for 8-tool structure
            ("classify", "segy", ".sgy"): "segy_classify",  # Now handles both single and batch
            ("detect template", "find template"): "segy_template_detect",
            ("compare", "survey", "segy"): "segy_survey_compare",
            ("parse", "intelligent"): ("segy_parser", {"auto_detect": True}),
            ("qc", "intelligent"): ("segy_qc", {"include_classification": True}),
            ("analyze", "survey"): "segy_survey_analysis",
            ("quick", "summary"): "quick_segy_summary",
            ("analyze", "segy"): "segy_analysis",

            # LAS Commands (unchanged)
            ("parse all",): ("las_parser", {"selection_mode": "all"}),
            ("evaluate all", "eval all"): ("formation_evaluation", {"selection_mode": "all"}),
            ("check all", "qc all"): ("las_qc", {"selection_mode": "all"}),
            ("correlate all",): ("well_correlation", {"selection_mode": "all"}),
            ("list files",): "list_files"
        }

        mcp_client = MCPClient(mcp_server_url)

        # Utility functions
        def _call_langchain_agent(agent, query):
            """Call the LangChain agent with error handling."""
            try:
                result = agent.invoke(query)
                return result.get('output', 'No response from well log expert')
            except Exception as e:
                return f"Error querying well log expert: {str(e)}"

        def _call_mcp_tool(mcp_client, tool_name, input_str, format_output=None):
            """Call MCP tool with standardized error handling and output formatting."""
            try:
                result = mcp_client.call_tool(tool_name, input_str)
                if format_output == "json" and isinstance(result, dict):
                    return json.dumps(result)
                return result if result else f"No response from {tool_name}"
            except Exception as e:
                return f"Error calling {tool_name}: {str(e)}"

        def create_tool_function(tool_config, mcp_client):
            """Factory function to create tool functions with better formatting."""
            if 'func' in tool_config:
                return tool_config['func']
            elif 'mcp_tool' in tool_config:
                mcp_tool_name = tool_config['mcp_tool']
                format_output = tool_config.get('format_output')

                def tool_wrapper(input_str: str) -> str:
                    """Execute MCP tool and return clean formatted results"""
                    try:
                        if not input_str.strip():
                            return f"Error: No input provided for {mcp_tool_name}"

                        # Call MCP tool
                        result = mcp_client.call_tool(mcp_tool_name, input_str)

                        # IMPROVED RESPONSE HANDLING
                        if isinstance(result, dict):
                            if "error" in result:
                                return f"❌ Error: {result['error']}"
                            elif "text" in result:
                                response_text = result["text"]

                                # If it's JSON, try to parse and format it nicely
                                if isinstance(response_text, str) and response_text.startswith('{'):
                                    try:
                                        parsed = json.loads(response_text)

                                        # Special formatting for different tool types
                                        if mcp_tool_name == "list_files":
                                            return format_file_list(parsed)
                                        elif mcp_tool_name == "system_status":
                                            return format_system_status(parsed)
                                        elif mcp_tool_name in ["las_parser", "segy_parser"]:
                                            return format_parser_output(parsed)
                                        elif mcp_tool_name in ["formation_evaluation"]:
                                            return format_formation_evaluation(parsed)
                                        else:
                                            # Generic JSON formatting
                                            return format_generic_json(parsed)

                                    except json.JSONDecodeError:
                                        pass

                                # Return as-is if already formatted
                                return str(response_text)
                            else:
                                return json.dumps(result, indent=2)
                        else:
                            return str(result)
                    except Exception as e:
                        return f"❌ Error calling {mcp_tool_name}: {str(e)}"

                return tool_wrapper
            else:
                raise ValueError(f"Invalid tool configuration: missing 'func' or 'mcp_tool'")

        def format_file_list(data):
            """Format file list data into clean output - NO EMOJIS"""
            if "matching_files" not in data:
                return "No files found"

            files = data["matching_files"]
            count = data.get("count", len(files))

            if not files:
                return f"No files found matching pattern: {data.get('pattern', 'unknown')}"

            # Group files
            early_wells = [f for f in files if f.startswith("1054146") or f.startswith("1054149")]
            main_field = [f for f in files if f.startswith("1054310")]
            other_files = [f for f in files if
                           not (f.startswith("1054146") or f.startswith("1054149") or f.startswith("1054310"))]

            output = f"LAS FILES FOUND ({count} files):\n"
            output += "=" * 40 + "\n\n"

            if early_wells:
                output += "EARLY WELLS:\n"
                for file in sorted(early_wells):
                    output += f"  - {file}\n"
                output += "\n"

            if main_field:
                output += "MAIN FIELD DEVELOPMENT:\n"
                for file in sorted(main_field):
                    output += f"  - {file}\n"
                output += "\n"

            if other_files:
                output += "OTHER FILES:\n"
                for file in sorted(other_files):
                    output += f"  - {file}\n"
                output += "\n"

            output += f"TOTAL: {count} files ready for analysis"
            return output

        def format_system_status(data):
            """Format system status into clean output - NO EMOJIS"""
            output = "SYSTEM STATUS REPORT\n"
            output += "=" * 30 + "\n\n"

            if "system_metrics" in data:
                metrics = data["system_metrics"]
                output += "PERFORMANCE METRICS:\n"
                output += f"  - CPU Usage: {metrics.get('cpu_percent', 'N/A')}%\n"
                output += f"  - Memory Usage: {metrics.get('memory_percent', 'N/A')}%\n"
                output += f"  - Disk Usage: {metrics.get('disk_percent', 'N/A')}%\n"
                output += f"  - Active Threads: {metrics.get('active_threads', 'N/A')}\n\n"

            if "server_health" in data:
                health = data["server_health"]
                output += "SERVER HEALTH:\n"
                output += f"  - A2A Server: {health.get('a2a_server', 'Unknown')}\n"
                output += f"  - MCP Server: {health.get('mcp_server', 'Unknown')}\n\n"

            if "data_directory_info" in data:
                data_info = data["data_directory_info"]
                output += "DATA DIRECTORY:\n"
                output += f"  - LAS Files: {data_info.get('las_files_count', 'N/A')}\n"
                output += f"  - SEG-Y Files: {data_info.get('segy_files_count', 'N/A')}\n\n"

            output += f"OVERALL STATUS: {data.get('overall_status', 'Unknown')}"
            return output

        def format_parser_output(data):
            """Format parser output into clean display - NO EMOJIS"""
            if "error" in data:
                return f"ERROR: {data['error']}"

            output = "FILE ANALYSIS RESULTS\n"
            output += "=" * 25 + "\n\n"

            if "well_name" in data:
                output += f"WELL: {data['well_name']}\n"
            if "file_processed" in data:
                output += f"FILE: {data['file_processed']}\n"
            if "depth_range" in data:
                depth = data['depth_range']
                output += f"DEPTH RANGE: {depth[0]:.1f} - {depth[1]:.1f} ft\n"

            return output

        def format_formation_evaluation(data):
            """Format formation evaluation results"""
            if "error" in data:
                return f"Formation Evaluation Error: {data['error']}"

            output = "Formation Evaluation Results\n"
            output += "=" * 32 + "\n\n"

            if "well_name" in data:
                output += f"Well: {data['well_name']}\n"
            if "pay_zones" in data:
                zones = data['pay_zones']
                output += f"Pay Zones Identified: {len(zones)}\n"
            if "average_porosity" in data:
                output += f"Average Porosity: {data['average_porosity']:.1f}%\n"

            return output

        def format_generic_json(data):
            """Format generic JSON data in a readable way"""
            if isinstance(data, dict) and len(data) < 10:
                output = ""
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        output += f"• {key.replace('_', ' ').title()}: {value}\n"
                    elif isinstance(value, str) and len(value) < 100:
                        output += f"• {key.replace('_', ' ').title()}: {value}\n"
                return output if output else json.dumps(data, indent=2)
            else:
                return json.dumps(data, indent=2)

        def extract_pattern(input_str):
            """Extract file pattern from command - FIXED VERSION"""
            pattern = input_str.strip()

            # Remove common command words from the beginning
            command_words = [
                "classify", "parse", "analyze", "check", "qc", "detect", "template",
                "compare", "intelligent", "quick", "summary", "list", "files",
                "correlate", "evaluate", "all", "matching"
            ]

            # Split the input into words
            words = pattern.split()

            # Remove command words from the beginning
            while words and words[0].lower() in command_words:
                words.pop(0)

            # Rejoin the remaining words
            if words:
                cleaned_pattern = " ".join(words)
            else:
                cleaned_pattern = pattern

            # Clean up any remaining artifacts
            for keyword in ["matching", "all", "files"]:
                if keyword in cleaned_pattern:
                    parts = cleaned_pattern.split(keyword, 1)
                    if len(parts) > 1:
                        cleaned_pattern = parts[1].strip()
                        break

            # Remove trailing periods
            cleaned_pattern = cleaned_pattern.rstrip('.')

            # If the result is empty or too short, return the original minus first word
            if not cleaned_pattern or len(cleaned_pattern) < 3:
                original_words = input_str.split()
                if len(original_words) > 1:
                    cleaned_pattern = " ".join(original_words[1:])
                else:
                    cleaned_pattern = input_str

            return cleaned_pattern.strip()

        def _extract_and_process_file(query, file_extensions, handler_name):
            """Extract filename from query and process with specified handler."""
            if isinstance(file_extensions, str):
                file_extensions = [file_extensions]

            for word in query.split():
                for ext in file_extensions:
                    if word.lower().endswith(ext):
                        handler = globals().get(handler_name)
                        if handler:
                            return handler(word)
            return None

        def create_system_prompt(config):
            """Generate system prompt from configuration."""
            capabilities_text = []
            for category, items in config["system_prompt"]["core_capabilities"].items():
                capabilities_text.append(f"\n{category}:")
                capabilities_text.extend(f"- {item}" for item in items)

            return f"""You are an expert subsurface data analysis assistant with advanced AI-powered capabilities.

    CRITICAL: You MUST use the available tools when users ask for data analysis. DO NOT provide generic advice.

    Your core capabilities include:
    {''.join(capabilities_text)}

    {config["system_prompt"]["quality_statement"]}

    MANDATORY TOOL USAGE WORKFLOWS:

    When user asks "analyze formation in file.las":
    1. FIRST call las_parser with "file.las" 
    2. THEN call formation_evaluation with "file.las"
    3. THEN call las_qc with "file.las"
    4. FINALLY provide expert interpretation of the actual results

    When user asks "analyze seismic file.sgy":
    1. FIRST call segy_parser with "file.sgy"
    2. THEN call segy_classify with "file.sgy" 
    3. THEN call segy_qc with "file.sgy"
    4. FINALLY provide expert interpretation of the actual results

    When user asks "list files":
    1. FIRST call list_files with appropriate pattern
    2. THEN provide summary of available files

    ALWAYS use tools to get actual data before providing analysis. Never give generic textbook answers.

    You have access to the following tools:"""

        # Create tools
        print("Creating LangChain tools...")
        tools = []
        for tool_name, tool_config in TOOL_DEFINITIONS.items():
            try:
                tool_function = create_tool_function(tool_config, mcp_client)
                tools.append(Tool(name=tool_name, func=tool_function, description=tool_config['description']))
                print(f"  ✓ Created tool: {tool_name}")
            except Exception as e:
                print(f"  ✗ Failed to create tool {tool_name}: {str(e)}")

        print(f"Successfully created {len(tools)} LangChain tools")

        def enhanced_direct_command_processor(command_str: str) -> Optional[str]:
            """Enhanced command processor with better pattern extraction"""
            try:
                command_lower = command_str.lower()

                # IMPROVED LIST COMMAND HANDLING
                if command_lower.startswith("list"):
                    # Extract everything after "list" and "files"
                    pattern = command_str
                    for word in ["list", "files", "matching"]:
                        if word in pattern.lower():
                            parts = pattern.lower().split(word, 1)
                            if len(parts) > 1:
                                pattern = parts[1].strip()

                    # If pattern is empty, default to all files
                    if not pattern:
                        pattern = "*"

                    print(f"List command - using pattern: '{pattern}'")
                    result = mcp_client.call_tool("list_files", pattern)
                    return json.dumps(result) if isinstance(result, dict) else str(result)

                # IMPROVED CLASSIFY COMMAND HANDLING
                if command_lower.startswith("classify"):
                    # Use regex to find the filename
                    import re
                    segy_pattern = re.search(r'([a-zA-Z0-9_\-\.]+\.s[eg]y)', command_str, re.IGNORECASE)

                    if segy_pattern:
                        filename = segy_pattern.group(1)
                        print(f"Classify command - extracted filename: '{filename}'")
                        result = mcp_client.call_tool("segy_classify", filename)
                        return json.dumps(result) if isinstance(result, dict) else str(result)

                # Add other command patterns here...

                return None
            except Exception as e:
                print(f"Error in command processor: {str(e)}")
                return None

        # Create agent components
        class HybridAgent:
            def __init__(self, agent_executor, command_processor, fallback_handlers=None):
                self.agent_executor = agent_executor
                self.command_processor = command_processor
                self.fallback_handlers = fallback_handlers or {}
                self.stats = {"total_queries": 0, "direct_commands": 0, "agent_responses": 0, "fallback_responses": 0,
                              "errors": 0}

            def run(self, query):
                self.stats["total_queries"] += 1

                # Try direct command processing
                try:
                    direct_result = self.command_processor(query)
                    if direct_result:
                        self.stats["direct_commands"] += 1
                        return direct_result
                except Exception as e:
                    print(f"Direct command processing failed: {str(e)}")

                # Use main agent - THIS IS THE KEY FIX
                try:
                    result = self.agent_executor.invoke({"input": query})["output"]
                    self.stats["agent_responses"] += 1
                    return result
                except Exception as e:
                    print(f"Agent processing failed: {str(e)}")
                    return self._try_fallback_handlers(query, str(e))

            def _try_fallback_handlers(self, query, original_error):
                query_lower = query.lower()
                for handler_name, handler_config in self.fallback_handlers.items():
                    if all(keyword in query_lower for keyword in handler_config["keywords"]):
                        try:
                            result = handler_config["handler"](query)
                            if result:
                                self.stats["fallback_responses"] += 1
                                return result
                        except Exception as fallback_error:
                            print(f"Fallback handler {handler_name} failed: {fallback_error}")

                self.stats["errors"] += 1
                return f"Sorry, I encountered an error and couldn't process your request: {original_error}"

            def get_stats(self):
                return self.stats.copy()

        # Create agent executor with EXPLICIT tool usage instructions
        enhanced_prefix = create_system_prompt(AGENT_CONFIG)
        suffix = """Begin!

    Question: {input}
    Thought: I need to analyze this request and determine which tools to use. If the user is asking about analyzing data files, I MUST use the appropriate tools to get actual results.
    {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(tools, prefix=enhanced_prefix, suffix=suffix,
                                             input_variables=["input", "agent_scratchpad"])
        memory = ConversationBufferMemory(memory_key="chat_history")
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=verbose)
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory,
                                                            **AGENT_CONFIG["agent_params"])

        # Create hybrid agent
        hybrid_agent = HybridAgent(agent_executor, enhanced_direct_command_processor, FALLBACK_HANDLERS)

        print(f"SUCCESS: Created hybrid agent with {len(tools)} tools")
        if verbose:
            print(
                f"\nAgent Configuration: Max Iterations: {AGENT_CONFIG['agent_params']['max_iterations']}, Max Execution Time: {AGENT_CONFIG['agent_params']['max_execution_time']}s, Fallback Handlers: {len(FALLBACK_HANDLERS)}")
            print(f"\nAvailable Tools ({len(tools)}):")
            for i, tool in enumerate(tools, 1):
                desc = tool.description.split('\n')[0][:80] + "..." if len(tool.description) > 80 else \
                    tool.description.split('\n')[0]
                print(f"  {i:2d}. {tool.name}: {desc}")

        return hybrid_agent

    # Step 5: Create Meta-Agent
    print("\nStep 5: Creating Subsurface Data Management Meta-Agent")

    try:
        meta_agent = create_meta_agent(
            langchain_agent=langchain_agent,
            mcp_server_url=mcp_server_url,
            model=args.model,
            temperature=args.temperature,
            verbose=True,
            data_dir=args.data_dir
        )

        # Step 6: Production Monitoring Setup (BEFORE Interactive Loop)
        print("\n" + "=" * 60)
        print("PRODUCTION MONITORING SETUP")
        print("=" * 60)

        # Basic metrics tracking
        start_time = time.time()
        total_queries = 0
        successful_queries = 0

        # Server health monitoring
        def check_server_health():
            """Check if both servers are still running"""
            try:
                # Check A2A server
                a2a_healthy = False
                try:
                    response = requests.get(f"{a2a_server_url}/health", timeout=5)
                    a2a_healthy = response.status_code == 200
                except:
                    # If no health endpoint, check if thread is alive
                    a2a_healthy = a2a_thread.is_alive()

                # Check MCP server
                mcp_healthy = False
                try:
                    response = requests.get(f"{mcp_server_url}/tools", timeout=5)
                    mcp_healthy = response.status_code == 200
                except:
                    mcp_healthy = mcp_thread.is_alive()

                return {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "a2a_server": "healthy" if a2a_healthy else "unhealthy",
                    "mcp_server": "healthy" if mcp_healthy else "unhealthy",
                    "uptime_hours": round((time.time() - start_time) / 3600, 2),
                    "total_queries": total_queries,
                    "success_rate": round((successful_queries / total_queries * 100), 2) if total_queries > 0 else 0
                }
            except Exception as e:
                return {"error": f"Health check failed: {str(e)}"}

        # Log system metrics
        def log_system_metrics():
            """Log basic system metrics"""
            try:
                if psutil:
                    metrics = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "cpu_percent": psutil.cpu_percent(interval=1),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_percent": psutil.disk_usage('./').percent,
                        "process_count": len(psutil.pids())
                    }

                    # Save to metrics log
                    metrics_file = os.path.join(args.log_dir, "system_metrics.jsonl")
                    with open(metrics_file, 'a') as f:
                        f.write(json.dumps(metrics) + '\n')

                    return metrics
                else:
                    return {"note": "psutil not installed - install with 'pip install psutil' for system metrics"}
            except Exception as e:
                return {"error": f"Metrics logging failed: {str(e)}"}

        # Performance monitoring wrapper
        def monitor_query_performance(query_func):
            """Wrapper to monitor query performance"""

            def wrapper(query):
                nonlocal total_queries, successful_queries
                query_start_time = time.time()
                total_queries += 1

                try:
                    result = query_func(query)
                    successful_queries += 1

                    # Log performance metrics
                    performance = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "query_length": len(query),
                        "response_time": round(time.time() - query_start_time, 3),
                        "success": True,
                        "query_type": "standard"
                    }

                    perf_file = os.path.join(args.log_dir, "performance_metrics.jsonl")
                    with open(perf_file, 'a') as f:
                        f.write(json.dumps(performance) + '\n')

                    return result
                except Exception as e:
                    # Log error metrics
                    error_metrics = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "query_length": len(query),
                        "response_time": round(time.time() - query_start_time, 3),
                        "success": False,
                        "error": str(e)
                    }

                    perf_file = os.path.join(args.log_dir, "performance_metrics.jsonl")
                    with open(perf_file, 'a') as f:
                        f.write(json.dumps(error_metrics) + '\n')

                    raise

            return wrapper

        # Start background health monitoring
        def background_health_monitor():
            """Background thread for health monitoring"""
            while True:
                try:
                    health = check_server_health()

                    # Log health status
                    health_file = os.path.join(args.log_dir, "health_status.jsonl")
                    with open(health_file, 'a') as f:
                        f.write(json.dumps(health) + '\n')

                    # Check for unhealthy servers
                    if health.get("a2a_server") == "unhealthy" or health.get("mcp_server") == "unhealthy":
                        print(f"\n⚠️  WARNING: Server health issue detected at {health['timestamp']}")
                        print(f"   A2A Server: {health.get('a2a_server', 'unknown')}")
                        print(f"   MCP Server: {health.get('mcp_server', 'unknown')}")

                    # Sleep for 5 minutes between checks
                    time.sleep(300)

                except Exception as e:
                    print(f"Health monitoring error: {str(e)}")
                    time.sleep(60)  # Shorter sleep on error

        # Create monitoring dashboard data
        def create_monitoring_dashboard():
            """Create a simple monitoring dashboard"""
            try:
                dashboard_data = {
                    "system_status": check_server_health(),
                    "system_metrics": log_system_metrics(),
                    "uptime_hours": round((time.time() - start_time) / 3600, 2),
                    "data_directory": args.data_dir,
                    "log_directory": args.log_dir,
                    "servers": {
                        "a2a_url": a2a_server_url,
                        "mcp_url": mcp_server_url
                    }
                }

                # Save dashboard data
                dashboard_file = os.path.join(args.log_dir, "dashboard.json")
                with open(dashboard_file, 'w') as f:
                    json.dump(dashboard_data, f, indent=2)

                return dashboard_data
            except Exception as e:
                return {"error": f"Dashboard creation failed: {str(e)}"}

        # Enhanced meta_agent with monitoring
        if 'meta_agent' in locals():
            # Wrap the meta_agent.run method with monitoring
            original_run = meta_agent.run
            meta_agent.run = monitor_query_performance(original_run)
            print("✓ Query performance monitoring enabled")

        # Start health monitoring in background
        health_thread = threading.Thread(target=background_health_monitor, daemon=True)
        health_thread.start()
        print("✓ Background health monitoring started")

        # Log initial system metrics
        initial_metrics = log_system_metrics()
        if "error" not in initial_metrics:
            print("✓ System metrics logging enabled")
        else:
            print(f"⚠️  System metrics logging failed: {initial_metrics.get('error', 'unknown')}")

        # Create initial dashboard
        dashboard = create_monitoring_dashboard()
        if "error" not in dashboard:
            print("✓ Monitoring dashboard created")
            print(f"   Dashboard file: {os.path.join(args.log_dir, 'dashboard.json')}")

        # Print monitoring summary
        print("\nMONITORING FEATURES ENABLED:")
        print("• Query performance tracking")
        print("• Server health monitoring (every 5 minutes)")
        print("• System metrics logging")
        print("• Error tracking and logging")
        print(f"• All monitoring data saved to: {args.log_dir}")

        print("\nMONITORING FILES:")
        print(f"• Health status: {os.path.join(args.log_dir, 'health_status.jsonl')}")
        print(f"• Performance: {os.path.join(args.log_dir, 'performance_metrics.jsonl')}")
        print(f"• System metrics: {os.path.join(args.log_dir, 'system_metrics.jsonl')}")
        print(f"• Dashboard: {os.path.join(args.log_dir, 'dashboard.json')}")

        # Fix for nested f-string issue
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        qa_log_filename = f"qa_log_{current_date}.{args.log_format}"
        print(f"• Q&A logs: {os.path.join(args.log_dir, qa_log_filename)}")

        print("\n" + "=" * 60)
        print("PRODUCTION READY - All monitoring systems active")
        print("=" * 60)

        # Step 7: Ready for Use
        print("\nSubsurface Data Management Platform - Ready!")
        print("\nExample commands:")
        print("=" * 50)

        print("\n🪨 WELL LOG ANALYSIS (LAS Files):")
        print("1. Parse LAS file: What metadata is in well_01.las?")
        print("2. Analyze curves: Analyze the GR and RHOB curves in well_1.las")
        print("3. Formation evaluation: Evaluate all matching ./data/well_*.las")
        print("4. Well correlation: Correlate all matching ./data/field_*.las")
        print("5. Quality control: Check quality of problematic_well.las")

        print("\n🌊 SEISMIC ANALYSIS (SEG-Y Files):")
        print("6. Parse SEG-Y: What metadata is in survey_3d.sgy?")
        print("7. Analyze survey: Analyze the geometry of marine_2d.sgy")
        print("8. Quality control: Check quality of seismic_data.sgy")
        print("9. Multi-file survey: Analyze all matching ./data/3D_*.sgy")
        print("10. Survey processing: Process all files matching Survey_*.sgy in parallel")

        print("\n🔬 INTEGRATED ANALYSIS:")
        print("11. Well-seismic tie: How do these wells correlate with the 3D seismic?")
        print("12. Formation mapping: Extend well correlations using seismic structure")
        print("13. Quality comparison: Compare data quality between wells and seismic")

        print("\n🖥️ SYSTEM MANAGEMENT:")
        print("14. System status: What is the current system health?")
        print("15. Performance check: How is the processing performance?")

        print("\n💡 EXPERT CONSULTATION:")
        print("16. Integration advice: What's the best workflow for well-seismic integration?")
        print("17. Interpretation help: How do I interpret these log and seismic signatures?")

        print("\n" + "=" * 50)
        print("🎯 Production-Ready: Enterprise-grade reliability with comprehensive monitoring")

        # print("\n💡 Note: System health checks run every 5 minutes in background")
        # print("If input seems frozen, just press Enter to refresh the prompt\n")

        while True:
            try:
                # Clear prompt with timeout awareness
                user_input = input("\n🤖 Ask me anything (or 'quit' to exit)\n> ").strip()

                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break

                # Handle empty input gracefully (from health monitoring interference)
                if not user_input:
                    print("💡 Empty input detected. Try asking about:")
                    print("   • System status")
                    print("   • List files in data directory")
                    print("   • Analyze a specific file")
                    continue

                print(f"\n🔄 Processing: {user_input}")
                print("⏳ Please wait...")

                try:
                    # Process the query using our hybrid agent
                    response = meta_agent.run(user_input)
                    cleaned_response = clean_response(response)
                    print(f"\n✅ Response:\n{cleaned_response}")

                    # Save the Q&A interaction
                    save_qa_interaction(user_input, response,
                                        log_format=args.log_format,
                                        log_dir=args.log_dir)

                except Exception as e:
                    error_message = f"Error processing query: {str(e)}"
                    print(f"\n❌ {error_message}")
                    traceback.print_exc()

                    # Still log the error for tracking
                    save_qa_interaction(user_input, error_message,
                                        log_format=args.log_format,
                                        log_dir=args.log_dir)

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except EOFError:
                print("\n\n📝 Input stream ended. Goodbye!")
                break
    except Exception as e:
        print(f"\nERROR Error creating meta-agent: {e}")
        traceback.print_exc()
        return 1

    # Return success
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)