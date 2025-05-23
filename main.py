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
from pathlib import Path

# Environment Setup
from dotenv import load_dotenv

# Agent-to-Agent Protocol (A2A)
from python_a2a import OpenAIA2AServer, run_server, A2AServer, AgentCard, AgentSkill
from python_a2a.langchain import to_langchain_agent, to_langchain_tool
from python_a2a.mcp import FastMCP

# LangChain Components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent, Tool, AgentType
from typing import List, Dict, Any, Optional, Union, Tuple
from langchain.tools import Tool
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler
import json
import traceback

# Data Processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lasio
import re

# Import enhanced MCP tools
from enhanced_mcp_tools import (
    enhanced_las_parser,
    enhanced_las_analysis,
    enhanced_las_qc,
    enhanced_formation_evaluation,
    enhanced_well_correlation,
    enhanced_well_correlation_with_qc,
    find_las_files_by_pattern,
    NumpyJSONEncoder,
    find_las_file,
    load_las_file
)
from formation_evaluation import (estimate_vshale)

# Import formation evaluation tools
from formation_evaluation import (
    evaluate_formation,
    create_formation_evaluation_summary
)

# # Add this at the top of main.py, after the imports
# import sys
# import io
# import os
#
# # Fix console encoding for Windows systems
# if sys.platform == 'win32':
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#     os.system('color')  # Enable ANSI color codes

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

# Add these imports at the top of main.py if not already present
import datetime
import csv
import os


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
    """Parse command line arguments for the application"""
    parser = argparse.ArgumentParser(description="LAS File Management Agent")

    # Server configuration
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument(
        "--host", default="localhost",
        help="Host to bind servers to (default: localhost)"
    )
    server_group.add_argument(
        "--a2a-port", type=int, default=None,
        help="Port to run the A2A server on (default: auto-select)"
    )
    server_group.add_argument(
        "--mcp-port", type=int, default=None,
        help="Port to run the MCP server on (default: auto-select)"
    )

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory containing LAS files (default: ./data)"
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model", type=str, default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)"
    )
    model_group.add_argument(
        "--temperature", type=float, default=0.0,
        help="Temperature for generation (default: 0.0)"
    )

    # Debug options
    debug_group = parser.add_argument_group("Debug Options")
    debug_group.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    debug_group.add_argument(
        "--verbose", action="store_true",
        help="Show verbose output from server processes"
    )

    # Add to parse_arguments() function
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument(
        "--log-format", type=str, default="csv",
        choices=["csv", "json", "text"],
        help="Format to save Q&A logs in (default: csv)"
    )
    logging_group.add_argument(
        "--log-dir", type=str, default="./logs",
        help="Directory to save Q&A logs in (default: ./logs)"
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

    # Find available ports for servers
    a2a_port = args.a2a_port or find_available_port(5000, 20)
    mcp_port = args.mcp_port or find_available_port(7000, 20)

    print(f" A2A server port: {a2a_port}")
    print(f" MCP server port: {mcp_port}")

    # Step 1: Create the A2A server for Well Log Analysis
    print("\nStep Step 1: Creating Well Log Analysis A2A Server")

    # Create an Agent Card for our expert agent
    agent_card = AgentCard(
        name="Well Log Analysis Expert",
        description="Specialized in interpreting well log data from LAS files",
        url=f"http://localhost:{a2a_port}",
        version="1.0.0",
        skills=[
            AgentSkill(
                name="Log Quality Control",
                description="Assessing data quality, identifying gaps, and flagging anomalies",
                examples=["Are there any quality issues with the density log?",
                          "What depth intervals have missing data?"]
            ),
            AgentSkill(
                name="Curve Interpretation",
                description="Understanding different log curve types and their geological significance",
                examples=["What does a high gamma ray reading indicate?",
                          "How do I interpret resistivity and density together?"]
            ),
            AgentSkill(
                name="Petrophysical Analysis",
                description="Basic petrophysical calculations and formation evaluation",
                examples=["How do I calculate porosity from the density log?",
                          "What cutoffs should I use to identify potential reservoirs?"]
            )
        ]
    )

    # Create the OpenAI-powered A2A server
    openai_server = OpenAIA2AServer(
        api_key=os.environ["OPENAI_API_KEY"],
        model=args.model,
        temperature=args.temperature,
        system_prompt="""You are a well log analysis expert specializing in subsurface data interpretation.
        Your expertise includes analyzing LAS files, understanding different log curves, and performing petrophysical calculations.
        Provide accurate, educational information about well log interpretation, petrophysical analysis, and formation evaluation.
        When discussing log curves, explain both their measurement principle and geological significance.
        Avoid making definitive interpretations without sufficient data, and acknowledge uncertainty where appropriate."""
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

    # Step 2: Create MCP Server with LAS Tools
    print("\nStep Step 2: Creating MCP Server with LAS Tools")

    # Create MCP server with tools
    mcp_server = FastMCP(
        name="LAS Tools",
        description="Advanced tools for LAS file processing and analysis"
    )

    # LAS Parser Tool
    @mcp_server.tool(
        name="las_parser",
        description="Parse and extract metadata from LAS files. Accepts JSON parameters or file paths."
    )
    def las_parser(file_path=None, **kwargs):
        """Parse LAS file and return metadata and curve information."""
        try:
            # Handle JSON string input
            if isinstance(file_path, str) and (file_path.startswith('{') or file_path.startswith('[')):
                try:
                    parsed_input = json.loads(file_path)
                    if isinstance(parsed_input, dict):
                        return enhanced_las_parser(**parsed_input)
                    elif isinstance(parsed_input, list):
                        # Handle list of files
                        return enhanced_las_parser(file_paths=parsed_input, data_dir=args.data_dir, **kwargs)
                except json.JSONDecodeError:
                    # Not valid JSON, continue with normal processing
                    pass

            # Handle dictionary input
            if isinstance(file_path, dict):
                return enhanced_las_parser(**file_path)

            # Normal string processing
            return enhanced_las_parser(file_path, data_dir=args.data_dir, **kwargs)
        except Exception as e:
            error_traceback = traceback.format_exc()
            return {"text": json.dumps({
                "error": f"Error processing input: {str(e)}",
                "details": error_traceback
            })}

    # LAS Analysis Tool
    @mcp_server.tool(
        name="las_analysis",
        description="Analyze curve data and perform basic calculations"
    )
    def las_analysis(file_path=None, curves=None, **kwargs):
        """Analyze specific curves from LAS file."""
        try:
            # Handle JSON string input
            if isinstance(file_path, str) and (file_path.startswith('{') or file_path.startswith('[')):
                try:
                    parsed_input = json.loads(file_path)
                    if isinstance(parsed_input, dict):
                        return enhanced_las_analysis(**parsed_input)
                    elif isinstance(parsed_input, list):
                        # Handle list of files
                        return enhanced_las_analysis(file_paths=parsed_input, data_dir=args.data_dir, **kwargs)
                except json.JSONDecodeError:
                    # Not valid JSON, continue with normal processing
                    pass

            # Handle dictionary input
            if isinstance(file_path, dict):
                return enhanced_las_analysis(**file_path)

            # Normal processing
            return enhanced_las_analysis(file_path, curves, data_dir=args.data_dir, **kwargs)
        except Exception as e:
            error_traceback = traceback.format_exc()
            return {"text": json.dumps({
                "error": f"Error processing input: {str(e)}",
                "details": error_traceback
            })}

    # LAS QC Tool
    @mcp_server.tool(
        name="las_qc",
        description="Perform quality control checks on LAS files"
    )
    def las_qc(file_path=None, **kwargs):
        """Perform quality control checks on LAS file."""
        try:
            # Handle JSON string input
            if isinstance(file_path, str) and (file_path.startswith('{') or file_path.startswith('[')):
                try:
                    parsed_input = json.loads(file_path)
                    if isinstance(parsed_input, dict):
                        return enhanced_las_qc(**parsed_input)
                    elif isinstance(parsed_input, list):
                        # Handle list of files
                        return enhanced_las_qc(file_paths=parsed_input, data_dir=args.data_dir, **kwargs)
                except json.JSONDecodeError:
                    # Not valid JSON, continue with normal processing
                    pass

            # Handle dictionary input
            if isinstance(file_path, dict):
                return enhanced_las_qc(**file_path)

            # Normal processing
            return enhanced_las_qc(file_path, data_dir=args.data_dir, **kwargs)
        except Exception as e:
            error_traceback = traceback.format_exc()
            return {"text": json.dumps({
                "error": f"Error processing input: {str(e)}",
                "details": error_traceback
            })}

    # Formation Evaluation Tool
    @mcp_server.tool(
        name="formation_evaluation",
        description="Perform petrophysical analysis and identify pay zones in LAS files"
    )
    def formation_evaluation(file_path=None, **kwargs):
        """Perform formation evaluation on LAS file."""
        try:
            # Handle JSON string input
            if isinstance(file_path, str) and (file_path.startswith('{') or file_path.startswith('[')):
                try:
                    parsed_input = json.loads(file_path)
                    if isinstance(parsed_input, dict):
                        return enhanced_formation_evaluation(**parsed_input)
                    elif isinstance(parsed_input, list):
                        # Handle list of files
                        return enhanced_formation_evaluation(file_paths=parsed_input, data_dir=args.data_dir, **kwargs)
                except json.JSONDecodeError:
                    # Not valid JSON, continue with normal processing
                    pass

            # Handle dictionary input
            if isinstance(file_path, dict):
                return enhanced_formation_evaluation(**file_path)

            # Normal processing
            return enhanced_formation_evaluation(file_path, data_dir=args.data_dir, **kwargs)
        except Exception as e:
            error_traceback = traceback.format_exc()
            return {"text": json.dumps({
                "error": f"Error processing input: {str(e)}",
                "details": error_traceback
            })}

    # Well Correlation Tool
    @mcp_server.tool(
        name="well_correlation",
        description="Correlate formations across multiple wells with enhanced algorithms"
    )
    def correlate_wells(well_list=None, marker_curve="GR", **kwargs):
        """Identify and correlate key formation tops across multiple wells."""
        try:
            # Handle JSON string input first
            if isinstance(well_list, str) and (well_list.startswith('{') or well_list.startswith('[')):
                try:
                    parsed_input = json.loads(well_list)
                    if isinstance(parsed_input, dict):
                        if 'data_dir' not in parsed_input:
                            parsed_input['data_dir'] = args.data_dir
                        # Use enhanced function
                        return enhanced_well_correlation_with_qc(**parsed_input)
                    elif isinstance(parsed_input, list):
                        return enhanced_well_correlation_with_qc(file_paths=parsed_input,
                                                                 data_dir=args.data_dir,
                                                                 llm_agent=langchain_agent, **kwargs)
                except json.JSONDecodeError:
                    pass

            if isinstance(well_list, dict):
                if 'data_dir' not in well_list:
                    well_list['data_dir'] = args.data_dir
                return enhanced_well_correlation_with_qc(**well_list)

            # Use enhanced function for normal processing
            return enhanced_well_correlation_with_qc(well_list, marker_curve, data_dir=args.data_dir,
                                                     llm_agent=langchain_agent, **kwargs)
        except Exception as e:
            error_traceback = traceback.format_exc()
            return {"text": json.dumps({
                "error": f"Error processing input: {str(e)}",
                "details": error_traceback
            })}

    @mcp_server.tool(
        name="list_files",
        description="List LAS files matching a pattern"
    )
    def list_las_files(pattern=None, data_dir="./data", **kwargs):
        """List all LAS files matching a pattern"""
        try:
            # Handle JSON string input
            if isinstance(pattern, str) and (pattern.startswith('{') or pattern.startswith('[')):
                try:
                    parsed_input = json.loads(pattern)
                    if isinstance(parsed_input, dict):
                        # Extract pattern from dict
                        if 'pattern' in parsed_input:
                            pattern = parsed_input.pop('pattern')
                        if 'data_dir' in parsed_input:
                            data_dir = parsed_input.pop('data_dir')
                        # Update kwargs with remaining items
                        kwargs.update(parsed_input)
                except json.JSONDecodeError:
                    # Not valid JSON, continue with normal processing
                    pass

            # Handle dictionary input
            if isinstance(pattern, dict):
                # Extract pattern from dict
                if 'pattern' in pattern:
                    new_pattern = pattern.pop('pattern')
                    # Update data_dir if provided
                    if 'data_dir' in pattern:
                        data_dir = pattern.pop('data_dir')
                    # Update kwargs with remaining items
                    kwargs.update(pattern)
                    pattern = new_pattern

            # Handle keyword arguments
            if 'input' in kwargs and kwargs['input'] is not None:
                pattern = kwargs['input']

            # Make sure we have a pattern
            if pattern is None:
                pattern = "*"  # Default to all files

            # If pattern doesn't end with .las, add it
            if not pattern.lower().endswith('.las'):
                pattern += ".las"

            # Find matching files
            matching_files = find_las_files_by_pattern(pattern, data_dir)

            # Return the list of matching files
            return {"text": json.dumps({
                "pattern": pattern,
                "matching_files": [os.path.basename(f) for f in matching_files],
                "full_paths": matching_files,
                "count": len(matching_files)
            }, cls=NumpyJSONEncoder)}

        except Exception as e:
            error_traceback = traceback.format_exc()
            return {"text": json.dumps({
                "error": f"Error listing LAS files: {str(e)}",
                "details": error_traceback
            }, cls=NumpyJSONEncoder)}

    @mcp_server.tool(
        name="calculate_shale_volume",
        description="Calculate volume of shale from gamma ray log using Larionov method"
    )
    def calculate_shale_volume(file_path=None, **kwargs):
        """Calculate shale volume from gamma ray log."""
        try:
            # Handle input from MCP layer - this is the key fix
            if 'input' in kwargs and kwargs['input'] is not None:
                input_data = kwargs['input']

                # Handle JSON string input
                if isinstance(input_data, str) and input_data.startswith('{'):
                    try:
                        parsed_input = json.loads(input_data)
                        file_path = parsed_input.get('file_path', file_path)
                        kwargs.update(parsed_input)
                    except json.JSONDecodeError:
                        # Not JSON, treat as file path
                        file_path = input_data
                else:
                    # Direct string input
                    file_path = input_data

            # Ensure we have a file_path
            if file_path is None:
                return {"text": json.dumps({"error": "No file path provided"})}

            print(f"DEBUG: Processing file_path: {file_path}")

            # Find the file using the existing helper function
            full_path = find_las_file(file_path, args.data_dir)
            print(f"DEBUG: Full path resolved to: {full_path}")

            if not os.path.isfile(full_path):
                return {"text": json.dumps({"error": f"File not found: {file_path}"})}

            # Load LAS file
            las, error = load_las_file(full_path)
            if error:
                return {"text": json.dumps({"error": f"Error loading LAS file: {error}"})}

            # Get gamma ray curve
            gr_curve = kwargs.get('curve', kwargs.get('curve_mnemonic', kwargs.get('gr_curve', 'GR')))
            print(f"DEBUG: Using GR curve: {gr_curve}")

            if not las.curve_exists(gr_curve):
                available_curves = las.get_curve_names()
                return {"text": json.dumps({
                    "error": f"Gamma ray curve '{gr_curve}' not found",
                    "available_curves": available_curves
                })}

            # Calculate shale volume
            gr_data = las.get_curve_data(gr_curve)
            vshale = estimate_vshale(gr_data)

            # Calculate statistics
            valid_vshale = vshale[~np.isnan(vshale)]

            if len(valid_vshale) == 0:
                return {"text": json.dumps({"error": "No valid shale volume data calculated"})}

            # Convert to percentages and calculate stats
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
                "summary": f"The average shale volume in well {las.well_info.get('WELL', 'Unknown')} is {round(avg_vshale_pct, 1)}%, ranging from {round(min_vshale_pct, 1)}% to {round(max_vshale_pct, 1)}%."
            }

            return {"text": json.dumps(result, cls=NumpyJSONEncoder)}

        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"DEBUG: Error in calculate_shale_volume: {str(e)}")
            print(f"DEBUG: Traceback: {error_traceback}")
            return {"text": json.dumps({
                "error": f"Error calculating shale volume: {str(e)}",
                "details": error_traceback
            })}

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
    print("\nStep Step 4: Converting MCP Tools to LangChain")
    try:
        las_parser_tool = to_langchain_tool(mcp_server_url, "las_parser")
        las_analysis_tool = to_langchain_tool(mcp_server_url, "las_analysis")
        las_qc_tool = to_langchain_tool(mcp_server_url, "las_qc")
        formation_eval_tool = to_langchain_tool(mcp_server_url, "formation_evaluation")
        well_correlation_tool = to_langchain_tool(mcp_server_url, "well_correlation")
        list_files_tool = to_langchain_tool(mcp_server_url, "list_files")
        calculate_shale_volume_tool = to_langchain_tool(mcp_server_url, "calculate_shale_volume")

        print("SUCCESS Successfully converted MCP tools to LangChain")

        command_processor = CommandProcessor(mcp_server_url=mcp_server_url, data_dir=args.data_dir)

        # # Create the command processor with our tools
        # command_processor = CommandProcessor(
        #     las_parser_tool=las_parser_tool,
        #     las_analysis_tool=las_analysis_tool,
        #     las_qc_tool=las_qc_tool,
        #     formation_eval_tool=formation_eval_tool,
        #     well_correlation_tool=well_correlation_tool,
        #     data_dir=args.data_dir
        # )
    except Exception as e:
        print(f"ERROR Error converting MCP tools to LangChain: {e}")
        print("\nContinuing with only the A2A agent...")
        las_parser_tool = None
        las_analysis_tool = None
        las_qc_tool = None
        formation_eval_tool = None
        well_correlation_tool = None
        list_files_tool = None
        calculate_shale_volume_tool = None
        command_processor = CommandProcessor(data_dir=args.data_dir)

    # Step 5: Create Meta-Agent
    print("\nStep Step 5: Creating LAS Management Meta-Agent")

    def create_meta_agent(
            langchain_agent: Any,
            mcp_server_url: str,
            model: str = "gpt-4",
            temperature: float = 0.0,
            verbose: bool = True,
            data_dir: str = "./data"
    ) -> Any:
        """Create a hybrid agent that combines LangChain and direct MCP access."""
        print("\nStep Creating Hybrid LAS Management Agent")

        # Create LLM for agent
        llm = ChatOpenAI(model=model, temperature=temperature)

        # Create direct MCP client for reliable tool execution
        class MCPClient:
            def __init__(self, server_url):
                self.server_url = server_url

            # In main.py - Update the MCPClient.call_tool method
            def call_tool(self, tool_name, input_data):
                """Make a direct call to an MCP tool"""
                import requests

                try:
                    # Convert input to appropriate format
                    if isinstance(input_data, dict):
                        # Dictionary input - convert to JSON string
                        input_json = json.dumps(input_data)
                    elif isinstance(input_data, str) and (input_data.startswith('{') or input_data.startswith('[')):
                        # It's already a JSON string - validate and ensure it's properly formatted
                        try:
                            parsed_data = json.loads(input_data)  # Parse to validate
                            input_json = input_data  # Use original string if valid
                        except json.JSONDecodeError:
                            # Not valid JSON, treat as regular string
                            input_json = input_data
                    else:
                        # Regular string or other type
                        input_json = input_data

                    # Make direct HTTP request to MCP server
                    response = requests.post(
                        f"{self.server_url}/tools/{tool_name}",
                        json={"input": input_json}
                    )

                    if response.status_code == 200:
                        return response.json()
                    else:
                        return {"error": f"HTTP error {response.status_code}: {response.text}"}
                except Exception as e:
                    return {"error": f"Error calling MCP tool: {str(e)}"}

        # Create MCP client
        mcp_client = MCPClient(mcp_server_url)

        # Create LangChain tools that use direct MCP calls
        tools = []

        # Add Well Log Expert tool
        def ask_well_log_expert(query: str) -> str:
            """Ask the well log expert a question."""
            try:
                result = langchain_agent.invoke(query)
                return result.get('output', 'No response from well log expert')
            except Exception as e:
                return f"Error querying well log expert: {str(e)}"

        tools.append(Tool(
            name="WellLogExpert",
            func=ask_well_log_expert,
            description="Ask the well log expert questions about log interpretation, petrophysics, etc."
        ))

        # Add LAS Parser Tool
        def parse_las_file(input_str: str) -> str:
            """Parse a LAS file and return metadata."""
            try:
                # Direct MCP call
                result = mcp_client.call_tool("las_parser", input_str)
                return result
            except Exception as e:
                return f"Error parsing LAS file: {str(e)}"

        tools.append(Tool(
            name="LASParser",
            func=parse_las_file,
            description="Parse a LAS file and extract metadata. Input can be a file path or JSON with parameters."
        ))

        # Add LAS Analysis Tool
        def analyze_las_file(input_str: str) -> str:
            """Analyze curves in a LAS file."""
            try:
                # Direct MCP call
                result = mcp_client.call_tool("las_analysis", input_str)
                return result
            except Exception as e:
                return f"Error analyzing LAS file: {str(e)}"

        tools.append(Tool(
            name="LASAnalysis",
            func=analyze_las_file,
            description="Analyze curves in a LAS file. Input can be a file path or JSON with parameters."
        ))

        # Add LAS QC Tool
        def check_las_quality(input_str: str) -> str:
            """Perform quality checks on a LAS file."""
            try:
                # Direct MCP call
                result = mcp_client.call_tool("las_qc", input_str)
                return result
            except Exception as e:
                return f"Error checking LAS file quality: {str(e)}"

        tools.append(Tool(
            name="LASQC",
            func=check_las_quality,
            description="Perform quality control checks on a LAS file. Input can be a file path or JSON with parameters."
        ))

        # Add Formation Evaluation Tool
        def evaluate_formation(input_str: str) -> str:
            """Perform petrophysical analysis on a LAS file."""
            try:
                # Direct MCP call
                result = mcp_client.call_tool("formation_evaluation", input_str)
                return result
            except Exception as e:
                return f"Error performing formation evaluation: {str(e)}"

        tools.append(Tool(
            name="FormationEvaluation",
            func=evaluate_formation,
            description="Analyze petrophysical properties and identify pay zones in a LAS file. Input can be a file path or JSON with parameters."
        ))

        # Add Well Correlation Tool
        def correlate_wells(input_str: str) -> str:
            """Correlate formations across multiple wells."""
            try:
                # Direct MCP call
                result = mcp_client.call_tool("well_correlation", input_str)
                return result
            except Exception as e:
                return f"Error correlating wells: {str(e)}"

        tools.append(Tool(
            name="WellCorrelation",
            func=correlate_wells,
            description="Identify and correlate key formation tops across multiple wells. Input can be a file path or JSON with parameters."
        ))

        # Add List Files Tool
        def list_matching_files(input_str: str) -> str:
            """List all LAS files matching a pattern."""
            try:
                # Direct MCP call
                result = mcp_client.call_tool("list_files", input_str)
                return result
            except Exception as e:
                return f"Error listing files: {str(e)}"

        tools.append(Tool(
            name="ListFiles",
            func=list_matching_files,
            description="List all LAS files matching a pattern. Input should be a file pattern that can include wildcards like * and ?."
        ))

        def calculate_shale_volume(input_str: str) -> str:
            """Calculate volume of shale from gamma ray log."""
            try:
                result = mcp_client.call_tool("calculate_shale_volume", input_str)
                return result
            except Exception as e:
                return f"Error calculating shale volume: {str(e)}"

        tools.append(Tool(
            name="CalculateShaleVolume",
            func=calculate_shale_volume,
            description="Calculate volume of shale from gamma ray log using Larionov method. Input can be a file path or JSON with parameters."
        ))

        # Create special command handling for multi-file operations
        def direct_command_processor(command_str: str) -> Optional[str]:
            """Process direct commands without using the agent"""
            try:
                # Parse all command
                if command_str.lower().startswith("parse all"):
                    pattern = extract_pattern(command_str)
                    print(f"Direct command: Parse all files matching {pattern}")
                    result = mcp_client.call_tool("las_parser", {
                        "file_path": pattern,
                        "selection_mode": "all"
                    })
                    return result

                # Evaluate all command
                elif command_str.lower().startswith(("evaluate all", "eval all")):
                    pattern = extract_pattern(command_str)
                    print(f"Direct command: Evaluate all files matching {pattern}")
                    result = mcp_client.call_tool("formation_evaluation", {
                        "file_path": pattern,
                        "selection_mode": "all"
                    })
                    return result

                # QC all command
                elif command_str.lower().startswith(("check all", "qc all")):
                    pattern = extract_pattern(command_str)
                    print(f"Direct command: QC all files matching {pattern}")
                    result = mcp_client.call_tool("las_qc", {
                        "file_path": pattern,
                        "selection_mode": "all"
                    })
                    return result

                # Correlate all command
                elif command_str.lower().startswith("correlate all"):
                    pattern = extract_pattern(command_str)
                    print(f"Direct command: Correlate all files matching {pattern}")
                    result = mcp_client.call_tool("well_correlation", {
                        "well_list": pattern,
                        "selection_mode": "all"
                    })
                    return result

                # List files command
                elif command_str.lower().startswith("list files"):
                    pattern = extract_pattern(command_str)
                    print(f"Direct command: List files matching {pattern}")
                    result = mcp_client.call_tool("list_files", pattern)
                    return result

                # No direct command matched
                return None

            except Exception as e:
                print(f"Error in direct command processor: {str(e)}")
                traceback.print_exc()
                return None

        def extract_pattern(input_str):
            """Extract file pattern from command"""
            pattern = input_str

            # Handle 'matching' keyword
            if "matching" in pattern:
                pattern = pattern.split("matching", 1)[1].strip()
            # Handle common command prefixes
            elif "all" in pattern:
                pattern = pattern.split("all", 1)[1].strip()
            elif "files" in pattern:
                pattern = pattern.split("files", 1)[1].strip()

            # Remove trailing period if present
            if pattern.endswith('.'):
                pattern = pattern[:-1]

            return pattern


        prefix = """You are an expert well log analysis assistant that works with LAS files.
    You have access to the following tools:"""

        suffix = """Begin!

    Question: {input}
    Thought: I need to analyze this request carefully to determine what tool to use.
    {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "agent_scratchpad"]
        )

        memory = ConversationBufferMemory(memory_key="chat_history")

        llm_chain = LLMChain(llm=llm, prompt=prompt)

        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=verbose)

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=verbose,
            memory=memory,
            handle_parsing_errors=True,
            max_iterations=10,  # Increase from default
            max_execution_time=300,  # 5 minutes timeout
            early_stopping_method="generate"  # Stop gracefully when approaching limits
        )

        # Return a hybrid agent that can process queries using either direct commands or the agent
        class HybridAgent:
            def __init__(self, agent_executor, command_processor):
                self.agent_executor = agent_executor
                self.command_processor = command_processor

            def run(self, query):
                """Process a query, trying direct commands first then falling back to agent"""
                # Try direct command processing first
                direct_result = self.command_processor(query)
                if direct_result:
                    return direct_result

                # Fall back to agent
                try:
                    return self.agent_executor.invoke({"input": query})["output"]
                except Exception as e:
                    print(f"Error in agent: {str(e)}")
                    # Try a fallback approach for common operations
                    if "metadata" in query.lower() and (".las" in query.lower() or "file" in query.lower()):
                        # Simple metadata request - try to extract filename
                        for word in query.split():
                            if word.lower().endswith(".las"):
                                return parse_las_file(word)
                    return f"Sorry, I encountered an error: {str(e)}"

        hybrid_agent = HybridAgent(agent_executor, direct_command_processor)

        print(f"SUCCESS Created hybrid agent with {len(tools)} tools")
        for i, tool in enumerate(tools):
            print(f"  {i + 1}. {tool.name}: {tool.description}")

        return hybrid_agent

    # Step 5: Create Meta-Agent
    print("\nStep Step 5: Creating LAS Management Meta-Agent")

    try:
        meta_agent = create_meta_agent(
            langchain_agent=langchain_agent,
            mcp_server_url=mcp_server_url,
            model=args.model,
            temperature=args.temperature,
            verbose=True,
            data_dir=args.data_dir
        )

        # Step 6: Ready for Use
        print("\nSUCCESS LAS Management Agent ready for use!")
        print("\nExample commands:")
        print("1. Parse a LAS file: What metadata is in file.las?")
        print("2. Parse multiple files: Parse all matching ./data/10543*.las")
        print("3. Analyze curves: Analyze the GR and RHOB curves in well_1.las")
        print("4. Quality check: Are there any quality issues with log_data.las?")
        print("5. Formation evaluation: Evaluate all matching ./data/10543*.las")
        print("6. Well correlation: Correlate all matching ./data/10543*.las")
        print("7. List files: List files matching *.las")
        print("8. Ask expert: What does a high neutron-density separation indicate?")

        print("\nLAS Management Agent is ready! Type your questions below (press Ctrl+C to exit):")
        try:
            while True:
                # Get user input
                user_input = input("\n> ")
                if user_input.strip().lower() in ['exit', 'quit']:
                    break

                print("\nProcessing your question...")

                # SOLUTION 3: Direct correlation handling
                if user_input.lower().startswith("correlate"):
                    try:
                        print("Using direct correlation mode...")
                        response = retry_correlation_with_fallback(meta_agent, user_input)
                        print(f"\nResponse:\n{response}")
                        save_qa_interaction(user_input, response,
                                            log_format=args.log_format,
                                            log_dir=args.log_dir)
                        continue  # Skip normal processing
                    except Exception as e:
                        print(f"Direct correlation failed, falling back to normal agent: {str(e)}")
                        # Fall through to normal processing

                try:
                    # Process the query using our hybrid agent
                    response = meta_agent.run(user_input)
                    print(f"\nResponse:\n{response}")

                    # Save the Q&A interaction
                    save_qa_interaction(user_input, response,
                                        log_format=args.log_format,
                                        log_dir=args.log_dir)
                except Exception as e:
                    error_message = f"Error processing query: {str(e)}"
                    print(error_message)
                    traceback.print_exc()

                    # Still log the error for tracking
                    save_qa_interaction(user_input, error_message)
        except KeyboardInterrupt:
            print("\nExiting...")
    except Exception as e:
        print(f"\nERROR Error creating meta-agent: {e}")
        traceback.print_exc()
        return 1

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