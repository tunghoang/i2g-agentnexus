"""
Interactive Shell
Clean, user-friendly command-line interface for the platform
"""

import time
import logging
import datetime
import csv
import json
from pathlib import Path
from typing import Optional

from config.settings import Config
from agents.meta_agent import MetaAgent


def save_qa_interaction(question: str, answer: str, log_format: str = "csv", log_dir: str = "./logs"):
    """
    Save a question and answer interaction to a log file

    Args:
        question: The user's question
        answer: The system's answer
        log_format: Format to save in ('csv', 'json', or 'text')
        log_dir: Directory to save logs in

    Returns:
        str: Path to the saved log file
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_only = datetime.datetime.now().strftime("%Y-%m-%d")

    if log_format.lower() == "csv":
        # Use a daily CSV file
        log_file = log_path / f"qa_log_{date_only}.csv"
        file_exists = log_file.exists()

        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header if new file
            if not file_exists:
                writer.writerow(["Timestamp", "Question", "Answer"])
            writer.writerow([timestamp, question, answer])

    elif log_format.lower() == "json":
        # Use a daily JSON file with an array of interactions
        log_file = log_path / f"qa_log_{date_only}.json"

        # Load existing data if file exists
        if log_file.exists():
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
        log_file = log_path / f"qa_log_{date_only}.txt"

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"=== {timestamp} ===\n")
            f.write(f"Question: {question}\n\n")
            f.write(f"Answer: {answer}\n\n")
            f.write("-" * 80 + "\n\n")

    return str(log_file)


def clean_response(response: str) -> str:
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


class InteractiveShell:
    """
    Interactive command-line shell for the Subsurface Data Platform

    Features:
    - Clean, intuitive interface
    - Command history and shortcuts
    - Error handling with helpful messages
    - Automatic logging of interactions
    - Platform status and help commands
    """

    def __init__(self, meta_agent: MetaAgent, config: Config, platform=None):
        self.meta_agent = meta_agent
        self.config = config
        self.platform = platform
        self.logger = logging.getLogger(__name__)

        # Shell state
        self.session_start = time.time()
        self.query_count = 0

    def run(self):
        """Run the interactive shell"""
        self._print_welcome()

        while True:
            try:
                # Get user input with timeout handling
                user_input = self._get_user_input()

                if not user_input:
                    self._handle_empty_input()
                    continue

                # Handle shell commands
                if self._handle_shell_command(user_input):
                    continue

                # Handle exit commands
                if user_input.lower() in ['exit', 'quit', 'q', 'bye']:
                    self._print_goodbye()
                    break

                # Process query
                self._process_query(user_input)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye!")
                break
            except EOFError:
                print("\n\n📝 Input stream ended. Goodbye!")
                break
            except Exception as e:
                self.logger.error(f"Shell error: {e}")
                print(f"Shell error: {str(e)}")

    def _print_welcome(self):
        """Print welcome message"""
        print("\n" + "=" * 80)
        print("SUBSURFACE DATA MANAGEMENT PLATFORM v2.0")
        print("=" * 80)
        print()
        print("💡 Quick Start:")
        print("   • 'list files' - See available data")
        print("   • 'system status' - Check platform health")
        print("   • 'help' - Get detailed assistance")
        print("   • 'quit' - Exit the platform")
        print()
        print("📚 Examples:")
        print("   • 'analyze formation in well_1054310_15.las'")
        print("   • 'classify survey F3_seismic.sgy'")
        print("   • 'correlate all wells matching *.las'")
        print("=" * 80)

    def _get_user_input(self) -> str:
        """Get user input with clean prompt"""
        try:
            return input("\nAsk me anything (or 'quit' to exit)\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as e:
            self.logger.warning(f"Input error: {e}")
            return ""

    def _handle_empty_input(self):
        """Handle empty input gracefully"""
        print("💡 Empty input detected. Try asking about:")
        print("   • System status")
        print("   • List files in data directory")
        print("   • Analyze a specific file")

    def _handle_shell_command(self, user_input: str) -> bool:
        """
        Handle special shell commands

        Returns:
            True if command was handled, False otherwise
        """
        command = user_input.lower().strip()

        # Help command
        if command in ['help', '?', 'h']:
            self._print_help()
            return True

        # Status commands
        elif command in ['status', 'health']:
            self._print_status()
            return True

        # Stats command
        elif command in ['stats', 'statistics']:
            self._print_stats()
            return True

        # Clear command
        elif command in ['clear', 'cls']:
            import os
            os.system('clear' if os.name == 'posix' else 'cls')
            return True

        # Version command
        elif command in ['version', 'ver']:
            self._print_version()
            return True

        return False

    def _print_help(self):
        """Print detailed help information"""
        print("\n" + "=" * 60)
        print("📚 SUBSURFACE DATA PLATFORM - HELP")
        print("=" * 60)
        print()
        print("SHELL COMMANDS:")
        print("   help, ?          - Show this help")
        print("   status, health   - Platform status")
        print("   stats           - Session statistics")
        print("   clear, cls      - Clear screen")
        print("   version         - Platform version")
        print("   quit, exit      - Exit platform")
        print()
        print("WELL LOG ANALYSIS (LAS Files):")
        print("   • 'parse well_data.las'")
        print("   • 'analyze formation in *.las'")
        print("   • 'check quality of all wells'")
        print("   • 'calculate shale volume for well_123.las'")
        print("   • 'correlate formations across *.las'")
        print()
        print("SEISMIC DATA ANALYSIS (SEG-Y Files):")
        print("   • 'classify survey_data.sgy'")
        print("   • 'analyze seismic quality *.segy'")
        print("   • 'extract metadata from F3_seismic.sgy'")
        print("   • 'quick summary of all *.sgy files'")
        print("   • 'extract survey polygon from seismic.sgy'")
        print()
        print("SYSTEM COMMANDS:")
        print("   • 'list files' or 'list *.las'")
        print("   • 'system status'")
        print("   • 'health check'")
        print("   • 'directory info'")
        print()
        print("TIPS:")
        print("   • Use wildcards: *.las, *.segy")
        print("   • Be specific: 'analyze formation' vs 'analyze'")
        print("   • Chain operations: 'parse and analyze well.las'")
        print("=" * 60)

    def _print_status(self):
        """Print platform status"""
        if self.platform:
            status = self.platform.get_status()
            print("\nPLATFORM STATUS:")
            print(f"   • Initialized: {'✅' if status['platform']['initialized'] else '❌'}")
            print(f"   • Uptime: {status['platform']['uptime_hours']:.1f} hours")
            print(f"   • A2A Server: {'✅' if status['servers']['a2a']['ready'] else '❌'}")
            print(f"   • MCP Server: {'✅' if status['servers']['mcp']['ready'] else '❌'}")
            print(f"   • Agents: {'✅' if status['agents']['meta_agent'] else '❌'}")
            print(f"   • Monitoring: {'✅' if status['monitoring']['enabled'] else '❌'}")
        else:
            print("\nAGENT STATUS:")
            try:
                agent_stats = self.meta_agent.get_stats()
                print(f"   • Uptime: {agent_stats.get('uptime_hours', 0):.1f} hours")
                print(f"   • Queries: {agent_stats.get('total_queries', 0)}")
                print(f"   • System: {agent_stats.get('system_type', 'Unknown')}")
            except Exception as e:
                print(f"   • Error getting status: {e}")

    def _print_stats(self):
        """Print session statistics"""
        try:
            agent_stats = self.meta_agent.get_stats()
            session_time = time.time() - self.session_start

            print("\n📈 SESSION STATISTICS:")
            print(f"   • Session time: {session_time / 3600:.1f} hours")
            print(f"   • Queries this session: {self.query_count}")
            print(f"   • Total agent queries: {agent_stats.get('total_queries', 0)}")
            print(f"   • Direct commands: {agent_stats.get('direct_commands', 0)}")
            print(f"   • Agent responses: {agent_stats.get('agent_responses', 0)}")
            print(f"   • Errors: {agent_stats.get('errors', 0)}")

            # Success rate
            total = agent_stats.get('total_queries', 0)
            errors = agent_stats.get('errors', 0)
            if total > 0:
                success_rate = ((total - errors) / total) * 100
                print(f"   • Success rate: {success_rate:.1f}%")

        except Exception as e:
            print(f"   • Error getting statistics: {e}")

    def _print_version(self):
        """Print version information"""
        print("\nSUBSURFACE DATA PLATFORM")
        print("   • Version: 2.0.0 (Complete Rewrite)")
        print("   • Architecture: Clean Modular Design")
        print("   • Agent: Meta-Agent with HybridAgent")
        print("   • AI Model: GPT-4o")
        print("   • Features: LAS + SEG-Y Analysis")

    def _process_query(self, user_input: str):
        """Process a user query"""
        self.query_count += 1

        print(f"\nProcessing: {user_input}")
        print("Please wait...")

        try:
            # Add rate limiting - wait 1 second before processing
            time.sleep(1)

            # Process the query
            response = self.meta_agent.run(user_input)
            cleaned_response = clean_response(response)

            print(f"\nResponse:\n{cleaned_response}")

            # Save the Q&A interaction
            try:
                log_file = save_qa_interaction(
                    user_input,
                    response,
                    log_format=self.config.logging.format,
                    log_dir=self.config.logging.directory
                )
                self.logger.debug(f"Interaction saved to {log_file}")
            except Exception as log_error:
                self.logger.warning(f"Failed to save interaction: {log_error}")

        except Exception as e:
            error_msg = str(e)

            if "rate limit" in error_msg.lower() or "429" in error_msg:
                print("Rate limit reached. Please wait 30 seconds before trying again.")
                time.sleep(30)
            elif "recursion" in error_msg.lower():
                print("Processing too complex. Try a simpler request or a smaller file.")
            else:
                print(f"Error: {error_msg}")

            # Still log the error for tracking
            try:
                save_qa_interaction(
                    user_input,
                    f"Error: {error_msg}",
                    log_format=self.config.logging.format,
                    log_dir=self.config.logging.directory
                )
            except Exception:
                pass  # Don't let logging errors break the shell

    def _print_goodbye(self):
        """Print goodbye message"""
        session_time = time.time() - self.session_start

        print("\n" + "=" * 60)
        print("GOODBYE!")
        print("=" * 60)
        print(f"Session Summary:")
        print(f"   • Duration: {session_time / 60:.1f} minutes")
        print(f"   • Queries processed: {self.query_count}")
        print(f"   • Logs saved to: {self.config.logging.directory}")
        print()
        print("✨ Thank you for using Subsurface Data Platform!")
        print("Your data analysis session has been saved.")
        print("=" * 60)


if __name__ == "__main__":
    # Test interactive shell
    from config.settings import load_config
    from agents.meta_agent import MetaAgent


    # Mock meta agent for testing
    class MockMetaAgent:
        def run(self, query):
            return f"Mock response to: {query}"

        def get_stats(self):
            return {
                "total_queries": 5,
                "uptime_hours": 1.5,
                "direct_commands": 2,
                "agent_responses": 3,
                "errors": 0
            }


    config = load_config()
    mock_meta = MockMetaAgent()

    shell = InteractiveShell(mock_meta, config)
    print("Interactive shell created - run shell.run() to start")