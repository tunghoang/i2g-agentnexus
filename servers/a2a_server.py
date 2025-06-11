"""
A2A Server Management
Handles Agent-to-Agent server lifecycle and configuration
"""

import os
import time
from typing import Optional

from python_a2a import OpenAIA2AServer, run_server, A2AServer, AgentCard, AgentSkill

from .base_server import BaseServer, HealthCheckMixin
from config.settings import A2AConfig, DataConfig


class A2AServerManager(BaseServer, HealthCheckMixin):
    """
    Manages A2A (Agent-to-Agent) server lifecycle

    Responsibilities:
    - Create OpenAI-powered A2A server
    - Configure expert agent capabilities
    - Manage server lifecycle
    - Health monitoring
    """

    def __init__(self, config: A2AConfig, data_config: DataConfig):
        super().__init__("A2A", config.host, config.port)
        self.config = config
        self.data_config = data_config
        self.openai_server: Optional[OpenAIA2AServer] = None
        self.wrapped_server: Optional[A2AServer] = None

    def _create_server(self):
        """Create A2A server with subsurface data expertise"""
        self.logger.info("Creating A2A server with subsurface expertise...")

        # Verify API key
        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("OPENAI_API_KEY not found in environment")

        # Create agent card with comprehensive capabilities
        agent_card = self._create_agent_card()

        # Create OpenAI-powered server with only valid parameters
        # Removed 'max_tokens' as it's not a valid constructor parameter
        try:
            self.openai_server = OpenAIA2AServer(
                api_key=os.environ["OPENAI_API_KEY"],
                model=self.config.model,
                temperature=self.config.temperature,
                # max_tokens=self.config.max_tokens,  # REMOVED - This parameter is invalid
                system_prompt=self._create_system_prompt()
            )
        except TypeError as e:
            self.logger.error(f"Error creating OpenAI A2A server: {e}")
            # Try with even fewer parameters
            try:
                self.logger.info("Trying with minimal parameters...")
                self.openai_server = OpenAIA2AServer(
                    api_key=os.environ["OPENAI_API_KEY"],
                    model=self.config.model,
                    system_prompt=self._create_system_prompt()
                )
            except TypeError as e2:
                self.logger.error(f"Minimal parameter creation also failed: {e2}")
                # Try with just required parameters
                try:
                    self.logger.info("Trying with just API key and model...")
                    self.openai_server = OpenAIA2AServer(
                        api_key=os.environ["OPENAI_API_KEY"],
                        model=self.config.model
                    )
                except Exception as e3:
                    self.logger.error(f"Even basic creation failed: {e3}")
                    raise e  # Re-raise the original error

        # Set agent card
        self.openai_server.agent_card = agent_card

        # Wrap in standard A2A server
        self.wrapped_server = SubsurfaceExpertServer(self.openai_server, agent_card)

        self.logger.info("A2A server created successfully")

    def _create_agent_card(self) -> AgentCard:
        """Create comprehensive agent card for subsurface data expert"""

        skills = [
            # Well Log Analysis Skills
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
                name="Intelligent Seismic Survey Classification",
                description="AI-powered automatic classification of seismic survey characteristics with confidence scoring",
                examples=[
                    "Automatically classify survey as 2D/3D, PreStack/PostStack",
                    "Determine optimal processing parameters based on survey type",
                    "Identify survey sorting methods and recommend templates",
                    "Provide confidence-based processing recommendations"
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

            # Integration and Production Skills
            AgentSkill(
                name="Integrated Subsurface Analysis",
                description="Combine well log and seismic data for comprehensive subsurface characterization",
                examples=[
                    "Integrate well logs with seismic data for reservoir characterization",
                    "Perform well-to-seismic calibration and tie analysis",
                    "Generate integrated subsurface models with uncertainty quantification",
                    "Provide geological interpretation combining multiple data types"
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

        return AgentCard(
            name="Subsurface Data Management Expert",
            description="Production-grade expert in well log and seismic data analysis with AI-powered automation capabilities",
            url=self.url,
            version="2.0.0",
            skills=skills
        )

    def _create_system_prompt(self) -> str:
        """Create comprehensive system prompt for the expert agent"""

        return """You are a production-grade subsurface data analysis expert specializing in both well log and seismic data interpretation with advanced AI automation capabilities.

CORE CAPABILITIES:

WELL LOG ANALYSIS (LAS Files):
- Robust parsing with comprehensive error recovery and validation
- Advanced petrophysical calculations and formation evaluation
- Quality control with actionable recommendations
- Multi-well correlation and pay zone identification
- Statistical analysis and curve characterization

INTELLIGENT SEG-Y ANALYSIS (AI-Powered):
- **Automatic Survey Classification**: AI-powered determination of 2D/3D, PreStack/PostStack, and sorting methods
- **Intelligent Template Detection**: Automatically finds optimal processing templates, eliminating manual selection
- **Batch Survey Analysis**: Processes multiple files with consistency analysis and progress reporting  
- **Survey Comparison**: Compares characteristics across files for processing compatibility
- Production-quality parsing with comprehensive validation and error recovery

INTEGRATED WORKFLOWS:
- Well-to-seismic calibration and correlation analysis
- Integrated subsurface interpretation combining both data types
- AI-powered geological interpretation with confidence scoring
- Formation correlation using both wells and seismic data
- Comprehensive quality assurance across all data types

INTELLIGENT AUTOMATION:
- Eliminates manual template selection through AI classification
- Provides confidence-based processing recommendations
- Automatically optimizes processing parameters based on survey characteristics
- Handles problematic files with intelligent error recovery and user guidance

PRODUCTION FEATURES:
- Comprehensive error handling with detailed recovery suggestions
- Real-time progress reporting for long operations
- Memory management and performance optimization for large datasets
- Detailed validation and quality control with actionable recommendations
- Robust template and configuration management
- System health monitoring and performance tracking

You provide reliable, production-quality analysis with comprehensive error handling, progress tracking, and detailed validation. Your intelligent capabilities solve common processing bottlenecks automatically while maintaining enterprise-grade reliability.

When discussing data integration, explain the complementary nature of well logs and seismic data in subsurface characterization. Always provide clear, actionable recommendations based on your analysis with appropriate confidence levels and uncertainty quantification."""

    def _start_server(self):
        """Start the A2A server in background thread"""
        if not self.wrapped_server:
            raise RuntimeError("Server not created")

        def run_a2a_server():
            run_server(self.wrapped_server, host=self.host, port=self.port)

        self.run_in_thread(run_a2a_server)

        # Give server time to start
        time.sleep(2)

    def _stop_server(self):
        """Stop the A2A server"""
        # A2A server stops when thread ends
        self._running = False

    def _check_health(self) -> bool:
        """Check A2A server health"""
        if not self.is_running():
            return False

        try:
            import requests
            # Try to access the agent card endpoint
            response = requests.get(f"{self.url}/agent", timeout=5)
            return response.status_code == 200
        except Exception:
            # Fallback to thread status
            return self.is_running()


class SubsurfaceExpertServer(A2AServer):
    """
    Wrapper class for OpenAI A2A server
    Implements the A2AServer interface
    """

    def __init__(self, openai_server: OpenAIA2AServer, agent_card: AgentCard):
        super().__init__(agent_card=agent_card)
        self.openai_server = openai_server

    def handle_message(self, message):
        """Handle incoming messages by delegating to OpenAI server"""
        return self.openai_server.handle_message(message)


if __name__ == "__main__":
    # Test A2A server manager
    from config.settings import A2AConfig, DataConfig

    a2a_config = A2AConfig(port=5001)
    data_config = DataConfig()

    server = A2AServerManager(a2a_config, data_config)

    print(f"Server status: {server.get_status()}")

    if os.getenv("OPENAI_API_KEY"):
        try:
            server.start()
            print("Server started successfully")

            if server.wait_ready(timeout=10):
                print("Server is ready!")
            else:
                print("Server not ready")

            time.sleep(2)
            server.stop()
            print("Server stopped")

        except Exception as e:
            print(f"Error: {e}")
    else:
        print("OPENAI_API_KEY not set - skipping server test")