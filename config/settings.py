"""
Configuration management for Subsurface Data Platform
Centralized, type-safe configuration with environment variable support
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class A2AConfig:
    """A2A Server Configuration"""
    port: int = 5000
    host: str = "localhost"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: int = 30

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class MCPConfig:
    """MCP Server Configuration"""
    port: int = 7000
    host: str = "localhost"
    timeout: int = 30

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class AgentConfig:
    """Agent Configuration"""
    max_iterations: int = 15
    max_execution_time: int = 300
    # early_stopping_method: str = "generate"
    handle_parsing_errors: bool = True
    verbose: bool = True
    temperature: float = 0.0


@dataclass
class MonitoringConfig:
    """Production Monitoring Configuration"""
    enabled: bool = True
    health_check_interval: int = 300  # seconds
    metrics_retention_days: int = 30
    performance_tracking: bool = True
    dashboard_enabled: bool = True


@dataclass
class LoggingConfig:
    """Logging Configuration"""
    level: str = "INFO"
    format: str = "csv"  # csv, json, text
    directory: str = "./logs"

    def __post_init__(self):
        # Ensure log directory exists
        Path(self.directory).mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data Directory Configuration"""
    data_dir: str = "./data"
    file_extensions: List[str] = field(default_factory=lambda: [".las", ".LAS", ".sgy", ".segy", ".SGY", ".SEGY"])
    max_files_batch: int = 50

    def __post_init__(self):
        # Ensure data directory exists
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class AnalysisConfig:
    """Analysis Configuration"""
    default_curves: List[str] = field(default_factory=lambda: ["GR", "RHOB", "NPHI", "RT", "SP"])
    correlation_tolerance: float = 5.0
    formation_water_resistivity: float = 0.1
    quality_threshold: str = "Good"  # Excellent, Good, Fair, Poor


@dataclass
class Config:
    """Main Configuration Class"""
    # Core components
    a2a: A2AConfig = field(default_factory=A2AConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    # Feature flags
    debug: bool = False
    dry_run: bool = False
    enable_segy_intelligent: bool = True
    parallel_processing: bool = False


def load_config() -> Config:
    """
    Load configuration from environment variables and defaults

    Environment Variables:
    - DATA_DIR: Data directory path
    - A2A_PORT: A2A server port
    - MCP_PORT: MCP server port
    - OPENAI_MODEL: OpenAI model name
    - LOG_LEVEL: Logging level
    - DEBUG: Enable debug mode
    """

    # Load from environment with fallbacks
    config = Config(
        # Data configuration
        data=DataConfig(
            data_dir=os.getenv("DATA_DIR", "./data")
        ),

        # A2A configuration
        a2a=A2AConfig(
            port=int(os.getenv("A2A_PORT", "5000")),
            host=os.getenv("A2A_HOST", "localhost"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None
        ),

        # MCP configuration
        mcp=MCPConfig(
            port=int(os.getenv("MCP_PORT", "7000")),
            host=os.getenv("MCP_HOST", "localhost")
        ),

        # Agent configuration
        agent=AgentConfig(
            max_iterations=int(os.getenv("MAX_ITERATIONS", "15")),
            max_execution_time=int(os.getenv("MAX_EXECUTION_TIME", "300")),
            verbose=os.getenv("VERBOSE", "true").lower() == "true"
        ),

        # Monitoring configuration
        monitoring=MonitoringConfig(
            enabled=os.getenv("MONITORING_ENABLED", "true").lower() == "true",
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "300"))
        ),

        # Logging configuration
        logging=LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "csv"),
            directory=os.getenv("LOG_DIR", "./logs")
        ),

        # Feature flags
        debug=os.getenv("DEBUG", "false").lower() == "true",
        dry_run=os.getenv("DRY_RUN", "false").lower() == "true",
        parallel_processing=os.getenv("PARALLEL_PROCESSING", "false").lower() == "true"
    )

    return config


def load_config_from_file(config_file: str) -> Config:
    """Load configuration from JSON/YAML file (future enhancement)"""
    # TODO: Implement file-based configuration loading
    raise NotImplementedError("File-based configuration loading not yet implemented")


# Configuration validation
def validate_config(config: Config) -> bool:
    """Validate configuration settings"""

    # Check required directories
    if not Path(config.data.data_dir).exists():
        print(f"Warning: Data directory does not exist: {config.data.data_dir}")
        return False

    # Check port availability (basic check)
    if config.a2a.port == config.mcp.port:
        print(f"Error: A2A and MCP servers cannot use the same port: {config.a2a.port}")
        return False

    # Check model name
    valid_models = ["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    if config.a2a.model not in valid_models:
        print(f"Warning: Unknown model: {config.a2a.model}")

    return True


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully:")
    print(f"Data directory: {config.data.data_dir}")
    print(f"A2A Server: {config.a2a.url}")
    print(f"MCP Server: {config.mcp.url}")
    print(f"Debug mode: {config.debug}")

    # Validate configuration
    if validate_config(config):
        print("Configuration is valid")
    else:
        print("Configuration has issues")