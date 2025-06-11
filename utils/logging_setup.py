"""
Logging setup utilities
Centralized logging configuration for the platform
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import LoggingConfig


def setup_logging(config: LoggingConfig, logger_name: Optional[str] = None):
    """
    Setup logging configuration

    Args:
        config: Logging configuration
        logger_name: Optional specific logger name
    """

    # Ensure log directory exists
    log_dir = Path(config.directory)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set logging level
    level = getattr(logging, config.level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get logger
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()

    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler
    log_file = log_dir / "platform.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # Set third-party library log levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    logger.info(f"Logging configured - Level: {config.level}, Directory: {config.directory}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, e: Exception, context: str = ""):
    """Log exception with context"""
    import traceback

    error_msg = f"Exception in {context}: {str(e)}" if context else f"Exception: {str(e)}"
    logger.error(error_msg)
    logger.debug(traceback.format_exc())


class LoggerMixin:
    """Mixin class to add logging to any class"""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return logging.getLogger(self.__class__.__name__)


if __name__ == "__main__":
    # Test logging setup
    from config.settings import LoggingConfig

    config = LoggingConfig(level="DEBUG", directory="./test_logs")
    setup_logging(config)

    logger = get_logger(__name__)
    logger.info("Test log message")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")