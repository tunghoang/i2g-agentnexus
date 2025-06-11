# utils/api_key_checker.py
"""
API Key validation utilities
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv


def check_api_key() -> bool:
    """
    Check if OpenAI API key is available

    Returns:
        True if API key is found, False otherwise
    """
    logger = logging.getLogger(__name__)

    # First check if API key is already in environment
    if "OPENAI_API_KEY" in os.environ:
        logger.info("OpenAI API key found in environment variables")
        return True

    # If not in environment, try to load from .env file
    logger.info(" API key not found in environment, checking .env file...")

    # Look for .env file in the current directory and parent directory
    env_paths = [Path(".env"), Path("../.env")]
    for env_path in env_paths:
        if env_path.exists():
            logger.info(f"📁 Found .env file at {env_path.resolve()}")
            load_dotenv(env_path)
            if "OPENAI_API_KEY" in os.environ:
                logger.info("Successfully loaded OpenAI API key from .env file")
                return True

    # If still not found, provide helpful instructions
    logger.error("OPENAI_API_KEY not found in environment variables or .env file")
    print("\n" + "=" * 60)
    print("OpenAI API Key Required")
    print("=" * 60)
    print("You have the following options:")
    print("1. Set the environment variable:")
    print("   export OPENAI_API_KEY=your_api_key")
    print("2. Create a .env file in the project directory:")
    print("   echo 'OPENAI_API_KEY=your_api_key' > .env")
    print("3. Set it for this session:")
    print("   OPENAI_API_KEY=your_api_key python main.py")
    print("=" * 60)

    return False