#!/usr/bin/env python3
"""
Test script to verify KG-Builder setup
"""

import os
import sys
from dotenv import load_dotenv
from loguru import logger

# Configure loguru logging for test
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)
logger.add(
    "logs/test-setup.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="5 MB",
    retention="3 days"
)

# Create logs directory if it doesn't exist
from pathlib import Path
Path("logs").mkdir(exist_ok=True)

def test_environment():
    """Test environment setup"""
    logger.info("Testing KG-Builder setup...")
    print("Testing KG-Builder setup...")
    
    # Load environment
    logger.debug("Loading environment variables...")
    load_dotenv()
    
    # Check API key
    logger.debug("Checking for OPENROUTER_API_KEY...")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found")
        print("❌ ERROR: OPENROUTER_API_KEY not found")
        print("Please create a .env file with your OpenRouter API key")
        return False
    else:
        logger.success("OPENROUTER_API_KEY found")
        print("✅ OPENROUTER_API_KEY found")
    
    # Test imports
    logger.debug("Testing kg-gen import...")
    try:
        from kg_gen import KGGen
        logger.success("kg-gen library imported successfully")
        print("✅ kg-gen library imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import kg-gen: {e}")
        print(f"❌ ERROR importing kg-gen: {e}")
        print("Please install requirements: uv add -r requirements.txt")
        return False
    
    logger.debug("Testing litellm import...")
    try:
        import litellm
        logger.success("litellm library imported successfully")
        print("✅ litellm library imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import litellm: {e}")
        print(f"❌ ERROR importing litellm: {e}")
        return False
    
    logger.success("All tests passed! KG-Builder is ready to use.")
    print("\n✅ All tests passed! KG-Builder is ready to use.")
    print("\nNext steps:")
    print("1. Copy env.sample to .env and add your API key")
    print("2. Run: python main.py --input input/sample-text.txt")
    
    return True

if __name__ == "__main__":
    test_environment()
