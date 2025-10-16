"""
Configuration file for Clustery LLM API keys.
Set your API keys here or use environment variables.
"""

import os
from typing import Optional

def load_api_keys() -> dict:
    """
    Load API keys from environment variables or config file.
    Returns a dictionary with API keys.
    """
    keys = {}
    
    # Try to load from environment variables first
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    # If not found in environment, try to load from .env file
    if not openai_key or not anthropic_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            openai_key = os.getenv("OPENAI_API_KEY")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        except ImportError:
            # dotenv not installed, that's okay
            pass
    
    keys["OPENAI_API_KEY"] = openai_key
    keys["ANTHROPIC_API_KEY"] = anthropic_key
    
    return keys

def get_openai_key() -> Optional[str]:
    """Get OpenAI API key."""
    return load_api_keys().get("OPENAI_API_KEY")

def get_anthropic_key() -> Optional[str]:
    """Get Anthropic API key."""
    return load_api_keys().get("ANTHROPIC_API_KEY")

# Example usage:
if __name__ == "__main__":
    keys = load_api_keys()
    print("API Keys Status:")
    print(f"OpenAI: {'✅ Set' if keys['OPENAI_API_KEY'] else '❌ Not set'}")
    print(f"Anthropic: {'✅ Set' if keys['ANTHROPIC_API_KEY'] else '❌ Not set'}")
