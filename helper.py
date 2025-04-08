# help other files to access api keys via dotenv
import os
import dotenv
from dotenv import load_dotenv
# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
# Get the API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Add more API keys as needed
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def get_api_key(key_name):
    """
    Get a specific API key by name
    
    Args:
        key_name (str): Name of the environment variable
        
    Returns:
        str: API key value or None if not found
    """
    return os.getenv(key_name)

def validate_api_keys():
    """
    Validate that required API keys exist
    
    Returns:
        dict: Dictionary with API key names and their status (True if exists)
    """
    keys = {
        'OPENAI_API_KEY': bool(OPENAI_API_KEY),
        'GEMINI_API_KEY': bool(GEMINI_API_KEY),
    }
    return keys

