"""
Centralized module for initializing and retrieving API clients.
"""
from dotenv import load_dotenv
import os
from pathlib import Path
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API keys
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not found in environment variables.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment variables.")

# Client instances (singleton pattern)
_pinecone_client = None
_openai_client = None

def get_pinecone_client() -> Pinecone:
    """Returns a Pinecone client instance."""
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    return _pinecone_client

def get_openai_client() -> OpenAI:
    """Returns an OpenAI client instance."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

def get_pinecone_index(index_name: str = "ai-qa-embeddings"):
    """Returns a Pinecone index instance."""
    client = get_pinecone_client()
    return client.Index(index_name)
