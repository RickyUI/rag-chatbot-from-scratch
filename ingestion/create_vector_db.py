# Import necessary libraries
from pinecone import PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv(dotenv_path=Path('.') / '.env')

api_key = os.getenv("PINECONE_API_KEY") # Pinecone API key

if not api_key:
    raise RuntimeError("PINECONE_API_KEY not found in environment variables.")

# Initialize Pinecone client
pc = PineconeClient(api_key=api_key)

# Creating the index in Pinecone
pc.create_index(
    name="ai-qa-embeddings",
    dimension=1536,  # Dimension of the embeddings
    spec =ServerlessSpec(cloud="aws", region="us-east-1")
)