# Import necessary libraries
from dotenv import load_dotenv
import os
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv(dotenv_path=Path('.') / '.env')

api_key = os.getenv("PINECONE_API_KEY") # Pinecone API key

if not api_key:
    raise RuntimeError("PINECONE_API_KEY not found in environment variables.")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Creating the index in Pinecone
pc.create_index(
    name="ai-qa-embeddings",
    dimension=1536,  # Dimension of the embeddings
    spec =ServerlessSpec(cloud="aws", region="us-east-1")
)

print("Pinecone index 'ai-qa-embeddings' created successfully.")