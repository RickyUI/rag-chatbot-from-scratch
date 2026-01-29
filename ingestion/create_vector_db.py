# Import necessary libraries
from pinecone import ServerlessSpec
from config.clients import get_pinecone_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone client
pc = get_pinecone_client()

# Index configuration
INDEX_NAME = "ai-qa-embeddings"
INDEX_DIMENSION = 1536
INDEX_METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"

# Check if index already exists and create if needed
def main():
    existing_indexes = pc.list_indexes()
    if INDEX_NAME in [idx.name for idx in existing_indexes]:
        logger.info(f"Index '{INDEX_NAME}' already exists.")
    else:
        logger.info(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=INDEX_DIMENSION,
            metric=INDEX_METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION)
        )
        logger.info(f"Index '{INDEX_NAME}' created successfully.")

if __name__ == "__main__":
    main()