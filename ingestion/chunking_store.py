# Import necessary libraries
import pandas as pd
from preprocess import preprocessed_ai_df
from uuid import uuid4
import numpy as np
from config.clients import get_openai_client, get_pinecone_index
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
openai_client = get_openai_client()
index_name = "ai-qa-embeddings"

batch_limit = 100

# Convert all column names to lowercase for consistency
def main():
    preprocessed_ai_df.columns = preprocessed_ai_df.columns.str.lower()

    for batch in np.array_split(preprocessed_ai_df, len(preprocessed_ai_df) / batch_limit):
        metadatas = [{"answer": row["answer"]} for _, row in batch.iterrows()]
        texts = batch["question"].tolist()
        ids = [str(uuid4()) for _ in range(len(texts))]

        # Generate embeddings using OpenAI API
        response = openai_client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )

        # Extract embeddings from the response
        embeds = [data.embedding for data in response.data]

        # Upserting embeddings into Pinecone
        index = get_pinecone_index(index_name)
        index.upsert(vectors=zip(ids, embeds, metadatas), namespace="ai-qa")

    # Print confirmation message
    logger.info("Data upserted to Pinecone index successfully.")

if __name__ == "__main__":
    main()