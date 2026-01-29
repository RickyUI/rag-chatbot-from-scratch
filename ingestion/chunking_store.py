# Import necessary libraries
import pandas as pd
from preprocess import preprocessed_ai_df
from uuid import uuid4
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv(dotenv_path=Path('.') / '.env')

# Load environment variables from .env file
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment variables.")

client = OpenAI(api_key=api_key)

batch_limit = 100

# Convert all column names to lowercase for consistency
preprocessed_ai_df.columns = preprocessed_ai_df.columns.str.lower()

for batch in np.array_split(preprocessed_ai_df, len(preprocessed_ai_df) / batch_limit):
    metadatas = [{"answer": row["answer"]} for _, row in batch.iterrows()]
    texts = batch["question"].tolist()
    ids = [str(uuid4()) for _ in range(len(texts))]

    # Generate embeddings using OpenAI API
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )

    # Extract embeddings from the response
    embeds = [data.embedding for data in response.data]
