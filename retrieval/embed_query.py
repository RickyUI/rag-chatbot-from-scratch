# Importing necessary libraries
from config.clients import get_openai_client, get_pinecone_index
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
openai_client = get_openai_client()
index = get_pinecone_index()

def retrieve(query: str, top_k: int = 5):
    """
    Embed a query and retrieve similar documents from Pinecone.
    
    Args:
        query (str): The query string to search for.
        top_k (int): Number of top results to return.
    
    Returns:
        dict: Query results from Pinecone index.
    """
    try:
        # Generate embedding for the query using OpenAI
        response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        
        query_embedding = response.data[0].embedding
        
        # Query the Pinecone index
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="ai-qa"
        )
        
        logger.info(f"Retrieved {len(results.matches)} results for query: {query}")
        
        retrieved_docs = []

        for doc in results['matches']:
            retrieved_docs.append(doc['metadata']['answer'])
        return retrieved_docs
    
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise

if __name__ == "__main__":
    # Test local
    test_results = retrieve("What does the deep learning model")
    print(test_results)
