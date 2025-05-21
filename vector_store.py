import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import json
import os
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Load environment variables
load_dotenv()

class VectorStore:
    def __init__(self, collection_name: str = "drawstrings_cases"):
        """Initialize the vector store with ChromaDB using OpenAI embeddings."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embedding_function = OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-ada-002"
        )
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_cases(self, cases: List[Dict]):
        """Add new cases to the vector store.
        
        Args:
            cases: List of dictionaries containing case information.
                  Each case should have:
                  - text: The text to embed (listing details)
                  - metadata: Additional information about the case
                  - id: Unique identifier for the case
        """
        # Extract data from cases
        ids = [case["id"] for case in cases]
        texts = [case["text"] for case in cases]
        metadatas = [case["metadata"] for case in cases]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
    
    def search_similar_cases(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar cases based on the query.
        
        Args:
            query: The text to search for similar cases
            n_results: Number of similar cases to return
            
        Returns:
            List of dictionaries containing similar cases with their metadata and correct outcome
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )        
        # Format results
        similar_cases = []
        for i in range(len(results["ids"][0])):
            similar_cases.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "correct_outcome": results["metadatas"][0][i].get("classification", "unknown")
            })
        
        return similar_cases
    
    def delete_collection(self):
        """Delete the current collection."""
        self.client.delete_collection(self.collection.name)
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the current collection."""
        return {
            "count": self.collection.count(),
            "name": self.collection.name
        }

    def debug_collection(self):
        """Debug the collection by printing stats and sample cases."""
        stats = self.get_collection_stats()
        print(f"Collection Stats: {stats}")
        if stats["count"] > 0:
            sample_cases = self.collection.get(limit=3)
            print(f"Sample Cases: {sample_cases}")
        else:
            print("No cases found in the collection.")

def format_case_for_storage(listing: Dict, classification: str, reasoning: str) -> Dict:
    """Format a case for storage in the vector store.
    
    Args:
        listing: The original product listing
        classification: The classification result
        reasoning: The reasoning for the classification
        
    Returns:
        Dictionary formatted for storage
    """
    # Create a text representation of the case
    text = f"""
    Title: {listing.get('title', '')}
    Description: {listing.get('description', '')}
    Category: {listing.get('category', '')}
    Keywords: {', '.join(listing.get('keywords', []))}
    Materials: {', '.join(listing.get('materials', []))}
    """
    
    # Create metadata
    metadata = {
        "classification": classification,
        "reasoning": reasoning,
        "has_images": bool(listing.get("images")),
        "num_images": len(listing.get("images", [])),
        "num_keywords": len(listing.get("keywords", [])),
        "num_materials": len(listing.get("materials", []))
    }
    
    # Create unique ID
    case_id = f"case_{hash(text)}"
    
    return {
        "id": case_id,
        "text": text,
        "metadata": metadata
    } 