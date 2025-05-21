import os
from dotenv import load_dotenv
from vector_store import VectorStore

# Load environment variables
load_dotenv()

def main():
    # Initialize vector store
    vector_store = VectorStore()
    
    # Debug the collection
    vector_store.debug_collection()
    
    # Example query
    query = "Category: clothing.gender_neutral_kids_clothing.hoodies_and_sweatshirts.hoodies\nDescription: Cozy hooded sweatshirt for children. Features a decorative toggle cord on the hood that can be adjusted. Perfect for layering in cool weather.\nKeywords: hoodie, sweatshirt, kids, hooded, toggle, children's clothing, youth apparel\nMaterials: cotton, polyester"
    
    # Search for similar cases
    similar_cases = vector_store.search_similar_cases(query, n_results=3)

if __name__ == "__main__":
    main() 