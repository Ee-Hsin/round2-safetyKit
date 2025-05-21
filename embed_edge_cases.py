import json
from vector_store import VectorStore, format_case_for_storage

def main():
    # Initialize vector store
    vector_store = VectorStore()
    
    # Load edge cases
    with open("example_cases_dataset.json", 'r') as f:
        edge_cases = json.load(f)
    
    # Format cases for storage
    formatted_cases = []
    for case in edge_cases["data"]:
        formatted_case = format_case_for_storage(
            listing=case["reviewInput"],
            classification=case["expectedOutcome"],
            reasoning=""  # Edge cases don't have reasoning in the dataset
        )
        formatted_cases.append(formatted_case)
        
    print(f"Formatted cases: {formatted_cases}")
    
    # Add cases to vector store
    vector_store.add_cases(formatted_cases)
    
    # Print collection stats
    stats = vector_store.get_collection_stats()
    print(f"Successfully embedded {stats['count']} cases into collection '{stats['name']}'")

if __name__ == "__main__":
    main() 