import sys
import os

# Add the current directory to sys.path to find src
sys.path.append(os.getcwd())

from src.crag import build_graph
from src.vectorstore import build_vectorstore

def main():
    print("--- Corrective RAG CLI (Self-Correction & Evaluation) ---")
    
    # 1. Ask for a URL to index
    url = input("Enter a URL to index (or leave blank to skip): ")
    if url:
        print(f"Indexing {url}...")
        try:
            build_vectorstore([url])
            print("Indexing complete!")
        except Exception as e:
            print(f"Indexing failed: {e}")
            return

    # 2. Build the graph
    app = build_graph()
    
    # 3. Chat loop
    while True:
        question = input("\nUser: ")
        if question.lower() in ['exit', 'quit', 'q']:
            break
            
        initial_state = {
            "question": question,
            "chat_history": "",
            "documents": [],
            "generation": "",
            "web_fallback": False,
            "is_safe": True,
            "suggestions": []
        }
        
        print("Assistant: Thinking...")
        try:
            # We use stream to show steps
            for output in app.stream(initial_state):
                for key, value in output.items():
                    print(f"  [Node: {key}]")
            
            # The final result is in the last output
            # But let's just get the final state from the last iteration
            final_state = value 
            print(f"\nResponse: {final_state.get('generation')}")
            
            if final_state.get('suggestions'):
                print("\nSuggestions:")
                for s in final_state.get('suggestions'):
                    print(f"  - {s}")
                    
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure LM Studio is running on localhost:1234")

if __name__ == "__main__":
    main()
