import sys
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from the .env file so Groq authenticates properly
load_dotenv(find_dotenv(), override=True)

# Add the project root to path so Python can find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.rag.chain import ask_rag

# Define test queries and expected keywords/topics for evaluation
TEST_SET = [
    {
        "query": "What are Person 1's tasks?",
        "expected_keyword": "chunk"
    },
    {
        "query": "What is the recommended project title?",
        "expected_keyword": "Vector Database"
    },
    {
        "query": "How many chunks were generated from the documents?",
        "expected_keyword": "cannot find"
    },
    {
        
    "query": "According to the project documentation, which vector database is used to store the embeddings?",
    "expected_keyword": "Chroma"

    }
]

def run_evaluation():
    print("🚀 Starting RAG Pipeline Evaluation...\n")
    passed = 0
    total = len(TEST_SET)

    for i, test in enumerate(TEST_SET, 1):
        query = test["query"]
        keyword = test["expected_keyword"]

        print(f"Test {i}: {query}")
        try:
            answer, docs = ask_rag(query, k=5)
            print(f"  💡 Answer: {answer}")
            print(f"  📚 Retrieved Sources: {len(docs)} chunks")

            # Show the cosine distance of each retrieved chunk (lower = more similar)
            for j, d in enumerate(docs, 1):
                dist = d.metadata.get("distance")
                if dist is not None:
                    print(f"     chunk {j}: distance={dist:.4f} | similarity={1 - dist:.4f}")

            # Check if expected keyword is in the response (simple evaluation metric)
            if keyword.lower() in answer.lower():
                print(f"  ✅ Status: PASSED (Found keyword: '{keyword}')\n")
                passed += 1
            else:
                print(f"  ⚠️ Status: REVIEW (Keyword '{keyword}' not explicitly found in answer)\n")
        except Exception as e:
            print(f"  ❌ Status: ERROR - {e}\n")

    print("-" * 40)
    print(f"Evaluation Complete: {passed}/{total} tests passed successfully.")

if __name__ == "__main__":
    run_evaluation()