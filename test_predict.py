#!/usr/bin/env python3
"""
Test script to validate the Qwen3-Reranker-8B model setup
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from predict import Predictor

def test_predictor():
    """Test the predictor with sample data"""
    print("Initializing Qwen3-Reranker-8B model...")

    try:
        # Initialize predictor
        predictor = Predictor()
        predictor.setup()
        print("✓ Model loaded successfully!")

        # Test data
        instruction = "Given a search query and a document, evaluate how relevant the document is to answering the query."
        query = "What is artificial intelligence?"
        documents = json.dumps([
            "Artificial intelligence is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence.",
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "Natural language processing is a field of artificial intelligence that helps computers understand human language.",
            "Deep learning is a type of machine learning that uses neural networks with multiple layers to process data."
        ])

        print("\nRunning reranking test...")
        result = predictor.predict(
            instruction=instruction,
            query=query,
            documents=documents,
            top_k=3,
            batch_size=2
        )

        # Parse and display results
        results = json.loads(result)
        print(f"\nQuery: {results['query']}")
        print(f"Total documents processed: {results['total_documents']}")
        print("\nTop ranked documents:")

        for i, doc_result in enumerate(results['results'], 1):
            print(f"{i}. Score: {doc_result['score']:.4f}")
            print(f"   Document: {doc_result['document'][:100]}...")
            print()

        print("✓ Test completed successfully!")
        return True

    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_predictor()
    sys.exit(0 if success else 1)