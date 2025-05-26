"""
CLI interface for running RAG pipeline from console.

This script allows you to:
- Process all documents (chunking, embedding, indexing).
- Enter a query manually via console.
- Get a response based on the processed documentation.

Usage:
$ python run_pipeline.py
"""

from utils.pipeline import run_data_pipeline, run_query_answer_pipeline


if __name__ == "__main__":

    print("[CLI MODE] Start pipeline...")
    run_data_pipeline()

    user_query = input("Input your query: ")
    run_query_answer_pipeline(query=user_query, 
                              run_mode="console",
                              llm_provider="groq")
