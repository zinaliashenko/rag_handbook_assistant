"""
The main module for running two RAG system pipelines:
1. run_data_pipeline() — performs document processing (chunking, embedding, indexing).
2. run_query_answer_pipeline() — processes a user query, finds relevant chunks, and calls LLM.

Modules:
- chunker: splitting documents into chunks.
- embedder: creating embeddings for chunks and queries.
- retriever: indexing and searching for similar chunks.
- llm_interface: preparing a query template, calling LLM, formatting the response.
"""

import os
from pathlib import Path
from time import perf_counter as timer

from .chunker import save_chunks_and_stats, split_into_chunks
from .config import CHUNKS_PATH, DATA_PATH, INDEX_PATH
from .embedder import embed_chunks, embed_query
from .llm_interface import (
    chat_template_groq,
    client_response_groq,
    print_response_console,
    print_response_markdown,
)
from .retriever import index_chunks, retrieve_top_k_chunks


def run_data_pipeline():
    """
    Full data processing pipeline:
    - creates necessary folders if missing;
    - finds all files in DATA_PATH with supported extensions (.pdf, .docx, .txt);
    - splits documents into chunks;
    - creates embeddings for chunks;
    - indexes embeddings with FAISS.
    """
    print("[INFO] Start data pipeline.")

    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)

    # Files search
    data_dir = DATA_PATH
    print(f"Data directory: {data_dir}")

    allowed_ext = (".pdf", ".docx", ".txt")
    files_paths = [
        os.path.join(data_dir, file_name)
        for file_name in os.listdir(data_dir)
        if file_name.lower().endswith(allowed_ext)
    ]

    for file in files_paths:
        file_path = Path(file)
        if not file_path.exists():
            print(f"[WARNING] File path {file_path} does not exist.")

    # chunker
    print("[INFO] Running chunker...")
    chunks = split_into_chunks(files_paths)
    save_chunks_and_stats(all_chunks=chunks)

    # embedder.py
    print("[INFO] Creating embeddings for chunks...")
    embed_chunks()

    # retriever.py
    print("[INFO] Indexing embeddings with FAISS...")
    index_chunks()
    print(f"Пробуємо знайти індекс за шляхом: {INDEX_PATH}")
    print(
        f"Файли в папці chunks/: {os.listdir(os.path.dirname(INDEX_PATH)) if os.path.exists(os.path.dirname(INDEX_PATH)) else 'папка відсутня'}"
    )
    print(
        f"Файли в data/: {os.listdir('data') if os.path.exists('data') else 'папка відсутня'}"
    )


def run_query_answer_pipeline(
    query: str = None, run_mode="console", llm_provider="groq"
):
    """
    Pipeline for responding to user request:
    - creates embedding for request;
    - finds top-K relevant chunks;
    - forms query template to LLM;
    - sends query to LLM and returns response.

    Parameters:
    - query (str): query text. If None — takes QUERY from config.py;
    - run_mode (str): "console" or "streamlit" — response output format;
    - llm_provider (str): which LLM is used (e.g., "groq").
    """
    start_time = timer()

    #  Query
    if query is None:
        from .config import QUERY

        query = QUERY

    # embedder.py
    print("[INFO] Embedding query...")
    query_embedding = embed_query(query=query)

    # retriever.py
    print("[INFO] Retrieving top-k relevant chunks...")
    indices = retrieve_top_k_chunks(query_embedding=query_embedding)

    # llm_interface.py
    print("[INFO] Generating prompt and calling LLM...")
    print(f"[INFO] LLM provider: {llm_provider} | Run mode: {run_mode}")

    dialogue_template = chat_template_groq(indices=indices, query=query)
    response_text = client_response_groq(dialogue_template=dialogue_template)

    end_time = timer()
    print(f"[INFO] Time: {end_time-start_time:.5f} seconds.")

    # Response printing
    if run_mode == "streamlit":
        print_response_markdown(response_text)
    else:
        print_response_console(response_text)
