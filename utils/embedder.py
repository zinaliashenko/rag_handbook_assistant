"""
A module for embedding chunks and query, and saving chunks embeddings.

Functions:
- embed_chunks(path_to_save, embedding_model): builds embeddings for loaded chunks and save them back to a file.
- embed_query(query, embeddings_model): builds embeddings for a query.

Uses SentenceTransformer.
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import save_to_json
from .config import CHUNKS_PATH, EMBEDDING_MODEL

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(
    path_to_save: str = CHUNKS_PATH,
    embedding_model: SentenceTransformer = embedding_model,
) -> None:
    """
    Loads chunks from JSON, builds embeddings for each of them, and saves back to a file.
    """
    # Open file with chunks
    with open(path_to_save, "r", encoding="utf-8") as f:
        chunks_and_statistics = json.load(f)

    print(f"[INFO] Chunks are loaded.")

    # Extract text from each chunk
    chunks_texts = [chunk["text"] for chunk in chunks_and_statistics]

    # Create embeddings in batches with a progress bar
    embeddings = embedding_model.encode(
        chunks_texts, batch_size=32, show_progress_bar=True
    ).astype("float32")
    print(f"[INFO] Chunks are embedded.")

    # Add embeddings to the chunk structure
    for chunk, embedding in zip(chunks_and_statistics, embeddings):
        chunk["embedding"] = embedding.tolist()

    # Save updated chunks with embeddings
    save_to_json(data=chunks_and_statistics, path=path_to_save)

    print(f"[INFO] Embeddins were saved.")


def embed_query(
    query: str, embedding_model: SentenceTransformer = embedding_model
) -> np.ndarray:
    """
    Returns the embedding for a single query in NumPy array format.
    """
    # Query embedding and returnes numpy array.
    query_embedding = embedding_model.encode([query]).astype("float32")
    print(f"[INFO] Query is embedded.")
    return query_embedding.squeeze(0)
