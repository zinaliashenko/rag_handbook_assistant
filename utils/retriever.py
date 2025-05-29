"""
Functions for indexing and finding the nearest chunks by embeddings.

- index_chunks(path_to_save, faiss_path): creates a FAISS index from the chunk embeddings and saves it.
- retrieve_top_k_chunks(query_embedding, k, faiss_path) -> list[int]: finds the indices of the k nearest chunks to the query.

The indices are used to retrieve the corresponding chunks in chunks_and_statistics.
"""

import json

import faiss
import numpy as np

from .config import CHUNKS_PATH, INDEX_PATH, TOP_K


def index_chunks(path_to_save: str = CHUNKS_PATH, faiss_path: str = INDEX_PATH) -> None:
    """
    Indexes embeddings from the chunks_and_statistics.json file in FAISS and stores the index.
    """
    # # Open file with chunks
    with open(path_to_save, "r", encoding="utf-8") as file:
        chunks_and_statistics = json.load(file)
    print(f"[INFO] Chunks are loaded.")

    # Get embeddings in numpy array
    embedding_array = np.array(
        [chunk["embedding"] for chunk in chunks_and_statistics], dtype="float32"
    )
    print(f"INFO Embedding shape: {embedding_array.shape}")

    # Create indices for embeddings
    index = faiss.IndexFlatL2(embedding_array.shape[1])
    index.add(embedding_array)

    print(f"[INFO] In FAISS index {index.ntotal} vectors")

    # Save indices to a file
    faiss.write_index(index, faiss_path)
    print(f"[INFO] Indices were saved.")


def retrieve_top_k_chunks(
    query_embedding: np.ndarray, k: int = TOP_K, faiss_path: str = INDEX_PATH
) -> list[int]:
    """
    Returns the top-k indices of the closest chunks for the given query embedding.
    """
    # Load indices
    index = faiss.read_index(faiss_path)
    print(f"[INFO] FAISS indices are read.")

    if query_embedding.ndim == 1:
        query_embedding = query_embedding[np.newaxis, :]

    # Search top k indices of the nearest embeddings
    distances, indices = index.search(query_embedding, k)
    print(f"[INFO] Retrieving top {k} neares chunks are finished.")

    print(f"[DEBUG]")
    # print top k chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks_and_statistics = json.load(file)
    chunks_texts = [chunk["text"] for chunk in chunks_and_statistics]
    print("[INFO] Nearest chunks:")
    for dist, idx in zip(distances[0], indices[0]):
        print(f"[INFO] Index: {idx}, Distance: {dist}")
        print(chunks_texts[idx])
        print("---")

    return indices[0].tolist()
