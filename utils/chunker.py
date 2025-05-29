"""
A module for splitting PDF documents into text chunks and saving statistics using PyMuPDF.

Functions:
- split_into_chunks(files_paths): splits PDF pages into text chunks.
- save_chunks_and_stats(all_chunks): saves chunks with metadata in JSON.

Uses PyMuPDF, NLTK, and Hugging Face tokenizer.
"""

import json
import re
from time import perf_counter as timer

import fitz  # PyMuPDF
import nltk
from transformers import AutoTokenizer

from .config import CHUNKS_PATH, TOKENIZER_MODEL

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


def download_punkt_if_needed():
    """
    Downloades NLTR 'punkt'.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("[INFO] Downloading NLTK 'punkt' tokenizer...")
        nltk.download("punkt")


def text_formatter(text: str) -> str:
    """
    Performs minor formatting on text.
    """
    return text.replace("\n", " ").strip()


def is_valid_title(text: str) -> str:
    """
    Set "Untitled" title for too short titles.
    """
    title = text.strip().split("\n")[0]
    return title if len(title) >= 5 else "Untitled"


def split_into_chunks(files_paths: list[str]) -> list[list[dict]]:
    """
    Splits PDFs into chunks per page (or optionally using headers).
    Returns a list of list of dicts per file.
    """
    download_punkt_if_needed()

    all_chunks = []

    for file_path in files_paths:
        start_time = timer()
        doc_chunks = []

        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            if not text.strip():
                continue

            cleaned_text = text_formatter(text)
            #title = is_valid_title(cleaned_text)

            doc_chunks.append(
                {
                    "file_directory": str(file_path).rsplit("/", 1)[0],
                    "file_type": "pdf",
                    "page_number": i + 1,
                    "page_char_count": len(text),
                    "page_word_count": len(re.findall(r"\w+", cleaned_text)),
                    "page_sentence_count_raw": len(text.split(". ")),
                    "page_token_count": len(tokenizer(cleaned_text)["input_ids"]),
                    "contains_text": True,
                    "text": cleaned_text,
                }
            )

        all_chunks.append(doc_chunks)
        end_time = timer()
        print(f"[INFO] Parsed '{file_path}' in {end_time - start_time:.3f} seconds")

    return all_chunks


def save_to_json(data, path):
    """
    Saves data to a chosen folder.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_chunks_and_stats(all_chunks: list[list], path_to_save: str = CHUNKS_PATH):
    """
    Saves chunks and their statistics to JSON file.
    """
    flat_chunks = [chunk for doc_chunks in all_chunks for chunk in doc_chunks]

    start_time = timer()
    save_to_json(flat_chunks, path_to_save)
    end_time = timer()

    print(
        f"[INFO] Saved {len(flat_chunks)} chunks in {end_time - start_time:.2f} seconds."
    )
