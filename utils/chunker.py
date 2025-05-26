"""
A module for splitting PDF documents into text chunks and saving statistics.

Functions:
- split_into_chunks(files_paths): splits documents into chunks by pages.
- save_chunks_and_stats(all_chunks): saves chunks with metadata (file, page, tokens, words, sentences, etc.) in JSON.

Uses `fitz` (PyMuPDF), `nltk` and Hugging Face `tokenizer`.
"""

import fitz  # PyMuPDF
from time import perf_counter as timer
import re
import nltk
from transformers import AutoTokenizer
import json
import os

from .config import TOKENIZER_MODEL, CHUNKS_PATH

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


def download_punkt_if_needed():
    """
    Downloads NLTK 'punkt' tokenizer if needed.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("[INFO] Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')


def split_into_chunks(files_paths: list[str]) -> list[list]:
    """
    Splits PDF files into chunks by page.
    Each page is treated as one chunk.
    """
    all_chunks = []
    for file_path in files_paths:
        start_time = timer()

        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}")
            continue

        doc = fitz.open(file_path)
        chunks = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text.strip():
                continue  # skip empty pages

            chunks.append({
                "file_directory": os.path.dirname(file_path),
                "file_name": os.path.basename(file_path),
                "file_type": "application/pdf",
                "page_number": page_num,
                "text": text
            })

        if not chunks:
            print(f"[WARNING] No text chunks extracted from {file_path}")
        else:
            all_chunks.append(chunks)

        end_time = timer()
        print(f"[INFO] Processed '{file_path}' in {end_time - start_time:.2f} seconds.")

    return all_chunks


def text_formatter(text: str) -> str:
    """
    Performs minor formatting on text.
    """
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text


def is_valid_title(title: str) -> str:
    """
    Set "Untitled" title for too short titles.
    """
    if not title or len(title.strip()) < 5:
        return "Untitled"
    return title


def save_to_json(data, path):
    """
    Saves data to a chosen folder.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_chunks_and_stats(all_chunks: list[list], 
                          tokenizer: AutoTokenizer=tokenizer,
                          path_to_save: str=CHUNKS_PATH) -> None:
    """
    Saves chunks and their statistics to a file.
    """
    download_punkt_if_needed()

    chunks_and_statistics = []
    start_time = timer()

    for chunks in all_chunks:
        for chunk in chunks:
            cleaned_text = text_formatter(chunk["text"])

            chunks_and_statistics.append({
                "file_directory": chunk["file_directory"],
                "file_name": chunk["file_name"],
                "file_type": chunk["file_type"],
                "page_number": chunk["page_number"],
                "page_char_count": len(chunk["text"]),
                "page_word_count": len(re.findall(r'\\w+', cleaned_text)),
                "page_sentence_count_raw": len(nltk.sent_tokenize(cleaned_text)),
                "page_token_count": len(tokenizer(cleaned_text)["input_ids"]),
                "origin_elements": ["PageText"],
                "title": f"Page {chunk['page_number']}",
                "contains_text": True,
                "text": cleaned_text
            })

    end_time = timer()
    print(f"[INFO] Statistics generated in {end_time - start_time:.2f} seconds.")

    save_to_json(data=chunks_and_statistics, 
                 path=path_to_save)

    print(f"[INFO] Chunks and statistics were saved.")