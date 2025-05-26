"""
A module for splitting PDF documents into text chunks and saving statistics.

Functions:
- split_into_chunks(files_paths): splits documents into chunks by headers.
- save_chunks_and_stats(all_chunks): saves chunks with metadata (file, page, tokens, words, sentences, etc.) in JSON.

Uses `unstructured`, `nltk` and Hugging Face `tokenizer`.
"""

from unstructured.partition.pdf import partition_pdf
from time import perf_counter as timer
from unstructured.documents.elements import NarrativeText, Title, Table
import re
import nltk
from transformers import AutoTokenizer
import json

from .config import TOKENIZER_MODEL, CHUNKS_PATH

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


def download_punkt_if_needed():
    """
    Downloades NLTR 'punkt'.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("[INFO] Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')


def split_into_chunks(files_paths: list[str]) -> list[list]:
    """
    Splits PDF files into chunks.
    """

    all_chunks = []
    for file_path in files_paths:
        start_time = timer()

        chunks = partition_pdf(
            filename=file_path,
            strategy="auto",  # не активує detectron2, якщо явно не потрібно
            extract_images_in_pdf=False,  # забороняє image extraction
            chunking_strategy="by_title",
            max_characters=2500,
            combine_text_under_n_chars=500,
            new_after_n_chars=2000,
        )
        """
        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True, # enables the detection and extraction of tables as structured elements
            strategy="fast", # "hi_res" - deep learning framework for object detection, segmentation, and layout analysis in documents or images
            extract_image_block_types=[], # ["Image", "Table"] specifies the types of blocks to be treated as images
            #extract_image_block_to_payload=True, # converts extracted images to a base64-encoded format
            use_ocr=False,
            ocr_strategy="no_ocr",
            chunking_strategy="by_title", # controls how the document is split into chunks for processing
            max_characters=2500, # sets the maximum number of characters in a chunk
            combine_text_under_n_chars=500, # combines small text fragments into a single chunk
            new_after_n_chars=2000, # creates a new chunk after reaching this character limit
        )"""
        if not chunks:
            print(f"[WARNING] No chunks extracted from {file_path}")
        else:
            all_chunks.append(chunks)
        end_time = timer()
        print(f"[INFO] Time: {end_time-start_time:.5f} seconds.")
    
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

            if not hasattr(chunk, "metadata") or chunk.metadata is None:
                continue

            elements = chunk.metadata.orig_elements
            element_types = list({el.category for el in elements})

            text_parts = [
                el.text for el in elements if isinstance(el, (NarrativeText, Title, Table))
            ]
            full_text = "\n".join(text_parts)
            cleaned_text = text_formatter(text=full_text)

            chunks_and_statistics.append({
                "file_directory": chunk.metadata.file_directory,
                "file_name": chunk.metadata.filename,
                "file_type": chunk.metadata.filetype,
                "page_number": getattr(chunk.metadata, "page_number", -1),
                "page_char_count": len(chunk.text),
                "page_word_count": len(re.findall(r'\w+', cleaned_text)),
                "page_sentence_count_raw": len(nltk.sent_tokenize(cleaned_text)),
                "page_token_count": len(tokenizer(cleaned_text)["input_ids"]),
                "origin_elements": element_types,
                "title": is_valid_title(next((el.text for el in elements if isinstance(el, Title)), None)),
                "contains_text": any(isinstance(el, (NarrativeText, Title, Table)) for el in elements),
                "text": cleaned_text
            })

    end_time = timer()
    print(f"[INFO] Time: {end_time-start_time:.5f} seconds.")

    save_to_json(data=chunks_and_statistics, 
                 path=path_to_save)
    
    print(f"[INFO] Chunks and statistics were saved.")
