"""
Streamlit program for interacting with the RAG system based on internal documentation.

This application allows you to:
- Run a document processing pipeline (chunking, embedding, indexing).
- Enter a query and get an answer based on information extracted from the documentation.

Author: zinaliashenko
Project: RAG Handbook Assistant
"""

import streamlit as st
import os
from utils.pipeline import run_data_pipeline, run_query_answer_pipeline

# Application title
st.title("🧠 RAG Handbook Assistant")

# Upload files section
uploaded_files = st.file_uploader(
    "📂 Load files (.pdf, .docx, .txt)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

DATA_PATH = "data"

if uploaded_files:
    # Save loaded files to data/ folder
    os.makedirs(DATA_PATH, exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"Loaded {len(uploaded_files)} files to {DATA_PATH}")

# Button to process documents
if st.button("📄 Data uploaded - process"):
    with st.spinner("Processing documents..."):
        run_data_pipeline()
    st.success("Documents processed! You can ask questions.")

# Input for a query
query = st.text_input("🔍 Enter a question:")

# Query processing and getting answer
if query:
    with st.spinner("Looking for an answer..."):
        run_query_answer_pipeline(query=query, 
                                  run_mode="streamlit",
                                  llm_provider="groq")
        
