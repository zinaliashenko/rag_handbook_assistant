"""
Streamlit program for interacting with the RAG system based on internal documentation.

This application allows you to:
- Run a document processing pipeline (chunking, embedding, indexing).
- Enter a query and get an answer based on information extracted from the documentation.

Author: zinaliashenko
Project: RAG Handbook Assistant
"""

import streamlit as st
from utils.pipeline import run_data_pipeline, run_query_answer_pipeline

# Application title
st.title("🧠 RAG Handbook Assistant")

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
