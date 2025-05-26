import os
# === LLM ===

def get_groq_api_key():
    """
    Returnes GROQ API key from Streamlit secrets or .env.
    """
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        return os.getenv("GROQ_API_KEY")

LLM_GROQ = {
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "base_url": "https://api.groq.com/openai/v1",
    "api_key": get_groq_api_key()
}

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHUNKS_PATH = os.path.join(BASE_DIR, "chunks", "chunks_and_statistics.json")
INDEX_PATH = os.path.join(BASE_DIR, "chunks", "faiss.index")
DATA_PATH = os.path.join(BASE_DIR, "data")

# === MODELS ===
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# === RETRIEVAL ===
TOP_K = 5

# === PROMPTS ===
BASE_PROMPT = """You are an AI assistant that helps employees understand internal company policies and handbooks.
    Your goal is to provide clear, concise, and accurate answers based strictly on the provided internal documentation.
    Start by identifying and extracting the most relevant information from the context before forming your answer.
    Do not include the extraction or reasoning in your output — only the final answer.
    If the answer is not directly supported by the context, politely state that no definitive answer can be provided.
    Be professional and helpful in tone, and use complete sentences with thorough explanations where appropriate.

    Use the following examples as a reference for the tone and style:

    \nExample 1:
    Query: What is the company's policy on remote work?
    Answer: According to the company's internal policy, employees are allowed to work remotely up to three days per week, provided their role permits it and they have prior approval from their manager. Employees must ensure they are accessible during core working hours and are responsible for maintaining productivity and data security while working remotely.

    \nExample 2:
    Query: How does the company handle parental leave?
    Answer: The company offers up to 16 weeks of paid parental leave for primary caregivers and 4 weeks for secondary caregivers. Employees must notify HR at least 30 days in advance and submit the required documentation. Leave can be taken continuously or split into two separate periods within one year of the child’s birth or adoption.

    \nExample 3:
    Query: Are employees allowed to accept gifts from clients?
    Answer: As per the internal code of conduct, employees may accept nominal gifts (e.g., promotional items worth under $50) from clients, but must disclose any gifts exceeding this amount to their direct manager or HR. Accepting gifts must never influence business decisions or create the appearance of a conflict of interest.

    \nNow use the following internal policy context to answer the employee's question:
    {context}

    \nRelevant passages: <extract relevant passages from the context here>
    Employee query: {query}
    Answer:"""

DIALOGUE_INSTRUCTION = {
            "role": "system",
            "content": """You are an AI assistant that helps employees understand internal policies.

    When answering, always follow **this strict JSON format**:

    {
    "thoughts": "<internal reasoning here>",
    "answer": "<final answer only>"
    }

    Only fill in the fields. Do not include any text outside this format.
    If you don't know the answer, write that in the 'answer' field only.
    """
        }


QUERY = "What are typical working hours?"
