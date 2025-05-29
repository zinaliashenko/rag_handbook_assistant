# 🧠 RAG Reference Assistant

An interactive Retrieval-Augmented Generation (RAG)-based answer retrieval system for technical documentation. The project allows you to upload PDF files, split them into chunks, create vector embeddings, and get answers to queries using Large Language Models (LLM).

## 🚀 Features

- Upload PDF files to local `data/` folder
- Automatically split documents into logical chunks with metadata
- Generate embeddings using `SentenceTransformer`
- Indexing using FAISS
- Handle user requests using LLM via Groq API
- Streamlit-based GUI
- Alternative launch from console


## 🗂️ Project structure

rag_handbook_assistant/
├── app.py # Streamlit interface
├── run_pipeline.py # Launch via console
├── data/ # PDF files are uploaded here
├── chunks/ # Save chunks and embeddings
├── utils/ # Main logic
│ ├── chunker.py # Chunking
│ ├── embedder.py # Embeddings for chunks and queries
│ ├── retriever.py # Finding similar chunks via FAISS
│ ├── llm_interface.py # Calling LLM
│ ├── pipeline.py # Universal pipeline file( called from CLI or Streamlit)
│ └── config.py # Constants (paths, base prompt, etc.)
├── .env # API key for Groq (local)
└── .streamlit/
└── secrets.toml # API key for Groq (in Streamlit Cloud)


## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/zinaliashenko/rag_handbook_assistant.git
cd rag_handbook_assistant
```

2. Create a virtual environment and activate it (recommended):
```bash
pip install -r requirements.txt
```

2. Create a .env file in the project root and add your Groq API key:
```bash
GROQ_API_KEY=your_api_key_here
```

## ▶️ Launching

- **Streamlit (GUI)**:
```bash
streamlit run app.py
```

- **Console (CLI)**:
```bash
python run_pipeline.py
```

## 🧪 Usage Example

1. Upload one or more PDF files to the data/ folder

2. In the Streamlit app, click "Run Processing"

3. Enter a query such as:

"What are typical working hours?"

The system will return an answer based on the most relevant chunks from the documents.


## 🛠️ Dependencies:

- openai
- sentence-transformers
- faiss-cpu
- PyMuPDF
- Streamlit

All features can be set via requirements.txt.

🔗 Links
- 💻 GitHub: https://github.com/zinaliashenko/rag_handbook_assistant

- 🌍 Demo: https://rag-handbook.streamlit.app/

## 📬 Contact

**Author:** [zinaliashenko](https://github.com/zinaliashenko)

📄 License
MIT License.