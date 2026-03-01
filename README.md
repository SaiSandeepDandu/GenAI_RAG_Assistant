# RAG GenAI Assistant

## Overview
This repository contains a **Retrieval-Augmented Generation (RAG) based AI assistant** designed to answer user queries using information from a PDF document. The assistant retrieves relevant content from the document using semantic search and generates responses via a large language model (LLM). This project demonstrates principles for GenAI applications, including modular architecture, vector based retrieval, and context aware LLM prompting.

## Features
- **RAG Architecture**: Retrieves relevant content from documents before generating responses to ensure factual accuracy.
- **Vector-based Search**: Uses **FAISS** for efficient similarity search over document embeddings.
- **LLM Integration**: Uses **Qwen-3** for generating context-aware answers.
- **Context-aware Assistant**: Maintains conversation history to provide coherent multi-turn responses.
- **Python Modular Design**: Clean separation of utility functions and main application logic.

## Tech Stack
- **Python 3.10+**  
- **Transformers** (Hugging Face) for LLM inference  
- **Sentence-Transformers** for embedding generation  
- **FAISS** for vector search  
- **LangChain & langchain_community** for document processing  
- **PyPDF** for PDF ingestion  

### Setup Instructions

1. Virtual environment creation - python -m venv venv
2. Activate virtual environment - venv\Scripts\activate
3. Install dependencies - pip install -r requirements.txt
4. Run application - python app.py
5. Type your questions in the console.
6. Type exit to quit.


## Future Improvements
1. Web-based interface (Streamlit or FastAPI) for easier interaction
2. Support for multiple document formats (Word, PDFs, etc.)