# RAG AI Agent

This project provides a modern PDF-based Retrieval-Augmented Generation (RAG) pipeline that can be run locally and deployed to GitHub.

## Features
- Accepts a PDF uploaded in the Streamlit app or one already placed in your workspace
- Answers questions strictly from the uploaded/selected PDF content
- Uses Hugging Face embeddings
- Supports optional LLM providers such as OpenAI, Groq, or Ollama
- Works from the command line and in Streamlit

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Place a PDF in the workspace

Drop your PDF file into the project folder (or set RAG_PDF_PATH to its location).

### 2. Run the app

```bash
streamlit run app.py
```

Then type your question in the input box and press Ask.

### 3. Use the command-line version

```bash
python rag.py --pdf "path/to/your.pdf" --rebuild
python rag.py --query "What does this document say about X?"
```

### 4. Use an LLM provider

Create a .env file from .env.example and add your API key:

```bash
cp .env.example .env
```

Then run:

```bash
python rag.py --pdf "path/to/your.pdf" --query "What is the main topic?" --provider groq --model llama-3.3-70b-versatile
```
