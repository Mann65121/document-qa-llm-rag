# document-qa-llm-rag

Atlas RAG Studio is a document question-answering application built with Flask and Ollama. It allows users to upload a PDF, extract its text, retrieve the most relevant context, and generate concise answers grounded in the document content.

## Overview

This project combines PDF text extraction, chunk-based retrieval, and local LLM generation to create a simple document QA workflow. The application runs locally and does not require a paid API.

## Features

- PDF upload and text extraction
- Context retrieval using chunked document text
- Concise grounded answer generation with Ollama
- Local fallback answering when Ollama is unavailable
- Confidence label for each answer
- Recent question history
- Reset workspace option
- Responsive web interface

## Tech stack

- Python
- Flask
- pypdf
- PyMuPDF
- Ollama
- HTML, CSS, JavaScript

## Project structure

```text
document-qa-llm-rag/
|-- backend/
|   |-- app.py
|   |-- chunking.py
|   |-- document_loader.py
|   |-- rag_pipeline.py
|   |-- vector_store.py
|   |-- static/
|   |   |-- app.js
|   |   `-- styles.css
|   `-- templates/
|       `-- index.html
|-- .env.example
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## How it works

1. Upload a PDF document.
2. Extract readable text from the document.
3. Split the text into overlapping chunks.
4. Retrieve the most relevant chunks for a question.
5. Send the retrieved context to Ollama for answer generation.
6. Return a concise answer with a confidence label.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the application

First, make sure Ollama is installed and the model is available:

```powershell
ollama pull llama3.2
```

Then run the Flask app:

```powershell
python backend\app.py
```

Open the application in your browser:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

## Optional environment variables

```powershell
$env:OLLAMA_MODEL="llama3.2"
$env:OLLAMA_URL="http://127.0.0.1:11434/api/generate"
```

## API endpoints

- `GET /api/health`
- `GET /api/history`
- `POST /api/upload`
- `POST /api/ask`
- `POST /api/reset`

## Notes

- The application is designed to run locally.
- Ollama is used as the primary answer generator.
- If Ollama is unavailable, the application falls back to a local grounded answer flow.
