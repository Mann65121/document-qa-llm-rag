# document-qa-llm-rag

Atlas RAG Studio is a document question-answering app built with Flask. It lets you upload a PDF, chunk its text, retrieve the most relevant sections, and answer questions through a custom frontend.

## Features

- PDF upload and text extraction
- Chunk-based retrieval with source snippets
- Grounded generative answers with Ollama and a local fallback
- Confidence labels and citation-style answers for better presentation
- Clean Flask API
- Distinctive responsive frontend
- Minimal dependencies for easier setup

## Generative mode

The app works in two modes:

- Preferred: Ollama-backed grounded generation through the backend
- Fallback: local grounded answer synthesis when Ollama is unavailable

Install the dependencies:

```powershell
pip install -r requirements.txt
```

Start Ollama and pull a model before running the app:

```powershell
ollama pull llama3.2
ollama serve
```

Optional: choose a different model explicitly.

```powershell
$env:OLLAMA_MODEL="llama3.2"
```

Then start the app:

```powershell
python backend\app.py
```

The app will use Ollama from the backend when it is available at `http://127.0.0.1:11434`, and it will fall back to the built-in local grounded answerer otherwise.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend\app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## API

- `GET /api/health`
- `POST /api/upload`
- `POST /api/ask`
