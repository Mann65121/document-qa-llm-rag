import os

from flask import Flask, jsonify, render_template, request

from chunking import chunk_text
from document_loader import load_document
from rag_pipeline import answer_question

app = Flask(__name__)

document_state = {
    "filename": None,
    "text": "",
    "chunks": [],
}
@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "document_loaded": bool(document_state["chunks"]),
            "chunk_count": len(document_state["chunks"]),
            "answer_mode": "ollama-or-local-grounded-generative",
            "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.2"),
            "ollama_url": os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        }
    )



