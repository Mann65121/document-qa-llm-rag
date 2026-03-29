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
@app.post("/api/upload")
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Please choose a PDF file to upload."}), 400

        file = request.files["file"]

        if not file or not file.filename:
            return jsonify({"error": "Please choose a PDF file to upload."}), 400

        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported."}), 400

        text = load_document(file)

        if not text.strip():
            return jsonify({"error": "The PDF does not contain readable text."}), 400

        chunks = chunk_text(text)
        if not chunks:
            return jsonify({"error": "The document could not be split into chunks."}), 400

        document_state["filename"] = file.filename
        document_state["text"] = text
        document_state["chunks"] = chunks

        return jsonify(
            {
                "message": "Document processed successfully.",
                "filename": file.filename,
                "total_characters": len(text),
                "total_chunks": len(chunks),
                "preview": text[:320],
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Upload failed: {exc}"}), 500



