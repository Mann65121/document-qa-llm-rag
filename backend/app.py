import os
from datetime import datetime

from flask import Flask, jsonify, render_template, request

from chunking import chunk_text
from document_loader import load_document
from rag_pipeline import answer_question

app = Flask(__name__)

document_state = {
    "filename": None,
    "text": "",
    "chunks": [],
    "uploaded_at": None,
    "history": [],
}


def reset_document_state():
    document_state["filename"] = None
    document_state["text"] = ""
    document_state["chunks"] = []
    document_state["uploaded_at"] = None
    document_state["history"] = []


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "document_loaded": bool(document_state["chunks"]),
            "chunk_count": len(document_state["chunks"]),
            "history_count": len(document_state["history"]),
            "filename": document_state["filename"],
            "answer_mode": "ollama-or-local-grounded-generative",
            "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.2"),
            "ollama_url": os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        }
    )


@app.get("/api/history")
def history():
    return jsonify(
        {
            "filename": document_state["filename"],
            "items": document_state["history"],
        }
    )


@app.post("/api/reset")
def reset():
    reset_document_state()
    return jsonify({"message": "Workspace cleared successfully."})


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

        reset_document_state()
        document_state["filename"] = file.filename
        document_state["text"] = text
        document_state["chunks"] = chunks
        document_state["uploaded_at"] = datetime.utcnow().isoformat() + "Z"

        return jsonify(
            {
                "message": "Document processed successfully.",
                "filename": file.filename,
                "total_characters": len(text),
                "total_chunks": len(chunks),
                "uploaded_at": document_state["uploaded_at"],
                "preview": text[:320],
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Upload failed: {exc}"}), 500


@app.post("/api/ask")
def ask():
    try:
        if not document_state["chunks"]:
            return jsonify({"error": "Upload a PDF before asking a question."}), 400

        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()

        if not question:
            return jsonify({"error": "Please enter a question."}), 400

        result = answer_question(document_state["chunks"], question)
        result["document"] = document_state["filename"]

        document_state["history"].insert(
            0,
            {
                "question": question,
                "answer": result["answer"],
                "confidence": result.get("confidence", "unknown"),
                "mode": result.get("generation", {}).get("mode", "unknown"),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )
        document_state["history"] = document_state["history"][:8]

        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Question processing failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
