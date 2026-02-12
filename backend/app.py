from flask import Flask, request, jsonify
from document_loader import load_document
from chunking import chunk_text
from embeddings import create_embeddings, model
from vector_store import create_index, search_index

app = Flask(__name__)

stored_chunks = []
faiss_index = None


@app.route("/")
def home():
    return "Document QA Backend Running"


# -------- Upload ----------
@app.route("/upload", methods=["POST"])
def upload():
    global stored_chunks, faiss_index

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        text = load_document(file)
        stored_chunks = chunk_text(text)

        embeddings = create_embeddings(stored_chunks)
        faiss_index = create_index(embeddings)

        return jsonify({
            "message": "Document indexed successfully",
            "chunks": len(stored_chunks)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------- Ask ----------
@app.route("/ask", methods=["POST"])
def ask():
    global stored_chunks, faiss_index

    try:
        if faiss_index is None:
            return jsonify({"error": "Upload document first"}), 400

        data = request.get_json()
        question = data["question"]

        query_embedding = model.encode(question)

        indices = search_index(faiss_index, query_embedding)

        relevant_chunks = [stored_chunks[i] for i in indices]

        return jsonify({"answer": " ".join(relevant_chunks)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
