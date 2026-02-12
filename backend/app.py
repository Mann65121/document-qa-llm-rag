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

@app.route("/upload", methods=["POST"])
def upload():
    global stored_chunks, faiss_index

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # Step 1: Extract text
        text = load_document(file)

        # Step 2: Split into chunks
        stored_chunks = chunk_text(text)

        # Step 3: Create embeddings
        embeddings = create_embeddings(stored_chunks)

        # Step 4: Create FAISS index
        faiss_index = create_index(embeddings)

        return jsonify({
            "message": "Document uploaded and indexed successfully",
            "total_chunks": len(stored_chunks)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/ask", methods=["POST"])
def ask():
    global stored_chunks, faiss_index

    try:
        if faiss_index is None:
            return jsonify({"error": "Upload a document first"}), 400

        data = request.get_json()
        question = data["question"]

        # Convert question to embedding
        query_embedding = model.encode([question])

        # Search similar chunks
        indices = search_index(faiss_index, query_embedding)

        # Get relevant text
        relevant_chunks = [stored_chunks[i] for i in indices]

        return jsonify({
            "answer": " ".join(relevant_chunks)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
