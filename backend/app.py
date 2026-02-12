from flask import Flask, request, jsonify
from document_loader import load_document
from embeddings import create_embeddings, model
from vector_store import create_index, search_index

faiss_index = None

embeddings = create_embeddings(stored_chunks)
faiss_index = create_index(embeddings)

query_embedding = model.encode([question])
indices = search_index(faiss_index, query_embedding)


relevant_chunks = [stored_chunks[i] for i in indices]

app = Flask(__name__)

@app.route("/")
def home():
    return "Document QA Backend Running"

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    text = load_document(file)
    return jsonify({"message": "File uploaded", "length": len(text)})

if __name__ == "__main__":
    app.run(debug=True)
