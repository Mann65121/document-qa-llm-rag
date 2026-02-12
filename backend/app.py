from flask import Flask, request, jsonify
from document_loader import load_document

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
