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


