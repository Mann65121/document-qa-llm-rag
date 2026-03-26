import json
import math
import os
import re
from collections import Counter
from urllib import error, request

STOP_WORDS = {
    "a","about","an","and","are","as","at","be","by","for","from","how",
    "in","into","is","it","its","of","on","or","that","the","their",
    "this","to","was","what","when","where","which","who","why","with",
}

QUESTION_HINTS = {
    "what": {"is", "are", "means", "defined", "refers"},
    "who": {"name", "person", "team", "member", "leader"},
    "when": {"date", "year", "month", "day", "time"},
    "where": {"location", "place", "address", "region"},
    "why": {"because", "reason", "due", "therefore"},
    "how": {"process", "steps", "method", "approach"},
}

DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")

def tokenize(text):
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return [word for word in words if word not in STOP_WORDS]

def split_sentences(text):
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()] or [text.strip()]

def build_document_frequency(chunks):
    freq = Counter()
    for chunk in chunks:
        freq.update(set(tokenize(chunk)))
    return freq

def query_type(query):
    words = re.findall(r"[A-Za-z]+", query.lower())
    return words[0] if words else ""

def score_text(text, query, doc_frequencies, total_docs):
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    text_tokens = tokenize(text)
    if not text_tokens:
        return 0.0

    counts = Counter(text_tokens)
    token_set = set(text_tokens)
    overlap = 0.0

    for token in query_tokens:
        if token not in token_set:
            continue
        idf = math.log((total_docs + 1) / (1 + doc_frequencies.get(token, 0))) + 1
        overlap += counts[token] * idf

    phrase_bonus = text.lower().count(query.lower()) * 3.5

    question_word = query_type(query)
    hint_bonus = 0.0
    if question_word in QUESTION_HINTS:
        hint_bonus = sum(0.4 for h in QUESTION_HINTS[question_word] if h in text.lower())

    density = overlap / math.sqrt(len(text_tokens))
    length_penalty = 1 / (1 + len(text_tokens) * 0.01)

    return (overlap + density + phrase_bonus + hint_bonus) * length_penalty

def retrieve_chunks(chunks, query, top_k=4):
    doc_freq = build_document_frequency(chunks)
    total_docs = max(1, len(chunks))
    scored = []

    for i, chunk in enumerate(chunks):
        score = score_text(chunk, query, doc_freq, total_docs)
        if score > 0:
            scored.append({"id": i + 1, "text": chunk, "score": round(score, 3)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def select_best_sentences(retrieved_chunks, query, max_sentences=5):
    pool = []
    all_sentences = []

    for chunk in retrieved_chunks:
        all_sentences.extend(split_sentences(chunk["text"]))

    doc_freq = build_document_frequency(all_sentences or [c["text"] for c in retrieved_chunks])
    total_docs = max(1, len(all_sentences))

    for chunk in retrieved_chunks:
        for sentence in split_sentences(chunk["text"]):
            if len(sentence.split()) < 5:
                continue

            score = score_text(sentence, query, doc_freq, total_docs)

            if score > 0:
                pool.append((score, sentence, chunk["id"]))

    pool.sort(reverse=True, key=lambda x: x[0])

    selected = []
    seen = set()

    for score, sentence, chunk_id in pool:
        norm = sentence.lower()
        if norm in seen:
            continue
        seen.add(norm)
        selected.append({"text": sentence, "score": round(score, 3), "chunk_id": chunk_id})

        if len(selected) == max_sentences:
            break

    return selected

def build_precise_answer(query, sentences):
    if not sentences:
        return "No relevant answer found in document."

    qtype = query_type(query)
    texts = [s["text"] for s in sentences]

    if qtype in {"what","who","when","where"}:
        return texts[0]

    return " ".join(texts[:2])[:300]

def answer_question(chunks, query):
    retrieved = retrieve_chunks(chunks, query)

    if not retrieved:
        return {
            "answer": "No relevant information found.",
            "confidence": "low",
            "sources": []
        }

    sentences = select_best_sentences(retrieved, query)
    answer = build_precise_answer(query, sentences)

    top_score = retrieved[0]["score"]

    if top_score > 8:
        confidence = "high"
    elif top_score > 4:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": retrieved
    }