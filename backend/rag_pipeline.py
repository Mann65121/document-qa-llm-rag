import json
import math
import os
import re
import unicodedata
from collections import Counter
from urllib import error, request


STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
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


def sanitize_output_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\uFFFD", " ")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_source_suffix(text):
    text = re.sub(r"\s*Sources:\s*Chunk\s+\d+(?:\s*,\s*Chunk\s+\d+)*\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def tokenize(text):
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return [word for word in words if word not in STOP_WORDS]


def split_sentences(text):
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [sanitize_output_text(sentence) for sentence in raw_sentences if sentence.strip()]
    return sentences or [sanitize_output_text(text.strip())]


def build_document_frequency(chunks):
    frequencies = Counter()
    for chunk in chunks:
        frequencies.update(set(tokenize(chunk)))
    return frequencies


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
        inverse_frequency = math.log((total_docs + 1) / (1 + doc_frequencies.get(token, 0))) + 1
        overlap += counts[token] * inverse_frequency

    phrase_bonus = text.lower().count(query.lower()) * 3.5
    question_word = query_type(query)
    hint_bonus = 0.0
    if question_word in QUESTION_HINTS:
        hint_bonus = sum(0.4 for hint in QUESTION_HINTS[question_word] if hint in text.lower())

    density = overlap / math.sqrt(len(text_tokens))
    return overlap + density + phrase_bonus + hint_bonus


def retrieve_chunks(chunks, query, top_k=4):
    doc_frequencies = build_document_frequency(chunks)
    total_docs = max(1, len(chunks))
    scored = []

    for index, chunk in enumerate(chunks):
        clean_chunk = sanitize_output_text(chunk)
        score = score_text(clean_chunk, query, doc_frequencies, total_docs)
        if score > 0:
            scored.append({"id": index + 1, "text": clean_chunk, "score": round(score, 3)})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def select_best_sentences(retrieved_chunks, query, max_sentences=5):
    sentence_pool = []
    all_sentences = []

    for chunk in retrieved_chunks:
        all_sentences.extend(split_sentences(chunk["text"]))

    doc_frequencies = build_document_frequency(all_sentences or [chunk["text"] for chunk in retrieved_chunks])
    total_docs = max(1, len(all_sentences))

    for chunk in retrieved_chunks:
        for sentence in split_sentences(chunk["text"]):
            score = score_text(sentence, query, doc_frequencies, total_docs)
            if score > 0:
                sentence_pool.append((score, sentence, chunk["id"]))

    sentence_pool.sort(key=lambda item: item[0], reverse=True)

    selected = []
    seen = set()
    for score, sentence, chunk_id in sentence_pool:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append({"text": sentence, "score": round(score, 3), "chunk_id": chunk_id})
        if len(selected) == max_sentences:
            break

    return selected


def detect_focus_terms(query, evidence_sentences):
    query_terms = tokenize(query)
    if query_terms:
        return query_terms[:6]

    combined = " ".join(sentence["text"] for sentence in evidence_sentences)
    return tokenize(combined)[:6]


def build_precise_answer(query, evidence_sentences):
    if not evidence_sentences:
        return "I could not find enough evidence in the uploaded document to answer that precisely."

    focus_terms = detect_focus_terms(query, evidence_sentences)
    filtered = []

    for sentence in evidence_sentences:
        text = sentence["text"]
        if any(term in text.lower() for term in focus_terms):
            filtered.append(text)

    if not filtered:
        filtered = [sentence["text"] for sentence in evidence_sentences[:3]]

    question_word = query_type(query)

    if question_word in {"what", "who", "when", "where"}:
        primary = filtered[0]
        if len(filtered) > 1 and len(primary) < 180:
            return sanitize_output_text(f"{primary} {filtered[1]}")
        return sanitize_output_text(primary)

    if question_word == "why":
        return sanitize_output_text(" ".join(filtered[:2]))

    if question_word == "how":
        return sanitize_output_text(f"{filtered[0]} {' '.join(filtered[1:2])}".strip())

    summary = " ".join(filtered[:2])
    if len(summary) > 360:
        summary = summary[:357].rsplit(" ", 1)[0] + "..."
    return sanitize_output_text(summary)


def build_generation_metadata(mode, evidence_sentences):
    evidence_strength = "high" if len(evidence_sentences) >= 3 else "medium" if len(evidence_sentences) == 2 else "low"
    return {
        "mode": mode,
        "evidence_strength": evidence_strength,
    }


def build_context_block(retrieved_chunks):
    context_parts = []
    for item in retrieved_chunks:
        context_parts.append(f"[Chunk {item['id']}] {item['text']}")
    return "\n\n".join(context_parts)


def build_generation_prompt(query, retrieved_chunks):
    context_block = build_context_block(retrieved_chunks)
    return f"""You are answering from a PDF document.

Rules:
- Use only the provided context.
- Answer in 2 to 4 concise sentences.
- Be precise and direct.
- If the answer is not fully supported, say so clearly.
- End with: Sources: Chunk X, Chunk Y

Question:
{query}

Context:
{context_block}
"""


def generate_with_ollama(query, retrieved_chunks):
    payload = {
        "model": DEFAULT_OLLAMA_MODEL,
        "prompt": build_generation_prompt(query, retrieved_chunks),
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 180,
        },
    }

    http_request = request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(http_request, timeout=25) as response:
        data = json.loads(response.read().decode("utf-8"))

    answer = sanitize_output_text((data.get("response") or "").strip())
    answer = remove_source_suffix(answer)
    if not answer:
        raise ValueError("Empty response from Ollama")
    return answer


def try_generate_answer(query, retrieved_chunks, evidence_sentences):
    try:
        answer = generate_with_ollama(query, retrieved_chunks)
        return {
            "answer": answer,
            "generation": build_generation_metadata(f"ollama:{DEFAULT_OLLAMA_MODEL}", evidence_sentences),
        }
    except (error.URLError, TimeoutError, ValueError, OSError):
        return {
            "answer": build_precise_answer(query, evidence_sentences),
            "generation": build_generation_metadata("local-grounded-generative", evidence_sentences),
        }


def answer_question(chunks, query):
    retrieved = retrieve_chunks(chunks, query, top_k=4)
    if not retrieved:
        return {
            "answer": "I could not find a strong match for that question in the uploaded document.",
            "sources": [],
            "generation": {
                "mode": "local-grounded-generative",
                "evidence_strength": "low",
            },
            "confidence": "low",
        }

    evidence_sentences = select_best_sentences(retrieved, query)
    generated = try_generate_answer(query, retrieved, evidence_sentences)

    sources = [
        {
            "chunk_id": item["id"],
            "score": item["score"],
            "snippet": item["text"][:260].strip(),
        }
        for item in retrieved
    ]

    top_score = sources[0]["score"] if sources else 0
    if top_score >= 8:
        confidence = "high"
    elif top_score >= 4:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "answer": remove_source_suffix(sanitize_output_text(generated["answer"])),
        "sources": sources,
        "generation": generated["generation"],
        "confidence": confidence,
    }
