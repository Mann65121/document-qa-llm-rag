import io
import re
import unicodedata

from pypdf import PdfReader

try:
    import fitz
except ImportError:
    fitz = None


def sanitize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\uFFFD", " ")
    text = text.replace("\u00A0", " ")
    text = text.replace("\u2022", " ")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_extracted_text(text):
    if not text:
        return -1

    # Penalize suspicious broken words like "descrip on" or "mul ple".
    broken_word_patterns = len(re.findall(r"\b[a-zA-Z]{3,}\s[a-zA-Z]{1,2}\b", text))
    alpha_count = len(re.findall(r"[A-Za-z]", text))
    return alpha_count - (broken_word_patterns * 10)


def extract_with_pymupdf(file_bytes):
    if fitz is None:
        return ""

    document = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page in document:
        page_text = page.get_text("text") or ""
        page_text = sanitize_text(page_text)
        if page_text:
            pages.append(page_text)
    return " ".join(pages).strip()


def extract_with_pypdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = sanitize_text(page_text)
        if page_text:
            pages.append(page_text)

    return " ".join(pages).strip()


def load_document(file_storage):
    stream = getattr(file_storage, "stream", file_storage)
    if hasattr(stream, "seek"):
        stream.seek(0)

    file_bytes = stream.read()
    if hasattr(stream, "seek"):
        stream.seek(0)

    pymupdf_text = extract_with_pymupdf(file_bytes)
    pypdf_text = extract_with_pypdf(file_bytes)

    if score_extracted_text(pymupdf_text) >= score_extracted_text(pypdf_text):
        return pymupdf_text
    return pypdf_text
