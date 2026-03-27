import re

from pypdf import PdfReader


def load_document(file_storage):
    stream = getattr(file_storage, "stream", file_storage)
    if hasattr(stream, "seek"):
        stream.seek(0)

    reader = PdfReader(stream)
    pages = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)

    text = "\n".join(pages)
    text = re.sub(r"\s+", " ", text).strip()
    return text
