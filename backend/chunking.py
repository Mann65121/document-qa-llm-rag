def chunk_text(text, chunk_size=120, overlap=30):
    words = text.split()
    if not words:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    step = max(1, chunk_size - overlap)
    chunks = []

    for index in range(0, len(words), step):
        chunk_words = words[index : index + chunk_size]
        if not chunk_words:
            continue

        chunk = " ".join(chunk_words).strip()
        if chunk and (not chunks or chunk != chunks[-1]):
            chunks.append(chunk)

    return chunks
