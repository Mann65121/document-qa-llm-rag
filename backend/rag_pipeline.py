def retrieve_chunks(chunks, query, top_k=2):
    scored_chunks = []

    for chunk in chunks:
        score = chunk.lower().count(query.lower())
        scored_chunks.append((score, chunk))

    # Sort by score descending
    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    return [chunk for score, chunk in scored_chunks[:top_k] if score > 0]
