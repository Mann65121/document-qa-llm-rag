from rag_pipeline import score_chunk


def create_index(chunks):
    return list(chunks)


def search_index(index, query, top_k=3):
    ranked = []

    for position, chunk in enumerate(index):
        score = score_chunk(chunk, query)
        if score > 0:
            ranked.append((score, position))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [position for _, position in ranked[:top_k]]
