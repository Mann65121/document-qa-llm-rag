from collections import Counter


def create_embeddings(text_chunks):
    return [Counter(chunk.lower().split()) for chunk in text_chunks]
