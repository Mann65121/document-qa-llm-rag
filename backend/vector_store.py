import faiss
import numpy as np

def create_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search_index(index, query_embedding, top_k=2):

    if len(query_embedding.shape) == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    query_embedding = query_embedding.astype("float32")

    distances, indices = index.search(query_embedding, top_k)
    return indices[0]
