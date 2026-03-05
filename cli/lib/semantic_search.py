#!/usr/bin/env python
import json
import re
from collections import defaultdict

import numpy as np
from config import (
    CHUNK_EMBEDDINGS_CACHE_PATH,
    CHUNK_METADATA_CACHE_PATH,
    DATA_PATH,
    EMBEDDINGS_CACHE_PATH,
)
from sentence_transformers import SentenceTransformer
from utils.load import load_movies


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings: list = None
        self.documents: list[dict] = []
        self.document_map = dict()

    def generate_embedding(self, text):
        text = text.strip()
        if not text:
            raise ValueError("Does not accept empty text.")
        return self.model.encode([text], show_progress_bar=True)[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        all_movies = []
        for doc in self.documents:
            title_description = f"{doc.get('title')} {doc.get('description')}"
            self.document_map[doc.get("id")] = title_description
            all_movies.append(title_description)

        self.embeddings = self.model.encode(all_movies, show_progress_bar=True)
        np.save(file=EMBEDDINGS_CACHE_PATH, arr=self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in self.documents:
            title_description = f"{doc.get('title')}\n{doc.get('description')}"
            self.document_map[doc.get("id")] = title_description

        if EMBEDDINGS_CACHE_PATH.is_file():
            self.embeddings = np.load(EMBEDDINGS_CACHE_PATH)

            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if len(self.embeddings) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embedding(query)
        cosine_distances: list[tuple[str, dict]] = []
        for emb, doc in zip(self.embeddings, self.documents):
            cosine_distances.append((cosine_similarity(query_embedding, emb), doc))

        cosine_distances = sorted(cosine_distances, key=lambda x: x[0], reverse=True)
        top_results = []
        for score, doc in cosine_distances[:limit]:
            top_results.append(
                dict(
                    score=score,
                    title=doc.get("title"),
                    description=doc.get("description"),
                )
            )

        return top_results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray
        self.chunk_metadata: list[dict] = []

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        all_chunks = []
        for i, doc in enumerate(self.documents):
            if not doc.get("description"):
                continue
            chunks = semantic_chunk(doc["description"], max_chunk_size=4, overlap=1)
            all_chunks += chunks
            len_chunks = len(chunks)
            for idx in range(len(chunks)):
                self.chunk_metadata.append(
                    {"movie_idx": i, "chunk_idx": idx, "total_chunks": len_chunks}
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        np.save(file=CHUNK_EMBEDDINGS_CACHE_PATH, arr=self.chunk_embeddings)
        with open(CHUNK_METADATA_CACHE_PATH, "w") as f:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            description = doc.get("description")
            self.document_map[doc.get("id")] = description

        if (
            CHUNK_METADATA_CACHE_PATH.is_file()
            and CHUNK_EMBEDDINGS_CACHE_PATH.is_file()
        ):
            with open(CHUNK_METADATA_CACHE_PATH, "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]

            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_CACHE_PATH)
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        q_embed = self.generate_embedding(query)
        chunk_score: list[dict] = []  # scores of all 72k chunks
        for cidx, embed in enumerate(self.chunk_embeddings):
            score = cosine_similarity(embed, q_embed)
            midx = self.chunk_metadata[cidx]["movie_idx"]
            chunk_score.append(
                {
                    "chunk_idx": cidx,
                    "movie_idx": midx,
                    "score": score,
                }
            )

        movie_score = defaultdict(float)  # movie_idx -> score (5000 movies)
        for cs in chunk_score:
            mov_idx = cs["movie_idx"]
            new_score = cs["score"]
            if new_score > movie_score[mov_idx]:
                movie_score[mov_idx] = new_score

        sorted_movie_score = sorted(
            movie_score.items(), key=lambda x: x[1], reverse=True
        )
        top_movies = sorted_movie_score[:limit]  # (mov_idx, score)
        top_results: list[dict] = []
        for idx, sim_score in top_movies:
            top_results.append(
                {
                    "id": idx,
                    "title": self.documents[idx].get("title"),
                    "document": self.documents[idx]["description"][:100],
                    "score": round(sim_score, 2),
                    "metadata": self.chunk_metadata[idx] or {},
                }
            )
        return top_results


def verify_model():
    sem_search = SemanticSearch()
    model = sem_search.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def embed_text(text: str):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    sem_search = SemanticSearch()
    documents = load_movies(DATA_PATH)
    embeddings = sem_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_chunk(text, max_chunk_size, overlap) -> list[str]:
    pattern = r"(?<=[.!?])\s+"
    text = text.strip()
    if text == "":
        return []
    sentences: list[str] = re.split(pattern, text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip() != ""]
    if not sentences:
        return []
    if len(sentences) == 1 and sentences[0][-1] in (".", "!", "?"):
        return sentences
    chunks: list[str] = []
    step_size = max_chunk_size - overlap

    for i in range(0, len(sentences), step_size):
        chunk_sentences = sentences[i : i + max_chunk_size]
        chunks.append(" ".join(chunk_sentences))
        if len(chunk_sentences) <= overlap:
            break
    return chunks
