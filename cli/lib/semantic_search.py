#!/usr/bin/env python
import numpy as np
from config import DATA_PATH, EMBEDDINGS_CACHE_PATH
from sentence_transformers import SentenceTransformer
from utils.load import load_movies


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
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
            title_description = f"{doc.get('title')}\n{doc.get('description')}"
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
