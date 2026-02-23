#!/usr/bin/env python

from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


def verify_model():
    sem_search = SemanticSearch()
    model = sem_search.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")
