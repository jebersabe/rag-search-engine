#!/usr/bin/env python

import pickle
import string
from collections import Counter, defaultdict

from config import (
    CACHE_PATH,
    DATA_PATH,
    DOCMAP_CACHE_PATH,
    INDEX_CACHE_PATH,
    STOPWORDS_PATH,
    TF_CACHE_PATH,
)
from nltk.stem import PorterStemmer
from utils.load import load_movies, load_stopwords

stopwords = load_stopwords(str(STOPWORDS_PATH))


def tokenize(text) -> list[str]:
    translator: dict = str.maketrans("", "", string.punctuation)
    stemmer = PorterStemmer()
    text_splits: list[str] = text.lower().translate(translator).split()
    tokens: list[str] = [
        stemmer.stem(t) for t in text_splits if t and t not in stopwords
    ]
    return tokens


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, str] = {}
        self.term_frequency: dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str):
        tokens: list[str] = tokenize(text)
        self.term_frequency[doc_id] = Counter(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def get_documents(self, term) -> list[int]:
        doc_ids: set[int] = self.index.get(term.lower(), set())
        return sorted(list(doc_ids))

    def build(self):
        movies: list[dict] = load_movies(str(DATA_PATH))
        for movie in movies:
            doc_id: int = int(movie.get("id"))
            self.__add_document(
                doc_id=doc_id, text=f"{movie['title']} {movie['description']}"
            )
            self.docmap[doc_id] = f"{movie['title']} {movie['description']}"

    def get_tf(self, doc_id, term) -> int:
        token = tokenize(term)
        if len(token) > 1:
            raise ValueError(f"Method only accepts 1 token, was given {len(token)}")
        freq = self.term_frequency[doc_id][token[0]]
        return freq

    def save(self):
        CACHE_PATH.mkdir(parents=True, exist_ok=True)

        with open(INDEX_CACHE_PATH, "wb") as f:
            pickle.dump(self.index, f)

        with open(DOCMAP_CACHE_PATH, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(TF_CACHE_PATH, "wb") as f:
            pickle.dump(self.term_frequency, f)

    def load(self):
        try:
            with open(INDEX_CACHE_PATH, "rb") as file:
                self.index = pickle.load(file)

            with open(DOCMAP_CACHE_PATH, "rb") as file:
                self.docmap = pickle.load(file)

            with open(TF_CACHE_PATH, "rb") as file:
                self.term_frequency = pickle.load(file)

        except FileNotFoundError as err:
            raise FileNotFoundError(f"No cache data found: {err}")
