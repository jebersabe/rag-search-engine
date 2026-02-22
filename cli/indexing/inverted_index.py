#!/usr/bin/env python
import math
import pickle
import string
from collections import Counter, defaultdict

from config import (
    BM25_B,
    BM25_K1,
    CACHE_PATH,
    DATA_PATH,
    DOC_LENGTHS_CACHE_PATH,
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
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str):
        tokens: list[str] = tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.term_frequency[doc_id] = Counter(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return float(0)
        return float(sum(self.doc_lengths.values()) / len(self.doc_lengths))

    def get_documents(self, term) -> list[int]:
        doc_ids: set[int] = self.index.get(term.lower(), set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id, term) -> int:
        token = tokenize(term)
        if len(token) > 1:
            raise ValueError(f"Method only accepts 1 token, was given {len(token)}")
        freq = self.term_frequency[doc_id][token[0]]
        return freq

    def get_bm25_idf(self, term) -> float:
        if len(term.split()) > 1:
            raise ValueError(f"Method accepts only 1 term. Was given {term.split()}")
        token: str = tokenize(term)[0]
        total_doc_count: int = len(self.docmap.keys())
        term_match_doc_count: int = len(self.index[token])

        bm25_idf = math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )
        return bm25_idf

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        if len(term.split()) > 1:
            raise ValueError(f"Method accepts only 1 term. Was given {term.split()}")
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )
        token: str = tokenize(term)[0]
        tf = self.get_tf(doc_id, token)
        bm25_tf_score = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf_score

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_score: float = self.get_bm25_idf(term) * self.get_bm25_tf(doc_id, term)
        return bm25_score

    def bm25_search(self, query, limit) -> list:
        tokens: list[str] = tokenize(query)
        scores = defaultdict(float)  # doc_id -> bm25 score
        for token in tokens:
            for doc_id in self.index[token]:
                scores[doc_id] += self.bm25(doc_id, token)

        sorted_scores: list[tuple] = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )
        return sorted_scores[:limit]

    def build(self):
        movies: list[dict] = load_movies(str(DATA_PATH))
        for movie in movies:
            doc_id: int = int(movie.get("id"))
            self.__add_document(
                doc_id=doc_id, text=f"{movie['title']} {movie['description']}"
            )
            self.docmap[doc_id] = f"{movie['title']}\n{movie['description']}"

    def save(self):
        CACHE_PATH.mkdir(parents=True, exist_ok=True)

        with open(INDEX_CACHE_PATH, "wb") as f:
            pickle.dump(self.index, f)

        with open(DOCMAP_CACHE_PATH, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(TF_CACHE_PATH, "wb") as f:
            pickle.dump(self.term_frequency, f)

        with open(DOC_LENGTHS_CACHE_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        try:
            with open(INDEX_CACHE_PATH, "rb") as file:
                self.index = pickle.load(file)

            with open(DOCMAP_CACHE_PATH, "rb") as file:
                self.docmap = pickle.load(file)

            with open(TF_CACHE_PATH, "rb") as file:
                self.term_frequency = pickle.load(file)

            with open(DOC_LENGTHS_CACHE_PATH, "rb") as file:
                self.doc_lengths = pickle.load(file)

        except FileNotFoundError as err:
            raise FileNotFoundError(f"No cache data found: {err}")
