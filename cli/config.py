#!/usr/bin/env python

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_DIR / "cache"
DATA_PATH = PROJECT_DIR / "data/movies.json"
STOPWORDS_PATH = PROJECT_DIR / "data/stopwords.txt"
INDEX_CACHE_PATH = PROJECT_DIR / "cache/index.pkl"
DOCMAP_CACHE_PATH = PROJECT_DIR / "cache/docmap.pkl"
TF_CACHE_PATH = PROJECT_DIR / "cache/term_frequency.pkl"
DOC_LENGTHS_CACHE_PATH = PROJECT_DIR / "cache/doc_lengths.pkl"
EMBEDDINGS_CACHE_PATH = PROJECT_DIR / "cache/movie_embeddings.npy"
BM25_K1 = 1.5
BM25_B = 0.75
