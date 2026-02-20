#!/usr/bin/env python3
import argparse
import string

from config import DATA_PATH, STOPWORDS_PATH
from indexing.inverted_index import InvertedIndex
from nltk.stem import PorterStemmer
from utils.load import load_movies, load_stopwords


def search(movies: list, query: str) -> set[str]:
    match_titles: list[str] = []
    translator: dict = str.maketrans("", "", string.punctuation)

    stemmer = PorterStemmer()
    stopwords: list[str] = load_stopwords(STOPWORDS_PATH)
    query: str = query.lower().translate(translator)
    query_ls: list[str] = query.split()
    query_ls_clean: list[str] = [q for q in query_ls if q not in stopwords]

    titles: list[str] = [movie.get("title") for movie in movies]
    for title in titles:
        if any(
            stemmer.stem(q) in stemmer.stem(word)
            for q in query_ls_clean
            for word in title.lower().translate(translator).split()
            if word not in stopwords
        ):
            match_titles.append(title)

    return set(match_titles)


def search_index(query: str, inverted_index: InvertedIndex):
    translator: dict = str.maketrans("", "", string.punctuation)

    PorterStemmer()
    stopwords: list[str] = load_stopwords(STOPWORDS_PATH)
    query: str = query.lower().translate(translator)
    query_ls: list[str] = query.split()
    tokens: list[str] = [q for q in query_ls if q not in stopwords]

    inverted_index.load()
    results: list[str] = []
    for token in tokens:
        doc_ids: int = inverted_index.get_documents(token)
        if len(doc_ids) > 5:
            results = doc_ids[:5]
            break
        results = results + doc_ids
        if len(results) > 5:
            results = results + doc_ids
            break

    movies = load_movies(DATA_PATH)
    for movie in movies:
        if movie.get("id") in results:
            print(f"Title: {movie.get('title')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index and save to disk")

    tf_parser = subparsers.add_parser("tf", help="Term frequency")
    tf_parser.add_argument("docid", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    args = parser.parse_args()

    inverted_index = InvertedIndex()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            inverted_index.load()
            load_movies(DATA_PATH)
            search_index(args.query, inverted_index)
        case "build":
            inverted_index.build()
            inverted_index.save()
        case "tf":
            inverted_index.load()
            freq = inverted_index.get_tf(args.docid, args.term)
            print(f"Freq of the word {args.term} in document ID {args.docid}: {freq}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
