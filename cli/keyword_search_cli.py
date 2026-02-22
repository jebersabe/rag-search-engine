#!/usr/bin/env python3
import argparse
import math
import string

from config import BM25_B, BM25_K1, DATA_PATH, STOPWORDS_PATH
from indexing.inverted_index import InvertedIndex, tokenize
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


def bm25_idf_command(term: str, inverted_index: InvertedIndex) -> float:
    if len(term.split()) > 1:
        raise ValueError(f"Multiple terms submitted: {term.split()}")
    token: str = tokenize(term)[0]
    inverted_index.load()
    bm25_score: float = inverted_index.get_bm25_idf(token)
    return bm25_score


def bm25_tf_command(
    doc_id: int, term: str, inverted_index: InvertedIndex, k1=BM25_K1, b=BM25_B
):
    inverted_index.load()
    bm25_tf_score = inverted_index.get_bm25_tf(doc_id, term, k1, b)
    return bm25_tf_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index and save to disk")

    tf_parser = subparsers.add_parser("tf", help="Term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Term frequency x inverse document frequency"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Limit results")

    args = parser.parse_args()

    inverted_index = InvertedIndex()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            inverted_index.load()
            search_index(args.query, inverted_index)
        case "build":
            inverted_index.build()
            inverted_index.save()
        case "tf":
            inverted_index.load()
            freq = inverted_index.get_tf(args.doc_id, args.term)
            print(f"Freq of the word {args.term} in document ID {args.doc_id}: {freq}")
        case "idf":
            inverted_index.load()
            token = tokenize(args.term)[0]
            total_doc_count: int = len(inverted_index.docmap.keys())
            term_match_doc_count: int = len(inverted_index.index[token])
            idf_value: float = math.log(
                (total_doc_count + 1) / (term_match_doc_count + 1)
            )
            print(f"Inverse document frequency of '{args.term}': {idf_value:.2f}")
        case "tfidf":
            inverted_index.load()
            token: str = tokenize(args.term)[0]
            tf: int = inverted_index.get_tf(args.doc_id, token)

            total_doc_count: int = len(inverted_index.docmap.keys())
            term_match_doc_count: int = len(inverted_index.index[token])
            idf: float = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            tf_idf: float = tf * idf
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            bm25_score = bm25_idf_command(term=args.term, inverted_index=inverted_index)
            print(f"BM25 IDF score of '{args.term}': {bm25_score:.2f}")
        case "bm25tf":
            bm25_tf_score = bm25_tf_command(
                args.doc_id, args.term, inverted_index, args.k1, args.b
            )
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf_score:.2f}"
            )
        case "bm25search":
            inverted_index.load()
            results = inverted_index.bm25_search(args.query, args.limit)
            load_movies(DATA_PATH)
            for doc_id, score in results:
                title = inverted_index.docmap[doc_id].split("\n")[0]
                print(f"({doc_id}) {title} - Score: {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
