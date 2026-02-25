#!/usr/bin/env python3

import argparse

from config import DATA_PATH
from lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)
from utils.load import load_movies


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify loaded model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Verify loaded model")
    embed_text_parser.add_argument("text", type=str, help="Text to encode")

    subparsers.add_parser("verify_embeddings", help="Verify loaded model")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed a query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Embed a query")
    search_parser.add_argument("query", type=str, help="Describe the movie you want")
    search_parser.add_argument("--limit", type=int, default=5, help="Results limit")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            sem_search = SemanticSearch()
            movies = load_movies(DATA_PATH)
            sem_search.load_or_create_embeddings(movies)
            results = sem_search.search(args.query, args.limit)
            for i, res in enumerate(results, start=1):
                print(
                    f"{i}. {res.get('title')} ({res.get('score')})\n{res.get('description')[:100]}"
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
