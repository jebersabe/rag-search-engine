#!/usr/bin/env python3

import argparse

from config import DATA_PATH
from lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    embed_query_text,
    embed_text,
    semantic_chunk,
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

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text before embedding")
    chunk_parser.add_argument("text", type=str, help="The text to be chunked")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Number of words for chunking"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Number of words overlap between chunks"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk texts with respect to semantic structure"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="The text to be chunked")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Max number of sentences per chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Number sentences overlap"
    )

    subparsers.add_parser("embed_chunks", help="Embed chunks")

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search chunks"
    )
    search_chunked_parser.add_argument("query", type=str, help="Query")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="Limit top results"
    )

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
        case "chunk":
            text_splits: list[str] = args.text.split()
            start: int = 0
            end: int = args.chunk_size
            counter: int = 1
            print(f"Chunking {len(args.text)} characters")
            while start < len(text_splits):
                if start == 0:
                    start_overlap = 0
                else:
                    start_overlap: int = start - args.overlap
                print(f"{counter}. {' '.join(text_splits[start_overlap:end])}")
                counter += 1
                start += args.chunk_size
                end += args.chunk_size
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
        case "embed_chunks":
            movies = load_movies(DATA_PATH)
            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            movies = load_movies(DATA_PATH)
            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(movies)
            results = css.search_chunks(args.query, args.limit)
            for i, res in enumerate(results, start=1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
