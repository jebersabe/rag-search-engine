#!/usr/bin/env python3
import argparse
import string

from nltk.stem import PorterStemmer
from utils.load import load_and_prep_data, load_stopwords


def search(movies: list, query: str) -> list[str]:
    match_titles: list[str] = []
    translator: dict = str.maketrans("", "", string.punctuation)

    stemmer = PorterStemmer()
    stopwords = load_stopwords("data/stopwords.txt")
    query: str = query.lower().translate(translator)
    query_ls: list[str] = query.split()
    query_ls_clean = [q for q in query_ls if q and q not in stopwords]

    titles: list[str] = [movie.get("title") for movie in movies]

    for title in titles:
        if any(
            stemmer.stem(q) in stemmer.stem(word)
            for q in query_ls_clean
            for word in title.lower().translate(translator).split()
            if word not in stopwords
        ):
            match_titles.append(title)

    return match_titles


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies = load_and_prep_data("data/movies.json")
            results = search(movies, args.query)
            if results:
                for i, result in enumerate(results, start=1):
                    if i <= 5:
                        print(f"{i}. {result}")
            else:
                print("No match found!")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
