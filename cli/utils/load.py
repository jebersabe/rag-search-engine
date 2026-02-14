#!/usr/bin/env python
import json


def load_and_prep_data(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data: str = f.read()

    data_dict: dict[str, list] = json.loads(data)
    movies: list[dict] = data_dict.get("movies", [])

    return movies


def load_stopwords(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        data: str = f.read()
        stopwords: list[str] = data.splitlines()

    return stopwords
