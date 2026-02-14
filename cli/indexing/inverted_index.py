#!/usr/bin/env python

import string

from nltk.stem import PorterStemmer

from cli.utils.load import load_stopwords

stopwords = load_stopwords("data/stopwords.txt")


class InvertedIndex:
    def __init__(self, index, docmap) -> None:
        self.index: dict[str, list[int]] = index
        self.docmap: dict = docmap

    def __add_document(self, doc_id, text: str):
        translator: dict = str.maketrans("", "", string.punctuation)
        stemmer = PorterStemmer()
        text_splits: list[str] = text.lower().translate(translator).split()
        tokens: list[str] = [
            stemmer.stem(t) for t in text_splits if t and t not in stopwords
        ]

        for token in tokens:
            self.index[token] = doc_id
