from os.path import dirname, abspath, join

import pandas as pd
import pickle5 as pickle
import numpy as np

"""This module is to compute the mean glove embedding feature vectors.
   To use it you must have the pickled glove embeddings stored in a "glove" directory."""

ROOT = dirname(dirname(abspath(__file__)))
DATA = join(ROOT, "data", "scraped-lyrics-v2-preprocessed.csv")
EMBEDDINGS = join(ROOT, "glove")


def load_word_embeddings(dim):
    """Load pickled pretrained glove embeddings with specified dimension."""
    with open(join(EMBEDDINGS, f"glove.6B.{dim}d.pickle"), "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


class MeanEmbeddingVectorizer(object):
    """Custom vectorizer to extract the mean glove embedding vector of the inputted lyrics."""
    def __init__(self, embedding_dim=50):
        """
        Initializes the MeanEmbeddingVectorizer.
            Parameters:
                embedding_dim (int): The desire pretrained glove embedding dimension (e.g, 50, 100, 200, 300).
        """
        self.word_embeddings = load_word_embeddings(embedding_dim)
        self.dim = len(self.word_embeddings)

    def fit(self, X, y):
        return self

    def transform(self, X):
        """
        Transforms the list of input lyrcs X into a list of mean glove embedding feature vectors.
            Parameters:
                X (list): Containing preprocesed lyrics.
        """
        features = []
        for lyrics in X:
            embeddings = [self.word_embeddings[w] for w in lyrics if w in self.word_embeddings]
            mean_embedding = np.mean(embeddings or [np.zeros(self.dim)], axis=0)
            features.append(mean_embedding)
        return np.array(features)


if __name__ == "__main__":
    """Sample usage"""
    df = pd.read_csv(DATA)
    lyrics = df["lyrics"]
    mev = MeanEmbeddingVectorizer()
    embedded_lyrics = mev.transform(lyrics)
    print(embedded_lyrics)
