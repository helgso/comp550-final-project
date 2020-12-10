from os.path import dirname, abspath, join

import pandas as pd
import pickle
import numpy as np


ROOT = dirname(dirname(abspath(__file__)))
DATA = join(ROOT, "data", "scraped-lyrics-v2-preprocessed.csv")
EMBEDDINGS = join(ROOT, "glove")
RESULTS = join(ROOT, "results")


def load_word_embeddings(dim):
    with open(join(EMBEDDINGS, f"glove.6B.{dim}d.pickle"), "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


class MeanEmbeddingVectorizer(object):
    """Extract mean embedding feature vectors,"""
    def __init__(self, embedding_dim=50):
        """
        Initializes the MeanEmbeddingVectorizer.
            Parameters:
                word_embeddings (dict): Containing mapping of words to their embeddings.
        """
        self.word_embeddings = load_word_embeddings(embedding_dim)
        self.dim = len(self.word_embeddings)

    def fit(self, X, y):
        return self

    def transform(self, X):
        """
        Transforms the input data X into mean embedding feature vectors.
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
    df = pd.read_csv(DATA)
    df["lyrics"] = df["lyrics"].apply(lambda x: x.split())
    lyrics = df.lyrics
    mev = MeanEmbeddingVectorizer()
    embeddings = mev.transform(lyrics)




