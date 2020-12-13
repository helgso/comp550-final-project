import pandas as pd
import transformers as trfmrs
from os.path import dirname, abspath, join
import numpy as np
import torch


"""This module is to compute the DistilBERT embedding features."""


ROOT = dirname(dirname(abspath(__file__)))
DATA = join(ROOT, "data", "scraped-lyrics-v2-preprocessed.csv")


def truncate(lyrics, strategy="middle", head=128, tail=382):
    if len(lyrics) <= 510:
        return lyrics
    accepted_strategies = ["middle", "head", "tail", "head-tail"]
    if strategy not in accepted_strategies:
        raise ValueError(f"The strategy must be {accepted_strategies}.")
    if strategy == "head":
        return lyrics[:510]
    if strategy == "tail":
        return lyrics[-510:]
    if strategy == "middle":
        return middle_tokens(lyrics)
    if strategy == "head-tail":
        if tail + head != 510:
            raise ValueError("For the head-tail truncation strategy, the head and tail must sum to 510.")
        return lyrics[:head] + lyrics[-tail:]


def middle_tokens(lyrics):
    # If length of lyrics is even
    if len(lyrics) % 2 == 0:
        right_middle = len(lyrics) // 2
        left_middle = right_middle - 1
        start = max(left_middle - 254, 0)
        end = min(right_middle + 255, len(lyrics))
    # If length of lyrics is odd
    else:
        middle = len(lyrics) // 2
        start = max(middle - 255, 0)
        end = min(middle + 255, len(lyrics))
    return lyrics[start: end]


def get_max_length(tokenized_vals):
    max = 0
    for i in tokenized_vals:
        if len(i) > max:
            max = len(i)
    return max


def get_bert_embeddings(lyrics):
    trunc_lyrics = [truncate(x) for x in lyrics]

    # Load DistilBERT tokenizer and model
    model_class, tokenizer_class, pretrained_weights = (
    trfmrs.DistilBertModel, trfmrs.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # Tokenize truncated lyrics
    tokenized = df["trunc_lyrics"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    # Pad lyrics so that they are all the same size
    max_length = get_max_length(tokenized.values)
    padded = np.array([i + [0] * (max_length - len(i)) for i in tokenized.values])

    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded).to(torch.int64)
    attention_mask = torch.tensor(attention_mask).to(torch.int64)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:, 0, :].numpy()
    return features


if __name__ == "__main__":
    """Sample usage."""
    df = pd.read_csv(DATA)
    features = get_bert_embeddings(df.lyrics.tolist())
