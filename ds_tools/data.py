import pickle
from pathlib import Path

def load_pickle(path):
    with Path(path).open('rb') as handle:
        return pickle.load(handle)
