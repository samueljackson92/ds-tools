import pickle
import yaml

from pathlib import Path

def load_pickle(path):
    with Path(path).open('rb') as handle:
        return pickle.load(handle)

def load_yaml(path):
    with open(path, 'r') as handle:
        return yaml.load(handle)
