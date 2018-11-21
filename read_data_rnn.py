'''
read all examples and present as a tuple of array of features and
corresponding labels
'''

import os
from multiprocessing import Pool
import numpy as np
from utils import data_utils
import json

def read_match_features(match_file):
    data = None
    with open(match_file, "r") as f:
        data = json.load(f)
    initials = data["init"]
    features = []
    labels = []
    for p in data["steps"]:
        features.append(p["x"])
        labels.append(p["y"])
    return (initials, features, labels)

def read_data(match_files):
    match_files = filter(data_utils.is_valid_match_file, match_files)
    pool = Pool()
    examples = pool.map(read_match_features, match_files)
    pool.close()
    pool.join()
    examples = reduce(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]), examples)
    initials = examples[0]
    features = np.array(examples[1])
    labels_arr = examples[2]
    m = len(labels_arr)
    labels = np.array(labels_arr).reshape(m, 1)
    n_x = len(initials[0])
    initials = map(lambda init: np.array(init).reshape(n_x, 1).transpose(), initials)
    return (initials, features, labels)

# if __name__ == "__main__":
#     match_files = []
#     match_files.append(os.path.join("data", "dataset_rnn", "dev", "3429340260.json"))
#     match_files.append(os.path.join("data", "dataset_rnn", "dev", "3432267900.json"))
#     read_data(match_files)
