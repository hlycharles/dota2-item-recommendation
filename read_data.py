import json
import os
from multiprocessing import Pool
import numpy as np

def is_valid_match_file(match_file):
    if (os.path.isdir(match_file)):
        return False
    return match_file.endswith(".json")

def read_match_features(match_file):
    data = None
    with open( '../examples/' + match_file, 'r') as f:
        data = json.load(f)
    examples = data['examples']
    features = []
    labels = []
    for example in examples:
        features.append(example['x'])
        labels.append(example['y'])
    return (features, labels)

def read_data(path):
    match_files = os.listdir(path)
    match_files = filter(is_valid_match_file, match_files)
    pool = Pool()
    examples = pool.map(read_match_features, match_files)
    pool.close()
    pool.join()
    examples = reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]), examples)
    features = np.transpose(np.array(examples[0]))
    labels_arr = examples[1]
    m = len(labels_arr)
    labels = np.transpose(np.array(labels_arr).reshape(m, 1))
    print features.shape
    print labels.shape
    return (features, labels)


if __name__ == "__main__":
    read_data('../examples')
