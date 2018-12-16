'''
read all examples and present as a tuple of array of features and
corresponding labels
'''

import os
from multiprocessing import Pool
import numpy as np
from utils import data_utils

def read_data(path):
    match_files = os.listdir(path)
    match_files = filter(data_utils.is_valid_match_file, match_files)
    match_files = map(lambda x: path + '/' + x, match_files)
    pool = Pool()
    examples = pool.map(data_utils.read_match_features, match_files)
    pool.close()
    pool.join()
    examples = reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]), examples)
    features = np.transpose(np.array(examples[0]))
    labels_arr = examples[1]
    m = len(labels_arr)
    labels = np.transpose(np.array(labels_arr).reshape(m, 1))
    return (features, labels)
