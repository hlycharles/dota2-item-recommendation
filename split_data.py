'''
split examples into train, dev, test sets
'''

import os
from multiprocessing import Pool
import random
import sys
from utils import data_utils

FOLDER = './examples'
OUT_FOLDER = 'dataset'

def split_data():
    match_files = os.listdir(FOLDER)
    match_files = filter(data_utils.is_valid_match_file, match_files)
    random.shuffle(match_files)
    dev_files = match_files[:1000]
    test_files = match_files[1000:2000]
    train_files = match_files[2000:]
    dev_files = map(lambda x: (OUT_FOLDER + '/dev/' + x, FOLDER + '/' + x), dev_files)
    test_files = map(lambda x: (OUT_FOLDER + '/test/' + x, FOLDER + '/' + x), test_files)
    train_files = map(lambda x: (OUT_FOLDER + '/train/' + x, FOLDER + '/' + x), train_files)
    pool = Pool()
    pool.map(data_utils.store_match_file, dev_files)
    pool.map(data_utils.store_match_file, test_files)
    pool.map(data_utils.store_match_file, train_files)
    pool.close()
    pool.join()

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        FOLDER = sys.argv[1]
    if (len(sys.argv) > 2):
        OUT_FOLDER = sys.argv[2]
    split_data()
