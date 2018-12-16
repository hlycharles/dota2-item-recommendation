'''
recurrent neural network with LSTM / GRU that predicts the end result of DOTA2 game
'''

import numpy as np
from keras.models import load_model, Model
from keras.layers import Dense, LSTM, Input, Lambda, Reshape, GRU
from keras.optimizers import Adam
from read_data_rnn import read_data
import random
import sys
import os
import json

NUM_EPOCH = 100
ID = "1"
n_a = 64
n_x = 408
infolder = "dataset_rnn"
Tx = 50
PRINT_COST = True

filenames = []
LSTMCell = LSTM(n_a, return_state = True)
GRUCell = GRU(n_a, return_state = False)
reshapor = Reshape((1, n_x))
initialDensor = Dense(n_a, input_shape=(n_x,))
outDensor = Dense(1, activation='sigmoid', input_shape=(n_a,))

def dprint(s):
    if (PRINT_COST):
        print s

def dirmodelLSTM(Tx):
    X = Input(shape=(Tx, n_x))
    aPre = Input(shape=(n_x,), name="a_pre")
    c0 = Input(shape=(n_a,), name="c0")
    a0 = initialDensor(aPre)
    a = a0
    c = c0

    outputs = []

    for t in range(Tx):
        x = Lambda(lambda x: x[:, t, :])(X)
        x = reshapor(x)
        a, _, c = LSTMCell(x, initial_state=[a, c])
        out = outDensor(a)
        outputs.append(out)

    model = Model(inputs = [X, aPre, c0], outputs = outputs)
    return model

def dirmodelGRU(Tx):
    X = Input(shape=(Tx, n_x))
    aPre = Input(shape=(n_x,), name="a_pre")
    a0 = initialDensor(aPre)
    a = a0

    outputs = []

    for t in range(Tx):
        x = Lambda(lambda x: x[:, t, :])(X)
        x = reshapor(x)
        a = GRUCell(x, initial_state = a)
        out = outDensor(a)
        outputs.append(out)

    model = Model(inputs = [X, aPre], outputs = outputs)
    return model

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        NUM_EPOCH = int(sys.argv[1])
    trainFileCount = -1
    if (len(sys.argv) > 2):
        trainFileCount = int(sys.argv[2])
    if (len(sys.argv) > 3):
        ID = sys.argv[3]
    if (len(sys.argv) > 4):
        infolder = sys.argv[4]
    if (len(sys.argv) > 5):
        ModelOpt = sys.argv[5]
    if (len(sys.argv) > 6):
        Tx = int(sys.argv[6])
    # read training data
    training_folder = os.path.join("data", infolder, "train")
    filenames = os.listdir(training_folder)
    filenames = map(lambda f: os.path.join("data", infolder, "train", f), filenames)
    random.shuffle(filenames)
    trainFiles = filenames
    if (trainFileCount > 0):
        trainFiles = trainFiles[:trainFileCount]
    initials, features, labels = read_data(trainFiles)

    dev_folder = os.path.join("data", infolder, "dev")
    dev_files = os.listdir(dev_folder)
    dev_files = map(lambda f: os.path.join("data", infolder, "dev", f), dev_files)
    random.shuffle(dev_files)
    dev_initials, dev_features, dev_labels = read_data(dev_files)

    learning_rate = 0.00025
    if ModelOpt == 'LSTM':
        model = dirmodelLSTM(Tx)
        learning_rate = 0.01
    elif ModelOpt == 'GRU':
        model = dirmodelGRU(Tx)
        learning_rate = 0.001
    opt = Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
    model.compile(
        optimizer = opt,
        loss = 'binary_crossentropy',
        metrics=['accuracy']
    )
    m = features.shape[0]

    c0 = np.zeros((m, n_a))
    dev_c0 = np.zeros((dev_features.shape[0], n_a))
    if (ModelOpt == "LSTM"):
        history_callback = model.fit(
            [features, initials, c0],
            list(labels),
            batch_size = 32,
            epochs = NUM_EPOCH,
            validation_data=([dev_features, dev_initials, dev_c0], list(dev_labels)),
            verbose = 0
        )
    elif (ModelOpt == "GRU"):
        history_callback = model.fit(
            [features, initials],
            list(labels),
            batch_size = 32,
            epochs = NUM_EPOCH,
            validation_data=([dev_features, dev_initials], list(dev_labels)),
            verbose = 0
        )

    with open("out/history_" + ID + ".json", "w") as f:
        json.dump(history_callback.history, f)

    model.save_weights("out/rnn_" + ID + "_weights.h5")
    with open("out/rnn_" + ID + "_architecture.json", "w") as f:
        json.dump(model.to_json(), f)
