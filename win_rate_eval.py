'''
Evaluate the win rate at each step with different item purchase logs
'''

from keras.models import load_model, Model
from keras.layers import Dense, LSTM, Input, Lambda, Reshape
from keras.optimizers import Adam
from keras.models import model_from_json
from read_data_rnn import read_data
import os
import numpy as np
from feature_generator_rnn import load_data, get_feature_at_time

from utils import constants
import random

item_map = constants.item_map
item_names = item_map.keys()

n_a = 64
n_x = 408

LSTMCell = LSTM(n_a, return_state = True)
reshapor = Reshape((1, n_x))
initialDensor = Dense(n_a, input_shape=(n_x,))
outDensor = Dense(1, activation='sigmoid', input_shape=(n_a,))

def dirmodel(Tx):
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

if __name__ == "__main__":
    model = dirmodel(50)
    opt = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
    model.compile(
        optimizer = opt,
        loss = 'binary_crossentropy',
        metrics=['accuracy']
    )
    model.load_weights("data/rnn_1_weights.h5")

    match_files = []
    match_files.append(os.path.join("data", "dataset_rnn", "test", "4184957624.json"))
    initials, features, labels = read_data(match_files)
    m = features.shape[0]

    c0 = np.zeros((m, n_a))
    score = model.predict([features, initials, c0])
    print score[25]

    match = load_data("data/data_pro/2/public_match_4184957624.json")

    players = match["players"]
    p = players[-1]
    new_log = []
    for pu in p["purchase_log"]:
        if (pu["time"] < 0):
            new_log.append(pu)
    for i in range(50):
        new_log.append({
            "key": random.choice(item_names),
            "time": (i + 1) * 60
        })
    p["purchase_log"] = new_log
    match["players"][-1] = p

    result = []
    for i in range(10):
        result.append([])

    time_slices = [t for t in range(51)]
    features = map(lambda t: get_feature_at_time(match, t), time_slices)

    init = map(lambda x: x[0], features[0])
    wins = map(lambda x: x[1], features[0])
    for t in range(1, len(features)):
        for p in range(len(features[t])):
            result[p].append(features[t][p][0])

    score1 = model.predict([result, init, c0])
    print score1
