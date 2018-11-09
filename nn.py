'''
plain neural network that predicts the end result of DOTA2 game

the number of layers and size of each hidden layer can be specified through
command line arguments. default hidden layer sizes are 100, 10
'''

import numpy as np
import tensorflow as tf
from read_data import read_data
import random
import sys

LEARNING_RATE = 0.00025
NUM_EPOCH = 30
MINIBATCH_SIZE = 32
HIDDEN_LAYERS = [100, 10]
ID = "1"

PRINT_COST = True

def dprint(s):
    if (PRINT_COST):
        print s

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None])
    Y = tf.placeholder(tf.float32, [n_y, None])
    return X, Y

def initialize_parameters(n_x, hidden_sizes, n_y):
    layers = [n_x] + hidden_sizes + [n_y]

    result = []
    for l in range(1, len(layers)):
        n_l = layers[l]
        n_in = layers[l - 1]
        W = tf.get_variable("W" + str(l), [n_l, n_in], initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b" + str(l), [n_l, 1], initializer = tf.zeros_initializer())
        result.append((W, b))

    return result

def forward_propagation(X, parameters):
    current_input = X
    for i in range(len(parameters) - 1):
        (W, b) = parameters[i]
        Z = tf.add(tf.matmul(W, current_input), b)
        A = tf.nn.relu(Z)
        current_input = A
    (W, b) = parameters[-1]
    Z_L = tf.add(tf.matmul(W, current_input), b)
    return Z_L

def compute_cost(Z, Y):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = Y))
    return cost

def compute_accuracy(Z_L, Y):
    A_L = tf.sigmoid(Z_L)
    Y_hat = tf.cast(tf.cast(A_L + 0.5, tf.int32), tf.float32)
    eq = tf.equal(Y_hat, Y)

    accuracy = tf.reduce_mean(tf.cast(eq, 'float'))
    return accuracy

def model(X_train, Y_train, X_test, Y_test):
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(n_x, HIDDEN_LAYERS, n_y)
    Z_L = forward_propagation(X, parameters)

    cost = compute_cost(Z_L, Y)
    accuracy = compute_accuracy(Z_L, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(cost)

    init = tf.global_variables_initializer()

    with open('out/out_' + ID + '.txt', 'w') as f:
        f.write("--Epoch costs--" + str(HIDDEN_LAYERS) + "\n")
    with open('out/accuracy_' + ID + '.txt', 'w') as f:
        f.write("")

    with tf.Session() as sess:
        sess.run(init)
        num_minibatches = int(m / MINIBATCH_SIZE)

        permutation = list(np.random.permutation(m))
        shuffled_X = X_train[:, permutation]
        shuffled_Y = Y_train[:, permutation]

        for epoch in range(NUM_EPOCH):
            epoch_cost = 0
            for mb in range(num_minibatches):
                minibatch_X = shuffled_X[:, (mb*MINIBATCH_SIZE):((mb+1)*MINIBATCH_SIZE)]
                minibatch_Y = shuffled_Y[:, (mb*MINIBATCH_SIZE):((mb+1)*MINIBATCH_SIZE)]
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X : minibatch_X, Y : minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            dprint("epoch: " + str(epoch))
            dprint("epoch cost: " + str(epoch_cost))
            with open('out/out_' + ID + '.txt', 'a') as f:
                f.write('epoch=' + str(epoch) + '\n')
                f.write('cost=' + str(epoch_cost) + '\n')

            train_accuracy = sess.run([accuracy], feed_dict = {X : X_train, Y : Y_train})[0]
            test_accuracy = sess.run([accuracy], feed_dict = {X : X_test, Y : Y_test})[0]
            dprint("train accuracy: " + str(train_accuracy))
            dprint("test accuracy: " + str(test_accuracy))
            with open('out/accuracy_' + ID + '.txt', 'a') as f:
                f.write('epoch=' + str(epoch) + '\n')
                f.write("train accuracy=" + str(train_accuracy) + '\n')
                f.write("test accuracy=" + str(test_accuracy) + '\n')


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        PRINT_COST = int(sys.argv[1]) > 0
    if (len(sys.argv) > 2):
        layer_strs = sys.argv[2].split(',')
        HIDDEN_LAYERS = map(int, layer_strs)
    if (len(sys.argv) > 3):
        ID = sys.argv[3]
    if (len(sys.argv) > 4):
        in_folder = sys.argv[4]

    dprint("hidden layers:" + str(HIDDEN_LAYERS))

    dprint("Reading data")
    (X_train, Y_train) = read_data(in_folder + '/train')
    (X_test, Y_test) = read_data(in_folder + '/dev')
    dprint(X_train.shape)
    dprint(Y_train.shape)
    dprint("Done: Reading data")
    model(X_train, Y_train, X_test, Y_test)
