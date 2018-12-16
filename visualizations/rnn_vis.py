import json
import sys
import csv

TRAINING_PREFIX = "dense_2_acc"
DEV_PREFIX = "val_dense_2_acc"
Tx = 50

def generate_acc_epoch_csv(content, t):
    training_key = TRAINING_PREFIX + "_" + str(t)
    dev_key = DEV_PREFIX + "_" + str(t)

    epoch_count = len(content[training_key])

    header = [i + 1 for i in range(epoch_count)]
    training_acc = content[training_key]
    dev_acc = content[dev_key]

    with open("out/rnn_epoch_vis.csv", "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        writer.writerow(header)
        writer.writerow(training_acc)
        writer.writerow(dev_acc)

def generate_acc_time_csv(content):
    header = [i + 1 for i in range(Tx)]
    training_acc = []
    dev_acc = []

    training_acc.append(content[TRAINING_PREFIX][-1])
    dev_acc.append(content[DEV_PREFIX][-1])
    for i in range(1, Tx):
        training_key = TRAINING_PREFIX + "_" + str(i)
        training_acc.append(content[training_key][-1])
        dev_key = DEV_PREFIX + "_" + str(i)
        dev_acc.append(content[dev_key][-1])

    with open("out/rnn_vis.csv", "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        writer.writerow(header)
        writer.writerow(training_acc)
        writer.writerow(dev_acc)


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        in_file = sys.argv[1]
        content = dict()
        with open(in_file, "r") as f:
            content = json.load(f)
        generate_acc_time_csv(content)
        generate_acc_epoch_csv(content, 20)

