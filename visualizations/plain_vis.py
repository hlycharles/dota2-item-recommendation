import csv
import sys

def get_num(line):
    eq_index = line.find('=')
    return float(line[eq_index + 1:])

def generate_csv(in_file):
    with open(in_file, "r") as f:
        lines = f.readlines()
    lines = map(lambda s: s.strip(), lines)

    train_accs = []
    test_accs = []
    for i in range(0, len(lines), 3):
        epoch = get_num(lines[i])
        train_acc = get_num(lines[i + 1])
        test_acc = get_num(lines[i + 2])

        if (epoch >= len(train_accs)):
            train_accs.append(train_acc)
            test_accs.append(test_acc)

    header = [i + 1 for i in range(len(train_accs))]
    with open("out/plain_vis.csv", "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        writer.writerow(header)
        writer.writerow(train_accs)
        writer.writerow(test_accs)


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        in_file = sys.argv[1]
        generate_csv(in_file)
