import json
import os

def is_valid_match_file(match_file):
    if (os.path.isdir(match_file)):
        return False
    return match_file.endswith(".json")

def read_match_features(match_file):
    data = None
    with open(match_file, 'r') as f:
        data = json.load(f)
    examples = data['examples']
    features = []
    labels = []
    for example in examples:
        features.append(example['x'])
        labels.append(example['y'])
    return (features, labels)

def store_match_file(t):
    target, source = t
    data = None
    with open(source, 'r') as f:
        data = json.load(f)
    with open(target, 'w') as f:
        json.dump(data, f)
    return True
