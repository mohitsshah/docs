import os
import re
import numpy as np
from collections import defaultdict


def extract_words(segments):
    words = []
    for index, segment in enumerate(segments):
        label = segment["label"]
        if label == "PARA":
            tmp_words = [w for w in segment["content"]]
            words.extend(tmp_words)
        elif label == "TABLE":
            for row in segment["content"]:
                for cell in row:
                    tmp_words = [w for w in cell]
                    words.extend(tmp_words)
    words = sorted(words, key=lambda x: (x[1], x[0]))
    return words


def extract_lines(words):
    unique_y = defaultdict(list)
    for word in words:
        unique_y[(word[1], word[3])].append(word)
    keys = list(unique_y.keys())
    keys = sorted(keys, key=lambda x: (x[0]))
    lines = []
    for key in keys:
        lines.append(unique_y[key])
    return lines


def get_text(line):
    words = []
    for word in line:
        words.append(word[-1])
    return " ".join(words)


def get_box(line):
    x_0 = min([w[0] for w in line])
    y_0 = min([w[1] for w in line])
    x_1 = max([w[2] for w in line])
    y_1 = max([w[3] for w in line])
    box = [x_0, y_0, x_1, y_1]
    return np.array(box).astype("float")
