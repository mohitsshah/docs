import os
import re
import numpy as np
from fuzzywuzzy import fuzz
from collections import defaultdict
import processors.utils.common as common_utils


def get_similarity(text_1, text_2, box_1, box_2):
    t_sim = float(fuzz.ratio(text_1, text_2)) / 100
    distance = np.sum((box_1 - box_2) * (box_1 - box_2))
    b_sim = np.exp(-distance)
    similarity = t_sim * b_sim
    return similarity


def add_headers_footers(document):
    page_lines = []
    page_widths = []
    page_heights = []
    for page in document["pages"]:
        page_widths.append(page["width"])
        page_heights.append(page["height"])
        words = common_utils.extract_words(page["segments"])
        lines = common_utils.extract_lines(words)
        page_lines.append(lines)
    num_pages = document["num_pages"]
    window_size = 5
    num_lines = 5
    h_scores = np.zeros((num_pages, num_lines))
    f_scores = np.zeros((num_pages, num_lines))
    h_weights = [1.0, 1.0, 0.5, 0.5, 0.5]
    f_weights = [1.0, 1.0, 0.5, 0.5, 0.5]
    num = np.zeros(num_pages)
    for i, page in enumerate(document["pages"]):
        lines_1 = page_lines[i]
        for j in range(0, window_size):
            if (i - j - 1) >= 0:
                lines_2 = page_lines[i-j-1]
                for n in range(num_lines):
                    try:
                        line_1 = lines_1[n]
                        line_2 = lines_2[n]
                        text_1 = common_utils.get_text(line_1)
                        text_1 = re.sub(r'\d', '@', text_1)
                        text_2 = common_utils.get_text(line_2)
                        text_2 = re.sub(r'\d', '@', text_2)
                        box_1 = common_utils.get_box(line_1)
                        box_2 = common_utils.get_box(line_2)
                        similarity = get_similarity(
                            text_1, text_2, box_1, box_2)
                        h_scores[i, n] += (similarity * h_weights[n])
                    except Exception:
                        pass
                for n in range(num_lines):
                    try:
                        line_1 = lines_1[len(lines_1) - n - 1]
                        line_2 = lines_2[len(lines_2) - n - 1]
                        text_1 = common_utils.get_text(line_1)
                        text_1 = re.sub(r'\d', '@', text_1)
                        text_2 = common_utils.get_text(line_2)
                        text_2 = re.sub(r'\d', '@', text_2)
                        box_1 = common_utils.get_box(line_1)
                        box_2 = common_utils.get_box(line_2)
                        similarity = get_similarity(
                            text_1, text_2, box_1, box_2)
                        f_scores[i, n] += (similarity * f_weights[n])
                    except Exception as e:
                        pass
                num[i] += 1
            if (i + j + 1) < num_pages:
                lines_2 = page_lines[i+j+1]
                for n in range(num_lines):
                    try:
                        line_1 = lines_1[n]
                        line_2 = lines_2[n]
                        text_1 = common_utils.get_text(line_1)
                        text_1 = re.sub(r'\d', '@', text_1)
                        text_2 = common_utils.get_text(line_2)
                        text_2 = re.sub(r'\d', '@', text_2)
                        box_1 = common_utils.get_box(line_1)
                        box_2 = common_utils.get_box(line_2)
                        similarity = get_similarity(
                            text_1, text_2, box_1, box_2)
                        h_scores[i, n] += (similarity * h_weights[n])
                    except Exception:
                        pass
                for n in range(num_lines):
                    try:
                        line_1 = lines_1[len(lines_1) - n - 1]
                        line_2 = lines_2[len(lines_2) - n - 1]
                        text_1 = common_utils.get_text(line_1)
                        text_1 = re.sub(r'\d', '@', text_1)
                        text_2 = common_utils.get_text(line_2)
                        text_2 = re.sub(r'\d', '@', text_2)
                        box_1 = common_utils.get_box(line_1)
                        box_2 = common_utils.get_box(line_2)
                        similarity = get_similarity(
                            text_1, text_2, box_1, box_2)
                        f_scores[i, n] += (similarity * f_weights[n])
                    except Exception as e:
                        pass
                num[i] += 1

    threshold = 0.5
    h_scores = h_scores / num.reshape(-1, 1)
    f_scores = f_scores / num.reshape(-1, 1)
    for i in range(num_pages):
        scores = h_scores[i, :]
        headers = []
        for line_num, score in enumerate(scores):
            if score > threshold:
                lines = page_lines[i]
                text = lines[line_num]
                if text:
                    headers.append(text)
        scores = f_scores[i, :]
        footers = []
        for line_num, score in enumerate(scores):
            if score > threshold:
                lines = page_lines[i]
                text = lines[len(lines) - line_num - 1]
                if text:
                    footers.append(text)

        document["pages"][i]["headers"] = headers
        document["pages"][i]["footers"] = footers
