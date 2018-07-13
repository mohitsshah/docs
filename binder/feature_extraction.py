import os
import re
import json
import spacy
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict, Counter

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


def run(source_file):
    nlp = spacy.load("en_core_web_sm")
    document = json.load(open(source_file, "r"))
    if document["errorFlag"]:
        return
    if document["source_format"] != "pdf":
        return
    data_frame = pd.DataFrame([], columns=["width", "height", "left", "top", "right",
                                           "bottom", "header", "footer", "min_size", "max_size", "median_size",
                                           "entities", "first_10_lines", "last_10_lines", "fonts", "content"])
    pages = document["documentPages"]
    for page in pages:
        features = {}
        features["width"] = float(page.get("pageWidth"))
        features["height"] = float(page.get("pageHeight"))
        features["left"] = float(page.get("marginLeft"))
        features["top"] = float(page.get("marginTop"))
        features["right"] = float(page.get("marginRight"))
        features["bottom"] = float(page.get("marginBottom"))

        page_headers = page.get("pageHeaders")
        headers = []
        for item in page_headers:
            line = []
            segment = item.get("paragraph")
            words = segment.get("words")
            for word in words:
                text = word.get("txt")
                line.append(text)
            headers.append(" ".join(line))
        features["header"] = "\n".join(headers)

        page_footers = page.get("pageFooters")
        footers = []
        for item in page_footers:
            line = []
            segment = item.get("paragraph")
            words = segment.get("words")
            for word in words:
                text = word.get("txt")
                line.append(text)
            footers.append(" ".join(line))
        features["footer"] = "\n".join(footers)
        page_items = []
        for segment in page.get("pageSegments"):
            if segment["isParagraph"]:
                paragraph = segment.get("paragraph")
                words = paragraph.get("words")
                for word in words:
                    item = []
                    coordinates = word.get("coordinates")
                    style = word.get("style")
                    item.append(coordinates["fromX"])
                    item.append(coordinates["fromY"])
                    item.append(coordinates["toX"])
                    item.append(coordinates["toY"])
                    item.append(style["font"])
                    item.append(style["fontSize"])
                    item.append(word.get("txt"))
                    page_items.append(item)
            elif segment["isTable"]:
                table = segment.get("table")
                cells = table.get("cells")
                for cell in cells:
                    for word in cell["words"]:
                        item = []
                        coordinates = word.get("coordinates")
                        style = word.get("style")
                        item.append(coordinates["fromX"])
                        item.append(coordinates["fromY"])
                        item.append(coordinates["toX"])
                        item.append(coordinates["toY"])
                        item.append(style["font"])
                        item.append(style["fontSize"])
                        item.append(word.get("txt"))
                        page_items.append(item)
        page_items = sorted(page_items, key=lambda x: (x[1], x[0]))
        sizes = []
        fonts = []
        words = []
        for item in page_items:
            font = item[4]
            if font:
                fonts.append(font)
            size = item[5]
            if size:
                sizes.append(float(size))
            words.append(item[-1])
        features["min_size"] = min(sizes) if sizes else 0
        features["max_size"] = max(sizes) if sizes else 0
        features["median_size"] = np.median(sizes) if sizes else 0
        features["fonts"] = Counter(fonts)
        lines = extract_lines(page_items)
        texts = [get_text(line) for line in lines]
        # first_10_lines = [re.sub(r'\d', '@', text) for text in texts[0:10]]
        # last_10_lines = [re.sub(r'\d', '@', text) for text in texts[-10:]]

        first_10_lines = []
        for text in texts[0:10]:
            pattern = re.compile("[^a-zA-Z]")
            text = re.sub(pattern, " ", text)
            tokens = [t for t in text.split() if t]
            text = " ".join(tokens).rstrip().lstrip()
            first_10_lines.append(text)

        last_10_lines = []
        for text in texts[-10:]:
            pattern = re.compile("[^a-zA-Z]")
            text = re.sub(pattern, " ", text)
            tokens = [t for t in text.split() if t]
            text = " ".join(tokens).rstrip().lstrip()
            last_10_lines.append(text)

        features["first_10_lines"] = first_10_lines
        features["last_10_lines"] = last_10_lines
        content = []
        entities = []
        allowed_ents = ["PERSON", "ORG", "NORP"]
        for text in texts:
            pattern = re.compile("[^a-zA-Z]")
            text = re.sub(pattern, " ", text)
            tokens = [t for t in text.split() if t]
            text = " ".join(tokens).rstrip().lstrip()
            content.append(text)
            if not text:
                continue
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in allowed_ents:
                    entities.append(ent.text)
        entities = list(set(entities))
        features["entities"] = entities
        features["content"] = content

        # features["median_size"] = np.median(sizes)
        # first_word = page_items[0]
        # last_word = page_items[-1]
        # print (first_word)
        # print (last_word)
        data_frame = data_frame.append(features, ignore_index=True)
    return data_frame
