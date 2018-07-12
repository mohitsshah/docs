import os
import re
import json
import spacy
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

nlp = spacy.load("en_core_web_sm")


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


def run(document):
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
    print(data_frame)
    data_frame.to_pickle("binder.pkl")


def split(data_frame):
    content = data_frame["content"]
    full_text = []
    for items in content:
        lines = [item for item in items if item]
        text = " ".join(lines).lower()
        full_text.append(text)
    full_vec = TfidfVectorizer(max_features=100, ngram_range=(
        1, 3), max_df=0.8, min_df=0.1, stop_words="english")
    full_text_vectors = full_vec.fit_transform(full_text).todense()
    full_text_similarity = np.dot(full_text_vectors, full_text_vectors.T)

    content = data_frame["first_10_lines"]
    first_text = []
    for items in content:
        lines = [item for item in items if item]
        text = " ".join(lines).lower()
        first_text.append(text)
    first_vec = TfidfVectorizer(max_features=100, ngram_range=(
        1, 3), max_df=0.8, min_df=0.1, stop_words="english")
    first_text_vectors = first_vec.fit_transform(first_text).todense()
    first_text_similarity = np.dot(first_text_vectors, first_text_vectors.T)

    content = data_frame["last_10_lines"]
    last_text = []
    for items in content:
        lines = [item for item in items if item]
        text = " ".join(lines).lower()
        last_text.append(text)
    last_vec = TfidfVectorizer(max_features=100, ngram_range=(
        1, 3), max_df=0.8, min_df=0.1, stop_words="english")
    last_text_vectors = last_vec.fit_transform(last_text).todense()
    last_text_similarity = np.dot(last_text_vectors, last_text_vectors.T)

    content = data_frame["fonts"]
    counter = Counter()
    for item in content:
        counter.update(item)
    fonts_vocab = list(counter.keys())
    # if "unknown" in fonts_vocab:
    #     fonts_vocab.remove("unknown")

    fonts = []
    for item in content:
        font = []
        for key, value in item.items():
            if key not in fonts_vocab:
                continue
            font.extend([key for _ in range(value)])
        fonts.append(" ".join(font))
    try:
        font_vec = TfidfVectorizer()
        font_vectors = font_vec.fit_transform(fonts).todense()
        font_similarity = np.dot(font_vectors, font_vectors.T)
    except:
        font_similarity = np.zeros((len(content), len(content)), dtype="float")

    entities = data_frame["entities"]
    widths = data_frame["width"]
    heights = data_frame["height"]
    left = data_frame["left"]
    right = data_frame["right"]
    top = data_frame["top"]
    bottom = data_frame["bottom"]
    num_pages = len(widths)

    stack = []
    segments = []
    history = 5
    future = 5
    for index in range(num_pages):
        if index == 0:
            stack.append(index)
            continue
        if index in stack:
            continue

        current_entities = set(entities[index])

        stack_text_score = 0
        stack_first_score = 0
        stack_last_score = 0
        stack_font_score = 0
        stack_entity_score = 0
        stack_dimensions_score = 0
        stack_margins_score = 0
        count = 0
        for num in stack[0:history]:
            stack_text_score += full_text_similarity[index, num]
            stack_first_score += first_text_similarity[index, num]
            stack_last_score += last_text_similarity[index, num]
            stack_font_score += font_similarity[index, num]
            stack_entities = set(entities[num])
            hits = float(len(current_entities.intersection(stack_entities)))
            if current_entities:
                stack_entity_score += (hits/len(current_entities))

            curr_dims = np.array([widths[index], heights[index]])
            stack_dims = np.array([widths[num], heights[num]])
            diff = curr_dims - stack_dims
            stack_dimensions_score += np.exp(-np.sum(diff * diff))

            curr_margins = np.array([left[index], top[index]])
            stack_margins = np.array([left[num], top[num]])
            diff = curr_margins - stack_margins
            stack_margins_score += np.exp(-np.sum(diff * diff))

            count += 1

        if count > 0:
            stack_text_score /= count
            stack_first_score /= count
            stack_last_score /= count
            stack_font_score /= count
            stack_dimensions_score /= count
            stack_margins_score /= count

        buffer_text_score = 0
        buffer_first_score = 0
        buffer_last_score = 0
        buffer_font_score = 0
        buffer_entity_score = 0
        buffer_dimensions_score = 0
        buffer_margins_score = 0
        count = 0
        buffer = []
        for num in range(index+1, index+future+1):
            try:
                buffer_text_score += full_text_similarity[index, num]
                buffer_first_score += first_text_similarity[index, num]
                buffer_last_score += last_text_similarity[index, num]
                buffer_font_score += font_similarity[index, num]
                buffer_entities = set(entities[num])
                hits = float(len(current_entities.intersection(buffer_entities)))
                if current_entities:
                    buffer_entity_score += (hits/len(current_entities))

                curr_dims = np.array([widths[index], heights[index]])
                buffer_dims = np.array([widths[num], heights[num]])
                diff = curr_dims - buffer_dims
                buffer_dimensions_score += np.exp(-np.sum(diff * diff))

                curr_margins = np.array([left[index], top[index]])
                buffer_margins = np.array([left[num], top[num]])
                diff = curr_margins - buffer_margins
                buffer_margins_score += np.exp(-np.sum(diff * diff))

                buffer.append(num)
                count += 1
            except:
                pass

        if count > 0:
            buffer_text_score /= count
            buffer_first_score /= count
            buffer_last_score /= count
            buffer_font_score /= count
            buffer_dimensions_score /= count
            buffer_margins_score /= count

        text_weight = 1.0
        first_weight = 1.0
        last_weight = 1.0
        font_weight = 1.0
        entity_weight = 1.0
        dimensions_weight = 1.0
        margins_weight = 1.0

        stack_score = text_weight*stack_text_score + first_weight*stack_first_score + last_weight * \
            stack_last_score + font_weight*stack_font_score + entity_weight*stack_entity_score + \
            dimensions_weight * stack_dimensions_score + margins_weight * stack_margins_score

        buffer_score = text_weight*buffer_text_score + first_weight*buffer_first_score + last_weight * \
            buffer_last_score + font_weight*buffer_font_score + \
            entity_weight*buffer_entity_score + dimensions_weight * buffer_dimensions_score + \
            margins_weight * buffer_margins_score

        norm = (text_weight + first_weight + last_weight + font_weight + entity_weight + dimensions_weight + margins_weight)
        stack_score /= norm
        buffer_score /= norm
        print(index+1, stack_score, buffer_score)

        if stack_score >= buffer_score:
            stack.append(index)
            stack.reverse()
        else:
            diff = buffer_score - stack_score
            if diff > 0.2:
                stack.sort()
                segments.append(stack)
                buffer.append(index)
                stack = list(buffer)
                stack.sort()
                stack.reverse()
            else:
                stack.append(index)
                stack.reverse()

    if stack:
        stack.sort()
        segments.append(stack)
        stack = []

    for segment in segments:
        tmp = [s+1 for s in segment]
        print(tmp)

    # for i in range(2, len(font_similarity)-2):
    #     print (i+1, font_similarity[i, i-2:i+3])


if __name__ == "__main__":
    # path = "tmp/merged1/merged1.json"
    path = "/Users/mohitshah/Projects/documents-caf/tmp/binder1/binder1.json"
    # path = "/Users/mohitshah/Projects/documents-caf/tmp/536210_SF_Industrial_Building/536210_SF_Industrial_Building.json"
    # path = "/Users/mohitshah/Projects/documents-caf/tmp/appraisal/appraisal.json"
    # path = "/Users/mohitshah/Projects/documents-caf/tmp/tiff_example/tiff_example.json"
    # path = "/Users/mohitshah/Projects/documents-caf/tmp/amazon/amazon.json"
    data = json.load(open(path, "r"))
    run(data)
    data = pd.read_pickle("binder.pkl")
    split(data)
