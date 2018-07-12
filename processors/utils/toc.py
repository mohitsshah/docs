import re
from collections import defaultdict
import processors.utils.common as common_utils


def is_number(word):
    try:
        x = float(word)
        return True
    except:
        return False


def is_roman(word):
    roman_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
    return re.search(roman_pattern, word.strip(), re.I)


def is_uppercase_alphabet(word):
    if len(word) > 1:
        return False
    char = word[0]
    if not char.isalpha():
        return False
    if not char.isupper():
        return False
    return True


def add_toc(document):
    search_terms = ["table of contents", "t a b l e", "c o n t e n t s"]
    pages = {}
    pages_with_toc = []
    num_pages = document["num_pages"]
    for index, page in enumerate(document["pages"]):
        words = common_utils.extract_words(page["segments"])
        lines = common_utils.extract_lines(words)
        texts = [common_utils.get_text(x) for x in lines]
        term_found = None
        line_found = None
        for term in search_terms:
            term = "\\b" + term + "\\b"
            for line_index, text in enumerate(texts):
                matches = re.findall(term, text.lower())
                if matches:
                    term_found = term
                    line_found = line_index
                    break
            if term_found and line_found:
                break
        if term_found:
            if not pages_with_toc:
                pages_with_toc.append(index)
                texts = texts[line_found+1:]
                pages[index] = texts
            else:
                last_index = pages_with_toc[-1]
                if index == last_index + 1:
                    pages_with_toc.append(index)
                    texts = texts[line_found+1:]
                    pages[index] = texts
                else:
                    break

    if not pages_with_toc:
        return

    pages_with_toc.sort()
    table_of_contents = []
    for index in pages_with_toc:
        texts = pages[index]
        for line_index, line in enumerate(texts):
            pattern = re.compile("[.]{2,}")
            line = re.sub(pattern, " ", line)
            line_words = line.split()
            # first_word = line_words[0]
            last_word = line_words[-1]

            # if is_number(first_word) or is_uppercase_alphabet(first_word):
            #     first_count += 1
            #     lines_to_keep.append(line_index)

            if (is_number(last_word) or is_roman(last_word)) and (len(line_words) > 1):
                if is_number(last_word):
                    if float(last_word) <= num_pages:
                        table_of_contents.append([" ".join(line_words[0:-1]).upper(), last_word])
                else:
                    table_of_contents.append([" ".join(line_words[0:-1]).upper(), last_word])

    document["toc"] = {
        "pages": [x+1 for x in pages_with_toc],
        "content": table_of_contents
    }

if __name__ == "__main__":
    import json
    data = json.load(open("sample.json"))
    find_toc(data)
    print (data["toc"])
