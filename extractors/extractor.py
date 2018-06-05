import extractors.utils.config as cfg
import extractors.utils.compile_defs as cdefs
import extractors.QANet.evaluate as evaluate
import spacy
import json
import pandas as pd
import re


class Extractor(object):
    def __init__(self, args):
        configs = cfg.make_configs(args)
        spacy_model = spacy.load("en_core_web_sm")
        qa_model = evaluate.Eval(configs)
        self.spacy_model = spacy_model
        self.qa_model = qa_model
        self.document = None
        self.content = None

    def set_defs(self, defs_path):
        defs = cdefs.run(defs_path)
        self.defs = defs
        pass

    def set_json(self, src):
        self.document = json.load(open(src))
        texts, blocks, key_values = self.parse_document()
        self.content = {"texts": texts, "blocks": blocks, "key_values": key_values}

    def parse_document(self):
        document = self.document
        labels = ["PARA", "TABLE"]
        blocks = []
        key_values = []
        texts = []
        for page in document["pages"]:
            blks = []
            kvs = []
            txts = []
            for segment in page["segments"]:
                if segment["label"] not in labels:
                    continue
                if segment["label"] == "PARA":
                    words = [item[-1] for item in segment["content"]]
                    txts.append(" ".join(words))
                    blks.append(" ".join(words))
                elif segment["label"] == "TABLE":
                    content = segment["content"]                    
                    for row in content:
                        r = []
                        for col in row:
                            tmp = [word[-1] for word in col]
                            if len(tmp) > 0:
                                r.append(" ".join(tmp))
                        if len(r) == 2:
                            kk = r[0]
                            vv = r[1]
                            tmp_k = kk.split("/")
                            tmp_v = vv.split("/")
                            if len(tmp_k) == len(tmp_v):
                                for xx in range(len(tmp_k)):
                                    kvs.append([tmp_k[xx], tmp_v[xx]])
                            elif len(tmp_k) > 1 and len(tmp_v) == 1:
                                for xx in range(len(tmp_k)):
                                    kvs.append([tmp_k[xx], tmp_v[0]])
                            else:
                                kvs.append(r)                            
                        else:
                            blks.append(" ".join(r))
                        txts.append(" ".join(r))

            texts.append(txts)
            blocks.append(blks)
            key_values.append(kvs)

        return texts, blocks, key_values

    def search_phrase_keys(self, phrase, key_values):
        pattern = re.compile("\\b%s\\b" % phrase.lower(), flags=re.IGNORECASE)
        match = []
        for bidx, block in enumerate(key_values):
            tt = block[0]
            tmp = [m.start() for m in re.finditer(pattern, tt.lower())]
            for t in tmp:
                match.append((bidx, t))
        return match

    def get_passages_keys(self, phrase, key_values):
        match = self.search_phrase_keys(phrase, key_values)
        passages = []
        indices = []
        for m in match:
            if m[0] not in indices:
                passages.append([m[0], key_values[m[0]]])
                indices.append(m[0])
        return passages

    def search_phrase_values(self, phrase, key_values):
        pattern = re.compile("\\b%s\\b" % phrase.lower(), flags=re.IGNORECASE)
        match = []
        for bidx, block in enumerate(key_values):
            tt = block[1]
            tmp = [m.start() for m in re.finditer(pattern, tt.lower())]
            for t in tmp:
                match.append((bidx, t))
        return match

    def get_passages_values(self, phrase, key_values):
        match = self.search_phrase_values(phrase, key_values)
        passages = []
        indices = []
        for m in match:
            if m[1] not in indices:
                passages.append([m[0], key_values[m[0]]])
                indices.append(m[1])
        return passages

    def search_phrase_blocks(self, phrase, blocks):
        pattern = re.compile("\\b%s\\b" % phrase.lower(), flags=re.IGNORECASE)
        match = []
        for bidx, block in enumerate(blocks):
            tmp = [m.start() for m in re.finditer(pattern, block.lower())]
            for t in tmp:
                match.append((bidx, t))
        return match

    def get_passages_blocks(self, phrase, blocks, window=2):
        match = self.search_phrase_blocks(phrase, blocks)
        passages = []
        indices = []
        for m in match:
            if m[0] not in indices:
                L = len(blocks)
                texts = []
                for j in range(1, window + 1):
                    if m[0] - j >= 0:
                        texts.append(blocks[m[0] - j])
                texts.reverse()
                texts.append(blocks[m[0]])
                for j in range(1, window + 1):
                    if m[0] + j < L:
                        texts.append(blocks[m[0] + j])
                passages.append([m[0], "\n".join(texts)])
                indices.append(m[0])
        return passages

    def search_terms(self, terms, include, exclude, pages_list):
        content = self.content
        texts = content["texts"]
        matches = []
        if len(pages_list) == 0:
            pages_list = range(len(texts))
        for page_num, text in enumerate(texts):
            if page_num in pages_list:
                text = text.lower()
                exclude_found = False
                if len(exclude) > 0:
                    for exc in exclude:
                        pattern = "\\b%s\\b" % exc.lower()
                        p = re.compile(pattern, flags=re.IGNORECASE)
                        m = re.findall(p, text)
                        if len(m) > 0:
                            exclude_found = True
                            break
                if exclude_found:
                    continue

                for term in terms:
                    context_found = False
                    pattern = "\\b%s\\b" % term.lower()
                    p = re.compile(pattern, flags=re.IGNORECASE)
                    m = re.findall(p, text)
                    if len(m) < 1:
                        pass
                    else:
                        if len(include) > 0:
                            for inc in include:
                                pattern = "\\b%s\\b" % inc.lower()
                                p = re.compile(pattern, flags=re.IGNORECASE)
                                mm = re.findall(p, text)
                                if len(mm) > 0:
                                    matches.append([page_num, 1, term])
                                    context_found = True
                            if not context_found:
                                matches.append([page_num, 0, term])
                        else:
                            matches.append([page_num, 0, term])
        return matches

    def extract_from_value(self, value, text_extract, expects):
        ans = str(value)
        if text_extract is not None:
            if text_extract["type"] == "SHORT":
                if text_extract["method"] == "QA":
                    question = text_extract["question"]
                    try:
                        ans = self.qa_model.extract(question, value)
                    except Exception:
                        pass
        if expects is None:
            return ans, "search", "text"
        patterns = expects["patterns"]
        for pat in patterns:
            res = re.search(re.compile(pat), ans)
            if res is not None:
                return ans, "search", "regex"

        entities = expects["entities"]
        nlp = self.spacy_model
        doc = nlp(ans)
        for ent in doc.ents:
            if ent.label_ in entities:
                return ans, "search", "entity"
        return None

    def find_regex_pattern(self, patterns):
        texts = self.content["texts"]
        for patt in patterns:
            pp = re.compile(patt)
            for p_num, text in enumerate(texts):
                m = re.findall(pp, text)
                if len(m) > 0:
                    return p_num, m[0]
        return None

    def _extract_item(self, item):
        content = self.content
        name = item["name"]
        method = item["type"]
        group = item["group"]
        pages_list = []
        if "pages" in item and item["pages"] is not None:
            tmp = item["pages"].split("-")
            if len(tmp) == 1:
                pages_list = [int(tmp[0])]
            elif len(tmp) == 2:
                pages_list = range(int(tmp[0]), int(tmp[1]))
        if method == "search":
            terms = item["search"]["terms"]
            include = item["search"]["include"]
            exclude = item["search"]["exclude"]
            regions = item["search"]["regions"]
            expects = item["expects"] if "expects" in item else None
            text_extract = item["text_extract"] if "text_extract" in item else None
            matches = self.search_terms(terms, include, exclude, pages_list)
            if len(matches) > 0:
                matches = sorted(matches, key=lambda x: (-x[1], x[0], -len(x[2])))
                if "keys" in regions:
                    for m in matches:
                        key_matches = self.get_passages_keys(m[2], content["key_values"][m[0]])
                        for key_match in key_matches:
                            item = key_match[1]
                            key, value = item
                            ret = self.extract_from_value(value, text_extract, expects)
                            if ret is not None:
                                ans, ans_method, ans_type = ret
                                res = {
                                    "Name": name,
                                    "Value": ans,
                                    "Type": ans_type,
                                    "Method": ans_method,
                                    "Source": " ".join([key, value]),
                                    "Region": "key",
                                    "Page": m[0],
                                    "Group": group
                                }
                                return res
                if "values" in regions:
                    for m in matches:
                        value_matches = self.get_passages_values(m[2], content["kv"][m[0]])
                        for value_match in value_matches:
                            item = value_match[1]
                            key, value = item
                            ret = self.extract_from_value(value, text_extract, expects)
                            if ret is not None:
                                ans, ans_method, ans_type = ret
                                res = {
                                    "Name": name,
                                    "Value": ans,
                                    "Type": ans_type,
                                    "Method": ans_method,
                                    "Source": " ".join([key, value]),
                                    "Region": "value",
                                    "Page": m[0],
                                    "Group": group
                                }
                                return res
                if "paragraphs" in regions:
                    for m in matches:
                        block_matches = self.get_passages_blocks(m[2], content["blocks"][m[0]])
                        for block_match in block_matches:
                            value = block_match[1]
                            ret = self.extract_from_value(value, text_extract, expects)
                            if ret is not None:
                                ans, ans_method, ans_type = ret
                                res = {
                                    "Name": name,
                                    "Value": ans,
                                    "Type": ans_type,
                                    "Method": ans_method,
                                    "Source": value,
                                    "Region": "paragraph",
                                    "Page": m[0],
                                    "Group": group
                                }
                                return res
            return {
                "Name": name,
                "Value": None,
                "Type": None,
                "Method": None,
                "Source": None,
                "Region": None,
                "Page": None,
                "Group": group
            }

        elif method == "lookup":
            ans_page = None
            terms = item["search"]["terms"]
            include = item["search"]["include"]
            exclude = item["search"]["exclude"]
            matches = self.search_terms(terms, include, exclude, pages_list)
            if len(matches) > 0:
                ans = item["values"][0]
                ans_page = matches[0][0]
            else:
                ans = item["values"][1]
            res = {
                "Name": name,
                "Value": ans,
                "Type": None,
                "Method": "lookup",
                "Source": None,
                "Region": None,
                "Page": ans_page,
                "Group": group
            }
            return res
        elif method == "regex":
            ans_page = None
            patterns = item["patterns"]
            ret = self.find_regex_pattern(patterns)
            if ret is not None:
                ans_page = ret[0]
                ans = ret[1]
            else:
                ans = None
            res = {
                "Name": name,
                "Value": ans,
                "Type": None,
                "Method": "regex",
                "Source": None,
                "Region": None,
                "Page": ans_page,
                "Group": group
            }
            return res

    def extract(self):
        defs = self.defs
        if defs is None:
            return None
        df = pd.DataFrame(columns=["Name", "Value", "Type", "Method", "Region", "Source", "Page", "Group"])
        for item in defs:
            r = self._extract_item(item)
            df = df.append(r, ignore_index=True)
        return df
