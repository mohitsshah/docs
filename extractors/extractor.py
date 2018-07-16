import extractors.utils.config as cfg
import extractors.utils.compile_defs as cdefs
import extractors.QANet.evaluate as evaluate
import spacy
import json
import pandas as pd
import re


class Extractor(object):
    def __init__(self, data_dir, model_dir, model_name):
        configs = cfg.make_configs(data_dir, model_dir, model_name)
        spacy_model = spacy.load("en_core_web_sm")
        qa_model = evaluate.Eval(configs)
        self.spacy_model = spacy_model
        self.qa_model = qa_model
        self.document = None
        self.content = None
        self.defs = None
        self.num_pages = None
        self.toc_pages = None

    def set_defs(self, defs_path):
        defs = cdefs.run(defs_path)
        self.defs = defs

    def set_json(self, src):
        self.document = json.load(open(src))
        if bool(self.document.get("errorFlag")):
            return False
        self.num_pages = self.document.get("numPages")
        self.toc_pages = []
        toc = self.document.get("TOC")
        if bool(toc.get("isGenerated")):
            self.toc_pages.extend(toc.get("tocPageNumbers"))
        self.parse_document(self.document.get("documentPages"))

    def parse_paragraph(self, paragraph):
        text = []
        for word in paragraph.get("words"):
            text.append(word.get("txt"))
        raw_text = " ".join(text)
        return raw_text

    def parse_table(self, table):
        num_rows = table.get("numRows")
        num_cols = table.get("numCols")
        data = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        for cell in table.get("cells"):
            row_index = cell.get("rowIndex")
            col_index = cell.get("colIndex")
            text = []
            for word in cell.get("words"):
                text.append(word.get("txt"))
            text = " ".join(text)
            data[row_index][col_index] = text
        lines = [" ".join(row) for row in data]
        raw_text = "\n".join(lines)
        return raw_text, data

    def parse_document(self, pages):
        self.content = {}
        for page in pages:
            page_number = int(page.get("pageNumber"))
            raw_text = []
            segments = []
            for segment in page.get("pageSegments"):
                if (segment.get("isParagraph")):
                    text = self.parse_paragraph(segment.get("paragraph"))
                    raw_text.append(text)
                    segments.append({"paragraph": text})
                elif (segment.get("isTable")):
                    text, table = self.parse_table(segment.get("table"))
                    raw_text.append(text)
                    segments.append({"table": table})
            raw_text = "\n".join(raw_text)
            self.content[page_number] = {
                "raw_text": raw_text,
                "segments": segments
            }

    def search_excludes(self, page_num, exclude):
        if exclude:
            for exc in exclude:
                pattern = re.compile("\\b%s\\b" % exc.lower(), flags=re.IGNORECASE)
                match = re.findall(pattern, self.content[page_num]["raw_text"])
                if match:
                    return True
        return False

    def search_includes(self, page_num, include):
        if include:
            for inc in include:
                pattern = re.compile("\\b%s\\b" % inc.lower(), flags=re.IGNORECASE)
                match = re.findall(pattern, self.content[page_num]["raw_text"])
                if match:
                    return True
        return False

    def search_paragraph(self, term, paragraph):
        pattern = re.compile("\\b%s\\b" % term.lower(), flags=re.IGNORECASE)
        matches = re.findall(pattern, paragraph)
        if not matches:
            return []
        match = [{
            "type": "paragraph",
            "term": term,
            "paragraph": paragraph
        }]
        return match

    def search_table(self, term, table):
        matches = []
        pattern = re.compile("\\b%s\\b" % term.lower(), flags=re.IGNORECASE)
        num_rows = len(table)
        for row_index, row in enumerate(table):
            cells = [x for x in row if len(x.rstrip().lstrip()) > 0]
            if len(cells) == 2:
                keys_values = []
                key = cells[0]
                value = cells[1]
                tmp_k = key.split("/")
                tmp_v = value.split("/")
                if len(tmp_k) == len(tmp_v):
                    for idx in range(len(tmp_k)):
                        keys_values.append([tmp_k[idx], tmp_v[idx]])
                elif len(tmp_k) > 1 and len(tmp_v) == 1:
                    for idx in range(len(tmp_k)):
                        keys_values.append([tmp_k[idx], tmp_v[0]])
                else:
                    keys_values.append([key, value])
                for key, value in keys_values:
                    match = re.findall(pattern, key)
                    if match:
                        matches.append({
                            "type": "key",
                            "term": term,
                            "key": key,
                            "value": value
                        })
                        continue
                    match = re.findall(pattern, value)
                    if match:
                        matches.append({
                            "type": "value",
                            "term": term,
                            "key": key,
                            "value": value
                        })
                        continue
            else:
                for col_index, col in enumerate(row):
                    match = re.findall(pattern, col)
                    if not match:
                        continue
                    if row_index == 0:
                        if num_rows > 1:
                            key = col
                            value = table[row_index+1][col_index]
                            matches.append({
                                "type": "key",
                                "term": term,
                                "key": key,
                                "value": value
                            })
                    matches.append({
                        "type": "paragraph",
                        "term": term,
                        "paragraph": " ".join(cells)
                    })
        return matches



    def search_single_term(self, page_num, term):
        matches = []
        for index, segment in enumerate(self.content[page_num]["segments"]):
            if "paragraph" in segment:
                results = self.search_paragraph(term, segment["paragraph"])
                for result in results:
                    result["segment_index"] = index
                    result["page_num"] = page_num
                matches.extend(results)
            if "table" in segment:
                results = self.search_table(term, segment["table"])
                for result in results:
                    result["segment_index"] = index
                    result["page_num"] = page_num
                matches.extend(results)
        return matches


    def search_terms(self, terms, include, exclude, pages_list):
        all_matches = []
        if not pages_list:
            pages_list = range(1, self.num_pages+1)
        for page_num in pages_list:
            exclude_found = self.search_excludes(page_num, exclude)
            if exclude_found:
                continue
            for term in terms:
                matches = self.search_single_term(page_num, term)
                if not matches:
                    continue
                context_found = self.search_includes(page_num, include)
                for match in matches:
                    match["context_found"] = context_found
                all_matches.extend(matches)
        return all_matches

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

    def find_regex_pattern(self, patterns, pages_list):
        if not pages_list:
            pages_list = range(1, self.num_pages+1)
        patterns = [re.compile(pattern) for pattern in patterns]
        for page_num in pages_list:
            text = self.content[page_num]["raw_text"]
            for pattern in patterns:
                match = re.findall(pattern, text)
                if match:
                    return page_num, match[0]
        return None

    def _extract_item(self, item):
        name = item["name"]
        method = item["type"]
        group = item["group"]
        result = {
            "Name": name,
            "Value": None,
            "Type": None,
            "Method": method,
            "Source": None,
            "Region": None,
            "Page": None,
            "Group": group,
            "Term": None
        }

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
            if not matches:
                return result

            matches = sorted(matches, key=lambda x: (-x["context_found"], x["page_num"]))

            if "keys" in regions:
                region_matches = [match for match in matches if match["type"] == "key"]
                for match in region_matches:
                    key = match["key"]
                    value = match["value"]
                    ret = self.extract_from_value(value, text_extract, expects)
                    if ret is not None:
                        ans, ans_method, ans_type = ret
                        result["Value"] = ans
                        result["Method"] = ans_method
                        result["Type"] = ans_type
                        result["Page"] = match["page_num"]
                        result["Region"] = "key"
                        result["Source"] = " ".join([key, value])
                        result["Term"] = match["term"]
                        return result

            if "values" in regions:
                region_matches = [match for match in matches if match["type"] == "value"]
                for match in region_matches:
                    key = match["key"]
                    value = match["value"]
                    ret = self.extract_from_value(value, text_extract, expects)
                    if ret is not None:
                        ans, ans_method, ans_type = ret
                        result["Value"] = ans
                        result["Method"] = ans_method
                        result["Type"] = ans_type
                        result["Page"] = match["page_num"]
                        result["Region"] = "value"
                        result["Source"] = " ".join([key, value])
                        result["Term"] = match["term"]
                        return result

            if "paragraphs" in regions:
                region_matches = [match for match in matches if match["type"] == "paragraph"]
                for match in region_matches:
                    paragraph = match["paragraph"]
                    ret = self.extract_from_value(paragraph, text_extract, expects)
                    if ret is not None:
                        ans, ans_method, ans_type = ret
                        result["Value"] = ans
                        result["Method"] = ans_method
                        result["Type"] = ans_type
                        result["Page"] = match["page_num"]
                        result["Region"] = "paragraph"
                        result["Source"] = paragraph
                        result["Term"] = match["term"]
                        return result

            return result

        elif method == "lookup":
            ans_page = None
            terms = item["search"]["terms"]
            include = item["search"]["include"]
            exclude = item["search"]["exclude"]
            matches = self.search_terms(terms, include, exclude, pages_list)
            if matches:
                ans = item["values"][0]
                ans_page = matches[0]["page_num"]
            else:
                ans = item["values"][1]
            result["Value"] = ans
            result["Page"] = ans_page
            result["Method"] = "lookup"
            return result

        elif method == "regex":
            ans_page = None
            patterns = item["patterns"]
            ret = self.find_regex_pattern(patterns, pages_list)
            if ret is not None:
                ans_page = ret[0]
                ans = ret[1]
            else:
                ans = None
            result["Value"] = ans
            result["Page"] = ans_page
            result["Method"] = "regex"
            return result

    def extract(self):
        defs = self.defs
        if defs is None:
            return None
        data_frame = pd.DataFrame(columns=["Name", "Value", "Type", "Method", "Region", "Source", "Page", "Group"])
        for item in defs:
            result = self._extract_item(item)
            data_frame = data_frame.append(result, ignore_index=True)
        return data_frame
