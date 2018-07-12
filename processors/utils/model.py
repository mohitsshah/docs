import os
import numpy as np

class DocumentModel(object):
    def __init__(self):
        self.document = {}

    def make_toc(self, toc):
        self.document["TOC"]["isGenerated"] = True
        self.document["TOC"]["tocPageNumbers"] = toc["pages"]
        for item in toc["content"]:
            node = {
                "txt": item[0],
                "pageNumber": item[1],
                "depth": 0,
                "point": None
            }
            self.document["TOC"]["tocNodes"].append(node)

    def make_style(self, item):
        style = {
            "font": None,
            "fontSize": None,
            "fontColor": None,
            "bold": False,
            "italic": False
        }
        data = item[-2]
        if data is None:
            return style
        if item[3] and item[1]:
            style["fontSize"] = item[3] - item[1]
        elif "font_size" in data:
            style["fontSize"] = data["font_size"]
        if "font_name" in data:
            style["font"] = data["font_name"]
        if "font_color" in data:
            style["fontColor"] = data["font_color"]
        if "bold" in data:
            style["bold"] = data["bold"]
        if "italic" in data:
            style["italic"] = data["italic"]
        return style

    def make_paragraph(self, page_segment, content):
        page_segment["isParagraph"] = True
        page_segment["paragraph"] = {"words": []}
        for item in content:
            word = {
                "txt": item[-1],
                "coordinates": {
                    "fromX": item[0],
                    "fromY": item[1],
                    "toX": item[2],
                    "toY": item[3]
                },
                "style": self.make_style(item)
            }
            page_segment["paragraph"]["words"].append(word)
        return page_segment

    def make_image(self, page_segment, content):
        page_segment["isImage"] = True
        page_segment["image"] = {
            "isGraph": False,
            "isPicture": True,
            "path": content
        }
        return page_segment

    def make_table(self, page_segment, content):
        page_segment["isTable"] = True
        page_segment["table"] = {
            "numRows": len(content),
            "numCols": len(content[0]),
            "headers": [],
            "cells": []
        }
        for row_index, row in enumerate(content):
            for col_index, cell_content in enumerate(row):
                cell = {
                    "rowIndex": row_index,
                    "colIndex": col_index,
                    "words": []
                }
                for cell_word in cell_content:
                    word = {
                        "txt": cell_word[-1],
                        "coordinates": {
                            "fromX": cell_word[0],
                            "fromY": cell_word[1],
                            "toX": cell_word[2],
                            "toY": cell_word[3]
                        },
                        "style": self.make_style(cell_word)
                    }
                    cell["words"].append(word)
                page_segment["table"]["cells"].append(cell)
        return page_segment


    def make_header(self, header):
        header_segment = {
            "isParagraph": True,
            "isTable": False,
            "isImage": False,
            "paragraph": {
                "words": []
            }
        }
        for item in header:
            word = {
                "txt": item[-1],
                "coordinates": {
                    "fromX": item[0],
                    "fromY": item[1],
                    "toX": item[2],
                    "toY": item[3]
                },
                "style": self.make_style(item)
            }
            header_segment["paragraph"]["words"].append(word)
        return header_segment


    def make_footer(self, footer):
        footer_segment = {
            "isParagraph": True,
            "isTable": False,
            "isImage": False,
            "paragraph": {
                "words": []
            }
        }
        for item in footer:
            word = {
                "txt": item[-1],
                "coordinates": {
                    "fromX": item[0],
                    "fromY": item[1],
                    "toX": item[2],
                    "toY": item[3]
                },
                "style": self.make_style(item)
            }
            footer_segment["paragraph"]["words"].append(word)
        return footer_segment

    def make_generic(self, data):
        self.document["numPages"] = data["num_pages"] if "num_pages" in data else None
        self.document["TOC"] = {
            "isGenerated": False,
            "tocPageNumbers": [],
            "tocNodes": []
        }
        if "toc" in data:
            self.make_toc(data["toc"])
        self.document["documentPages"] = []
        for page in data["pages"]:
            document_page = {"pageNumber": page["page_number"]}
            if "error" in page:
                document_page["errorFlag"] = True
                document_page["error"] = page["error"]
                self.document["documentPages"].append(document_page)
                continue
            document_page["pageWidth"] = int(page["width"]) if "width" in page else None
            document_page["pageHeight"] = int(page["height"]) if "height" in page else None
            document_page["pageImageOrientation"] = page["orientation"] if "orientation" in page else None
            document_page["isPageImage"] = page["is_page_image"] if "is_page_image" in page else None
            document_page["pageSegments"] = []
            document_page["pageHeaders"] = []
            document_page["pageFooters"] = []
            boxes = []
            for segment in page["segments"]:
                box = segment["bbox"]
                boxes.append(box)
                label = segment["label"]
                content = segment["content"]
                page_segment = {
                    "isParagraph": False,
                    "isTable": False,
                    "isImage": False,
                    "coordinates": {
                        "fromX": box[0],
                        "fromY": box[1],
                        "toX": box[2],
                        "toY": box[3]
                    }
                }
                if label == "PARA":
                    page_segment = self.make_paragraph(page_segment, content)

                elif label == "TABLE":
                    page_segment = self.make_table(page_segment, content)

                elif label == "IMAGE":
                    page_segment = self.make_image(page_segment, content)

                document_page["pageSegments"].append(page_segment)

            for header in page["headers"]:
                document_page["pageHeaders"].append(self.make_header(header))

            for footer in page["footers"]:
                document_page["pageFooters"].append(self.make_footer(footer))

            boxes = np.array(boxes).astype("int")
            left, top = list(np.min(boxes[:, 0:2], axis=0))
            right, bottom = list(np.max(boxes[:, 2:4], axis=0))
            document_page["marginLeft"] = int(left)
            document_page["marginTop"] = int(top)
            document_page["marginRight"] = int(page["width"]) - int(right)
            document_page["marginBottom"] = int(page["height"]) - int(bottom)
            self.document["documentPages"].append(document_page)


    def create(self, data, source_file, source_format, error=None):
        self.document["source_file"] = source_file
        self.document["source_format"] = source_format
        self.document["errorFlag"] = False
        self.document["error"] = None
        if data is None:
            self.document["errorFlag"] = True
            self.document["error"] = error
            return self.document

        self.make_generic(data)
        return self.document