import os
import re
import json
import docx
import zipfile
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from docx.document import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

class Processor(object):
    def __init__(self, source_file, dst, overwrite, cleanup):
        _, filename = os.path.split(source_file)
        name, _ = os.path.splitext(filename)
        tmp_dir = os.path.join(dst, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        self.source_file = source_file
        self.filename = filename
        self.name = name
        self.output_dir = dst
        self.tmp_dir = tmp_dir
        self.overwrite = overwrite
        self.cleanup = cleanup

    def save_document(self, document):
        """Serialize the final document to a JSON file

        Arguments:
            document {object} -- Document object
        """

        path = os.path.join(self.output_dir, self.name + ".json")
        with open(path, "w") as fi:
            json.dump(document, fi)

    def qn(self, tag):
        """
        Stands for 'qualified name', a utility function to turn a namespace
        prefixed tag name into a Clark-notation qualified tag name for lxml. For
        example, ``qn('p:cSld')`` returns ``'{http://schemas.../main}cSld'``.
        Source: https://github.com/python-openxml/python-docx/
        """
        nsmap = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        prefix, tagroot = tag.split(':')
        uri = nsmap[prefix]
        return '{{{}}}{}'.format(uri, tagroot)

    def text_from_xml(self, xml_content):
        """
        A string representing the textual content of this run, with content
        child elements like ``<w:tab/>`` translated to their Python
        equivalent.
        Adapted from: https://github.com/python-openxml/python-docx/
        """
        text = u''
        root = ET.fromstring(xml_content)
        for child in root.iter():
            attr = child.attrib
            if child.tag == self.qn('w:t'):
                t_text = child.text
                text += t_text if t_text is not None else ''
            elif child.tag == self.qn('w:tab'):
                text += '\t'
            elif child.tag in (self.qn('w:br'), self.qn('w:cr')):
                text += '\n'
            elif child.tag == self.qn("w:p"):
                text += '\n\n'
        return text

    def get_headers(self, zipf, files_list):
        header_xmls = re.compile('word/header[0-9]*.xml')
        headers = [self.text_from_xml(zipf.read(fname))
                   for fname in files_list if header_xmls.match(fname)]
        header_segments = []
        for header in headers:
            segment = {"paragraph": {"words": []}}
            for word in header.split():
                segment["paragraph"]["words"].append({
                    "style": None,
                    "coordinates": None,
                    "txt": word
                })
            header_segments.append(segment)
        return header_segments

    def get_footers(self, zipf, files_list):
        footer_xmls = re.compile('word/footer[0-9]*.xml')
        footers = [self.text_from_xml(zipf.read(fname))
                   for fname in files_list if footer_xmls.match(fname)]
        footer_segments = []
        for footer in footers:
            segment = {"paragraph": {"words": []}}
            for word in footer.split():
                segment["paragraph"]["words"].append({
                    "style": None,
                    "coordinates": None,
                    "txt": word
                })
            footer_segments.append(segment)
        return footer_segments

    def get_images(self, zipf, files_list):
        for fname in files_list:
            _, extension = os.path.splitext(fname)
            if extension in [".jpg", ".jpeg", ".png", ".bmp"]:
                dst_fname = os.path.join(self.tmp_dir, os.path.basename(fname))
                with open(dst_fname, "wb") as dst_f:
                    dst_f.write(zipf.read(fname))
        image_files = [os.path.join(self.tmp_dir, xx) for xx in os.listdir(self.tmp_dir)]
        return image_files

    def text_from_paragraph(self, paragraph):
        text = []
        for run in paragraph.runs:
            words = run.text.split()
            font_info = run.font
            font_name = font_info.name
            pointsize = font_info.size.pt if font_info.size else None
            is_bold = font_info.bold
            text.extend([w.rstrip().lstrip() for w in words])
        return text

    def get_segments(self, tree):
        segments = []
        for child in tree.element.body.iterchildren():
            if isinstance(child, CT_P):
                segment = {"paragraph": {"words": []}}
                paragraph = Paragraph(child, tree)
                words = self.text_from_paragraph(paragraph)
                for word in words:
                    segment["paragraph"]["words"].append({
                        "style": None,
                        "coordinates": None,
                        "txt": word
                    })
                segments.append({"label": "PARA", "segment": segment})
            elif isinstance(child, CT_Tbl):
                table = Table(child, tree)
                num_rows = len(table.rows)
                num_cols = len(table.columns)
                segment = {
                    "table": {
                        "numRows": num_rows,
                        "numCols": num_cols,
                        "headers": [],
                        "cells": []
                    }
                }
                for i in range(num_rows):
                    for j in range(num_cols):
                        cell_items = {
                            "rowIndex": i,
                            "colIndex": j,
                            "words": []
                        }
                        cell = table.cell(i, j)
                        words = []
                        paragraphs = cell.paragraphs
                        for paragraph in paragraphs:
                            words.extend(self.text_from_paragraph(paragraph))
                        for word in words:
                            cell_items["words"].append({
                                "style": None,
                                "coordinates": None,
                                "txt": word
                            })
                        segment["table"]["cells"].append(cell_items)
                segments.append({"label": "TABLE", "segment": segment})
        return segments

    def make_json(self, headers, footers, segments, error=None):
        document = {}
        document["source_file"] = self.source_file
        document["source_format"] = "docx"
        document["errorFlag"] = False
        document["error"] = None

        if error:
            document["error"] = error
            document["errorFlag"] = True
            return document

        document["numPages"] = 1
        document["TOC"] = {
            "isGenerated": False,
            "tocPageNumbers": [],
            "tocNodes": []
        }


        document["documentPages"] = []
        document_page = {"pageNumber": 1}

        document["documentPages"].append(document_page)

        document_page["pageSegments"] = []
        document_page["pageHeaders"] = []
        document_page["pageFooters"] = []

        for segment in segments:
            label = segment["label"]
            page_segment = {
                "isParagraph": False,
                "isTable": False,
                "isImage": False,
                "coordinates": None
            }
            if label == "PARA":
                page_segment["isParagraph"] = True
                page_segment["paragraph"] = segment["segment"]["paragraph"]
            elif label == "TABLE":
                page_segment["isTable"] = True
                page_segment["table"] = segment["segment"]["table"]
            document_page["pageSegments"].append(page_segment)

        for header in headers:
            header_segment = {
                "isParagraph": True,
                "isTable": False,
                "isImage": False,
                "coordinates": None,
                "paragraph": None
            }
            header_segment["paragraph"] = header["paragraph"]
            document_page["pageHeaders"].append(header_segment)

        for footer in footers:
            footer_segment = {
                "isParagraph": True,
                "isTable": False,
                "isImage": False,
                "coordinates": None,
                "paragraph": None
            }
            footer_segment["paragraph"] = footer["paragraph"]
            document_page["pageFooters"].append(footer_segment)

        document["documentPages"].append(document_page)
        return document

    def run(self):
        source_file = self.source_file
        docx_tree = docx.Document(source_file)
        zipf = zipfile.ZipFile(source_file)
        files_list = zipf.namelist()
        headers = self.get_headers(zipf, files_list)
        footers = self.get_footers(zipf, files_list)
        segments = self.get_segments(docx_tree)
        document = self.make_json(headers, footers, segments)
        self.save_document(document)
        return True