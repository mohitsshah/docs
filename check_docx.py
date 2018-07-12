import re
import docx
import zipfile
import xml.etree.ElementTree as ET
from docx.document import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph


class Processor(object):
    def __init__(self, source_file):
        self.source_file = source_file

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
        return headers

    def get_footers(self, zipf, files_list):
        footer_xmls = re.compile('word/footer[0-9]*.xml')
        footers = [self.text_from_xml(zipf.read(fname))
                   for fname in files_list if footer_xmls.match(fname)]
        return footers

    def text_from_paragraph(self, paragraph):
        text = []
        font_name = None
        pointsize = None
        is_bold = None
        for run in paragraph.runs:
            words = run.text.split()
            font_info = run.font
            font_name = font_info.name
            pointsize = font_info.size.pt if font_info.size else None
            is_bold = font_info.bold
            text.extend([[w.rstrip().lstrip(), font_name, pointsize, is_bold] for w in words])
        return text

    def get_segments(self, tree):
        segments = []
        for child in tree.element.body.iterchildren():
            if isinstance(child, CT_P):
                paragraph = Paragraph(child, tree)
                words = self.text_from_paragraph(paragraph)
                segments.append(["PARA", words])
            elif isinstance(child, CT_Tbl):
                cells = []
                table = Table(child, tree)
                num_rows = len(table.rows)
                num_cols = len(table.columns)
                for i in range(num_rows):
                    for j in range(num_cols):
                        cell = table.cell(i, j)
                        words = []
                        paragraphs = cell.paragraphs
                        for paragraph in paragraphs:
                            words.extend(self.text_from_paragraph(paragraph))
                        cells.append([i, j, words])
                segments.append(["TABLE", cells])
        return segments

    def run(self):
        source_file = self.source_file
        docx_tree = docx.Document(source_file)
        zipf = zipfile.ZipFile(source_file)
        files_list = zipf.namelist()
        headers = self.get_headers(zipf, files_list)
        footers = self.get_footers(zipf, files_list)
        segments = self.get_segments(docx_tree)
        # print(headers, footers)
        print (segments)


if __name__ == "__main__":
    p = Processor("../../Downloads/sample.docx")
    p.run()