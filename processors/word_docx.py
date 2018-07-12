import os
import re
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
import processors.utils.image as img_utils
import processors.utils.ocr as tess_utils
import processors.utils.segmentation as seg_utils

class Processor(object):
    def __init__(self, source_file, dst, tessdata, overwrite, cleanup, oem, store_results):
        _, filename = os.path.split(source_file)
        name, _ = os.path.splitext(filename)
        # output_dir = os.path.join(dst, name)
        tmp_dir = os.path.join(dst, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        images_dir = os.path.join(dst, "images")
        os.makedirs(images_dir, exist_ok=True)

        self.source_file = source_file
        self.filename = filename
        self.name = name
        self.output_dir = dst
        self.tmp_dir = tmp_dir
        self.images_dir = images_dir
        self.tessdata = tessdata
        self.overwrite = overwrite
        self.cleanup = cleanup
        self.oem_mode = oem
        self.store_results = store_results
        self.images_to_keep = []


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

    def get_words_from_ocr(self, image_file, x_offset=0, y_offset=0, pad_offset=0, osd=True):
        """Perform OCR on a given image

        Arguments:
            image_file {str} -- Path to the image file on which OCR needs to be applied

        Keyword Arguments:
            x_offset {int} -- Required for cropped images to get the correct bounding box
            y_offset {int} -- Required for cropped images to get the correct bounding box
            pad_offset {int} -- Padding offset
            osd {bool} -- If True, perform orientation detection

        Returns:
            words [list] -- List of words along with their bounding boxes
            orientation[float] -- Orientation of the image
        """

        orientation = None
        if osd:
            osd = tess_utils.OSD(image_file, self.tessdata)
            rotation = osd.perform_osd()
            if rotation != 0:
                orientation = 360. - rotation
                image_file = img_utils.rotate_image(image_file, rotation)
        ocr = tess_utils.OCR(image_file, self.tessdata, self.oem_mode)
        words = ocr.perform_ocr(
            x_offset=x_offset, y_offset=y_offset, pad_offset=pad_offset)
        words = [[int(w[0]), int(w[1]), int(w[2]),
                  int(w[3]), w[4], w[-1]] for w in words]
        return words, orientation

    def make_segments(self, words, width, height):
        """[summary]

        Arguments:
            words {[type]} -- [description]
            image_regions {[type]} -- [description]
            image_words {[type]} -- [description]
            page_id {[type]} -- [description]
            width {[type]} -- [description]
            height {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        _, page_matrix, median_height = seg_utils.make_image_from_words(
            words, width, height)

        tmp1 = seg_utils.cut_segment(page_matrix)
        tmp2 = tmp1[1:] + [np.shape(page_matrix)[0]]
        tb_cuts = [(t1, t2) for t1, t2 in zip(tmp1, tmp2)]
        lr_cuts = []
        # Collect cuts for each row of the page
        for tb in tb_cuts:
            segment_image = page_matrix[tb[0]:tb[1], :].T
            lr = seg_utils.cut_segment(segment_image)
            lr_cuts.append(lr[0:-1])

        segments = seg_utils.label_segments(
            tb_cuts, lr_cuts, words, width, height, median_height)
        segments = seg_utils.merge_segments(
            segments, page_matrix, words, width, height, median_height)
        segments = seg_utils.merge_consecutive_tables(segments)
        blocks = seg_utils.make_blocks(segments, page_matrix, words)

        blocks = sorted(blocks, key=lambda x: (x[1], x[0]))
        return blocks

    def run(self):
        source_file = self.source_file
        docx_tree = docx.Document(source_file)
        zipf = zipfile.ZipFile(source_file)
        files_list = zipf.namelist()
        headers = self.get_headers(zipf, files_list)
        footers = self.get_footers(zipf, files_list)
        segments = self.get_segments(docx_tree)
        # print(headers, footers)
        image_files = self.get_images(zipf, files_list)
        for image_file in image_files:
            _, filename = os.path.split(image_file)
            name, _ = os.path.splitext(filename)
            outfile = os.path.join(self.images_dir, name + ".png")
            image_file = img_utils.convert_image(image_file, outfile, self.overwrite)
            image_words = self.get_words_from_ocr(image_file, pad_offset=None)
            image = cv2.imread(image_file)
            height, width, _ = image.shape
            width = int(float(width)*72/300.)
            height = int(float(height)*72/300.)
            print (width, height)
            print (image_file)
            blocks = self.make_segments(image_words[0], width, height)
            print (blocks)
            # print (image_words)
            print ()

        # TODO: Pack everything in a JSON file