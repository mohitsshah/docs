import sys
import numpy as np
import cv2
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter
from pdfminer.cmapdb import CMapDB
from pdfminer.layout import LAParams
from pdfminer.image import ImageWriter

def get_attribs(items):
    obj = {}
    for item in items:
        obj[item[0]] = item[1]
    return obj

def format_xml_box(box, width, height):
    box = [float(b) for b in box]
    box[1] = height - box[1]
    box[3] = height - box[3]
    tmp = box[1]
    box[1] = box[3]
    box[3] = tmp
    return box

def make_word_box(chars):
    chars = sorted(chars, key=lambda x: (x[1], x[0]))
    tmp = [c[-1] for c in chars]
    word_box = [chars[0][0], chars[0][1],
                chars[-1][2], chars[-1][3], chars[0][4], "".join(tmp)]
    return word_box

def is_y_similar(ry0, ry1, y0, y1):
    if ry0 == y0:
        return True
    if ry0 < y0 < ry1:
        return True
    return False

def extract_image_regions(page_matrix):
    regions = []
    page_matrix = np.array(page_matrix).astype("uint8")
    _, contours, _ = cv2.findContours(
        page_matrix, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [list(x) for xx in contour for x in xx]
        points = np.array(points)
        x0, y0 = np.min(points, axis=0)
        x1, y1 = np.max(points, axis=0)
        x1 += 1
        y1 += 1
        regions.append([x0, y0, x1, y1])
    regions = sorted(regions, key=lambda x: (x[1], x[0]))
    return regions

def scan_figures(tree, width, height):
    figures = tree.findall("figure")
    unique_y = {}
    chars = []
    for figure in figures:
        for child in figure:
            if child.tag == "text":
                box = format_xml_box(child.attrib["bbox"].split(","), width, height)
                unique_y[(box[1], box[3])] = True
                font_info = {"font_name": None, "pointsize": None}
                if "font" in child.attrib:
                    font_info["font_name"] = child.attrib["font"]
                if "size" in child.attrib:
                    font_info["pointsize"] = int(
                        float(child.attrib["size"]))
                box.append(font_info)
                box.append(child.text)
                chars.append(box)

    chars = sorted(chars, key=lambda x: (x[1], x[0]))
    words = []
    fig_inds = []
    for y in unique_y.keys():
        ref_y0 = y[0]
        ref_y1 = y[1]
        C = []
        inds = []
        for idx, c in enumerate(chars):
            if idx not in fig_inds:
                y0 = c[1]
                y1 = c[3]
                if is_y_similar(ref_y0, ref_y1, y0, y1):
                    C.append(c)
                    inds.append(idx)
        C = sorted(C, key=lambda x: (x[1], x[0]))
        inds = list(set(inds))
        fig_inds.extend(inds)
        widths = []
        for i, cc in enumerate(C):
            widths.append(cc[2] - cc[0])
        med = np.mean(widths) if len(widths) > 0 else 0.
        word = []
        for i, cc in enumerate(C):
            if i == 0:
                word.append(cc)
            else:
                g = cc[0] - C[i - 1][2]
                if (g >= 2. * med) or (len(cc[-1].rstrip().lstrip()) == 0):
                    tmp = []
                    for l in word:
                        tmp.append(l[-1])
                    if len(tmp) > 0:
                        bb = [word[0][0], word[0][1], word[-1]
                                [2], word[-1][3], word[0][4], "".join(tmp)]
                        words.append(bb)
                    if g >= 2. * med:
                        word = [cc]
                    else:
                        word = []
                else:
                    word.append(cc)
        tmp = []
        for l in word:
            tmp.append(l[-1])
        if len(tmp) > 0:
            bb = [word[0][0], word[0][1], word[-1]
                    [2], word[-1][3], word[0][4], "".join(tmp)]
            words.append(bb)
    words = [[int(w[0]), int(w[1]), int(w[2]), int(w[3]), w[4], w[-1]]
                for w in words]
    words = sorted(words, key=lambda x: (x[1], x[0]))
    return words

def scan_texts(tree, width, height):
    words = []
    text_boxes = tree.findall("textbox")
    for textbox in text_boxes:
        for textline in textbox:
            if textline.tag == "textline":
                chars = []
                for word in textline:
                    if word.tag == "text":
                        ch = word.text
                        ch = ch.rstrip().lstrip()
                        font_info = {"font_name": None, "pointsize": None}
                        if "bbox" in word.attrib:
                            box = format_xml_box(word.attrib["bbox"].split(","), width, height)
                            if len(ch) == 0:
                                if len(chars) > 0:
                                    words.append(make_word_box(chars))
                                    chars = []
                            else:
                                font_info["font_name"] = word.attrib["font"]
                                font_info["pointsize"] = int(
                                    float(word.attrib["size"]))
                                tmp = box + [font_info, ch]
                                chars.append(tmp)
                        else:
                            if len(chars) > 0:
                                words.append(make_word_box(chars))
                                chars = []
                if len(chars) > 0:
                    words.append(make_word_box(chars))
                    chars = []
    words = [[int(w[0]), int(w[1]), int(w[2]), int(w[3]), w[4], w[-1]]
                for w in words]
    words = sorted(words, key=lambda x: (x[1], x[0]))
    return words

def scan_images(tree, width, height):
    figures = tree.findall("figure")
    if len(figures) == 0:
        return []
    page_matrix = np.zeros((int(height), int(width)))
    for figure in figures:
        box = format_xml_box(figure.attrib["bbox"].split(","), width, height)
        box[0] = np.clip(box[0], 0, width)
        box[1] = np.clip(box[1], 0, height)
        box[2] = np.clip(box[2], 0, width)
        box[3] = np.clip(box[3], 0, height)
        box = [int(x) for x in box]
        for child in figure:
            if child.tag == "image":
                page_matrix[box[1]:box[3], box[0]:box[2]] = 255
    image_blocks = extract_image_regions(page_matrix)
    return image_blocks

def convert(infile, outfile, rotation=0):
    debug = 0
    password = ''
    pagenos = set()
    maxpages = 0
    codec = 'utf-8'
    caching = True
    laparams = LAParams()

    PDFDocument.debug = debug
    PDFParser.debug = debug
    CMapDB.debug = debug
    PDFResourceManager.debug = debug
    PDFPageInterpreter.debug = debug
    PDFDevice.debug = debug

    rsrcmgr = PDFResourceManager(caching=caching)
    outfp = open(outfile, 'wb')
    device = XMLConverter(rsrcmgr, outfp, codec=codec, laparams=laparams)
    fp = open(infile, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.get_pages(fp, pagenos,
                                    maxpages=maxpages, password=password,
                                    caching=caching, check_extractable=True):
        page.rotate = (page.rotate + rotation) % 360
        interpreter.process_page(page)
    fp.close()
    device.close()
    outfp.close()