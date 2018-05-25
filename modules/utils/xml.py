import sys
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
                chars[-1][2], chars[-1][3], "".join(tmp)]
    return word_box

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