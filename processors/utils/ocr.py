import os
import subprocess
from tesserocr import PyTessBaseAPI, RIL, PSM, iterate_level, OEM


class OSD(object):
    def __init__(self, image_file, tessdata):
        self.image_file = image_file
        self.tessdata = tessdata
        api = PyTessBaseAPI(path=tessdata, psm=PSM.OSD_ONLY)
        api.SetImageFile(image_file)
        self.api = api

    def perform_osd(self):
        api = self.api
        osd = api.DetectOrientationScript()
        if osd is not None:
            orientation = osd["orient_deg"]
            rotation = (360 - orientation) if orientation != 0 else 0
        else:
            rotation = 0
        return rotation


class OCR(object):
    def __init__(self, image_file, tessdata, oem_mode):
        if oem_mode == "v3":
            oem = OEM.TESSERACT_ONLY
        else:
            oem = OEM.LSTM_ONLY
        api = PyTessBaseAPI(path=tessdata, psm=PSM.AUTO_OSD, oem=oem)
        api.SetImageFile(image_file)
        self.api = api

    def perform_ocr(self, x_offset=0, y_offset=0, pad_offset=None):
        dpi = 300
        api = self.api
        api.Recognize()
        ri = api.GetIterator()
        words = []
        level = RIL.WORD
        for r in iterate_level(ri, level):
            try:
                word = r.GetUTF8Text(level)
                font_info = r.WordFontAttributes()
                bbox = list(r.BoundingBox(level))
                bbox = [float(b) for b in bbox]
                bbox = [float(b) * 72 / dpi for b in bbox]
                bbox[0] += x_offset
                bbox[2] += x_offset
                bbox[1] += y_offset
                bbox[3] += y_offset
                if pad_offset is not None:
                    bbox[0] += pad_offset[0]
                    bbox[1] += pad_offset[1]
                    bbox[2] += pad_offset[0]
                    bbox[3] += pad_offset[1]
                word = word.rstrip().lstrip()
                if word:
                    bbox.append(font_info)
                    bbox.append(word)
                    words.append(bbox)
            except Exception as e:
                pass

        words = sorted(words, key=lambda x: (x[1], x[0]))
        return words
