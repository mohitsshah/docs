import os
import subprocess
from tesserocr import PyTessBaseAPI, RIL, PSM, iterate_level, OEM


class OSD(object):
    def __init__(self, image_file, tessdata, mode="tesserocr"):
        self.mode = mode
        self.image_file = image_file
        self.tessdata = tessdata
        if mode == "tesserocr":
            api = PyTessBaseAPI(path=tessdata, psm=PSM.OSD_ONLY)
            api.SetImageFile(image_file)
            self.api = api

    def perform_osd(self):
        if self.mode == "tesserocr":
            api = self.api
            osd = api.DetectOrientationScript()
            if osd is not None:
                orientation = osd["orient_deg"]
                rotation = (360 - orientation) if orientation != 0 else 0
            else:
                rotation = 0
            return rotation
        else:
            image_dir = "/".join(self.image_file.split("/")[0:-1])
            or_file = os.path.join(image_dir, "orientation.txt")
            options = "--psm 0 --tessdata-dir " + os.path.abspath(self.tessdata)
            cmd = "tesseract " + self.image_file + " stdout " + options + " > " + or_file
            try:
                subprocess.check_output(cmd, shell=True)
            except subprocess.CalledProcessError:
                if os.path.exists(or_file):
                    os.remove(or_file)
                raise Exception("Legacy Mode OSD Error.")
            or_text = open(or_file).read()
            orientation = 0
            if len(or_text) > 0:
                lines = or_text.split("\n")
                lines = [l for l in lines if len(l) > 0]
                tmp = {}
                for l in lines:
                    items = l.split(":")
                    tmp[items[0]] = items[1].rstrip().lstrip()
                orientation = int(tmp["Orientation in degrees"])
            if os.path.exists(or_file):
                os.remove(or_file)
            rotation = (360 - orientation) if orientation != 0 else 0
            return rotation


class OCR(object):
    def __init__(self, image_file, tessdata):
        api = PyTessBaseAPI(path=tessdata, psm=PSM.AUTO_OSD)
        api.SetImageFile(image_file)
        self.api = api

    def perform_ocr(self, x_offset=0, y_offset=0):
        dpi = 300
        api = self.api
        api.Recognize()
        ri = api.GetIterator()
        words = []
        level = RIL.WORD
        for r in iterate_level(ri, level):
            try:
                word = r.GetUTF8Text(level)
                bbox = list(r.BoundingBox(level))
                bbox = [float(b) for b in bbox]
                bbox = [float(b) * 72 / dpi for b in bbox]
                bbox[0] += x_offset
                bbox[2] += x_offset
                bbox[1] += y_offset
                bbox[3] += y_offset
                w = word.rstrip().lstrip()
                if len(w) > 0:
                    bbox.append(w)
                    words.append(bbox)
            except Exception:
                pass
        words = sorted(words, key=lambda x: (x[1], x[0]))
        return words



if __name__ == "__main__":
    # Mode: tesserocr
    m = OSD(image_file="/Users/mohitshah/Others/processed_docs/pdf_example/images/pdf_example-1.png",
            tessdata="/Users/mohitshah/Others/tessdata/v4")
    o = m.perform_osd()
    print(o)

    # Mode: legacy
    m = OSD(image_file="/Users/mohitshah/Others/processed_docs/pdf_example/images/pdf_example-1.png",
            tessdata="/Users/mohitshah/Others/tessdata/v4", mode="legacy")
    o = m.perform_osd()
    print(o)
