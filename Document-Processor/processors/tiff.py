import os
import pickle
import json
import xml.etree.ElementTree
import numpy as np
import subprocess
import cv2
import shutil
import processors.utils.ocr as tess_utils
import processors.utils.image as img_utils
import processors.utils.xml as xml_utils
import processors.utils.segmentation as seg_utils
import processors.utils.export as export_utils
import re


class Processor(object):
    def __init__(self, output_dir, args):
        self.output_dir = output_dir
        self.tessdata = args["tessdata"]
        self.overwrite = args["overwrite"]
        self.osd_mode = args["osd"]
        self.cleanup = args["cleanup"]
        self.raw_dir = os.path.join(self.output_dir, "raw")
        self.tmp_dir = os.path.join(self.output_dir, "tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.filename = os.listdir(self.raw_dir)[0]
        self.name, self.ext = self.filename.split(".")
        self.image_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

    def clean_dirs(self):
        if os.path.exists(self.raw_dir): shutil.rmtree(self.raw_dir)
        if os.path.exists(self.tmp_dir): shutil.rmtree(self.tmp_dir)
        if os.path.exists(self.image_dir): shutil.rmtree(self.image_dir)

    def is_y_similar(self, ry0, ry1, y0, y1):
        if ry0 == y0:
            return True
        if ry0 < y0 < ry1:
            return True
        return False

    def make_segments(self, words, image_regions, image_words, page_id, width, height):
        page_matrix_raw, page_matrix, median_height = seg_utils.image_from_words(words, width, height)

        tmp1 = seg_utils.cut_segment(page_matrix)
        tmp2 = tmp1[1:] + [np.shape(page_matrix)[0]]
        tb_cuts = [(t1, t2) for t1, t2 in zip(tmp1, tmp2)]
        lr_cuts = []
        # Collect cuts for each row of the page
        for tb in tb_cuts:
            segment_image = page_matrix[tb[0]:tb[1], :].T
            lr = seg_utils.cut_segment(segment_image)
            lr_cuts.append(lr[0:-1])

        segments = seg_utils.label_segments(tb_cuts, lr_cuts, words, width, height, median_height)
        segments = seg_utils.merge_segments(segments, page_matrix, words, width, height, median_height)
        segments = seg_utils.merge_consecutive_tables(segments)
        blocks = seg_utils.make_blocks(segments, page_matrix, words)

        for region, region_words in zip(image_regions, image_words):
            if len(region_words) > 0:
                region_matrix_raw, region_matrix, median_height = seg_utils.image_from_words(region_words, width, height)
                tmp1 = seg_utils.cut_segment(region_matrix)
                tmp2 = tmp1[1:] + [np.shape(region_matrix)[0]]
                tb_cuts = [(t1, t2) for t1, t2 in zip(tmp1, tmp2)]
                lr_cuts = []
                # Collect cuts for each row of the region
                for tb in tb_cuts:
                    segment_image = region_matrix[tb[0]:tb[1], :].T
                    lr = seg_utils.cut_segment(segment_image)
                    lr_cuts.append(lr[0:-1])

                segments = seg_utils.label_segments(tb_cuts, lr_cuts, region_words, width, height, median_height)
                segments = seg_utils.merge_segments(segments, region_matrix, region_words, width, height, median_height)
                segments = seg_utils.merge_consecutive_tables(segments)
                region_blocks = seg_utils.make_blocks(segments, region_matrix, region_words)
                blocks += region_blocks
            else:
                block = list(region)
                block.extend(["IMAGE", []])
                blocks.append(block)

        blocks = sorted(blocks, key=lambda x: (x[1], x[0]))
        return blocks

    def make_page(self, image_file, page_id, width, height):
        osd = tess_utils.OSD(image_file, self.tessdata, self.osd_mode)
        rotation = osd.perform_osd()
        if rotation != 0:
            image_file = img_utils.rotate_image(image_file, rotation)
        ocr = tess_utils.OCR(image_file, self.tessdata)
        words = ocr.perform_ocr()
        image_regions = []
        image_words = []
        segments = self.make_segments(words, image_regions, image_words, page_id, width, height)
        return segments


    def make_json(self, images):
        document = {"total_pages": len(images), "pages": []}        
        for page_id, image_path in enumerate(images, 1):
            page = {"width": None, "height": None, "is_page_image": True, "segments": []}
            image = cv2.imread(image_path)
            height, width, channels = image.shape
            page["width"] = width
            page["height"] = height
            segments = self.make_page(image_path, page_id, width, height)                    
            for segment in segments:
                x0, y0, x1, y1, label, data = segment
                page_segment = {"bbox": [int(x0), int(y0), int(x1), int(y1)], "label": label, "content": data}
                page["segments"].append(page_segment)
            document["pages"].append(page)
        return document

    def convert_to_png(self):
        def tryint(s):
            try:
                return int(s)
            except:
                return s

        def alphanum_key(s):
            """ Turn a string into a list of string and number chunks.
                "z23a" -> ["z", 23, "a"]
            """
            return [ tryint(c) for c in re.split('([0-9]+)', s) ]

        def sort_nicely(l):
            """ Sort the given list in the way that humans expect.
            """
            l.sort(key=alphanum_key)

        dpi = 300
        if os.path.exists(self.image_dir) and len(os.listdir(self.image_dir)) > 0 and not self.overwrite:
            files = os.listdir(self.image_dir)
            files = [os.path.join(self.image_dir, f) for f in files]
            sort_nicely(files)            
            return files
        outfile = os.path.join(self.image_dir, self.name + ".png")
        infile = os.path.join(self.raw_dir, self.filename)
        cmd = "convert -density %s -units PixelsPerInch %s %s" % (dpi, infile, outfile)
        try:
            subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError:
            raise Exception("TIFF to PNG Conversion Error")
        files = os.listdir(self.image_dir)
        sort_nicely(files)
        files = [os.path.join(self.image_dir, f) for f in files]
        return files

    def run(self):
        json_file = os.path.join(self.output_dir, self.name + ".json")
        if os.path.exists(json_file) and not self.overwrite:
            with open(json_file, "r") as fi:
                json_document = json.load(fi)
        else:
            images = self.convert_to_png()            
            json_document = self.make_json(images)
            with open(json_file, "w") as fi:
                fi.write(json.dumps(json_document))
        if self.cleanup:
            self.clean_dirs()