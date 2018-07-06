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
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed
import re


class Processor(object):
    def __init__(self, source_file, args):
        _, filename = os.path.split(source_file)
        name, _ = os.path.splitext(filename)
        output_dir = os.path.join(args["dst"], name)
        tmp_dir = os.path.join(output_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        self.source_file = source_file
        self.filename = filename
        self.name = name
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir
        self.images_dir = images_dir
        self.tessdata = args["tessdata"]
        self.overwrite = args["overwrite"]
        self.cleanup = args["cleanup"]
        self.oem_mode = args["oem"]
        self.store_results = args["store_results"]
        self.images_to_keep = []

    def clean_dirs(self):
        if not self.store_results:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
        if os.path.exists(self.images_dir):
            images_list = [os.path.join(self.images_dir, x)
                           for x in os.listdir(self.images_dir)]
            images_to_remove = [
                x for x in images_list if x not in self.images_to_keep]
            for x in images_to_remove:
                os.remove(x)
        os.remove(self.source_file)

    def save_document(self, document):
        path = os.path.join(self.output_dir, self.name + ".json")
        with open(path, "w") as fi:
            json.dump(document, fi)

    def save_intermediate_results(self, page_id, results):
        path = os.path.join(self.tmp_dir, "intermediate.pkl")
        if os.path.exists(path):
            data = pickle.load(open(path, "rb"))
        else:
            data = {}
        data[page_id] = results
        with open(path, "wb") as fi:
            pickle.dump(data, fi)

    def load_intermediate_results(self, page_id):
        path = os.path.join(self.tmp_dir, "intermediate.pkl")
        if not os.path.exists(path):
            return None
        data = pickle.load(open(path, "rb"))
        try:
            results = data[page_id]
            return results
        except KeyError:
            return None

    def get_words_from_ocr(self, image_file, x_offset=0, y_offset=0, pad_offset=0, osd=True):
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

    def get_words_from_image(self, image_blocks, width, height, page_id):
        regions = []
        regions_words = []
        raw_images = []
        is_page_image = False
        padding = 10
        orientation = 0.
        image_file = None
        if len(image_blocks) == 0:
            return regions, regions_words, raw_images, is_page_image, padding, orientation
        image_file = img_utils.convert_page_to_image(page_id, self.source_file, os.path.join(
            self.images_dir, "%s-%s.png" % (self.name, page_id)), self.overwrite)

        for box in image_blocks:
            if int(box[2] - box[0]) == int(width) and int(box[3] - box[1]) == int(height):
                is_page_image = True
                words, orientation = self.get_words_from_ocr(
                    image_file, pad_offset=None)
                regions_words.append(words)
                regions.append(box)
                raw_images.append(image_file)
                break
            else:
                cropped_file, pad_offset = img_utils.crop_image(
                    image_file, box, self.overwrite, width, height, padding=padding)
                words, _ = self.get_words_from_ocr(
                    cropped_file, x_offset=box[0], y_offset=box[1], pad_offset=pad_offset, osd=False)
                regions_words.append(words)
                regions.append(box)
                raw_images.append(cropped_file)
        self.images_to_keep.extend(raw_images)
        return regions, regions_words, raw_images, is_page_image, padding, orientation

    def make_segments(self, words, image_regions, image_words, page_id, width, height):
        page_matrix_raw, page_matrix, median_height = seg_utils.image_from_words(
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

        for region, region_words in zip(image_regions, image_words):
            if len(region_words) > 0:
                _, region_matrix, median_height = seg_utils.image_from_words(
                    region_words, width, height)
                tmp1 = seg_utils.cut_segment(region_matrix)
                tmp2 = tmp1[1:] + [np.shape(region_matrix)[0]]
                tb_cuts = [(t1, t2) for t1, t2 in zip(tmp1, tmp2)]
                lr_cuts = []
                # Collect cuts for each row of the region
                for tb in tb_cuts:
                    segment_image = region_matrix[tb[0]:tb[1], :].T
                    lr = seg_utils.cut_segment(segment_image)
                    lr_cuts.append(lr[0:-1])

                segments = seg_utils.label_segments(
                    tb_cuts, lr_cuts, region_words, width, height, median_height)
                segments = seg_utils.merge_segments(
                    segments, region_matrix, region_words, width, height, median_height)
                segments = seg_utils.merge_consecutive_tables(segments)
                region_blocks = seg_utils.make_blocks(
                    segments, region_matrix, region_words)
                blocks += region_blocks
            else:
                block = list(region)
                block.extend(["IMAGE", []])
                blocks.append(block)

        blocks = sorted(blocks, key=lambda x: (x[1], x[0]))
        return blocks

    def make_page(self, root=None, image_file=None, page_id=None, width=None, height=None):
        text_words = []
        text_figure_words = []
        image_words = []
        image_regions = []
        raw_images = []

        if root is not None:
            selector = "./page[@id='%s']" % page_id
            tree = root.find(selector)
            text_words = xml_utils.scan_texts(tree, width, height)
            text_figure_words = xml_utils.scan_figures(tree, width, height)
            image_blocks = xml_utils.scan_images(tree, width, height)
            image_regions, image_words, raw_images, is_page_image, padding, orientation = self.get_words_from_image(
                image_blocks, width, height, page_id)

        if (image_file is not None) or len(text_words + text_figure_words) == 0 and len(image_regions) == 0:
            is_page_image = True
            padding = 0
            image_file = img_utils.convert_page_to_image(page_id,
                                                         self.source_file,
                                                         os.path.join(
                                                             self.images_dir, "%s-%s.png" % (self.name, page_id)),
                                                         self.overwrite)
            ocr_words, orientation = self.get_words_from_ocr(
                image_file, pad_offset=None)
            image_words.append(ocr_words)
            image_regions.append([0, 0, width, height])
            raw_images.append(image_file)
            self.images_to_keep.extend(raw_images)

        intermediate_data = {
            "text_words": text_words,
            "text_figure_words": text_figure_words,
            "image_regions": image_regions,
            "image_words": image_words,
            "raw_images": raw_images,
            "is_page_image": is_page_image,
            "padding": padding,
            "orientation": orientation,
            "images_to_keep": self.images_to_keep
        }

        if self.store_results:
            self.save_intermediate_results(page_id, intermediate_data)

        if is_page_image:
            words = list(image_words[0])
            image_regions = []
            image_words = []
        else:
            words = text_words + text_figure_words

        segments = self.make_segments(
            words, image_regions, image_words, page_id, width, height)

        page = {"page_number": int(page_id), "width": width, "height": height,
                "is_page_image": is_page_image, "orientation": orientation, "segments": []}
        for segment in segments:
            x0, y0, x1, y1, label, data = segment
            page_segment = {"bbox": [int(x0), int(y0), int(
                x1), int(y1)], "label": label, "content": data}
            page["segments"].append(page_segment)

        return page

    def make_json(self, xml_file=None, images=None):
        document = {"num_pages": 0, "pages": []}
        if xml_file is not None:
            tree = xml.etree.ElementTree.parse(xml_file)
            root = tree.getroot()
            for child in root:
                tag = child.tag
                if tag == 'page':
                    obj = xml_utils.get_attribs(child.items())
                    page_id = obj["id"]
                    page_orientation = float(obj["rotate"])
                    if page_orientation != 0:
                        page_orientation = 360. - page_orientation
                    width, height = obj["bbox"].split(",")[2:]
                    width = float(width)
                    height = float(height)
                    page = self.make_page(
                        root=root, page_id=page_id, width=width, height=height)
                    document["pages"].append(page)
                    document["num_pages"] += 1
        elif images is not None:
            for index, image_file in enumerate(images, 1):
                page_id = str(index)
                image = cv2.imread(image_file)
                height, width, channels = image.shape
                width = float(width)
                height = float(height)
                page = self.make_page(
                    image_file=image_file, page_id=page_id, width=width, height=height)
                document["pages"].append(page)
                document["num_pages"] += 1

        return document

    def make_xml(self):
        xml_file = os.path.join(self.tmp_dir, self.name + ".xml")
        if os.path.exists(xml_file) and not self.overwrite:
            try:
                _ = xml.etree.ElementTree.parse(xml_file)
                return xml_file
            except Exception:
                os.remove(xml_file)
        xml_utils.convert(self.source_file, xml_file)
        return xml_file

    def run(self):
        json_file = os.path.join(self.output_dir, self.name + ".json")
        if os.path.exists(json_file) and not self.overwrite:
            return True
        else:
            try:
                xml_file = self.make_xml()
                document = self.make_json(xml_file=xml_file, images=None)
            except PDFTextExtractionNotAllowed:
                images = img_utils.convert_pdf_to_image(infile=self.source_file, outfile=os.path.join(
                    self.images_dir, self.name + ".png"), overwrite=self.overwrite)
                document = self.make_json(xml_file=None, images=images)
            self.save_document(document)
            if self.cleanup:
                self.clean_dirs()
