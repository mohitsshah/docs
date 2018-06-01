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

    def scan_texts(self, tree, width, height):
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
                            if "bbox" in word.attrib:
                                box = xml_utils.format_xml_box(
                                    word.attrib["bbox"].split(","), width, height)
                                if len(ch) == 0:
                                    if len(chars) > 0:
                                        words.append(xml_utils.make_word_box(chars))
                                        chars = []
                                else:
                                    tmp = box + [ch]
                                    chars.append(tmp)
                            else:
                                if len(chars) > 0:
                                    words.append(xml_utils.make_word_box(chars))
                                    chars = []
                    if len(chars) > 0:
                        words.append(xml_utils.make_word_box(chars))
                        chars = []
        words = [[int(w[0]), int(w[1]), int(w[2]), int(w[3]), w[-1]] for w in words]
        words = sorted(words, key=lambda x: (x[1], x[0]))
        return words

    def is_y_similar(self, ry0, ry1, y0, y1):
        if ry0 == y0:
            return True
        if ry0 < y0 < ry1:
            return True
        return False

    def scan_figures(self, tree, width, height):
        figures = tree.findall("figure")
        unique_y = {}
        chars = []
        for figure in figures:
            for child in figure:
                if child.tag == "text":
                    box = xml_utils.format_xml_box(
                        child.attrib["bbox"].split(","), width, height)
                    unique_y[(box[1], box[3])] = True
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
                    if self.is_y_similar(ref_y0, ref_y1, y0, y1):
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
                    if g > 2. * med:
                        tmp = []
                        for l in word:
                            tmp.append(l[-1])
                        if len(tmp) > 0:
                            bb = [word[0][0], word[0][1], word[-1]
                                  [2], word[-1][3], "".join(tmp)]
                            words.append(bb)
                        word = [cc]
                    else:
                        word.append(cc)
            tmp = []
            for l in word:
                tmp.append(l[-1])
            if len(tmp) > 0:
                bb = [word[0][0], word[0][1], word[-1]
                      [2], word[-1][3], "".join(tmp)]
                words.append(bb)
        words = [[int(w[0]), int(w[1]), int(w[2]), int(w[3]), w[-1]] for w in words]
        words = sorted(words, key=lambda x: (x[1], x[0]))
        return words

    def scan_images(self, tree, width, height, page_id):
        regions = []
        regions_words = []
        figures = tree.findall("figure")
        image_file = None
        if len(figures) == 0:
            return regions, regions_words, False
        image_file = img_utils.convert_page_to_image(page_id,
                                                  os.path.join(
                                                      self.raw_dir, self.filename),
                                                  os.path.join(
                                                      self.image_dir, "%s-%s.png" % (self.name, page_id)),
                                                  self.overwrite)
        page_matrix = np.zeros((int(height), int(width)))
        for figure in figures:
            box = xml_utils.format_xml_box(
                figure.attrib["bbox"].split(","), width, height)
            box[0] = np.clip(box[0], 0, width)
            box[1] = np.clip(box[1], 0, height)
            box[2] = np.clip(box[2], 0, width)
            box[3] = np.clip(box[3], 0, height)
            box = [int(x) for x in box]
            for child in figure:
                if child.tag == "image":
                    page_matrix[box[1]:box[3], box[0]:box[2]] = 255

        regions = img_utils.extract_image_regions(page_matrix)
        is_page_image = False
        for box in regions:
            if box[2] == width and box[3] == height:
                is_page_image = True
                osd = tess_utils.OSD(image_file, self.tessdata, self.osd_mode)
                rotation = osd.perform_osd()
                if rotation != 0:
                    image_file = img_utils.rotate_image(image_file, rotation)
                ocr = tess_utils.OCR(image_file, self.tessdata)
                words = ocr.perform_ocr(x_offset=0, y_offset=0)
                words = [[int(w[0]), int(w[1]), int(w[2]), int(w[3]), w[-1]] for w in words]
                regions_words.append(words)
            else:
                cropped_file = img_utils.crop_image(
                    image_file, box, self.overwrite)
                ocr = tess_utils.OCR(cropped_file, self.tessdata)
                words = ocr.perform_ocr(
                    x_offset=box[0], y_offset=box[1])
                words = [[int(w[0]), int(w[1]), int(w[2]), int(w[3]), w[-1]] for w in words]
                regions_words.append(words)

        return regions, regions_words, is_page_image

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

    def make_page(self, root, page_id, width, height):
        selector = "./page[@id='%s']" % page_id
        tree = root.find(selector)
        text_words = self.scan_texts(tree, width, height)
        text_words += self.scan_figures(tree, width, height)
        image_regions, image_words, is_page_image = self.scan_images(
            tree, width, height, page_id)
        if is_page_image:
            words = list(image_words[0])
            image_regions = []
            image_words = []
        else:
            if len(text_words) == 0 and len(image_regions) == 0:
                image_file = img_utils.convert_page_to_image(page_id,
                                                        os.path.join(
                                                            self.raw_dir, self.filename),
                                                        os.path.join(
                                                            self.image_dir, "%s-%s.png" % (self.name, page_id)),
                                                        self.overwrite)
                osd = tess_utils.OSD(image_file, self.tessdata, self.osd_mode)
                rotation = osd.perform_osd()
                if rotation != 0:
                    image_file = img_utils.rotate_image(image_file, rotation)
                ocr = tess_utils.OCR(image_file, self.tessdata)
                words = ocr.perform_ocr()
                image_regions = []
                image_words = []
                is_page_image = True
            else:
                words = text_words
        segments = self.make_segments(words, image_regions, image_words, page_id, width, height)
        return segments, is_page_image


    def make_json(self, xml_file):
        tree = xml.etree.ElementTree.parse(xml_file)
        root = tree.getroot()
        document = {"total_pages": None, "pages": []}
        page_count = 0
        for child in root:
            tag = child.tag
            if tag == 'page':
                page = {"width": None, "height": None, "is_page_image": None, "segments": []}
                obj = xml_utils.get_attribs(child.items())
                page_id = obj["id"]
                width, height = obj["bbox"].split(",")[2:]
                width = float(width)
                height = float(height)
                page["width"] = width
                page["height"] = height
                segments, is_page_image = self.make_page(root, page_id, width, height)
                page["is_page_image"] = is_page_image
                for segment in segments:
                    x0, y0, x1, y1, label, data = segment
                    page_segment = {"bbox": [int(x0), int(y0), int(x1), int(y1)], "label": label, "content": data}
                    page["segments"].append(page_segment)
                document["pages"].append(page)
                page_count += 1
        document["total_pages"] = page_count
        return document

    def make_xml(self):
        xml_file = os.path.join(self.tmp_dir, self.name + ".xml")
        if os.path.exists(xml_file) and not self.overwrite:
            return xml_file
        xml_utils.convert(os.path.join(self.raw_dir, self.filename), xml_file)
        return xml_file

    def run(self):
        json_file = os.path.join(self.output_dir, self.name + ".json")
        if os.path.exists(json_file) and not self.overwrite:
            with open(json_file, "r") as fi:
                json_document = json.load(fi)
        else:
            xml_file = self.make_xml()
            json_document = self.make_json(xml_file)
            with open(json_file, "w") as fi:
                fi.write(json.dumps(json_document))
        if self.cleanup:
            self.clean_dirs()