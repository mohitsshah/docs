"""PDF to JSON Conversion
This module accepts a PDF file (text, scanned, or a mix of both) and converts it to a JSON file.
The JSON file captures the words in the document along with their bounding boxes.
Returns:
    boolean -- Flag indicates if the file was successfully converted.
"""

import os
import pickle
import json
import shutil
import xml.etree.ElementTree
import cv2
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed
import processors.utils.ocr as tess_utils
import processors.utils.image as img_utils
import processors.utils.xml as xml_utils
import processors.utils.segmentation as seg_utils
import processors.utils.headers_footers as hf_utils
import processors.utils.toc as toc_utils
from processors.utils.model import DocumentModel

class Processor(object):
    """
    PDF Processor
    """

    def __init__(self, source_file, dst, tessdata, overwrite, cleanup, oem, store_results, debug_segmentation, restore_results):
        """Initialization

        Arguments:
            source_file {str} -- Path to the source file
            dst {str} -- Path to the destination directory
            tessdata {str} -- Path to tessdata model
            overwrite {bool} -- If True, overwrite existing json file
            cleanup {true} -- If True, remove unwanted files after completion
            oem {string} -- Tesseract OCR Engine Mode (OEM). Two options - v4 (LSTM-based) or v3 (Legacy Tesseract)
            store_results {true} -- If True, store intermediate results. Useful for debugging and experimenting.
        """

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
        self.restore_results = restore_results
        self.debug_segmentation = debug_segmentation
        self.images_to_keep = []

    def clean_dirs(self):
        """
        Clean up unwanted directories and files after completion.
        """
        if not self.store_results:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
        if os.path.exists(self.images_dir):
            images_list = [os.path.join(self.images_dir, x)
                           for x in os.listdir(self.images_dir)]
            images_to_remove = [
                x for x in images_list if x not in self.images_to_keep]
            for image in images_to_remove:
                os.remove(image)
        os.remove(self.source_file)

    def save_document(self, document):
        """Serialize the final document to a JSON file

        Arguments:
            document {object} -- Document object
        """

        path = os.path.join(self.output_dir, self.name + ".json")
        with open(path, "w") as fi:
            json.dump(document, fi)

    def save_intermediate_results(self, page_id, results):
        """Save intermediate results

        Arguments:
            page_id {str} -- Page number
            results {object} -- Intermediate results object
        """

        path = os.path.join(self.tmp_dir, "intermediate.pkl")
        if os.path.exists(path):
            data = pickle.load(open(path, "rb"))
        else:
            data = {}
        data[page_id] = results
        with open(path, "wb") as fi:
            pickle.dump(data, fi)

    def load_intermediate_results(self, page_id):
        """Load intermediate results for a specific page

        Arguments:
            page_id {str} -- Page number

        Returns:
            object -- Intermediate results object
        """

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

    def get_words_from_image(self, image_blocks, width, height, page_id):
        """Process image blocks found in the page
        1. Convert the page to image
        2. Identify if the page is an image
        3. (Optional) Crop the image
        4. Perform OCR

        Arguments:
            image_blocks {list} -- List of bounding box information for each image block
            width {int} -- Width of the page
            height {int} -- Height of the page
            page_id {str} -- Page number

        Returns:
            regions [list] -- List of bounding box information for each image region
            regions_words[list] -- List of words along with bounding boxes for each image region
            raw_images[list] -- List of file paths for each image
            is_page_image[bool] -- If True, the entire page is detected as an image
            padding[float] -- Padding amount applied to the image
            orientation[float] -- Orientation of the image as detected by OSD
        """

        regions = []
        regions_words = []
        raw_images = []
        is_page_image = False
        padding = 10
        orientation = 0.
        image_file = None
        if not image_blocks:
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

    def make_segments(self, words, image_regions, image_words, image_paths, page_id, width, height, is_page_image):
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
        blocks = []

        for region, region_words, image_path in zip(image_regions, image_words, image_paths):
            if region_words:
                words.extend(region_words)
            else:
                block = list(region)
                block.extend(["IMAGE", image_path])
                blocks.append(block)

        words = sorted(words, key=lambda x: (x[1], x[0]))
        new_words = []
        for word in words:
            if (word[2] - word[0] == int(width)) and (word[3] - word[1] == int(height)):
                continue
            new_words.append(word)
        words = new_words

        if self.debug_segmentation:
            debug_dir = os.path.join(self.output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            outfile = os.path.join(debug_dir, "%s-%s.png" %
                                   (self.name, page_id))
            debug_image = img_utils.convert_page_to_image(
                page_id, self.source_file, outfile, overwrite=False, resample=False)
            debug_image = cv2.imread(debug_image)

        _, page_matrix, median_height = seg_utils.make_image_from_words(
            words, width, height)

        tb_cuts = seg_utils.cut_segment(page_matrix)
        lr_cuts = []
        for t_b in tb_cuts:
            segment_image = page_matrix[t_b[0]:t_b[1], :].T
            l_r = seg_utils.cut_segment(segment_image)
            lr_cuts.append(l_r)

            # Uncomment lines below to see top-bottom and left-right cuts

            # if self.debug_segmentation:
                # cv2.rectangle(debug_image, (0, t_b[0]), (int(
                #     width), t_b[1]), (255, 255, 0), 1)
                # for xx in l_r:
                #     cv2.rectangle(debug_image, (xx[0], t_b[0]),
                #                   (xx[1], t_b[1]), (255, 0, 255), 1)

        t_b_threshold = 0.75
        if is_page_image:
            t_b_threshold = 1.5
        segments = seg_utils.label_segments(tb_cuts, lr_cuts, page_matrix)
        segments = seg_utils.merge_table_neighbors(
            segments, page_matrix, words, median_height, t_b_threshold)
        segments = seg_utils.merge_paragraph_neighbors(
            segments, page_matrix, words, median_height, t_b_threshold)
        segments = seg_utils.merge_consecutive_tables(segments, tb_cuts, lr_cuts, page_matrix)
        blocks.extend(seg_utils.make_blocks(segments, page_matrix, words))
        if self.debug_segmentation:
            for block in blocks:
                x_0, y_0, x_1, y_1, label, _ = block
                if label == "PARA":
                    color = (0, 255, 0)
                elif label == "TABLE":
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(debug_image, (x_0, y_0),
                              (x_1, y_1), color, 1)
            cv2.imshow("Page", debug_image)
            cv2.waitKey()

        return blocks

    def make_page(self, root=None, image_file=None, page_id=None, width=None, height=None):
        """[summary]

        Keyword Arguments:
            root {[type]} -- [description] (default: {None})
            image_file {[type]} -- [description] (default: {None})
            page_id {[type]} -- [description] (default: {None})
            width {[type]} -- [description] (default: {None})
            height {[type]} -- [description] (default: {None})

        Returns:
            [type] -- [description]
        """

        text_words = []
        text_figure_words = []
        image_words = []
        image_regions = []
        raw_images = []

        if not self.restore_results:
            if root is not None:
                selector = "./page[@id='%s']" % page_id
                tree = root.find(selector)
                text_words = xml_utils.scan_texts(tree, width, height)
                text_figure_words = xml_utils.scan_figures(tree, width, height)
                image_blocks = xml_utils.scan_images(tree, width, height)
                image_regions, image_words, raw_images, is_page_image, padding, orientation = self.get_words_from_image(
                    image_blocks, width, height, page_id)

            if (image_file) or (not text_words and not text_figure_words and not image_regions):
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

        else:
            intermediate_data = self.load_intermediate_results(page_id)
            text_words = intermediate_data["text_words"]
            text_figure_words = intermediate_data["text_figure_words"]
            image_regions = intermediate_data["image_regions"]
            image_words = intermediate_data["image_words"]
            raw_images = intermediate_data["raw_images"]
            is_page_image = intermediate_data["is_page_image"]
            padding = intermediate_data["padding"]
            orientation = intermediate_data["orientation"]
            self.images_to_keep = intermediate_data["images_to_keep"]

        if is_page_image:
            words = list(image_words[0])
            image_regions = []
            image_words = []
        else:
            words = text_words + text_figure_words

        segments = self.make_segments(
            words, image_regions, image_words, raw_images, page_id, width, height, is_page_image)

        page = {"page_number": int(page_id), "width": width, "height": height,
                "is_page_image": is_page_image, "orientation": orientation, "segments": []}
        for segment in segments:
            x_0, y_0, x_1, y_1, label, data = segment
            page_segment = {"bbox": [int(x_0), int(y_0), int(
                x_1), int(y_1)], "label": label, "content": data}
            page["segments"].append(page_segment)

        return page

    def make_json(self, xml_file=None, images=None, downsample=True):
        """[summary]

        Keyword Arguments:
            xml_file {[type]} -- [description] (default: {None})
            images {[type]} -- [description] (default: {None})

        Returns:
            [type] -- [description]
        """

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
                height, width, _ = image.shape
                height = (height*72/300.)
                width = (width*72/300.)
                width = float(width)
                height = float(height)
                page = self.make_page(
                    image_file=image_file, page_id=page_id, width=width, height=height)
                document["pages"].append(page)
                document["num_pages"] += 1
        toc_utils.add_toc(document)
        hf_utils.add_headers_footers(document)
        return document

    def make_xml(self):
        """Convert PDF to XML using pdfminer

        Returns:
            xml_file [str] -- Path to xml file
        """

        xml_file = os.path.join(self.tmp_dir, self.name + ".xml")
        if os.path.exists(xml_file) and not self.overwrite:
            try:
                _ = xml.etree.ElementTree.parse(xml_file)
                return xml_file
            except Exception:
                raise PDFTextExtractionNotAllowed
        xml_utils.convert(self.source_file, xml_file)
        return xml_file

    def run(self):
        """Run the PDF to JSON conversion on a single PDF

        Returns:
            status [bool] -- If True, conversion was successful
        """

        json_file = os.path.join(self.output_dir, self.name + ".json")
        if os.path.exists(json_file) and not self.overwrite:
            return True

        if self.source_file.endswith(".pdf"):
            try:
                xml_file = self.make_xml()
                data = self.make_json(xml_file=xml_file, images=None)
            except PDFTextExtractionNotAllowed:
                outfile = os.path.join(self.images_dir, self.name + ".png")
                images = img_utils.convert_pdf_to_image(
                    infile=self.source_file, outfile=outfile, overwrite=self.overwrite)
                data = self.make_json(xml_file=None, images=images)
        else:
            outfile = os.path.join(self.images_dir, self.name + ".png")
            images = img_utils.convert_tiff_to_image(
                infile=self.source_file, outfile=outfile, overwrite=self.overwrite)
            data = self.make_json(xml_file=None, images=images)

        document = DocumentModel()
        self.save_document(document.create(data, self.source_file, "pdf"))
        if self.cleanup:
            self.clean_dirs()
        return True
