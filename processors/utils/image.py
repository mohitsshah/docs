import subprocess
import os
import numpy as np
import cv2
import re

# TODO: Hide subprocess output messages


def convert_image(infile, outfile, overwrite):
    dpi = 300
    if os.path.exists(outfile) and not overwrite:
        return outfile
    cmd = "convert -density %d -units PixelsPerInch %s -background white %s" % (
        dpi, infile, outfile)
    try:
        subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        raise Exception("Image Conversion Error, File: %s" % infile)
    return outfile


def convert_page_to_image(page_id, infile, outfile, overwrite, resample=True):
    dpi = 300
    num = int(page_id)
    if os.path.exists(outfile) and not overwrite:
        return outfile
    if resample:
        cmd = "convert -density %d -units PixelsPerInch %s[%d] %s" % (
            dpi, infile, num - 1, outfile)
    else:
        cmd = "convert -units PixelsPerInch %s[%d] %s" % (
            infile, num - 1, outfile)
    try:
        subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        raise Exception("Image Conversion Error, Page: %s" % page_id)
    return outfile


def convert_pdf_to_image(infile, outfile, overwrite):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    def sort_nicely(l):
        """ Sort the given list in the way that humans expect.
        """
        l.sort(key=alphanum_key)

    dpi = 300
    images_dir, _ = os.path.split(outfile)
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0 and not overwrite:
        files = os.listdir(images_dir)
        files = [os.path.join(images_dir, f) for f in files]
        sort_nicely(files)
        return files

    cmd = "convert -scene 1 -density %s -units PixelsPerInch %s %s" % (
        dpi, infile, outfile)
    try:
        subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        raise Exception("PDF to PNG Conversion Error")
    files = os.listdir(images_dir)
    sort_nicely(files)
    files = [os.path.join(images_dir, f) for f in files]
    return files


def crop_image(infile, box, overwrite, page_width, page_height, padding=0):
    pad = padding // 2
    new_box = list(box)
    new_box[0] = np.clip(box[0] - pad, 0, page_width)
    new_box[1] = np.clip(box[1] - pad, 0, page_height)
    new_box[2] = np.clip(box[2] + pad, 0, page_width)
    new_box[3] = np.clip(box[3] + pad, 0, page_height)
    offsets = [box[0] - new_box[0], box[1] - new_box[1],
               box[2] - new_box[2], box[3] - new_box[3]]
    bbox = [300 * float(b) / 72 for b in new_box]
    width = int(bbox[2] - bbox[0])
    height = int(bbox[3] - bbox[1])
    x0 = int(bbox[0])
    y0 = int(bbox[1])
    crop_params = str(width) + 'x' + str(height) + \
        '+' + str(x0) + '+' + str(y0)
    outfile = infile[0:-4] + '-'
    outfile += crop_params + '.png'
    if os.path.exists(outfile) and not overwrite:
        return outfile, offsets
    cmd = 'convert -crop ' + crop_params + ' ' + infile + ' ' + outfile
    try:
        subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        raise Exception("Image Conversion Error during Cropping.")

    return outfile, offsets


def rotate_image(image_file, rotation):
    cmd = "convert -density 300 -units PixelsPerInch -rotate %d %s %s" % (
        rotation, image_file, image_file)
    try:
        subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        raise Exception("Image Conversion Error during Rotation.")
    return image_file
