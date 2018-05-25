import subprocess
import os
import numpy as np
import cv2

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
    
def crop_image(infile, box, overwrite):
    bbox = [300 * float(b) / 72 for b in box]
    width = int(bbox[2] - bbox[0])
    height = int(bbox[3] - bbox[1])
    x0 = int(bbox[0])
    y0 = int(bbox[1])
    crop_params = str(width) + 'x' + str(height) + \
        '+' + str(x0) + '+' + str(y0)
    outfile = infile[0:-4] + '-'
    outfile += crop_params + '.png'
    if os.path.exists(outfile) and not overwrite:
        return outfile
    cmd = 'convert -crop ' + crop_params + ' ' + infile + ' ' + outfile
    try:
        subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        raise Exception("Image Conversion Error during Cropping.")
    return outfile

def rotate_image(image_file, rotation):
    cmd = "convert -density 300 -units PixelsPerInch -rotate %d %s %s" % (
        rotation, image_file, image_file)
    try:
        subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        raise Exception("Image Conversion Error during Rotation.")
    return image_file

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
