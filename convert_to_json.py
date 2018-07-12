"""Convert Documents to JSON
A program to convert documents (PDFs, TIFFs, Word and Excel documents) to a standard JSON format.
"""

import os
import argparse
import shutil
import glob
import logging
import datetime
from processors import pdf, xlsx, word_docx

# Setting up logger
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, str(datetime.datetime.now()) + ".log")
logging.basicConfig(filename=LOG_FILE,
                    format='%(levelname)s (%(asctime)s): %(message)s', level=logging.INFO)

ALLOWED_EXTENSIONS = ["pdf", "doc", "docx",
                      "xls", "xlsx", "xlsm", "tiff", "tif"]

# ALLOWED_EXTENSIONS = ["docx"]

def get_files(src, formats):
    """Find files in the source directory matching the allowed or user-selected extensions

    Arguments:
        src {str} -- Path to the source directory
        formats {list} -- List of file formats

    Returns:
        [list] -- List of absolute paths to files
    """

    files = []
    for extension in formats:
        ext_files = glob.glob(os.path.join(
            src, "**/*." + extension), recursive=True)
        files += ext_files
    return files


def process_files(files, dst, tessdata, oem, overwrite, cleanup, store_results, debug_segmentation, restore_results):
    """Process files and convert them to JSON.

    Arguments:
        files {list} -- List of absolute paths to files
        dst {str} -- Absolute path of output directory
        tessdata {str} -- Absolute path to Tesseract model. Required only for PDF formats.
        oem {str} -- OEM Mode to use for Tesseract. Options: v4 (LSTM) or v3 (LEGACY TESSERACT). Required only for PDF formats.
        overwrite {bool} -- Overwrite existing results. If True, will perform the entire conversion process from scratch.
        cleanup {bool} -- Cleanup files/directories after conversion.
        store_results {bool} -- Store intermediate results. Useful for debugging or resuming a process later on.
    """

    logging.info("Total Files: %d", len(files))
    for num, file in enumerate(files, 1):
        _, filename = os.path.split(file)
        name, ext = os.path.splitext(filename)
        # Sanitize the file name. No whitespaces are allowed.
        clean_name = name.replace(" ", "_")
        clean_filename = filename.replace(" ", "_")
        # Make the output directory.
        output_dir = os.path.join(dst, clean_name)
        os.makedirs(output_dir, exist_ok=True)
        source_file = os.path.join(output_dir, clean_filename)
        shutil.copy2(file, source_file)
        logging.info("Processing File: %s (%d/%d)", file, num, len(files))
        if ext == ".pdf" or ext.startswith(".tif"):
            job = pdf.Processor(source_file, output_dir,
                                tessdata, overwrite, cleanup, oem, store_results, debug_segmentation, restore_results)
            job.run()
        elif ext == ".doc":
            pass
        elif ext == ".docx":
            job = word_docx.Processor(source_file, output_dir,
                                tessdata, overwrite, cleanup, oem, store_results)
            job.run()
        elif ext.startswith(".xls"):
            job = xlsx.Processor(source_file, output_dir, overwrite, cleanup)
            job.run()
        else:
            raise Exception("Unknown extension (%s)" % file)


def run(src, dst, formats, tessdata, oem, overwrite, cleanup, store_results, debug_segmentation, restore_results):
    src = os.path.abspath(src)
    if not os.path.exists(src):
        raise Exception("Source directory (%s) does not exist." %
                        (src))
    if not os.path.isdir(src):
        raise Exception("Source (%s) is not a directory." % (src))
    formats = formats if formats else ALLOWED_EXTENSIONS
    tessdata = os.path.abspath(
        tessdata) if tessdata else None
    if tessdata is not None:
        if not os.path.exists(tessdata):
            raise Exception(
                "Tessdata directory (%s) does not exist." % (tessdata))
    overwrite = bool(overwrite)
    cleanup = bool(cleanup)
    debug_segmentation = bool(debug_segmentation)
    store_results = bool(store_results)
    restore_results = bool(restore_results)
    files = get_files(src, formats)
    if not files:
        raise Exception("Found 0 files in %s" % (src))
    dst = os.path.abspath(dst)
    os.makedirs(dst, exist_ok=True)
    process_files(files, dst, tessdata,
                  oem, overwrite, cleanup, store_results, debug_segmentation, restore_results)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        "Command line arguments for Document Conversion to JSON")
    PARSER.add_argument("-s",
                        "--src",
                        type=str,
                        required=True,
                        dest="src",
                        help="Source directory of files/Path to a single file")
    PARSER.add_argument("-d",
                        "--dst",
                        type=str,
                        required=True,
                        dest="dst",
                        help="Destination directory")
    PARSER.add_argument("-f",
                        "--formats",
                        nargs="*",
                        type=str,
                        choices=["pdf", "tiff", "tif", "xls",
                                 "xlsx", "xlsm", "doc", "docx"],
                        dest="formats",
                        help="File formats to process. Leave empty for all formats.")
    PARSER.add_argument("-t",
                        "--tessdata",
                        type=str,
                        dest="tessdata",
                        required=True,
                        help="Path to Tessdata model (v4) for tesserocr. Applies to PDF/TIFF/Images")
    PARSER.add_argument("-oem",
                        type=str,
                        dest="oem",
                        choices=["v4", "v3"],
                        default="v4",
                        help="OEM Mode for Tesseract")
    PARSER.add_argument("-o",
                        "--overwrite",
                        type=int,
                        choices=[0, 1],
                        default=0,
                        dest="overwrite",
                        help="Overwrite files")
    PARSER.add_argument("-c",
                        "--cleanup",
                        type=int,
                        choices=[0, 1],
                        default=1,
                        dest="cleanup",
                        help="Clean up temporary files/directories")
    PARSER.add_argument("-st",
                        "--store_results",
                        type=int,
                        choices=[0, 1],
                        default=1,
                        dest="store_results",
                        help="Store intermediate results (useful for debugging)")
    PARSER.add_argument("-ds",
                        "--debug_segmentation",
                        type=int,
                        choices=[0, 1],
                        default=0,
                        dest="debug_segmentation",
                        help="Debug segmentation results")
    PARSER.add_argument("-rst",
                        "--restore_results",
                        type=int,
                        choices=[0, 1],
                        default=0,
                        dest="restore_results",
                        help="Restore intermediate results (useful for debugging)")
    FLAGS = PARSER.parse_args()

    # try:
    run(**vars(FLAGS))
    # except Exception as error:
    #     logging.critical(str(error))
