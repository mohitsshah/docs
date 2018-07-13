import os
import json
import glob
import argparse
import pandas
import binder.feature_extraction as feature_extraction
import binder.segmentation as segmentation

ALLOWED_EXTENSIONS = ["json"]


def get_files(src):
    files = []
    for extension in ALLOWED_EXTENSIONS:
        ext_files = glob.glob(os.path.join(
            src, "**/*." + extension), recursive=True)
        files += ext_files
    return files


def process_files(files, dst, overwrite):
    for file in files:
        _, file_name = os.path.split(file)
        name, _ = os.path.splitext(file_name)
        output_name = name + ".pkl"
        output_path = os.path.join(dst, output_name)
        features = None
        if not os.path.exists(output_path) or overwrite:
            features = feature_extraction.run(file)
            features.to_pickle(output_path)
        else:
            features = pandas.read_pickle(output_path)
        segments = segmentation.run(features)
        print (file)
        for segment in segments:
            print (segment)
        output_name = name + ".txt"
        output_path = os.path.join(dst, output_name)
        segments = "\n".join([", ".join(x) for x in segments])
        open(output_path, "w").write(segments)


def run(src, dst, overwrite):
    src = os.path.abspath(src)
    if not os.path.exists(src):
        raise Exception("Directory/File (%s) does not exist." % (src))
    files = get_files(src)
    if not files:
        raise Exception("Found 0 files in %s" % (src))

    dst = os.path.abspath(dst)
    os.makedirs(dst, exist_ok=True)

    process_files(files, dst, overwrite)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        "Command line arguments for Information Extraction")
    PARSER.add_argument("-s",
                        "--src",
                        type=str,
                        required=True,
                        dest="src",
                        help="Source directory of files")
    PARSER.add_argument("-d",
                        "--dst",
                        type=str,
                        required=True,
                        dest="dst",
                        help="Destination directory")
    PARSER.add_argument("-o",
                        "--overwrite",
                        type=int,
                        choices=[0, 1],
                        default=1,
                        dest="overwrite",
                        help="Overwrite files")
    FLAGS = PARSER.parse_args()
    run(**vars(FLAGS))
