"""[summary]

Raises:
    Exception -- [description]
    Exception -- [description]

Returns:
    [type] -- [description]
"""

import argparse
import glob
import json
import os
from processors.utils.export import export_text as export

ALLOWED_EXTENSIONS = ["json"]


def get_files(src):
    """[summary]

    Arguments:
        src {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    files = []
    for extension in ALLOWED_EXTENSIONS:
        ext_files = glob.glob(os.path.join(
            src, "**/*." + extension), recursive=True)
        files += ext_files
    return files


def process_files(files, dst):
    """[summary]

    Arguments:
        files {[type]} -- [description]
        dst {[type]} -- [description]
    """

    for file in files:
        _, file_name = os.path.split(file)
        name, _ = os.path.splitext(file_name)
        document = json.load(open(file))
        dst_dir = os.path.join(dst, name)
        os.makedirs(dst_dir, exist_ok=True)
        export(document, dst_dir, name)


def run(src, dst):
    """[summary]

    Arguments:
        src {[type]} -- [description]
        dst {[type]} -- [description]

    Raises:
        Exception -- [description]
        Exception -- [description]
    """

    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if not os.path.exists(src):
        raise Exception("Directory/File (%s) does not exist." % (src))
    files = get_files(src)
    if not files:
        raise Exception("Found 0 files in %s" % (src))
    os.makedirs(dst, exist_ok=True)
    process_files(files, dst)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        "Command line arguments for Table Export from JSON Document(s)")
    PARSER.add_argument("-s",
                        "--src",
                        type=str,
                        required=True,
                        dest="src",
                        help="Source directory of JSON files")
    PARSER.add_argument("-d",
                        "--dst",
                        type=str,
                        required=True,
                        dest="dst",
                        help="Destination directory")
    FLAGS = PARSER.parse_args()
    run(**vars(FLAGS))
