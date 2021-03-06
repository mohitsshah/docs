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

from processors.utils.export import export_tables as export

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


def process_files(files, dst, fmt):
    """[summary]

    Arguments:
        files {[type]} -- [description]
        dst {[type]} -- [description]
        fmt {[type]} -- [description]
    """

    for file in files:
        document = json.load(open(file))
        _, file_name = os.path.split(file)
        name, _ = os.path.splitext(file_name)
        dst_dir = os.path.join(dst, name)
        os.makedirs(dst_dir, exist_ok=True)
        export(document, dst_dir, name, fmt)


def run(src, dst, fmt):
    """[summary]

    Arguments:
        src {[type]} -- [description]
        dst {[type]} -- [description]
        fmt {[type]} -- [description]

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
    process_files(files, dst, fmt)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        "Command line arguments for Table Export from JSON Document(s)")
    PARSER.add_argument("-s",
                        "--src",
                        type=str,
                        required=True,
                        dest="src",
                        help="Source directory of files/Single JSON file")
    PARSER.add_argument("-d",
                        "--dst",
                        type=str,
                        required=True,
                        dest="dst",
                        help="Destination directory")
    PARSER.add_argument("-f",
                        "--fmt",
                        type=str,
                        required=True,
                        choices=["txt", "csv", "xlsx", "df"],
                        dest="fmt",
                        help="Format to write Tables.")
    FLAGS = PARSER.parse_args()
    run(**vars(FLAGS))
