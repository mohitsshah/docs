"""[summary]

Raises:
    Exception -- [description]
    Exception -- [description]
    Exception -- [description]
    Exception -- [description]

Returns:
    [type] -- [description]
"""

import argparse
import glob
import os
import extractors.extractor as extractor

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


def process_files(files, dst, defs, data_dir, model_dir, model_name, overwrite):
    """[summary]

    Arguments:
        files {[type]} -- [description]
        dst {[type]} -- [description]
        defs {[type]} -- [description]
        data_dir {[type]} -- [description]
        model_dir {[type]} -- [description]
        model_name {[type]} -- [description]
        overwrite {[type]} -- [description]
    """

    model = extractor.Extractor(data_dir, model_dir, model_name)
    model.set_defs(defs)
    for file in files:
        _, file_name = os.path.split(file)
        name, _ = os.path.splitext(file_name)
        output_name = name + ".xlsx"
        output_path = os.path.join(dst, output_name)
        if os.path.exists(output_path) and not overwrite:
            continue
        model.set_json(file)
        # data_frame = model.extract()


def run(src, dst, defs, data_dir, model_dir, model_name, overwrite):
    """[summary]

    Arguments:
        src {[type]} -- [description]
        dst {[type]} -- [description]
        defs {[type]} -- [description]
        data_dir {[type]} -- [description]
        model_dir {[type]} -- [description]
        model_name {[type]} -- [description]
        overwrite {[type]} -- [description]

    Raises:
        Exception -- [description]
        Exception -- [description]
        Exception -- [description]
        Exception -- [description]
    """

    src = os.path.abspath(src)
    if not os.path.exists(src):
        raise Exception("Directory/File (%s) does not exist." % (src))
    files = get_files(src)
    if not files:
        raise Exception("Found 0 files in %s" % (src))

    dst = os.path.abspath(dst)
    os.makedirs(dst, exist_ok=True)

    defs = os.path.abspath(defs)
    if not os.path.exists(defs):
        raise Exception("Definitions file (%s) does not exist." % (defs))

    process_files(files, dst, defs, data_dir, model_dir, model_name, overwrite)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        "Command line arguments for Information Extraction")
    PARSER.add_argument("-s",
                        "--src",
                        type=str,
                        required=True,
                        dest="src",
                        help="Source directory of files/Single file (JSON)")
    PARSER.add_argument("-d",
                        "--dst",
                        type=str,
                        required=True,
                        dest="dst",
                        help="Destination directory")
    PARSER.add_argument("-df",
                        "--defs",
                        type=str,
                        required=True,
                        dest="defs",
                        help="Path to Definitions (JSON) file")
    PARSER.add_argument("-dd",
                        "--data_dir",
                        type=str,
                        required=True,
                        dest="data_dir",
                        help="Path to QANet data directory")
    PARSER.add_argument("-md",
                        "--model_dir",
                        type=str,
                        required=True,
                        dest="model_dir",
                        help="Path to QANet model directory")
    PARSER.add_argument("-mn",
                        "--model_name",
                        type=str,
                        default="FRC",
                        required=True,
                        dest="model_name",
                        help="QANet model name (Default: FRC)")
    PARSER.add_argument("-o",
                        "---overwrite",
                        type=int,
                        choices=[0, 1],
                        default=1,
                        dest="overwrite",
                        help="Overwrite files")
    FLAGS = PARSER.parse_args()
    run(**vars(FLAGS))
