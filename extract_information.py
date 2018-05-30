import argparse
import json
import glob
import os

accepted_extensions = ["json"]

def get_files(src):
    files = []
    for extension in accepted_extensions:
        ext_files = glob.glob(os.path.join(
            src, "**/*." + extension), recursive=True)
        files += ext_files
    return files

def process_files(files, defs_path):
    pass

def run(args):
    src = os.path.abspath(args["src"])
    defs = os.path.abspath(args["defs"])
    if not os.path.exists(src):
        raise Exception("Directory/File (%s) does not exist." % (src))
    is_directory = None
    if os.path.isdir(src):
        is_directory = True
    if src.endswith(".json"):
        is_directory = False
    if is_directory is None:
        raise Exception("Check Source Directory/File.")
    if is_directory is True:
        files = get_files(src)
    else:
        files = [src]
    if len(files) == 0:
        raise Exception("Found 0 files in %s" % (src))

    if not os.path.exists(defs):
        raise Exception("Definitions File (%s) does not exist." % (defs))

    process_files(files, defs)

if __name__ == '__main__':
    flags = argparse.ArgumentParser(
        "Command line arguments for Information Extraction")
    flags.add_argument("-src",
                       type=str,
                       required=True,
                       help="Source directory of files/Single JSON file")
    flags.add_argument("-defs",
                       type=str,
                       required=True,
                       help="Path to Definitions (JSON) file")
    flags.add_argument("-overwrite",
                       type=bool,
                       default=False,
                       help="Overwrite files")
    args = flags.parse_args()
    args = (vars(args))
    run(args)