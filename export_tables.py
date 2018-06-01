import argparse
import json
import glob
import os
import os
from processors.utils.export import export_tables as export
accepted_extensions = ["json"]

def get_files(src):
    files = []
    for extension in accepted_extensions:
        ext_files = glob.glob(os.path.join(
            src, "**/*." + extension), recursive=True)
        files += ext_files
    return files

def process_files(files, args):
    dst = args["dst"]
    for f in files:
        document = json.load(open(f))
        name = f.split("/")[-1].split(".")[0]
        dst_dir = os.path.join(dst, name)
        os.makedirs(dst_dir, exist_ok=True)
        export(document, dst_dir, name, fmt=args["fmt"], debug=args["debug"])



def run(args):
    src = os.path.abspath(args["src"])
    dst = os.path.abspath(args["dst"])
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
    os.makedirs(dst, exist_ok=True)
    args["src"] = src
    args["dst"] = dst
    process_files(files, args)

if __name__ == '__main__':
    flags = argparse.ArgumentParser(
        "Command line arguments for Information Extraction")
    flags.add_argument("-src",
                       type=str,
                       required=True,
                       help="Source directory of files/Single JSON file")
    flags.add_argument("-dst",
                       type=str,
                       required=True,
                       help="Destination directory")
    flags.add_argument("-debug",
                       type=bool,
                       default=False,
                       help="Write messages to console")
    flags.add_argument("-fmt",
                       type=str,
                       required=True,
                       choices=["txt", "csv", "xlsx", "df"],
                       help="Format to write Tables. Choices - txt (Plain Text), csv, xlsx, df (Pandas dataframe pickled)")
    args = flags.parse_args()
    args = (vars(args))
    run(args)
