import os
import argparse
import shutil
import glob
from processors import pdf, tiff

accepted_extensions = ["pdf", "doc", "docx",
                       "xls", "xlsx", "xlsm", "tiff", "tif"]


def get_files(src):
    files = []
    for extension in accepted_extensions:
        ext_files = glob.glob(os.path.join(
            src, "**/*." + extension), recursive=True)
        files += ext_files
    return files


def process_files(files, dst, args):
    print("Total %d files" % len(files))
    for f in files:
        filename = f.split("/")[-1]
        name, ext = filename.split(".")
        output_dir = os.path.join(dst, name)
        raw_dir = os.path.join(output_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        raw_file = os.path.join(raw_dir, filename)
        shutil.copy(f, raw_file)
        print("Working on %s..." % filename)
        if ext == "pdf":
            job = pdf.Processor(output_dir, args)
            job.run()
        elif ext.startswith("tif"):
            job = tiff.Processor(output_dir, args)
            job.run()
        elif ext == "doc":
            pass
        elif ext == "docx":
            pass
        elif ext.startswith("xls"):
            pass
        elif ext == "msg":
            pass
        else:
            raise Exception("Unknown extension.")
        print("Complete.")


def run(args):
    src = os.path.abspath(args["src"])
    dst = os.path.abspath(args["dst"])    
    args["tessdata"] = os.path.abspath(args["tessdata"])
    if not os.path.isdir(args["tessdata"]):
        raise Exception("Invalid tessdata directory (%s)" % (args["tessdata"]))
    if not os.path.exists(src):
        raise Exception("Directory (%s) does not exist." % (src))
    if not os.path.isdir(src):
        raise Exception("%s is not a directory." % (src))
    files = get_files(src)
    if len(files) == 0:
        raise Exception("Found 0 files in %s" % (src))
    if not os.path.exists(dst):
        print("Creating output directory at %s" % (dst))
        os.makedirs(dst, exist_ok=True)

    process_files(files, dst, args)


if __name__ == '__main__':
    flags = argparse.ArgumentParser(
        "Command line arguments for Document Conversion to JSON")
    flags.add_argument("-src",
                       type=str,
                       required=True,
                       help="Source directory of files")
    flags.add_argument("-dst",
                       type=str,
                       required=True,
                       help="Destination directory")
    flags.add_argument("-overwrite",
                       type=bool,                       
                       default=False, 
                       help="Overwrite files")
    flags.add_argument("-tessdata",
                       type=str,
                       required=True,
                       help="Path to Tessdata model (v4) for tesserocr. Applies to PDF/TIFF/Images")
    flags.add_argument("-osd",
                       type=str,
                       choices=["legacy", "tesserocr"],
                       default="legacy",
                       help="Choose mode for Orientation detection. Legacy mode uses the tesseract command line, Tesserocr mode uses the python library. Applies to PDF/TIFF/Images")
    flags.add_argument("-cleanup",
                       type=bool,
                       default=True,
                       help="Clean up temporary files/directories")
    args = flags.parse_args()
    args = (vars(args))
    run(args)
