import os
import argparse
import shutil
import glob
from processors import pdf, tiff, xlsx

accepted_extensions = ["pdf", "doc", "docx",
                       "xls", "xlsx", "xlsm", "tiff", "tif"]

def get_files(src, formats):
    files = []
    for extension in formats:
        ext_files = glob.glob(os.path.join(
            src, "**/*." + extension), recursive=True)
        files += ext_files
    return files


def process_files(files, args):
    for num, f in enumerate(files, 1):
        head, filename = os.path.split(f)
        name, ext = os.path.splitext(filename)
        clean_name = name.replace(" ", "_")
        clean_filename = filename.replace(" ", "_")
        output_dir = os.path.join(args["dst"], clean_name)
        os.makedirs(output_dir, exist_ok=True)
        source_file = os.path.join(output_dir, clean_filename)
        shutil.copy2(f, source_file)
        print("**** Processing %s (%d/%d)" % (f, num, len(files)))
        if ext == ".pdf":
            job = pdf.Processor(source_file, args)
            job.run()
        elif ext.startswith(".tif"):
            job = tiff.Processor(source_file, args)
            job.run()
        elif ext == ".doc":
            pass
        elif ext == ".docx":
            pass
        elif ext.startswith(".xls"):
            job = xlsx.Processor(f, args)
            job.run()
        else:
            raise Exception("Unknown extension.")


def run(args):
    args["src"] = os.path.abspath(args["src"])
    assert os.path.exists(args["src"]), ("Source directory (%s) does not exist." % (args["src"]))
    assert os.path.isdir(args["src"]), ("Source (%s) is not a directory." % (args["src"]))
    args["formats"] = args["formats"] if (args["formats"] and len(args["formats"]) > 0) else accepted_extensions
    args["tessdata"] = os.path.abspath(args["tessdata"]) if args["tessdata"] else None
    if args["tessdata"] is not None:
        assert os.path.exists(args["tessdata"]), ("Tessdata directory (%s) does not exist." % (args["tessdata"]))
    args["overwrite"] = bool(args["overwrite"])
    args["cleanup"] = bool(args["cleanup"])
    args["store_results"] = bool(args["store_results"])
    files = get_files(args["src"], args["formats"])
    assert len(files) > 0, ("Found 0 files in %s" % (args["src"]))
    args["dst"] = os.path.abspath(args["dst"])
    os.makedirs(args["dst"], exist_ok=True)
    process_files(files, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Command line arguments for Document Conversion to JSON")
    parser.add_argument("-s",
                       "--src",
                       type=str,
                       required=True,
                       dest="src",
                       help="Source directory of files/Path to a single file")
    parser.add_argument("-d",
                       "--dst",
                       type=str,
                       required=True,
                       dest="dst",
                       help="Destination directory")
    parser.add_argument("-f",
                       "--formats",
                       nargs="*",
                       type=str,
                       choices=["pdf", "tiff", "tif", "xls", "xlsx", "xlsm", "doc", "docx"],
                       dest="formats",
                       help="File formats to process. Leave empty for all formats.")
    parser.add_argument("-t",
                       "--tessdata",
                       type=str,
                       dest="tessdata",
                       help="Path to Tessdata model (v4) for tesserocr. Applies to PDF/TIFF/Images")
    parser.add_argument("-oem",
                       type=str,
                       dest="oem",
                       choices=["v4", "v3"],
                       default="v4",
                       help="OEM Mode for Tesseract")
    parser.add_argument("-o",
                       "--overwrite",
                       type=int,
                       choices=[0,1],
                       default=0,
                       dest="overwrite",
                       help="Overwrite files")
    parser.add_argument("-c",
                       "--cleanup",
                       type=int,
                       choices=[0,1],
                       default=1,
                       dest="cleanup",
                       help="Clean up temporary files/directories")
    parser.add_argument("-st",
                       "--store_results",
                       type=int,
                       choices=[0,1],
                       default=1,
                       dest="store_results",
                       help="Store intermediate results (useful for debugging)")
    args = parser.parse_args()
    if args.formats and ("pdf" in args.formats or "tiff" in args.formats) and not args.tessdata:
        parser.error('Missing argument -t or --tessdata.')
    run(vars(args))
