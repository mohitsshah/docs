import os
from xlrd import open_workbook
import pandas as pd

class Processor(object):
    def __init__(self, output_dir, args):
        self.output_dir = output_dir
        self.overwrite = args["overwrite"]
        self.cleanup = args["cleanup"]
        self.raw_dir = os.path.join(self.output_dir, "raw")
        self.filename = os.listdir(self.raw_dir)[0]
        self.name, self.ext = self.filename.split(".")

    def make_page(self, sheet):
        table = []
        for row in range(sheet.nrows):
            r = []
            for col in range(sheet.ncols):
                r.append(sheet.cell(row, col).value)
            table.append(r)
        print (table)
        
    def run(self):
        file_path = os.path.join(self.raw_dir, self.filename)
        workbook = open_workbook(file_path)
        for sheet in workbook.sheets():
            self.make_page(sheet)

if __name__ == "__main__":
    import argparse
    flags = argparse.ArgumentParser("Command line for standalone evaluation")
    flags.add_argument("-output_dir", type=str, required=True)
    flags.add_argument("-overwrite", type=bool, default=False)
    flags.add_argument("-cleanup", type=bool, default=True)
    args = flags.parse_args()
    args = vars(args)
    output_dir = args["output_dir"]
    P = Processor(output_dir, args)
    P.run()