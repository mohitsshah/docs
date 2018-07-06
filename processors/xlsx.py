import os
from xlrd import open_workbook
import pandas as pd
import processors.utils.segmentation as seg_utils
import numpy as np

class Processor(object):
    def __init__(self, output_dir, args):
        self.output_dir = output_dir
        self.overwrite = args["overwrite"]
        self.cleanup = args["cleanup"]
        self.raw_dir = os.path.join(self.output_dir, "raw")
        self.filename = os.listdir(self.raw_dir)[0]
        self.name, self.ext = self.filename.split(".")

    def cut_segment(self, segment):
        gaps = np.any(segment, axis=1).astype('int')
        tmp = np.array(gaps[1:]) - np.array(gaps[0:-1])
        breaks = np.where(tmp==1)[0]
        breaks = [x+1 for x in breaks]
        return breaks
        
    def make_page(self, sheet):        
        table = []
        num_rows = sheet.nrows
        num_cols = sheet.ncols
        table_image = np.zeros((num_rows, num_cols))
        print (num_rows, num_cols)
        for row in range(sheet.nrows):
            r = []
            for col in range(sheet.ncols):
                v = str(sheet.cell(row, col).value)
                if len(v) > 0:
                    table_image[row, col] = 255
                r.append(v)
            table.append(r)
        lr_cuts = self.cut_segment(table_image.T)
        print (lr_cuts)

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