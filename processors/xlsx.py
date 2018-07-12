import os
import json
import numpy as np
from xlrd import open_workbook
import processors.utils.segmentation as seg_utils
from processors.utils.model import DocumentModel

class Processor(object):
    def __init__(self, source_file, output_dir, overwrite, cleanup):
        _, filename = os.path.split(source_file)
        name, _ = os.path.splitext(filename)

        self.source_file = source_file
        self.filename = filename
        self.name = name
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.cleanup = cleanup

    def save_document(self, document):
        """Serialize the final document to a JSON file

        Arguments:
            document {object} -- Document object
        """

        path = os.path.join(self.output_dir, self.name + ".json")
        with open(path, "w") as fi:
            json.dump(document, fi)

    def get_lr_segments(self, page_matrix, threshold=1):
        lr_cuts = seg_utils.cut_segment(page_matrix.T)
        processed = []
        merge_list = []
        for idx, l_r in enumerate(lr_cuts):
            merge = []
            if idx in processed:
                continue
            processed.append(idx)
            nxt = int(idx)
            while True:
                nxt += 1
                if nxt >= len(lr_cuts) or nxt in processed:
                    break
                gap = lr_cuts[nxt][0] - lr_cuts[nxt-1][1]
                if gap > threshold:
                    break
                merge.append(nxt)
                processed.append(nxt)
            if merge:
                merge.append(idx)
                merge.sort()
                processed.append(idx)
                merge_list.append(merge)

        segments = []
        indices = []
        for items in merge_list:
            indices.extend(items)
            start = lr_cuts[items[0]][0]
            stop = lr_cuts[items[-1]][1]
            segments.append([start, stop])
        indices = list(set(indices))
        indices.sort()

        for index, l_r in enumerate(lr_cuts):
            if index not in indices:
                segments.append(l_r)

        segments = sorted(segments, key=lambda x: (x[0]))
        return segments

    def get_tb_segments(self, page_matrix, segments, threshold=1):
        tables = []
        for segment in segments:
            col_matrix = page_matrix[:, segment[0]:segment[1]]
            tb_cuts = seg_utils.cut_segment(col_matrix)
            processed = []
            merge_list = []
            for idx, t_b in enumerate(tb_cuts):
                merge = []
                if idx in processed:
                    continue
                processed.append(idx)
                nxt = int(idx)
                while True:
                    nxt += 1
                    if nxt >= len(tb_cuts) or nxt in processed:
                        break
                    gap = tb_cuts[nxt][0] - tb_cuts[nxt-1][1]
                    if gap > threshold:
                        break
                    merge.append(nxt)
                    processed.append(nxt)
                if merge:
                    merge.append(idx)
                    merge.sort()
                    processed.append(idx)
                    merge_list.append(merge)

            new_segments = []
            indices = []
            for items in merge_list:
                indices.extend(items)
                m1 = items[0]
                s1 = tb_cuts[m1][0]
                m2 = items[-1]
                s2 = tb_cuts[m2][1]
                new_segments.append([s1, s2])
            indices = list(set(indices))
            indices.sort()

            for index, t_b in enumerate(tb_cuts):
                if index not in indices:
                    new_segments.append(t_b)
            for row_segment in new_segments:
                tables.append([segment[0], row_segment[0], segment[1], row_segment[1], None])
        return tables

    def get_table_data(self, tables, sheet):
        for table in tables:
            table_data = []
            for row_index in range(table[1], table[3]):
                row = []
                for col_index in range(table[0], table[2]):
                    value = sheet.cell(row_index, col_index).value
                    row.append(value)
                table_data.append(row)
            table.append(table_data)
        return tables

    def make_page(self, sheet):
        sheet_name = sheet.name
        page_matrix = []
        for row in range(sheet.nrows):
            line_matrix = []
            for col in range(sheet.ncols):
                value = str(sheet.cell(row, col).value)
                if value:
                    line_matrix.append(1)
                else:
                    line_matrix.append(0)
            page_matrix.append(line_matrix)
        page_matrix = np.array(page_matrix).astype("uint8")
        segments = self.get_lr_segments(page_matrix)
        tables = self.get_tb_segments(page_matrix, segments)
        tables = self.get_table_data(tables, sheet)
        page = {"sheet_name": sheet_name, "tables": tables}
        return page

    def run(self):
        data = {"num_sheets": 0, "sheets": []}
        workbook = open_workbook(self.source_file)
        for sheet in workbook.sheets():
            page = self.make_page(sheet)
            data["sheets"].append(page)
            data["num_sheets"] += 1
        model = DocumentModel()
        document = model.create(data, self.source_file, source_format="xlsx")
        self.save_document(document)
        return True

