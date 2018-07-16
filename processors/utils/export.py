import pandas
import tabulate
import os

def export_tables(document, output_dir, name, fmt="txt", debug=False):
    if document["errorFlag"]:
        return
    table_number = 0
    for page in document.get("documentPages"):
        page_number = page.get("pageNumber")
        for segment in page.get("pageSegments"):
            if not segment.get("isTable"):
                continue
            table = segment.get("table")
            num_rows = table.get("numRows")
            num_cols = table.get("numCols")
            cells = table.get("cells")
            data = [[[] for j in range(num_cols)] for i in range(num_rows)]
            for cell in cells:
                row_index = cell.get("rowIndex")
                col_index = cell.get("colIndex")
                words = cell.get("words")
                text = []
                for word in words:
                    text.append(word.get("txt"))
                data[row_index][col_index] = " ".join(text)
            data_frame = pandas.DataFrame(data)
            file_name = name + "-%d-%d.%s" % (page_number, table_number + 1, fmt)
            file_path = os.path.join(output_dir, file_name)
            if fmt == "txt":
                table_string = [tabulate.tabulate(data_frame, tablefmt="psql")]
                open(file_path, "w").write("\n".join(table_string))
                if debug:
                    print ("\n".join(table_string))
            elif fmt == "csv":
                data_frame.to_csv(file_path)
            elif fmt == "xlsx":
                writer = pandas.ExcelWriter(file_path)
                data_frame.to_excel(writer)
                writer.save()
            table_number += 1

def parse_paragraph(paragraph):
    text = []
    for word in paragraph.get("words"):
        text.append(word.get("txt"))
    raw_text = " ".join(text)
    return raw_text

def parse_table(table):
    num_rows = table.get("numRows")
    num_cols = table.get("numCols")
    data = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
    for cell in table.get("cells"):
        row_index = cell.get("rowIndex")
        col_index = cell.get("colIndex")
        text = []
        for word in cell.get("words"):
            text.append(word.get("txt"))
        text = " ".join(text)
        data[row_index][col_index] = text
    lines = [" ".join(row) for row in data]
    raw_text = "\n".join(lines)
    return raw_text, data

def export_text(document, output_dir, name):
    for page in document.get("documentPages"):
        raw_text = []
        page_number = page.get("pageNumber")
        for segment in page.get("pageSegments"):
            if (segment.get("isParagraph")):
                text = parse_paragraph(segment.get("paragraph"))
                raw_text.append(text)
            elif (segment.get("isTable")):
                text, table = parse_table(segment.get("table"))
                raw_text.append(text)
        raw_text = "\n".join(raw_text)
        file_name = name + "-%d.txt" % (page_number)
        file_path = os.path.join(output_dir, file_name)
        open(file_path, "w").write(raw_text)