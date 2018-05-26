import pandas
import tabulate
import os

def export_tables(document, output_dir, name, fmt="txt"):
    pages = document["pages"]
    tables = [(page_num, segment["content"], segment["bbox"]) for page_num, page in enumerate(pages) for segment in page["segments"] if segment["label"] == "TABLE"]
    for table_num, (page_num, table, bbox) in enumerate(tables):
        table_string = []
        table_string.append("Page: %d" % (page_num + 1))
        num_rows = len(table)
        num_cols = len(table[0])
        bbox = [str(x) for x in bbox]
        bbox = ", ".join(bbox)
        table_string.append("Bounding Box: (%s)" % (bbox))
        table_string.append("No. of Rows: %d" % (num_rows))
        table_string.append("No. of Columns: %d" % (num_cols))
        table_data = []
        for row in table:
            row_data = []
            for col in row:
                col = sorted(col, key=lambda x: (x[1], x[0]))
                words = [c[-1] for c in col]
                row_data.append(" ".join(words))
            table_data.append(row_data)
        df = pandas.DataFrame(table_data)
        file_name = name + "-%d-%d.%s" % (page_num + 1, table_num + 1, fmt)
        file_path = os.path.join(output_dir, file_name)        
        if fmt == "txt":
            table_string.append(tabulate.tabulate(df, tablefmt="psql"))            
            open(file_path, "w").write("\n".join(table_string))            
            print ("\n".join(table_string))
        elif fmt == "csv":
            df.to_csv(file_path)
        elif fmt == "xlsx":
            writer = pandas.ExcelWriter(file_path)
            df.to_excel(writer)
            writer.save()
