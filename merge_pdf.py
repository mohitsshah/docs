import os
from PyPDF2 import PdfFileWriter, PdfFileReader

filenames = [
    "2016_Complete_63190464135216030900_CapitalOne_Statement_072015_9938.pdf",
    "2016_Complete_63199458125216033000_7800_Aug_Bank.pdf",
    "2016_Complete_63113779800216121900_CHASE_063016.pdf"
]

pdf_writer = PdfFileWriter()

for filename in filenames:
    path = os.path.join("/Users/mohitshah/Others/statements", filename)
    pdf_reader = PdfFileReader(path)
    for page in range(pdf_reader.getNumPages()):
        pdf_writer.addPage(pdf_reader.getPage(page))

with open("merged1.pdf", 'wb') as fh:
    pdf_writer.write(fh)
