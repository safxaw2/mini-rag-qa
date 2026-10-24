import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.pdf_parse import PDFParser


def test_pdf_pares():
    parser=PDFParser()
    result=parser.parse("./data/WBGAR25_Chinese.pdf")

    print(result)

    with open('./tests/test_pdf.md', 'w', encoding='utf-8') as f:
        f.write(result.raw_text)

if __name__=="__main__":
    test_pdf_pares()