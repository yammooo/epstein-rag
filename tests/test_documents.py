import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from documents import DOCUMENT_COLUMNS, _ocr_result_text, load_or_ocr_pdfs


class OcrTest(unittest.TestCase):
    def test_extracts_nonempty_recognized_lines(self):
        result = SimpleNamespace(
            json={"res": {"rec_texts": [" First line ", "", None, "Second line"]}}
        )

        self.assertEqual(_ocr_result_text(result), "First line\nSecond line")

    def test_empty_pdf_directory_does_not_load_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            documents = load_or_ocr_pdfs(directory, directory / "ocr.parquet")

        self.assertTrue(documents.empty)
        self.assertEqual(documents.columns.tolist(), DOCUMENT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
