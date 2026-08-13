import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from documents import DOCUMENT_COLUMNS, _ocr_result_text, load_or_ocr_pdfs


class OcrTest(unittest.TestCase):
    def test_orders_columns_and_keeps_text_from_image_labeled_region(self):
        result = SimpleNamespace(
            json={
                "res": {
                    "overall_ocr_res": {
                        "rec_texts": ["Left top", "Right top", "Left bottom", "Right bottom"],
                        "rec_boxes": [
                            [0, 0, 40, 10],
                            [60, 0, 100, 10],
                            [0, 20, 40, 30],
                            [60, 20, 100, 30],
                        ],
                    },
                    "parsing_res_list": [
                        {
                            "block_bbox": [0, 0, 40, 40],
                            "block_label": "image",
                            "block_order": 1,
                        },
                        {
                            "block_bbox": [60, 0, 100, 40],
                            "block_label": "text",
                            "block_order": 2,
                        },
                    ],
                }
            }
        )

        self.assertEqual(
            _ocr_result_text(result),
            "Left top Left bottom\n\nRight top Right bottom",
        )

    def test_empty_pdf_directory_does_not_load_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            documents = load_or_ocr_pdfs(directory, directory / "ocr.parquet")

        self.assertTrue(documents.empty)
        self.assertEqual(documents.columns.tolist(), DOCUMENT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
