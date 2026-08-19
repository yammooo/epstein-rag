import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation import generate_ground_truth


class GroundTruthSamplingTest(unittest.TestCase):
    def test_generates_equal_pdf_and_email_cases(self):
        documents = pd.DataFrame(
            [
                {
                    "source_ids": [f"{source_type}-{index}"],
                    "source_type": source_type,
                    "title": "Test document",
                    "text": "x" * 1001,
                    "date": "",
                    "sender": "",
                    "recipients": "",
                    "page": index if source_type == "pdf" else pd.NA,
                }
                for source_type in ("pdf", "email")
                for index in range(2)
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "test_set.json"
            test_set = generate_ground_truth(
                documents,
                lambda _: json.dumps({"question": "Question?", "answer": "Answer."}),
                cache_path,
                n=4,
                request_delay=0,
            )

        self.assertEqual(len(test_set), 4)
        self.assertEqual(sum(item["source_type"] == "pdf" for item in test_set), 2)
        self.assertEqual(sum(item["source_type"] == "email" for item in test_set), 2)


if __name__ == "__main__":
    unittest.main()
