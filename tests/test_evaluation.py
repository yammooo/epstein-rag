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


class JsonParserTest(unittest.TestCase):
    def test_parses_plain_json(self):
        from evaluation import _parse_json_response

        res = _parse_json_response('{"score": 8, "reasoning": "accurate"}')
        self.assertEqual(res, {"score": 8, "reasoning": "accurate"})

    def test_parses_json_with_code_fence(self):
        from evaluation import _parse_json_response

        res = _parse_json_response('```json\n{"score": 9, "reasoning": "good"}\n```')
        self.assertEqual(res, {"score": 9, "reasoning": "good"})

    def test_parses_json_with_unlabeled_fence_and_preamble(self):
        from evaluation import _parse_json_response

        text = 'Here is the result:\n```\n{"score": 10, "reasoning": "perfect"}\n```\nDone.'
        res = _parse_json_response(text)
        self.assertEqual(res, {"score": 10, "reasoning": "perfect"})


class TestSetValidationTest(unittest.TestCase):
    def test_is_balanced_test_set(self):
        from evaluation import _is_balanced_test_set

        valid_set = [
            {"source_ids": ["pdf:1"], "source_type": "pdf"},
            {"source_ids": ["email:1"], "source_type": "email"},
        ]
        self.assertTrue(_is_balanced_test_set(valid_set, 2))
        self.assertFalse(_is_balanced_test_set(valid_set, 4))
        self.assertFalse(_is_balanced_test_set([], 2))


if __name__ == "__main__":
    unittest.main()
