import unittest

import pandas as pd

from documents import (
    add_email_fields,
    build_header,
    combine_documents,
    deduplicate_emails,
    emails_to_documents,
    normalize_body,
    normalize_email_fields,
)


class DocumentPreparationTests(unittest.TestCase):
    def test_email_cleanup_and_conversion_preserve_provenance(self):
        emails = pd.DataFrame(
            {
                "doc_id": ["a", "b"],
                "subject": [" Subject\t", "Subject"],
                "preview": [None, None],
                "from_name": ["Alice", "Alice"],
                "from_email": ["alice@example.com", "alice@example.com"],
                "to": ["Bob <bob@example.com>", "Bob <bob@example.com>"],
                "date": ["2020-01-01", "2020-01-01"],
                "body": ["One\r\n\r\n\r\nTwo", "One\n\nTwo"],
            }
        )

        prepared = add_email_fields(normalize_email_fields(emails))
        grouped = deduplicate_emails(prepared)
        documents = emails_to_documents(grouped)

        self.assertEqual(normalize_body(None), "")
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents.loc[0, "source_ids"], ["a", "b"])
        self.assertEqual(documents.loc[0, "source_type"], "email")
        self.assertEqual(documents.loc[0, "text"], "One\n\nTwo")

        pdf_page = pd.DataFrame(
            [{
                "source_ids": ["pdf:file:p3"],
                "source_type": "pdf",
                "title": "file",
                "text": "OCR text",
                "date": "",
                "sender": "",
                "recipients": "",
                "page": 3,
                "n_original_rows": 1,
            }]
        )
        combined = combine_documents(documents, pdf_page)
        self.assertEqual(combined["source_type"].tolist(), ["email", "pdf"])
        self.assertIn("Page: 3", build_header(combined.iloc[1]))


if __name__ == "__main__":
    unittest.main()
