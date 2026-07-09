import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import resolve_pdf_path


class ResolvePdfPathTests(unittest.TestCase):
    def test_uses_environment_path_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "selected.pdf"
            pdf_path.touch()

            with patch.dict(os.environ, {"RAG_PDF_PATH": str(pdf_path)}, clear=False):
                self.assertEqual(resolve_pdf_path(Path(tmpdir)), pdf_path)

    def test_finds_first_pdf_in_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            pdf_path = workspace / "docs" / "guide.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.touch()

            self.assertEqual(resolve_pdf_path(workspace), pdf_path)


if __name__ == "__main__":
    unittest.main()
