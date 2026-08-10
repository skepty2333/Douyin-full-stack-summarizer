"""Offline regression tests for readable vector formula rendering."""
from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path
import unittest

from app.services import pdf_generator


FORMULA = r"C_{wire} = f_{out} + f_{intermediary} + f_{in} + s_{fx}"


class FormulaRenderingTests(unittest.TestCase):
    def test_block_formula_is_vector_and_uses_readable_style(self) -> None:
        rendered = pdf_generator.process_latex_in_markdown(f"成本为：\n\n$${FORMULA}$$")

        match = re.search(r"data:image/svg\+xml;base64,([^\"]+)", rendered)
        self.assertIsNotNone(match)
        assert match is not None
        svg = base64.b64decode(match.group(1))
        self.assertIn(b"<svg", svg)
        dimensions = re.search(rb'width="([0-9.]+)pt" height="([0-9.]+)pt"', svg)
        self.assertIsNotNone(dimensions)
        assert dimensions is not None
        self.assertGreater(float(dimensions.group(1)), 200.0)
        self.assertGreater(float(dimensions.group(2)), 20.0)
        self.assertIn('class="formula-block"', rendered)
        self.assertIn('class="block-formula"', rendered)
        self.assertIn("min-height: 1.65em", pdf_generator.GITHUB_PDF_CSS)

    def test_inline_formula_has_vector_baseline_sizing(self) -> None:
        rendered = pdf_generator.process_latex_in_markdown(r"费用包含 $f_{out}$。")
        self.assertIn("data:image/svg+xml;base64,", rendered)
        self.assertIn('class="inline-formula"', rendered)
        self.assertIn("height: 1.3em", pdf_generator.GITHUB_PDF_CSS)

    def test_formula_pdf_generation_succeeds_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            output = Path(temp_name) / "formula.pdf"
            success = pdf_generator.generate_pdf(
                f"# 公式测试\n\n成本为：\n\n$${FORMULA}$$\n\n其中 $f_{{out}}$ 为汇出费。",
                str(output),
            )
            self.assertTrue(success)
            self.assertTrue(output.read_bytes().startswith(b"%PDF-"))
            self.assertGreater(output.stat().st_size, 2000)

    def test_pdf_fetcher_still_rejects_external_assets(self) -> None:
        with self.assertRaises(ValueError):
            pdf_generator._safe_pdf_url_fetcher("file:///etc/passwd")

    def test_video_frames_have_print_height_and_atomic_pagination_limits(self) -> None:
        css = pdf_generator.GITHUB_PDF_CSS
        self.assertIn("max-height: 155mm", css)
        self.assertIn("break-inside: avoid", css)
        self.assertIn("page-break-inside: avoid", css)


if __name__ == "__main__":
    unittest.main()
