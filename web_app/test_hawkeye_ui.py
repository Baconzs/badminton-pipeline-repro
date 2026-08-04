"""Static UI contract tests for Hawk-Eye rejection diagnostics."""

from __future__ import annotations

import unittest
from pathlib import Path


class HawkEyeUIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        javascript_path = Path(__file__).resolve().parent / "static" / "app.js"
        cls.javascript = javascript_path.read_text(encoding="utf-8")
        start = cls.javascript.index("  function hawkeyeEmptyText(report)")
        end = cls.javascript.index("  function hawkeyeCourt(candidate, court)")
        cls.empty_renderer = cls.javascript[start:end]

    def test_known_rejection_reason_has_concise_product_copy(self):
        self.assertIn('landing_out_of_bounds: "终点超出可信球场范围"', self.javascript)
        self.assertIn("diagnostics.reason_counts", self.empty_renderer)
        self.assertIn("未生成落点：", self.empty_renderer)

    def test_unknown_or_legacy_reason_falls_back_to_safe_notice(self):
        self.assertIn("if (!primary) return fallback", self.empty_renderer)
        self.assertIn("Object.prototype.hasOwnProperty.call", self.empty_renderer)

    def test_only_observed_landing_can_show_a_line_distance(self):
        self.assertIn('candidate.method === "visible_rest"', self.javascript)
        self.assertIn("未生成可量化距离", self.javascript)


if __name__ == "__main__":
    unittest.main()
