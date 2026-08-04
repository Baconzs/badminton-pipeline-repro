"""Static UI contract tests for Motion Coach's selective presentation.

The browser bundle is dependency-free and this repository does not ship a
JavaScript test runtime.  These assertions keep the product boundary explicit:
per-stroke candidates stay internal and only reliable aggregate advice is
rendered.
"""

from __future__ import annotations

import unittest
from pathlib import Path


class MotionCoachUIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        static_root = Path(__file__).resolve().parent / "static"
        cls.javascript = (static_root / "app.js").read_text(encoding="utf-8")
        cls.html = (static_root / "index.html").read_text(encoding="utf-8")
        cls.styles = (static_root / "app.css").read_text(encoding="utf-8")
        start = cls.javascript.index("  function renderCoaching(raw)")
        end = cls.javascript.index("  function renderStrokeTypes(raw)")
        cls.coaching_renderer = cls.javascript[start:end]

    def test_renderer_requires_reliable_aggregate_recommendations(self):
        self.assertIn('item.status !== "reliable"', self.javascript)
        self.assertIn("repeatCount < 3", self.javascript)
        self.assertIn("!RELIABLE_COACH_FAMILIES.has(family)", self.javascript)
        self.assertIn("!signalFamilies || !signalFamilies.has(family)", self.javascript)
        self.assertIn("UNCERTAIN_COACH_COPY.test(`${problem} ${action}`)", self.javascript)
        self.assertIn("reliableCoachRecommendations(player)", self.coaching_renderer)
        self.assertNotIn("stroke_reviews", self.coaching_renderer)
        self.assertNotIn("evaluation.suggestions", self.coaching_renderer)

    def test_renderer_uses_concise_problem_and_training_copy(self):
        self.assertIn('<i>问题</i>', self.javascript)
        self.assertIn('<i>训练</i>', self.javascript)
        self.assertIn("可靠建议 0 条", self.coaching_renderer)
        self.assertIn("UNCERTAIN_COACH_COPY", self.javascript)

    def test_initial_motion_coach_copy_matches_selective_renderer(self):
        self.assertIn("仅展示通过证据门控的可靠建议", self.html)
        self.assertNotIn("逐拍动作建议", self.html)
        self.assertIn(".coach-recommendation-card", self.styles)

    def test_aggregate_family_labels_and_signal_compatibility_are_explicit(self):
        for family, label in (
            ("preparation", "准备衔接"),
            ("overhead", "上手球"),
            ("frontcourt", "前场球"),
            ("backcourt", "后场球"),
            ("recovery", "击球恢复"),
        ):
            with self.subTest(family=family):
                self.assertIn(f'{family}: "{label}"', self.javascript)
                self.assertIn(f'"{family}"', self.javascript)

        for signal in (
            "preparation_support",
            "overhead_preparation",
            "recovery_stability",
        ):
            with self.subTest(signal=signal):
                self.assertIn(f'["{signal}", new Set', self.javascript)

        # Aggregate categories accept the corresponding legacy evidence
        # signals while exact legacy stroke families stay in the same map.
        self.assertIn('["overhead_extension", new Set(["clear", "drive_clear", "drop", "smash", "overhead"])]', self.javascript)
        self.assertIn('["front_support", new Set(["push", "lift", "cross_net", "net", "net_kill", "frontcourt"])]', self.javascript)
        self.assertIn('["back_support", new Set(["clear", "drive_clear", "drop", "smash", "defensive_drive", "backcourt"])]', self.javascript)

    def test_narrow_layout_stretches_panels_and_reflows_kpis(self):
        self.assertIn("align-items: stretch", self.styles)
        self.assertIn(".kpi-grid { grid-template-columns: repeat(3,minmax(0,1fr)); }", self.styles)


if __name__ == "__main__":
    unittest.main()
