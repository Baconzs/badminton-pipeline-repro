"""Dependency-free contract tests for the optional BST adapter."""

from __future__ import annotations

import unittest

import numpy as np

from web_app.bst_adapter import (
    assess_bst_hit_gates,
    build_bst_sequences,
    decode_bst_logits,
    resolve_bst_configuration,
    run_bst_stroke_inference,
)


def _pose():
    points = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index in range(17):
        points[index] = [float(index + 10), float(index + 20), 0.95]
    return {
        "keypoints": points,
        "box": [0.0, 0.0, 100.0, 200.0],
        "court_position": [3.05, 10.0],
    }


class BSTFeatureTests(unittest.TestCase):
    def test_batched_jnbone_shape_and_normalization(self):
        pose = _pose()
        pose_by_frame = {
            frame: {"far": pose, "near": pose}
            for frame in range(0, 21)
        }
        features, metadata = build_bst_sequences(
            [{"frame": 10, "rally_id": 2, "hit_index": 3, "hitter": "near"}],
            pose_by_frame,
            {frame: (320.0, 240.0) for frame in range(0, 21)},
            fps=10.0,
            width=640,
            height=480,
            seq_len=20,
            window_seconds=1.0,
        )
        self.assertEqual(features["human_pose"].shape, (1, 20, 2, 36, 2))
        self.assertEqual(features["positions"].shape, (1, 20, 2, 2))
        self.assertEqual(features["shuttles"].shape, (1, 20, 2))
        self.assertEqual(int(features["video_lengths"][0]), 20)
        self.assertEqual(metadata[0]["pose_coverage"], 1.0)
        self.assertAlmostEqual(float(features["shuttles"][0, 10, 0]), 0.5)
        # Center-aligned bbox normalization keeps the bbox center at (0, 0).
        self.assertAlmostEqual(
            float(features["human_pose"][0, 10, 0, 0, 0]),
            -40.0 / (100.0**2 + 200.0**2) ** 0.5,
            places=4,
        )

    def test_other_upstream_pose_styles_have_expected_channel_counts(self):
        pose = _pose()
        pose_by_frame = {0: {"far": pose, "near": pose}}
        events = [{"frame": 0}]
        ball = {0: (1.0, 2.0)}
        expected = {"J_only": 17, "JnB_interp": 36, "JnB_bone": 36, "Jn2B": 55}
        for style, channels in expected.items():
            features, _ = build_bst_sequences(
                events, pose_by_frame, ball, fps=25, width=100, height=100,
                seq_len=8, window_seconds=.35, pose_style=style,
            )
            self.assertEqual(features["human_pose"].shape[-2], channels)

    def test_top3_decoder_is_bounded_and_probabilities_sum(self):
        metadata = [{"frame": 12, "seconds": .48, "timecode": "00:00"}]
        decoded = decode_bst_logits(
            np.asarray([[2.0, 0.0, 1.0, -1.0]], dtype=np.float32),
            ["Top_長球", "Bottom_殺球", "Top_挑球", "未知球種"],
            metadata,
        )
        self.assertEqual(decoded[0]["model_label"], "高远球")
        self.assertEqual(decoded[0]["model_side"], "far")
        self.assertLessEqual(len(decoded[0]["top3"]), 3)
        self.assertAlmostEqual(sum(item["probability"] for item in decoded[0]["top3"]), 0.968, places=2)
        self.assertAlmostEqual(decoded[0]["top1_probability"], decoded[0]["top3"][0]["probability"])
        self.assertGreater(decoded[0]["top1_top2_margin"], 0.0)
        self.assertIn("coach_ready", decoded[0])

    def test_motion_coach_gate_is_stricter_than_type_report_gate(self):
        hit = {
            "confidence": 82.0,
            "top3": [
                {"probability": .82},
                {"probability": .11},
                {"probability": .07},
            ],
            "event_confirmed": True,
            "pose_coverage": .50,
            "ball_coverage": .30,
        }

        gate = assess_bst_hit_gates(hit)

        self.assertTrue(gate["type_ready"])
        self.assertFalse(gate["coach_ready"])
        self.assertEqual(gate["coach_gate_reason"], "low_confidence")
        self.assertAlmostEqual(gate["top1_top2_margin"], .71)

    def test_motion_coach_gate_requires_margin_and_strict_coverage(self):
        strong = {
            "confidence": 91.0,
            "top3": [
                {"probability": .91},
                {"probability": .05},
                {"probability": .04},
            ],
            "event_confirmed": True,
            "pose_coverage": .91,
            "ball_coverage": .70,
        }
        gate = assess_bst_hit_gates(strong)
        self.assertTrue(gate["coach_ready"])
        self.assertEqual(gate["coach_gate_reason"], "ready")
        self.assertEqual(gate["top1_probability"], .91)
        self.assertEqual(gate["top1_top2_margin"], .86)

        missing_margin = dict(strong, top3=[])
        gate = assess_bst_hit_gates(missing_margin)
        self.assertFalse(gate["coach_ready"])
        self.assertEqual(gate["coach_gate_reason"], "margin_unavailable")

        sparse_pose = dict(strong, pose_coverage=.84)
        gate = assess_bst_hit_gates(sparse_pose)
        self.assertTrue(gate["type_ready"])
        self.assertFalse(gate["coach_ready"])
        self.assertEqual(gate["coach_gate_reason"], "low_pose_coverage")

        sparse_ball = dict(strong, ball_coverage=.54)
        gate = assess_bst_hit_gates(sparse_ball)
        self.assertTrue(gate["type_ready"])
        self.assertFalse(gate["coach_ready"])
        self.assertEqual(gate["coach_gate_reason"], "low_ball_coverage")

    def test_missing_optional_configuration_is_safe(self):
        config = resolve_bst_configuration("/definitely/not/a/repository")
        self.assertFalse(config["configured"])
        self.assertEqual(config["class_count"], 25)

    def test_report_exposes_separate_type_and_coaching_policies(self):
        report = run_bst_stroke_inference(
            "/does/not/exist.mp4",
            "/does/not/exist.csv",
            fps=25,
            pose_weights="/does/not/exist.pt",
            repo_root="/does/not/exist",
        )

        policy = report["gate_policy"]
        self.assertEqual(policy["type_report"]["min_probability"], .60)
        self.assertEqual(policy["type_report"]["min_pose_coverage"], .08)
        self.assertEqual(policy["motion_coach"]["min_probability"], .85)
        self.assertEqual(policy["motion_coach"]["min_top1_top2_margin"], .25)
        self.assertEqual(policy["motion_coach"]["min_pose_coverage"], .85)
        self.assertEqual(policy["motion_coach"]["min_ball_coverage"], .55)


if __name__ == "__main__":
    unittest.main()
