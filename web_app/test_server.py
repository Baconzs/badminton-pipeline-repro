"""Small Flask contract tests for the local Web UI.

These tests intentionally exercise only the HTTP boundary.  They do not enqueue
an inference job (which would require model weights and a long-running worker),
but they do create the same application object used by ``run_web_ui.sh``.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from web_app.server import _public_job, create_app


class ServerRouteTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.data_root = Path(self._temporary.name) / "runtime"
        repository = Path(__file__).resolve().parents[1]
        self.app = create_app(
            repo_root=repository,
            data_root=self.data_root,
            python_bin=sys.executable,
        )
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self):
        # JobManager owns a daemon worker.  Stop the idle worker explicitly so
        # repeated unittest runs do not accumulate background threads.
        manager = self.app.extensions.get("job_manager")
        worker_queue = getattr(manager, "_queue", None)
        if worker_queue is not None:
            worker_queue.put(None)
        self._temporary.cleanup()

    def test_health_endpoint_returns_runtime_capabilities(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        self.assertTrue(payload.get("ok") in {True, False})
        self.assertIn("weights", payload)
        self.assertIn("bst", payload)
        self.assertIn("ffmpeg", payload)
        self.assertIn("queue_mode", payload)

    def test_homepage_is_served(self):
        response = self.client.get("/")
        try:
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/html", response.content_type)
            self.assertGreater(len(response.data), 0)
        finally:
            # ``send_file`` keeps a file handle on the response until it is
            # explicitly closed when used through Flask's test client.
            response.close()

    def test_server_path_outside_allowlist_is_rejected(self):
        outside_video = Path(self._temporary.name) / "outside.mp4"
        # The bytes do not need to be a valid video: allow-list validation must
        # reject the path before any decoder is called.
        outside_video.write_bytes(b"not a video")

        response = self.client.post(
            "/api/videos/import",
            json={"path": str(outside_video)},
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        self.assertIn("error", payload)
        self.assertIn("允许目录", str(payload["error"]))

    def test_conflicting_bullet_time_aliases_are_rejected(self):
        response = self.client.post(
            "/api/jobs",
            json={
                "video_id": "missing",
                "bullet_time_enabled": False,
                "bullet_time_fx": True,
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        self.assertIn("冲突", str(payload["error"]))

    def test_public_job_coaching_is_path_free_and_schema_limited(self):
        public = _public_job({
            "id": "aaaaaaaaaaaa",
            "video": {}, "outputs": {}, "rallies": [], "logs": [],
            "coaching": {
                "status": "ready", "notice": "safe", "players": [{
                    "id": "near", "label": "attacker", "score": 130,
                    "sample_count": 2, "path": "/private/report.json",
                    "recommendations": [{
                        "priority": "high", "title": "提示", "detail": "详情",
                        "metric": "指标", "evidence": [{
                            "frame": 12, "seconds": 0.48, "timecode": "00:00",
                            "path": "/private/frame.png",
                        }],
                    }],
                }],
            },
            "hawkeye": {
                "status": "ready",
                "notice": "safe",
                "summary": {"reviewed_rallies": 1},
                "candidates": [{
                    "rally_id": 1,
                    "frame": 30,
                    "seconds": 1.2,
                    "timecode": "00:01",
                    "hit_frame": 10,
                    "hitter": "near",
                    "landing": {"x_m": 3.0, "y_m": 2.0, "path": "/private/point.json"},
                    "decision": "in",
                    "method": "visible_rest",
                    "line_distance_cm": 100.0,
                    "confidence": 88,
                    "confidence_level": "high",
                    "evidence_frames": 5,
                    "path": "/private/hawkeye.json",
                }],
            },
        })
        coaching = public["coaching"]
        self.assertEqual(coaching["players"][0]["label"], "下方球员")
        self.assertEqual(coaching["players"][0]["score"], 100)
        self.assertNotIn("path", str(coaching))
        hawkeye = public["hawkeye"]
        self.assertEqual(hawkeye["candidates"][0]["decision"], "in")
        self.assertNotIn("path", str(hawkeye))

    def test_public_hawkeye_diagnostics_are_allowlisted_and_bounded(self):
        public = _public_job({
            "id": "eeeeeeeeeeee", "video": {}, "outputs": {}, "rallies": [], "logs": [],
            "coaching": {},
            "hawkeye": {
                "status": "insufficient_evidence",
                "notice": "safe",
                "summary": {"reviewed_rallies": 2},
                "candidates": [],
                "diagnostics": {
                    "reason_counts": [
                        {
                            "code": "landing_out_of_bounds",
                            "count": 2,
                            "detail": "/private/track.csv",
                        },
                        {"code": "unknown_reason", "count": 9},
                        {"code": "wrong_target_half", "count": -1},
                        {"code": "low_confidence", "count": "3"},
                        {"code": "landing_out_of_bounds", "count": 99},
                    ],
                    "ignored_unconfirmed_hits": 1,
                    "fallback_candidate_count": 9999999,
                    "path": "/private/report.json",
                },
            },
        })

        diagnostics = public["hawkeye"]["diagnostics"]
        self.assertEqual(diagnostics["reason_counts"], [
            {"code": "landing_out_of_bounds", "count": 2},
            {"code": "low_confidence", "count": 3},
        ])
        self.assertEqual(diagnostics["ignored_unconfirmed_hits"], 1)
        self.assertEqual(diagnostics["fallback_candidate_count"], 999999)
        self.assertNotIn("private", str(diagnostics))

    def test_legacy_endpoint_proxy_is_hidden_as_no_landing(self):
        public = _public_job({
            "id": "ffffffffffff", "video": {}, "outputs": {},
            "rallies": [], "logs": [], "coaching": {},
            "hawkeye": {
                "status": "ready",
                "notice": "旧报告",
                "summary": {"reviewed_rallies": 1},
                "diagnostics": {
                    "reason_counts": [{"code": "landing_out_of_bounds", "count": 1}],
                },
                "candidates": [{
                    "rally_id": 1, "frame": 907, "seconds": 36.28,
                    "timecode": "00:36", "hit_frame": 856,
                    "hitter": "unknown", "landing": {"x_m": 4.789, "y_m": -1.716},
                    "decision": "review", "method": "terminal_endpoint_proxy",
                    "line_distance_cm": 171.6, "confidence": 79,
                    "confidence_level": "medium", "evidence_frames": 3,
                }],
            },
        })
        hawkeye = public["hawkeye"]
        self.assertEqual(hawkeye["status"], "insufficient_evidence")
        self.assertEqual(hawkeye["candidates"], [])
        self.assertEqual(
            hawkeye["diagnostics"]["reason_counts"],
            [{"code": "no_landing_evidence", "count": 1}],
        )
        self.assertNotIn("171.6", str(hawkeye))

    def test_public_v2_coaching_only_exposes_reliable_repeated_advice(self):
        public = _public_job({
            "id": "dddddddddddd", "video": {}, "outputs": {}, "rallies": [], "logs": [],
            "coaching": {
                "version": 2, "status": "ready", "notice": "safe", "players": [{
                    "id": "near", "score": 81, "sample_count": 7,
                    "recommendations": [
                        {
                            "status": "candidate", "priority": "high", "family": "smash",
                            "signal": "overhead_extension", "problem": "候选问题",
                            "action": "候选训练", "repeat_count": 4,
                        },
                        {
                            "status": "suppressed", "priority": "medium", "family": "smash",
                            "signal": "back_support", "problem": "内部诊断",
                            "action": "不应展示", "repeat_count": 5,
                        },
                        {
                            "status": "candidate", "priority": "medium", "family": "drop",
                            "signal": "overhead_extension", "problem": "第三条内部候选",
                            "action": "仍不应展示", "repeat_count": 6,
                        },
                        {
                            "status": "reliable", "priority": "high", "family": "smash",
                            "signal": "overhead_extension", "problem": "动作可能存在问题",
                            "action": "待确认后再练习", "repeat_count": 5,
                        },
                        {
                            "status": "reliable", "priority": "good", "family": "smash",
                            "signal": "overhead_extension", "problem": "不支持的旧优先级",
                            "action": "不应进入 v2", "repeat_count": 5,
                        },
                        {
                            "status": "reliable", "priority": "high", "family": "smash",
                            "signal": "overhead_extension", "problem": "连续上手击球伸展不足。",
                            "action": "提前架拍，在身体前上方完成触球。", "repeat_count": 4,
                            "title": "上手伸展", "detail": "兼容字段", "metric": "4 拍",
                            "path": "/private/recommendation.json",
                            "evidence": [{
                                "frame": 120, "seconds": 4.8, "timecode": "00:04",
                                "window_start": 4.3, "window_end": 5.6,
                                "real_points": 8, "path": "/private/frame.png",
                            }],
                        },
                    ],
                }],
            },
            "hawkeye": {},
        })

        recommendations = public["coaching"]["players"][0]["recommendations"]
        self.assertEqual(len(recommendations), 1)
        recommendation = recommendations[0]
        self.assertEqual(recommendation["status"], "reliable")
        self.assertEqual(recommendation["family"], "smash")
        self.assertEqual(recommendation["signal"], "overhead_extension")
        self.assertEqual(recommendation["problem"], "连续上手击球伸展不足。")
        self.assertEqual(recommendation["action"], "提前架拍，在身体前上方完成触球。")
        self.assertEqual(recommendation["repeat_count"], 4)
        self.assertNotIn("path", str(public["coaching"]))

    def test_public_v2_coaching_accepts_aggregate_and_legacy_family_signal_pairs(self):
        compatible_pairs = (
            ("preparation", "preparation_support"),
            ("overhead", "overhead_preparation"),
            ("overhead", "overhead_extension"),
            ("frontcourt", "front_support"),
            ("frontcourt", "front_contact_space"),
            ("backcourt", "back_support"),
            ("recovery", "recovery_stability"),
            # Existing exact stroke families remain valid.
            ("smash", "overhead_extension"),
            ("lift", "front_support"),
            ("clear", "back_support"),
        )
        for family, signal in compatible_pairs:
            with self.subTest(family=family, signal=signal):
                public = _public_job({
                    "id": "ffffffffffff", "video": {}, "outputs": {}, "rallies": [], "logs": [],
                    "coaching": {
                        "version": 2, "status": "ready", "notice": "safe", "players": [{
                            "id": "near", "score": 80, "sample_count": 5,
                            "recommendations": [{
                                "status": "reliable", "priority": "high",
                                "family": family, "signal": signal,
                                "problem": "连续三拍动作结构不稳定。",
                                "action": "分组练习并保持动作节奏。",
                                "repeat_count": 3,
                            }],
                        }],
                    },
                    "hawkeye": {},
                })

                recommendations = public["coaching"]["players"][0]["recommendations"]
                self.assertEqual(len(recommendations), 1)
                self.assertEqual(recommendations[0]["family"], family)
                self.assertEqual(recommendations[0]["signal"], signal)

    def test_public_v2_coaching_rejects_mismatched_aggregate_family_signal_pairs(self):
        for family, signal in (
            ("recovery", "overhead_extension"),
            ("overhead", "recovery_stability"),
            ("frontcourt", "back_support"),
            ("backcourt", "front_contact_space"),
        ):
            with self.subTest(family=family, signal=signal):
                public = _public_job({
                    "id": "999999999999", "video": {}, "outputs": {}, "rallies": [], "logs": [],
                    "coaching": {
                        "version": 2, "status": "ready", "notice": "safe", "players": [{
                            "id": "near", "score": 80, "sample_count": 5,
                            "recommendations": [{
                                "status": "reliable", "priority": "high",
                                "family": family, "signal": signal,
                                "problem": "连续三拍动作结构不稳定。",
                                "action": "分组练习并保持动作节奏。",
                                "repeat_count": 3,
                            }],
                        }],
                    },
                    "hawkeye": {},
                })

                self.assertEqual(public["coaching"]["players"][0]["recommendations"], [])

    def test_public_v2_aggregate_coaching_still_requires_repeated_fixed_copy(self):
        invalid_fields = (
            {"repeat_count": 2},
            {"problem": ""},
            {"action": ""},
            {"problem": "动作可能不稳定。"},
            {"action": "待确认后再练习。"},
        )
        for overrides in invalid_fields:
            with self.subTest(overrides=overrides):
                recommendation = {
                    "status": "reliable", "priority": "high",
                    "family": "overhead", "signal": "overhead_preparation",
                    "problem": "连续三拍准备时机偏晚。",
                    "action": "提前架拍并保持身体前倾。",
                    "repeat_count": 3,
                    **overrides,
                }
                public = _public_job({
                    "id": "888888888888", "video": {}, "outputs": {}, "rallies": [], "logs": [],
                    "coaching": {
                        "version": 2, "status": "ready", "notice": "safe", "players": [{
                            "id": "near", "score": 80, "sample_count": 5,
                            "recommendations": [recommendation],
                        }],
                    },
                    "hawkeye": {},
                })

                self.assertEqual(public["coaching"]["players"][0]["recommendations"], [])

    def test_public_job_exposes_bounded_stroke_review_with_video_window(self):
        public = _public_job({
            "id": "bbbbbbbbbbbb", "video": {}, "outputs": {}, "rallies": [], "logs": [],
            "coaching": {
                "version": 2, "status": "ready", "notice": "safe", "players": [{
                    "id": "near", "score": 77, "sample_count": 4, "review_count": 1,
                    "gated_review_count": 3, "evidence_quality": 71,
                    "stroke_summary": [{"family": "lift", "label": "挑球候选", "count": 1, "average_confidence": 70}],
                    "stroke_reviews": [{
                        "rally_id": 1, "hit_index": 2,
                        "stroke": {
                            "family": "lift", "label": "正手候选挑球候选", "base_label": "挑球候选",
                            "area": "front", "confidence": 72, "confidence_level": "medium",
                            "hand_label": "正手候选", "hand_status": "candidate",
                            "reasons": ["路线"], "path": "/private/stroke.json",
                        },
                        "classification": {
                            "source": "bst", "status": "agreement", "family": "lift",
                            "label": "正手挑球", "rule_family": "lift", "rule_label": "挑球",
                            "rule_confidence": 70, "bst_family": "lift", "bst_label": "挑球",
                            "bst_raw_label": "Bottom_挑球", "bst_confidence": 82.4,
                            "bst_margin": 31.2, "type_conflict": False,
                            "advice_eligible": True, "reason": "BST 与球路类型一致",
                            "coaching_gate": {
                                "eligible": True, "reason_code": "ready",
                                "source": "bst_agreement", "bst_confidence": 91.3,
                                "bst_margin": 31.2, "pose_coverage": .92,
                                "ball_coverage": .71, "path": "/private/gate.json",
                            },
                            "path": "/private/classification.json",
                        },
                        "evaluation": {
                            "grade": "focus", "title": "前场动作观察", "detail": "详情",
                            "observations": ["观察"], "suggestions": ["建议"], "caveat": "限制",
                        },
                        "advice_eligible": True,
                        "observed": ["真实球点"],
                        "evidence": {
                            "frame": 100, "seconds": 4.0, "timecode": "00:04",
                            "pre_seconds": .55, "post_seconds": 1.2, "window_start": 3.45,
                            "window_end": 5.2, "real_points": 6, "path": "/private/clip.mp4",
                        },
                    }],
                    "recommendations": [],
                }],
            },
            "hawkeye": {},
        })
        coaching = public["coaching"]
        review = coaching["players"][0]["stroke_reviews"][0]
        self.assertEqual(coaching["version"], 2)
        self.assertEqual(coaching["players"][0]["gated_review_count"], 3)
        self.assertEqual(review["stroke"]["family"], "lift")
        self.assertEqual(review["classification"]["source"], "bst")
        self.assertEqual(review["classification"]["bst_confidence"], 82.4)
        self.assertTrue(review["classification"]["advice_eligible"])
        self.assertEqual(review["classification"]["coaching_gate"]["reason_code"], "ready")
        self.assertEqual(review["classification"]["coaching_gate"]["bst_margin"], 31.2)
        self.assertTrue(review["advice_eligible"])
        self.assertEqual(review["evaluation"]["suggestions"], [])
        self.assertEqual(review["evidence"]["window_start"], 3.45)
        self.assertNotIn("path", str(coaching))

    def test_public_job_exposes_safe_bst_top3_cards(self):
        public = _public_job({
            "id": "cccccccccccc", "video": {}, "outputs": {}, "rallies": [], "logs": [],
            "coaching": {}, "hawkeye": {},
            "stroke_types": {
                "version": 1, "status": "ready", "model": "BST_0",
                "dataset": "shuttleset_25", "class_count": 25,
                "total_hit_count": 1, "predicted_hit_count": 1,
                "gate_policy": {
                    "version": 1,
                    "type_report": {"min_probability": .60, "min_pose_coverage": .08, "path": "/private/type"},
                    "motion_coach": {"min_probability": .85, "min_top1_top2_margin": .25, "min_pose_coverage": .85, "min_ball_coverage": .55},
                },
                "notice": "safe", "quality_notice": "type only",
                "hits": [{
                    "frame": 20, "seconds": .8, "timecode": "00:00",
                    "window_start": .2, "window_end": 1.4,
                    "rally_id": 1, "hit_index": 2, "hitter": "near",
                    "model_side": "near", "model_label": "杀球",
                    "confidence": 74, "confidence_level": "ready",
                    "top1_probability": .74, "top1_top2_margin": .31,
                    "margin_available": True, "type_ready": True,
                    "type_gate_reason": "ready", "coach_ready": False,
                    "coach_gate_reason": "low_confidence",
                    "pose_coverage": .9, "ball_coverage": .8,
                    "rule_candidate": "杀球候选", "path": "/private/model.json",
                    "top3": [{"class_id": 2, "raw_label": "Bottom_殺球", "label": "杀球", "probability": .74, "confidence": 74, "path": "/private/top.json"}],
                }],
            },
        })
        report = public["stroke_types"]
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["hits"][0]["model_label"], "杀球")
        self.assertEqual(report["hits"][0]["top3"][0]["class_id"], 2)
        self.assertEqual(report["hits"][0]["top1_top2_margin"], .31)
        self.assertFalse(report["hits"][0]["coach_ready"])
        self.assertEqual(report["gate_policy"]["motion_coach"]["min_probability"], .85)
        self.assertNotIn("path", str(report))


if __name__ == "__main__":
    unittest.main()
