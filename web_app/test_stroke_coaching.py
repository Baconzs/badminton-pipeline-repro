"""Regression tests for conservative per-stroke Motion Coach rules."""

import unittest

from web_app.stroke_coaching import (
    classify_stroke,
    evaluate_stroke,
    extract_pose_metrics,
    select_bst_stroke_type,
    summarize_reviews,
    zone_for_player_position,
)


def pose(*, right_wrist=(126, 70), right_elbow=(102, 80), knee_y=224):
    points = [[0.0, 0.0, 0.0] for _ in range(17)]
    entries = {
        5: (42, 100), 6: (82, 100),
        7: (23, 109), 8: right_elbow,
        9: (8, 118), 10: right_wrist,
        11: (45, 166), 12: (78, 166),
        13: (45, knee_y), 14: (78, knee_y),
        15: (45, 260), 16: (78, 260),
    }
    for index, (x, y) in entries.items():
        points[index] = [float(x), float(y), 0.94]
    return points


def phase_pose(
    *,
    scale=1.0,
    right_wrist=(90, 110),
    right_elbow=(88, 96),
    shoulder_tilt=0.0,
    hip_tilt=0.0,
    knee_shift=10.0,
):
    """Synthetic COCO-17 pose with independently changing phase signals."""
    points = pose(right_wrist=right_wrist, right_elbow=right_elbow)
    points[5][1] += shoulder_tilt / 2.0
    points[6][1] -= shoulder_tilt / 2.0
    points[11][1] += hip_tilt / 2.0
    points[12][1] -= hip_tilt / 2.0
    points[13][0] -= knee_shift
    points[14][0] += knee_shift
    for point in points:
        if len(point) >= 3 and point[2] > 0:
            point[0] *= scale
            point[1] *= scale
    return points


def phase_samples(*, scale=1.0):
    """A preparation/contact/recovery timeline around frame 100."""
    timeline = (
        (80, (82, 132), (84, 111), 0, 0, 16),
        (85, (84, 128), (85, 108), 1, 0, 15),
        (90, (91, 114), (89, 101), 4, -2, 12),
        (95, (108, 86), (97, 84), 10, -4, 6),
        (100, (126, 70), (102, 80), 16, -8, 0),
        (105, (114, 82), (99, 84), 12, -5, 4),
        (110, (98, 98), (92, 91), 6, -2, 8),
        (115, (90, 106), (88, 96), 2, 0, 10),
        (120, (90, 106), (88, 96), 2, 0, 10),
        (125, (90, 106), (88, 96), 2, 0, 10),
    )
    return [
        {
            "frame": frame,
            "keypoints": phase_pose(
                scale=scale,
                right_wrist=wrist,
                right_elbow=elbow,
                shoulder_tilt=shoulder_tilt,
                hip_tilt=hip_tilt,
                knee_shift=knee_shift,
            ),
            "court_position": (3.0, 5.0),
        }
        for frame, wrist, elbow, shoulder_tilt, hip_tilt, knee_shift in timeline
    ]


def samples():
    return [
        {"frame": 94, "keypoints": pose(), "court_position": (3.0, 4.9)},
        {"frame": 100, "keypoints": pose(), "court_position": (3.0, 5.0)},
        {"frame": 106, "keypoints": pose(), "court_position": (3.1, 5.2)},
    ]


def event(**extra):
    value = {
        "confirmed": True,
        "rally_hit_index": 2,
        "frame": 100,
        "hit_point": (132.0, 70.0),
    }
    value.update(extra)
    return value


def trajectory(**extra):
    value = {
        "real_points": 6,
        "arc_signal": 0.18,
        "flight_seconds": 1.25,
        "speed_reliable": True,
        "speed_percentile": 0.60,
        "speed_kmh": 48.0,
    }
    value.update(extra)
    return value


def bst_hit(**extra):
    value = {
        "model_label": "杀球",
        "model_raw_label": "Bottom_殺球",
        "confidence": 91.0,
        "confidence_level": "ready",
        "event_confirmed": True,
        "pose_coverage": 0.90,
        "ball_coverage": 0.76,
        "hitter": "near",
        "model_side": "near",
        "top3": [
            {"label": "杀球", "confidence": 91.0},
            {"label": "高远球", "confidence": 54.0},
            {"label": "吊球", "confidence": 12.0},
        ],
    }
    value.update(extra)
    return value


class StrokeCoachingTests(unittest.TestCase):
    def test_high_confidence_bst_can_supply_type_and_recompute_advice(self):
        rule = {
            "family": "unknown", "label": "球路信息不足", "base_label": "球路信息不足",
            "area": "unknown", "confidence": 48, "confidence_level": "review",
            "hand_label": "正反手信息不足", "hand_status": "review", "reasons": ["路线不足"],
        }
        selected, classification = select_bst_stroke_type(rule, bst_hit(), player="near")
        self.assertEqual(selected["family"], "smash")
        self.assertEqual(classification["source"], "bst")
        self.assertEqual(classification["status"], "adopted")
        self.assertFalse(classification["advice_eligible"])
        self.assertEqual(classification["coaching_gate"]["reason_code"], "route_type_unconfirmed")
        self.assertEqual(rule["family"], "unknown")
        metrics = extract_pose_metrics(samples(), hit_frame=100, hit_point=(132, 70))
        evaluation = evaluate_stroke(selected, metrics, trajectory(speed_percentile=.91, arc_signal=.08))
        self.assertTrue(any("杀球" in item for item in evaluation["suggestions"]))

    def test_bst_low_confidence_keeps_rule_type(self):
        rule = {
            "family": "clear", "label": "高远球", "base_label": "高远球", "area": "back",
            "confidence": 72, "confidence_level": "medium", "reasons": ["球路"],
        }
        selected, classification = select_bst_stroke_type(rule, bst_hit(confidence=59.9, confidence_level="review"), player="near")
        self.assertEqual(selected["family"], "clear")
        self.assertEqual(classification["source"], "rule")
        self.assertEqual(classification["status"], "rule_fallback")
        self.assertFalse(classification["type_conflict"])
        self.assertTrue(classification["advice_eligible"])
        self.assertTrue(classification["coaching_gate"]["eligible"])
        self.assertEqual(classification["coaching_gate"]["source"], "rule_fallback")

    def test_bst_conflict_keeps_rule_and_exposes_model_alternative(self):
        rule = {
            "family": "clear", "label": "高远球", "base_label": "高远球", "area": "back",
            "confidence": 72, "confidence_level": "medium", "reasons": ["球路"],
        }
        selected, classification = select_bst_stroke_type(rule, bst_hit(), player="near")
        self.assertEqual(selected["family"], "clear")
        self.assertEqual(classification["status"], "conflict")
        self.assertTrue(classification["type_conflict"])
        self.assertFalse(classification["advice_eligible"])
        self.assertFalse(classification["coaching_gate"]["eligible"])
        self.assertEqual(classification["bst_label"], "杀球")
        self.assertEqual(classification["rule_label"], "高远球")

    def test_bst_side_or_unmapped_type_falls_back(self):
        rule = {
            "family": "lift", "label": "挑球", "base_label": "挑球", "area": "front",
            "confidence": 70, "confidence_level": "medium", "reasons": ["球路"],
        }
        selected, classification = select_bst_stroke_type(rule, bst_hit(model_side="far"), player="near")
        self.assertEqual(selected["family"], "lift")
        self.assertEqual(classification["status"], "side_mismatch")
        self.assertFalse(classification["advice_eligible"])
        self.assertFalse(classification["coaching_gate"]["eligible"])
        _, weak_side_mismatch = select_bst_stroke_type(
            rule,
            bst_hit(model_side="far", confidence=59.9, confidence_level="review"),
            player="near",
        )
        self.assertEqual(weak_side_mismatch["status"], "side_mismatch")
        self.assertFalse(weak_side_mismatch["advice_eligible"])
        selected, classification = select_bst_stroke_type(rule, bst_hit(model_label="平球", model_raw_label="Bottom_平球"), player="near")
        self.assertEqual(selected["family"], "lift")
        self.assertEqual(classification["status"], "unsupported")

    def test_only_strict_bst_route_agreement_can_enter_coaching(self):
        rule = {
            "family": "smash", "label": "杀球", "base_label": "杀球", "area": "back",
            "confidence": 82, "confidence_level": "medium", "reasons": ["球路"],
        }
        selected, classification = select_bst_stroke_type(rule, bst_hit(), player="near")
        self.assertEqual(selected["family"], "smash")
        self.assertEqual(classification["status"], "agreement")
        self.assertTrue(classification["advice_eligible"])
        self.assertEqual(classification["bst_margin"], 37.0)

        strict_failures = (
            bst_hit(confidence=84.9),
            bst_hit(top3=[{"confidence": 91}, {"confidence": 70}]),
            bst_hit(pose_coverage=0.849),
            bst_hit(ball_coverage=0.549),
        )
        for hint in strict_failures:
            with self.subTest(hint=hint):
                _, rejected = select_bst_stroke_type(rule, hint, player="near")
                self.assertEqual(rejected["status"], "rule_fallback")
                self.assertEqual(rejected["bst_status"], "agreement")
                self.assertEqual(rejected["source"], "rule")
                self.assertTrue(rejected["advice_eligible"])
                self.assertTrue(rejected["coaching_gate"]["eligible"])

    def test_player_zone_uses_the_players_own_net_side(self):
        self.assertEqual(zone_for_player_position((3.0, 7.6), "near"), "front")
        self.assertEqual(zone_for_player_position((3.0, 12.1), "near"), "back")
        self.assertEqual(zone_for_player_position((3.0, 5.8), "far"), "front")
        self.assertEqual(zone_for_player_position((3.0, 1.2), "far"), "back")
        self.assertIsNone(zone_for_player_position((3.0, 8.9), "near"))  # boundary buffer

    def test_front_lift_requires_route_evidence_and_can_add_forehand_label(self):
        metrics = extract_pose_metrics(samples(), hit_frame=100, hit_point=(132, 70))
        stroke = classify_stroke(
            event(), trajectory(), origin_zone="front", receiver_zone="back",
            pose=metrics, dominant_hand="right",
        )
        self.assertEqual(stroke["family"], "lift")
        self.assertEqual(stroke["label"], "正手挑球")
        self.assertGreaterEqual(stroke["confidence"], 60)

    def test_hand_is_held_for_review_without_declared_handedness(self):
        metrics = extract_pose_metrics(samples(), hit_frame=100, hit_point=(132, 70))
        stroke = classify_stroke(
            event(), trajectory(), origin_zone="front", receiver_zone="back",
            pose=metrics, dominant_hand="auto",
        )
        self.assertEqual(stroke["label"], "挑球")
        self.assertEqual(stroke["hand_label"], "设置持拍手可区分正反手")
        self.assertEqual(stroke["hand_status"], "review")

    def test_low_quality_measured_hit_keeps_a_downgraded_route_candidate(self):
        metrics = extract_pose_metrics(samples(), hit_frame=100, hit_point=(132, 70))
        stroke = classify_stroke(
            event(confirmed=False, route_candidate=True), trajectory(),
            origin_zone="front", receiver_zone="back", pose=metrics,
        )
        self.assertEqual(stroke["family"], "lift")
        self.assertLessEqual(stroke["confidence"], 58)
        self.assertIn("击球信号较弱", " ".join(stroke["reasons"]))

    def test_attribution_penalties_are_applied_after_score_cap(self):
        metrics = extract_pose_metrics(samples(), hit_frame=100, hit_point=(132, 70))
        clean = classify_stroke(
            event(), trajectory(), origin_zone="front", receiver_zone="back", pose=metrics,
        )
        inferred_hitter = classify_stroke(
            event(hitter_inferred=True), trajectory(),
            origin_zone="front", receiver_zone="back", pose=metrics,
        )
        inferred_receiver = classify_stroke(
            event(), trajectory(receiver_side_inferred=True),
            origin_zone="front", receiver_zone="back", pose=metrics,
        )

        self.assertEqual(clean["confidence"], 88)
        self.assertEqual(inferred_hitter["confidence"], 76)
        self.assertEqual(inferred_receiver["confidence"], 80)

    def test_backcourt_smash_has_a_compact_product_label(self):
        metrics = extract_pose_metrics(samples(), hit_frame=100, hit_point=(132, 70))
        stroke = classify_stroke(
            event(), trajectory(speed_percentile=0.91, arc_signal=0.08),
            origin_zone="back", receiver_zone="front", pose=metrics,
            dominant_hand="right",
        )
        self.assertEqual(stroke["family"], "smash")
        self.assertEqual(stroke["label"], "杀球")
        self.assertNotIn("候选", stroke["label"])
        self.assertNotIn("待确认", stroke["label"])

    def test_product_copy_avoids_review_placeholders(self):
        metrics = extract_pose_metrics(samples(), hit_frame=100, hit_point=(132, 70))
        stroke = classify_stroke(
            event(), trajectory(), origin_zone="front", receiver_zone="back",
            pose=metrics, dominant_hand="auto",
        )
        evaluation = evaluate_stroke(stroke, metrics, trajectory())
        visible_text = " ".join([
            str(stroke.get("label", "")),
            str(stroke.get("base_label", "")),
            str(stroke.get("hand_label", "")),
            str(evaluation.get("title", "")),
            str(evaluation.get("detail", "")),
            " ".join(evaluation.get("suggestions", [])),
        ])
        for placeholder in ("候选", "待确认", "待复核"):
            self.assertNotIn(placeholder, visible_text)

    def test_front_compact_stroke_is_not_scored_by_overhead_extension_rule(self):
        metrics = extract_pose_metrics(samples(), hit_frame=100, hit_point=(132, 70))
        stroke = classify_stroke(
            event(), trajectory(arc_signal=0.03, flight_seconds=0.45),
            origin_zone="front", receiver_zone="mid", pose=metrics,
        )
        evaluation = evaluate_stroke(stroke, metrics, trajectory())
        joined = " ".join(evaluation["suggestions"])
        self.assertEqual(stroke["family"], "push")
        self.assertNotIn("上手高点击球", joined)

    def test_evaluation_exposes_candidate_but_not_reliable_single_hit_advice(self):
        metrics = extract_pose_metrics(samples(), hit_frame=100, hit_point=(132, 70))
        stroke = classify_stroke(
            event(), trajectory(), origin_zone="front", receiver_zone="back",
            pose=metrics, dominant_hand="right",
        )
        evaluation = evaluate_stroke(stroke, metrics, trajectory())
        self.assertEqual(evaluation["advice_status"], "candidate")
        self.assertTrue(evaluation["coaching_gate"]["eligible_for_aggregation"])
        self.assertEqual(evaluation["advice_candidates"][0]["problem"], "最后一步支撑连续偏直或偏窄。")
        self.assertEqual(evaluation["advice_candidates"][0]["action"], "最后一步落稳并轻微屈膝，再完成出拍。")

    def test_low_pose_or_trajectory_evidence_suppresses_issue_candidates(self):
        metrics = extract_pose_metrics(samples(), hit_frame=100, hit_point=(132, 70))
        stroke = classify_stroke(
            event(), trajectory(), origin_zone="front", receiver_zone="back",
            pose=metrics, dominant_hand="right",
        )
        for weak_metrics, weak_trajectory, reason in (
            ({**metrics, "pose_confidence": 0.79}, trajectory(), "pose_confidence_low"),
            (metrics, trajectory(real_points=5), "trajectory_points_low"),
        ):
            with self.subTest(reason=reason):
                evaluation = evaluate_stroke(stroke, weak_metrics, weak_trajectory)
                self.assertEqual(evaluation["advice_status"], "suppressed")
                self.assertEqual(evaluation["eligible_signals"], [])
                self.assertIn(reason, evaluation["coaching_gate"]["reason_codes"])

    def test_empty_keypoint_arrays_do_not_count_as_contact_pose_samples(self):
        missing_pose = [[0.0, 0.0, 0.0] for _ in range(17)]
        sparse_samples = [
            {"frame": 94, "keypoints": pose(), "court_position": (3.0, 4.9)},
            {"frame": 100, "keypoints": missing_pose, "court_position": (3.0, 5.0)},
            {"frame": 106, "keypoints": missing_pose, "court_position": (3.1, 5.2)},
        ]
        metrics = extract_pose_metrics(sparse_samples, hit_frame=100, hit_point=(132, 70))
        stroke = {
            "family": "lift", "label": "挑球", "base_label": "挑球",
            "area": "front", "confidence": 82, "confidence_level": "medium",
        }
        evaluation = evaluate_stroke(stroke, metrics, trajectory())

        self.assertEqual(metrics["contact_samples"], 1)
        self.assertEqual(evaluation["advice_status"], "suppressed")
        self.assertIn("contact_samples_low", evaluation["coaching_gate"]["reason_codes"])

    def test_one_frame_limb_outlier_cannot_create_an_issue_signal(self):
        partial = pose()
        for point in partial:
            if len(point) >= 3 and point[2] > 0:
                point[2] = 1.0
        # Torso/arms stay clear, but missing ankles make knee and stance
        # measurements unavailable in two of the three contact frames.
        partial[15] = [0.0, 0.0, 0.0]
        partial[16] = [0.0, 0.0, 0.0]
        sparse_limb_samples = [
            {"frame": 94, "keypoints": pose(), "court_position": (3.0, 4.9)},
            {"frame": 100, "keypoints": partial, "court_position": (3.0, 5.0)},
            {"frame": 106, "keypoints": partial, "court_position": (3.1, 5.2)},
        ]
        metrics = extract_pose_metrics(sparse_limb_samples, hit_frame=100, hit_point=(220, 30))
        stroke = {
            "family": "lift", "label": "挑球", "base_label": "挑球",
            "area": "front", "confidence": 82, "confidence_level": "medium",
        }
        evaluation = evaluate_stroke(stroke, metrics, trajectory())

        self.assertEqual(metrics["contact_samples"], 3)
        self.assertGreaterEqual(metrics["pose_confidence"], 0.80)
        self.assertEqual(metrics["knee_samples"], 1)
        self.assertEqual(metrics["stance_samples"], 1)
        self.assertNotIn("front_support", evaluation["eligible_signals"])

    def test_phase_metrics_are_scale_normalized_across_a_full_stroke_timeline(self):
        normal = extract_pose_metrics(
            phase_samples(scale=1.0),
            hit_frame=100,
            hit_point=(126, 70),
            fps=25.0,
            dominant_hand="right",
        )
        doubled = extract_pose_metrics(
            phase_samples(scale=2.0),
            hit_frame=100,
            hit_point=(252, 140),
            fps=25.0,
            dominant_hand="right",
        )

        for result in (normal, doubled):
            gate = result["phase_gate"]
            self.assertTrue(gate["eligible"])
            self.assertEqual(gate["arm"], "right")
            self.assertGreaterEqual(gate["pose_confidence"], 0.90)
            self.assertIsInstance(gate["reason_codes"], list)

        fields = (
            "prep_wrist_height_over_torso",
            "prep_knee_angle_deg",
            "contact_wrist_speed_torso_s",
            "peak_shoulder_hip_change_deg",
            "peak_elbow_angular_speed_deg_s",
            "peak_knee_angular_speed_deg_s",
            "recovery_stable_seconds",
        )
        for field in fields:
            with self.subTest(field=field):
                first = normal["phase_metrics"][field]
                second = doubled["phase_metrics"][field]
                self.assertIsNotNone(first)
                self.assertIsNotNone(second)
                self.assertAlmostEqual(first, second, delta=0.03)

        self.assertGreater(normal["phase_metrics"]["contact_wrist_speed_torso_s"], 0)
        self.assertGreater(normal["phase_metrics"]["peak_elbow_angular_speed_deg_s"], 0)
        self.assertGreater(normal["phase_metrics"]["peak_knee_angular_speed_deg_s"], 0)
        self.assertGreater(normal["phase_metrics"]["recovery_stable_seconds"], 0)

    def test_phase_metrics_need_more_than_a_three_frame_contact_window(self):
        sparse = [
            sample for sample in phase_samples()
            if sample["frame"] in {95, 100, 105}
        ]
        metrics = extract_pose_metrics(
            sparse,
            hit_frame=100,
            hit_point=(126, 70),
            fps=25.0,
            dominant_hand="right",
        )

        # Contact-only rules still have their historical three visible frames,
        # but that is not enough evidence for preparation/contact/recovery
        # timing claims.
        self.assertEqual(metrics["contact_samples"], 3)
        self.assertFalse(metrics["phase_gate"]["eligible"])
        self.assertTrue(metrics["phase_gate"]["reason_codes"])
        self.assertTrue(any(
            "sample" in str(code)
            for code in metrics["phase_gate"]["reason_codes"]
        ))

    def test_unknown_route_can_emit_only_an_independently_gated_phase_signal(self):
        frames = (88, 92, 95, 98, 100, 102, 106, 110, 114, 120)
        timeline = [
            {"frame": frame, "keypoints": pose(), "court_position": (3.0, 5.0)}
            for frame in frames
        ]
        metrics = extract_pose_metrics(
            timeline,
            hit_frame=100,
            hit_point=(132, 70),
            fps=25.0,
            dominant_hand="right",
        )
        evaluation = evaluate_stroke(
            {"family": "unknown", "confidence": 58},
            metrics,
            trajectory(event_confirmed=True),
        )

        self.assertEqual(evaluation["signals"], [])
        self.assertEqual(evaluation["eligible_signals"], [])
        self.assertEqual(evaluation["phase_eligible_signals"], ["preparation_support"])
        self.assertTrue(evaluation["phase_advice_eligible"])
        self.assertTrue(evaluation["phase_gate"]["eligible"])

    @staticmethod
    def _review(frame, *, family="lift", signal="front_support", **extra):
        value = {
            "rally_id": frame // 100,
            "hit_index": 1,
            "stroke": {"family": family, "confidence": 82},
            "evaluation": {"signals": [signal]},
            "evidence": {"frame": frame, "seconds": frame / 25, "timecode": f"00:{frame // 25:02d}"},
        }
        value.update(extra)
        return value

    def test_recommendations_need_three_independent_same_family_signals(self):
        reviews = [self._review(frame) for frame in (100, 200, 300)]
        summaries, recommendations, quality = summarize_reviews(reviews)
        self.assertEqual(summaries[0]["family"], "lift")
        self.assertEqual(len(recommendations), 1)
        advice = recommendations[0]
        self.assertEqual(advice["status"], "reliable")
        self.assertEqual(advice["family"], "frontcourt")
        self.assertEqual(advice["repeat_count"], 3)
        self.assertEqual(advice["problem"], "最后一步支撑连续偏直或偏窄。")
        self.assertEqual(advice["action"], "最后一步落稳并轻微屈膝，再完成出拍。")
        self.assertGreaterEqual(quality, 70)

    def test_overhead_extension_combines_compatible_overhead_families(self):
        reviews = [
            self._review(100, family="clear", signal="overhead_extension"),
            self._review(200, family="drive_clear", signal="overhead_extension"),
            self._review(300, family="smash", signal="overhead_extension"),
        ]
        _, recommendations, _ = summarize_reviews(reviews)

        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["signal"], "overhead_extension")
        self.assertEqual(recommendations[0]["repeat_count"], 3)
        self.assertEqual(recommendations[0]["status"], "reliable")

        # Compatibility does not lower the independent-hit threshold.
        _, only_two, _ = summarize_reviews(reviews[:2])
        self.assertEqual(only_two, [])
        _, duplicate_third, _ = summarize_reviews(reviews[:2] + [reviews[0]])
        self.assertEqual(duplicate_third, [])

    def test_unknown_type_can_aggregate_only_explicitly_gated_phase_signals(self):
        def phase_review(frame, *, eligible=True):
            return self._review(
                frame,
                family="unknown",
                evaluation={
                    "signals": [],
                    "eligible_signals": [],
                    "phase_signals": ["overhead_preparation"],
                    "phase_advice_eligible": eligible,
                    "phase_gate": {
                        "eligible": eligible,
                        "reason_codes": [] if eligible else ["phase_samples_low"],
                    },
                },
            )

        reviews = [phase_review(frame) for frame in (100, 200, 300)]
        _, recommendations, _ = summarize_reviews(reviews)
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["signal"], "overhead_preparation")
        self.assertEqual(recommendations[0]["repeat_count"], 3)

        # A blocked phase record contributes no vote, leaving only two hits.
        blocked = reviews[:2] + [phase_review(300, eligible=False)]
        _, recommendations, _ = summarize_reviews(blocked)
        self.assertEqual(recommendations, [])

    def test_duplicates_do_not_combine_and_compatible_front_families_do(self):
        duplicate = self._review(100)
        _, recommendations, _ = summarize_reviews([duplicate, duplicate, duplicate])
        self.assertEqual(recommendations, [])

        # Refining a contact frame must not turn one logical rally hit into
        # three independent votes.
        shifted_duplicates = [
            self._review(frame, rally_id=7, hit_index=2)
            for frame in (100, 101, 102)
        ]
        _, recommendations, _ = summarize_reviews(shifted_duplicates)
        self.assertEqual(recommendations, [])

        mixed = [
            self._review(100, family="lift"),
            self._review(200, family="push"),
            self._review(300, family="net"),
        ]
        _, recommendations, _ = summarize_reviews(mixed)
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["family"], "frontcourt")
        self.assertEqual(recommendations[0]["repeat_count"], 3)

        # Conflicting copies of the same physical frames cannot each build a
        # separate three-hit family aggregate.
        conflicting_copies = []
        for frame in (100, 200, 300):
            conflicting_copies.extend([
                self._review(frame, family="lift"),
                self._review(frame, family="push"),
            ])
        _, recommendations, _ = summarize_reviews(conflicting_copies)
        self.assertEqual(recommendations, [])

    def test_low_evidence_and_type_conflicts_cast_no_advice_vote(self):
        good = [self._review(frame) for frame in (100, 200)]
        conflict = self._review(
            300,
            classification={
                "source": "rule", "status": "conflict", "advice_eligible": False,
                "coaching_gate": {"eligible": False},
            },
        )
        _, recommendations, _ = summarize_reviews(good + [conflict])
        self.assertEqual(recommendations, [])

        low_pose = self._review(
            300,
            evaluation={
                "signals": ["front_support"],
                "eligible_signals": [],
                "coaching_gate": {"eligible_for_aggregation": False},
            },
        )
        _, recommendations, _ = summarize_reviews(good + [low_pose])
        self.assertEqual(recommendations, [])

        unverified_hitter = self._review(
            300,
            attribution_gate={
                "eligible": False,
                "reason_code": "hitter_attribution_unverified",
            },
        )
        _, recommendations, _ = summarize_reviews(good + [unverified_hitter])
        self.assertEqual(recommendations, [])


if __name__ == "__main__":
    unittest.main()
