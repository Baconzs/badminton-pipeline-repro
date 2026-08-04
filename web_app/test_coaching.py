import unittest

from web_app.bst_adapter import assess_bst_hit_gates
from web_app.coaching import (
    _apply_pose_hitter_associations,
    _correct_rally_hitter_alternation,
    _eligible_coaching_reviews,
    _review_for_event,
    _point_in_convex_quad,
    _rally_hitter_balance,
    build_coaching_report,
    run_pose_coaching,
)


def pose(*, wrist_x=20, knee_y=180, ankle_right_x=30):
    """A confident COCO-17 pose with deliberately configurable signals."""
    points = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, x, y in [
        (5, 0, 0), (6, 100, 0), (7, 10, 10), (8, 90, 10),
        (9, wrist_x, 20), (10, 100 - wrist_x, 20),
        (11, 0, 100), (12, 100, 100), (13, 0, knee_y),
        (14, 100, knee_y), (15, 0, 260), (16, ankle_right_x, 260),
    ]:
        points[index] = [float(x), float(y), 0.95]
    return points


def hit_event(frame, rally_id, hitter, confidence=0.95, **extra):
    """Minimal sidecar-derived event used by hitter-attribution tests."""
    event = {
        "frame": frame,
        "rally_id": rally_id,
        "hitter": hitter,
        "hitter_confidence": confidence,
    }
    event.update(extra)
    return event


class CoachingReportTests(unittest.TestCase):
    def test_only_advice_eligible_reviews_reach_visible_aggregation(self):
        accepted = {"classification": {"advice_eligible": True}, "evidence": {"frame": 10}}
        conflict = {"classification": {"advice_eligible": False}, "evidence": {"frame": 20}}

        self.assertEqual(_eligible_coaching_reviews([conflict, accepted]), [accepted])

    def test_review_recomputes_advice_after_bst_type_selection(self):
        pose_by_frame = {
            frame: {"near": {"keypoints": pose(), "court_position": None}}
            for frame in (94, 100, 106)
        }
        event = {
            "frame": 100,
            "rally_id": 1,
            "rally_hit_index": 2,
            "confirmed": True,
            "route_candidate": True,
            "hit_point": (20.0, 20.0),
            "sample_offsets": [-6, 0, 6],
            "trajectory": {
                "real_points": 6, "arc_signal": .08, "flight_seconds": .5,
                "speed_reliable": True, "speed_percentile": .91, "speed_kmh": 80,
            },
        }
        hint = {
            "frame": 100, "rally_id": 1, "hit_index": 2, "hitter": "near",
            "model_side": "near", "model_label": "杀球", "model_raw_label": "Bottom_殺球",
            "confidence": 84, "confidence_level": "ready", "event_confirmed": True,
            "pose_coverage": .8, "ball_coverage": .8,
        }
        review = _review_for_event(
            event, player="near", dominant_hand="right", fps=25,
            pose_by_frame=pose_by_frame, bst_hint=hint,
        )
        self.assertEqual(review["classification"]["source"], "bst")
        self.assertEqual(review["stroke"]["family"], "smash")
        self.assertTrue(any("杀球" in item for item in review["evaluation"]["suggestions"]))
        self.assertFalse(review["advice_eligible"])

    def test_low_confidence_bst_falls_back_to_reliable_rule_for_coaching(self):
        pose_by_frame = {
            frame: {"near": {"keypoints": pose(), "court_position": (3.0, 12.0)}}
            for frame in (94, 100, 106)
        }
        pose_by_frame[120] = {
            "far": {"keypoints": pose(), "court_position": (3.0, 5.8)},
        }
        event = {
            "frame": 100,
            "rally_id": 1,
            "rally_hit_index": 2,
            "confirmed": True,
            "route_candidate": True,
            "receiver": "far",
            "shot_end_frame": 120,
            "hit_point": (20.0, 20.0),
            "sample_offsets": [-6, 0, 6],
            "trajectory": {
                "real_points": 6, "arc_signal": .08, "flight_seconds": .5,
                "speed_reliable": True, "speed_percentile": .70, "speed_kmh": 55,
            },
        }
        hint = {
            "frame": 100, "rally_id": 1, "hit_index": 2,
            "hitter": "near", "model_side": "near",
            "model_label": "杀球", "model_raw_label": "Bottom_殺球",
            "confidence": 59.9, "confidence_level": "review",
            "event_confirmed": True, "pose_coverage": .92, "ball_coverage": .72,
        }

        review = _review_for_event(
            event,
            player="near",
            dominant_hand="right",
            fps=25,
            pose_by_frame=pose_by_frame,
            bst_hint=hint,
        )

        self.assertEqual(review["stroke"]["family"], "drop")
        self.assertEqual(review["classification"]["source"], "rule")
        self.assertEqual(review["classification"]["status"], "rule_fallback")
        self.assertTrue(review["classification"]["advice_eligible"])
        self.assertTrue(review["classification"]["coaching_gate"]["eligible"])
        self.assertTrue(review["advice_eligible"])
        self.assertEqual(_eligible_coaching_reviews([review]), [review])

    def test_strict_bst_rule_agreement_can_enter_coaching(self):
        pose_by_frame = {
            frame: {"near": {"keypoints": pose(), "court_position": (3.0, 12.0)}}
            for frame in (94, 100, 106)
        }
        pose_by_frame[120] = {
            "far": {"keypoints": pose(), "court_position": (3.0, 5.8)},
        }
        event = {
            "frame": 100,
            "rally_id": 1,
            "rally_hit_index": 2,
            "confirmed": True,
            "route_candidate": True,
            "receiver": "far",
            "shot_end_frame": 120,
            "hit_point": (20.0, 20.0),
            "sample_offsets": [-6, 0, 6],
            "trajectory": {
                "real_points": 6, "arc_signal": .08, "flight_seconds": .5,
                "speed_reliable": True, "speed_percentile": .91, "speed_kmh": 80,
            },
        }
        hint = {
            "frame": 100, "rally_id": 1, "hit_index": 2, "hitter": "near",
            "model_side": "near", "model_label": "吊球", "model_raw_label": "Bottom_切球",
            "confidence": 92, "confidence_level": "ready", "event_confirmed": True,
            "pose_coverage": .92, "ball_coverage": .72,
            "top3": [
                {"probability": .92}, {"probability": .05}, {"probability": .03},
            ],
        }
        hint.update(assess_bst_hit_gates(hint))

        review = _review_for_event(
            event, player="near", dominant_hand="right", fps=25,
            pose_by_frame=pose_by_frame, bst_hint=hint,
        )

        self.assertEqual(review["classification"]["status"], "agreement")
        self.assertTrue(review["classification"]["coaching_gate"]["eligible"])
        self.assertTrue(review["advice_eligible"])
        self.assertEqual(_eligible_coaching_reviews([review]), [review])

    def test_bst_route_conflict_is_removed_from_coaching(self):
        pose_by_frame = {
            frame: {"near": {"keypoints": pose(), "court_position": (3.0, 12.0)}}
            for frame in (94, 100, 106)
        }
        pose_by_frame[120] = {
            "far": {"keypoints": pose(), "court_position": (3.0, 5.8)},
        }
        event = {
            "frame": 100, "rally_id": 1, "rally_hit_index": 2,
            "confirmed": True, "route_candidate": True, "receiver": "far",
            "shot_end_frame": 120, "hit_point": (20.0, 20.0),
            "sample_offsets": [-6, 0, 6],
            "trajectory": {
                "real_points": 6, "arc_signal": .08, "flight_seconds": .5,
                "speed_reliable": True, "speed_percentile": .70, "speed_kmh": 55,
            },
        }
        hint = {
            "frame": 100, "rally_id": 1, "hit_index": 2, "hitter": "near",
            "model_side": "near", "model_label": "杀球", "model_raw_label": "Bottom_殺球",
            "confidence": 93, "confidence_level": "ready", "event_confirmed": True,
            "pose_coverage": .94, "ball_coverage": .75,
            "top3": [
                {"probability": .93}, {"probability": .04}, {"probability": .03},
            ],
        }
        hint.update(assess_bst_hit_gates(hint))

        review = _review_for_event(
            event, player="near", dominant_hand="right", fps=25,
            pose_by_frame=pose_by_frame, bst_hint=hint,
        )

        self.assertEqual(review["classification"]["status"], "conflict")
        self.assertFalse(review["advice_eligible"])
        self.assertEqual(_eligible_coaching_reviews([review]), [])

    def test_rally_alternation_hitter_never_drives_action_advice(self):
        pose_by_frame = {
            frame: {"near": {"keypoints": pose(), "court_position": (3.0, 12.0)}}
            for frame in (94, 100, 106)
        }
        pose_by_frame[120] = {
            "far": {"keypoints": pose(), "court_position": (3.0, 5.8)},
        }
        event = {
            "frame": 100, "rally_id": 1, "rally_hit_index": 2,
            "confirmed": True, "route_candidate": True, "receiver": "far",
            "shot_end_frame": 120, "hit_point": (20.0, 20.0),
            "sample_offsets": [-6, 0, 6],
            "hitter_inferred": True, "hitter_confidence": .55,
            "hitter_provenance": "rally_alternation_inferred",
            "trajectory": {
                "real_points": 6, "arc_signal": .08, "flight_seconds": .5,
                "speed_reliable": True, "speed_percentile": .70, "speed_kmh": 55,
            },
        }

        review = _review_for_event(
            event, player="near", dominant_hand="right", fps=25,
            pose_by_frame=pose_by_frame,
        )

        self.assertTrue(review["classification"]["advice_eligible"])
        self.assertFalse(review["advice_eligible"])
        self.assertEqual(review["attribution_gate"]["reason_code"], "hitter_attribution_unverified")
        self.assertEqual(review["evaluation"]["advice_status"], "suppressed")
        self.assertEqual(review["evaluation"]["eligible_signals"], [])
        self.assertEqual(_eligible_coaching_reviews([review]), [])

    def test_low_confidence_motion_signals_have_timestamped_advice(self):
        report = build_coaching_report([
            {"frame": 25, "player": "near", "keypoints": pose()},
            {"frame": 50, "player": "near", "keypoints": pose()},
        ], fps=25.0)

        self.assertEqual(report["status"], "ready")
        player = report["players"][0]
        self.assertEqual(player["label"], "下方球员")
        self.assertLess(player["score"], 58)
        self.assertTrue(player["recommendations"])
        evidence = player["recommendations"][0]["evidence"]
        self.assertEqual(evidence[0]["timecode"], "00:01")
        self.assertIn("seconds", evidence[0])

    def test_one_pose_frame_does_not_create_a_score(self):
        report = build_coaching_report([
            {"frame": 25, "player": "far", "keypoints": pose(wrist_x=150, ankle_right_x=120)},
        ], fps=25.0)

        self.assertEqual(report["status"], "insufficient_evidence")
        self.assertEqual(report["players"], [])

    def test_missing_player_side_is_disclosed_in_notice(self):
        report = build_coaching_report([
            {"frame": 25, "player": "near", "keypoints": pose()},
            {"frame": 50, "player": "near", "keypoints": pose()},
        ], fps=25.0)

        self.assertEqual(report["status"], "ready")
        self.assertIn("上方球员姿态样本不足", report["notice"])
        self.assertNotIn("待确认", report["notice"])
        self.assertNotIn("待复核", report["notice"])

    def test_court_quad_filter_is_orientation_independent(self):
        clockwise = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        counter_clockwise = list(reversed(clockwise))

        self.assertTrue(_point_in_convex_quad((50.0, 50.0), clockwise))
        self.assertTrue(_point_in_convex_quad((50.0, 50.0), counter_clockwise))
        self.assertFalse(_point_in_convex_quad((120.0, 50.0), clockwise))

    def test_missing_model_is_safe_optional_result(self):
        report = run_pose_coaching(
            "/does/not/exist.mp4",
            "/does/not/exist.csv",
            fps=25,
            pose_weights="/does/not/exist.pt",
        )
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["players"], [])


class RallyHitterAlternationTests(unittest.TestCase):
    """Contract tests for conservative per-rally hitter attribution repair.

    Alternation is only supporting evidence.  It may fill an unknown label or
    correct a weak contradictory label surrounded by clear opposite-side
    contacts; it must never overwrite a confident observation merely to make
    the sequence look like a perfect rally.
    """

    def test_keeps_normal_high_confidence_alternation_and_does_not_mutate_input(self):
        events = [
            hit_event(10, 1, "near"),
            hit_event(22, 1, "far"),
            hit_event(34, 1, "near"),
        ]

        corrected = _correct_rally_hitter_alternation(events)

        self.assertEqual([event["hitter"] for event in corrected], ["near", "far", "near"])
        self.assertEqual([event["hitter"] for event in events], ["near", "far", "near"])
        self.assertTrue(all(not event["hitter_inferred"] for event in corrected))
        self.assertTrue(all(not event["hitter_corrected"] for event in corrected))
        self.assertTrue(all(event["hitter_provenance"] == "sidecar" for event in corrected))

    def test_infers_unknown_from_clear_in_rally_alternation(self):
        events = [
            hit_event(10, 1, "near", 0.93),
            hit_event(22, 1, "unknown", 0.0),
            hit_event(34, 1, "near", 0.94),
        ]

        corrected = _correct_rally_hitter_alternation(events)

        inferred = corrected[1]
        self.assertEqual(inferred["hitter"], "far")
        self.assertTrue(inferred["hitter_inferred"])
        self.assertFalse(inferred["hitter_corrected"])
        self.assertEqual(inferred["hitter_provenance"], "rally_alternation_inferred")

    def test_never_flips_consecutive_confident_same_side_events_after_missed_contact(self):
        events = [
            hit_event(10, 1, "near", 0.96),
            hit_event(22, 1, "far", 0.95),
            # A missed opponent contact can make two confirmed contacts from
            # the same side appear adjacent in the exported event sequence.
            hit_event(34, 1, "far", 0.97),
            hit_event(46, 1, "near", 0.95),
        ]

        corrected = _correct_rally_hitter_alternation(events)

        self.assertEqual([event["hitter"] for event in corrected], ["near", "far", "far", "near"])
        self.assertTrue(all(not event["hitter_corrected"] for event in corrected))
        self.assertEqual(corrected[2]["hitter_provenance"], "sidecar")

    def test_does_not_carry_alternation_evidence_across_rally_boundary(self):
        events = [
            hit_event(10, 1, "near", 0.96),
            hit_event(22, 1, "far", 0.95),
            # No supporting hit exists inside rally 2, so the previous rally
            # must not supply an inferred first hitter.
            hit_event(60, 2, "unknown", 0.0),
        ]

        corrected = _correct_rally_hitter_alternation(events)

        self.assertEqual(corrected[2]["hitter"], "unknown")
        self.assertFalse(corrected[2]["hitter_inferred"])
        self.assertFalse(corrected[2]["hitter_corrected"])
        self.assertEqual(corrected[2]["hitter_provenance"], "unknown")

    def test_corrects_only_low_confidence_wrong_label_with_two_sided_evidence(self):
        events = [
            hit_event(10, 1, "near", 0.96),
            # It contradicts two strong near contacts and is explicitly weak;
            # this is the only case where alternation may overwrite a label.
            hit_event(22, 1, "near", 0.28, hitter_direction="far", hitter_direction_confidence=0.68),
            hit_event(34, 1, "near", 0.97),
        ]

        corrected = _correct_rally_hitter_alternation(events, low_confidence_threshold=0.60)

        repaired = corrected[1]
        self.assertEqual(repaired["hitter"], "far")
        self.assertFalse(repaired["hitter_inferred"])
        self.assertTrue(repaired["hitter_corrected"])
        self.assertEqual(repaired["hitter_provenance"], "rally_alternation_corrected")

    def test_does_not_synthesize_a_long_unknown_span_by_global_parity(self):
        events = [
            hit_event(10, 1, "near", 0.96),
            hit_event(22, 1, "unknown", 0.0),
            hit_event(34, 1, "unknown", 0.0),
            # With two unknown events, this endpoint is parity-consistent.
            # It is still insufficient evidence to manufacture both sides.
            hit_event(46, 1, "far", 0.96),
        ]

        corrected = _correct_rally_hitter_alternation(events)

        self.assertEqual([event["hitter"] for event in corrected], ["near", "unknown", "unknown", "far"])
        self.assertTrue(all(not event["hitter_inferred"] for event in corrected[1:3]))
        self.assertTrue(all(event["hitter_provenance"] == "unknown" for event in corrected[1:3]))

    def test_balance_is_a_per_rally_diagnostic_not_a_relabel_target(self):
        events = [
            hit_event(10, 1, "near"),
            hit_event(22, 1, "far"),
            hit_event(34, 1, "unknown", 0.0),
            hit_event(60, 2, "near"),
            hit_event(72, 2, "near"),
            hit_event(84, 2, "near"),
        ]

        balance = _rally_hitter_balance(events)

        self.assertEqual(balance, [
            {
                "rally_id": 1,
                "near_count": 1,
                "far_count": 1,
                "unknown_count": 1,
                "difference": 0,
                "within_one": True,
                "complete": False,
            },
            {
                "rally_id": 2,
                "near_count": 3,
                "far_count": 0,
                "unknown_count": 0,
                "difference": 3,
                "within_one": False,
                "complete": True,
            },
        ])


class PoseAssociationSafetyTests(unittest.TestCase):
    def test_missing_pose_evidence_uses_initialized_previous_attribution_state(self):
        events = [
            hit_event(25, 1, "near", 0.48, hitter_provenance="sidecar"),
        ]

        associated = _apply_pose_hitter_associations(events, {})

        self.assertEqual(associated, 0)
        self.assertEqual(events[0]["hitter"], "unknown")
        self.assertEqual(events[0]["hitter_confidence"], 0.0)
        self.assertEqual(events[0]["hitter_provenance"], "pose_contact_unresolved")


if __name__ == "__main__":
    unittest.main()
