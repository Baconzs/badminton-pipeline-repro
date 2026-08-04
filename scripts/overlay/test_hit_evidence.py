#!/usr/bin/env python3
"""Focused regression tests for exported hit-event evidence gates."""

import csv
import tempfile
import unittest

from ball_tracking import (
    BallPoint,
    HitEvent,
    detect_hit_points,
    detect_top_edge_exit_hits,
    validate_hit_event_evidence,
)
from overlay_player_analytics import (
    apply_strict_pose_hitter_associations,
    has_confirmed_model_audio_evidence,
    infer_single_unknown_hitter_from_rally_anchors,
    merge_hit_events,
    merge_visual_hit_candidates,
    write_ball_tracking_csv,
)


def make_track(real_frames=(3, 4, 6, 7), length=12):
    track = [
        BallPoint(frame, 0.0, 0.0, False, "missing", False, 0.0)
        for frame in range(length)
    ]
    for frame in real_frames:
        track[frame] = BallPoint(
            frame,
            float(frame * 10),
            50.0,
            True,
            "model",
            True,
            1.0,
        )
    return track


class HitEvidenceValidationTests(unittest.TestCase):
    def test_top_edge_point_cannot_be_exported_as_hit(self):
        track = make_track()
        track[5] = BallPoint(5, 50.0, 4.0, True, "model_edge", True, 0.35)
        event = HitEvent(5, 50.0, 4.0, 0.95, 12.0, source="curvature")
        accepted, stats = validate_hit_event_evidence(
            track, [event], audio_frames=[5]
        )
        self.assertEqual(accepted, [])
        self.assertEqual(stats.get("edge_or_derived_event"), 1)

    def test_classical_gap_point_cannot_be_exported_as_hit(self):
        track = make_track()
        track[5] = BallPoint(5, 50.0, 50.0, True, "classical_gap", True, 0.4)
        event = HitEvent(5, 50.0, 50.0, 0.95, 12.0, source="curvature")
        accepted, stats = validate_hit_event_evidence(
            track, [event], audio_frames=[5]
        )
        self.assertEqual(accepted, [])
        self.assertEqual(stats.get("edge_or_derived_event"), 1)

    def test_pose_only_event_is_never_exported(self):
        event = HitEvent(5, 50.0, 50.0, 0.95, 0.0, source="audio_pose")
        accepted, stats = validate_hit_event_evidence(
            make_track(), [event], audio_frames=[5]
        )
        self.assertEqual(accepted, [])
        self.assertEqual(stats.get("pose_only"), 1)

    def test_player_body_overlap_is_a_hard_veto(self):
        event = HitEvent(5, 50.0, 50.0, 0.92, 12.0, source="curvature")
        accepted, stats = validate_hit_event_evidence(
            make_track(),
            [event],
            audio_frames=[5],
            player_boxes={5: [(40.0, 20.0, 70.0, 90.0)]},
        )
        self.assertEqual(accepted, [])
        self.assertEqual(stats.get("player_overlap"), 1)

    def test_player_box_overlap_near_racket_contact_is_allowed(self):
        event = HitEvent(5, 50.0, 50.0, 0.92, 12.0, source="curvature")
        accepted, stats = validate_hit_event_evidence(
            make_track(),
            [event],
            audio_frames=[5],
            player_boxes={5: [(40.0, 20.0, 70.0, 90.0)]},
            player_contacts={5: [(52.0, 52.0)]},
            contact_tolerance_px=45.0,
        )
        self.assertEqual([item.frame for item in accepted], [5])
        self.assertEqual(stats["accepted"], 1)

    def test_curvature_requires_audio_and_bidirectional_real_points(self):
        event = HitEvent(5, 50.0, 50.0, 0.92, 12.0, source="curvature")
        accepted, stats = validate_hit_event_evidence(
            make_track(real_frames=(2, 3, 4)), [event], audio_frames=[5]
        )
        self.assertEqual(accepted, [])
        self.assertEqual(stats.get("missing_bidirectional_trajectory"), 1)

        accepted, stats = validate_hit_event_evidence(
            make_track(), [event], audio_frames=[]
        )
        self.assertEqual(accepted, [])
        self.assertEqual(stats.get("missing_audio"), 1)

    def test_trajectory_plus_audio_is_accepted(self):
        event = HitEvent(5, 50.0, 50.0, 0.92, 12.0, source="curvature")
        accepted, stats = validate_hit_event_evidence(
            make_track(), [event], audio_frames=[7], audio_tolerance_frames=2
        )
        self.assertEqual([item.frame for item in accepted], [5])
        self.assertEqual(stats["accepted"], 1)
        self.assertFalse(accepted[0].ball_observed)

    def test_audio_endpoint_may_use_one_sided_measured_trajectory(self):
        event = HitEvent(7, 70.0, 50.0, 0.80, 10.0, source="classical_endpoint")
        accepted, stats = validate_hit_event_evidence(
            make_track(real_frames=(5, 6, 7)), [event], audio_frames=[7]
        )
        self.assertEqual([item.frame for item in accepted], [7])
        self.assertEqual(stats["accepted"], 1)
        self.assertTrue(accepted[0].ball_observed)

    def test_pose_or_audio_cannot_relocate_hit_away_from_ball(self):
        event = HitEvent(5, 400.0, 400.0, 0.90, 0.0, source="audio_ball")
        accepted, stats = validate_hit_event_evidence(
            make_track(), [event], audio_frames=[5], position_tolerance_px=50.0
        )
        self.assertEqual(accepted, [])
        self.assertEqual(stats.get("trajectory_position_mismatch"), 1)

    def test_strict_audio_contact_identity_propagates_without_shifting_visual_frame(self):
        visual = HitEvent(
            50, 500.0, 220.0, 0.76, 24.0,
            source="curvature", hitter="unknown",
        )
        audio = HitEvent(
            52, 560.0, 260.0, 0.94, 0.0,
            source="audio_ball", hitter="near", uncertain=True,
            hitter_confidence=0.82, hitter_source="pose_contact",
        )
        merged = merge_hit_events([visual], [audio], min_separation_frames=8)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].frame, 50)
        self.assertEqual((merged[0].x, merged[0].y), (500.0, 220.0))
        self.assertEqual(merged[0].source, "curvature")
        self.assertEqual(merged[0].hitter, "near")
        self.assertEqual(merged[0].hitter_source, "pose_contact")
        self.assertAlmostEqual(merged[0].hitter_confidence, 0.82)

    def test_strict_pose_association_labels_visual_candidates_before_merge(self):
        visual = HitEvent(
            50, 500.0, 220.0, 0.76, 24.0,
            source="curvature", hitter="unknown",
        )
        pose_evidence = {
            50: {
                "players": [
                    # The virtual racket tip is close to the measured ball.
                    {"side": "near", "contact": (510.0, 230.0), "torso": 100.0},
                    # Duplicate same-side detections must be collapsed before
                    # comparing near/far contact distances.
                    {"side": "near", "contact": (513.0, 223.0), "torso": 100.0},
                    # A visible opponent is sufficiently farther away to make
                    # this a strict, rather than nearest-body, association.
                    {"side": "far", "contact": (500.0, 20.0), "torso": 100.0},
                ],
            },
        }

        stats = apply_strict_pose_hitter_associations([visual], pose_evidence)

        self.assertEqual(visual.hitter, "near")
        self.assertEqual(visual.hitter_source, "pose_contact")
        self.assertGreaterEqual(visual.hitter_confidence, 0.75)
        self.assertEqual(stats["associated"], 1)
        self.assertEqual(stats["updated"], 1)

    def test_loose_audio_side_hint_does_not_propagate_to_visual_hit(self):
        visual = HitEvent(
            50, 500.0, 220.0, 0.76, 24.0,
            source="curvature", hitter="unknown",
        )
        # Legacy nearest-body hints carried neither a strict contact source
        # nor an attribution confidence.  They remain timing evidence only.
        audio = HitEvent(
            52, 560.0, 260.0, 0.94, 0.0,
            source="audio_ball", hitter="near", uncertain=True,
        )
        merged = merge_hit_events([visual], [audio], min_separation_frames=8)
        self.assertEqual(merged[0].hitter, "unknown")
        self.assertEqual(merged[0].hitter_source, "unknown")

    def test_ambiguous_nearby_visual_turns_do_not_receive_audio_identity(self):
        first = HitEvent(50, 500.0, 220.0, 0.76, 24.0, source="curvature")
        second = HitEvent(53, 540.0, 230.0, 0.75, 22.0, source="curvature")
        audio = HitEvent(
            52, 520.0, 225.0, 0.94, 0.0,
            source="audio_ball", hitter="near", uncertain=True,
            hitter_confidence=0.84, hitter_source="pose_contact",
        )
        # Keep both visual turns through NMS while retaining the three-frame
        # identity-association window; the audio event is deliberately
        # ambiguous and must not choose either one.
        merged = merge_hit_events([first, second], [audio], min_separation_frames=3)
        self.assertEqual([event.hitter for event in merged], ["unknown", "unknown"])

    def test_sidecar_exports_hitter_confidence_and_source(self):
        track = [BallPoint(0, 10.0, 20.0, True, "model", True, 1.0)]
        event = HitEvent(
            0, 10.0, 20.0, 0.91, 18.0,
            hitter="near", hitter_confidence=0.83,
            hitter_source="pose_contact",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/sidecar.csv"
            write_ball_tracking_csv(
                path,
                {0: (1, 10, 20, "model", 1.0)},
                track,
                [event],
                [1],
            )
            with open(path, encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
        self.assertEqual(row["HitHitter"], "near")
        self.assertEqual(row["HitHitterConfidence"], "0.8300")
        self.assertEqual(row["HitHitterSource"], "pose_contact")

    def test_rally_alternation_fills_only_single_unknown_between_pose_anchors(self):
        events = [
            HitEvent(
                12, 100.0, 200.0, 0.9, 20.0,
                hitter="near", rally=3,
                hitter_confidence=0.75, hitter_source="pose_contact",
            ),
            HitEvent(28, 120.0, 190.0, 0.88, 22.0, rally=3),
            HitEvent(
                44, 140.0, 180.0, 0.91, 24.0,
                hitter="near", rally=3,
                hitter_confidence=0.87, hitter_source="pose_contact",
            ),
        ]
        original_center = (
            events[1].frame, events[1].x, events[1].y, events[1].score,
            events[1].source, events[1].uncertain, events[1].rally,
        )

        stats = infer_single_unknown_hitter_from_rally_anchors(events)

        self.assertEqual(events[1].hitter, "far")
        self.assertAlmostEqual(events[1].hitter_confidence, 0.55)
        self.assertEqual(events[1].hitter_source, "rally_alternation_inferred")
        self.assertEqual(
            (
                events[1].frame, events[1].x, events[1].y, events[1].score,
                events[1].source, events[1].uncertain, events[1].rally,
            ),
            original_center,
        )
        self.assertEqual(stats["filled"], 1)

    def test_rally_alternation_never_uses_itself_to_balance_counts(self):
        events = [
            HitEvent(
                12, 100.0, 200.0, 0.9, 20.0,
                hitter="near", rally=3,
                hitter_confidence=0.91, hitter_source="pose_contact",
            ),
            HitEvent(28, 120.0, 190.0, 0.88, 22.0, rally=3),
            # Opposite anchor sides cannot uniquely explain one missing hit,
            # so the unknown remains available for human video review.
            HitEvent(
                44, 140.0, 180.0, 0.91, 24.0,
                hitter="far", rally=3,
                hitter_confidence=0.89, hitter_source="pose_contact",
            ),
            # A different rally must not supply an endpoint to make hit
            # totals look balanced.
            HitEvent(
                60, 150.0, 175.0, 0.90, 23.0,
                hitter="near", rally=4,
                hitter_confidence=0.92, hitter_source="pose_contact",
            ),
        ]

        stats = infer_single_unknown_hitter_from_rally_anchors(events)

        self.assertEqual(events[1].hitter, "unknown")
        self.assertEqual(events[1].hitter_source, "unknown")
        self.assertEqual(stats["filled"], 0)
        self.assertEqual(stats["skipped_opposite_anchors"], 1)

    def test_rally_alternation_requires_strong_pose_contact_anchors(self):
        events = [
            HitEvent(
                12, 100.0, 200.0, 0.9, 20.0,
                hitter="near", rally=3,
                hitter_confidence=0.99, hitter_source="sidecar",
            ),
            HitEvent(28, 120.0, 190.0, 0.88, 22.0, rally=3),
            HitEvent(
                44, 140.0, 180.0, 0.91, 24.0,
                hitter="near", rally=3,
                hitter_confidence=0.74, hitter_source="pose_contact",
            ),
        ]

        stats = infer_single_unknown_hitter_from_rally_anchors(events)

        self.assertEqual(events[1].hitter, "unknown")
        self.assertEqual(stats["filled"], 0)

    def test_rally_alternation_skips_close_or_endpoint_candidates(self):
        def pose_anchor(frame):
            return HitEvent(
                frame, 100.0, 200.0, 0.9, 20.0,
                hitter="near", rally=3,
                hitter_confidence=0.82, hitter_source="pose_contact",
            )

        close_events = [pose_anchor(12), HitEvent(20, 120.0, 190.0, 0.88, 22.0, rally=3), pose_anchor(36)]
        close_stats = infer_single_unknown_hitter_from_rally_anchors(close_events)
        self.assertEqual(close_events[1].hitter, "unknown")
        self.assertEqual(close_stats["skipped_close_events"], 1)

        endpoint_events = [
            pose_anchor(12),
            HitEvent(28, 120.0, 190.0, 0.88, 22.0, rally=3, source="edge_exit"),
            pose_anchor(44),
        ]
        endpoint_stats = infer_single_unknown_hitter_from_rally_anchors(endpoint_events)
        self.assertEqual(endpoint_events[1].hitter, "unknown")
        self.assertEqual(endpoint_stats["skipped_collision_sources"], 1)

    def test_audio_without_visual_candidate_is_not_exported(self):
        audio = HitEvent(
            55, 560.0, 260.0, 0.94, 0.0,
            source="audio_ball", uncertain=True,
        )
        self.assertEqual(merge_hit_events([], [audio]), [])

    def test_primary_and_fused_visual_candidates_are_unioned(self):
        """A valid fused-only turn must not disappear behind a nonempty primary list."""
        primary = [HitEvent(273, 1249.0, 581.0, 0.95, 24.0, source="curvature")]
        fused = [HitEvent(319, 718.0, 567.0, 0.70, 43.0, source="curvature")]
        merged = merge_visual_hit_candidates(primary, fused, min_separation_frames=8)
        self.assertEqual([event.frame for event in merged], [273, 319])

    def test_raw_model_turn_with_audio_is_confirmed_below_legacy_score_threshold(self):
        point = BallPoint(655, 1188.0, 220.0, True, "model", True, 1.0)
        event = HitEvent(
            655, 1188.0, 220.0, 0.70, 42.0,
            source="curvature", ball_observed=True,
        )
        self.assertTrue(has_confirmed_model_audio_evidence(
            event, point, audio_frames=[657], audio_tolerance_frames=6,
        ))

    def test_interpolated_or_audio_only_turn_is_not_promoted_to_confirmed(self):
        point = BallPoint(655, 1188.0, 220.0, True, "interp", False, 1.0)
        event = HitEvent(
            655, 1188.0, 220.0, 0.70, 42.0,
            source="audio_ball", ball_observed=False,
        )
        self.assertFalse(has_confirmed_model_audio_evidence(
            event, point, audio_frames=[655], audio_tolerance_frames=6,
        ))

    def test_top_edge_exit_recovers_last_measured_contact_not_edge_point(self):
        """A far-player clear can leave the frame before curvature has rhs support."""
        track = [
            BallPoint(0, 100.0, 170.0, True, "model", True, 1.0),
            BallPoint(1, 103.0, 184.0, True, "model", True, 1.0),
            BallPoint(2, 106.0, 201.0, True, "model", True, 1.0),
            BallPoint(3, 98.0, 160.0, True, "model_edge", True, 0.35),
            BallPoint(4, 88.0, 82.0, True, "model_edge", True, 0.35),
        ]
        events = detect_top_edge_exit_hits(track)
        self.assertEqual([(event.frame, event.source) for event in events], [
            (2, "edge_exit"),
        ])
        accepted, stats = validate_hit_event_evidence(
            track, events, audio_frames=[5], audio_tolerance_frames=3,
        )
        self.assertEqual([event.frame for event in accepted], [2])
        self.assertEqual(stats["accepted"], 1)

    def test_edge_exit_beats_primary_curvature_at_same_contact(self):
        """Keep the valid last measured point when primary curvature lands on an edge."""
        primary = [HitEvent(44, 1077.0, 225.0, 0.998, 31.0, source="curvature")]
        edge_exit = [HitEvent(44, 1077.0, 225.0, 0.960, 42.0, source="edge_exit")]
        merged = merge_visual_hit_candidates(primary, edge_exit)
        self.assertEqual([(event.frame, event.source) for event in merged], [
            (44, "edge_exit"),
        ])

    def test_slow_defensive_turn_survives_adaptive_speed_gate(self):
        """A lift may nearly stop at contact while the rest of the rally is fast."""
        coordinates = [
            (0.0, 0.0), (0.0, 13.0), (0.0, 26.0), (0.0, 39.0),
            (0.0, 52.0), (0.0, 65.0), (0.0, 78.0), (0.0, 91.0),
            (3.0, 91.0), (11.0, 77.0), (19.0, 63.0), (27.0, 49.0),
        ]
        track = [
            BallPoint(frame, x, y, True, "model", True, 1.0)
            for frame, (x, y) in enumerate(coordinates)
        ]
        events = detect_hit_points(track, fps=25.0, min_speed_px=7.0)
        self.assertIn(8, [event.frame for event in events])


if __name__ == "__main__":
    unittest.main()
