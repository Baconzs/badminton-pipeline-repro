#!/usr/bin/env python3
"""Focused regression tests for exported hit-event evidence gates."""

import unittest

from ball_tracking import (
    BallPoint,
    HitEvent,
    detect_hit_points,
    detect_top_edge_exit_hits,
    validate_hit_event_evidence,
)
from overlay_player_analytics import (
    has_confirmed_model_audio_evidence,
    merge_hit_events,
    merge_visual_hit_candidates,
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

    def test_audio_confirmation_does_not_shift_visual_hit_frame(self):
        visual = HitEvent(
            50, 500.0, 220.0, 0.76, 24.0,
            source="curvature", hitter="unknown",
        )
        audio = HitEvent(
            55, 560.0, 260.0, 0.94, 0.0,
            source="audio_ball", hitter="near", uncertain=True,
        )
        merged = merge_hit_events([visual], [audio], min_separation_frames=8)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].frame, 50)
        self.assertEqual((merged[0].x, merged[0].y), (500.0, 220.0))
        self.assertEqual(merged[0].source, "curvature")
        self.assertEqual(merged[0].hitter, "near")

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
