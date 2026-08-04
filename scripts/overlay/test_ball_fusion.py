import unittest
from types import SimpleNamespace

import numpy as np

from ball_tracking import (
    BallPoint,
    DERIVED_MEASUREMENT_SOURCES,
    add_offscreen_bridges,
    coverage,
    detect_hit_points,
    filter_ball_track_geometry,
    segment_rallies,
    smooth_ball_track,
)
from overlay_player_analytics import (
    fill_tracknet_gaps_with_classical,
    fuse_ball_observations,
    longest_observation_gap,
    merge_tracknet_phase_observations,
    find_dual_phase_coherent_motion_frames,
    find_edge_reentry_frames,
    find_top_exit_branch_frames,
    find_top_entry_lead_frames,
    find_top_edge_interstitial_branch_frames,
    veto_top_edge_interstitial_branches,
    build_arg_parser,
    veto_isolated_phase_observations,
    veto_model_observations_in_player_boxes,
    veto_short_unsupported_phase_branches,
)


def classical_result(points):
    observations = {
        int(frame): SimpleNamespace(
            x=float(x),
            y=float(y),
            confidence=0.9,
            track_id=int(track_id),
        )
        for frame, x, y, track_id in points
    }
    return SimpleNamespace(observations=observations)


class BallFusionTests(unittest.TestCase):
    def test_gap_only_classical_fill_requires_two_model_anchors(self):
        # This mirrors the safe f317--318 recovery: the model is vetoed in a
        # short hole, while a multi-point classical track follows the same
        # flight.  Existing model anchors remain untouched.
        model = {
            0: (1, 740, 490, "model", 1.0),
            1: (1, 720, 515, "model", 1.0),
            2: (0, 705, 538, "phase_veto", 0.0),
            3: (0, 701, 563, "phase_veto", 0.0),
            4: (0, 0, 0, "phase_veto", 0.0),
            5: (1, 780, 515, "model", 1.0),
        }
        classical = classical_result([
            (0, 740, 490, 4), (1, 720, 515, 4),
            (2, 705, 538, 4), (3, 701, 563, 4),
            (4, 690, 545, 4), (5, 780, 515, 4),
        ])
        filled, ids, stats = fill_tracknet_gaps_with_classical(
            model, classical, frame_count=6, support_window=2,
            bridge_residual_px=72.0,
        )
        self.assertEqual(filled[2][3], "classical_gap")
        self.assertEqual(filled[3][3], "classical_gap")
        self.assertEqual(filled[0][3], "model")
        self.assertEqual(ids, {2: 4, 3: 4})
        self.assertEqual(stats["filled"], 2)

    def test_gap_only_fill_rejects_long_or_unanchored_holes(self):
        model = {
            0: (1, 100, 100, "model", 1.0),
            1: (0, 0, 0, "phase_veto", 0.0),
            2: (0, 0, 0, "phase_veto", 0.0),
            3: (0, 0, 0, "phase_veto", 0.0),
            4: (0, 0, 0, "phase_veto", 0.0),
            5: (0, 0, 0, "phase_veto", 0.0),
            6: (1, 200, 100, "model", 1.0),
        }
        classical = classical_result([
            (1, 110, 100, 7), (2, 120, 100, 7),
            (3, 130, 100, 7), (4, 140, 100, 7),
        ])
        filled, _, stats = fill_tracknet_gaps_with_classical(
            model, classical, frame_count=7, support_window=2,
            max_gap_frames=3,
        )
        self.assertTrue(all(filled[frame][0] == 0 for frame in range(1, 5)))
        self.assertGreater(stats["rejected_long_gap"], 0)

    def test_classical_gap_is_not_coverage_or_hit_evidence(self):
        track = [
            BallPoint(0, 100, 100, True, "model", True, 1.0),
            BallPoint(1, 110, 100, True, "classical_gap", True, 0.4),
            BallPoint(2, 120, 100, True, "model", True, 1.0),
        ]
        total, real, derived = coverage(track)
        self.assertEqual((total, real, derived), (2 / 3, 2 / 3, 0.0))
        events = detect_hit_points(track, fps=25.0, min_speed_px=1.0)
        self.assertEqual(events, [])
        self.assertIn("classical_gap", DERIVED_MEASUREMENT_SOURCES)

    def test_isolated_single_phase_body_jump_is_vetoed(self):
        primary = {
            558: (1, 1275, 33),
            559: (1, 1282, 11),
            560: (1, 1286, 3),
            563: (0, 0, 0),
            574: (0, 0, 0),
        }
        secondary = dict(primary)
        secondary[558] = (1, 1275, 33)
        secondary[559] = (1, 1282, 11)
        secondary[560] = (1, 1286, 7)
        secondary[563] = (1, 1031, 427)
        output = {frame: (value[0], value[1], value[2], "model", 1.0)
                  for frame, value in secondary.items()}
        filtered, vetoed, stats = veto_isolated_phase_observations(
            output,
            primary,
            secondary,
            pose_evidence={},
            candidate_frames={563},
            support_window=5,
        )
        self.assertEqual(vetoed, [563])
        self.assertEqual(filtered[563][0], 0)
        self.assertEqual(stats["vetoed"], 1)

    def test_isolated_phase_point_with_coherent_bridge_survives(self):
        primary = {frame: (1, 100 + 10 * frame, 100) for frame in (0, 1, 2, 3, 4)}
        secondary = dict(primary)
        primary[2] = (0, 0, 0)
        secondary[2] = (1, 120, 100)
        output = {
            frame: (value[0], value[1], value[2], "model", 1.0)
            for frame, value in secondary.items()
        }
        filtered, vetoed, stats = veto_isolated_phase_observations(
            output,
            primary,
            secondary,
            pose_evidence={},
            candidate_frames={2},
            support_window=3,
        )
        self.assertEqual(vetoed, [])
        self.assertEqual(filtered[2][0], 1)
        self.assertEqual(stats["bridge_supported"], 1)

    def test_sparse_one_sided_support_uses_frame_ids_not_mapping_length(self):
        """A sparse diagnostic mapping must still retain a coherent point.

        The production phase merge returns a dense mapping, while callers
        inspecting a short segment often pass only the visible rows.  The
        right-side support at frames 3--5 should therefore be considered even
        though ``len(output)`` is only four and the largest key is 5.
        """
        primary = {
            2: (0, 0, 0),
            3: (1, 130, 100),
            4: (1, 140, 100),
            5: (1, 150, 100),
        }
        secondary = {
            2: (1, 120, 100),
            3: (1, 130, 100),
            4: (1, 140, 100),
            5: (1, 150, 100),
        }
        output = {
            frame: (value[0], value[1], value[2], "model", 1.0)
            for frame, value in secondary.items()
        }
        filtered, vetoed, stats = veto_isolated_phase_observations(
            output,
            primary,
            secondary,
            pose_evidence={},
            candidate_frames={2},
            support_window=3,
        )
        self.assertEqual(vetoed, [])
        self.assertEqual(filtered[2][0], 1)
        self.assertEqual(stats["one_sided_supported"], 1)

    def test_fill_and_near_agreement_preserve_real_source(self):
        model = {
            0: (1, 100, 100),
            1: (0, 0, 0),
            2: (1, 120, 120),
        }
        classical = classical_result([
            (0, 103, 98, 7),
            (1, 110, 110, 7),
        ])
        fused, track_ids, stats = fuse_ball_observations(
            model, classical, frame_count=3
        )
        self.assertEqual(fused[0][1:4], (100, 100, "model"))
        self.assertEqual(fused[1][1:4], (110, 110, "classical"))
        self.assertEqual(track_ids, {1: 7})
        self.assertEqual(stats["classical_fill"], 1)

    def test_classical_fusion_does_not_drop_edge_provenance(self):
        model = {0: (1, 100, 4, "model_edge", 0.35)}
        fused, _, _ = fuse_ball_observations(
            model, classical_result([]), frame_count=1
        )
        self.assertEqual(fused[0][3], "model_edge")

    def test_bridge_replaces_clear_model_excursion(self):
        model = {
            0: (1, 100, 100),
            1: (1, 300, 300),
            2: (1, 120, 100),
        }
        classical = classical_result([(1, 110, 100, 4)])
        fused, track_ids, stats = fuse_ball_observations(
            model, classical, frame_count=3
        )
        self.assertEqual(fused[1][1:4], (110, 100, "classical"))
        self.assertEqual(track_ids, {1: 4})
        self.assertEqual(stats["conflict_use_classical"], 1)

    def test_bridge_keeps_model_at_real_fast_contact(self):
        # The model point is closer to the two-sided flight bridge; a
        # conflicting classical body response must not win by fixed priority.
        model = {
            0: (1, 825, 435),
            1: (1, 903, 435),
            2: (1, 1020, 412),
        }
        classical = classical_result([(1, 938, 395, 2)])
        fused, track_ids, stats = fuse_ball_observations(
            model, classical, frame_count=3
        )
        self.assertEqual(fused[1][1:4], (903, 435, "model"))
        self.assertEqual(track_ids, {})
        self.assertEqual(stats["conflict_keep_model"], 1)

    def test_conflict_bridge_never_crosses_scene_cut(self):
        model = {
            0: (1, 100, 100),
            1: (1, 400, 400),
            2: (1, 120, 100),
        }
        classical = classical_result([(1, 110, 100, 9)])
        fused, track_ids, _ = fuse_ball_observations(
            model, classical, frame_count=3, scene_cuts=[2]
        )
        self.assertEqual(fused[1][1:4], (400, 400, "model"))
        self.assertEqual(track_ids, {})

    def test_long_local_gap_is_detected_even_with_high_global_coverage(self):
        model = {frame: (1, frame + 1, 100) for frame in range(100)}
        for frame in range(40, 53):
            model[frame] = (0, 0, 0)
        self.assertEqual(longest_observation_gap(model, 100), 13)

    def test_kalman_reacquires_after_prediction_window(self):
        model = {
            0: (1, 100, 100),
            1: (1, 110, 100),
            10: (1, 700, 300),
        }
        track = smooth_ball_track(
            model,
            frame_count=12,
            fps=25,
            max_interp_gap=2,
            max_predict_gap=3,
        )
        self.assertTrue(track[10].visible)
        self.assertEqual(track[10].source, "model")
        self.assertAlmostEqual(track[10].x, 700.0, delta=2.0)

    def test_reset_frame_blocks_interpolation_across_phase_veto(self):
        model = {
            0: (1, 100, 100),
            1: (1, 110, 90),
            5: (1, 500, 20),
            6: (1, 510, 10),
        }
        track = smooth_ball_track(
            model,
            frame_count=8,
            fps=25,
            max_interp_gap=8,
            max_predict_gap=0,
            reset_frames=[2],
        )
        self.assertFalse(track[2].visible)
        self.assertFalse(track[3].visible)
        self.assertFalse(track[4].visible)
        self.assertEqual(track[5].source, "model")

    def test_static_gate_keeps_slow_high_clear(self):
        model = {
            frame: (1, 500 + 10 * frame, 120, "model", 1.0)
            for frame in range(30)
        }
        track = smooth_ball_track(model, 30, 25, max_predict_gap=0)
        filtered = filter_ball_track_geometry(
            track,
            court_polygon=None,
            static_window=20,
            static_disp=8.0,
        )
        self.assertEqual(sum(point.visible for point in filtered), 30)

    def test_static_gate_removes_tight_raw_model_lock(self):
        model = {
            frame: (1, 500 + (frame % 2), 120, "model", 1.0)
            for frame in range(24)
        }
        track = smooth_ball_track(model, 24, 25, max_predict_gap=0)
        filtered = filter_ball_track_geometry(
            track,
            court_polygon=None,
            static_window=20,
            static_disp=8.0,
        )
        self.assertLess(sum(point.visible for point in filtered), 5)

    def test_phase_bridge_selects_secondary_through_conflict_group(self):
        primary = {
            0: (1, 100, 100),
            1: (1, 450, 400),
            2: (1, 460, 410),
            3: (1, 130, 100),
        }
        secondary = {
            0: (1, 102, 100),
            1: (1, 110, 100),
            2: (1, 120, 100),
            3: (1, 132, 100),
        }
        merged, suspicious, stats = merge_tracknet_phase_observations(
            primary, secondary, frame_count=4
        )
        self.assertEqual(merged[1][1:3], (110, 100))
        self.assertEqual(merged[2][1:3], (120, 100))
        self.assertTrue({1, 2}.issubset(suspicious))
        self.assertEqual(stats["bridge_secondary"], 2)

    def test_phase_conflict_without_two_sided_anchors_stays_missing(self):
        primary = {frame: (1, 100 + frame, 100) for frame in range(5)}
        secondary = {frame: (1, 500 + frame, 400) for frame in range(5)}
        merged, _, stats = merge_tracknet_phase_observations(
            primary, secondary, frame_count=5
        )
        self.assertFalse(any(int(merged[frame][0]) for frame in range(5)))
        self.assertEqual(stats["ambiguous_missing"], 5)

    def test_long_phase_conflict_uses_group_anchors_not_primary_fallback(self):
        primary = {0: (1, 100, 100), 13: (1, 230, 100)}
        secondary = {0: (1, 100, 100), 13: (1, 230, 100)}
        for frame in range(1, 13):
            primary[frame] = (1, 700, 400)
            secondary[frame] = (1, 100 + 10 * frame, 100)
        merged, _, stats = merge_tracknet_phase_observations(
            primary,
            secondary,
            frame_count=14,
            bridge_search_frames=6,
        )
        self.assertTrue(all(merged[frame][1] == 100 + 10 * frame for frame in range(1, 13)))
        self.assertEqual(stats["bridge_secondary"], 12)

    def test_phase_bridge_does_not_cross_scene_cut(self):
        primary = {
            0: (1, 100, 100),
            1: (1, 400, 400),
            2: (1, 120, 100),
        }
        secondary = {
            0: (1, 102, 100),
            1: (1, 110, 100),
            2: (1, 122, 100),
        }
        merged, _, _ = merge_tracknet_phase_observations(
            primary, secondary, frame_count=3, scene_cuts=[2]
        )
        self.assertEqual(merged[1], (0, 0, 0))

    def test_dual_agree_top_edge_is_low_confidence_render_support(self):
        # The final three points are a smooth ascent to y≈0.  They should
        # bypass the short phase/body veto, but remain excluded from hit
        # evidence via their explicit source label.
        primary = {
            0: (1, 100, 220),
            1: (1, 108, 170),
            2: (1, 116, 92),
            3: (1, 124, 14),
        }
        secondary = {
            0: (1, 101, 220),
            1: (1, 108, 169),
            2: (1, 116, 91),
            3: (1, 124, 13),
        }
        merged, _, stats = merge_tracknet_phase_observations(
            primary, secondary, frame_count=4, edge_y_px=35.0
        )
        self.assertEqual(merged[1][3], "model_edge")
        self.assertEqual(merged[2][3], "model_edge")
        self.assertEqual(merged[3][3], "model_edge")
        self.assertEqual(stats["dual_edge"], 3)

    def test_single_phase_edge_stub_is_limited_and_labelled(self):
        primary = {
            0: (0, 0, 0),
            1: (0, 0, 0),
            2: (0, 0, 0),
            3: (1, 720, 26),
            4: (1, 720, 60),
        }
        secondary = {
            0: (0, 0, 0),
            1: (1, 720, 3),
            2: (0, 0, 0),
            3: (1, 720, 26),
            4: (1, 720, 60),
        }
        merged, _, stats = merge_tracknet_phase_observations(
            primary, secondary, frame_count=5, edge_y_px=35.0
        )
        self.assertEqual(merged[1][3], "single_phase_edge")
        self.assertEqual(stats["single_edge"], 1)

    def test_top_entry_lead_protects_only_dual_phase_ascent(self):
        # The edge anchor and its two lead-in points mirror f445--447 in the
        # reference video.  A pose box deliberately covers all three points;
        # the protection set must keep them available for rendering while the
        # edge source remains outside ordinary hit evidence.
        observations = {
            0: (1, 1090, 250, "model", 1.0),
            1: (1, 1065, 176, "model_edge", 0.35),
            2: (1, 1046, 78, "model_edge", 0.35),
            3: (1, 1027, 15, "model_edge", 0.35),
        }
        lead = find_top_entry_lead_frames(
            observations,
            frame_count=4,
            top_y=30,
            max_lead_frames=8,
        )
        self.assertEqual(lead, [1, 2, 3])
        evidence = {
            frame: {"players": [{"box": (980, 0, 1120, 260)}]}
            for frame in lead
        }
        filtered, vetoed = veto_model_observations_in_player_boxes(
            observations,
            evidence,
            frames=lead,
            box_margin_px=0,
            protected_frames=lead,
        )
        self.assertEqual(vetoed, [])
        self.assertTrue(all(filtered[frame][0] == 1 for frame in lead))

        # A single-phase edge response and a non-monotone branch are not
        # protected, even if they happen to touch the same top zone.
        single = dict(observations)
        single[2] = (1, 1046, 78, "single_phase_edge", 0.22)
        self.assertEqual(
            find_top_entry_lead_frames(single, 4, top_y=30),
            [],
        )
        wrong_way = dict(observations)
        wrong_way[2] = (1, 1046, 220, "model_edge", 0.35)
        self.assertEqual(
            find_top_entry_lead_frames(wrong_way, 4, top_y=30),
            [],
        )

    def test_top_entry_lead_never_crosses_scene_cut(self):
        observations = {
            0: (1, 1000, 200, "model_edge", 0.35),
            1: (1, 990, 120, "model_edge", 0.35),
            2: (1, 980, 20, "model_edge", 0.35),
        }
        self.assertEqual(
            find_top_entry_lead_frames(
                observations,
                frame_count=3,
                scene_cuts=[1],
                top_y=30,
            ),
            [],
        )

    def test_edge_top_exit_branch_requires_body_jump(self):
        smooth = {
            0: (1, 1000, 120, "model", 1.0),
            1: (1, 990, 20, "model_edge", 0.35),
            2: (1, 985, 45, "model", 1.0),
            3: (1, 980, 75, "model", 1.0),
            4: (1, 975, 110, "model", 1.0),
        }
        self.assertEqual(
            find_top_exit_branch_frames(smooth, 5, top_y=30, body_y=300),
            [],
        )
        false_branch = dict(smooth)
        false_branch[2] = (1, 985, 450, "model", 1.0)
        false_branch[3] = (1, 980, 452, "model", 1.0)
        self.assertEqual(
            find_top_exit_branch_frames(false_branch, 5, top_y=30, body_y=300),
            [2, 3],
        )

    def test_top_interstitial_branch_vetoes_high_residual_middle_points(self):
        """Only a body/background branch between two edge flights is vetoed."""
        observations = {
            # Monotone approach to the top edge.
            0: (1, 900, 100, "model", 1.0),
            1: (1, 880, 60, "model", 1.0),
            2: (1, 860, 20, "model", 1.0),
            # The true straight bridge would stay near x≈820.  These visible
            # points instead jump onto a distant body/scoreboard response.
            5: (1, 500, 340, "model", 1.0),
            6: (1, 470, 300, "model", 1.0),
            # Monotone re-entry from the top edge.
            10: (1, 780, 20, "model", 1.0),
            11: (1, 760, 60, "model", 1.0),
            12: (1, 740, 100, "model", 1.0),
        }
        candidates, bridges, stats = find_top_edge_interstitial_branch_frames(
            observations,
            frame_count=13,
            edge_top_y=110,
            edge_core_y=35,
            min_anchor_frames=3,
            max_interstitial_gap=20,
            min_branch_residual_px=110,
        )
        self.assertEqual(candidates, [5, 6])
        self.assertEqual(stats["candidate_frames"], 2)
        self.assertIn(5, bridges)
        filtered, vetoed, veto_stats = veto_top_edge_interstitial_branches(
            observations,
            bridges,
            pose_evidence={},
        )
        self.assertEqual(vetoed, [5, 6])
        self.assertEqual(veto_stats["vetoed"], 2)
        self.assertEqual(filtered[5][0], 0)
        self.assertEqual(filtered[5][3], "phase_top_branch_veto")
        self.assertEqual(filtered[6][3], "phase_top_branch_veto")
        # The edge anchors themselves are retained verbatim.
        self.assertEqual(filtered[2], observations[2])
        self.assertEqual(filtered[10], observations[10])

    def test_top_interstitial_default_broad_envelope_catches_f448_style_branch(self):
        """The 180 px envelope admits only the lead-in, not the core gate."""
        observations = {
            # f445--447-like ascent: the first point is above the old 110 px
            # envelope but the final point still reaches the strict 35 px
            # core.  The right side mirrors the f478 re-entry.
            0: (1, 1065, 165, "model_edge", 0.35),
            1: (1, 1046, 78, "model_edge", 0.35),
            2: (1, 1027, 15, "model_edge", 0.35),
            # f448--455-like body lock, deliberately far from the edge bridge.
            5: (1, 986, 446, "model", 1.0),
            6: (1, 990, 442, "model", 1.0),
            7: (1, 990, 442, "model", 1.0),
            8: (1, 986, 446, "model", 1.0),
            9: (1, 990, 450, "model", 1.0),
            10: (1, 986, 453, "model", 1.0),
            11: (1, 990, 457, "model", 1.0),
            14: (1, 787, 11, "model_edge", 0.35),
            15: (1, 783, 41, "model", 1.0),
            16: (1, 780, 78, "model", 1.0),
        }
        broad, _, broad_stats = find_top_edge_interstitial_branch_frames(
            observations,
            frame_count=17,
            edge_core_y=35,
            min_anchor_frames=3,
            max_interstitial_gap=20,
            min_branch_residual_px=110,
        )
        narrow, _, narrow_stats = find_top_edge_interstitial_branch_frames(
            observations,
            frame_count=17,
            edge_top_y=110,
            edge_core_y=35,
            min_anchor_frames=3,
            max_interstitial_gap=20,
            min_branch_residual_px=110,
        )
        self.assertEqual(broad, list(range(5, 12)))
        self.assertEqual(narrow, [])
        self.assertEqual(broad_stats["candidate_frames"], 7)
        self.assertEqual(narrow_stats["candidate_frames"], 0)
        # The broad envelope is a discovery aid only; the strict core remains
        # 35 px and the parser default is kept in sync with the helper.
        self.assertEqual(build_arg_parser().parse_args([]).tracknet_phase_top_interstitial_edge_y, 180.0)
        self.assertEqual(build_arg_parser().parse_args([]).tracknet_phase_edge_protect_y, 30.0)

    def test_top_interstitial_low_residual_flight_is_not_vetoed(self):
        """A smooth ordinary trajectory has no high-residual candidate."""
        observations = {
            0: (1, 900, 100, "model", 1.0),
            1: (1, 860, 60, "model", 1.0),
            2: (1, 820, 20, "model", 1.0),
            # Points lie on the line from (820,20) to (620,20).
            5: (1, 760, 20, "model", 1.0),
            6: (1, 720, 20, "model", 1.0),
            10: (1, 620, 20, "model", 1.0),
            11: (1, 580, 60, "model", 1.0),
            12: (1, 540, 100, "model", 1.0),
        }
        candidates, bridges, stats = find_top_edge_interstitial_branch_frames(
            observations,
            frame_count=13,
            edge_top_y=110,
            edge_core_y=35,
            min_anchor_frames=3,
            max_interstitial_gap=20,
            min_branch_residual_px=110,
        )
        self.assertEqual(candidates, [])
        self.assertEqual(bridges, {})
        self.assertEqual(stats["candidate_frames"], 0)
        filtered, vetoed, _ = veto_top_edge_interstitial_branches(
            observations, bridges, pose_evidence={}
        )
        self.assertEqual(vetoed, [])
        self.assertEqual(filtered, observations)

    def test_top_interstitial_veto_protects_edge_contact_and_explicit_frames(self):
        """Protection gates keep valid edge/contact responses untouched."""
        observations = {
            5: (1, 500, 340, "model_edge", 0.35),
            6: (1, 470, 300, "model", 1.0),
            7: (1, 440, 260, "model", 1.0),
        }
        bridges = {
            frame: ((2, 860.0, 20.0), (10, 780.0, 20.0), 500.0)
            for frame in observations
        }
        pose = {
            6: {
                "players": [{
                    "wrists": [(470, 300)],
                    "contact": (470, 300),
                }]
            }
        }
        filtered, vetoed, stats = veto_top_edge_interstitial_branches(
            observations,
            bridges,
            pose_evidence=pose,
            protected_frames={7},
            contact_protect_px=38,
        )
        self.assertEqual(vetoed, [])
        self.assertEqual(filtered, observations)
        self.assertEqual(stats["edge_protected"], 2)
        self.assertEqual(stats["contact_protected"], 1)

    def test_top_interstitial_veto_does_not_cross_scene_cut(self):
        """Top-edge anchors on different shots cannot create a bridge."""
        observations = {
            0: (1, 900, 100, "model", 1.0),
            1: (1, 880, 60, "model", 1.0),
            2: (1, 860, 20, "model", 1.0),
            5: (1, 500, 340, "model", 1.0),
            6: (1, 470, 300, "model", 1.0),
            10: (1, 780, 20, "model", 1.0),
            11: (1, 760, 60, "model", 1.0),
            12: (1, 740, 100, "model", 1.0),
        }
        candidates, _, stats = find_top_edge_interstitial_branch_frames(
            observations,
            frame_count=13,
            scene_cuts=[6],
            edge_top_y=110,
            edge_core_y=35,
            min_anchor_frames=3,
            max_interstitial_gap=20,
            min_branch_residual_px=110,
        )
        self.assertEqual(candidates, [])
        self.assertEqual(stats["bridges"], 0)

    def test_f478_like_single_top_reentry_is_not_interstitial_branch(self):
        """A normal top-edge re-entry (f478-style) must remain untouched."""
        observations = {
            # There is only a top-edge re-entry and no preceding monotone
            # approach run in this local window.
            0: (1, 787, 11, "model_edge", 0.35),
            1: (1, 783, 41, "model", 1.0),
            2: (1, 780, 78, "model", 1.0),
            3: (1, 776, 116, "model", 1.0),
            4: (1, 772, 161, "model", 1.0),
            5: (1, 768, 202, "model", 1.0),
            6: (1, 765, 243, "model", 1.0),
        }
        candidates, bridges, stats = find_top_edge_interstitial_branch_frames(
            observations,
            frame_count=7,
            edge_top_y=110,
            edge_core_y=35,
            min_anchor_frames=3,
            max_interstitial_gap=20,
            min_branch_residual_px=110,
        )
        self.assertEqual(candidates, [])
        self.assertEqual(bridges, {})
        self.assertEqual(stats["candidate_frames"], 0)

    def test_f445_to_f478_interstitial_branch_uses_broad_edge_lead(self):
        """The f445--447 lead must catch the body lock before f478 re-entry."""
        observations = {
            # The first lead point is above the old 110 px envelope, but it is
            # explicitly edge-labelled and part of a clear monotone ascent.
            0: (1, 1065, 176, "model_edge", 0.35),
            1: (1, 1046, 78, "model_edge", 0.35),
            2: (1, 1027, 15, "model_edge", 0.35),
            # Shared model/body lock in the unseen interval.
            3: (1, 990, 442, "model", 1.0),
            4: (1, 990, 442, "model", 1.0),
            5: (1, 986, 446, "model", 1.0),
            6: (1, 990, 450, "model", 1.0),
            7: (1, 986, 453, "model", 1.0),
            8: (1, 990, 457, "model", 1.0),
            # Normal top-edge re-entry.
            31: (1, 787, 11, "model_edge", 0.35),
            32: (1, 783, 41, "model", 1.0),
            33: (1, 780, 78, "model", 1.0),
        }
        candidates, bridges, stats = find_top_edge_interstitial_branch_frames(
            observations,
            frame_count=34,
            edge_top_y=180,
            edge_core_y=35,
            min_anchor_frames=3,
            max_interstitial_gap=40,
            min_branch_residual_px=110,
        )
        self.assertEqual(candidates, [3, 4, 5, 6, 7, 8])
        self.assertEqual(stats["candidate_frames"], 6)
        self.assertTrue(all(frame in bridges for frame in candidates))

    def test_top_entry_edge_points_are_not_hit_or_rally_evidence(self):
        track = [
            BallPoint(frame, 100.0 + frame * 8.0, 220.0 - frame * 70.0,
                      True, "model_edge", True, 0.35)
            for frame in range(4)
        ]
        # Even a sharp apparent turn in edge-only points must not produce a
        # curvature event or a synthetic rally segment.
        events = detect_hit_points(track, fps=25.0, min_speed_px=1.0)
        labels, segments = segment_rallies(track, events, fps=25.0)
        self.assertEqual(events, [])
        self.assertEqual(segments, [])
        self.assertTrue(all(label == 0 for label in labels))

    def test_edge_gap_does_not_get_interp_or_stale_kalman(self):
        model = {
            0: (1, 600, 3),
            3: (1, 800, 420),
        }
        track = smooth_ball_track(
            model, frame_count=5, fps=25, max_interp_gap=8, max_predict_gap=3
        )
        self.assertFalse(track[1].visible)
        self.assertFalse(track[2].visible)
        self.assertEqual(track[3].source, "model")

    def test_short_body_lock_times_out_without_erasing_top_flight(self):
        body = {
            frame: (1, 1128 + (frame % 2), 468, "model", 1.0)
            for frame in range(8)
        }
        body_track = smooth_ball_track(body, 8, 25, max_predict_gap=0)
        body_filtered = filter_ball_track_geometry(
            body_track,
            court_polygon=None,
            static_window=0,
            static_lock_window=0,
            static_lock_timeout_window=4,
            static_lock_timeout_disp=10,
            static_lock_min_y=80,
            static_lock_preroll=2,
        )
        self.assertLess(sum(point.visible for point in body_filtered), 4)

        edge = {
            frame: (1, 1000, 8, "model", 1.0)
            for frame in range(8)
        }
        edge_track = smooth_ball_track(edge, 8, 25, max_predict_gap=0)
        edge_filtered = filter_ball_track_geometry(
            edge_track,
            court_polygon=None,
            static_window=0,
            static_lock_window=0,
            static_lock_timeout_window=4,
            static_lock_timeout_disp=10,
            static_lock_min_y=80,
            static_lock_preroll=2,
        )
        self.assertEqual(sum(point.visible for point in edge_filtered), 8)

    def test_edge_reentry_mask_keeps_only_confirmed_outside_continuation(self):
        """A narrow edge continuation survives geometry without ROI widening."""
        court = np.asarray(
            [[0, 0], [100, 0], [100, 200], [0, 200]],
            dtype=np.float32,
        )
        observations = {
            0: (1, 105, 3, "model_edge", 0.35),
            1: (1, 106, 20, "model", 1.0),
            2: (1, 107, 40, "model", 1.0),
            3: (1, 98, 60, "model", 1.0),
        }
        frames = find_edge_reentry_frames(
            observations,
            frame_count=4,
            active_quad=court,
            edge_y=5,
            max_frames=4,
            max_y=80,
            min_outside_frames=2,
            max_horizontal_step=20,
            max_vertical_step=30,
        )
        self.assertEqual(frames, [1, 2])

        track = [
            BallPoint(
                frame,
                observations[frame][1],
                observations[frame][2],
                True,
                observations[frame][3],
                True,
                observations[frame][4],
            )
            for frame in range(4)
        ]
        without_mask = filter_ball_track_geometry(
            track,
            court_polygon=court,
            top_extension=0,
            side_padding=0,
            bottom_padding=0,
            static_window=0,
            static_lock_window=0,
            static_lock_timeout_window=0,
        )
        with_mask = filter_ball_track_geometry(
            track,
            court_polygon=court,
            top_extension=0,
            side_padding=0,
            bottom_padding=0,
            static_window=0,
            static_lock_window=0,
            static_lock_timeout_window=0,
            edge_reentry_frames=frames,
        )
        self.assertFalse(without_mask[1].visible)
        self.assertFalse(without_mask[2].visible)
        self.assertTrue(with_mask[1].visible)
        self.assertTrue(with_mask[2].visible)
        self.assertTrue(with_mask[3].visible)

    def test_edge_reentry_mask_rejects_body_jump_without_confirmation(self):
        court = np.asarray(
            [[0, 0], [100, 0], [100, 200], [0, 200]],
            dtype=np.float32,
        )
        observations = {
            0: (1, 105, 3, "model_edge", 0.35),
            1: (1, 106, 20, "model", 1.0),
            2: (1, 107, 45, "model", 1.0),
            # No later in-quad continuation: this is an unsupported branch.
        }
        self.assertEqual(
            find_edge_reentry_frames(
                observations,
                frame_count=3,
                active_quad=court,
                edge_y=5,
                max_frames=4,
                max_y=80,
                min_outside_frames=2,
                max_horizontal_step=20,
                max_vertical_step=30,
            ),
            [],
        )

    def test_player_box_veto_only_removes_requested_model_points(self):
        observations = {
            1: (1, 120, 140, "model", 1.0),
            2: (1, 125, 145, "model", 1.0),
            3: (1, 120, 140, "classical", 0.9),
        }
        evidence = {
            frame: {"players": [{"box": (100, 100, 160, 200)}]}
            for frame in (1, 2, 3)
        }
        output, vetoed = veto_model_observations_in_player_boxes(
            observations, evidence, frames={1, 3}
        )
        self.assertEqual(vetoed, [1])
        self.assertEqual(output[1][0], 0)
        self.assertEqual(output[2][0], 1)
        self.assertEqual(output[3][3], "classical")

    def test_short_wrong_way_branch_is_removed_but_real_fast_branch_survives(self):
        false_branch = {
            0: (1, 104, 212, "model", 1.0),
            1: (1, 1065, 176, "model", 1.0),
            2: (1, 1046, 78, "model", 1.0),
            3: (1, 1027, 15, "model", 1.0),
        }
        far_pose = {
            frame: {
                "players": [{
                    "box": (1000, 100, 1150, 450),
                    "side": "far",
                    "wrists": [(1080, 220)],
                    "contact": (1080, 220),
                }]
            }
            for frame in range(4)
        }
        filtered, vetoed, _ = veto_short_unsupported_phase_branches(
            false_branch, far_pose, {1, 2, 3}
        )
        self.assertEqual(vetoed, [1, 2, 3])
        self.assertTrue(all(filtered[frame][0] == 0 for frame in (1, 2, 3)))

        real_branch = {
            frame: value
            for frame, value in enumerate([
                (1, 735, 408, "model", 1.0),
                (1, 772, 371, "model", 1.0),
                (1, 836, 236, "model", 1.0),
                (1, 881, 161, "model", 1.0),
                (1, 911, 101, "model", 1.0),
                (1, 933, 60, "model", 1.0),
                (1, 952, 30, "model", 1.0),
                (1, 971, 7, "model", 1.0),
            ])
        }
        near_pose = {
            frame: {
                "players": [{
                    "box": (700, 300, 850, 700),
                    "side": "near",
                    "wrists": [(760, 400)],
                    "contact": (760, 400),
                }]
            }
            for frame in real_branch
        }
        filtered, vetoed, _ = veto_short_unsupported_phase_branches(
            real_branch, near_pose, set(real_branch)
        )
        self.assertEqual(vetoed, [])
        self.assertTrue(all(filtered[frame][0] == 1 for frame in real_branch))

    def test_dual_phase_coherent_motion_protects_f317_contact_run(self):
        """A complete dual-phase hit/reversal run must survive a body box."""
        coordinates = [
            (712, 491), (708, 517),
            (701, 543), (701, 566), (723, 562), (783, 517), (825, 483),
            (862, 457), (892, 438),
        ]
        primary = {
            frame: (1, x, y)
            for frame, (x, y) in enumerate(coordinates)
        }
        secondary = {
            frame: (1, x + (4 if frame % 3 == 0 else 0), y)
            for frame, (x, y) in enumerate(coordinates)
        }
        merged = {
            frame: (1, x, y, "model", 1.0)
            for frame, (x, y) in enumerate(coordinates)
        }
        candidates = {2, 3, 4, 5, 6}
        protected, stats = find_dual_phase_coherent_motion_frames(
            primary,
            secondary,
            merged,
            candidates,
            frame_count=len(coordinates),
            agreement_px=25,
            support_frames=2,
        )
        self.assertEqual(protected, [2, 3, 4, 5, 6])
        self.assertEqual(stats["protected_runs"], 1)

        # A relaxed player box can make the compact f317--319 portion look
        # static.  The dual-phase protection must take precedence over that
        # pose-only veto; contact points are allowed to overlap the hitter.
        pose = {
            frame: {
                "players": [{
                    "box": (650, 500, 780, 620),
                    "side": "near",
                    "wrists": [(500, 300)],
                    "contact": (500, 300),
                }],
                "veto_players": [{
                    "box": (650, 500, 780, 620),
                    "side": "near",
                    "wrists": [(500, 300)],
                    "contact": (500, 300),
                }],
            }
            for frame in candidates
        }
        _, unprotected_veto, _ = veto_short_unsupported_phase_branches(
            merged,
            pose,
            candidates,
            max_branch_frames=3,
        )
        self.assertTrue({2, 3, 4}.issubset(unprotected_veto))
        filtered, vetoed, _ = veto_short_unsupported_phase_branches(
            merged,
            pose,
            candidates,
            max_branch_frames=3,
            protected_frames=protected,
        )
        self.assertEqual(vetoed, [])
        self.assertTrue(all(filtered[frame][0] == 1 for frame in protected))

    def test_dual_phase_coherent_motion_accepts_edge_supported_f856_run(self):
        coordinates = [
            (746, 285), (742, 311),
            (738, 341), (738, 378), (735, 408),
            (772, 371), (836, 236), (881, 161),
            (911, 101), (933, 60),
        ]
        primary = {
            frame: (1, x, y)
            for frame, (x, y) in enumerate(coordinates)
        }
        secondary = {
            frame: (1, x, y + (4 if frame % 2 else 0))
            for frame, (x, y) in enumerate(coordinates)
        }
        merged = {
            frame: (
                1, x, y,
                "model_edge" if frame >= 8 else "model",
                0.35 if frame >= 8 else 1.0,
            )
            for frame, (x, y) in enumerate(coordinates)
        }
        # The edge-labelled f859/f860-style points are support only; the
        # protected candidate run ends at the ordinary f858-style point.
        protected, stats = find_dual_phase_coherent_motion_frames(
            primary,
            secondary,
            merged,
            candidate_frames=set(range(2, 10)),
            frame_count=len(coordinates),
            edge_y=30,
        )
        self.assertEqual(protected, list(range(2, 8)))
        self.assertEqual(stats["protected_frames"], 6)

    def test_dual_phase_motion_protection_rejects_static_and_top_branch(self):
        static_primary = {
            frame: (1, 700 + frame % 2, 500)
            for frame in range(7)
        }
        static_secondary = dict(static_primary)
        static_merged = {
            frame: (1, value[1], value[2], "model", 1.0)
            for frame, value in static_primary.items()
        }
        protected, stats = find_dual_phase_coherent_motion_frames(
            static_primary,
            static_secondary,
            static_merged,
            candidate_frames={2, 3, 4},
            frame_count=7,
        )
        self.assertEqual(protected, [])
        self.assertGreater(stats["rejected_motion"], 0)

        moving = {
            frame: (1, 900 - 20 * frame, 120 + 25 * frame)
            for frame in range(7)
        }
        moving_merged = {
            frame: (1, value[1], value[2], "model", 1.0)
            for frame, value in moving.items()
        }
        # Any excluded interstitial frame contaminates the local support, so
        # neither side of a known top/body branch receives protection.
        protected, stats = find_dual_phase_coherent_motion_frames(
            moving,
            moving,
            moving_merged,
            candidate_frames={2, 3, 4},
            frame_count=7,
            excluded_frames={3},
        )
        self.assertEqual(protected, [])
        self.assertGreater(stats["rejected_length"], 0)

        # A phase-isolated jump (the f563/f863 pattern) cannot establish the
        # dual-observation guarantee even when its surrounding coordinates
        # happen to move quickly.
        single_secondary = dict(moving)
        single_secondary[3] = (0, 0, 0)
        protected, _ = find_dual_phase_coherent_motion_frames(
            moving,
            single_secondary,
            moving_merged,
            candidate_frames={2, 3, 4},
            frame_count=7,
        )
        self.assertEqual(protected, [])

    def test_offscreen_bridge_is_render_only_and_top_direction_gated(self):
        track = [
            BallPoint(frame, 0.0, 0.0, False, "missing", False, 0.0)
            for frame in range(15)
        ]
        # The shuttle approaches the top edge (negative image-y velocity), is
        # absent for nine frames, then re-enters moving downwards.
        for frame, x, y in (
            (0, 600, 100), (1, 590, 70), (2, 580, 25),
            (12, 520, 20), (13, 515, 55), (14, 510, 90),
        ):
            track[frame] = BallPoint(frame, x, y, True, "model", True, 1.0)

        bridged = add_offscreen_bridges(track)
        self.assertTrue(all(bridged[frame].visible for frame in range(3, 12)))
        self.assertTrue(all(bridged[frame].source == "offscreen_bridge" for frame in range(3, 12)))
        self.assertTrue(all(not bridged[frame].raw_visible for frame in range(3, 12)))
        self.assertEqual(coverage(track)[1], coverage(bridged)[1])

        # A render bridge must not become a measured point for downstream
        # callers; only source-labelled model/classical points count as real.
        self.assertEqual(sum(point.source == "offscreen_bridge" for point in bridged), 9)

        # A low-confidence dual-phase edge anchor may define the visual
        # crossing even though it is intentionally excluded from evidence.
        edge_track = list(track)
        edge_track[2] = BallPoint(2, 580, 25, True, "model_edge", True, 0.35)
        edge_track[12] = BallPoint(12, 520, 20, True, "model_edge", True, 0.35)
        edge_bridged = add_offscreen_bridges(edge_track)
        self.assertTrue(all(
            edge_bridged[frame].source == "offscreen_bridge"
            for frame in range(3, 12)
        ))
        self.assertEqual(coverage(edge_track)[1], coverage(edge_bridged)[1])

    def test_offscreen_bridge_rejects_cut_wrong_direction_and_long_hole(self):
        def make_track(right_frame, right_y, left_y=20):
            length = right_frame + 3
            result = [
                BallPoint(frame, 0.0, 0.0, False, "missing", False, 0.0)
                for frame in range(length)
            ]
            result[0] = BallPoint(0, 500, 80, True, "model", True, 1.0)
            result[1] = BallPoint(1, 490, left_y, True, "model", True, 1.0)
            result[right_frame] = BallPoint(right_frame, 450, right_y, True, "model", True, 1.0)
            result[right_frame + 1] = BallPoint(right_frame + 1, 440, right_y - 15, True, "model", True, 1.0)
            result[right_frame + 2] = BallPoint(right_frame + 2, 430, right_y - 30, True, "model", True, 1.0)
            return result

        # The endpoint motion does not leave the top edge downwards.
        wrong = make_track(12, 20)
        self.assertFalse(any(point.source == "offscreen_bridge" for point in add_offscreen_bridges(wrong)))

        valid = make_track(12, 20, left_y=20)
        cut_result = add_offscreen_bridges(valid, scene_cuts=[6])
        self.assertFalse(any(point.source == "offscreen_bridge" for point in cut_result))

        long_hole = make_track(45, 20, left_y=20)
        self.assertFalse(any(point.source == "offscreen_bridge" for point in add_offscreen_bridges(long_hole)))

    def test_offscreen_bridge_accepts_edge_labelled_anchors_render_only(self):
        track = [
            BallPoint(frame, 0.0, 0.0, False, "missing", False, 0.0)
            for frame in range(15)
        ]
        track[0] = BallPoint(0, 600, 70, True, "model", True, 1.0)
        track[1] = BallPoint(1, 590, 20, True, "model_edge", True, 0.35)
        track[12] = BallPoint(12, 520, 20, True, "single_phase_edge", True, 0.22)
        track[13] = BallPoint(13, 515, 55, True, "model", True, 1.0)
        track[14] = BallPoint(14, 510, 90, True, "model", True, 1.0)
        bridged = add_offscreen_bridges(track)
        self.assertTrue(all(bridged[frame].source == "offscreen_bridge"
                            for frame in range(2, 12)))
        self.assertTrue(all(not bridged[frame].raw_visible
                            for frame in range(2, 12)))
        self.assertEqual(coverage(track)[1], coverage(bridged)[1])

    def test_offscreen_bridge_rejects_mixed_edge_and_body_anchor(self):
        track = [
            BallPoint(frame, 0.0, 0.0, False, "missing", False, 0.0)
            for frame in range(15)
        ]
        track[0] = BallPoint(0, 600, 70, True, "model", True, 1.0)
        track[1] = BallPoint(1, 590, 20, True, "model_edge", True, 0.35)
        track[12] = BallPoint(12, 520, 20, True, "model", True, 1.0)
        track[13] = BallPoint(13, 515, 55, True, "model", True, 1.0)
        track[14] = BallPoint(14, 510, 90, True, "model", True, 1.0)
        bridged = add_offscreen_bridges(track)
        self.assertFalse(any(point.source == "offscreen_bridge" for point in bridged))

    def test_mixed_edge_bridge_recovers_f698_to_f720_pattern_render_only(self):
        """A strict ordinary→edge apex bridge is visual-only evidence.

        The real clip has an ordinary TrackNet point at f698 just before the
        shuttle exits the top edge, then a phase-labelled edge point at f720
        on re-entry.  The three-frame monotone sides are retained while the
        21-frame interior is fully missing.
        """
        frame_count = 723
        track = [
            BallPoint(frame, 0.0, 0.0, False, "missing", False, 0.0)
            for frame in range(frame_count)
        ]
        for frame, x, y in (
            (696, 996, 93),
            (697, 976, 56),
            (698, 956, 15),
        ):
            track[frame] = BallPoint(frame, x, y, True, "model", True, 1.0)
        for frame, x, y, source in (
            (720, 742, 11, "model_edge"),
            (721, 735, 30, "model_edge"),
            (722, 731, 60, "model"),
        ):
            track[frame] = BallPoint(frame, x, y, True, source, True, 0.35)

        bridged = add_offscreen_bridges(
            track,
            top_y=30.0,
            allow_mixed_edge_bridge=True,
        )
        self.assertTrue(all(
            bridged[frame].visible and
            bridged[frame].source == "offscreen_bridge" and
            not bridged[frame].raw_visible
            for frame in range(699, 720)
        ))
        # Anchors and measured coverage are untouched.
        self.assertEqual(bridged[698].source, "model")
        self.assertEqual(bridged[720].source, "model_edge")
        self.assertEqual(coverage(track), coverage(bridged))
        # Synthetic points cannot create hit or rally evidence downstream.
        hits = detect_hit_points(bridged, fps=30.0, min_separation_frames=2)
        self.assertFalse(any(699 <= event.frame < 720 for event in hits))
        labels, _ = segment_rallies(bridged, [], fps=30.0, gap_frames=5)
        self.assertTrue(all(labels[frame] == 0 for frame in range(699, 720)))

    def test_mixed_edge_bridge_rejects_observed_hole_and_inconsistent_sides(self):
        frame_count = 723

        def make_track():
            result = [
                BallPoint(frame, 0.0, 0.0, False, "missing", False, 0.0)
                for frame in range(frame_count)
            ]
            for frame, x, y in (
                (696, 996, 93),
                (697, 976, 56),
                (698, 956, 15),
            ):
                result[frame] = BallPoint(frame, x, y, True, "model", True, 1.0)
            for frame, x, y, source in (
                (720, 742, 11, "model_edge"),
                (721, 735, 30, "model_edge"),
                (722, 731, 60, "model"),
            ):
                result[frame] = BallPoint(frame, x, y, True, source, True, 0.35)
            return result

        # A visible model/Kalman point in the hole must never be overwritten.
        observed_hole = make_track()
        observed_hole[710] = BallPoint(710, 850, 120, True, "model", True, 1.0)
        self.assertFalse(any(
            point.source == "offscreen_bridge"
            for point in add_offscreen_bridges(
                observed_hole, allow_mixed_edge_bridge=True
            )
        ))

        # A horizontal direction reversal is not a credible unseen apex.
        reversed_x = make_track()
        reversed_x[721] = BallPoint(721, 760, 30, True, "model_edge", True, 0.35)
        reversed_x[722] = BallPoint(722, 790, 60, True, "model", True, 0.35)
        self.assertFalse(any(
            point.source == "offscreen_bridge"
            for point in add_offscreen_bridges(
                reversed_x, allow_mixed_edge_bridge=True
            )
        ))

    def test_broad_interstitial_envelope_does_not_protect_ordinary_y176_point(self):
        """The broad search band must not shield an interior false branch.

        In the production clip f715 is a normal TrackNet response around
        y=176 between the f698 top exit and f720 re-entry.  ``edge_top_y=180``
        is needed to discover this branch, but only the strict core (30 px)
        is an edge observation.  The point must therefore be vetoed so the
        render-only mixed bridge can span f699--719.
        """
        frame_count = 723
        observations = {
            696: (1, 996, 93, "model", 1.0),
            697: (1, 976, 56, "model", 1.0),
            698: (1, 956, 15, "model", 1.0),
            720: (1, 742, 11, "model_edge", 0.35),
            721: (1, 735, 30, "model_edge", 0.35),
            722: (1, 731, 60, "model", 1.0),
        }
        for frame, y in (
            # f700--706 follow a distant body/background response; f703 is
            # already missing in the phase fusion and is intentionally
            # omitted here.
            (700, 502), (701, 513), (702, 525),
            (704, 532), (705, 528), (706, 525), (711, 150),
            (715, 176),
        ):
            observations[frame] = (1, 978 if frame <= 706 else 855, y, "model", 1.0)

        candidates, bridges, _ = find_top_edge_interstitial_branch_frames(
            observations,
            frame_count=frame_count,
            edge_top_y=180,
            edge_core_y=30,
            min_anchor_frames=3,
            max_interstitial_gap=40,
            min_branch_residual_px=110,
        )
        self.assertIn(715, candidates)
        filtered, vetoed, _ = veto_top_edge_interstitial_branches(
            observations,
            bridges,
            edge_top_y=180,
            edge_core_y=30,
        )
        self.assertIn(715, vetoed)
        self.assertEqual(filtered[715][3], "phase_top_branch_veto")

        track = [
            BallPoint(frame, 0, 0, False, "missing", False, 0.0)
            for frame in range(frame_count)
        ]
        for frame, value in filtered.items():
            if int(value[0]) > 0:
                track[frame] = BallPoint(
                    frame,
                    value[1],
                    value[2],
                    True,
                    value[3],
                    True,
                    value[4] if len(value) >= 5 else 1.0,
                )
        bridged = add_offscreen_bridges(
            track,
            top_y=30,
            allow_mixed_edge_bridge=True,
            mixed_support_frames=3,
        )
        self.assertTrue(all(
            bridged[frame].source == "offscreen_bridge"
            for frame in range(699, 720)
        ))

    def test_f445_to_f478_interstitial_veto_enables_strict_edge_bridge(self):
        """The f445--478 high-clear gap is visual-only after branch veto."""
        frame_count = 481
        observations = {}
        # Actual phase-fused provenance around the high-clear exit/re-entry:
        # the short dual-phase lead is edge-labelled, followed by the body
        # lock f448--455 and a detector outage through f477.
        for frame, x, y, source, confidence in (
            (445, 1065, 165, "model_edge", 0.35),
            (446, 1046, 78, "model_edge", 0.35),
            (447, 1027, 15, "model_edge", 0.35),
            (478, 787, 11, "model_edge", 0.35),
            (479, 783, 41, "model", 1.0),
            (480, 780, 78, "model", 1.0),
        ):
            observations[frame] = (1, x, y, source, confidence)
        for frame, (x, y) in enumerate(
            [(986, 446), (990, 442), (990, 442), (990, 442),
             (986, 446), (990, 450), (986, 453), (990, 457)],
            start=448,
        ):
            observations[frame] = (1, x, y, "model", 1.0)

        candidates, bridges, stats = find_top_edge_interstitial_branch_frames(
            observations,
            frame_count=frame_count,
            edge_core_y=35,
            min_anchor_frames=3,
            max_interstitial_gap=40,
            min_branch_residual_px=110,
        )
        self.assertEqual(candidates, list(range(448, 456)))
        self.assertEqual(stats["candidate_frames"], 8)
        filtered, vetoed, _ = veto_top_edge_interstitial_branches(
            observations,
            bridges,
            edge_top_y=180,
        )
        self.assertEqual(vetoed, candidates)

        track = []
        for frame in range(frame_count):
            value = filtered.get(frame)
            if value is None or int(value[0]) <= 0:
                track.append(BallPoint(frame, 0, 0, False, "missing", False, 0.0))
            else:
                track.append(BallPoint(
                    frame, value[1], value[2], True, value[3], True,
                    value[4] if len(value) >= 5 else 1.0,
                ))
        bridged = add_offscreen_bridges(
            track,
            top_y=30,
            allow_mixed_edge_bridge=True,
            mixed_support_frames=3,
        )
        self.assertTrue(all(
            bridged[frame].source == "offscreen_bridge"
            and bridged[frame].visible
            and not bridged[frame].raw_visible
            for frame in range(448, 478)
        ))
        self.assertEqual(bridged[447].source, "model_edge")
        self.assertEqual(bridged[478].source, "model_edge")


if __name__ == "__main__":
    unittest.main()
