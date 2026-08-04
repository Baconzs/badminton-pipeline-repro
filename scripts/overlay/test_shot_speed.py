#!/usr/bin/env python3
"""Regression tests for conservative projected shuttle-speed estimates."""

import unittest

import numpy as np

from ball_tracking import BallPoint, estimate_shot_average_speed


def make_track(length=11, source="model"):
    return [
        BallPoint(
            frame=frame,
            x=float(frame),
            y=5.0,
            visible=True,
            source=source,
            raw_visible=(source == "model"),
            confidence=1.0,
        )
        for frame in range(length)
    ]


class ShotSpeedTests(unittest.TestCase):
    def test_constant_projected_motion_has_expected_speed(self):
        estimate = estimate_shot_average_speed(
            make_track(), 0, 10, 10.0, np.eye(3),
            court_width_m=20.0, court_length_m=20.0,
        )
        self.assertTrue(estimate["available"])
        self.assertAlmostEqual(estimate["distance_m"], 10.0, places=5)
        self.assertAlmostEqual(estimate["observed_seconds"], 1.0, places=5)
        self.assertAlmostEqual(estimate["speed_mps"], 10.0, places=5)
        self.assertAlmostEqual(estimate["speed_kmh"], 36.0, places=5)

    def test_missing_or_render_only_points_do_not_bridge_distance(self):
        track = make_track()
        track[4] = BallPoint(4, 4.0, 5.0, False, "missing", False, 0.0)
        track[5] = BallPoint(5, 5.0, 5.0, True, "offscreen_bridge", False, 0.1)
        track[6] = BallPoint(6, 6.0, 5.0, True, "kalman", False, 0.4)
        estimate = estimate_shot_average_speed(
            track, 0, 10, 10.0, np.eye(3),
            court_width_m=20.0, court_length_m=20.0,
            min_coverage=0.80,
        )
        self.assertFalse(estimate["available"])
        self.assertEqual(estimate["quality"], "insufficient_coverage")
        # 0--3 and 7--10 are the only valid contiguous segments.
        self.assertAlmostEqual(estimate["observed_seconds"], 0.6, places=5)

    def test_horizon_projection_is_rejected_instead_of_creating_fake_speed(self):
        track = make_track()
        for point in track:
            point.y = 0.0
        track[5] = BallPoint(5, 5.0, 5.0, True, "model", True, 1.0)
        # This maps y=5 to an unbounded / far-away court coordinate, which
        # lies outside the provided court margin.
        homography = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -0.19, 1.0],
        ])
        estimate = estimate_shot_average_speed(
            track, 0, 10, 10.0, homography,
            court_width_m=20.0, court_length_m=20.0, court_margin_m=1.0,
            min_coverage=0.90,
        )
        self.assertFalse(estimate["available"])
        self.assertIn(estimate["quality"], {"insufficient_segments", "insufficient_coverage"})

    def test_capped_perspective_proxy_recovers_high_flight_motion(self):
        track = make_track()
        for point in track:
            point.y = -4.0
        # Direct floor projection above the baseline is outside the tiny court, while a
        # local scale evaluated at the supplied far-baseline y remains stable.
        homography = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -0.19, 1.0],
        ])
        estimate = estimate_shot_average_speed(
            track, 0, 10, 10.0, homography,
            court_width_m=20.0, court_length_m=20.0, court_margin_m=1.0,
            proxy_floor_y=0.0,
            proxy_min_coverage=0.8,
            proxy_min_real_points=8,
            proxy_min_segments=8,
            proxy_max_scale_m_per_px=10.0,
        )
        self.assertTrue(estimate["available"])
        self.assertEqual(estimate["method"], "perspective_proxy")
        self.assertEqual(estimate["quality"], "proxy")
        self.assertAlmostEqual(estimate["speed_mps"], 10.0, places=5)


if __name__ == "__main__":
    unittest.main()
