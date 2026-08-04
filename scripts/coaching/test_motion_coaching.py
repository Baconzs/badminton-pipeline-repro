"""Tests for conservative 2-D coaching rules; no model or video is required."""

import unittest

from motion_coaching import analyze_motion


def pose(knee_y=150, shoulder_x=100, hip_x=100, elbow=(135, 75), wrist=(165, 75)):
    points = [[0.0, 0.0] for _ in range(17)]
    # Symmetric body; hip-knee-ankle is almost straight when knee_y=150.
    points[5], points[6] = [shoulder_x - 20, 70], [shoulder_x + 20, 70]
    points[11], points[12] = [hip_x - 18, 120], [hip_x + 18, 120]
    points[13], points[14] = [hip_x - 18, knee_y], [hip_x + 18, knee_y]
    points[15], points[16] = [hip_x - 18, 190], [hip_x + 18, 190]
    points[7], points[8] = list(elbow), [shoulder_x - (elbow[0] - shoulder_x), elbow[1]]
    points[9], points[10] = list(wrist), [shoulder_x - (wrist[0] - shoulder_x), wrist[1]]
    return points


def sample(frame, phase="ready", **extra):
    result = {"frame": frame, "phase": phase, "keypoints": pose(), "confidences": [0.92] * 17}
    result.update(extra)
    return result


class MotionCoachingTests(unittest.TestCase):
    def ids(self, report):
        return {item["id"] for item in report["suggestions"]}

    def test_straight_ready_knees_create_explainable_prompt(self):
        report = analyze_motion([sample(i) for i in range(18)], fps=30, player_side="bottom")
        item = next(item for item in report["suggestions"] if item["id"] == "ready_knee_flexion")
        self.assertEqual(item["confidence"], "high")
        self.assertGreaterEqual(item["observed"]["median_knee_angle_deg"], 165)
        self.assertIn("二维", item["caveat"])

    def test_missing_ready_phase_suppresses_preparation_conclusion(self):
        report = analyze_motion([sample(i, phase="movement") for i in range(30)], fps=30, player_side="top")
        self.assertNotIn("ready_knee_flexion", self.ids(report))
        self.assertIn("ready_knee_flexion", {item["id"] for item in report["suppressed_checks"]})

    def test_large_ready_torso_sway_creates_prompt_only_in_ready_phase(self):
        samples = [sample(i, keypoints=pose(shoulder_x=100 + i * 2.5, hip_x=100)) for i in range(18)]
        report = analyze_motion(samples, fps=30, player_side="top")
        item = next(item for item in report["suggestions"] if item["id"] == "torso_stability")
        self.assertGreaterEqual(item["observed"]["shoulder_sway_over_torso"], 0.32)
        self.assertGreaterEqual(item["observed"]["torso_tilt_spread_deg"], 15.0)

    def test_overhead_compact_arm_needs_three_contacts(self):
        samples = [sample(i) for i in range(18)]
        # Bent right arm and a short wrist reach relative to torso.
        for frame in (20, 30, 40):
            samples.append(sample(frame, phase="contact", shot_context="overhead", hitting_arm="right",
                                  keypoints=pose(elbow=(125, 75), wrist=(123, 85))))
        report = analyze_motion(samples, fps=30, player_side="top")
        self.assertIn("overhead_arm_extension", self.ids(report))
        item = next(item for item in report["suggestions"] if item["id"] == "overhead_arm_extension")
        self.assertEqual(item["title"], "上手击球伸展")
        self.assertNotIn("复核", item["message"])

    def test_non_overhead_contact_does_not_get_arm_extension_prompt(self):
        samples = [sample(i) for i in range(18)]
        samples.extend(sample(frame, phase="contact", shot_context="drive", keypoints=pose(elbow=(125, 75), wrist=(126, 100))) for frame in (20, 30, 40))
        report = analyze_motion(samples, fps=30, player_side="top")
        self.assertNotIn("overhead_arm_extension", self.ids(report))

    def test_recovery_needs_calibrated_positions_and_repeated_opportunities(self):
        samples = [sample(i, court_xy_m=(3.0, 3.0)) for i in range(20)]
        for hit_frame in (30, 80, 130):
            samples.append(sample(hit_frame, phase="contact", hit=True, court_xy_m=(5.0, 3.0)))
            # Remain far away for the required post-hit observation window.
            for later in (hit_frame + 18, hit_frame + 24, hit_frame + 30):
                samples.append(sample(later, phase="movement", court_xy_m=(5.0, 3.0)))
        report = analyze_motion(samples, fps=30, player_side="bottom", court_calibrated=True)
        self.assertIn("movement_recovery", self.ids(report))
        self.assertEqual(next(item for item in report["suggestions"] if item["id"] == "movement_recovery")["evidence"]["opportunities"], 3)

    def test_uncalibrated_positions_never_create_recovery_prompt(self):
        report = analyze_motion([sample(i, court_xy_m=(3.0, 3.0)) for i in range(30)], fps=30, player_side="bottom")
        self.assertNotIn("movement_recovery", self.ids(report))
        self.assertIn("court_coordinates_not_calibrated", {item["reason"] for item in report["suppressed_checks"]})


if __name__ == "__main__":
    unittest.main()
