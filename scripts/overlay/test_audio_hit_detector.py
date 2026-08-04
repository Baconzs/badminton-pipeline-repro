#!/usr/bin/env python3
"""Regression coverage for silent-video audio evidence fallback."""

import unittest
from unittest.mock import patch

import numpy as np

import audio_hit_detector as detector
from ball_tracking import BallPoint, HitEvent, validate_hit_event_evidence


class AudioAvailabilityTests(unittest.TestCase):
    def test_legacy_call_stays_a_list_and_silent_source_reports_unavailable(self):
        with patch.object(detector, "_decode_mono_pcm", return_value=np.empty(0, dtype=np.float32)):
            legacy = detector.detect_audio_hit_candidates("silent.mp4", 25.0)
            candidates, available = detector.detect_audio_hit_candidates(
                "silent.mp4", 25.0, return_audio_available=True
            )
        self.assertEqual(legacy, [])
        self.assertEqual(candidates, [])
        self.assertFalse(available)

    def test_decoded_samples_report_available_even_without_a_hit_candidate(self):
        # A constant PCM signal deliberately contains no onset, but it still
        # represents a valid audio stream and must keep the audio gate active.
        samples = np.zeros(2048, dtype=np.float32)
        with patch.object(detector, "_decode_mono_pcm", return_value=samples):
            candidates, available = detector.detect_audio_hit_candidates(
                "quiet.mp4", 25.0, return_audio_available=True
            )
        self.assertEqual(candidates, [])
        self.assertTrue(available)

    def test_visual_event_is_allowed_only_when_audio_gate_is_disabled(self):
        track = [
            BallPoint(frame, float(frame * 10), 50.0, True, "model", True, 1.0)
            for frame in range(8)
        ]
        event = HitEvent(4, 40.0, 50.0, 0.9, 12.0, source="curvature")
        accepted, stats = validate_hit_event_evidence(
            track, [event], audio_frames=[], require_audio=False
        )
        self.assertEqual([item.frame for item in accepted], [4])
        self.assertEqual(stats["accepted"], 1)


if __name__ == "__main__":
    unittest.main()
