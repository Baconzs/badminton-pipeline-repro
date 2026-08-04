import csv
import json
import tempfile
import unittest
from pathlib import Path

from web_app.pipeline_service import JobManager, summarize_sidecar


class SidecarSummaryTests(unittest.TestCase):
    def test_summary_counts_hits_rallies_and_real_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracking.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "Frame", "Visibility", "Source", "Hit", "RallyID",
                    "ShotAvgSpeedKmh",
                ])
                writer.writeheader()
                writer.writerows([
                    {"Frame": 0, "Visibility": 1, "Source": "model", "Hit": 1,
                     "RallyID": 1, "ShotAvgSpeedKmh": 120},
                    {"Frame": 1, "Visibility": 1, "Source": "interp", "Hit": 0,
                     "RallyID": 1, "ShotAvgSpeedKmh": ""},
                    {"Frame": 2, "Visibility": 0, "Source": "missing", "Hit": 0,
                     "RallyID": 0, "ShotAvgSpeedKmh": ""},
                    {"Frame": 3, "Visibility": 1, "Source": "classical", "Hit": 1,
                     "RallyID": 2, "ShotAvgSpeedKmh": 80},
                ])
            summary = summarize_sidecar(path, fps=2.0)
        self.assertEqual(summary["hit_count"], 2)
        self.assertEqual(summary["rally_count"], 2)
        self.assertEqual(summary["real_coverage"], 0.5)
        self.assertEqual(summary["duration_seconds"], 2.0)
        self.assertEqual(summary["average_shot_speed_kmh"], 100.0)


class JobManifestRecoveryTests(unittest.TestCase):
    def test_unhashable_status_is_recovered_as_failed(self):
        """A damaged manifest must not prevent the service from starting."""
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            job_id = "aaaaaaaaaaaa"
            job_dir = data_root / "jobs" / job_id
            job_dir.mkdir(parents=True)
            (job_dir / "job.json").write_text(
                json.dumps({"id": job_id, "status": []}),
                encoding="utf-8",
            )

            manager = JobManager(Path(__file__).resolve().parents[1], data_root)
            try:
                recovered = manager.get_job(job_id)
                self.assertEqual(recovered["status"], "failed")
                self.assertEqual(recovered["stage"], "服务重启，任务已中断")
            finally:
                # Stop this test manager's daemon worker so repeated test
                # runs do not leave idle worker threads behind.
                manager._queue.put(None)
                manager._worker.join(timeout=1)


class MotionCoachJobConfigTests(unittest.TestCase):
    def test_target_player_and_handedness_are_persisted_with_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"placeholder")
            manager = JobManager(Path(__file__).resolve().parents[1], root / "runtime")
            try:
                job = manager.create_job(
                    {"id": "video", "name": "source.mp4", "path": str(source), "metadata": {}},
                    split_rallies=False,
                    professional_analysis=False,
                    hawkeye_review=False,
                    court_points=[],
                    coach_target="far",
                    near_dominant_hand="left",
                    far_dominant_hand="right",
                )
                self.assertEqual(job["config"]["coach_target"], "far")
                self.assertEqual(job["config"]["near_dominant_hand"], "left")
                self.assertEqual(job["config"]["far_dominant_hand"], "right")
                with self.assertRaises(ValueError):
                    manager.create_job(
                        {"id": "video", "name": "source.mp4", "path": str(source), "metadata": {}},
                        split_rallies=False,
                        professional_analysis=False,
                        hawkeye_review=False,
                        court_points=[],
                        coach_target="diagonal",
                    )
            finally:
                manager._queue.put(None)
                manager._worker.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
