"""Regression tests for the explicit bullet-time pipeline option."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from web_app.pipeline_service import JobManager


class _ProfessionalPipelineProbe(JobManager):
    """Small in-memory probe that records the shell command only."""

    def __init__(self, root: Path) -> None:
        # Avoid starting JobManager's worker: _run_professional_pipeline only
        # needs the repository path, an update hook, and a command hook.
        self.repo_root = root
        self.python_bin = "python3"
        self._updates = []
        self.command = None
        self.command_kwargs = {}

    def _update(self, job_id: str, **values) -> None:  # type: ignore[override]
        self._updates.append((job_id, values))

    def _run_command(self, job_id, command, **kwargs) -> None:  # type: ignore[override]
        self.command = list(command)
        self.command_kwargs = kwargs


def _make_pipeline_artifacts(work_root: Path) -> None:
    work_root.mkdir(parents=True, exist_ok=True)
    # The service only checks existence here; media decoding is covered by the
    # normal pipeline tests and would make this switch test needlessly heavy.
    for name in (
        "end1_ball_tracking_official_fused_ball_tracking.csv",
        "end1_ball_tracking_official_fused_h264.mp4",
        "end1_ball_tracking_official_fused.mp4",
        "end1_ball_tracking_official_fused_fx_h264.mp4",
        "end1_ball_tracking_official_fused_fx.mp4",
    ):
        (work_root / name).write_bytes(b"placeholder")


class BulletTimeSwitchTests(unittest.TestCase):
    def _run(self, env_value, selected=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            _make_pipeline_artifacts(work)
            probe = _ProfessionalPipelineProbe(root)
            environment = {} if env_value is None else {
                "BADMINTON_ENABLE_BULLET_TIME_FX": env_value,
            }
            with patch.dict(os.environ, environment, clear=True):
                result = probe._run_professional_pipeline(
                    "job",
                    root / "input.mp4",
                    work,
                    "0,0,1,0,1,1,0,1",
                    "auto",
                    False,
                    selected,
                )
            return probe, result

    def test_web_default_explicitly_disables_fx_and_prefers_normal_video(self):
        probe, result = self._run(None)

        self.assertIn("--no-bullet-time-fx", probe.command)
        self.assertNotIn("--enable-bullet-time-fx", probe.command)
        self.assertTrue(str(result[1]).endswith("end1_ball_tracking_official_fused_h264.mp4"))
        self.assertEqual(
            probe.command_kwargs["stage_markers"]["[STEP 3/3]"][1],
            "整理普通速度分析视频",
        )

    def test_environment_opt_in_enables_fx_and_prefers_fx_video(self):
        probe, result = self._run("1")

        self.assertIn("--enable-bullet-time-fx", probe.command)
        self.assertNotIn("--no-bullet-time-fx", probe.command)
        self.assertTrue(str(result[1]).endswith("end1_ball_tracking_official_fused_fx_h264.mp4"))
        self.assertEqual(
            probe.command_kwargs["stage_markers"]["[STEP 3/3]"][1],
            "生成视觉特效",
        )

    def test_explicit_false_overrides_legacy_environment_opt_in(self):
        probe, result = self._run("1", selected=False)

        self.assertIn("--no-bullet-time-fx", probe.command)
        self.assertNotIn("--enable-bullet-time-fx", probe.command)
        self.assertTrue(str(result[1]).endswith("end1_ball_tracking_official_fused_h264.mp4"))

    def test_explicit_true_overrides_legacy_environment_default(self):
        probe, result = self._run(None, selected=True)

        self.assertIn("--enable-bullet-time-fx", probe.command)
        self.assertNotIn("--no-bullet-time-fx", probe.command)
        self.assertTrue(str(result[1]).endswith("end1_ball_tracking_official_fused_fx_h264.mp4"))

    def test_selected_fx_also_enables_professional_job_for_direct_clients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"placeholder")
            # This test only checks normalized job configuration.  Keep the
            # background worker idle so no video pipeline is started.
            with patch.object(JobManager, "_worker_loop", return_value=None):
                manager = JobManager(root, root / "runtime")
            try:
                job = manager.create_job(
                    {"id": "video", "name": source.name, "path": str(source), "metadata": {}},
                    split_rallies=False,
                    professional_analysis=False,
                    hawkeye_review=False,
                    bullet_time_enabled=True,
                    court_points=[0, 0, 1, 0, 1, 1, 0, 1],
                )
                self.assertTrue(job["config"]["professional_analysis"])
                self.assertTrue(job["config"]["bullet_time_enabled"])
                self.assertTrue(job["config"]["bullet_time_fx"])
            finally:
                manager._queue.put(None)
                manager._worker.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
