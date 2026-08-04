"""Single-worker job orchestration for the local badminton Web UI."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import csv
import datetime as dt
import json
import os
import queue
import re
import signal
import shutil
import subprocess
import sys
import threading
import traceback
import uuid

from .rally_cutter import RallyCutCancelled, cut_rallies, read_rally_segments
from .bst_adapter import attach_rule_candidates, run_bst_stroke_inference
from .coaching import run_pose_coaching
from .hawkeye import run_hawkeye_review


TERMINAL_STATES = {"completed", "failed", "cancelled"}
_PROGRESS_RE = re.compile(r"(?P<current>\d+)\s*/\s*(?P<total>\d+)")
_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_PATH_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _safe_float(value, fallback=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _env_bullet_time_enabled() -> bool:
    """Return the legacy environment opt-in for the cinematic FX pass.

    The Web UI now sends an explicit ``bullet_time_enabled`` value for every
    new job.  Keeping this tiny fallback preserves compatibility with command
    line/API clients created before that field existed (and with direct calls
    to :meth:`_run_professional_pipeline`).
    """
    raw = str(os.environ.get("BADMINTON_ENABLE_BULLET_TIME_FX", "0"))
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def summarize_sidecar(path: Path | str, fps: float) -> Dict[str, object]:
    total = 0
    visible = 0
    real = 0
    hits = 0
    rallies = set()
    speeds: List[float] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            total += 1
            if int(_safe_float(row.get("Visibility"), 0)) > 0:
                visible += 1
            if str(row.get("Source", "")) in {"model", "classical"}:
                real += 1
            if int(_safe_float(row.get("Hit"), 0)) > 0:
                hits += 1
                speed = _safe_float(row.get("ShotAvgSpeedKmh"), -1)
                if speed >= 0:
                    speeds.append(speed)
            rally_id = int(_safe_float(row.get("RallyID"), 0))
            if rally_id > 0:
                rallies.add(rally_id)
    return {
        "frame_count": total,
        "duration_seconds": round(total / max(1.0, float(fps)), 2),
        "hit_count": hits,
        "rally_count": len(rallies),
        "visible_coverage": round(visible / total, 4) if total else 0.0,
        "real_coverage": round(real / total, 4) if total else 0.0,
        "max_shot_speed_kmh": round(max(speeds), 1) if speeds else None,
        "average_shot_speed_kmh": (
            round(sum(speeds) / len(speeds), 1) if speeds else None
        ),
    }


class JobCancelled(RuntimeError):
    pass


class JobManager:
    """Persist jobs and execute at most one GPU pipeline at a time."""

    def __init__(
        self,
        repo_root: Path | str,
        data_root: Path | str,
        python_bin: Optional[Path | str] = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.data_root = Path(data_root).resolve()
        self.jobs_root = self.data_root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Dict[str, object]] = {}
        self._lock = threading.RLock()
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._active_process: Dict[str, subprocess.Popen] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        requested_python = str(python_bin or sys.executable)
        if not Path(requested_python).is_absolute():
            requested_python = shutil.which(requested_python) or requested_python
        self.python_bin = requested_python
        self._load_existing_jobs()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="badminton-pipeline-worker",
            daemon=True,
        )
        self._worker.start()

    def _load_existing_jobs(self) -> None:
        for manifest in self.jobs_root.glob("*/job.json"):
            try:
                job = json.loads(manifest.read_text(encoding="utf-8"))
                job_id = str(job["id"])
            except Exception:
                continue
            if not isinstance(job, dict):
                continue
            # Never let a hand-edited job manifest turn persistence into a
            # path traversal (``jobs_root / job_id`` is used below).
            if not _JOB_ID_RE.fullmatch(job_id) or job_id != manifest.parent.name:
                continue
            # Treat malformed status values (for example a hand-edited list
            # or object) as interrupted jobs too.  Checking membership on an
            # unhashable value would otherwise abort service initialization.
            status = job.get("status")
            if not isinstance(status, str) or status not in TERMINAL_STATES:
                job["status"] = "failed"
                job["stage"] = "服务重启，任务已中断"
                job["error"] = "Job interrupted by server restart"
                job["updated_at"] = _now()
            self._jobs[job_id] = job
            self._persist(job_id)

    def _persist(self, job_id: str) -> None:
        job = self._jobs[job_id]
        directory = self.jobs_root / job_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "job.json"
        temporary = directory / "job.json.tmp"
        temporary.write_text(
            json.dumps(job, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(target)

    def create_job(
        self,
        video: Mapping[str, object],
        *,
        split_rallies: bool,
        professional_analysis: bool,
        court_points: Sequence[float],
        device: str = "auto",
        continue_after_scene_cuts: bool = False,
        hawkeye_review: bool = False,
        coach_target: str = "near",
        near_dominant_hand: str = "auto",
        far_dominant_hand: str = "auto",
        bullet_time_enabled: Optional[bool] = None,
        bullet_time_fx: Optional[bool] = None,
    ) -> Dict[str, object]:
        source = Path(str(video.get("path", ""))).resolve()
        if not source.is_file():
            raise ValueError("视频文件不存在")
        if (
            not isinstance(split_rallies, bool)
            or not isinstance(professional_analysis, bool)
            or not isinstance(continue_after_scene_cuts, bool)
            or not isinstance(hawkeye_review, bool)
        ):
            raise ValueError("分析开关必须是布尔值")
        if bullet_time_enabled is not None and not isinstance(bullet_time_enabled, bool):
            raise ValueError("子弹镜头开关必须是布尔值")
        if bullet_time_fx is not None and not isinstance(bullet_time_fx, bool):
            raise ValueError("子弹镜头开关必须是布尔值")
        if bullet_time_enabled is None:
            bullet_time_enabled = bullet_time_fx
        elif bullet_time_fx is not None and bullet_time_fx != bullet_time_enabled:
            raise ValueError("子弹镜头开关参数冲突")
        # Bullet-time is produced by the professional TrackNet/pose pipeline.
        # The browser already mirrors this dependency in its controls, but
        # enforce it here as well for direct API calls and older cached pages.
        # A missing option still keeps the legacy environment fallback from
        # unexpectedly upgrading a lightweight job.
        if bullet_time_enabled is True:
            professional_analysis = True
        points = [float(value) for value in court_points]
        if (split_rallies or professional_analysis or hawkeye_review) and len(points) != 8:
            raise ValueError("智能处理需要按 TL → TR → BR → BL 标定四个球场角点")
        if any(not (value >= 0.0 and value < 100000.0) for value in points):
            raise ValueError("球场标定坐标无效")
        normalized_device = str(device or "auto").lower()
        if normalized_device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("不支持的计算设备")
        normalized_coach_target = str(coach_target or "near").lower()
        if normalized_coach_target not in {"near", "far", "both"}:
            raise ValueError("动作教练分析对象必须是下方、上方或双方球员")
        normalized_near_hand = str(near_dominant_hand or "auto").lower()
        normalized_far_hand = str(far_dominant_hand or "auto").lower()
        if normalized_near_hand not in {"left", "right", "auto"} or normalized_far_hand not in {"left", "right", "auto"}:
            raise ValueError("持拍手必须是左手、右手或自动识别")

        # ``None`` means an older client did not send the new option.  Resolve
        # that case from the historical environment switch, but persist the
        # resolved value so a resumed job always reproduces the selected
        # output.  FX is implemented by the professional pipeline only; an
        # accidental opt-in on a lightweight job is therefore safely ignored.
        requested_bullet_time = (
            _env_bullet_time_enabled()
            if bullet_time_enabled is None
            else bool(bullet_time_enabled)
        )
        resolved_bullet_time = bool(requested_bullet_time and professional_analysis)

        job_id = uuid.uuid4().hex[:12]
        created = _now()
        job: Dict[str, object] = {
            "id": job_id,
            "video_id": str(video.get("id", "")),
            "video": {
                "name": str(video.get("name", source.name)),
                "path": str(source),
                "metadata": deepcopy(dict(video.get("metadata", {}) or {})),
            },
            "config": {
                "split_rallies": bool(split_rallies),
                "professional_analysis": bool(professional_analysis),
                "hawkeye_review": bool(hawkeye_review),
                "court_points": points,
                "device": normalized_device,
                "continue_after_scene_cuts": bool(continue_after_scene_cuts),
                # These values only affect the optional professional Motion
                # Coach pass, but are persisted with every job so historical
                # reports remain reproducible when reopened in the UI.
                "coach_target": normalized_coach_target,
                "near_dominant_hand": normalized_near_hand,
                "far_dominant_hand": normalized_far_hand,
                "bullet_time_enabled": resolved_bullet_time,
                # Keep the descriptive FX alias for API clients that use it.
                "bullet_time_fx": resolved_bullet_time,
            },
            "status": "queued",
            "progress": 0,
            "stage": "等待 GPU 任务队列",
            "created_at": created,
            "updated_at": created,
            "logs": [],
            "outputs": {},
            "rallies": [],
            "stats": {},
            # Produced only by professional jobs.  Keeping it inline in the
            # manifest avoids another path-bearing public artifact.
            "coaching": {},
            # Optional BST classifier output.  This is deliberately separate
            # from Motion Coach's evidence-based technique suggestions.
            "stroke_types": {},
            # A compact, path-free terminal-trajectory review.  It is
            # deliberately independent of the optional pose-coaching report.
            "hawkeye": {},
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._cancel_events[job_id] = threading.Event()
            if not split_rallies and not professional_analysis and not hawkeye_review:
                job["status"] = "completed"
                job["progress"] = 100
                job["stage"] = "视频已导入，无额外处理"
                job["updated_at"] = _now()
                self._persist(job_id)
            else:
                self._persist(job_id)
                self._queue.put(job_id)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Dict[str, object]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return deepcopy(self._jobs[job_id])

    def list_jobs(self, limit: int = 20) -> List[Dict[str, object]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda job: str(job.get("created_at", "")),
                reverse=True,
            )
            return deepcopy(jobs[: max(1, int(limit))])

    def output_path(self, job_id: str, key: str) -> Path:
        if not _PATH_ID_RE.fullmatch(str(job_id)):
            raise KeyError(key)
        job = self.get_job(job_id)
        raw_outputs = job.get("outputs", {})
        output = raw_outputs.get(key) if isinstance(raw_outputs, Mapping) else None
        if not isinstance(output, Mapping) or not output.get("path"):
            raise KeyError(key)
        path = Path(str(output["path"])).resolve()
        job_root = (self.jobs_root / job_id).resolve()
        try:
            job_root.relative_to(self.jobs_root.resolve())
            path.relative_to(job_root)
        except ValueError as exc:
            raise KeyError(key) from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def cancel(self, job_id: str) -> Dict[str, object]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
            if job.get("status") in TERMINAL_STATES:
                return deepcopy(job)
            event = self._cancel_events.setdefault(job_id, threading.Event())
            event.set()
            if job.get("status") == "queued":
                # A queued task has no child process yet.  Make its terminal
                # state visible immediately so this browser is not locked
                # behind an unrelated long GPU job ahead of it; the worker
                # will simply skip it when it reaches the queue item.
                job["status"] = "cancelled"
                job["stage"] = "任务已取消"
                job["updated_at"] = _now()
                self._persist(job_id)
                return deepcopy(job)
            job["stage"] = "正在取消任务"
            job["updated_at"] = _now()
            process = self._active_process.get(job_id)
            self._persist(job_id)
        if process is not None and process.poll() is None:
            self._terminate_process(process)
        return self.get_job(job_id)

    def _update(self, job_id: str, **values) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if "progress" in values:
                try:
                    values["progress"] = max(
                        int(job.get("progress", 0)), int(values["progress"])
                    )
                except (TypeError, ValueError):
                    values.pop("progress", None)
            job.update(values)
            job["updated_at"] = _now()
            self._persist(job_id)

    def _log(self, job_id: str, message: str) -> None:
        clean = " ".join(str(message).replace("\r", " ").split())
        if not clean:
            return
        with self._lock:
            job = self._jobs[job_id]
            logs = deque(job.get("logs", []), maxlen=160)
            logs.append({"time": _now(), "message": clean[-1200:]})
            job["logs"] = list(logs)
            job["updated_at"] = _now()
            self._persist(job_id)

    def _cancelled(self, job_id: str) -> bool:
        return self._cancel_events.get(job_id, threading.Event()).is_set()

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            try:
                self._execute_job(job_id)
            finally:
                self._queue.task_done()

    def _execute_job(self, job_id: str) -> None:
        if self._cancelled(job_id):
            self._update(job_id, status="cancelled", stage="任务已取消")
            return
        self._update(job_id, status="running", progress=2, stage="初始化任务")
        try:
            job = self.get_job(job_id)
            config = dict(job["config"])
            source = Path(str(dict(job["video"])["path"])).resolve()
            job_root = (self.jobs_root / job_id).resolve()
            work_root = job_root / "work"
            work_root.mkdir(parents=True, exist_ok=True)
            points = ",".join(
                str(int(round(float(value))))
                for value in config.get("court_points", [])
            )

            if config.get("professional_analysis"):
                sidecar, analysis_video, rally_clip_source = self._run_professional_pipeline(
                    job_id,
                    source,
                    work_root,
                    points,
                    str(config.get("device", "auto")),
                    bool(config.get("continue_after_scene_cuts", False)),
                    bullet_time_enabled=bool(
                        config.get("bullet_time_enabled", config.get("bullet_time_fx", False))
                    ),
                )
            else:
                sidecar = self._run_rally_index_pipeline(
                    job_id,
                    source,
                    work_root,
                    points,
                    str(config.get("device", "auto")),
                    bool(config.get("continue_after_scene_cuts", False)),
                )
                analysis_video = None
                rally_clip_source = source

            if self._cancelled(job_id):
                raise JobCancelled()
            fps = _safe_float(
                dict(dict(job["video"]).get("metadata", {}) or {}).get("fps"),
                25.0,
            )
            stats = summarize_sidecar(sidecar, fps)
            if self._cancelled(job_id):
                raise JobCancelled()
            hawkeye: Dict[str, object] = {}
            if config.get("hawkeye_review"):
                self._update(job_id, progress=92, stage="生成鹰眼辅助落点复核")
                hawkeye = run_hawkeye_review(
                    sidecar,
                    fps=fps,
                    court_points=config.get("court_points", []),
                    force_review=bool(config.get("continue_after_scene_cuts", False)),
                    video_path=source,
                )
                if self._cancelled(job_id):
                    raise JobCancelled()

            coaching: Dict[str, object] = {}
            stroke_types: Dict[str, object] = {}
            if config.get("professional_analysis"):
                # Run BST first so Motion Coach can use a validated type hint
                # while its original pose/trajectory evidence is still in
                # memory.  BST remains optional and isolated from core jobs.
                self._update(job_id, progress=95, stage="BST 击球类型识别")
                try:
                    stroke_types = run_bst_stroke_inference(
                        source,
                        sidecar,
                        fps=fps,
                        pose_weights=self.repo_root / "weights" / "yolov8s-pose.pt",
                        device=str(config.get("device", "auto")),
                        court_points=config.get("court_points", []),
                        repo_root=self.repo_root,
                    )
                except Exception:
                    # Keep the optional model fully isolated from the core
                    # report, and avoid persisting a traceback/path in the
                    # browser-facing manifest.
                    stroke_types = {
                        "version": 1,
                        "provider": "BST",
                        "status": "unavailable",
                        "notice": "BST 类型识别暂不可用，已保留规则型击球候选。",
                        "quality_notice": "BST 只识别击球类型，不判断动作好坏。",
                        "model": "BST",
                        "dataset": "",
                        "class_count": 0,
                        "total_hit_count": 0,
                        "predicted_hit_count": 0,
                        "hits": [],
                    }
                if self._cancelled(job_id):
                    raise JobCancelled()
                self._update(job_id, progress=96, stage="生成动作改进建议")
                coaching = run_pose_coaching(
                    source,
                    sidecar,
                    fps=fps,
                    pose_weights=self.repo_root / "weights" / "yolov8s-pose.pt",
                    device=str(config.get("device", "auto")),
                    court_points=config.get("court_points", []),
                    coach_target=str(config.get("coach_target", "near")),
                    near_dominant_hand=str(config.get("near_dominant_hand", "auto")),
                    far_dominant_hand=str(config.get("far_dominant_hand", "auto")),
                    stroke_type_report=stroke_types,
                )
                if self._cancelled(job_id):
                    raise JobCancelled()
                stroke_types = attach_rule_candidates(stroke_types, coaching)
            outputs: Dict[str, Dict[str, object]] = {}
            if analysis_video is not None and analysis_video.is_file():
                outputs["analysis"] = self._output_record(
                    analysis_video, "专业分析视频", "video"
                )
            sidecar_label = (
                "专业逐帧分析数据"
                if config.get("professional_analysis")
                else "轻量回合索引 CSV（部分击球事件字段有限）"
            )
            outputs["sidecar"] = self._output_record(sidecar, sidecar_label, "csv")

            rallies: List[Dict[str, object]] = []
            if config.get("split_rallies"):
                # If the optional post-process reports have already run, do
                # not send SSE progress backwards before clip export.
                clip_start = (
                    97 if config.get("professional_analysis")
                    else 94 if config.get("hawkeye_review")
                    else 91
                )
                self._update(job_id, progress=clip_start, stage="生成回合短视频")
                duration = _safe_float(
                    dict(dict(job["video"]).get("metadata", {}) or {}).get("duration"),
                    0.0,
                ) or None
                segments = read_rally_segments(
                    sidecar,
                    fps,
                    pre_roll_seconds=0.8,
                    post_roll_seconds=1.0,
                    total_duration_seconds=duration,
                )
                clips_dir = job_root / "rallies"

                def on_clip(position, total, _segment):
                    progress = clip_start + int(
                        (99 - clip_start) * position / max(1, total)
                    )
                    self._update(
                        job_id,
                        progress=min(98, progress),
                        stage=f"正在导出回合 {position}/{total}",
                    )
                    if self._cancelled(job_id):
                        raise JobCancelled()

                def register_clip_process(process: Optional[subprocess.Popen]) -> None:
                    with self._lock:
                        if process is None:
                            self._active_process.pop(job_id, None)
                        else:
                            self._active_process[job_id] = process

                try:
                    rallies = cut_rallies(
                        rally_clip_source,
                        clips_dir,
                        segments,
                        progress_callback=on_clip,
                        cancel_requested=lambda: self._cancelled(job_id),
                        process_callback=register_clip_process,
                    )
                except RallyCutCancelled as exc:
                    raise JobCancelled() from exc
                for rally in rallies:
                    key = f"rally_{int(rally['index']):03d}"
                    outputs[key] = self._output_record(
                        Path(str(rally["path"])),
                        f"回合 {int(rally['index']):02d}",
                        "video",
                    )

            if self._cancelled(job_id):
                raise JobCancelled()
            self._update(
                job_id,
                status="completed",
                progress=100,
                stage="处理完成",
                outputs=outputs,
                rallies=rallies,
                stats=stats,
                coaching=coaching,
                stroke_types=stroke_types,
                hawkeye=hawkeye,
            )
        except JobCancelled:
            self._update(job_id, status="cancelled", stage="任务已取消")
        except Exception as exc:
            self._log(job_id, traceback.format_exc(limit=8))
            self._update(
                job_id,
                status="failed",
                stage="处理失败",
                error=str(exc),
            )

    def _run_professional_pipeline(
        self,
        job_id: str,
        source: Path,
        work_root: Path,
        court_points: str,
        device: str,
        continue_after_scene_cuts: bool,
        bullet_time_enabled: Optional[bool] = None,
    ) -> tuple[Path, Path, Path]:
        self._update(job_id, progress=5, stage="启动专业分析 Pipeline")
        command = [
            "bash",
            str(self.repo_root / "run_all_mac.sh"),
            "--input-video", str(source),
            "--work-root", str(work_root),
            "--court-points", court_points,
            "--python", self.python_bin,
        ]
        # New callers pass an explicit value from the UI.  ``None`` retains
        # the pre-existing environment opt-in for older command/API clients.
        if bullet_time_enabled is None:
            bullet_time_enabled = _env_bullet_time_enabled()
        elif not isinstance(bullet_time_enabled, bool):
            raise ValueError("子弹镜头开关必须是布尔值")
        command.append(
            "--enable-bullet-time-fx"
            if bullet_time_enabled
            else "--no-bullet-time-fx"
        )
        if device != "auto":
            command.extend(["--yolo-device", device, "--tracknet-device", device])
        if continue_after_scene_cuts:
            command.append("--continue-ball-after-scene-cut")
        stage_markers = {
            "[STEP 1/3]": (8, "TrackNet 羽球定位"),
            "[STEP 1b/3]": (30, "TrackNet 双相位复核"),
            "[STEP 2/3]": (52, "球员姿态与专业分析"),
            "[STEP 3/3]": (
                84,
                "生成视觉特效" if bullet_time_enabled else "整理普通速度分析视频",
            ),
            "[DONE]": (90, "整理分析产物"),
        }
        self._run_command(
            job_id,
            command,
            stage_markers=stage_markers,
            progress_windows={
                "[STEP 1/3]": (8, 30),
                "[STEP 1b/3]": (30, 52),
                "[STEP 2/3]": (52, 84),
                "[STEP 3/3]": (84, 90),
                "[DONE]": (90, 94),
            },
        )
        sidecar = work_root / "end1_ball_tracking_official_fused_ball_tracking.csv"
        normal_speed_candidates = [
            work_root / "end1_ball_tracking_official_fused_h264.mp4",
            work_root / "end1_ball_tracking_official_fused.mp4",
        ]
        fx_candidates = [
            work_root / "end1_ball_tracking_official_fused_fx_h264.mp4",
            work_root / "end1_ball_tracking_official_fused_fx.mp4",
        ]
        # Prefer the artifact matching the selected mode.  The shell script
        # still emits the historical *_fx.mp4 alias when FX is disabled, so
        # older consumers remain compatible while the UI receives normal
        # speed footage by default.
        analysis_candidates = (
            [*fx_candidates, *normal_speed_candidates]
            if bullet_time_enabled
            else [*normal_speed_candidates, *fx_candidates]
        )
        clip_candidates = [
            work_root / "end1_ball_tracking_official_fused_h264.mp4",
            work_root / "end1_ball_tracking_official_fused.mp4",
        ]
        analysis = next(
            (path for path in analysis_candidates if path.is_file()),
            analysis_candidates[-1],
        )
        clip_source = next(
            (path for path in clip_candidates if path.is_file()),
            clip_candidates[-1],
        )
        if not sidecar.is_file() or not analysis.is_file() or not clip_source.is_file():
            raise RuntimeError("专业分析完成，但缺少视频或 sidecar 产物")
        return sidecar, analysis, clip_source

    def _run_rally_index_pipeline(
        self,
        job_id: str,
        source: Path,
        work_root: Path,
        court_points: str,
        device: str,
        continue_after_scene_cuts: bool,
    ) -> Path:
        weight = self.repo_root / "weights" / "TrackNet_official_best.pt"
        if not weight.is_file():
            weight = self.repo_root / "weights" / "TrackNet_best.pt"
        if not weight.is_file():
            raise RuntimeError("TrackNet 权重不存在")
        primary_dir = work_root / "tracknet_official_result"
        secondary_dir = work_root / "tracknet_official_result_phase_half"
        primary_dir.mkdir(parents=True, exist_ok=True)
        secondary_dir.mkdir(parents=True, exist_ok=True)
        stem = source.stem
        primary_csv = primary_dir / f"{stem}_ball.csv"
        secondary_csv = secondary_dir / f"{stem}_ball.csv"
        predict = self.repo_root / "scripts" / "tracknet_runtime" / "predict.py"
        common = [
            self.python_bin,
            str(predict),
            "--video_file", str(source),
            "--tracknet_file", str(weight),
            "--device", device,
            "--large_video",
            "--batch_size", "4",
            "--max_sample_num", "300",
            "--eval_mode", "nonoverlap",
            "--tracknet-court-top-extension", "480",
            "--tracknet-court-points", court_points,
        ]
        environment = dict(os.environ)
        environment["TRACKNET_VIS_THRESH"] = "0.20"
        self._update(job_id, progress=8, stage="基础球路分析：主相位")
        self._run_command(
            job_id,
            common + ["--save_dir", str(primary_dir)],
            environment=environment,
            progress_window=(8, 38),
        )
        offset = self._tracknet_half_offset(weight)
        self._update(job_id, progress=38, stage="基础球路分析：复核相位")
        self._run_command(
            job_id,
            common + [
                "--save_dir", str(secondary_dir),
                "--nonoverlap-offset", str(offset),
                "--tracknet-court-side-padding", "40",
                "--tracknet-court-bottom-padding", "30",
            ],
            environment=environment,
            progress_window=(38, 68),
        )
        if not primary_csv.is_file() or not secondary_csv.is_file():
            raise RuntimeError("基础球路分析未生成双相位 CSV")

        self._update(job_id, progress=68, stage="构建回合时间轴")
        overlay = self.repo_root / "scripts" / "overlay" / "overlay_player_analytics.py"
        index_video = work_root / "rally_index.mp4"
        command = [
            self.python_bin,
            str(overlay),
            "--video_path", str(source),
            "--output_path", str(index_video),
            "--ball_csv", str(primary_csv),
            "--ball_csv_secondary", str(secondary_csv),
            "--yolo_model", str(self.repo_root / "weights" / "yolov8s-pose.pt"),
            "--tracker_cfg", "bytetrack.yaml",
            "--skip_player_detector",
            "--no_draw_pose",
            "--no_draw_player_heatmap",
            "--no_audio_hit_detection",
            "--no_select_court_points",
            "--court_points", court_points,
            "--ball_top_extension", "480",
            "--max_frames", "1",
        ]
        command.append(
            "--continue_ball_after_scene_cut"
            if continue_after_scene_cuts
            else "--stop_ball_at_first_scene_cut"
        )
        self._run_command(job_id, command, progress_window=(68, 90))
        sidecar = work_root / "rally_index_ball_tracking.csv"
        if not sidecar.is_file():
            raise RuntimeError("回合时间轴 sidecar 未生成")
        return sidecar

    def _tracknet_half_offset(self, weight: Path) -> int:
        try:
            import torch

            checkpoint = torch.load(str(weight), map_location="cpu", weights_only=False)
            sequence_length = int(checkpoint["param_dict"]["seq_len"])
            return max(1, sequence_length // 2)
        except Exception as exc:
            raise RuntimeError(f"无法读取 TrackNet 序列长度: {exc}") from exc

    @staticmethod
    def _output_record(path: Path, label: str, media_type: str) -> Dict[str, object]:
        return {
            "path": str(path.resolve()),
            "filename": path.name,
            "label": label,
            "type": media_type,
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        }

    def _run_command(
        self,
        job_id: str,
        command: Sequence[str],
        *,
        environment: Optional[Mapping[str, str]] = None,
        stage_markers: Optional[Mapping[str, tuple[int, str]]] = None,
        progress_window: Optional[tuple[int, int]] = None,
        progress_windows: Optional[Mapping[str, tuple[int, int]]] = None,
    ) -> None:
        if self._cancelled(job_id):
            raise JobCancelled()
        env = dict(environment or os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        self._log(job_id, "$ " + " ".join(str(part) for part in command))
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        with self._lock:
            self._active_process[job_id] = process
        try:
            assert process.stdout is not None
            active_window = progress_window
            for line in process.stdout:
                if self._cancelled(job_id):
                    self._terminate_process(process)
                    raise JobCancelled()
                message = line.strip()
                if message:
                    self._log(job_id, message)
                    for marker, (progress, stage) in (stage_markers or {}).items():
                        if marker in message:
                            self._update(job_id, progress=progress, stage=stage)
                            active_window = (progress_windows or {}).get(marker, active_window)
                            break
                    # TrackNet/tqdm and the overlay scripts expose progress as
                    # ``N/M`` progress lines.  Interpolate
                    # within the current stage so SSE consumers see movement
                    # during long inference passes instead of only at stage
                    # boundaries.  Restrict parsing to progress-like lines to
                    # avoid mistaking arbitrary log ratios for percentages.
                    if active_window is not None and (
                        "progress" in message.lower()
                        or ("%" in message and "|" in message)
                    ):
                        match = _PROGRESS_RE.search(message)
                        if match:
                            current = int(match.group("current"))
                            total = int(match.group("total"))
                            if total > 0 and 0 <= current <= total:
                                start, end = active_window
                                fraction = current / float(total)
                                candidate = int(round(start + fraction * (end - start)))
                                with self._lock:
                                    previous_progress = int(
                                        self._jobs[job_id].get("progress", 0)
                                    )
                                if candidate > previous_progress:
                                    self._update(
                                        job_id,
                                        progress=min(99, candidate),
                                    )
            return_code = process.wait()
        finally:
            with self._lock:
                self._active_process.pop(job_id, None)
        if self._cancelled(job_id):
            raise JobCancelled()
        if return_code != 0:
            raise RuntimeError(f"Pipeline 子进程退出码 {return_code}")

    @staticmethod
    def _terminate_process(process: subprocess.Popen) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=8)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()


__all__ = ["JobManager", "TERMINAL_STATES", "summarize_sidecar"]
