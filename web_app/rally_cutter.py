"""Build and render rally clips from an overlay ball-tracking sidecar."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import csv
import json
import os
import signal
import shutil
import subprocess
import time


@dataclass(frozen=True)
class RallySegment:
    index: int
    rally_id: int
    start_frame: int
    end_frame: int
    trajectory_start_frame: int
    trajectory_end_frame: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    hit_count: int
    quality: str
    start_reason: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class RallyCutCancelled(RuntimeError):
    """Raised when a caller cancels while ffmpeg is rendering a clip."""


def _group_by_rally_id(rows: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    # Render-only top-edge bridge rows are deliberately exported with
    # RallyID=0.  Group by positive ID instead of contiguous non-zero spans,
    # otherwise one high clear leaving the image would split a real rally.
    grouped: Dict[int, Dict[str, object]] = {}
    for row in rows:
        try:
            frame = int(float(row.get("Frame", "0") or 0))
            rally_id = int(float(row.get("RallyID", "0") or 0))
            hit = int(float(row.get("Hit", "0") or 0)) > 0
        except (TypeError, ValueError):
            continue
        if rally_id <= 0:
            continue
        active = grouped.get(rally_id)
        if active is None:
            active = {
                "rally_id": rally_id,
                "start_frame": frame,
                "end_frame": frame,
                "hit_frames": [],
            }
            grouped[rally_id] = active
        else:
            active["start_frame"] = min(int(active["start_frame"]), frame)
            active["end_frame"] = max(int(active["end_frame"]), frame)
        if hit:
            active["hit_frames"].append(frame)
    return sorted(grouped.values(), key=lambda item: int(item["start_frame"]))


def read_rally_segments(
    sidecar_csv: Path | str,
    fps: float,
    *,
    pre_roll_seconds: float = 0.8,
    post_roll_seconds: float = 1.0,
    min_duration_seconds: float = 1.5,
    total_duration_seconds: Optional[float] = None,
) -> List[RallySegment]:
    """Convert contiguous positive RallyID spans into padded clip boundaries.

    The sidecar begins at the first measured shuttle point and ends at the
    last one.  A short pre/post roll turns that detector span into the more
    useful human concept of serve preparation through the dead-ball moment.
    Adjacent padded clips are split at their midpoint so exported clips never
    overlap or duplicate footage.
    """

    rate = max(1.0, float(fps))
    path = Path(sidecar_csv)
    with path.open("r", encoding="utf-8", newline="") as stream:
        groups = _group_by_rally_id(csv.DictReader(stream))

    provisional: List[Dict[str, float | int]] = []
    for group in groups:
        hit_frames = [int(frame) for frame in group.get("hit_frames", [])]
        logical_start_frame = (
            hit_frames[0] if hit_frames else int(group["start_frame"])
        )
        logical_end_frame = int(group["end_frame"])
        raw_start = float(logical_start_frame) / rate
        raw_end = float(logical_end_frame + 1) / rate
        start = max(0.0, raw_start - max(0.0, float(pre_roll_seconds)))
        end = raw_end + max(0.0, float(post_roll_seconds))
        if total_duration_seconds is not None:
            end = min(end, max(0.0, float(total_duration_seconds)))
        if end - start < max(0.0, float(min_duration_seconds)):
            continue
        provisional.append({
            "rally_id": int(group["rally_id"]),
            "start_frame": logical_start_frame,
            "end_frame": logical_end_frame,
            "trajectory_start_frame": int(group["start_frame"]),
            "trajectory_end_frame": int(group["end_frame"]),
            "hit_count": len(hit_frames),
            "quality": (
                "hit_anchored" if len(hit_frames) >= 2 else
                "single_hit" if hit_frames else
                "trajectory_fallback"
            ),
            "start_reason": "first_hit" if hit_frames else "first_trajectory",
            "start_seconds": start,
            "end_seconds": end,
        })

    for left, right in zip(provisional, provisional[1:]):
        if float(left["end_seconds"]) <= float(right["start_seconds"]):
            continue
        midpoint = 0.5 * (
            float(left["end_seconds"]) + float(right["start_seconds"])
        )
        left["end_seconds"] = midpoint
        right["start_seconds"] = midpoint

    segments: List[RallySegment] = []
    for index, item in enumerate(provisional, start=1):
        start = float(item["start_seconds"])
        end = float(item["end_seconds"])
        segments.append(RallySegment(
            index=index,
            rally_id=int(item["rally_id"]),
            start_frame=int(item["start_frame"]),
            end_frame=int(item["end_frame"]),
            trajectory_start_frame=int(item["trajectory_start_frame"]),
            trajectory_end_frame=int(item["trajectory_end_frame"]),
            start_seconds=round(start, 3),
            end_seconds=round(end, 3),
            duration_seconds=round(max(0.0, end - start), 3),
            hit_count=int(item["hit_count"]),
            quality=str(item["quality"]),
            start_reason=str(item["start_reason"]),
        ))
    return segments


def resolve_ffmpeg() -> Optional[str]:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    return executable or None


def cut_rallies(
    source_video: Path | str,
    output_dir: Path | str,
    segments: Iterable[RallySegment],
    *,
    ffmpeg_bin: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, RallySegment], None]] = None,
    cancel_requested: Optional[Callable[[], bool]] = None,
    process_callback: Optional[Callable[[Optional[subprocess.Popen]], None]] = None,
) -> List[Dict[str, object]]:
    """Render frame-accurate H.264/AAC rally clips and a JSON manifest."""

    executable = ffmpeg_bin or resolve_ffmpeg()
    if not executable:
        raise RuntimeError("ffmpeg is required for rally clipping")
    source = Path(source_video).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    items = list(segments)
    rendered: List[Dict[str, object]] = []
    total = len(items)
    for position, segment in enumerate(items, start=1):
        if cancel_requested and cancel_requested():
            raise RallyCutCancelled("Rally clipping cancelled")
        output = destination / f"rally_{segment.index:03d}.mp4"
        command = [
            executable,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", f"{segment.start_seconds:.3f}",
            "-i", str(source),
            "-t", f"{segment.duration_seconds:.3f}",
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "21",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        if process_callback:
            process_callback(process)
        try:
            while True:
                try:
                    detail, _ = process.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    if cancel_requested and cancel_requested():
                        _terminate_render_process(process)
                        raise RallyCutCancelled("Rally clipping cancelled")
                    # Avoid a busy loop while still responding to cancellation
                    # at sub-second granularity for a long re-encode.
                    time.sleep(0.02)
        finally:
            if process_callback:
                process_callback(None)
        if cancel_requested and cancel_requested():
            raise RallyCutCancelled("Rally clipping cancelled")
        if process.returncode != 0:
            detail = (detail or "").strip()
            raise RuntimeError(
                f"ffmpeg failed for rally {segment.index}: {detail[-1000:]}"
            )
        record = segment.to_dict()
        record.update({
            "filename": output.name,
            "path": str(output),
            "size_bytes": output.stat().st_size,
        })
        rendered.append(record)
        if progress_callback:
            progress_callback(position, total, segment)

    manifest = destination / "rallies.json"
    manifest.write_text(
        json.dumps({"rallies": rendered}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rendered


def _terminate_render_process(process: subprocess.Popen) -> None:
    """Terminate a dedicated ffmpeg process group without touching its parent."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=4)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


__all__ = [
    "RallySegment",
    "RallyCutCancelled",
    "cut_rallies",
    "read_rally_segments",
    "resolve_ffmpeg",
]
