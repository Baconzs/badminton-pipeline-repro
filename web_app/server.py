"""Flask server for the ShuttleVision local Web workbench."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import argparse
import json
import math
import mimetypes
import os
import re
import time
import uuid

import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_file,
    stream_with_context,
)
from werkzeug.utils import secure_filename

from .pipeline_service import JobManager, TERMINAL_STATES
from .bst_adapter import public_bst_configuration
from .rally_cutter import resolve_ffmpeg


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024**3
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w])(?:\"(?:/[^\"]+|[A-Za-z]:[\\/][^\"]+)\"|"
    r"'(?:/[^']+|[A-Za-z]:[\\/][^']+)'|"
    r"/[A-Za-z0-9._+@%~=,:-]+(?:/[A-Za-z0-9._+@%~=,:-]+)*|"
    r"[A-Za-z]:[\\/][^\s]+)"
)


class UploadTooLarge(ValueError):
    """Raised when a streamed upload exceeds the configured byte limit."""


def _inside(path: Path, roots: Iterable[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _video_metadata(path: Path) -> Dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("文件无法作为视频解码")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError("无法读取视频分辨率、帧率或时长")
        return {
            "fps": round(fps, 4),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": round(frame_count / fps, 3),
            "size_bytes": path.stat().st_size,
        }
    finally:
        capture.release()


def _write_poster(video_path: Path, poster_path: Path, metadata: Mapping[str, object]) -> None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(metadata.get("fps", 25.0) or 25.0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(fps * 0.75))))
        ok, image = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = capture.read()
        if not ok:
            raise ValueError("无法提取视频封面")
        maximum_width = 1280
        if image.shape[1] > maximum_width:
            scale = maximum_width / float(image.shape[1])
            image = cv2.resize(
                image,
                (maximum_width, max(1, int(round(image.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        poster_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(poster_path), image):
            raise ValueError("无法写入视频封面")
    finally:
        capture.release()


class VideoLibrary:
    def __init__(
        self,
        data_root: Path,
        allowed_roots: Iterable[Path],
        *,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    ) -> None:
        self.data_root = data_root.resolve()
        self.manifest_root = self.data_root / "videos"
        self.upload_root = self.data_root / "uploads"
        self.allowed_roots = [root.resolve() for root in allowed_roots]
        self.max_upload_bytes = max(1, int(max_upload_bytes))
        self.manifest_root.mkdir(parents=True, exist_ok=True)
        self.upload_root.mkdir(parents=True, exist_ok=True)
        # Imported files are restricted to configured roots; generated uploads
        # are restricted to this application's own upload directory.  Keeping
        # this list in one place also makes manifest validation defensive
        # against a hand-edited JSON record.
        self.source_roots = [*self.allowed_roots, self.upload_root.resolve()]

    def _save(self, record: Mapping[str, object]) -> Dict[str, object]:
        video_id = str(record["id"])
        target = self.manifest_root / f"{video_id}.json"
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(dict(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(target)
        return dict(record)

    def get(self, video_id: str) -> Dict[str, object]:
        normalized_id = str(video_id)
        if (
            not normalized_id
            or len(normalized_id) > 64
            or any(character not in "0123456789abcdef" for character in normalized_id)
        ):
            raise KeyError(video_id)
        path = self.manifest_root / f"{normalized_id}.json"
        if not path.is_file():
            raise KeyError(normalized_id)
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise KeyError(normalized_id)
        if str(record.get("id", "")) != normalized_id:
            raise KeyError(normalized_id)
        source = Path(str(record.get("path", ""))).resolve()
        if not source.is_file() or not _inside(source, self.source_roots):
            raise FileNotFoundError(source)
        return record

    def list(self, limit: int = 16):
        records = []
        manifests = sorted(
            self.manifest_root.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in manifests[: max(1, int(limit))]:
            try:
                records.append(self.get(path.stem))
            except Exception:
                continue
        return records

    def import_path(self, raw_path: str) -> Dict[str, object]:
        path = Path(str(raw_path or "")).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("请选择受支持的视频文件")
        if not _inside(path, self.allowed_roots):
            raise ValueError("服务器路径必须位于允许目录")
        return self._register(path, path.name, uploaded=False)

    def upload(self, storage) -> Dict[str, object]:
        # ``secure_filename`` strips non-ASCII characters and can turn a
        # perfectly valid name such as ``羽毛球比赛.mp4`` into ``mp4`` (losing
        # the extension).  Resolve the extension from the raw basename first,
        # then use a sanitized/fallback display name; the actual stored path
        # remains UUID-based and never depends on this input.
        raw_name = str(storage.filename or "")
        if len(raw_name) > 255:
            raise ValueError("文件名过长")
        raw_suffix = Path(raw_name.replace("\\", "/")).suffix.lower()
        original_name = secure_filename(raw_name)
        suffix = raw_suffix
        if suffix not in VIDEO_EXTENSIONS:
            raise ValueError("支持 MP4、MOV、MKV、AVI、WEBM、M4V 视频")
        if not original_name or Path(original_name).suffix.lower() != suffix:
            original_name = f"upload{suffix}"
        video_id = uuid.uuid4().hex[:16]
        directory = self.upload_root / video_id
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"source{suffix}"
        temporary = directory / f"source{suffix}.part"
        try:
            written = 0
            with temporary.open("wb") as stream:
                while True:
                    chunk = storage.stream.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > self.max_upload_bytes:
                        raise UploadTooLarge("视频超过服务器允许的上传大小")
                    stream.write(chunk)
            if written <= 0:
                raise ValueError("上传文件为空")
            temporary.replace(destination)
            return self._register(destination, original_name, uploaded=True, video_id=video_id)
        except Exception:
            # Do not leave an invalid/oversized upload consuming the runtime
            # volume.  A failed upload is never registered as a playable source.
            for failed in (temporary, destination, self.manifest_root / f"{video_id}.jpg"):
                try:
                    failed.unlink()
                except FileNotFoundError:
                    pass
            try:
                directory.rmdir()
            except OSError:
                pass
            raise

    def _register(
        self,
        path: Path,
        name: str,
        *,
        uploaded: bool,
        video_id: Optional[str] = None,
    ) -> Dict[str, object]:
        identifier = video_id or uuid.uuid4().hex[:16]
        path = path.resolve(strict=True)
        if not _inside(path, self.source_roots):
            raise ValueError("视频路径不在允许目录内")
        metadata = _video_metadata(path)
        poster = self.manifest_root / f"{identifier}.jpg"
        _write_poster(path, poster, metadata)
        record = {
            "id": identifier,
            "name": name or path.name,
            "path": str(path.resolve()),
            "uploaded": bool(uploaded),
            "metadata": metadata,
            "poster_path": str(poster.resolve()),
            "created_at": time.time(),
        }
        return self._save(record)


def _public_video(record: Mapping[str, object]) -> Dict[str, object]:
    result = {
        key: value for key, value in dict(record).items()
        if key not in {"path", "poster_path"}
    }
    video_id = str(record["id"])
    result["content_url"] = f"/api/videos/{video_id}/content"
    result["poster_url"] = f"/api/videos/{video_id}/poster"
    return result


def _public_coaching(raw: object) -> Dict[str, object]:
    """Expose a deliberately narrow, path-free coaching report schema.

    Job manifests persist across server restarts and can be edited locally, so
    this endpoint boundary must not pass arbitrary strings/objects from a
    manifest into the browser.
    """
    report = dict(raw) if isinstance(raw, Mapping) else {}
    status = str(report.get("status", ""))
    if status not in {"ready", "insufficient_evidence", "unavailable"}:
        return {}
    try:
        report_version = int(report.get("version", 1))
    except (TypeError, ValueError):
        report_version = 1
    result: Dict[str, object] = {
        "version": 2 if report_version >= 2 else 1,
        "status": status,
        "notice": str(report.get("notice", ""))[:520],
        "players": [],
    }
    players = report.get("players", [])
    if not isinstance(players, (list, tuple)):
        return result
    safe_players = []
    for raw_player in players[:2]:
        if not isinstance(raw_player, Mapping):
            continue
        player_id = str(raw_player.get("id", ""))
        if player_id not in {"near", "far"}:
            continue
        try:
            score = max(0, min(100, int(raw_player.get("score", 0))))
            sample_count = max(0, min(999, int(raw_player.get("sample_count", 0))))
        except (TypeError, ValueError):
            continue
        recommendations = []
        raw_recommendations = raw_player.get("recommendations", [])
        if not isinstance(raw_recommendations, (list, tuple)):
            raw_recommendations = []
        # Scan a bounded set, then cap the *accepted* output.  Otherwise a few
        # suppressed/candidate records at the front of a hand-edited manifest
        # could hide a later reliable recommendation.
        for raw_recommendation in raw_recommendations[:24]:
            if len(recommendations) >= 3:
                break
            if not isinstance(raw_recommendation, Mapping):
                continue
            priority = str(raw_recommendation.get("priority", ""))
            if priority not in {"high", "medium", "good"}:
                continue
            if report_version >= 2 and priority not in {"high", "medium"}:
                continue
            # Version 2 has a deliberately selective product contract.  Per-
            # stroke candidates and suppressed diagnostics stay in the local
            # manifest; only a repeated, explicitly reliable recommendation
            # may cross the HTTP boundary.
            recommendation_status = str(raw_recommendation.get("status", ""))
            recommendation_family = str(raw_recommendation.get("family", ""))
            recommendation_signal = str(raw_recommendation.get("signal", ""))
            problem = str(raw_recommendation.get("problem", "")).strip()
            action = str(raw_recommendation.get("action", "")).strip()
            repeat_count = 0
            if report_version >= 2:
                allowed_recommendation_families = {
                    "serve", "push", "lift", "cross_net", "net", "net_kill",
                    "clear", "drive_clear", "drop", "smash", "defensive_drive",
                    "preparation", "overhead", "frontcourt", "backcourt", "recovery",
                }
                allowed_recommendation_signals = {
                    "serve_preparation", "serve_first_step", "front_support",
                    "front_contact_space", "overhead_extension", "back_support",
                    "preparation_support", "overhead_preparation", "recovery_stability",
                }
                recommendation_signal_families = {
                    "serve_preparation": {"serve"},
                    "serve_first_step": {"serve"},
                    "front_support": {"push", "lift", "cross_net", "net", "net_kill", "frontcourt"},
                    "front_contact_space": {"push", "lift", "cross_net", "net", "net_kill", "frontcourt"},
                    "overhead_extension": {"clear", "drive_clear", "drop", "smash", "overhead"},
                    "back_support": {"clear", "drive_clear", "drop", "smash", "defensive_drive", "backcourt"},
                    "preparation_support": {"preparation"},
                    "overhead_preparation": {"overhead"},
                    "recovery_stability": {"recovery"},
                }
                uncertain_copy = re.compile(
                    r"待确认|待复核|信息不足|不确定|疑似|可能|或许|候选|需要确认|建议复核|请复核"
                )
                try:
                    repeat_count = max(0, min(999, int(raw_recommendation.get("repeat_count", 0))))
                except (TypeError, ValueError):
                    continue
                if (
                    recommendation_status != "reliable"
                    or recommendation_family not in allowed_recommendation_families
                    or recommendation_signal not in allowed_recommendation_signals
                    or recommendation_family not in recommendation_signal_families.get(recommendation_signal, set())
                    or repeat_count < 3
                    or not problem
                    or not action
                    or uncertain_copy.search(f"{problem} {action}") is not None
                ):
                    continue
            evidence = []
            raw_evidence = raw_recommendation.get("evidence", [])
            if not isinstance(raw_evidence, (list, tuple)):
                raw_evidence = []
            for raw_item in raw_evidence[:3]:
                if not isinstance(raw_item, Mapping):
                    continue
                try:
                    frame = max(0, int(raw_item.get("frame", 0)))
                    seconds = max(0.0, min(86400.0, float(raw_item.get("seconds", 0))))
                except (TypeError, ValueError):
                    continue
                timecode = str(raw_item.get("timecode", ""))
                if not re.fullmatch(r"\d{2,3}:\d{2}", timecode):
                    continue
                item = {"frame": frame, "seconds": round(seconds, 2), "timecode": timecode}
                # Version 2 cards seek the original video over a bounded
                # pre/post window.  Old reports simply omit these fields.
                try:
                    pre_seconds = max(0.0, min(5.0, float(raw_item.get("pre_seconds", 0))))
                    post_seconds = max(0.0, min(5.0, float(raw_item.get("post_seconds", 0))))
                    window_start = max(0.0, min(86400.0, float(raw_item.get("window_start", seconds - pre_seconds))))
                    window_end = max(window_start, min(86400.0, float(raw_item.get("window_end", seconds + post_seconds))))
                    real_points = max(0, min(999, int(raw_item.get("real_points", 0))))
                except (TypeError, ValueError):
                    pre_seconds = post_seconds = 0.0
                    window_start = window_end = seconds
                    real_points = 0
                if report_version >= 2:
                    item.update({
                        "pre_seconds": round(pre_seconds, 2),
                        "post_seconds": round(post_seconds, 2),
                        "window_start": round(window_start, 2),
                        "window_end": round(window_end, 2),
                        "real_points": real_points,
                    })
                evidence.append(item)
            public_recommendation = {
                "priority": priority,
                "title": str(raw_recommendation.get("title", ""))[:80],
                "detail": str(raw_recommendation.get("detail", ""))[:480],
                "metric": str(raw_recommendation.get("metric", ""))[:100],
                "evidence": evidence,
            }
            if report_version >= 2:
                public_recommendation.update({
                    "status": "reliable",
                    "family": recommendation_family,
                    "signal": recommendation_signal,
                    "problem": problem[:180],
                    "action": action[:220],
                    "repeat_count": repeat_count,
                })
            recommendations.append(public_recommendation)
        try:
            review_count = max(0, min(999, int(raw_player.get("review_count", 0))))
            gated_review_count = max(0, min(999, int(raw_player.get("gated_review_count", 0))))
            evidence_quality = max(0, min(100, int(raw_player.get("evidence_quality", score))))
        except (TypeError, ValueError):
            review_count = 0
            gated_review_count = 0
            evidence_quality = score
        stroke_summary = []
        raw_summary = raw_player.get("stroke_summary", [])
        if isinstance(raw_summary, (list, tuple)):
            allowed_families = {
                "serve", "push", "lift", "cross_net", "net", "net_kill",
                "clear", "drive_clear", "drop", "smash", "defensive_drive",
            }
            for raw_item in raw_summary[:12]:
                if not isinstance(raw_item, Mapping):
                    continue
                family = str(raw_item.get("family", ""))
                if family not in allowed_families:
                    continue
                try:
                    count = max(0, min(999, int(raw_item.get("count", 0))))
                    average_confidence = max(0, min(100, int(raw_item.get("average_confidence", 0))))
                except (TypeError, ValueError):
                    continue
                stroke_summary.append({
                    "family": family,
                    "label": str(raw_item.get("label", ""))[:64],
                    "count": count,
                    "average_confidence": average_confidence,
                })
        stroke_reviews = []
        raw_reviews = raw_player.get("stroke_reviews", [])
        if isinstance(raw_reviews, (list, tuple)):
            allowed_families = {
                "serve", "push", "lift", "cross_net", "net", "net_kill",
                "clear", "drive_clear", "drop", "smash", "defensive_drive",
                "front_unknown", "back_unknown", "unknown",
            }
            for raw_review in raw_reviews[:12]:
                if not isinstance(raw_review, Mapping):
                    continue
                raw_stroke = raw_review.get("stroke", {})
                raw_evaluation = raw_review.get("evaluation", {})
                raw_evidence = raw_review.get("evidence", {})
                if not isinstance(raw_stroke, Mapping) or not isinstance(raw_evaluation, Mapping) or not isinstance(raw_evidence, Mapping):
                    continue
                family = str(raw_stroke.get("family", ""))
                level = str(raw_stroke.get("confidence_level", ""))
                grade = str(raw_evaluation.get("grade", ""))
                if family not in allowed_families or level not in {"medium", "review"} or grade not in {"focus", "steady", "review"}:
                    continue
                try:
                    confidence = max(0, min(100, int(raw_stroke.get("confidence", 0))))
                    rally_id = max(0, min(999999, int(raw_review.get("rally_id", 0))))
                    hit_index = max(0, min(9999, int(raw_review.get("hit_index", 0))))
                    frame = max(0, int(raw_evidence.get("frame", 0)))
                    seconds = max(0.0, min(86400.0, float(raw_evidence.get("seconds", 0))))
                    pre_seconds = max(0.0, min(5.0, float(raw_evidence.get("pre_seconds", 0))))
                    post_seconds = max(0.0, min(5.0, float(raw_evidence.get("post_seconds", 0))))
                    window_start = max(0.0, min(86400.0, float(raw_evidence.get("window_start", seconds - pre_seconds))))
                    window_end = max(window_start, min(86400.0, float(raw_evidence.get("window_end", seconds + post_seconds))))
                    real_points = max(0, min(999, int(raw_evidence.get("real_points", 0))))
                except (TypeError, ValueError):
                    continue
                timecode = str(raw_evidence.get("timecode", ""))
                if not re.fullmatch(r"\d{2,3}:\d{2}", timecode):
                    continue
                def safe_texts(value: object, maximum: int = 4, length: int = 260):
                    if not isinstance(value, (list, tuple)):
                        return []
                    return [str(item)[:length] for item in value[:maximum] if isinstance(item, str)]
                public_classification = {}
                raw_classification = raw_review.get("classification")
                if isinstance(raw_classification, Mapping) and raw_classification:
                    source = str(raw_classification.get("source", "rule"))
                    if source not in {"bst", "rule"}:
                        source = "rule"
                    classification_status = str(raw_classification.get("status", "unavailable"))
                    if classification_status not in {
                        "adopted", "agreement", "rule_fallback", "low_confidence", "insufficient_evidence",
                        "conflict", "side_mismatch", "unsupported", "unavailable",
                    }:
                        classification_status = "unavailable"
                    classification_family = str(raw_classification.get("family", family))
                    if classification_family not in allowed_families:
                        classification_family = family
                    rule_family = str(raw_classification.get("rule_family", family))
                    if rule_family not in allowed_families:
                        rule_family = family
                    bst_family = str(raw_classification.get("bst_family", ""))
                    if bst_family not in allowed_families:
                        bst_family = ""
                    try:
                        rule_confidence = max(0, min(100, int(raw_classification.get("rule_confidence", confidence))))
                        bst_confidence = max(0.0, min(100.0, float(raw_classification.get("bst_confidence", 0))))
                    except (TypeError, ValueError):
                        rule_confidence, bst_confidence = confidence, 0.0
                    raw_gate = raw_classification.get("coaching_gate", {})
                    public_gate = {}
                    if isinstance(raw_gate, Mapping):
                        gate_source = str(raw_gate.get("source", ""))
                        reason_code = str(raw_gate.get("reason_code", ""))
                        allowed_gate_sources = {
                            "rule_only", "rule_fallback", "bst_blocked", "bst_agreement", "bst_adopted",
                        }
                        allowed_gate_reasons = {
                            "ready", "rule_ready", "rule_evidence_low",
                            "bst_type_confidence_low", "event_unconfirmed",
                            "bst_type_coverage_low", "side_mismatch",
                            "bst_family_unsupported", "type_conflict",
                            "route_type_unconfirmed", "bst_coach_confidence_low",
                            "bst_margin_missing", "bst_margin_low",
                            "bst_pose_coverage_low", "bst_ball_coverage_low",
                            "low_confidence", "margin_unavailable", "low_margin",
                            "low_pose_coverage", "low_ball_coverage",
                            "bst_adapter_gate_failed",
                        }
                        if gate_source in allowed_gate_sources and reason_code in allowed_gate_reasons:
                            try:
                                gate_confidence = max(0.0, min(100.0, float(raw_gate.get("bst_confidence", 0))))
                                raw_margin = raw_gate.get("bst_margin")
                                gate_margin = None if raw_margin is None else max(0.0, min(100.0, float(raw_margin)))
                                gate_pose = max(0.0, min(1.0, float(raw_gate.get("pose_coverage", 0))))
                                gate_ball = max(0.0, min(1.0, float(raw_gate.get("ball_coverage", 0))))
                            except (TypeError, ValueError):
                                public_gate = {}
                            else:
                                public_gate = {
                                    "eligible": raw_gate.get("eligible") is True,
                                    "reason_code": reason_code,
                                    "source": gate_source,
                                    "bst_confidence": round(gate_confidence, 1),
                                    "bst_margin": round(gate_margin, 1) if gate_margin is not None else None,
                                    "pose_coverage": round(gate_pose, 3),
                                    "ball_coverage": round(gate_ball, 3),
                                }
                    try:
                        raw_bst_margin = raw_classification.get("bst_margin")
                        bst_margin = None if raw_bst_margin is None else max(0.0, min(100.0, float(raw_bst_margin)))
                    except (TypeError, ValueError):
                        bst_margin = None
                    public_classification = {
                        "source": source,
                        "status": classification_status,
                        "family": classification_family,
                        "label": str(raw_classification.get("label", raw_stroke.get("label", "")))[:100],
                        "rule_family": rule_family,
                        "rule_label": str(raw_classification.get("rule_label", raw_stroke.get("label", "")))[:100],
                        "rule_confidence": rule_confidence,
                        "bst_family": bst_family,
                        "bst_label": str(raw_classification.get("bst_label", ""))[:100],
                        "bst_raw_label": str(raw_classification.get("bst_raw_label", ""))[:120],
                        "bst_confidence": round(bst_confidence, 1),
                        "bst_margin": round(bst_margin, 1) if bst_margin is not None else None,
                        "type_conflict": raw_classification.get("type_conflict") is True,
                        "advice_eligible": raw_classification.get("advice_eligible") is True,
                        "reason": str(raw_classification.get("reason", ""))[:220],
                        **({"coaching_gate": public_gate} if public_gate else {}),
                    }
                stroke_reviews.append({
                    "rally_id": rally_id,
                    "hit_index": hit_index,
                    "stroke": {
                        "family": family,
                        "label": str(raw_stroke.get("label", ""))[:100],
                        "base_label": str(raw_stroke.get("base_label", ""))[:64],
                        "area": str(raw_stroke.get("area", "unknown")) if str(raw_stroke.get("area", "unknown")) in {"serve", "front", "back", "unknown"} else "unknown",
                        "confidence": confidence,
                        "confidence_level": level,
                        "hand_label": str(raw_stroke.get("hand_label", ""))[:100],
                        "hand_status": str(raw_stroke.get("hand_status", "review")) if str(raw_stroke.get("hand_status", "review")) in {"candidate", "review"} else "review",
                        "reasons": safe_texts(raw_stroke.get("reasons"), 3, 180),
                    },
                    "evaluation": {
                        "grade": grade,
                        "title": str(raw_evaluation.get("title", ""))[:100],
                        "detail": str(raw_evaluation.get("detail", ""))[:480],
                        "observations": safe_texts(raw_evaluation.get("observations"), 4, 220),
                        # Version 2 treats every per-stroke prompt as an
                        # internal candidate.  Only player.recommendations can
                        # carry reliable advice across the public boundary.
                        "suggestions": [] if report_version >= 2 else safe_texts(raw_evaluation.get("suggestions"), 3, 280),
                        "caveat": str(raw_evaluation.get("caveat", ""))[:340],
                    },
                    "observed": safe_texts(raw_review.get("observed"), 4, 180),
                    "advice_eligible": raw_review.get("advice_eligible") is True,
                    "evidence": {
                        "frame": frame,
                        "seconds": round(seconds, 2),
                        "timecode": timecode,
                        "pre_seconds": round(pre_seconds, 2),
                        "post_seconds": round(post_seconds, 2),
                        "window_start": round(window_start, 2),
                        "window_end": round(window_end, 2),
                        "real_points": real_points,
                    },
                    **({"classification": public_classification} if public_classification else {}),
                })
        safe_players.append({
            "id": player_id,
            "label": "下方球员" if player_id == "near" else "上方球员",
            "score": score,
            "sample_count": sample_count,
            "review_count": review_count,
            "gated_review_count": gated_review_count,
            "evidence_quality": evidence_quality,
            "stroke_summary": stroke_summary,
            "stroke_reviews": stroke_reviews,
            "recommendations": recommendations,
        })
    result["players"] = safe_players
    return result


def _public_stroke_types(raw: object) -> Dict[str, object]:
    """Expose the optional BST report without leaking model/runtime paths.

    The adapter keeps its payload primitive and path-free already, but job
    manifests are local persistent state and may be edited by an operator.
    Keep this boundary strict just like ``_public_coaching``.
    """
    report = dict(raw) if isinstance(raw, Mapping) else {}
    status = str(report.get("status", ""))
    if status not in {"ready", "insufficient_evidence", "unavailable", "not_configured"}:
        return {}
    try:
        version = max(1, min(3, int(report.get("version", 1))))
        class_count = max(0, min(512, int(report.get("class_count", 0))))
        total_hits = max(0, min(1000000, int(report.get("total_hit_count", 0))))
        predicted_hits = max(0, min(total_hits, int(report.get("predicted_hit_count", 0))))
    except (TypeError, ValueError):
        version, class_count, total_hits, predicted_hits = 1, 0, 0, 0
    result: Dict[str, object] = {
        "version": version,
        "provider": "BST",
        "status": status,
        "notice": str(report.get("notice", ""))[:700],
        "quality_notice": str(report.get("quality_notice", ""))[:420],
        "model": str(report.get("model", ""))[:80],
        "dataset": str(report.get("dataset", ""))[:80],
        "class_count": class_count,
        "total_hit_count": total_hits,
        "predicted_hit_count": predicted_hits,
        "hits": [],
    }
    # Keep the research/type gate visible as a small numeric contract, while
    # never forwarding arbitrary adapter configuration or filesystem paths.
    raw_gate_policy = report.get("gate_policy")
    if isinstance(raw_gate_policy, Mapping):
        public_gate_policy = {"version": 1, "type_report": {}, "motion_coach": {}}
        for section in ("type_report", "motion_coach"):
            raw_section = raw_gate_policy.get(section)
            if not isinstance(raw_section, Mapping):
                continue
            section_values = {}
            for key in ("min_probability", "min_top1_top2_margin", "min_pose_coverage", "min_ball_coverage"):
                if key not in raw_section:
                    continue
                try:
                    value = float(raw_section.get(key))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    section_values[key] = round(max(0.0, min(100.0, value)), 4)
            public_gate_policy[section] = section_values
        result["gate_policy"] = public_gate_policy
    raw_hits = report.get("hits", [])
    if not isinstance(raw_hits, (list, tuple)):
        return result
    safe_hits = []
    for raw_hit in raw_hits[:2048]:
        if not isinstance(raw_hit, Mapping):
            continue
        try:
            frame = max(0, int(raw_hit.get("frame", 0)))
            seconds = max(0.0, min(86400.0, float(raw_hit.get("seconds", 0))))
            rally_id = max(0, min(1000000, int(raw_hit.get("rally_id", 0))))
            hit_index = max(0, min(100000, int(raw_hit.get("hit_index", 0))))
            pose_coverage = max(0.0, min(1.0, float(raw_hit.get("pose_coverage", 0))))
            ball_coverage = max(0.0, min(1.0, float(raw_hit.get("ball_coverage", 0))))
            confidence = max(0.0, min(100.0, float(raw_hit.get("confidence", 0))))
            window_start = max(0.0, min(86400.0, float(raw_hit.get("window_start", seconds))))
            window_end = max(window_start, min(86400.0, float(raw_hit.get("window_end", seconds))))
            top1_probability = max(0.0, min(1.0, float(raw_hit.get("top1_probability", confidence / 100.0))))
            raw_margin = raw_hit.get("top1_top2_margin")
            top1_top2_margin = None if raw_margin is None else max(0.0, min(1.0, float(raw_margin)))
        except (TypeError, ValueError):
            continue
        timecode = str(raw_hit.get("timecode", ""))
        if not re.fullmatch(r"\d{2,3}:\d{2}", timecode):
            continue
        hitter = str(raw_hit.get("hitter", ""))
        if hitter not in {"near", "far", "unknown"}:
            hitter = "unknown"
        model_side = str(raw_hit.get("model_side", ""))
        if model_side not in {"near", "far"}:
            model_side = ""
        level = str(raw_hit.get("confidence_level", "review"))
        if level not in {"ready", "review"}:
            level = "review"
        event_confirmed = raw_hit.get("event_confirmed") is True
        margin_available = (
            raw_hit.get("margin_available") is True
            if "margin_available" in raw_hit
            else top1_top2_margin is not None
        )
        type_ready = (
            raw_hit.get("type_ready") is True
            if "type_ready" in raw_hit
            else level == "ready"
        )
        coach_ready = raw_hit.get("coach_ready") is True
        type_gate_reason = str(raw_hit.get("type_gate_reason", ""))
        if type_gate_reason not in {
            "ready", "low_confidence", "event_unconfirmed",
            "low_pose_coverage", "low_ball_coverage",
        }:
            type_gate_reason = ""
        coach_gate_reason = str(raw_hit.get("coach_gate_reason", ""))
        if coach_gate_reason not in {
            "ready", "low_confidence", "margin_unavailable", "low_margin",
            "event_unconfirmed", "low_pose_coverage", "low_ball_coverage",
        }:
            coach_gate_reason = ""
        top3 = []
        raw_top3 = raw_hit.get("top3", [])
        if isinstance(raw_top3, (list, tuple)):
            for raw_top in raw_top3[:3]:
                if not isinstance(raw_top, Mapping):
                    continue
                try:
                    class_id = max(0, min(9999, int(raw_top.get("class_id", 0))))
                    probability = max(0.0, min(1.0, float(raw_top.get("probability", 0))))
                    top_confidence = max(0.0, min(100.0, float(raw_top.get("confidence", probability * 100))))
                except (TypeError, ValueError):
                    continue
                candidate_side = str(raw_top.get("model_side", ""))
                if candidate_side not in {"near", "far"}:
                    candidate_side = ""
                top3.append({
                    "class_id": class_id,
                    "raw_label": str(raw_top.get("raw_label", ""))[:100],
                    "label": str(raw_top.get("label", "待复核"))[:80],
                    "probability": round(probability, 4),
                    "confidence": round(top_confidence, 1),
                    "model_side": candidate_side,
                })
        safe_hits.append({
            "frame": frame,
            "seconds": round(seconds, 2),
            "timecode": timecode,
            "window_start": round(window_start, 2),
            "window_end": round(window_end, 2),
            "rally_id": rally_id,
            "hit_index": hit_index,
            "hitter": hitter,
            "event_confirmed": event_confirmed,
            "model_side": model_side,
            "model_label": str(raw_hit.get("model_label", "待复核"))[:80],
            "model_raw_label": str(raw_hit.get("model_raw_label", ""))[:100],
            "confidence": round(confidence, 1),
            "top1_probability": round(top1_probability, 4),
            "top1_top2_margin": round(top1_top2_margin, 4) if top1_top2_margin is not None else None,
            "margin_available": margin_available,
            "type_ready": type_ready,
            "type_gate_reason": type_gate_reason,
            "coach_ready": coach_ready,
            "coach_gate_reason": coach_gate_reason,
            "confidence_level": level,
            "pose_coverage": round(pose_coverage, 3),
            "ball_coverage": round(ball_coverage, 3),
            "review_reason": str(raw_hit.get("review_reason", ""))[:180],
            "rule_candidate": str(raw_hit.get("rule_candidate", ""))[:120],
            "top3": top3,
        })
    result["hits"] = safe_hits
    return result


_HAWKEYE_DIAGNOSTIC_CODES = frozenset({
    "no_rallies",
    "no_confirmed_hit",
    "no_post_hit_observation",
    "shot_continues_to_next_contact",
    "later_hit_unconfirmed",
    "discontinuous_terminal_track",
    "flight_too_short",
    "no_landing_evidence",
    "projection_failed",
    "landing_out_of_bounds",
    "wrong_target_half",
    "low_confidence",
    "invalid_calibration",
    "source_unavailable",
    "read_failed",
})


def _public_hawkeye(raw: object) -> Dict[str, object]:
    """Expose a compact, path-free single-camera landing-review report.

    The manifest is local persistence and may be hand edited, so the browser
    receives only a strict set of bounded primitive values.  In particular it
    never receives raw paths, arbitrary SVG/HTML, full CSV rows, or a legacy
    terminal-endpoint proxy masquerading as a landing.
    """
    report = dict(raw) if isinstance(raw, Mapping) else {}
    status = str(report.get("status", ""))
    if status not in {"ready", "insufficient_evidence", "unavailable"}:
        return {}
    result: Dict[str, object] = {
        "version": 1,
        "status": status,
        "notice": str(report.get("notice", ""))[:420],
        "court": {
            "width_m": 6.1,
            "length_m": 13.4,
            "line_margin_cm": 20,
        },
        "summary": {
            "reviewed_rallies": 0,
            "candidate_count": 0,
            "in_count": 0,
            "out_count": 0,
            "review_count": 0,
        },
        "candidates": [],
    }
    raw_candidates = report.get("candidates", [])
    if not isinstance(raw_candidates, (list, tuple)):
        return result
    legacy_proxy_only = bool(raw_candidates) and all(
        isinstance(item, Mapping)
        and str(item.get("method", "")) == "terminal_endpoint_proxy"
        for item in raw_candidates
    )
    candidates = []
    for raw_candidate in raw_candidates[:80]:
        if not isinstance(raw_candidate, Mapping):
            continue
        try:
            rally_id = int(raw_candidate.get("rally_id", 0))
            frame = int(raw_candidate.get("frame", -1))
            hit_frame = int(raw_candidate.get("hit_frame", -1))
            seconds = float(raw_candidate.get("seconds", -1))
            confidence = int(raw_candidate.get("confidence", -1))
            evidence_frames = int(raw_candidate.get("evidence_frames", 0))
            line_distance = float(raw_candidate.get("line_distance_cm", -1))
        except (TypeError, ValueError):
            continue
        method = str(raw_candidate.get("method", ""))
        recovered_method = method == "visible_terminal_rest_after_occlusion"
        landing = raw_candidate.get("landing", {})
        if not isinstance(landing, Mapping):
            landing = {}
        try:
            x_m = float(landing.get("x_m"))
            y_m = float(landing.get("y_m"))
        except (TypeError, ValueError):
            x_m = y_m = float("nan")
        if (
            not (1 <= rally_id <= 999999 and frame >= 0 and hit_frame >= 0)
            or not math.isfinite(seconds)
            or not (0.0 <= seconds <= 86400.0)
            or (not recovered_method and not all(math.isfinite(value) for value in (line_distance, x_m, y_m)))
            or (not recovered_method and not (0.0 <= line_distance <= 1000.0))
            or (recovered_method and not all(math.isfinite(value) for value in (x_m, y_m)))
            or (recovered_method and not (-3.0 <= x_m <= 9.1 and -3.0 <= y_m <= 16.4))
            or (not recovered_method and not (-3.0 <= x_m <= 9.1 and -3.0 <= y_m <= 16.4))
        ):
            continue
        timecode = str(raw_candidate.get("timecode", ""))
        hitter = str(raw_candidate.get("hitter", ""))
        decision = str(raw_candidate.get("decision", ""))
        level = str(raw_candidate.get("confidence_level", ""))
        boundary = str(raw_candidate.get("boundary", ""))
        call_hint = str(raw_candidate.get("call_hint", ""))
        if (
            not re.fullmatch(r"\d{2,3}:\d{2}", timecode)
            or hitter not in {"near", "far", "unknown"}
            or decision not in {"in", "out", "review"}
            # A terminal endpoint proxy was accepted by older manifests, but
            # it is not a measured landing and must never reach the browser.
            or method not in {"visible_rest", "visible_terminal_rest_after_occlusion"}
            or level not in {"high", "medium"}
            or (recovered_method and boundary not in {
                "far_baseline", "near_baseline", "left_sideline", "right_sideline",
            })
            or (recovered_method and decision not in {"out", "review"})
        ):
            continue
        safe_candidate = {
            "rally_id": rally_id,
            "frame": frame,
            "seconds": round(seconds, 2),
            "timecode": timecode,
            "hit_frame": hit_frame,
            "hitter": hitter,
            "decision": decision,
            "method": method,
            "confidence": max(0, min(100, confidence)),
            "confidence_level": level,
            "evidence_frames": max(0, min(99, evidence_frames)),
        }
        if math.isfinite(x_m) and math.isfinite(y_m):
            safe_candidate["landing"] = {"x_m": round(x_m, 3), "y_m": round(y_m, 3)}
        if math.isfinite(line_distance) and 0.0 <= line_distance <= 1000.0:
            safe_candidate["line_distance_cm"] = round(line_distance, 1)
        if recovered_method:
            safe_candidate["boundary"] = boundary
            safe_candidate["call_hint"] = call_hint if call_hint in {"out", "review"} else "out"
            raw_range = raw_candidate.get("line_distance_range_cm", [])
            if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
                try:
                    range_values = [float(raw_range[0]), float(raw_range[1])]
                except (TypeError, ValueError):
                    range_values = []
                if (
                    len(range_values) == 2
                    and all(math.isfinite(value) for value in range_values)
                    and 0.0 <= range_values[0] <= range_values[1] <= 1000.0
                ):
                    safe_candidate["line_distance_range_cm"] = [
                        round(range_values[0], 1), round(range_values[1], 1)
                    ]
            safe_candidate["occluded"] = True
        candidates.append(safe_candidate)
    result["candidates"] = candidates
    # Completed jobs persist their reports.  Older reports may contain only a
    # terminal proxy; hide that stale pseudo-result and expose the same safe
    # no-call contract as a newly processed job.
    if legacy_proxy_only and not candidates:
        result["status"] = "insufficient_evidence"
        result["notice"] = "未生成落点：未观察到可靠落地事件，请回看视频。"
    result["summary"] = {
        "reviewed_rallies": max(0, min(999999, int(_safe_public_int(report.get("summary"), "reviewed_rallies")))),
        "candidate_count": len(candidates),
        "in_count": sum(item["decision"] == "in" for item in candidates),
        "out_count": sum(item["decision"] == "out" for item in candidates),
        "review_count": sum(item["decision"] == "review" for item in candidates),
    }
    raw_diagnostics = report.get("diagnostics", {})
    if isinstance(raw_diagnostics, Mapping):
        safe_reasons = []
        seen_codes = set()
        if legacy_proxy_only and not candidates:
            safe_reasons.append({"code": "no_landing_evidence", "count": 1})
            seen_codes.add("no_landing_evidence")
        raw_reasons = raw_diagnostics.get("reason_counts", [])
        if isinstance(raw_reasons, (list, tuple)):
            for raw_reason in raw_reasons[:12]:
                if not isinstance(raw_reason, Mapping):
                    continue
                code = str(raw_reason.get("code", ""))
                if legacy_proxy_only and code in {
                    "landing_out_of_bounds",
                    "low_confidence",
                }:
                    continue
                if code not in _HAWKEYE_DIAGNOSTIC_CODES or code in seen_codes:
                    continue
                try:
                    count = int(raw_reason.get("count", 0))
                except (TypeError, ValueError):
                    continue
                if count <= 0:
                    continue
                safe_reasons.append({
                    "code": code,
                    "count": min(999999, count),
                })
                seen_codes.add(code)
        safe_diagnostics: Dict[str, object] = {}
        if safe_reasons:
            safe_diagnostics["reason_counts"] = safe_reasons
        for key in ("ignored_unconfirmed_hits", "fallback_candidate_count"):
            if legacy_proxy_only and key == "fallback_candidate_count":
                continue
            try:
                value = int(raw_diagnostics.get(key, 0))
            except (TypeError, ValueError):
                continue
            if value > 0:
                safe_diagnostics[key] = min(999999, value)
        if safe_diagnostics:
            result["diagnostics"] = safe_diagnostics
    return result


def _safe_public_int(raw: object, key: str) -> int:
    """Read one bounded-report counter without making a malformed manifest fatal."""
    if not isinstance(raw, Mapping):
        return 0
    try:
        return int(raw.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _public_job(job: Mapping[str, object]) -> Dict[str, object]:
    result = dict(job)

    private_paths = []
    raw_video_value = job.get("video", {})
    raw_video = (
        dict(raw_video_value)
        if isinstance(raw_video_value, Mapping)
        else {}
    )
    if raw_video.get("path"):
        private_paths.append(str(raw_video["path"]))
    raw_outputs_for_paths = job.get("outputs", {})
    if not isinstance(raw_outputs_for_paths, Mapping):
        raw_outputs_for_paths = {}
    for raw_output in raw_outputs_for_paths.values():
        if isinstance(raw_output, Mapping) and raw_output.get("path"):
            private_paths.append(str(raw_output["path"]))
    raw_rallies_for_paths = job.get("rallies", [])
    if not isinstance(raw_rallies_for_paths, (list, tuple)):
        raw_rallies_for_paths = []
    for raw_rally in raw_rallies_for_paths:
        if isinstance(raw_rally, Mapping) and raw_rally.get("path"):
            private_paths.append(str(raw_rally["path"]))
    private_paths.sort(key=len, reverse=True)

    def redact(value: object) -> str:
        # Pipeline logs contain absolute source/output paths.  They are useful
        # to the local operator but should not be disclosed verbatim if the
        # development server is accidentally bound to a LAN interface.
        text = str(value)
        for private_path in private_paths:
            text = text.replace(private_path, "<path>")
        return _ABSOLUTE_PATH_RE.sub("<path>", text)

    result["error"] = (
        redact(result.get("error")) if result.get("error") else None
    )
    raw_logs = result.get("logs", [])
    if not isinstance(raw_logs, (list, tuple)):
        raw_logs = []
    result["logs"] = [
        {
            **dict(entry),
            "message": redact(dict(entry).get("message", "")),
        }
        for entry in raw_logs
        if isinstance(entry, Mapping)
    ]
    job_id = str(result["id"])
    outputs = {}
    raw_outputs = result.get("outputs", {})
    if not isinstance(raw_outputs, Mapping):
        raw_outputs = {}
    for key, raw in raw_outputs.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_]+", key) or not isinstance(raw, Mapping):
            continue
        output = {item: value for item, value in dict(raw).items() if item != "path"}
        output["url"] = f"/api/jobs/{job_id}/files/{key}"
        outputs[key] = output
    result["outputs"] = outputs
    rallies = []
    raw_rallies = result.get("rallies", [])
    if not isinstance(raw_rallies, (list, tuple)):
        raw_rallies = []
    for raw in raw_rallies:
        if not isinstance(raw, Mapping) or "index" not in raw:
            continue
        try:
            index = int(raw["index"])
        except (TypeError, ValueError):
            continue
        if index < 1 or index > 999999:
            continue
        rally = {item: value for item, value in dict(raw).items() if item != "path"}
        rally["index"] = index
        key = f"rally_{index:03d}"
        rally["url"] = f"/api/jobs/{job_id}/files/{key}"
        rallies.append(rally)
    result["rallies"] = rallies
    result["coaching"] = _public_coaching(job.get("coaching"))
    result["stroke_types"] = _public_stroke_types(job.get("stroke_types"))
    result["hawkeye"] = _public_hawkeye(job.get("hawkeye"))
    raw_video_public = result.get("video", {})
    video = dict(raw_video_public) if isinstance(raw_video_public, Mapping) else {}
    video.pop("path", None)
    result["video"] = video
    return result


def create_app(
    *,
    repo_root: Optional[Path | str] = None,
    data_root: Optional[Path | str] = None,
    python_bin: Optional[Path | str] = None,
) -> Flask:
    repository = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    runtime = Path(
        data_root
        or os.environ.get("BADMINTON_UI_DATA_ROOT", repository / "runtime" / "web_ui")
    ).resolve()
    configured_roots = os.environ.get("BADMINTON_UI_INPUT_ROOTS", "")
    allowed_roots = [repository]
    if configured_roots:
        allowed_roots.extend(
            Path(value).expanduser().resolve()
            for value in configured_roots.split(os.pathsep)
            if value.strip()
        )

    app = Flask(
        __name__,
        static_folder=str(Path(__file__).resolve().parent / "static"),
        static_url_path="/static",
    )
    try:
        max_upload_bytes = int(
            os.environ.get("BADMINTON_UI_MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES))
        )
    except (TypeError, ValueError):
        max_upload_bytes = DEFAULT_MAX_UPLOAD_BYTES
    max_upload_bytes = max(1, max_upload_bytes)
    app.config["MAX_CONTENT_LENGTH"] = max_upload_bytes
    library = VideoLibrary(
        runtime,
        allowed_roots,
        max_upload_bytes=max_upload_bytes,
    )
    manager = JobManager(repository, runtime, python_bin=python_bin)
    app.extensions["video_library"] = library
    app.extensions["job_manager"] = manager

    @app.get("/")
    def index():
        index_path = Path(app.static_folder or "") / "index.html"
        if not index_path.is_file():
            # A missing frontend bundle is a deployment/configuration issue;
            # return a useful response instead of an opaque Flask 500.
            return jsonify({"error": "Web UI 首页资源不存在"}), 503
        return send_file(index_path)

    @app.get("/api/health")
    def health():
        accelerator = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                accelerator = f"cuda · {torch.cuda.get_device_name(0)}"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                accelerator = "mps"
        except Exception:
            pass
        weights = {
            "tracknet": any(
                (repository / "weights" / filename).is_file()
                for filename in ("TrackNet_official_best.pt", "TrackNet_best.pt")
            ),
            "pose": (repository / "weights" / "yolov8s-pose.pt").is_file(),
        }
        return jsonify({
            "ok": all(weights.values()),
            "accelerator": accelerator,
            "ffmpeg": bool(resolve_ffmpeg()),
            "weights": weights,
            "bst": public_bst_configuration(repository),
            "queue_mode": "single-worker",
            # Do not disclose the server's absolute filesystem layout from a
            # health endpoint that may be reachable over a LAN.
            "data_root": runtime.name,
        })

    @app.get("/api/videos")
    def list_videos():
        return jsonify([_public_video(record) for record in library.list()])

    @app.post("/api/videos/upload")
    def upload_video():
        storage = request.files.get("video")
        if storage is None or not storage.filename:
            return jsonify({"error": "请选择视频文件"}), 400
        try:
            return jsonify(_public_video(library.upload(storage))), 201
        except UploadTooLarge as exc:
            return jsonify({"error": str(exc)}), 413
        except (ValueError, OSError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/videos/import")
    def import_video():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "请求体必须是 JSON 对象"}), 400
        try:
            record = library.import_path(str(payload.get("path", "")))
            return jsonify(_public_video(record)), 201
        except FileNotFoundError:
            return jsonify({"error": "找不到可导入的视频文件"}), 400
        except PermissionError:
            return jsonify({"error": "没有权限读取该视频文件"}), 400
        except (ValueError, OSError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/videos/<video_id>/content")
    def video_content(video_id):
        try:
            record = library.get(video_id)
            path = Path(str(record["path"])).resolve()
        except (KeyError, FileNotFoundError, OSError, ValueError, TypeError):
            return jsonify({"error": "视频不存在"}), 404
        return send_file(
            path,
            conditional=True,
            mimetype=mimetypes.guess_type(path.name)[0] or "video/mp4",
        )

    @app.get("/api/videos/<video_id>/poster")
    def video_poster(video_id):
        try:
            record = library.get(video_id)
            path = Path(str(record["poster_path"])).resolve()
            if not path.is_file() or not _inside(path, [library.manifest_root]):
                raise FileNotFoundError(path)
        except (KeyError, FileNotFoundError, OSError, ValueError, TypeError):
            return jsonify({"error": "封面不存在"}), 404
        return send_file(path, conditional=True, mimetype="image/jpeg")

    @app.get("/api/jobs")
    def list_jobs():
        return jsonify([_public_job(job) for job in manager.list_jobs()])

    @app.post("/api/jobs")
    def create_job():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "请求体必须是 JSON 对象"}), 400
        try:
            split_value = payload.get("split_rallies", False)
            professional_value = payload.get("professional_analysis", False)
            hawkeye_value = payload.get("hawkeye_review", False)
            continue_scene_value = payload.get("continue_after_scene_cuts", False)
            # ``None`` keeps compatibility with older API clients: the job
            # manager can fall back to BADMINTON_ENABLE_BULLET_TIME_FX. New
            # UI clients send the canonical field, while the aliases remain
            # accepted when they agree.  Reject contradictory aliases instead
            # of silently changing the selected output.
            bullet_time_fields = [
                payload[key]
                for key in ("bullet_time_enabled", "bullet_time_fx", "bullet_time")
                if key in payload and payload[key] is not None
            ]
            if any(not isinstance(value, bool) for value in bullet_time_fields):
                raise ValueError("子弹镜头开关必须是布尔值")
            if len(set(bullet_time_fields)) > 1:
                raise ValueError("子弹镜头开关参数冲突")
            bullet_time_value = bullet_time_fields[0] if bullet_time_fields else None
            coach_target = payload.get("coach_target", "near")
            near_dominant_hand = payload.get("near_dominant_hand", "auto")
            far_dominant_hand = payload.get("far_dominant_hand", "auto")
            if (
                not isinstance(split_value, bool)
                or not isinstance(professional_value, bool)
                or not isinstance(hawkeye_value, bool)
                or not isinstance(continue_scene_value, bool)
                or (
                    bullet_time_value is not None
                    and not isinstance(bullet_time_value, bool)
                )
            ):
                raise ValueError("分析开关必须是布尔值")
            # Keep the HTTP contract self-contained: selecting FX always
            # requests the professional pass, even if an older client did
            # not mirror the browser's dependent toggle.
            professional_value = professional_value or bullet_time_value is True
            if (
                not isinstance(coach_target, str)
                or not isinstance(near_dominant_hand, str)
                or not isinstance(far_dominant_hand, str)
            ):
                raise ValueError("动作教练参数格式无效")
            video = library.get(str(payload.get("video_id", "")))
            job = manager.create_job(
                video,
                split_rallies=split_value,
                professional_analysis=professional_value,
                hawkeye_review=hawkeye_value,
                court_points=list(payload.get("court_points", []) or []),
                device=str(payload.get("device", "auto")),
                continue_after_scene_cuts=continue_scene_value,
                coach_target=coach_target,
                near_dominant_hand=near_dominant_hand,
                far_dominant_hand=far_dominant_hand,
                bullet_time_enabled=bullet_time_value,
            )
            return jsonify(_public_job(job)), 201
        except KeyError:
            return jsonify({"error": "视频不存在"}), 404
        except (ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        try:
            return jsonify(_public_job(manager.get_job(job_id)))
        except KeyError:
            return jsonify({"error": "任务不存在"}), 404

    @app.get("/api/jobs/<job_id>/events")
    def job_events(job_id):
        try:
            manager.get_job(job_id)
        except KeyError:
            return jsonify({"error": "任务不存在"}), 404

        @stream_with_context
        def generate():
            previous = None
            event_id = 0
            while True:
                try:
                    snapshot = manager.get_job(job_id)
                except KeyError:
                    return
                signature = str(snapshot.get("updated_at", ""))
                if signature != previous:
                    event_id += 1
                    payload = json.dumps(_public_job(snapshot), ensure_ascii=False)
                    yield f"id: {event_id}\ndata: {payload}\n\n"
                    previous = signature
                else:
                    yield ": heartbeat\n\n"
                if snapshot.get("status") in TERMINAL_STATES:
                    return
                time.sleep(1.0)

        response = Response(generate(), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel_job(job_id):
        try:
            return jsonify(_public_job(manager.cancel(job_id)))
        except KeyError:
            return jsonify({"error": "任务不存在"}), 404

    @app.get("/api/jobs/<job_id>/files/<key>")
    def job_file(job_id, key):
        try:
            path = manager.output_path(job_id, key)
        except (KeyError, FileNotFoundError):
            return jsonify({"error": "产物不存在"}), 404
        return send_file(
            path,
            conditional=True,
            as_attachment=False,
            download_name=path.name,
            mimetype=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "视频超过服务器允许的上传大小"}), 413

    return app


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="ShuttleVision Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--python", default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    app = create_app(
        data_root=args.data_root or None,
        python_bin=args.python or None,
    )
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
