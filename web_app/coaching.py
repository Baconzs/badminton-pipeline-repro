"""Evidence-led, local motion-coaching summaries for professional jobs.

The overlay pipeline intentionally keeps its CSV limited to ball and hit
provenance.  This module performs a small *post-process* pose pass around
already detected hits, rather than changing that pipeline or claiming that a
ball trajectory alone can diagnose a player's technique.

Reports are designed for the Web UI: every recommendation carries timestamped
evidence and is omitted when the required joints were not confidently visible.
They are training prompts, not medical or biomechanical diagnoses.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from .stroke_coaching import (
    classify_stroke,
    evaluate_stroke,
    extract_pose_metrics,
    select_bst_stroke_type,
    summarize_reviews,
    zone_for_player_position,
)


PLAYER_LABELS = {"near": "下方球员", "far": "上方球员"}
_MIN_SAMPLES = 2


def _number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _point(keypoints: Sequence[Sequence[object]], index: int):
    if index >= len(keypoints):
        return None
    point = keypoints[index]
    if not isinstance(point, (list, tuple)) or len(point) < 3 or _number(point[2]) < 0.28:
        return None
    return (_number(point[0]), _number(point[1]))


def _distance(first, second) -> float | None:
    if first is None or second is None:
        return None
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _midpoint(first, second):
    if first is None or second is None:
        return None
    return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)


def _angle(first, vertex, third) -> float | None:
    if first is None or vertex is None or third is None:
        return None
    left = (first[0] - vertex[0], first[1] - vertex[1])
    right = (third[0] - vertex[0], third[1] - vertex[1])
    denominator = math.hypot(*left) * math.hypot(*right)
    if denominator <= 1e-6:
        return None
    cosine = max(-1.0, min(1.0, (left[0] * right[0] + left[1] * right[1]) / denominator))
    return math.degrees(math.acos(cosine))


def _clip_score(value: float) -> int:
    return int(round(max(0.0, min(100.0, value))))


def _time_label(frame: int, fps: float) -> str:
    seconds = max(0.0, frame / max(1.0, fps))
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes:02d}:{remainder:02d}"


def _metrics(keypoints: Sequence[Sequence[object]]) -> dict[str, float]:
    """Calculate scale-normalized, view-dependent movement signals."""
    left_shoulder, right_shoulder = _point(keypoints, 5), _point(keypoints, 6)
    left_wrist, right_wrist = _point(keypoints, 9), _point(keypoints, 10)
    left_hip, right_hip = _point(keypoints, 11), _point(keypoints, 12)
    left_knee, right_knee = _point(keypoints, 13), _point(keypoints, 14)
    left_ankle, right_ankle = _point(keypoints, 15), _point(keypoints, 16)

    shoulder_center = _midpoint(left_shoulder, right_shoulder)
    hip_center = _midpoint(left_hip, right_hip)
    torso = _distance(shoulder_center, hip_center)
    result: dict[str, float] = {}
    if torso and torso > 4:
        reaches = [
            _distance(left_shoulder, left_wrist),
            _distance(right_shoulder, right_wrist),
        ]
        reaches = [reach / torso for reach in reaches if reach is not None]
        if reaches:
            result["reach"] = max(reaches)
    knees = [
        _angle(left_hip, left_knee, left_ankle),
        _angle(right_hip, right_knee, right_ankle),
    ]
    knees = [angle for angle in knees if angle is not None]
    if knees:
        result["knee_angle"] = min(knees)
    shoulder_width = _distance(left_shoulder, right_shoulder)
    ankle_width = _distance(left_ankle, right_ankle)
    if shoulder_width and shoulder_width > 4 and ankle_width is not None:
        result["stance"] = ankle_width / shoulder_width
    return result


def build_coaching_report(
    observations: Iterable[Mapping[str, Any]],
    *,
    fps: float,
) -> dict[str, Any]:
    """Turn verified pose observations into a compact, UI-safe report.

    ``observations`` contains no file paths and is intentionally easy to unit
    test.  A player needs at least two valid pose samples before any score or
    recommendation is shown.
    """
    grouped: dict[str, list[dict[str, Any]]] = {"near": [], "far": []}
    for raw in observations:
        player = str(raw.get("player", ""))
        keypoints = raw.get("keypoints")
        if player not in grouped or not isinstance(keypoints, Sequence):
            continue
        metrics = _metrics(keypoints)
        if metrics:
            grouped[player].append({"frame": int(_number(raw.get("frame"))), "metrics": metrics})

    players = []
    for player, samples in grouped.items():
        if len(samples) < _MIN_SAMPLES:
            continue
        values: dict[str, list[float]] = {"reach": [], "knee_angle": [], "stance": []}
        for sample in samples:
            for key, value in sample["metrics"].items():
                values[key].append(value)
        scores: dict[str, int] = {}
        if values["reach"]:
            # Wrist-to-shoulder reach normalized by torso height.  It is a
            # consistency signal, not a claim of ideal stroke mechanics.
            scores["reach"] = _clip_score((mean(values["reach"]) - 0.70) / 0.70 * 100)
        if values["knee_angle"]:
            scores["knee_angle"] = _clip_score((170.0 - mean(values["knee_angle"])) / 35.0 * 100)
        if values["stance"]:
            scores["stance"] = _clip_score((mean(values["stance"]) - 0.50) / 0.55 * 100)
        if not scores:
            continue
        overall = _clip_score(mean(scores.values()))
        recommendations = []
        signals = [
            ("reach", "触球附近的手臂伸展", "已检测击球附近的可见手臂伸展信号偏小。可在慢速多球中练习先到位、在身体前上方完成触球；不要为追求伸展而耸肩。", "手臂伸展信号"),
            ("knee_angle", "触球附近的下肢弹性", "已检测击球附近的膝关节较直。可在准备和出拍前更早做轻微屈膝与分腿垫步，为蹬地和回位保留空间。", "膝关节信号"),
            ("stance", "触球附近的支撑面", "已检测击球附近的双脚支撑面偏窄。练习击球前最后一步落稳、击球后先恢复平衡再回中。", "双脚间距信号"),
        ]
        for metric, title, detail, metric_label in signals:
            if metric not in scores or scores[metric] >= 58:
                continue
            evidence = sorted(
                (sample for sample in samples if metric in sample["metrics"]),
                key=lambda sample: sample["metrics"][metric],
            )[:3]
            recommendations.append({
                "priority": "high" if scores[metric] < 35 else "medium",
                "title": title,
                "detail": detail,
                "metric": f"{metric_label} {scores[metric]}/100（二维信号）",
                "evidence": [
                    {"frame": item["frame"], "seconds": round(item["frame"] / max(1.0, fps), 2), "timecode": _time_label(item["frame"], fps)}
                    for item in evidence
                ],
            })
        if not recommendations:
            recommendations.append({
                "priority": "good",
                "title": "基础动作信号稳定",
                "detail": "可见击球帧里的伸展、下肢和支撑面稳定；保持当前节奏。",
                "metric": f"可见动作信号 {overall}/100（非技术评分）",
                "evidence": [
                    {"frame": sample["frame"], "seconds": round(sample["frame"] / max(1.0, fps), 2), "timecode": _time_label(sample["frame"], fps)}
                    for sample in samples[:3]
                ],
            })
        players.append({
            "id": player,
            "label": PLAYER_LABELS[player],
            "score": overall,
            "sample_count": len(samples),
            "recommendations": recommendations[:3],
        })
    insufficient_labels = [
        PLAYER_LABELS[player]
        for player, samples in grouped.items()
        if len(samples) < _MIN_SAMPLES
    ]
    notice = "基于击球附近可见姿态给出训练建议；分数是动作信号，不是技术评分。"
    if insufficient_labels:
        notice += f" {'、'.join(insufficient_labels)}姿态样本不足，暂不生成建议。"
    return {
        "version": 1,
        "status": "ready" if players else "insufficient_evidence",
        "notice": notice,
        "players": players,
    }


def _hit_frames(sidecar_path: Path, maximum: int = 32) -> list[tuple[int, str, float | None]]:
    """Return a bounded, evenly distributed set of *confirmed* hit frames."""
    hits: list[tuple[int, str, float | None]] = []
    with sidecar_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(_number(row.get("Hit"))) <= 0:
                continue
            frame = int(_number(row.get("Frame"), -1))
            if frame < 0:
                continue
            hitter = str(row.get("HitHitter", "")).strip().lower()
            ball_y = _number(row.get("HitY"), -1)
            hits.append((frame, hitter if hitter in PLAYER_LABELS else "", ball_y if ball_y >= 0 else None))
    if len(hits) <= maximum:
        return hits
    stride = len(hits) / float(maximum)
    return [hits[min(len(hits) - 1, int(index * stride))] for index in range(maximum)]


def _point_in_convex_quad(point: tuple[float, float], quad: Sequence[tuple[float, float]]) -> bool:
    """Return whether ``point`` is in a clockwise/counter-clockwise court quad."""
    if len(quad) != 4:
        return False
    signs = []
    for index, start in enumerate(quad):
        end = quad[(index + 1) % len(quad)]
        cross = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])
        signs.append(cross)
    return min(signs) >= -1e-4 or max(signs) <= 1e-4


def _pose_foot(keypoints: Sequence[Sequence[object]], box: Sequence[object]) -> tuple[float, float] | None:
    """Prefer visible ankles; use the box bottom only when ankles are absent."""
    ankles = [point for point in (_point(keypoints, 15), _point(keypoints, 16)) if point is not None]
    if ankles:
        return (
            sum(point[0] for point in ankles) / len(ankles),
            sum(point[1] for point in ankles) / len(ankles),
        )
    try:
        x1, _y1, x2, y2 = (float(value) for value in box[:4])
    except (TypeError, ValueError, IndexError):
        return None
    return ((x1 + x2) / 2.0, y2)


_MEASURED_SOURCES = {"model", "classical"}
_MAX_STROKE_EVENTS = 36
_MAX_STROKE_REVIEWS = 12


def _integer(value: object, default: int = 0) -> int:
    number = _number(value, float(default))
    try:
        return int(number if number is not None else default)
    except (TypeError, ValueError, OverflowError):
        return default


def _measured_ball_point(row: Mapping[str, object]) -> tuple[float, float] | None:
    """Return only a detector/classical observation, never an interpolation."""
    raw_source = str(row.get("RawSource", "")).strip().lower()
    source = str(row.get("Source", "")).strip().lower()
    candidates = (
        (raw_source, row.get("RawX"), row.get("RawY"), row.get("RawVisibility")),
        (source, row.get("X"), row.get("Y"), row.get("Visibility")),
    )
    for candidate_source, raw_x, raw_y, visibility in candidates:
        x, y = _number(raw_x), _number(raw_y)
        if candidate_source in _MEASURED_SOURCES and _integer(visibility) > 0 and x is not None and y is not None:
            return x, y
    return None


def _trajectory_metrics(
    event: Mapping[str, Any],
    measured_rows: Mapping[int, tuple[float, float]],
    *,
    fps: float,
    court_height_px: float,
) -> dict[str, Any]:
    start = _integer(event.get("frame"), -1)
    end = _integer(event.get("shot_end_frame"), -1)
    points = []
    if start >= 0 and end > start:
        for frame in sorted(frame for frame in measured_rows if start <= frame <= end):
            points.append((frame, measured_rows[frame]))
    arc = 0.0
    if len(points) >= 4 and points[-1][0] > points[0][0] and court_height_px > 1.0:
        first_frame, first = points[0]
        last_frame, last = points[-1]
        residuals = []
        for frame, point in points[1:-1]:
            progress = (frame - first_frame) / float(last_frame - first_frame)
            line_y = first[1] + (last[1] - first[1]) * progress
            residuals.append(abs(point[1] - line_y) / court_height_px)
        if residuals:
            arc = max(residuals)
    observed = _number(event.get("shot_observed_seconds"))
    if observed is None or observed <= 0.0:
        observed = (points[-1][0] - points[0][0]) / max(1.0, fps) if len(points) >= 2 else 0.0
    speed = _number(event.get("speed_kmh"))
    coverage = _number(event.get("shot_coverage"), 0.0) or 0.0
    quality = str(event.get("shot_speed_quality", "")).lower()
    method = str(event.get("shot_speed_method", "")).lower()
    return {
        "real_points": len(points),
        "flight_seconds": round(max(0.0, observed), 2),
        "arc_signal": round(arc, 3),
        "speed_kmh": round(speed, 1) if speed is not None and speed >= 0 else None,
        "speed_reliable": bool(speed is not None and speed > 0 and quality == "ok" and method == "court_plane" and coverage >= 0.55),
        "shot_coverage": round(coverage, 2),
    }


def _link_event_receivers(events: Sequence[dict[str, Any]]) -> None:
    """Link a stroke to the next identified hitter within its rally only."""
    for index, event in enumerate(events):
        next_in_rally = next(
            (candidate for candidate in events[index + 1:] if candidate["rally_id"] == event["rally_id"]),
            None,
        )
        end = _integer(event.get("shot_end_frame"), -1)
        if end <= int(event["frame"]) and next_in_rally is not None:
            end = int(next_in_rally["frame"])
            event["shot_end_frame"] = end
        receiver = None
        if next_in_rally is not None and end > int(event["frame"]) and abs(int(next_in_rally["frame"]) - end) <= 3:
            candidate_hitter = str(next_in_rally.get("hitter", ""))
            if candidate_hitter in PLAYER_LABELS and candidate_hitter != event["hitter"]:
                receiver = candidate_hitter
        event["receiver"] = receiver


def _outgoing_direction_hitter(
    event: Mapping[str, Any],
    measured_rows: Mapping[int, tuple[float, float]],
    *,
    fps: float,
    court_height_px: float,
) -> tuple[str, float] | None:
    """Infer a weak side hint from the initial visible shuttle direction.

    In the calibrated broadcast convention, a far-side stroke initially moves
    towards larger image ``y`` and a near-side stroke towards smaller ``y``.
    A shuttle's arc and perspective make this a supporting cue only; it never
    becomes a standalone high-confidence attribution.
    """
    frame = _integer(event.get("frame"), -1)
    if frame < 0:
        return None
    end = _integer(event.get("shot_end_frame"), -1)
    if end <= frame:
        end = frame + max(5, int(round(0.55 * max(1.0, fps))))
    stop = min(end - 1, frame + max(6, int(round(0.55 * max(1.0, fps)))))
    points = [
        (candidate, measured_rows[candidate])
        for candidate in range(frame + 1, stop + 1)
        if candidate in measured_rows
    ]
    if len(points) < 3:
        return None
    slopes = [
        (right[1][1] - left[1][1]) / float(right[0] - left[0])
        for left, right in zip(points, points[1:])
        if 0 < right[0] - left[0] <= 2
    ]
    if len(slopes) < 2:
        return None
    vertical_speed = float(median(slopes))
    minimum_speed = max(7.0, float(court_height_px) * 0.008)
    if abs(vertical_speed) < minimum_speed:
        return None
    side = "near" if vertical_speed < 0.0 else "far"
    confidence = min(
        0.76,
        0.42
        + 0.20 * min(1.0, abs(vertical_speed) / max(minimum_speed * 3.0, 1.0))
        + 0.12 * min(1.0, (len(points) - 2) / 5.0),
    )
    return side, round(confidence, 3)


def _hitter_confidence(event: Mapping[str, Any], *, default_for_sidecar: float = 0.48) -> float:
    """Return attribution confidence without confusing it with ``HitScore``.

    Older tracking sidecars only expose a hit-quality score.  That says a
    shuttle turn was detected, not that its hitter was identified, so a legacy
    ``near``/``far`` label starts as deliberately weak evidence here.
    """
    side = str(event.get("hitter", "")).strip().lower()
    if side not in PLAYER_LABELS:
        return 0.0
    raw = _number(event.get("hitter_confidence"), -1.0)
    if raw > 1.0:
        raw /= 100.0
    if raw < 0.0:
        return float(default_for_sidecar)
    return max(0.0, min(1.0, raw))


def _opponent(side: str) -> str:
    return "far" if side == "near" else "near"


def _is_sequence_anchor(event: Mapping[str, Any], threshold: float) -> bool:
    """Whether this side label is strong enough to anchor rally parity."""
    side = str(event.get("hitter", "")).strip().lower()
    if side not in PLAYER_LABELS or _hitter_confidence(event) < threshold:
        return False
    source = str(event.get("hitter_provenance", "")).strip().lower()
    # A legacy sidecar label is not an anchor unless a newer sidecar supplied
    # explicit, very high attribution confidence.
    return source != "sidecar" or _hitter_confidence(event) >= 0.85


def _sequence_has_collision(events: Sequence[Mapping[str, Any]], start: int, end: int) -> bool:
    """Do not infer through near-duplicate hit candidates."""
    for event in events[max(0, start):min(len(events), end + 1)]:
        if str(event.get("hit_source", "")).strip().lower() in {"edge_exit", "classical_endpoint"}:
            return True
    for index in range(max(0, start), min(len(events) - 1, end)):
        left = _integer(events[index].get("frame"), -1)
        right = _integer(events[index + 1].get("frame"), -1)
        if left >= 0 and right > left and right - left <= 10:
            return True
    return False


def _direction_supports(event: Mapping[str, Any], expected: str) -> bool:
    side = str(event.get("hitter_direction", "")).strip().lower()
    confidence = _number(event.get("hitter_direction_confidence"), 0.0)
    if confidence > 1.0:
        confidence /= 100.0
    return side == expected and confidence >= 0.45


def _correct_rally_hitter_alternation(
    events: Sequence[Mapping[str, Any]],
    *,
    low_confidence_threshold: float = 0.60,
) -> list[dict[str, Any]]:
    """Conservatively repair hitter labels using the within-rally sequence.

    Badminton contacts alternate sides, but a detector can miss a contact or
    emit a duplicate turn.  Strong contact-pose labels become anchors; only a
    parity-consistent, collision-free gap between two anchors can fill an
    unknown label.  This is a consistency check, never a count normalizer.

    The input remains untouched so callers can retain the raw sidecar record
    for diagnostics.
    """
    threshold = max(0.0, min(1.0, float(low_confidence_threshold)))
    prepared: list[dict[str, Any]] = []
    for raw in events:
        if not isinstance(raw, Mapping):
            continue
        event = dict(raw)
        side = str(event.get("hitter", "")).strip().lower()
        event["hitter"] = side if side in PLAYER_LABELS else "unknown"
        confidence = _hitter_confidence(event)
        event["hitter_confidence"] = confidence
        if not isinstance(event.get("hitter_provenance"), str) or not event.get("hitter_provenance"):
            event["hitter_provenance"] = "sidecar" if side in PLAYER_LABELS else "unknown"
        event["hitter_inferred"] = bool(event.get("hitter_inferred", False))
        event["hitter_corrected"] = bool(event.get("hitter_corrected", False))
        prepared.append(event)

    groups: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for index, event in enumerate(prepared):
        rally_id = _integer(event.get("rally_id"), 0)
        groups.setdefault(rally_id, []).append((index, event))

    for rally_id, grouped in groups.items():
        if rally_id <= 0:
            continue
        grouped.sort(key=lambda item: (_integer(item[1].get("frame"), -1), item[0]))
        sequence = [event for _index, event in grouped]

        anchors = [
            index for index, event in enumerate(sequence)
            if _is_sequence_anchor(event, threshold)
        ]
        for left_index, right_index in zip(anchors, anchors[1:]):
            gap_size = right_index - left_index - 1
            # Alternation is a local tiebreaker, not a mechanism for creating
            # a complete synthetic rally.  A long unlabelled span can contain
            # missed contacts, duplicate TrackNet turns, or both; filling it
            # by parity would make an appealing but invented hit sequence.
            # Keep the repair to one ambiguous contact bounded by two anchors.
            if gap_size != 1 or _sequence_has_collision(sequence, left_index, right_index):
                continue
            left = sequence[left_index]
            right = sequence[right_index]
            left_side = str(left.get("hitter", ""))
            expected_right = left_side if (right_index - left_index) % 2 == 0 else _opponent(left_side)
            if str(right.get("hitter", "")) != expected_right:
                # A missed contact or duplicate candidate breaks this parity
                # chain; preserve the labels for video review.
                continue
            for index in range(left_index + 1, right_index):
                center = sequence[index]
                expected = left_side if (index - left_index) % 2 == 0 else _opponent(left_side)
                center_side = str(center.get("hitter", ""))
                center_confidence = _hitter_confidence(center)
                if center_side == "unknown":
                    center.update({
                        "hitter": expected,
                        "hitter_confidence": 0.55,
                        "hitter_inferred": True,
                        "hitter_provenance": "rally_alternation_inferred",
                    })
                elif (
                    center_side != expected
                    and center_confidence < threshold
                    and _direction_supports(center, expected)
                ):
                    center.update({
                        "hitter": expected,
                        "hitter_confidence": min(0.58, max(0.45, center_confidence)),
                        "hitter_corrected": True,
                        "hitter_provenance": "rally_alternation_corrected",
                    })

    return prepared


def _rally_hitter_balance(events: Sequence[Mapping[str, Any]]) -> list[dict[str, int | bool]]:
    """Summarize the alternation sanity check without changing any labels.

    In a complete, clean badminton rally the two hitters alternate, so their
    counts differ by at most one.  Missing contacts and duplicated trajectory
    turns are common enough that this is a *diagnostic*, never a relabelling
    target.  Unknown events stay excluded from player totals and make the
    rally incomplete for display purposes.
    """
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for event in events:
        if not isinstance(event, Mapping):
            continue
        rally_id = _integer(event.get("rally_id"), 0)
        if rally_id > 0:
            grouped.setdefault(rally_id, []).append(event)
    summary: list[dict[str, int | bool]] = []
    for rally_id in sorted(grouped):
        near_count = sum(str(event.get("hitter", "")).strip().lower() == "near" for event in grouped[rally_id])
        far_count = sum(str(event.get("hitter", "")).strip().lower() == "far" for event in grouped[rally_id])
        unknown_count = len(grouped[rally_id]) - near_count - far_count
        difference = abs(near_count - far_count)
        summary.append({
            "rally_id": rally_id,
            "near_count": near_count,
            "far_count": far_count,
            "unknown_count": unknown_count,
            "difference": difference,
            "within_one": difference <= 1,
            "complete": unknown_count == 0,
        })
    return summary


def _read_stroke_events(sidecar_path: Path, *, fps: float, court_height_px: float) -> tuple[list[dict[str, Any]], int]:
    """Parse sidecar evidence and retain provenance gates for every stroke."""
    measured_rows: dict[int, tuple[float, float]] = {}
    events: list[dict[str, Any]] = []
    unknown_hitter_count = 0
    with sidecar_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            frame = _integer(row.get("Frame"), -1)
            if frame < 0:
                continue
            measured = _measured_ball_point(row)
            if measured is not None:
                measured_rows[frame] = measured
            if _integer(row.get("Hit")) <= 0:
                continue
            hitter = str(row.get("HitHitter", "")).strip().lower()
            hitter_known = hitter in PLAYER_LABELS
            if not hitter_known:
                unknown_hitter_count += 1
            raw_hitter_confidence = _number(row.get("HitHitterConfidence"), -1.0)
            if raw_hitter_confidence > 1.0:
                raw_hitter_confidence /= 100.0
            hitter_confidence = (
                max(0.0, min(1.0, raw_hitter_confidence))
                if hitter_known and raw_hitter_confidence >= 0.0
                # Legacy sidecars stored only ``HitScore`` (hit quality), not
                # hitter-attribution confidence.  Keep that hint deliberately
                # weak instead of treating it as a pose-confirmed identity.
                else (0.48 if hitter_known else 0.0)
            )
            hitter_source = str(row.get("HitHitterSource", "")).strip().lower()
            if not hitter_source:
                hitter_source = "sidecar" if hitter_known else "unknown"
            hit_x, hit_y = _number(row.get("HitX")), _number(row.get("HitY"))
            hit_score = _number(row.get("HitScore"), 0.0) or 0.0
            observed = _integer(row.get("HitBallObserved")) > 0
            uncertain = _integer(row.get("HitUncertain")) > 0
            events.append({
                "frame": frame,
                "hitter": hitter if hitter_known else "unknown",
                "hitter_confidence": hitter_confidence,
                "hitter_provenance": hitter_source,
                "hit_point": (hit_x, hit_y) if hit_x is not None and hit_y is not None else None,
                "hit_score": hit_score,
                "hit_source": str(row.get("HitSource", "")).strip().lower(),
                "rally_id": _integer(row.get("RallyID")),
                "rally_hit_index": _integer(row.get("HitRallyIndex")),
                "shot_end_frame": _integer(row.get("ShotEndFrame"), -1),
                "shot_observed_seconds": _number(row.get("ShotObservedSeconds"), 0.0) or 0.0,
                "speed_kmh": _number(row.get("ShotAvgSpeedKmh")),
                "shot_coverage": _number(row.get("ShotCoverage"), 0.0) or 0.0,
                "shot_speed_quality": str(row.get("ShotSpeedQuality", "")),
                "shot_speed_method": str(row.get("ShotSpeedMethod", "")),
                # A low-quality hit is not allowed to become a high-confidence
                # technical call, but a measured ball point plus a coherent
                # route can still be useful as a clearly downgraded candidate.
                "route_candidate": bool(observed and hit_score >= 0.55 and hit_x is not None and hit_y is not None),
                "confirmed": bool(observed and not uncertain and hit_score >= 0.60 and hit_x is not None and hit_y is not None),
                "hitter_inferred": False,
                "hitter_corrected": False,
            })
    events.sort(key=lambda item: int(item["frame"]))
    _link_event_receivers(events)
    for event in events:
        event["trajectory"] = _trajectory_metrics(event, measured_rows, fps=fps, court_height_px=court_height_px)
        direction = _outgoing_direction_hitter(
            event,
            measured_rows,
            fps=fps,
            court_height_px=court_height_px,
        )
        if direction is not None:
            event["hitter_direction"], event["hitter_direction_confidence"] = direction
    reliable_speeds = [
        float(event["trajectory"]["speed_kmh"])
        for event in events
        if event["trajectory"].get("speed_reliable") and event["trajectory"].get("speed_kmh") is not None
    ]
    for event in events:
        speed = event["trajectory"].get("speed_kmh")
        if event["trajectory"].get("speed_reliable") and speed is not None and reliable_speeds:
            event["trajectory"]["speed_percentile"] = round(sum(value <= speed for value in reliable_speeds) / len(reliable_speeds), 2)
        else:
            event["trajectory"]["speed_percentile"] = None
    return events, unknown_hitter_count


def _evenly_spaced(items: Sequence[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if len(items) <= maximum:
        return list(items)
    stride = len(items) / float(maximum)
    return [items[min(len(items) - 1, int(index * stride))] for index in range(maximum)]


def _court_context(court_points: Sequence[float]):
    """Build a calibrated pixel-to-court mapper, or return no mapper."""
    if not isinstance(court_points, (list, tuple)) or len(court_points) != 8:
        return [], None, None, 1.0
    try:
        values = [float(value) for value in court_points]
        quad = [(values[index], values[index + 1]) for index in range(0, 8, 2)]
        if max(point[1] for point in quad) - min(point[1] for point in quad) <= 1.0:
            return [], None, None, 1.0
        import cv2
        import numpy as np

        source = np.float32(quad)
        destination = np.float32([(0.0, 0.0), (6.10, 0.0), (6.10, 13.40), (0.0, 13.40)])
        matrix = cv2.getPerspectiveTransform(source, destination)

        def mapper(point: tuple[float, float]) -> tuple[float, float] | None:
            converted = cv2.perspectiveTransform(np.float32([[[point[0], point[1]]]]), matrix)[0][0]
            x, y = float(converted[0]), float(converted[1])
            if not math.isfinite(x) or not math.isfinite(y):
                return None
            return x, y

        return quad, sum(point[1] for point in quad) / 4.0, mapper, max(point[1] for point in quad) - min(point[1] for point in quad)
    except Exception:
        return [], None, None, 1.0


def _extract_pose_sides(result: Any, *, court_quad, court_mid_y, mapper) -> dict[str, dict[str, Any]]:
    """Assign detected people to the calibrated top/bottom court half only."""
    if result.keypoints is None or result.keypoints.xy is None or result.boxes is None or result.boxes.xyxy is None:
        return {}
    points = result.keypoints.xy.cpu().tolist()
    boxes = result.boxes.xyxy.cpu().tolist()
    confidences = result.keypoints.conf.cpu().tolist() if result.keypoints.conf is not None else []
    candidates = []
    for index, keypoint_set in enumerate(points):
        if index >= len(boxes):
            continue
        merged = [
            [point[0], point[1], confidences[index][joint] if index < len(confidences) and joint < len(confidences[index]) else 0.0]
            for joint, point in enumerate(keypoint_set)
        ]
        foot = _pose_foot(merged, boxes[index])
        if foot is None or (court_quad and not _point_in_convex_quad(foot, court_quad)):
            continue
        position = mapper(foot) if mapper is not None else None
        # Keep the detector box alongside COCO-17 points.  The optional BST
        # adapter uses the same bbox-diagonal normalization as its training
        # preprocessing; existing Motion Coach consumers ignore this field.
        candidates.append({
            "foot": foot,
            "court_position": position,
            "keypoints": merged,
            "box": [float(value) for value in boxes[index][:4]],
        })
    if not candidates or court_mid_y is None:
        return {}
    def is_far(item: Mapping[str, Any]) -> bool:
        # The BST preprocessing orders players by court-plane y.  Prefer that
        # calibrated coordinate under perspective; use the image midpoint as
        # a fallback for legacy/un-calibrated pose observations.
        position = item.get("court_position")
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            try:
                position_y = float(position[1])
                if math.isfinite(position_y):
                    return position_y < 6.70  # 13.40 m court midpoint
            except (TypeError, ValueError):
                pass
        return item["foot"][1] < court_mid_y

    far = [item for item in candidates if is_far(item)]
    near = [item for item in candidates if not is_far(item)]

    def sort_y(item: Mapping[str, Any]) -> float:
        position = item.get("court_position")
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            try:
                value = float(position[1])
                if math.isfinite(value):
                    return value
            except (TypeError, ValueError):
                pass
        return float(item["foot"][1])

    result_by_side: dict[str, dict[str, Any]] = {}
    if far:
        result_by_side["far"] = min(far, key=sort_y)
    if near:
        result_by_side["near"] = max(near, key=sort_y)
    return result_by_side


def _pose_frames(
    video: Path,
    *,
    model: Any,
    requested_frames: Sequence[int],
    device: str,
    court_quad,
    court_mid_y,
    mapper,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Run a bounded batched pose pass and keep only calibrated court people."""
    import cv2

    requested_device = str(device).lower()
    # Ultralytics accepts a CUDA device index reliably across releases; the
    # literal ``cuda`` can terminate inference before it returns a report in
    # some local installations.
    model_device = 0 if requested_device == "cuda" else (requested_device if requested_device in {"mps", "cpu"} else None)
    output: dict[int, dict[str, dict[str, Any]]] = {}

    def process_chunk(chunk: Sequence[tuple[int, Any]]) -> None:
        if not chunk:
            return
        kwargs: dict[str, Any] = {
            "verbose": False,
            "classes": [0],
            "conf": 0.20,
            "imgsz": 960,
        }
        if model_device is not None:
            kwargs["device"] = model_device
        results = list(model.predict(source=[image for _frame, image in chunk], **kwargs))
        for (frame_number, _image), result in zip(chunk, results):
            output[frame_number] = _extract_pose_sides(
                result,
                court_quad=court_quad,
                court_mid_y=court_mid_y,
                mapper=mapper,
            )
        # Ultralytics can retain batch tensors in CUDA's allocator across a
        # long local-video pass.  The report stores only scalar keypoints, so
        # returning the completed batch cache immediately keeps the optional
        # coach process bounded on smaller GPUs.
        del results
        try:
            del result
        except UnboundLocalError:
            pass
        if model_device == 0:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    # A full professional pass can request 200+ HD frames.  Keep only one
    # inference batch in RAM; retaining every decoded frame can exceed a
    # local GPU process's memory budget before pose inference even begins.
    capture = cv2.VideoCapture(str(video))
    pending: list[tuple[int, Any]] = []
    try:
        # Seeking for every requested frame is unexpectedly expensive on an
        # H.264 match recording (and can make a 30-hit review look as though
        # pose inference timed out).  The requested frames are already
        # sorted, so decode forward once and only retain frames needed by the
        # bounded batch.  We still start at the first requested frame to avoid
        # decoding a long unneeded pre-roll.
        target_frames = sorted({max(0, int(frame)) for frame in requested_frames})
        if not target_frames:
            return output
        capture.set(cv2.CAP_PROP_POS_FRAMES, target_frames[0])
        next_frame = target_frames[0]
        for frame_number in target_frames:
            while next_frame < frame_number:
                if not capture.grab():
                    process_chunk(pending)
                    return output
                next_frame += 1
            ok, image = capture.read()
            if not ok:
                process_chunk(pending)
                return output
            next_frame += 1
            pending.append((frame_number, image))
            if len(pending) >= 18:
                process_chunk(pending)
                pending = []
        process_chunk(pending)
    finally:
        capture.release()
    return output


def _contact_association_distance(raw_pose: Mapping[str, Any], hit_point: object) -> float | None:
    """Distance from a virtual racket tip to an observed ball, normalized.

    This is intentionally used only to associate an otherwise unknown hit to a
    court half.  It does not identify a racket, a forehand/backhand, or a
    stroke type.  A wide separation gate is applied by the caller.
    """
    keypoints = raw_pose.get("keypoints") if isinstance(raw_pose, Mapping) else None
    if not isinstance(keypoints, Sequence):
        return None
    ball = hit_point if isinstance(hit_point, (list, tuple)) and len(hit_point) >= 2 else None
    if ball is None:
        return None
    ball_x, ball_y = _number(ball[0]), _number(ball[1])
    if ball_x is None or ball_y is None:
        return None
    left_shoulder = _point(keypoints, 5)
    right_shoulder = _point(keypoints, 6)
    left_hip = _point(keypoints, 11)
    right_hip = _point(keypoints, 12)
    torso = _distance(_midpoint(left_shoulder, right_shoulder), _midpoint(left_hip, right_hip))
    if torso is None or torso < 4.0:
        return None
    distances = []
    for shoulder_i, elbow_i, wrist_i in ((5, 7, 9), (6, 8, 10)):
        shoulder, elbow, wrist = _point(keypoints, shoulder_i), _point(keypoints, elbow_i), _point(keypoints, wrist_i)
        if shoulder is None or elbow is None or wrist is None:
            continue
        tip = (wrist[0] + 0.35 * (wrist[0] - elbow[0]), wrist[1] + 0.35 * (wrist[1] - elbow[1]))
        distances.append(math.hypot(tip[0] - ball_x, tip[1] - ball_y) / torso)
    return min(distances) if distances else None


def _contact_pose_hitter_evidence(
    event: Mapping[str, Any],
    pose_by_frame: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> tuple[str, float] | None:
    """Return a strict virtual-racket contact association when visible."""
    frame = _integer(event.get("frame"), -1)
    hit_point = event.get("hit_point")
    if frame < 0 or hit_point is None:
        return None
    candidates = []
    for side, raw_pose in pose_by_frame.get(frame, {}).items():
        if side not in PLAYER_LABELS or not isinstance(raw_pose, Mapping):
            continue
        distance = _contact_association_distance(raw_pose, hit_point)
        if distance is not None:
            candidates.append((distance, side))
    if not candidates:
        return None
    candidates.sort()
    best_distance, best_side = candidates[0]
    second_distance = candidates[1][0] if len(candidates) > 1 else None
    # The absolute gate keeps a player body far from the shuttle from becoming
    # the hitter; if only one side is visible, make it materially stricter.
    maximum_distance = 0.75 if second_distance is None else 1.05
    if best_distance > maximum_distance:
        return None
    if second_distance is not None and second_distance - best_distance < 0.35:
        return None
    closeness = max(0.0, min(1.0, 1.0 - best_distance / maximum_distance))
    separation = (
        1.0
        if second_distance is None
        else max(0.0, min(1.0, (second_distance - best_distance - 0.35) / 0.75))
    )
    confidence = 0.63 + 0.19 * closeness + 0.12 * separation
    return best_side, round(min(0.94, confidence), 3)


def _infer_hitter_from_contact_pose(event: Mapping[str, Any], pose_by_frame: Mapping[int, Mapping[str, Mapping[str, Any]]]) -> str | None:
    """Compatibility wrapper for callers that only need the associated side."""
    evidence = _contact_pose_hitter_evidence(event, pose_by_frame)
    return evidence[0] if evidence is not None else None


def _apply_pose_hitter_associations(
    events: Sequence[dict[str, Any]],
    pose_by_frame: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> int:
    """Prefer a clearly visible contact arm over a weak legacy side hint."""
    associated = 0
    for event in events:
        # These snapshots must be taken before the evidence branch: the
        # no-pose branch deliberately uses them to decide whether a weak
        # legacy side hint should be downgraded to unknown.
        previous_side = str(event.get("hitter", "")).strip().lower()
        previous_confidence = _hitter_confidence(event)
        previous_source = str(event.get("hitter_provenance", "")).lower()
        evidence = _contact_pose_hitter_evidence(event, pose_by_frame)
        if evidence is None:
            # A legacy sidecar's label came from a loose nearest-body hint.
            # When the stricter virtual-racket test cannot support it, expose
            # the uncertainty rather than keeping a plausible-looking but
            # potentially wrong player assignment (notably for high clears).
            if previous_source == "sidecar" and previous_confidence < 0.60:
                event.update({
                    "hitter": "unknown",
                    "hitter_confidence": 0.0,
                    "hitter_provenance": "pose_contact_unresolved",
                })
            continue
        side, confidence = evidence
        # Do not overwrite a future sidecar's explicit high-confidence pose
        # result with another sparse pose sample.  Existing sidecars have no
        # such confidence and therefore remain eligible for correction.
        should_replace = (
            previous_side not in PLAYER_LABELS
            or previous_source == "sidecar"
            or confidence >= previous_confidence + 0.16
        )
        if not should_replace:
            continue
        changed = previous_side != side
        event.update({
            "hitter": side,
            "hitter_confidence": confidence,
            "hitter_provenance": "pose_contact",
            "hitter_inferred": True,
        })
        if changed and previous_side in PLAYER_LABELS:
            event["hitter_corrected"] = True
        associated += 1
    return associated


def _median_position(values: Iterable[object]) -> tuple[float, float] | None:
    coordinates = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        x, y = _number(value[0]), _number(value[1])
        if x is not None and y is not None:
            coordinates.append((x, y))
    if not coordinates:
        return None
    return (mean(item[0] for item in coordinates), mean(item[1] for item in coordinates))


def _time_label_with_seconds(seconds: float) -> str:
    minutes, remainder = divmod(int(max(0.0, seconds)), 60)
    return f"{minutes:02d}:{remainder:02d}"


def _bst_hits_by_frame(report: Mapping[str, Any] | None) -> dict[int, list[dict[str, Any]]]:
    """Index a ready BST report without making it a required dependency."""
    if not isinstance(report, Mapping) or str(report.get("status", "")) != "ready":
        return {}
    raw_hits = report.get("hits", [])
    if not isinstance(raw_hits, (list, tuple)):
        return {}
    indexed: dict[int, list[dict[str, Any]]] = {}
    for raw in raw_hits:
        if not isinstance(raw, Mapping):
            continue
        frame = _integer(raw.get("frame"), -1)
        if frame < 0:
            continue
        indexed.setdefault(frame, []).append(dict(raw))
    return indexed


def _bst_hint_for_event(
    indexed: Mapping[int, Sequence[Mapping[str, Any]]],
    event: Mapping[str, Any],
    player: str,
) -> Mapping[str, Any] | None:
    """Match BST and Coach on frame, then verify rally/hit identity."""
    frame = _integer(event.get("frame"), -1)
    candidates = list(indexed.get(frame, ()))
    if frame < 0 or not candidates:
        return None
    event_rally = _integer(event.get("rally_id"), 0)
    event_hit = _integer(event.get("rally_hit_index"), 0)
    filtered = []
    for candidate in candidates:
        candidate_rally = _integer(candidate.get("rally_id"), 0)
        candidate_hit = _integer(candidate.get("hit_index"), 0)
        if event_rally and candidate_rally and event_rally != candidate_rally:
            continue
        if event_hit and candidate_hit and event_hit != candidate_hit:
            continue
        filtered.append(candidate)
    if not filtered:
        return None
    # Prefer a side-consistent card, but leave a side-mismatched card visible
    # to the selector so it can explain why the rule fallback was used.
    return max(
        filtered,
        key=lambda item: (
            str(item.get("hitter", "")) == player,
            str(item.get("model_side", "")) == player,
            _number(item.get("confidence"), 0.0),
        ),
    )


def _review_for_event(
    event: Mapping[str, Any],
    *,
    player: str,
    dominant_hand: str,
    fps: float,
    pose_by_frame: Mapping[int, Mapping[str, Mapping[str, Any]]],
    bst_hint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hit_frame = _integer(event.get("frame"), 0)
    offsets = event.get("sample_offsets", [])
    samples = []
    for offset in offsets if isinstance(offsets, Sequence) else []:
        frame = max(0, hit_frame + _integer(offset))
        raw_pose = pose_by_frame.get(frame, {}).get(player)
        if not isinstance(raw_pose, Mapping):
            continue
        samples.append({
            "frame": frame,
            "keypoints": raw_pose.get("keypoints", []),
            "court_position": raw_pose.get("court_position"),
        })
    closest = pose_by_frame.get(hit_frame, {}).get(player)
    contact_position_samples = sorted(
        (raw for raw in samples if isinstance(raw, Mapping)),
        key=lambda raw: abs(_integer(raw.get("frame"), hit_frame) - hit_frame),
    )[:3]
    origin_position = _median_position(
        [raw.get("court_position") for raw in contact_position_samples]
        + ([closest.get("court_position")] if isinstance(closest, Mapping) else [])
    )
    receiver = str(event.get("receiver", ""))
    endpoint = _integer(event.get("shot_end_frame"), -1)
    receiver_side_inferred = False
    # A confirmed shuttle stroke crosses the net.  If the next hit identity is
    # unavailable but the official shot-end frame has a calibrated opponent
    # pose, its *court half* can be used as lower-confidence route evidence.
    # This is not presented as a stable player identity.
    if receiver not in PLAYER_LABELS and endpoint > hit_frame:
        receiver = "far" if player == "near" else "near"
        receiver_side_inferred = True
    receiver_pose = pose_by_frame.get(endpoint, {}).get(receiver) if receiver in PLAYER_LABELS else None
    receiver_position = receiver_pose.get("court_position") if isinstance(receiver_pose, Mapping) else None
    origin_zone = zone_for_player_position(origin_position, player)
    receiver_zone = zone_for_player_position(receiver_position, receiver) if receiver in PLAYER_LABELS else None
    trajectory = dict(event.get("trajectory", {}) if isinstance(event.get("trajectory"), Mapping) else {})
    # Carry the event gate into the phase evaluator.  Direct unit callers that
    # construct a trajectory retain the historical default (confirmed).
    trajectory["event_confirmed"] = bool(event.get("confirmed", True))
    trajectory["receiver_side_inferred"] = receiver_side_inferred
    if origin_position is not None and isinstance(receiver_position, (list, tuple)) and len(receiver_position) >= 2:
        receiver_x = _number(receiver_position[0])
        if receiver_x is not None:
            trajectory["lateral_player_m"] = round(receiver_x - origin_position[0], 2)
    hit_point = event.get("hit_point")
    pose = extract_pose_metrics(
        samples,
        hit_frame=hit_frame,
        hit_point=hit_point,
        fps=fps,
        dominant_hand=dominant_hand,
    )
    rule_stroke = classify_stroke(
        event,
        trajectory,
        origin_zone=origin_zone,
        receiver_zone=receiver_zone,
        pose=pose,
        dominant_hand=dominant_hand,
    )
    stroke, classification = select_bst_stroke_type(
        rule_stroke,
        bst_hint,
        player=player,
    )
    evaluation = evaluate_stroke(stroke, pose, trajectory)
    attribution_source = str(event.get("hitter_provenance", "")).strip().lower()
    attribution_confidence = max(0.0, min(1.0, _number(event.get("hitter_confidence"), 0.0)))
    blocked_attribution_sources = {
        "rally_alternation_inferred",
        "rally_alternation_corrected",
    }
    attribution_eligible = attribution_source not in blocked_attribution_sources
    attribution_gate = {
        "eligible": attribution_eligible,
        "reason_code": "ready" if attribution_eligible else "hitter_attribution_unverified",
        "source": attribution_source or "unknown",
        "confidence": round(attribution_confidence, 3),
    }
    if not attribution_eligible:
        # Sequence alternation is useful for rally statistics, but it is not
        # observed contact evidence.  Keep the diagnostic review while making
        # it impossible for its pose signal to enter recommendation voting.
        evaluation = dict(evaluation)
        evaluation["advice_status"] = "suppressed"
        evaluation["eligible_signals"] = []
        evaluation["phase_eligible_signals"] = []
        evaluation["phase_advice_eligible"] = False
        evaluation["advice_candidates"] = []
        raw_evaluation_gate = evaluation.get("coaching_gate", {})
        evaluation_gate = dict(raw_evaluation_gate) if isinstance(raw_evaluation_gate, Mapping) else {}
        raw_reason_codes = evaluation_gate.get("reason_codes", [])
        reason_codes = [
            str(code) for code in raw_reason_codes
            if isinstance(code, str)
        ] if isinstance(raw_reason_codes, (list, tuple)) else []
        if "hitter_attribution_unverified" not in reason_codes:
            reason_codes.insert(0, "hitter_attribution_unverified")
        evaluation_gate.update({
            "eligible_for_aggregation": False,
            "reason_code": "hitter_attribution_unverified",
            "reason_codes": reason_codes,
        })
        evaluation["coaching_gate"] = evaluation_gate
    hit_seconds = hit_frame / max(1.0, fps)
    end_seconds = endpoint / max(1.0, fps) if endpoint > hit_frame else hit_seconds + 0.85
    window_start = max(0.0, hit_seconds - 0.55)
    window_end = min(hit_seconds + 2.50, max(hit_seconds + 0.70, end_seconds + 0.20))
    observed = []
    if origin_zone:
        observed.append(f"击球者在{ {'front': '前场', 'mid': '中场', 'back': '后场'}[origin_zone] }")
    if receiver_zone:
        endpoint_label = "对方端接触窗口" if receiver_side_inferred else "下一拍接球者"
        observed.append(f"{endpoint_label}在{ {'front': '前场', 'mid': '中场', 'back': '后场'}[receiver_zone] }")
    real_points = _integer(trajectory.get("real_points"))
    if real_points:
        observed.append(f"{real_points} 个真实出球球点")
    speed = _number(trajectory.get("speed_kmh"))
    percentile = _number(trajectory.get("speed_percentile"))
    if bool(trajectory.get("speed_reliable")) and speed is not None and percentile is not None:
        observed.append(f"视频内投影速度 {speed:.1f} km/h（P{int(round(percentile * 100))}）")
    advice_eligible = (
        classification.get("advice_eligible", True) is True
        and attribution_eligible
    )
    phase_signal_list = evaluation.get("phase_eligible_signals", [])
    phase_gate = evaluation.get("phase_gate", {})
    classification_status = str(classification.get("status", ""))
    phase_blocked_by_type = (
        bool(classification.get("type_conflict"))
        or classification_status in {"conflict", "side_mismatch", "unsupported"}
    )
    phase_advice_eligible = bool(
        isinstance(phase_signal_list, (list, tuple))
        and phase_signal_list
        and isinstance(phase_gate, Mapping)
        and phase_gate.get("eligible_for_aggregation", phase_gate.get("eligible")) is True
        and bool(event.get("confirmed", True))
        and attribution_eligible
        and not phase_blocked_by_type
    )
    evaluation = dict(evaluation)
    evaluation["phase_advice_eligible"] = phase_advice_eligible
    return {
        "rally_id": max(0, _integer(event.get("rally_id"))),
        "hit_index": max(0, _integer(event.get("rally_hit_index"))),
        "stroke": stroke,
        "classification": classification,
        # Stable overall gate used by aggregation and clients that should not
        # need to combine type and hitter-attribution failure states.
        "advice_eligible": advice_eligible,
        "phase_advice_eligible": phase_advice_eligible,
        "attribution_gate": attribution_gate,
        "evaluation": evaluation,
        "observed": observed[:4],
        "evidence": {
            "frame": hit_frame,
            "seconds": round(hit_seconds, 2),
            "timecode": _time_label_with_seconds(hit_seconds),
            "pre_seconds": round(hit_seconds - window_start, 2),
            "post_seconds": round(window_end - hit_seconds, 2),
            "window_start": round(window_start, 2),
            "window_end": round(window_end, 2),
            "real_points": max(0, real_points),
        },
    }


def _eligible_coaching_reviews(reviews: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only reviews explicitly allowed to drive visible suggestions.

    BST diagnostics remain in the separate stroke-type report.  Dropping a
    gated review here prevents a conflicting or weak type from leaking into
    either the representative cards or the multi-hit recommendation summary.
    Legacy rule-only reviews without a gate retain their prior behaviour.
    """
    eligible = []
    for review in reviews:
        phase_ok = review.get("phase_advice_eligible") is True
        exact_ok = "advice_eligible" not in review or review.get("advice_eligible") is True
        classification = review.get("classification", {})
        if isinstance(classification, Mapping) and "advice_eligible" in classification:
            exact_ok = exact_ok and classification.get("advice_eligible") is True
            status = str(classification.get("status", ""))
            # Explicit type contradictions remain a hard stop even for a
            # generic phase signal.  A weak/missing BST card is not a stop.
            if status in {"conflict", "side_mismatch", "unsupported"} or classification.get("type_conflict") is True:
                phase_ok = False
        attribution_gate = review.get("attribution_gate", {})
        if isinstance(attribution_gate, Mapping) and attribution_gate.get("eligible") is False:
            phase_ok = False
        if not (exact_ok or phase_ok):
            continue
        eligible.append(review)
    return eligible


def _representative_reviews(reviews: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Show varied, strongest evidence first while keeping a small UI payload."""
    ordered = sorted(
        reviews,
        key=lambda item: (
            str(dict(item.get("stroke", {})).get("family", "unknown")) in {"unknown", "front_unknown", "back_unknown"},
            -_integer(dict(item.get("stroke", {})).get("confidence")),
            _integer(dict(item.get("evidence", {})).get("frame")),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_families = set()
    for review in ordered:
        family = str(dict(review.get("stroke", {})).get("family", "unknown"))
        if family not in seen_families:
            selected.append(review)
            seen_families.add(family)
        if len(selected) >= _MAX_STROKE_REVIEWS:
            return selected
    for review in ordered:
        if review in selected:
            continue
        selected.append(review)
        if len(selected) >= _MAX_STROKE_REVIEWS:
            break
    return selected


def run_pose_coaching(
    video_path: Path | str,
    sidecar_path: Path | str,
    *,
    fps: float,
    pose_weights: Path | str,
    device: str = "auto",
    court_points: Sequence[float] = (),
    coach_target: str = "near",
    near_dominant_hand: str = "auto",
    far_dominant_hand: str = "auto",
    stroke_type_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build conservative, per-stroke coaching cards with original-video proof.

    This intentionally reviews only a bounded representative set.  A long
    match may contain hundreds of contacts, and running dense pose inference
    over every frame would make the optional coach block the core pipeline.
    Every selected card still carries a pre/post original-video time window.
    """
    sidecar, video, weights = Path(sidecar_path), Path(video_path), Path(pose_weights)
    target = str(coach_target or "near").lower()
    if target not in {"near", "far", "both"}:
        target = "near"
    hands = {
        "near": str(near_dominant_hand or "auto").lower(),
        "far": str(far_dominant_hand or "auto").lower(),
    }
    hands = {side: hand if hand in {"left", "right", "auto"} else "auto" for side, hand in hands.items()}
    selected_players = ("near", "far") if target == "both" else (target,)
    bst_hits = _bst_hits_by_frame(stroke_type_report)
    if not sidecar.is_file() or not video.is_file() or not weights.is_file():
        return {"version": 2, "status": "unavailable", "notice": "动作分析所需模型或数据不可用。", "players": []}
    try:
        import cv2  # Imported here so a no-model installation remains optional.
        from ultralytics import YOLO

        court_quad, court_mid_y, mapper, court_height = _court_context(court_points)
        if not court_quad or court_mid_y is None or mapper is None:
            return {"version": 2, "status": "insufficient_evidence", "notice": "需要四点球场标定才能生成动作建议。", "players": []}
        events, unknown_hitter_count = _read_stroke_events(sidecar, fps=max(1.0, float(fps)), court_height_px=court_height)
        # A broken early side hint must not determine which frames receive the
        # only pose pass.  Sample the whole rally first, then use strict pose
        # contact and the sequence constraint to decide player attribution.
        sampled_indices = _evenly_spaced(list(range(len(events))), _MAX_STROKE_EVENTS)
        sampled_events = [events[index] for index in sampled_indices]
        if not sampled_events:
            return {
                "version": 2,
                "status": "insufficient_evidence",
                "notice": "没有检测到可用击球。",
                "players": [],
            }
        offsets = sorted({
            int(round(seconds * fps))
            for seconds in (
                -0.48, -0.32, -0.20, -0.08,
                0.0,
                0.08, 0.22, 0.38, 0.58, 0.82,
            )
        })
        requested_frames = []
        for event in sampled_events:
            event["sample_offsets"] = offsets
            frame = _integer(event.get("frame"), 0)
            requested_frames.extend(max(0, frame + offset) for offset in offsets)
            endpoint = _integer(event.get("shot_end_frame"), -1)
            if endpoint > frame:
                requested_frames.append(endpoint)
        model = YOLO(str(weights))
        pose_by_frame = _pose_frames(
            video,
            model=model,
            requested_frames=requested_frames,
            device=device,
            court_quad=court_quad,
            court_mid_y=court_mid_y,
            mapper=mapper,
        )
        pose_hitter_count = _apply_pose_hitter_associations(sampled_events, pose_by_frame)
        # Correct only a weak/failed side sequence.  It is deliberately run
        # after contact-pose evidence so an airborne ball's nearest box cannot
        # dominate the whole rally.
        events = _correct_rally_hitter_alternation(events)
        sampled_events = [events[index] for index in sampled_indices]
        _link_event_receivers(events)
        players = []
        for side in selected_players:
            reviewed_events = [
                _review_for_event(
                    event,
                    player=side,
                    dominant_hand=hands[side],
                    fps=max(1.0, float(fps)),
                    pose_by_frame=pose_by_frame,
                    bst_hint=_bst_hint_for_event(bst_hits, event, side),
                )
                for event in sampled_events
                if event.get("hitter") == side
            ]
            reviews = _eligible_coaching_reviews(reviewed_events)
            if not reviews:
                continue
            display_reviews = _representative_reviews(reviews)
            summaries, recommendations, evidence_quality = summarize_reviews(reviews)
            pose_samples = sum(
                int(_number(dict(review.get("stroke", {})).get("confidence"), 0) or 0) >= 55
                for review in reviews
            )
            players.append({
                "id": side,
                "label": PLAYER_LABELS[side],
                "sample_count": pose_samples,
                "review_count": len(reviews),
                "gated_review_count": len(reviewed_events) - len(reviews),
                "evidence_quality": evidence_quality,
                # Kept for compatibility with clients that rendered the old
                # non-technical score.  The new UI calls this evidence quality
                # and does not present it as a technique grade.
                "score": evidence_quality,
                "stroke_summary": summaries,
                "stroke_reviews": display_reviews,
                "recommendations": recommendations,
            })
        notice = "动作建议基于真实球点、脚点和击球附近姿态；证据质量不是技术评分。"
        alternation_inferred_count = sum(
            str(event.get("hitter_provenance", "")) == "rally_alternation_inferred"
            for event in events
        )
        alternation_corrected_count = sum(
            str(event.get("hitter_provenance", "")) == "rally_alternation_corrected"
            for event in events
        )
        if pose_hitter_count:
            notice += f" 已用可见持拍臂校正 {pose_hitter_count} 个击球归属。"
        if alternation_inferred_count or alternation_corrected_count:
            notice += (
                f" {alternation_inferred_count} 个击球归属按回合顺序补足，{alternation_corrected_count} 个低可信归属已调整；"
                "相关卡片的证据质量已下调。"
            )
        balance = _rally_hitter_balance(events)
        if balance:
            unbalanced = [item for item in balance if not bool(item["within_one"])]
            incomplete = [item for item in balance if not bool(item["complete"])]
            if unbalanced:
                rally_labels = "、".join(f"Rally {item['rally_id']}" for item in unbalanced)
                notice += (
                    f" {rally_labels} 的击球归属不平衡，保留原始记录。"
                )
            else:
                notice += f" 已检查 {len(balance)} 个回合的击球交替。"
            if incomplete:
                notice += f" 其中 {len(incomplete)} 个回合含未归属击球，未计入球员统计。"
        if unknown_hitter_count:
            remaining_unknown = sum(event.get("hitter") == "unknown" for event in events)
            if remaining_unknown:
                notice += f" 另有 {remaining_unknown} 个击球未归属球员，未计入统计。"
        if any(hands[side] == "auto" for side in selected_players):
            notice += " 设置持拍手后可显示正手/反手。"
        return {
            "version": 2,
            "status": "ready" if players else "insufficient_evidence",
            "notice": notice,
            "players": players,
        }
    except Exception:
        # Keep coaching optional.  An unavailable pose runtime must never turn
        # a successful professional ball/trajectory analysis into a failed job.
        return {"version": 2, "status": "unavailable", "notice": "动作分析暂时不可用。", "players": []}


__all__ = ["build_coaching_report", "run_pose_coaching"]
