"""Rule based badminton movement observations from COCO-17 keypoints.

This module deliberately does *not* diagnose injury, judge technique, or try to
name a stroke from a single broadcast camera.  It emits a small number of
training prompts only when the caller supplies enough high-confidence 2-D pose
evidence and the relevant phase label.  It is independent of the renderer so
it can be used by an offline report or a future Web UI.

Coordinates are image pixels except ``court_xy_m`` which is an optional,
calibrated floor-plane coordinate.  The COCO-17 ordering is the one returned
by YOLO pose: shoulders 5/6, elbows 7/8, wrists 9/10, hips 11/12, knees 13/14
and ankles 15/16.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA_VERSION = "1.0"
MIN_KEYPOINT_CONFIDENCE = 0.45
MIN_READY_SAMPLES = 12
MIN_READY_SECONDS = 0.40
MIN_OVERHEAD_CONTACTS = 3
MIN_RECOVERY_OPPORTUNITIES = 3

_LEFT = {"shoulder": 5, "elbow": 7, "wrist": 9, "hip": 11, "knee": 13, "ankle": 15}
_RIGHT = {"shoulder": 6, "elbow": 8, "wrist": 10, "hip": 12, "knee": 14, "ankle": 16}


def _finite_point(value: Any) -> Optional[Tuple[float, float]]:
    """Return a finite two-dimensional point or ``None`` for malformed input."""
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _point(sample: Dict[str, Any], index: int, min_confidence: float) -> Optional[Tuple[float, float]]:
    points = sample.get("keypoints")
    if not isinstance(points, Sequence) or index >= len(points):
        return None
    confidences = sample.get("confidences")
    if isinstance(confidences, Sequence) and index < len(confidences):
        try:
            if float(confidences[index]) < min_confidence:
                return None
        except (TypeError, ValueError):
            return None
    return _finite_point(points[index])


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _midpoint(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle(a: Tuple[float, float], vertex: Tuple[float, float], b: Tuple[float, float]) -> Optional[float]:
    """Return the included joint angle in degrees, or None for a degenerate limb."""
    va = (a[0] - vertex[0], a[1] - vertex[1])
    vb = (b[0] - vertex[0], b[1] - vertex[1])
    na, nb = math.hypot(*va), math.hypot(*vb)
    if na < 1e-6 or nb < 1e-6:
        return None
    cosine = max(-1.0, min(1.0, (va[0] * vb[0] + va[1] * vb[1]) / (na * nb)))
    return math.degrees(math.acos(cosine))


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sample_confidence(sample: Dict[str, Any], indexes: Iterable[int]) -> Optional[float]:
    values = sample.get("confidences")
    if not isinstance(values, Sequence):
        return None
    valid = []
    for index in indexes:
        if index >= len(values):
            continue
        try:
            value = float(values[index])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            valid.append(value)
    return sum(valid) / len(valid) if valid else None


def _phase(sample: Dict[str, Any]) -> str:
    return str(sample.get("phase", "")).strip().lower()


def _frame(sample: Dict[str, Any]) -> Optional[int]:
    try:
        value = int(sample["frame"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _duration_seconds(samples: Sequence[Dict[str, Any]], fps: float) -> float:
    frames = [value for value in (_frame(sample) for sample in samples) if value is not None]
    return (max(frames) - min(frames)) / fps if len(frames) >= 2 else 0.0


def _confidence_label(mean_confidence: Optional[float], evidence_ratio: float) -> str:
    score = (mean_confidence or 0.0) * max(0.0, min(1.0, evidence_ratio))
    if score >= 0.72:
        return "high"
    if score >= 0.57:
        return "medium"
    return "low"


def _suggestion(
    identifier: str,
    category: str,
    title: str,
    message: str,
    observed: Dict[str, Any],
    threshold: Dict[str, Any],
    evidence: Dict[str, Any],
    confidence: str,
    caveat: str,
) -> Dict[str, Any]:
    return {
        "id": identifier,
        "category": category,
        "title": title,
        "message": message,
        "observed": observed,
        "threshold": threshold,
        "evidence": evidence,
        "confidence": confidence,
        "priority": "review" if confidence == "medium" else "practice",
        "caveat": caveat,
    }


def _ready_pose_metrics(samples: Sequence[Dict[str, Any]], min_conf: float) -> Tuple[List[float], List[float], List[float], List[float]]:
    knees: List[float] = []
    sway_positions: List[Tuple[float, float]] = []
    torso_lengths: List[float] = []
    torso_tilts: List[float] = []
    for sample in samples:
        for side in (_LEFT, _RIGHT):
            hip, knee, ankle = (_point(sample, side[name], min_conf) for name in ("hip", "knee", "ankle"))
            if hip and knee and ankle:
                angle = _angle(hip, knee, ankle)
                if angle is not None:
                    knees.append(angle)
        left_shoulder, right_shoulder = _point(sample, 5, min_conf), _point(sample, 6, min_conf)
        left_hip, right_hip = _point(sample, 11, min_conf), _point(sample, 12, min_conf)
        if left_shoulder and right_shoulder and left_hip and right_hip:
            shoulder_mid, hip_mid = _midpoint(left_shoulder, right_shoulder), _midpoint(left_hip, right_hip)
            torso = _distance(shoulder_mid, hip_mid)
            if torso >= 4.0:
                sway_positions.append(shoulder_mid)
                torso_lengths.append(torso)
                torso_tilts.append(math.degrees(math.atan2(abs(shoulder_mid[0] - hip_mid[0]), abs(shoulder_mid[1] - hip_mid[1]) + 1e-9)))
    return knees, torso_lengths, torso_tilts, [coord for point in sway_positions for coord in point]


def _torso_sway_ratio(samples: Sequence[Dict[str, Any]], min_conf: float) -> Tuple[Optional[float], Optional[float], int]:
    positions: List[Tuple[float, float]] = []
    lengths: List[float] = []
    tilts: List[float] = []
    for sample in samples:
        shoulder_a, shoulder_b = _point(sample, 5, min_conf), _point(sample, 6, min_conf)
        hip_a, hip_b = _point(sample, 11, min_conf), _point(sample, 12, min_conf)
        if not (shoulder_a and shoulder_b and hip_a and hip_b):
            continue
        shoulder_mid, hip_mid = _midpoint(shoulder_a, shoulder_b), _midpoint(hip_a, hip_b)
        length = _distance(shoulder_mid, hip_mid)
        if length < 4.0:
            continue
        positions.append(shoulder_mid)
        lengths.append(length)
        tilts.append(math.degrees(math.atan2(abs(shoulder_mid[0] - hip_mid[0]), abs(shoulder_mid[1] - hip_mid[1]) + 1e-9)))
    if len(positions) < 2:
        return None, None, len(positions)
    xs, ys = [p[0] for p in positions], [p[1] for p in positions]
    sway = math.hypot((_percentile(xs, 0.90) or 0) - (_percentile(xs, 0.10) or 0), (_percentile(ys, 0.90) or 0) - (_percentile(ys, 0.10) or 0))
    tilt_spread = (_percentile(tilts, 0.90) or 0.0) - (_percentile(tilts, 0.10) or 0.0)
    return sway / max(1.0, median(lengths)), tilt_spread, len(positions)


def _overhead_arm_metrics(samples: Sequence[Dict[str, Any]], min_conf: float) -> Tuple[List[float], List[float], List[float]]:
    """Use explicitly labelled overhead contacts only; compact strokes are valid otherwise."""
    angles: List[float] = []
    reaches: List[float] = []
    confidences: List[float] = []
    for sample in samples:
        if _phase(sample) != "contact" or str(sample.get("shot_context", "")).lower() != "overhead":
            continue
        side_name = str(sample.get("hitting_arm", "")).lower()
        sides = [_LEFT] if side_name == "left" else [_RIGHT] if side_name == "right" else [_LEFT, _RIGHT]
        candidates = []
        for side in sides:
            shoulder, elbow, wrist = (_point(sample, side[name], min_conf) for name in ("shoulder", "elbow", "wrist"))
            hip_a, hip_b = _point(sample, 11, min_conf), _point(sample, 12, min_conf)
            if not (shoulder and elbow and wrist and hip_a and hip_b):
                continue
            torso = _distance(_midpoint(shoulder, _point(sample, 6 if side is _LEFT else 5, min_conf) or shoulder), _midpoint(hip_a, hip_b))
            angle = _angle(shoulder, elbow, wrist)
            if angle is not None and torso >= 4.0:
                candidates.append((angle, _distance(shoulder, wrist) / torso, _sample_confidence(sample, list(side.values())) or 0.0))
        if candidates:
            # Without a labelled racket arm, the longer visible arm is the least
            # speculative proxy.  It still cannot create a prompt unless three
            # explicitly labelled overhead contacts agree.
            angle, reach, conf = max(candidates, key=lambda item: item[1])
            angles.append(angle)
            reaches.append(reach)
            confidences.append(conf)
    return angles, reaches, confidences


def _court_point(sample: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    return _finite_point(sample.get("court_xy_m"))


def _median_point(points: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    return (median([p[0] for p in points]), median([p[1] for p in points])) if points else None


def analyze_motion(
    samples: Sequence[Dict[str, Any]],
    *,
    fps: float,
    player_side: str,
    court_calibrated: bool = False,
    min_keypoint_confidence: float = MIN_KEYPOINT_CONFIDENCE,
) -> Dict[str, Any]:
    """Produce conservative coaching observations for one player's samples.

    ``samples`` must all belong to the same tracked player and should be sorted
    by ``frame``.  ``phase='ready'`` and ``phase='contact'`` are intentional
    caller-provided labels; this function will not infer those semantic phases
    from pose alone.  Recovery additionally requires calibrated ``court_xy_m``.
    """
    if not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    normalized = [dict(sample) for sample in samples if isinstance(sample, dict) and _frame(sample) is not None]
    normalized.sort(key=lambda sample: _frame(sample) or 0)
    side = str(player_side).lower()
    if side not in {"top", "bottom"}:
        raise ValueError("player_side must be 'top' or 'bottom'")
    min_conf = max(0.0, min(1.0, float(min_keypoint_confidence)))
    ready = [sample for sample in normalized if _phase(sample) == "ready"]
    contacts = [sample for sample in normalized if _phase(sample) == "contact"]
    suggestions: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, str]] = []

    all_confidences = []
    for sample in normalized:
        confidence = _sample_confidence(sample, range(5, 17))
        if confidence is not None:
            all_confidences.append(confidence)
    quality = {
        "input_samples": len(normalized),
        "duration_seconds": round(_duration_seconds(normalized, float(fps)), 3),
        "ready_samples": len(ready),
        "contact_samples": len(contacts),
        "mean_keypoint_confidence": round(sum(all_confidences) / len(all_confidences), 3) if all_confidences else None,
        "calibrated_court_coordinates": bool(court_calibrated),
    }

    # Preparation: restrict to caller-labelled low-intensity/ready moments.
    knees, _, _, _ = _ready_pose_metrics(ready, min_conf)
    ready_seconds = _duration_seconds(ready, float(fps))
    if len(knees) < MIN_READY_SAMPLES or ready_seconds < MIN_READY_SECONDS:
        suppressed.append({"id": "ready_knee_flexion", "reason": "insufficient_labelled_ready_pose"})
    else:
        straight_share = sum(angle >= 160.0 for angle in knees) / len(knees)
        mean_conf = median(all_confidences) if all_confidences else None
        confidence = _confidence_label(mean_conf, min(1.0, len(knees) / 24.0))
        if median(knees) >= 165.0 and straight_share >= 0.70 and confidence != "low":
            suggestions.append(_suggestion(
                "ready_knee_flexion", "preparation", "准备姿态可增加屈膝", 
                "在标记为准备的画面里，膝角较直。可在舒适、可控的范围内略微屈膝，让启动和制动更从容。",
                {"median_knee_angle_deg": round(median(knees), 1), "straight_frame_share": round(straight_share, 2)},
                {"median_angle_deg_gte": 165.0, "straight_share_gte": 0.70},
                {"joint_observations": len(knees), "ready_seconds": round(ready_seconds, 2)}, confidence,
                "这是单机位二维姿态的准备期观察，不代表关节活动度或伤病判断。",
            ))

    # Torso: movement frames are intentionally excluded; torso sway is normal during a stroke.
    sway, tilt_spread, torso_samples = _torso_sway_ratio(ready, min_conf)
    if torso_samples < MIN_READY_SAMPLES or ready_seconds < MIN_READY_SECONDS:
        suppressed.append({"id": "torso_stability", "reason": "insufficient_labelled_ready_pose"})
    else:
        mean_conf = median(all_confidences) if all_confidences else None
        confidence = _confidence_label(mean_conf, min(1.0, torso_samples / 24.0))
        if sway is not None and tilt_spread is not None and sway >= 0.32 and tilt_spread >= 15.0 and confidence != "low":
            suggestions.append(_suggestion(
                "torso_stability", "torso_control", "准备时躯干可更稳定", 
                "准备窗口中肩部相对骨盆摆动较大。练习小碎步时保持躯干安定，用髋膝完成重心调整，帮助下一拍衔接。",
                {"shoulder_sway_over_torso": round(sway, 2), "torso_tilt_spread_deg": round(tilt_spread, 1)},
                {"sway_over_torso_gte": 0.32, "tilt_spread_deg_gte": 15.0},
                {"pose_samples": torso_samples, "ready_seconds": round(ready_seconds, 2)}, confidence,
                "快速启动、镜头抖动和遮挡也会放大该信号。",
            ))

    # Arm extension: only explicit overhead contexts are reviewed.
    angles, reaches, arm_confidences = _overhead_arm_metrics(contacts, min_conf)
    if len(angles) < MIN_OVERHEAD_CONTACTS:
        suppressed.append({"id": "overhead_arm_extension", "reason": "fewer_than_three_high_confidence_overhead_contacts"})
    else:
        # Three contacts are sufficient to raise a *review* prompt, while six
        # independent observations are required before the evidence quantity
        # can contribute its full weight to a high-confidence label.
        evidence_ratio = 0.70 + 0.30 * min(1.0, (len(angles) - MIN_OVERHEAD_CONTACTS) / 3.0)
        confidence = _confidence_label(median(arm_confidences) if arm_confidences else None, evidence_ratio)
        if median(angles) < 145.0 and median(reaches) < 1.05 and confidence != "low":
            suggestions.append(_suggestion(
                "overhead_arm_extension", "stroke_shape", "上手击球伸展", 
                "上手击球时拉开引拍、触球点和随挥空间，肩膀保持放松。",
                {"median_elbow_angle_deg": round(median(angles), 1), "median_wrist_shoulder_over_torso": round(median(reaches), 2)},
                {"median_elbow_angle_deg_lt": 145.0, "median_reach_over_torso_lt": 1.05},
                {"overhead_contacts": len(angles)}, confidence,
                "平抽、封网和刻意的紧凑击球并不需要大幅伸展；本提示只适用于已标记的上手目标。",
            ))

    # Recovery: use floor-plane positions and an explicit ready-position baseline only.
    ready_court = [_court_point(sample) for sample in ready]
    base = _median_point([point for point in ready_court if point is not None])
    hit_samples = [sample for sample in normalized if bool(sample.get("hit")) and _court_point(sample) is not None]
    opportunities = []
    if not court_calibrated:
        suppressed.append({"id": "movement_recovery", "reason": "court_coordinates_not_calibrated"})
    elif base is None:
        suppressed.append({"id": "movement_recovery", "reason": "no_ready_position_baseline"})
    else:
        for hit in hit_samples:
            hit_frame, hit_point = _frame(hit), _court_point(hit)
            if hit_frame is None or hit_point is None or _distance(hit_point, base) < 0.90:
                continue
            after = []
            for sample in normalized:
                frame, point = _frame(sample), _court_point(sample)
                if frame is None or point is None:
                    continue
                elapsed = (frame - hit_frame) / float(fps)
                if 0.55 <= elapsed <= 1.25:
                    after.append(point)
            if len(after) < 3:
                continue
            final_distance = _distance(_median_point(after) or after[-1], base)
            path_length = sum(_distance(a, b) for a, b in zip(after, after[1:]))
            opportunities.append((final_distance, path_length, hit_frame))
        if len(opportunities) < MIN_RECOVERY_OPPORTUNITIES:
            suppressed.append({"id": "movement_recovery", "reason": "fewer_than_three_observable_far_court_hits"})
        else:
            unrecovered = [item for item in opportunities if item[0] >= 0.75 and item[1] < 0.35]
            mean_conf = median(all_confidences) if all_confidences else None
            evidence_ratio = 0.70 + 0.30 * min(1.0, (len(opportunities) - MIN_RECOVERY_OPPORTUNITIES) / 3.0)
            confidence = _confidence_label(mean_conf, evidence_ratio)
            if len(unrecovered) / len(opportunities) >= 0.67 and confidence != "low":
                suggestions.append(_suggestion(
                    "movement_recovery", "movement_recovery", "击球后的回位可更早启动", 
                    "多次远离准备位置的击球后，在可观察的 0.55–1.25 秒内没有看到明显回位。可练习落地后的第一步和回到自己准备区域的节奏，再衔接下一拍。",
                    {"unrecovered_opportunity_share": round(len(unrecovered) / len(opportunities), 2), "median_remaining_distance_m": round(median([item[0] for item in unrecovered]), 2)},
                    {"unrecovered_share_gte": 0.67, "remaining_distance_m_gte": 0.75, "post_hit_path_m_lt": 0.35},
                    {"opportunities": len(opportunities), "unrecovered": len(unrecovered), "base_position_m": [round(base[0], 2), round(base[1], 2)]}, confidence,
                    "回位目标会随战术、搭档与来球而变；该提示只描述相对于已标记准备位置的可见移动。",
                ))

    status = "ok" if normalized else "insufficient_evidence"
    return {
        "schema_version": SCHEMA_VERSION,
        "player_side": side,
        "status": status,
        "disclaimer": "仅基于视频二维关键点的训练观察，不构成医疗、伤病或专业技术诊断。",
        "quality": quality,
        "suggestions": suggestions,
        "suppressed_checks": suppressed,
    }
