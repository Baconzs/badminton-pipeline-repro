"""Optional adapter for the open-source BST stroke-type model.

The main video pipeline deliberately does not vendor BST (or its research
dependencies).  A user can clone the upstream repository and place a trained
checkpoint in ``weights/``; this module then turns the pose/ball/court signals
already produced by the Web UI into BST inputs.  If either the repository or
checkpoint is absent, the adapter returns a path-free ``not_configured``
report instead of making the professional job fail.

BST is a stroke *type* classifier.  It is not a technical-quality or medical
assessment model.  The report therefore keeps model confidence, evidence
coverage, and the existing rule candidate separate.
"""

from __future__ import annotations

import csv
import importlib
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PLAYER_LABELS = {"near": "下方球员", "far": "上方球员"}
_POSE_STYLE = "JnB_bone"
_DEFAULT_SEQ_LEN = 100
_DEFAULT_WINDOW_SECONDS = 1.6
_DEFAULT_MAX_EVENTS = 512
_MAX_TOPK = 3

# BST cards and Motion Coach intentionally use different evidence policies.
# The research classifier can remain useful as a visible/auditable type
# candidate with sparse broadcast evidence.  A type that drives a technical
# suggestion must be much more selective: high Top-1 probability, clear
# separation from Top-2, and a substantially complete pose/ball window.
_TYPE_MIN_CONFIDENCE = 0.60
_TYPE_MIN_POSE_COVERAGE = 0.08
_TYPE_MIN_BALL_COVERAGE = 0.08
_COACH_MIN_CONFIDENCE = 0.85
_COACH_MIN_MARGIN = 0.25
_COACH_MIN_POSE_COVERAGE = 0.85
_COACH_MIN_BALL_COVERAGE = 0.55


# The order is copied from the upstream ShuttleSet helper.  Keeping the
# mapping local means the Web UI can still decode a checkpoint when the
# research repository is installed in a separate virtual environment.
_SHUTTLESET_12 = [
    "放小球", "擋小球", "殺球", "挑球", "長球", "平球",
    "切球", "推球", "撲球", "勾球", "發短球", "發長球",
]
_SHUTTLESET_17 = [
    "放小球", "擋小球", "殺球", "點扣", "挑球", "防守回挑",
    "長球", "平球", "後場抽平球", "切球", "過渡切球", "推球",
    "撲球", "防守回抽", "勾球", "發短球", "發長球",
]
_BADMINTONDB_18 = [
    "Bottom-Block", "Bottom-Clear", "Bottom-Drive", "Bottom-Dropshot",
    "Bottom-Net-Kill", "Bottom-Net-Lift", "Bottom-Net-Shot", "Bottom-Serve",
    "Bottom-Smash", "Top-Block", "Top-Clear", "Top-Drive",
    "Top-Dropshot", "Top-Net-Kill", "Top-Net-Lift", "Top-Net-Shot",
    "Top-Serve", "Top-Smash",
]

_FAMILY_LABELS = {
    "放小球": "搓放网",
    "擋小球": "挡网",
    "殺球": "杀球",
    "點扣": "点扣",
    "挑球": "挑球",
    "防守回挑": "防守回挑",
    "長球": "高远球",
    "平球": "平球",
    "後場抽平球": "后场平抽",
    "切球": "吊球",
    "過渡切球": "过渡吊球",
    "推球": "推球",
    "撲球": "扑球",
    "防守回抽": "防守回抽",
    "勾球": "勾对角",
    "發短球": "短发球",
    "發長球": "长发球",
    "Bottom-Block": "挡网",
    "Bottom-Clear": "高远球",
    "Bottom-Drive": "平抽",
    "Bottom-Dropshot": "吊球",
    "Bottom-Net-Kill": "扑球",
    "Bottom-Net-Lift": "挑球",
    "Bottom-Net-Shot": "网前球",
    "Bottom-Serve": "发球",
    "Bottom-Smash": "杀球",
    "Top-Block": "挡网",
    "Top-Clear": "高远球",
    "Top-Drive": "平抽",
    "Top-Dropshot": "吊球",
    "Top-Net-Kill": "扑球",
    "Top-Net-Lift": "挑球",
    "Top-Net-Shot": "网前球",
    "Top-Serve": "发球",
    "Top-Smash": "杀球",
}


def _finite(value: object, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _integer(value: object, default: int = 0) -> int:
    number = _finite(value)
    if number is None:
        return default
    try:
        return int(number)
    except (TypeError, ValueError, OverflowError):
        return default


def _probability(value: object, default: float = 0.0) -> float:
    """Normalize a probability supplied either as 0..1 or as a percent."""
    number = _finite(value, default)
    if number is None:
        return default
    if number > 1.0:
        number /= 100.0
    return max(0.0, min(1.0, float(number)))


def assess_bst_hit_gates(hit: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable type/report and Motion-Coach readiness metadata.

    The Top-1/Top-2 separation must come from the decoded probability vector;
    a standalone confidence number cannot prove that the winning class is
    well separated.  This deliberately rejects legacy/custom cards without
    at least two probability entries from driving coaching advice while still
    leaving those cards available in the BST diagnostic report.
    """
    raw_top = hit.get("top3", []) if isinstance(hit, Mapping) else []
    probabilities: list[float] = []
    if isinstance(raw_top, (list, tuple)):
        for candidate in raw_top:
            if not isinstance(candidate, Mapping):
                continue
            probability = _finite(candidate.get("probability"))
            if probability is None:
                continue
            probabilities.append(_probability(probability))
    probabilities.sort(reverse=True)
    top1 = probabilities[0] if probabilities else _probability(hit.get("confidence", 0.0))
    margin_available = len(probabilities) >= 2
    margin = max(0.0, top1 - probabilities[1]) if margin_available else 0.0
    pose_coverage = _probability(hit.get("pose_coverage", 0.0))
    ball_coverage = _probability(hit.get("ball_coverage", 0.0))
    confirmed = hit.get("event_confirmed", False) is True

    type_reason = "ready"
    if top1 < _TYPE_MIN_CONFIDENCE:
        type_reason = "low_confidence"
    elif not confirmed:
        type_reason = "event_unconfirmed"
    elif pose_coverage < _TYPE_MIN_POSE_COVERAGE:
        type_reason = "low_pose_coverage"
    elif ball_coverage < _TYPE_MIN_BALL_COVERAGE:
        type_reason = "low_ball_coverage"
    type_ready = type_reason == "ready"

    coach_reason = "ready"
    if top1 < _COACH_MIN_CONFIDENCE:
        coach_reason = "low_confidence"
    elif not margin_available:
        coach_reason = "margin_unavailable"
    elif margin < _COACH_MIN_MARGIN:
        coach_reason = "low_margin"
    elif not confirmed:
        coach_reason = "event_unconfirmed"
    elif pose_coverage < _COACH_MIN_POSE_COVERAGE:
        coach_reason = "low_pose_coverage"
    elif ball_coverage < _COACH_MIN_BALL_COVERAGE:
        coach_reason = "low_ball_coverage"
    coach_ready = coach_reason == "ready"
    return {
        "top1_probability": round(top1, 4),
        "top1_top2_margin": round(margin, 4),
        "margin_available": margin_available,
        "type_ready": type_ready,
        "type_gate_reason": type_reason,
        "coach_ready": coach_ready,
        "coach_gate_reason": coach_reason,
    }


def _gate_policy() -> dict[str, Any]:
    """Return a fresh JSON-safe copy of the selective gate contract."""
    return {
        "version": 1,
        "type_report": {
            "min_probability": _TYPE_MIN_CONFIDENCE,
            "min_pose_coverage": _TYPE_MIN_POSE_COVERAGE,
            "min_ball_coverage": _TYPE_MIN_BALL_COVERAGE,
        },
        "motion_coach": {
            "min_probability": _COACH_MIN_CONFIDENCE,
            "min_top1_top2_margin": _COACH_MIN_MARGIN,
            "min_pose_coverage": _COACH_MIN_POSE_COVERAGE,
            "min_ball_coverage": _COACH_MIN_BALL_COVERAGE,
        },
    }


def _timecode(seconds: float) -> str:
    minutes, remainder = divmod(int(max(0.0, seconds)), 60)
    return f"{minutes:02d}:{remainder:02d}"


def _env_first(*names: str) -> str:
    for name in names:
        value = str(os.environ.get(name, "")).strip()
        if value:
            return value
    return ""


def _path_from_env(*names: str) -> Path | None:
    raw = _env_first(*names)
    if not raw:
        return None
    try:
        path = Path(raw).expanduser().resolve()
    except (OSError, ValueError):
        return None
    return path


def _first_file(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path is None:
            continue
        try:
            if path.is_file():
                return path.resolve()
        except OSError:
            continue
    return None


def _first_dir(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path is None:
            continue
        try:
            if path.is_dir():
                return path.resolve()
        except OSError:
            continue
    return None


def _class_labels(dataset: str, repo: Path | None = None) -> list[str]:
    normalized = str(dataset or "").strip().lower().replace("-", "_")
    if normalized in {"shuttleset", "shuttleset_25", "shuttle_25", "merged"}:
        return ["未知球种"] + [f"Top_{item}" for item in _SHUTTLESET_12] + [f"Bottom_{item}" for item in _SHUTTLESET_12]
    if normalized in {"shuttleset_35", "shuttle_35", "full"}:
        return [f"Top_{item}" for item in _SHUTTLESET_17] + [f"Bottom_{item}" for item in _SHUTTLESET_17] + ["未知球种"]
    if normalized in {"badmintondb", "badmintondb_18", "badminton_db"}:
        # Prefer the CSV shipped by the upstream repository, because it is
        # the authoritative class-id ordering for that dataset.
        if repo is not None:
            csv_path = repo / "BadmintonDB" / "after_generating" / "merged_strokes.csv"
            try:
                with csv_path.open("r", encoding="utf-8", newline="") as stream:
                    rows = csv.DictReader(stream)
                    by_id: dict[int, str] = {}
                    for row in rows:
                        class_id = _integer(row.get("class_id"), -1)
                        label = str(row.get("merged_type", "")).strip()
                        if class_id >= 0 and label:
                            by_id.setdefault(class_id, label)
                    if by_id and set(by_id) == set(range(max(by_id) + 1)):
                        return [by_id[index] for index in range(max(by_id) + 1)]
            except (OSError, UnicodeError, csv.Error):
                pass
        return list(_BADMINTONDB_18)
    return []


def _friendly_label(raw_label: str) -> str:
    value = str(raw_label or "").strip()
    if value in {"未知球種", "unknown", "Unknown"}:
        return "未知球种"
    side_removed = value
    for prefix in ("Top_", "Bottom_", "Top-", "Bottom-"):
        if side_removed.startswith(prefix):
            side_removed = side_removed[len(prefix):]
            break
    return _FAMILY_LABELS.get(value, _FAMILY_LABELS.get(side_removed, side_removed or "待复核"))


def _label_side(raw_label: str) -> str | None:
    value = str(raw_label or "")
    if value.startswith(("Top_", "Top-")):
        return "far"
    if value.startswith(("Bottom_", "Bottom-")):
        return "near"
    return None


def resolve_bst_configuration(repo_root: Path | str) -> dict[str, Any]:
    """Resolve optional BST paths/settings without exposing absolute paths."""
    root = Path(repo_root).resolve()
    configured_repo = _path_from_env("BADMINTON_BST_REPO", "BST_REPO")
    research_repo = _first_dir(
        [
            configured_repo,
            root / "third_party" / "BST-Badminton-Stroke-type-Transformer",
            root / "external" / "BST-Badminton-Stroke-type-Transformer",
            root / "BST-Badminton-Stroke-type-Transformer",
        ]
    )
    configured_weights = _path_from_env("BADMINTON_BST_WEIGHTS", "BST_WEIGHTS")
    explicit_weight = configured_weights is not None
    weight_candidates = [configured_weights] if explicit_weight else []
    weight_root = root / "weights"
    for pattern in ("bst*.pt", "BST*.pt", "*bst*.pth", "*BST*.pth"):
        try:
            weight_candidates.extend(sorted(weight_root.glob(pattern)))
        except OSError:
            pass
    dataset_env = _env_first("BADMINTON_BST_DATASET", "BST_DATASET")
    dataset = dataset_env or "shuttleset_25"
    model_env = _env_first("BADMINTON_BST_MODEL", "BST_MODEL")
    raw_model_name = model_env or "BST_CG_AP"
    model_aliases = {
        "bst": "BST", "bst_0": "BST_0", "bst_cg": "BST_CG",
        "bst_ap": "BST_AP", "bst_cg_ap": "BST_CG_AP",
    }
    model_name = model_aliases.get(raw_model_name.lower().replace("-", "_"), raw_model_name)
    pose_style = _env_first("BADMINTON_BST_POSE_STYLE", "BST_POSE_STYLE") or _POSE_STYLE
    try:
        seq_len = max(8, min(512, int(_env_first("BADMINTON_BST_SEQ_LEN", "BST_SEQ_LEN") or _DEFAULT_SEQ_LEN)))
    except (TypeError, ValueError):
        seq_len = _DEFAULT_SEQ_LEN
    try:
        window_seconds = max(0.35, min(8.0, float(_env_first("BADMINTON_BST_WINDOW_SECONDS", "BST_WINDOW_SECONDS") or _DEFAULT_WINDOW_SECONDS)))
    except (TypeError, ValueError):
        window_seconds = _DEFAULT_WINDOW_SECONDS
    try:
        max_events = max(1, min(2048, int(_env_first("BADMINTON_BST_MAX_EVENTS", "BST_MAX_EVENTS") or _DEFAULT_MAX_EVENTS)))
    except (TypeError, ValueError):
        max_events = _DEFAULT_MAX_EVENTS
    # When the operator did not explicitly choose a checkpoint, avoid an
    # arbitrary lexicographic pick if several research weights are present.
    # A uniquely recognizable filename is safe to auto-select and can also
    # supply the model/dataset metadata omitted from bare state_dict files.
    if not explicit_weight:
        unique_candidates = []
        seen = set()
        for candidate in weight_candidates:
            if candidate is None:
                continue
            try:
                key = str(candidate.resolve())
            except OSError:
                continue
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)
        if len(unique_candidates) == 1:
            weights = unique_candidates[0]
        elif unique_candidates:
            normalized_dataset = str(dataset).lower().replace("-", "_")
            desired_model = str(model_name).lower()

            def candidate_score(candidate: Path) -> int:
                name = candidate.name.lower()
                score = 0
                if normalized_dataset in {"shuttleset_25", "shuttle_25", "merged"} and "merged" in name:
                    score += 4
                if normalized_dataset in {"shuttleset_35", "shuttle_35", "full"} and "merged" not in name:
                    score += 3
                if normalized_dataset.startswith("badminton") and "bad" in name:
                    score += 4
                if desired_model and desired_model in name:
                    score += 3
                if str(pose_style).lower() in name:
                    score += 1
                if f"seq_{seq_len}" in name:
                    score += 1
                return score

            scored = sorted(((candidate_score(item), item) for item in unique_candidates), key=lambda pair: pair[0], reverse=True)
            if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]) and scored[0][0] > 0:
                weights = scored[0][1]
            else:
                weights = None
        else:
            weights = None
    else:
        # An explicit path is authoritative; do not silently substitute a
        # different checkpoint found under ``weights/``.
        weights = configured_weights if configured_weights is not None and configured_weights.is_file() else None
    if weights is not None and not model_env:
        filename = weights.name.lower()
        for token, alias in (("bst_cg_ap", "BST_CG_AP"), ("bst_cg", "BST_CG"), ("bst_ap", "BST_AP"), ("bst_0", "BST_0"), ("bst_", "BST")):
            if token in filename:
                model_name = alias
                break
    if weights is not None and not dataset_env:
        filename = weights.name.lower()
        if "badminton" in filename:
            dataset = "badmintondb_18"
        elif "merged" in filename:
            dataset = "shuttleset_25"
        elif "shuttle" in filename:
            dataset = "shuttleset_35"
    labels = _class_labels(dataset, research_repo)
    return {
        "configured": bool(research_repo and weights and labels),
        "repository_found": bool(research_repo),
        "weights_found": bool(weights),
        "dataset": str(dataset),
        "model": str(model_name),
        "pose_style": str(pose_style),
        "class_count": len(labels),
        "seq_len": seq_len,
        "window_seconds": window_seconds,
        "max_events": max_events,
        # Internal-only paths.  Callers must not serialize this dictionary
        # directly into a public job manifest.
        "_repo_path": research_repo,
        "_weights_path": weights,
        "_labels": labels,
    }


def public_bst_configuration(repo_root: Path | str) -> dict[str, Any]:
    """Return the small safe subset used by ``/api/health``."""
    config = resolve_bst_configuration(repo_root)
    return {
        "configured": bool(config["configured"]),
        "repository_found": bool(config["repository_found"]),
        "weights_found": bool(config["weights_found"]),
        "dataset": config["dataset"],
        "model": config["model"],
        "pose_style": config["pose_style"],
        "class_count": config["class_count"],
    }


def _read_sidecar(path: Path) -> tuple[list[dict[str, Any]], dict[int, tuple[float, float]], int]:
    events: list[dict[str, Any]] = []
    measured: dict[int, tuple[float, float]] = {}
    maximum_frame = -1
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            frame = _integer(row.get("Frame"), -1)
            if frame < 0:
                continue
            maximum_frame = max(maximum_frame, frame)
            source = str(row.get("RawSource", "")).strip().lower()
            if not source:
                source = str(row.get("Source", "")).strip().lower()
            visibility = _integer(row.get("RawVisibility"), _integer(row.get("Visibility"), 0))
            x = _finite(row.get("RawX"), _finite(row.get("X")))
            y = _finite(row.get("RawY"), _finite(row.get("Y")))
            if source in {"model", "classical"} and visibility > 0 and x is not None and y is not None and x >= 0 and y >= 0 and (x > 0 or y > 0):
                measured[frame] = (x, y)
            if _integer(row.get("Hit"), 0) <= 0:
                continue
            hitter = str(row.get("HitHitter", "")).strip().lower()
            if hitter not in {"near", "far"}:
                hitter = "unknown"
            event = {
                "frame": frame,
                "seconds": 0.0,
                "hitter": hitter,
                "rally_id": max(0, _integer(row.get("RallyID"), 0)),
                "hit_index": max(0, _integer(row.get("HitRallyIndex"), 0)),
                "hit_score": max(0.0, min(1.0, _finite(row.get("HitScore"), 0.0) or 0.0)),
                "confirmed": _integer(row.get("HitBallObserved"), 0) > 0 and _integer(row.get("HitUncertain"), 0) <= 0,
            }
            events.append(event)
    events.sort(key=lambda item: item["frame"])
    return events, measured, maximum_frame


def _sample_frames(start: int, end: int, anchor: int, target_len: int) -> tuple[list[int], int]:
    start = max(0, int(start))
    end = max(start, int(end))
    target_len = max(1, int(target_len))
    raw_len = end - start + 1
    if raw_len <= target_len:
        indices = list(range(start, end + 1))
    else:
        # Mirror ``make_seq_len_same`` in the upstream repository: retain an
        # integer temporal stride and right-pad when the stride yields fewer
        # than ``target_len`` observations.  This preserves the distribution
        # expected by released checkpoints better than arbitrary interpolation.
        need_padding = (raw_len % target_len) > (target_len // 2)
        stride = raw_len // target_len + int(need_padding)
        indices = list(range(start, end + 1, max(1, stride)))[:target_len]
        nearest = min(range(len(indices)), key=lambda index: abs(indices[index] - anchor))
        indices[nearest] = max(start, min(end, int(anchor)))
        indices = [value for index, value in enumerate(indices) if index == 0 or value != indices[index - 1]]
    video_len = min(target_len, len(indices))
    indices.extend([0] * (target_len - len(indices)))
    return indices[:target_len], video_len


def _valid_keypoint(point: object, confidence_threshold: float = 0.28) -> tuple[float, float] | None:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    x, y = _finite(point[0]), _finite(point[1])
    confidence = _finite(point[2], 1.0) if len(point) >= 3 else 1.0
    if x is None or y is None or confidence is None or confidence < confidence_threshold:
        return None
    return x, y


def _normalize_pose(raw_pose: Mapping[str, Any] | None) -> tuple[list[list[float]], bool]:
    """Apply BST's bbox-diagonal, center-aligned COCO-17 normalization."""
    zero = [[0.0, 0.0] for _ in range(17)]
    if not isinstance(raw_pose, Mapping):
        return zero, False
    keypoints = raw_pose.get("keypoints")
    box = raw_pose.get("box")
    if not isinstance(keypoints, Sequence) or len(keypoints) < 17 or not isinstance(box, Sequence) or len(box) < 4:
        return zero, False
    try:
        x1, y1, x2, y2 = (float(box[index]) for index in range(4))
    except (TypeError, ValueError, IndexError):
        return zero, False
    diagonal = math.hypot(x2 - x1, y2 - y1)
    if not math.isfinite(diagonal) or diagonal < 1e-4:
        return zero, False
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    output: list[list[float]] = []
    valid = 0
    for point in keypoints[:17]:
        value = _valid_keypoint(point)
        if value is None:
            output.append([0.0, 0.0])
            continue
        valid += 1
        output.append([(value[0] - center_x) / diagonal, (value[1] - center_y) / diagonal])
    return output, valid >= 4


_COCO_BONE_PAIRS = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
]


def _interpolate_joints(joints: Any) -> Any:
    """Return joints plus the 19 midpoint channels used by BST."""
    import numpy as np

    midpoints = []
    for start, end in _COCO_BONE_PAIRS:
        first = joints[:, :, :, start, :]
        second = joints[:, :, :, end, :]
        midpoints.append(np.where((first != 0.0) & (second != 0.0), (first + second) / 2.0, 0.0))
    return np.concatenate((joints, np.stack(midpoints, axis=-2)), axis=-2)


def _add_bones(joints: Any) -> Any:
    """Return ``JnB_bone`` using the exact 19 COCO pairs from BST."""
    import numpy as np
    return np.concatenate((joints, _bone_channels(joints)), axis=-2)


def _bone_channels(joints: Any) -> Any:
    """Return only the 19 bone channels (batched input)."""
    import numpy as np
    bones = []
    for start, end in _COCO_BONE_PAIRS:
        # ``joints`` is batched here: (batch, time, players, joints, xy).
        first = joints[:, :, :, start, :]
        second = joints[:, :, :, end, :]
        bones.append(np.where((first != 0.0) & (second != 0.0), second - first, 0.0))
    return np.stack(bones, axis=-2)


def build_bst_sequences(
    events: Sequence[Mapping[str, Any]],
    pose_by_frame: Mapping[int, Mapping[str, Mapping[str, Any]]],
    measured_ball: Mapping[int, tuple[float, float]],
    *,
    fps: float,
    width: int,
    height: int,
    court_width_m: float = 6.10,
    court_length_m: float = 13.40,
    seq_len: int = _DEFAULT_SEQ_LEN,
    window_seconds: float = _DEFAULT_WINDOW_SECONDS,
    max_events: int = _DEFAULT_MAX_EVENTS,
    pose_style: str = _POSE_STYLE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build fixed-length BST samples and path-free event metadata.

    The returned feature arrays are intentionally kept internal to the
    adapter.  Tests and an optional custom predictor may inspect them, but no
    raw skeleton is written into the job manifest.
    """
    import numpy as np

    ordered = sorted((dict(event) for event in events if isinstance(event, Mapping)), key=lambda item: _integer(item.get("frame"), -1))
    total = len(ordered)
    if total > max_events:
        stride = total / float(max_events)
        selected = [ordered[min(total - 1, int(index * stride))] for index in range(max_events)]
    else:
        selected = ordered
    seq_len = max(8, int(seq_len))
    fps = max(1.0, float(fps))
    width = max(1, int(width))
    height = max(1, int(height))
    radius = max(1, int(round(max(0.35, float(window_seconds)) * fps)))
    poses = np.zeros((len(selected), seq_len, 2, 17, 2), dtype=np.float32)
    positions = np.zeros((len(selected), seq_len, 2, 2), dtype=np.float32)
    shuttles = np.zeros((len(selected), seq_len, 2), dtype=np.float32)
    lengths = np.zeros((len(selected),), dtype=np.int64)
    metadata: list[dict[str, Any]] = []
    for sample_index, event in enumerate(selected):
        frame = _integer(event.get("frame"), -1)
        indices, video_len = _sample_frames(frame - radius, frame + radius, frame, seq_len)
        lengths[sample_index] = video_len
        pose_valid = 0
        ball_valid = 0
        for time_index, source_frame in enumerate(indices[:video_len]):
            if source_frame <= 0 and not (frame - radius <= 0):
                continue
            frame_poses = pose_by_frame.get(int(source_frame), {})
            for player_index, side in enumerate(("far", "near")):
                raw_pose = frame_poses.get(side)
                normalized, usable = _normalize_pose(raw_pose)
                poses[sample_index, time_index, player_index] = np.asarray(normalized, dtype=np.float32)
                raw_position = raw_pose.get("court_position") if isinstance(raw_pose, Mapping) else None
                if isinstance(raw_position, (list, tuple)) and len(raw_position) >= 2:
                    x = _finite(raw_position[0])
                    y = _finite(raw_position[1])
                    if x is not None and y is not None:
                        positions[sample_index, time_index, player_index] = (
                            np.clip(x / max(1e-6, court_width_m), -0.25, 1.25),
                            np.clip(y / max(1e-6, court_length_m), -0.25, 1.25),
                        )
                if usable:
                    pose_valid += 1
            ball = measured_ball.get(int(source_frame))
            if isinstance(ball, (list, tuple)) and len(ball) >= 2:
                x, y = _finite(ball[0]), _finite(ball[1])
                if x is not None and y is not None and x >= 0 and y >= 0:
                    shuttles[sample_index, time_index] = (
                        np.clip(x / width, 0.0, 1.0),
                        np.clip(y / height, 0.0, 1.0),
                    )
                    ball_valid += 1
        metadata.append({
            "frame": max(0, frame),
            "seconds": round(max(0.0, frame / fps), 2),
            "timecode": _timecode(frame / fps),
            "rally_id": max(0, _integer(event.get("rally_id"), 0)),
            "hit_index": max(0, _integer(event.get("hit_index"), 0)),
            "hitter": str(event.get("hitter", "unknown")) if str(event.get("hitter", "unknown")) in {"near", "far"} else "unknown",
            "event_confirmed": bool(event.get("confirmed", False)),
            "pose_coverage": round(pose_valid / max(1, video_len * 2), 3),
            "ball_coverage": round(ball_valid / max(1, video_len), 3),
            "window_start": round(max(0.0, (frame - radius) / fps), 2),
            "window_end": round((frame + radius) / fps, 2),
        })
    # Augment channels exactly as the upstream preprocessing script does.
    normalized_style = str(pose_style or _POSE_STYLE)
    if normalized_style == "J_only":
        human_pose = poses
    elif normalized_style == "JnB_interp":
        human_pose = _interpolate_joints(poses)
    elif normalized_style == "Jn2B":
        interpolated = _interpolate_joints(poses)
        human_pose = __import__("numpy").concatenate((interpolated, _bone_channels(interpolated)), axis=-2)
    else:
        human_pose = _add_bones(poses)
    return {
        "human_pose": human_pose,
        "positions": positions,
        "shuttles": shuttles,
        "video_lengths": lengths,
    }, metadata


@contextmanager
def _bst_import_context(repo: Path):
    source_root = repo / "stroke_classification"
    inserted = [str(source_root), str(repo)]
    old_path = list(sys.path)
    for item in reversed(inserted):
        if item not in sys.path:
            sys.path.insert(0, item)
    try:
        yield
    finally:
        sys.path[:] = old_path


def _load_bst_model(config: Mapping[str, Any], device: str):
    """Load an upstream model/checkpoint lazily; raises a useful RuntimeError."""
    import torch

    repo = config.get("_repo_path")
    weight = config.get("_weights_path")
    if not isinstance(repo, Path) or not isinstance(weight, Path):
        raise RuntimeError("BST 仓库或权重未配置")
    model_name = str(config.get("model", "BST_CG_AP"))
    pose_style = str(config.get("pose_style", _POSE_STYLE))
    if pose_style not in {"JnB_bone", "J_only", "JnB_interp", "Jn2B"}:
        raise RuntimeError(f"BST pose_style 不支持：{pose_style}")
    labels = list(config.get("_labels", []))
    if not labels:
        raise RuntimeError("BST 数据集类别映射为空")
    requested = str(device or "auto").lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA，但当前 PyTorch 没有可用 CUDA")
        torch_device = torch.device("cuda")
    elif requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("请求 MPS，但当前 PyTorch 没有可用 MPS")
        torch_device = torch.device("mps")
    elif requested == "cpu":
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extra = {"J_only": 0, "JnB_bone": 1, "JnB_interp": 1, "Jn2B": 2}[pose_style]
    in_dim = (17 + 19 * extra) * 2
    seq_len = int(config.get("seq_len", _DEFAULT_SEQ_LEN))
    classes = {
        "BST": "BST",
        "BST_0": "BST_0",
        "BST_CG": "BST_CG",
        "BST_AP": "BST_AP",
        "BST_CG_AP": "BST_CG_AP",
    }
    if model_name not in classes:
        raise RuntimeError(f"BST 模型架构不支持：{model_name}")
    with _bst_import_context(repo):
        module = importlib.import_module("model.bst")
        architecture = getattr(module, classes[model_name], None)
        if architecture is None:
            raise RuntimeError(f"BST 仓库中没有架构：{model_name}")
        network = architecture(
            in_dim=in_dim,
            n_class=len(labels),
            seq_len=seq_len,
            depth_tem=2,
            depth_inter=1,
        ).to(torch_device)
    try:
        try:
            state = torch.load(str(weight), map_location=torch_device, weights_only=True)
        except TypeError:
            state = torch.load(str(weight), map_location=torch_device)
    except Exception as exc:
        raise RuntimeError(f"无法读取 BST 权重：{exc}") from exc
    if isinstance(state, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = state.get(key)
            if isinstance(candidate, Mapping):
                state = candidate
                break
    if not isinstance(state, Mapping):
        raise RuntimeError("BST 权重不是 state_dict")
    cleaned = {str(key).removeprefix("module."): value for key, value in state.items()}
    try:
        network.load_state_dict(cleaned, strict=True)
    except Exception as exc:
        raise RuntimeError(f"BST 权重与 {model_name}/{len(labels)} 类别不匹配：{exc}") from exc
    network.eval()
    return network, torch_device, labels


def _decode_logits(logits: Any, labels: Sequence[str], metadata: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Decode logits into bounded, JSON-safe Top-3 event cards."""
    import numpy as np

    values = logits
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().numpy()
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] != len(metadata):
        raise ValueError("BST logits shape 与击球事件数量不一致")
    outputs = []
    for row, item in zip(values, metadata):
        row = row - np.max(row)
        probabilities = np.exp(row)
        denominator = float(np.sum(probabilities))
        if not math.isfinite(denominator) or denominator <= 0:
            probabilities = np.zeros_like(row)
        else:
            probabilities /= denominator
        order = np.argsort(-probabilities)[: min(_MAX_TOPK, len(labels))]
        top = []
        for class_id in order:
            index = int(class_id)
            raw_label = str(labels[index])
            probability = float(probabilities[index])
            top.append({
                "class_id": index,
                "raw_label": raw_label,
                "label": _friendly_label(raw_label),
                "probability": round(max(0.0, min(1.0, probability)), 4),
                "confidence": round(max(0.0, min(100.0, probability * 100.0)), 1),
                "model_side": _label_side(raw_label),
            })
        best = top[0] if top else {"class_id": -1, "raw_label": "", "label": "待复核", "probability": 0.0, "confidence": 0.0, "model_side": None}
        event = dict(item)
        event.update({
            "model_label": best["label"],
            "model_raw_label": best["raw_label"],
            "model_side": best.get("model_side"),
            "confidence": best["confidence"],
            "top3": top,
        })
        gate = assess_bst_hit_gates(event)
        event.update(gate)
        # ``confidence_level`` remains the display/report gate for backwards
        # compatibility.  Motion Coach consumes the independent, stricter
        # ``coach_ready`` gate instead.
        event["confidence_level"] = "ready" if gate["type_ready"] else "review"
        if not gate["type_ready"]:
            event["review_reason"] = {
                "low_confidence": "BST 类型置信度不足",
                "event_unconfirmed": "击球事件本身未达到确认门槛",
                "low_pose_coverage": "击球窗口可见姿态不足",
                "low_ball_coverage": "击球窗口真实球点不足",
            }.get(str(gate["type_gate_reason"]), "BST 类型证据不足")
        outputs.append(event)
    return outputs


def decode_bst_logits(logits: Any, labels: Sequence[str], metadata: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Public, dependency-light wrapper used by regression tests/custom heads."""
    return _decode_logits(logits, labels, metadata)


def _unavailable_report(status: str, notice: str, *, events: Sequence[Mapping[str, Any]] = (), config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    return {
        "version": 1,
        "provider": "BST",
        "status": status,
        "notice": notice,
        "model": str(config.get("model", "BST_CG_AP")),
        "dataset": str(config.get("dataset", "shuttleset_25")),
        "class_count": max(0, _integer(config.get("class_count"), 0)),
        "gate_policy": _gate_policy(),
        "total_hit_count": len(events),
        "predicted_hit_count": 0,
        "hits": [],
        "quality_notice": "BST 只识别击球类型，不判断动作好坏；技术动作建议仍来自可解释规则并需视频复核。",
    }


def run_bst_stroke_inference(
    video_path: Path | str,
    sidecar_path: Path | str,
    *,
    fps: float,
    pose_weights: Path | str,
    device: str = "auto",
    court_points: Sequence[float] = (),
    repo_root: Path | str = ".",
    predictor: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Run optional BST inference and return a path-free per-hit report.

    ``predictor`` is intentionally injectable for tests and for deployments
    that export BST to another runtime.  It receives the feature dictionary
    returned by :func:`build_bst_sequences` and must return one logit row per
    metadata item.
    """
    sidecar = Path(sidecar_path)
    video = Path(video_path)
    config = resolve_bst_configuration(repo_root)
    try:
        events, measured, _maximum_frame = _read_sidecar(sidecar)
    except (OSError, csv.Error, UnicodeError):
        return _unavailable_report("unavailable", "无法读取击球 sidecar，未运行 BST。", config=config)
    if not events:
        return _unavailable_report("insufficient_evidence", "未检测到可用于 BST 识别的击球事件。", events=events, config=config)
    if len(court_points) != 8:
        return _unavailable_report("insufficient_evidence", "BST 需要四点球场标定来生成球员场上位置。", events=events, config=config)
    if predictor is None and not config.get("configured"):
        missing = []
        if not config.get("repository_found"):
            missing.append("BST_REPO")
        if not config.get("weights_found"):
            missing.append("BST_WEIGHTS")
        if not missing:
            missing.append("有效的 BST_DATASET/模型配置")
        return _unavailable_report(
            "not_configured",
            "未配置 BST 研究代码或训练权重；已保留现有规则型击球候选。设置 " + "、".join(missing) + " 后重跑专业分析。",
            events=events,
            config=config,
        )
    if not video.is_file() or not sidecar.is_file():
        return _unavailable_report("unavailable", "原视频或击球 sidecar 不可用，未运行 BST。", events=events, config=config)
    # Read dimensions without making OpenCV a module-import requirement.
    try:
        import cv2

        capture = cv2.VideoCapture(str(video))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
    except Exception:
        return _unavailable_report("unavailable", "无法读取视频分辨率，未运行 BST。", events=events, config=config)
    if width <= 0 or height <= 0:
        return _unavailable_report("unavailable", "无法读取视频分辨率，未运行 BST。", events=events, config=config)
    try:
        from .coaching import _court_context, _pose_frames

        court_quad, court_mid_y, mapper, _court_height = _court_context(court_points)
    except Exception:
        return _unavailable_report("unavailable", "球场坐标转换不可用，未运行 BST。", events=events, config=config)
    if not court_quad or court_mid_y is None or mapper is None:
        return _unavailable_report("insufficient_evidence", "球场四点标定无效，未生成 BST 类型。", events=events, config=config)
    # Choose all events up to the configured safety bound.  The report makes
    # the processed/total count explicit when a very long match is sampled.
    max_events = int(config.get("max_events", _DEFAULT_MAX_EVENTS))
    selected = events if len(events) <= max_events else [events[min(len(events) - 1, int(index * len(events) / float(max_events)))] for index in range(max_events)]
    seq_len = int(config.get("seq_len", _DEFAULT_SEQ_LEN))
    radius = max(1, int(round(float(config.get("window_seconds", _DEFAULT_WINDOW_SECONDS)) * max(1.0, float(fps)))))
    requested_frames = []
    for event in selected:
        frame = _integer(event.get("frame"), 0)
        requested_frames.extend(range(max(0, frame - radius), frame + radius + 1))
    # A caller-provided predictor can supply its own pose/features and avoids
    # importing either Ultralytics or PyTorch in tests.
    pose_by_frame: Mapping[int, Mapping[str, Mapping[str, Any]]] = {}
    if predictor is None:
        pose_file = Path(pose_weights)
        if not pose_file.is_file():
            return _unavailable_report("unavailable", "YOLO 姿态权重不可用，未运行 BST。", events=events, config=config)
        try:
            from ultralytics import YOLO

            model = YOLO(str(pose_file))
            pose_by_frame = _pose_frames(
                video,
                model=model,
                requested_frames=sorted(set(requested_frames)),
                device=device,
                court_quad=court_quad,
                court_mid_y=court_mid_y,
                mapper=mapper,
            )
        except Exception:
            return _unavailable_report("unavailable", "姿态序列提取失败，未运行 BST。", events=events, config=config)
    else:
        # Custom predictors may expose a precomputed map through a conventional
        # attribute; otherwise a zero-pose feature set remains valid for shape
        # and decoder tests, but is marked as low evidence in the report.
        pose_by_frame = getattr(predictor, "pose_by_frame", {}) or {}
    features, metadata = build_bst_sequences(
        selected,
        pose_by_frame,
        measured,
        fps=fps,
        width=width,
        height=height,
        seq_len=seq_len,
        window_seconds=float(config.get("window_seconds", _DEFAULT_WINDOW_SECONDS)),
        max_events=max_events,
        pose_style=str(config.get("pose_style", _POSE_STYLE)),
    )
    if not metadata:
        return _unavailable_report("insufficient_evidence", "没有可构建的 BST 序列。", events=events, config=config)
    try:
        if predictor is not None:
            logits = predictor(features)
        else:
            network, torch_device, labels = _load_bst_model(config, device)
            import torch

            human_pose = torch.from_numpy(features["human_pose"]).to(torch_device)
            positions = torch.from_numpy(features["positions"]).to(torch_device)
            shuttles = torch.from_numpy(features["shuttles"]).to(torch_device)
            video_lengths = torch.from_numpy(features["video_lengths"]).to(torch_device)
            # Official inference flattens the final (joint,xy) dimensions.
            human_pose = human_pose.view(*human_pose.shape[:-2], -1)
            with torch.no_grad():
                if str(config.get("model", "")) == "BST_0":
                    logits = network(human_pose, shuttles, video_lengths)
                else:
                    logits = network(human_pose, shuttles, positions, video_lengths)
        labels = list(config.get("_labels", []))
        if predictor is not None:
            labels = list(config.get("_labels", [])) or [f"class_{index}" for index in range(len(logits[0]))]
        hits = _decode_logits(logits, labels, metadata)
    except Exception:
        return _unavailable_report(
            "unavailable",
            "BST 推理暂不可用，未生成模型类型；请检查上游模型依赖、架构和 checkpoint 是否匹配。",
            events=events,
            config=config,
        )
    predicted = len(hits)
    sampled_notice = ""
    if predicted < len(events):
        sampled_notice = f" 长视频按安全上限抽样了 {predicted}/{len(events)} 拍；可通过 BADMINTON_BST_MAX_EVENTS 调高上限。"
    return {
        "version": 1,
        "provider": "BST",
        "status": "ready" if hits else "insufficient_evidence",
        "notice": f"BST 输出击球类型候选；类型报告与 Motion Coach 使用独立证据门槛。{sampled_notice}",
        "model": str(config.get("model", "BST_CG_AP")),
        "dataset": str(config.get("dataset", "shuttleset_25")),
        "class_count": len(labels),
        "gate_policy": _gate_policy(),
        "total_hit_count": len(events),
        "predicted_hit_count": predicted,
        "hits": hits,
        "quality_notice": "击球类型识别不等于动作好坏评分；请结合 Motion Coach 的证据型建议和原视频复核。",
    }


def attach_rule_candidates(report: Mapping[str, Any], coaching: Mapping[str, Any]) -> dict[str, Any]:
    """Attach existing rule candidates to matching BST cards without guessing."""
    result = dict(report) if isinstance(report, Mapping) else {}
    raw_hits = result.get("hits", [])
    if not isinstance(raw_hits, (list, tuple)):
        return result
    by_frame: dict[int, str] = {}
    players = coaching.get("players", []) if isinstance(coaching, Mapping) else []
    if isinstance(players, (list, tuple)):
        for player in players:
            if not isinstance(player, Mapping):
                continue
            reviews = player.get("stroke_reviews", [])
            if not isinstance(reviews, (list, tuple)):
                continue
            for review in reviews:
                if not isinstance(review, Mapping):
                    continue
                evidence = review.get("evidence", {})
                stroke = review.get("stroke", {})
                classification = review.get("classification", {})
                if not isinstance(evidence, Mapping) or not isinstance(stroke, Mapping):
                    continue
                frame = _integer(evidence.get("frame"), -1)
                label = ""
                if isinstance(classification, Mapping):
                    label = str(classification.get("rule_label", "")).strip()
                if not label:
                    label = str(stroke.get("rule_label", stroke.get("label", ""))).strip()
                if frame >= 0 and label:
                    by_frame[frame] = label
    enriched = []
    for raw in raw_hits:
        if not isinstance(raw, Mapping):
            continue
        hit = dict(raw)
        frame = _integer(hit.get("frame"), -1)
        if frame in by_frame:
            hit["rule_candidate"] = by_frame[frame]
        else:
            hit["rule_candidate"] = ""
        enriched.append(hit)
    result["hits"] = enriched
    return result


__all__ = [
    "assess_bst_hit_gates",
    "attach_rule_candidates",
    "build_bst_sequences",
    "decode_bst_logits",
    "public_bst_configuration",
    "resolve_bst_configuration",
    "run_bst_stroke_inference",
]
