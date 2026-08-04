import argparse
import csv
import math
import os
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from ball_tracking import (
    BallPoint,
    DERIVED_MEASUREMENT_SOURCES,
    EDGE_MEASUREMENT_SOURCES,
    HitEvent,
    add_offscreen_bridges,
    coverage as ball_coverage,
    detect_hit_points,
    detect_top_edge_exit_hits,
    filter_ball_track_geometry,
    _point_near_top_edge,
    estimate_shot_average_speed,
    segment_rallies,
    smooth_ball_track,
    validate_hit_event_evidence,
)


DEFAULT_VIDEO_PATH = r"D:\yumaoqiu\tracknet_v3_result\866ba79f9b46ce0d9b8b1d55eb82832c_tracknetv3.mp4"
DEFAULT_OUTPUT_PATH = r"D:\yumaoqiu\end1.mp4"


FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
_FONT_CACHE = {}

ZH_TITLE = "\u4e0a\u534a\u573a\u7403\u5458\u7edf\u8ba1"
ZH_RALLY = "\u56de\u5408"
ZH_TOP = "\u4e0a\u534a\u573a"
ZH_BOTTOM = "\u4e0b\u534a\u573a"
ZH_ERR_COURT = "\u672a\u63d0\u4f9b --court_points\uff0c\u4e14\u5df2\u5173\u95ed\u624b\u52a8\u9009\u70b9\u3002\u8bf7\u5f00\u542f --select_court_points \u6216\u4f20\u5165 --court_points\u3002"


def parse_court_points(value):
    if not value:
        return None
    tokens = [v.strip() for v in value.split(",") if v.strip() != ""]
    if len(tokens) != 8:
        raise ValueError("court_points must be x1,y1,x2,y2,x3,y3,x4,y4")
    vals = [float(v) for v in tokens]
    return np.array(
        [[vals[0], vals[1]], [vals[2], vals[3]], [vals[4], vals[5]], [vals[6], vals[7]]],
        dtype=np.float32,
    )


def infer_ball_csv(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    if base.endswith("_tracknetv3"):
        stem = base.replace("_tracknetv3", "")
    else:
        stem = base
    candidates = [
        os.path.join(os.path.dirname(video_path), f"{stem}_ball.csv"),
        os.path.join(os.path.dirname(video_path), f"{base}_ball.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def load_ball_dict(csv_path):
    ball = {}
    if not csv_path or not os.path.exists(csv_path):
        return ball
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(float(row.get("Frame", "0")))
                vis = int(float(row.get("Visibility", "0")))
                x = int(float(row.get("X", "0")))
                y = int(float(row.get("Y", "0")))
                ball[idx] = (vis, x, y)
            except Exception:
                continue
    return ball


def _normalized_ball_observation(value, default_source="model"):
    """Return one visible observation with explicit source/confidence."""
    if value is None or len(value) < 3:
        return None
    try:
        visible = int(value[0]) > 0
        x = float(value[1])
        y = float(value[2])
    except (TypeError, ValueError, IndexError):
        return None
    if not visible or x <= 0 or y <= 0 or not np.isfinite([x, y]).all():
        return None
    source = str(value[3]) if len(value) >= 4 else str(default_source)
    try:
        confidence = float(value[4]) if len(value) >= 5 else 1.0
    except (TypeError, ValueError):
        confidence = 1.0
    return (
        1,
        int(round(x)),
        int(round(y)),
        source,
        float(np.clip(confidence, 0.0, 1.0)),
    )


def find_edge_reentry_frames(
    ball_dict,
    frame_count,
    active_quad,
    scene_cuts=None,
    edge_y=35.0,
    max_frames=6,
    max_y=150.0,
    min_outside_frames=2,
    max_horizontal_step=70.0,
    max_vertical_step=80.0,
):
    """Return a narrow mask for ordinary points just after top-edge re-entry.

    The expanded court polygon is intentionally conservative around the
    far-baseline corners.  A genuine shuttle can nevertheless re-enter along
    a short, monotone path which remains a few pixels outside that polygon
    (the supplied clip has f576--578 at ``x=1338``).  Widening the ROI globally
    admits scoreboard/player responses, so this helper requires all of the
    following instead:

    * an explicit ``model_edge``/``single_phase_edge`` anchor at ``y<=edge_y``;
    * at least ``min_outside_frames`` consecutive ordinary model points with
      small horizontal/vertical steps and increasing image-y;
    * a later ordinary point on the same monotone run which is inside
      ``active_quad``.

    Only the outside points are returned.  The in-quad confirmation is never
    masked and no coordinates are synthesized.  Scene cuts terminate a run.
    """
    n = max(0, int(frame_count))
    if n <= 0 or active_quad is None:
        return []
    try:
        quad = np.asarray(active_quad, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return []
    if len(quad) != 4 or not np.isfinite(quad).all():
        return []
    edge_limit = max(0.0, float(edge_y))
    frame_limit = max(1, int(max_frames))
    y_limit = max(edge_limit + 1.0, float(max_y))
    required = max(2, int(min_outside_frames))
    horizontal_limit = max(1.0, float(max_horizontal_step))
    vertical_limit = max(1.0, float(max_vertical_step))
    cuts = sorted({int(cut) for cut in (scene_cuts or ()) if 0 < int(cut) < n})

    def same_scene(left, right):
        return _same_scene(left, right, cuts)

    observations = {}
    for frame in range(n):
        value = _normalized_ball_observation(ball_dict.get(frame))
        if value is not None:
            observations[frame] = value

    selected = set()
    edge_sources = {"model_edge", "single_phase_edge"}
    for anchor in sorted(observations):
        anchor_value = observations[anchor]
        if (
            str(anchor_value[3]) not in edge_sources
            or float(anchor_value[2]) > edge_limit
        ):
            continue
        anchor_xy = np.asarray(anchor_value[1:3], dtype=np.float64)
        if not np.isfinite(anchor_xy).all():
            continue
        outside = []
        previous_xy = anchor_xy
        confirmed = False
        for frame in range(anchor + 1, min(n, anchor + frame_limit + 1)):
            if not same_scene(anchor, frame):
                break
            value = observations.get(frame)
            if value is None or str(value[3]) != "model":
                # Missing frames and another edge-labelled point indicate a
                # separate phase; do not leap over them into a body branch.
                break
            xy = np.asarray(value[1:3], dtype=np.float64)
            if not np.isfinite(xy).all():
                break
            step = xy - previous_xy
            if (
                float(xy[1]) <= float(previous_xy[1]) + 2.0
                or float(xy[1]) > y_limit
                or abs(float(step[0])) > horizontal_limit
                or float(step[1]) > vertical_limit
            ):
                break
            inside = bool(point_in_quad((float(xy[0]), float(xy[1])), quad))
            if not inside:
                outside.append(frame)
            elif len(outside) >= required:
                # The first in-quad ordinary point confirms the continuation;
                # do not include it in the mask.
                confirmed = True
                break
            else:
                # A point returning inside before the minimum outside run is
                # reached is ordinary geometry and needs no exception.
                break
            previous_xy = xy
        if confirmed and len(outside) >= required:
            selected.update(outside)
    return sorted(selected)


def _same_scene(left, right, scene_cuts):
    """Whether two frame ids belong to the same hard-cut segment."""
    lo, hi = sorted((int(left), int(right)))
    return not any(lo < int(cut) <= hi for cut in (scene_cuts or ()))


def longest_observation_gap(ball_dict, frame_count, scene_cuts=None):
    """Return the longest raw-observation hole inside one camera segment."""
    n = max(0, int(frame_count))
    cuts = sorted({int(cut) for cut in (scene_cuts or ()) if 0 < int(cut) < n})
    boundaries = [0] + cuts + [n]
    longest = 0
    for segment_start, segment_stop in zip(boundaries, boundaries[1:]):
        run = 0
        for frame in range(segment_start, segment_stop):
            if _normalized_ball_observation(ball_dict.get(frame)) is None:
                run += 1
                longest = max(longest, run)
            else:
                run = 0
    return int(longest)


def fuse_ball_observations(
    model_dict,
    classical_result,
    frame_count,
    scene_cuts=None,
    agreement_px=20.0,
    bridge_search_frames=6,
    bridge_classical_residual_px=28.0,
    bridge_model_residual_px=60.0,
):
    """Fuse real model/classical candidates without inventing coordinates.

    Near-agreeing observations keep the official TrackNet coordinate because
    it is usually more precisely localized.  A disagreement is not resolved
    by blindly preferring either detector: consecutive conflict frames are
    compared with the TrackNet trajectory immediately before and after the
    cluster.  Classical replaces TrackNet only when it closely follows that
    two-sided bridge and the model candidate is a clear excursion.  This
    preserves real high-speed contacts such as frame 240 in ``output.mp4``
    while correcting body/banner jumps such as frame 408.
    """
    n = max(0, int(frame_count))
    cuts = sorted({int(cut) for cut in (scene_cuts or ()) if 0 < int(cut) < n})
    fused = {}
    model_visible = {}
    for frame in range(n):
        normalized = _normalized_ball_observation(model_dict.get(frame), "model")
        if normalized is None:
            fused[frame] = (0, 0, 0)
        else:
            # A TrackNet-only CSV has no source column.  Preserve the explicit
            # low-confidence edge labels created by phase fusion; otherwise a
            # classical pass would silently turn them back into ordinary
            # model evidence and re-enable body/hit vetoes downstream.
            normalized = (
                normalized[0], normalized[1], normalized[2],
                normalized[3] if normalized[3] in {"model", "model_edge", "single_phase_edge"} else "model",
                normalized[4],
            )
            fused[frame] = normalized
            model_visible[frame] = normalized

    observations = getattr(classical_result, "observations", {}) or {}
    classical = {}
    classical_ids = {}
    for frame, observation in observations.items():
        frame = int(frame)
        if frame < 0 or frame >= n:
            continue
        try:
            value = (
                1,
                int(round(float(observation.x))),
                int(round(float(observation.y))),
                "classical",
                float(np.clip(float(observation.confidence), 0.0, 1.0)),
            )
            if not np.isfinite([value[1], value[2], value[4]]).all():
                continue
        except (TypeError, ValueError, AttributeError):
            continue
        classical[frame] = value
        classical_ids[frame] = int(getattr(observation, "track_id", -1))

    stats = {
        "model": len(model_visible),
        "classical": len(classical),
        "model_only": 0,
        "classical_fill": 0,
        "agree_keep_model": 0,
        "conflict_keep_model": 0,
        "conflict_use_classical": 0,
    }
    conflicts = []
    selected_classical_ids = {}
    agreement = max(0.0, float(agreement_px))
    for frame, classical_value in classical.items():
        model_value = model_visible.get(frame)
        if model_value is None:
            fused[frame] = classical_value
            selected_classical_ids[frame] = classical_ids[frame]
            stats["classical_fill"] += 1
            continue
        distance = euclidean(model_value[1:3], classical_value[1:3])
        if distance <= agreement:
            stats["agree_keep_model"] += 1
        else:
            conflicts.append(frame)

    # Split conflict frames into consecutive clusters and never bridge a cut.
    groups = []
    for frame in sorted(conflicts):
        if (
            not groups
            or frame != groups[-1][-1] + 1
            or not _same_scene(groups[-1][-1], frame, cuts)
        ):
            groups.append([frame])
        else:
            groups[-1].append(frame)

    search = max(1, int(bridge_search_frames))
    classical_limit = max(1.0, float(bridge_classical_residual_px))
    model_limit = max(1.0, float(bridge_model_residual_px))
    conflict_set = set(conflicts)
    for group in groups:
        first, last = group[0], group[-1]
        left = next(
            (
                frame for frame in range(first - 1, max(-1, first - search - 1), -1)
                if frame in model_visible
                and frame not in conflict_set
                and _same_scene(frame, first, cuts)
            ),
            None,
        )
        right = next(
            (
                frame for frame in range(last + 1, min(n, last + search + 1))
                if frame in model_visible
                and frame not in conflict_set
                and _same_scene(last, frame, cuts)
            ),
            None,
        )
        for frame in group:
            use_classical = False
            if left is not None and right is not None and right > left:
                alpha = float(frame - left) / float(right - left)
                bridge = np.asarray(model_visible[left][1:3], dtype=np.float64) * (1.0 - alpha)
                bridge += np.asarray(model_visible[right][1:3], dtype=np.float64) * alpha
                model_residual = float(np.linalg.norm(
                    np.asarray(model_visible[frame][1:3], dtype=np.float64) - bridge
                ))
                classical_residual = float(np.linalg.norm(
                    np.asarray(classical[frame][1:3], dtype=np.float64) - bridge
                ))
                use_classical = (
                    classical_residual <= classical_limit
                    and model_residual >= max(model_limit, 4.0 * classical_residual)
                )
            if use_classical:
                fused[frame] = classical[frame]
                selected_classical_ids[frame] = classical_ids[frame]
                stats["conflict_use_classical"] += 1
            else:
                stats["conflict_keep_model"] += 1

    stats["model_only"] = max(
        0,
        len(model_visible) - stats["agree_keep_model"]
        - stats["conflict_keep_model"] - stats["conflict_use_classical"],
    )
    stats["fused_visible"] = sum(
        _normalized_ball_observation(value) is not None for value in fused.values()
    )
    return fused, selected_classical_ids, stats


def fill_tracknet_gaps_with_classical(
    ball_dict,
    classical_result,
    frame_count,
    scene_cuts=None,
    support_window=8,
    min_track_points=3,
    max_gap_frames=12,
    bridge_residual_px=72.0,
    max_step_px=180.0,
    min_confidence=0.45,
):
    """Fill only proven post-veto TrackNet holes with classical points.

    The normal model/classical fusion runs before the final sparse phase/body
    veto.  A point which is rejected there therefore cannot be recovered by
    that pass, even when an independent classical track follows the same
    flight (the useful ``f317--318`` case in ``output.mp4``).  This helper is
    deliberately narrower than :func:`fuse_ball_observations`:

    * the destination frame must currently be missing (including a vetoed
      model tuple); existing TrackNet points are never replaced;
    * the classical observation must belong to a track with at least
      ``min_track_points`` observations;
    * the *whole* missing run must be bounded by ordinary ``model`` anchors in
      the same scene; edge/derived anchors are not accepted;
    * each candidate must stay close to the two-sided linear bridge and obey a
      per-frame step limit.

    Accepted points use the ``classical_gap`` provenance.  That source is
    renderable and retained by the Kalman stage, but is intentionally absent
    from ``REAL_MEASUREMENT_SOURCES``/``MEASUREMENT_SOURCES`` so it cannot
    inflate coverage, hit evidence, or rally segmentation.

    Returns ``(updated_dict, track_ids, stats)``.  The input mapping is copied.
    """
    n = max(0, int(frame_count))
    output = dict(ball_dict or {})
    observations = getattr(classical_result, "observations", {}) or {}
    support = max(1, int(support_window))
    minimum_track = max(1, int(min_track_points))
    maximum_gap = max(1, int(max_gap_frames))
    bridge_limit = max(1.0, float(bridge_residual_px))
    step_limit = max(1.0, float(max_step_px))
    confidence_limit = float(np.clip(float(min_confidence), 0.0, 1.0))
    cuts = sorted({int(cut) for cut in (scene_cuts or ()) if 0 < int(cut) < n})

    def same_scene(left, right):
        return _same_scene(left, right, cuts)

    # Normalize classical observations once and retain their detector track
    # ids.  A missing/invalid id cannot establish the required multi-point
    # support, so it is intentionally not eligible for gap filling.
    candidates = {}
    track_frames = {}
    for frame, value in observations.items():
        try:
            frame_i = int(frame)
        except (TypeError, ValueError):
            continue
        if frame_i < 0 or frame_i >= n:
            continue
        # ``ClassicalBallDetectionResult`` stores dataclass observations,
        # whereas unit/diagnostic callers may provide tuples.  Normalize both
        # representations before applying the shared gates.
        if hasattr(value, "x") and hasattr(value, "y"):
            try:
                value_for_normalize = (
                    1,
                    float(value.x),
                    float(value.y),
                    "classical",
                    float(getattr(value, "confidence", 1.0)),
                )
            except (TypeError, ValueError):
                value_for_normalize = None
        else:
            value_for_normalize = value
        normalized = _normalized_ball_observation(value_for_normalize, "classical")
        if normalized is None:
            continue
        try:
            track_id = int(getattr(value, "track_id", -1))
        except (TypeError, ValueError):
            track_id = -1
        if track_id < 0:
            continue
        try:
            confidence = float(normalized[4])
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < confidence_limit:
            continue
        candidates[frame_i] = (normalized, track_id)
        track_frames.setdefault(track_id, []).append(frame_i)
    supported_tracks = {
        track_id
        for track_id, frames in track_frames.items()
        if len(set(frames)) >= minimum_track
    }

    # Identify contiguous missing runs in the post-veto mapping.  A model
    # tuple with visibility=0 is missing even when it retains its old x/y for
    # diagnostics.  We use the complete run for anchor lookup, so a candidate
    # cannot sneak through a longer unsupported hole by borrowing an anchor
    # from its middle.
    missing_runs = []
    current = []
    for frame in range(n):
        if _normalized_ball_observation(output.get(frame)) is None:
            if not current or frame == current[-1] + 1:
                current.append(frame)
            else:
                missing_runs.append(current)
                current = [frame]
        elif current:
            missing_runs.append(current)
            current = []
    if current:
        missing_runs.append(current)

    stats = {
        "candidate_frames": len(candidates),
        "supported_tracks": len(supported_tracks),
        "missing_runs": len(missing_runs),
        "eligible_runs": 0,
        "filled": 0,
        "rejected_track_support": 0,
        "rejected_long_gap": 0,
        "rejected_anchor": 0,
        "rejected_bridge": 0,
        "rejected_step": 0,
    }
    selected_track_ids = {}

    for run in missing_runs:
        if not run:
            continue
        # A long absence is much more likely to contain an unseen apex or a
        # camera/background response.  Keep the render-only top-edge bridge
        # responsible for those cases instead of inventing interior points.
        if len(run) > maximum_gap:
            if any(frame in candidates for frame in run):
                stats["rejected_long_gap"] += sum(frame in candidates for frame in run)
            continue

        left = None
        for distance in range(1, support + 1):
            frame = run[0] - distance
            if frame < 0 or not same_scene(frame, run[0]):
                break
            value = _normalized_ball_observation(output.get(frame))
            if value is not None:
                if str(value[3]) == "model":
                    left = (frame, value)
                break
        right = None
        for distance in range(1, support + 1):
            frame = run[-1] + distance
            if frame >= n or not same_scene(run[-1], frame):
                break
            value = _normalized_ball_observation(output.get(frame))
            if value is not None:
                if str(value[3]) == "model":
                    right = (frame, value)
                break
        if left is None or right is None or right[0] <= left[0]:
            if any(frame in candidates for frame in run):
                stats["rejected_anchor"] += sum(frame in candidates for frame in run)
            continue
        stats["eligible_runs"] += 1
        left_frame, left_value = left
        right_frame, right_value = right
        left_xy = np.asarray(left_value[1:3], dtype=np.float64)
        right_xy = np.asarray(right_value[1:3], dtype=np.float64)
        span = float(right_frame - left_frame)
        if span <= 0 or not np.isfinite([left_xy, right_xy]).all():
            stats["rejected_anchor"] += sum(frame in candidates for frame in run)
            continue

        for frame in run:
            candidate_entry = candidates.get(frame)
            if candidate_entry is None:
                continue
            candidate, track_id = candidate_entry
            if track_id not in supported_tracks:
                stats["rejected_track_support"] += 1
                continue
            alpha = float(frame - left_frame) / span
            bridge = left_xy * (1.0 - alpha) + right_xy * alpha
            candidate_xy = np.asarray(candidate[1:3], dtype=np.float64)
            residual = float(np.linalg.norm(candidate_xy - bridge))
            if not np.isfinite(candidate_xy).all() or residual > bridge_limit:
                stats["rejected_bridge"] += 1
                continue

            # Bound both sides independently.  This catches a candidate which
            # happens to lie near the bridge's midpoint but would require a
            # teleport from one anchor or into the other.
            left_step = float(np.linalg.norm(candidate_xy - left_xy)) / max(
                1.0, float(frame - left_frame)
            )
            right_step = float(np.linalg.norm(right_xy - candidate_xy)) / max(
                1.0, float(right_frame - frame)
            )
            if max(left_step, right_step) > step_limit:
                stats["rejected_step"] += 1
                continue

            # Do not overwrite a point which appeared after an earlier
            # candidate was selected.  This is mostly defensive for sparse
            # mappings passed by unit tests; production mappings are dense.
            if _normalized_ball_observation(output.get(frame)) is not None:
                continue
            confidence = min(0.45, float(candidate[4]) * 0.65)
            output[frame] = (
                1,
                int(candidate[1]),
                int(candidate[2]),
                "classical_gap",
                confidence,
            )
            selected_track_ids[frame] = track_id
            stats["filled"] += 1

    return output, selected_track_ids, stats


def merge_tracknet_phase_observations(
    primary_dict,
    secondary_dict,
    frame_count,
    agreement_px=25.0,
    bridge_search_frames=6,
    bridge_residual_px=60.0,
    scene_cuts=None,
    edge_y_px=35.0,
    edge_lead_frames=3,
    single_edge_max_frames=2,
):
    """Merge two non-overlap temporal phases from the same TrackNet model.

    Moving the 8-frame window by half a sequence exposes a different temporal
    context without using InpaintNet.  Agreeing coordinates are retained as a
    real model observation.  Consecutive disagreements are resolved as one
    group using an agreement anchor before the group and another after it.
    This prevents a long conflict from repeatedly falling back to the primary
    phase.  When neither phase follows the bridge, the frame stays missing.
    The returned ``suspicious_frames`` are suitable for a sparse player-body
    pre-veto.
    """
    n = max(0, int(frame_count))
    primary = {
        frame: _normalized_ball_observation(primary_dict.get(frame), "model")
        for frame in range(n)
    }
    secondary = {
        frame: _normalized_ball_observation(secondary_dict.get(frame), "model")
        for frame in range(n)
    }
    agreement = max(0.0, float(agreement_px))
    agreement_frames = set()
    for frame in range(n):
        left, right = primary[frame], secondary[frame]
        if left is not None and right is not None:
            if euclidean(left[1:3], right[1:3]) <= agreement:
                agreement_frames.add(frame)

    search = max(1, int(bridge_search_frames))
    bridge_limit = max(1.0, float(bridge_residual_px))
    cuts = sorted({
        int(cut) for cut in (scene_cuts or ())
        if 0 < int(cut) < n
    })
    merged = {
        frame: (primary[frame] if frame in agreement_frames else (0, 0, 0))
        for frame in range(n)
    }
    low_support = set()
    stats = {
        "primary_visible": sum(value is not None for value in primary.values()),
        "secondary_visible": sum(value is not None for value in secondary.values()),
        "phase_agree": 0,
        "bridge_primary": 0,
        "bridge_secondary": 0,
        "single_phase": 0,
        "ambiguous_missing": 0,
        "dual_edge": 0,
        "single_edge": 0,
    }

    stats["phase_agree"] = len(agreement_frames)

    unresolved = [
        frame for frame in range(n)
        if frame not in agreement_frames
        and (primary[frame] is not None or secondary[frame] is not None)
    ]
    groups = []
    for frame in unresolved:
        crosses_cut = bool(groups) and any(
            groups[-1][-1] < cut <= frame for cut in cuts
        )
        if not groups or frame != groups[-1][-1] + 1 or crosses_cut:
            groups.append([frame])
        else:
            groups[-1].append(frame)

    for group in groups:
        first_frame, last_frame = group[0], group[-1]
        low_support.update(group)
        left = next(
            (
                candidate
                for candidate in range(
                    first_frame - 1,
                    max(-1, first_frame - search - 1),
                    -1,
                )
                if candidate in agreement_frames
                and _same_scene(candidate, first_frame, cuts)
            ),
            None,
        )
        right = next(
            (
                candidate
                for candidate in range(
                    last_frame + 1,
                    min(n, last_frame + search + 1),
                )
                if candidate in agreement_frames
                and _same_scene(last_frame, candidate, cuts)
            ),
            None,
        )
        has_bridge = left is not None and right is not None and right > left

        for frame in group:
            candidates = [
                (name, value)
                for name, value in (
                    ("primary", primary[frame]),
                    ("secondary", secondary[frame]),
                )
                if value is not None
            ]
            selected = None
            selected_phase = ""
            if has_bridge:
                alpha = float(frame - left) / float(right - left)
                bridge = np.asarray(primary[left][1:3], dtype=np.float64) * (1.0 - alpha)
                bridge += np.asarray(primary[right][1:3], dtype=np.float64) * alpha
                residuals = [
                    float(np.linalg.norm(
                        np.asarray(value[1:3], dtype=np.float64) - bridge
                    ))
                    for _, value in candidates
                ]
                best_index = int(np.argmin(residuals))
                decisive = (
                    len(residuals) == 1
                    or max(residuals) - min(residuals) >= 15.0
                )
                residual_limit = bridge_limit * (1.5 if len(candidates) == 1 else 1.0)
                if residuals[best_index] <= residual_limit and decisive:
                    selected_phase, selected = candidates[best_index]
            elif len(candidates) == 1:
                # A single phase can genuinely recover an edge/window miss.
                # Keep it as low-support evidence so the sparse body pass and
                # later geometry/trajectory stages can still reject it.
                selected_phase, selected = candidates[0]
            # With two conflicting candidates and no two-sided bridge, there
            # is no principled primary-phase preference.  Preserve missing.

            if selected is None:
                merged[frame] = (0, 0, 0)
                stats["ambiguous_missing"] += 1
            else:
                merged[frame] = (
                    1, selected[1], selected[2], "model", selected[4]
                )
                if len(candidates) == 1:
                    stats["single_phase"] += 1
                elif selected_phase == "primary":
                    stats["bridge_primary"] += 1
                else:
                    stats["bridge_secondary"] += 1

    # Preserve a very small amount of edge provenance.  A dual-phase point
    # which approaches y≈0 smoothly is stronger than a single phase response,
    # but it still has no trustworthy image evidence once the shuttle leaves
    # the frame.  Marking it ``model_edge`` lets the renderer keep the flight
    # while the branch/body veto and hit validator can intentionally ignore it.
    edge_limit = max(0.0, float(edge_y_px))
    edge_tail = max(1, int(edge_lead_frames))
    single_limit = max(1, int(single_edge_max_frames))
    dual_edge_frames = set()
    single_edge_frames = set()

    # Consecutive agreement runs.  Walk backwards from a top-edge point along
    # a monotone ascent (image-y decreasing) and retain at most ``edge_tail``
    # frames.  This recovers the f445--447 high-clear lead-in without opening
    # a broad exemption for arbitrary body jumps.
    agreement_sorted = sorted(agreement_frames)
    agreement_runs = []
    for frame in agreement_sorted:
        if (
            not agreement_runs
            or frame != agreement_runs[-1][-1] + 1
            or not _same_scene(agreement_runs[-1][-1], frame, cuts)
        ):
            agreement_runs.append([frame])
        else:
            agreement_runs[-1].append(frame)
    for run in agreement_runs:
        run_has_anchor = (
            len(run) >= 3
            or (run[0] > 0 and run[0] - 1 in agreement_frames)
            or (run[-1] + 1 < n and run[-1] + 1 in agreement_frames)
        )
        if not run_has_anchor:
            # Do not exempt an isolated two-frame agreement island.  Such
            # islands are frequently a shared body response after a long gap.
            continue
        for position, frame in enumerate(run):
            value = merged.get(frame)
            if value is None or _normalized_ball_observation(value) is None:
                continue
            y = float(value[2])
            if y > edge_limit:
                continue
            # A two-frame agreement island immediately after a detector gap
            # is often a shared body/background response (the end-of-video
            # f965--966 pattern).  Require either a preceding agreement anchor
            # or a short >=3-frame edge run before granting the exemption.
            if (
                position == 0
                and len(run) < 3
                and (position + 1 >= len(run) or float(merged[run[position + 1]][2]) <= edge_limit)
            ):
                continue
            tail = [frame]
            cursor = position - 1
            while len(tail) < edge_tail and cursor >= 0:
                previous_frame = run[cursor]
                previous = merged.get(previous_frame)
                if previous is None or _normalized_ball_observation(previous) is None:
                    break
                # A clear upward image-y trend is required; a pair of static
                # pixels at y=3 is not enough to bypass a body veto.
                if float(previous[2]) <= float(merged[tail[-1]][2]) + 1.0:
                    break
                tail.append(previous_frame)
                cursor -= 1
            dual_edge_frames.update(tail)

    # Single-phase edge points are useful for reacquisition, but only retain a
    # short entry/exit stub.  Look ahead through agreement points so a lone
    # secondary f574 or f797 point is recognized when the next frame descends
    # into the court.  Never use these points as hit evidence.
    for frame in range(n):
        if frame in agreement_frames:
            continue
        value = primary[frame] if primary[frame] is not None else secondary[frame]
        other = secondary[frame] if primary[frame] is not None else primary[frame]
        if value is None or other is not None or float(value[2]) > edge_limit:
            continue
        direction = 0
        for step in range(1, 4):
            candidate_frame = frame + step
            if candidate_frame >= n or not _same_scene(frame, candidate_frame, cuts):
                break
            candidate = merged.get(candidate_frame)
            if candidate is None or _normalized_ball_observation(candidate) is None:
                continue
            if float(candidate[2]) >= float(value[2]) + 3.0:
                direction = 1
                break
            # A flat top-edge response can last one or two frames before the
            # shuttle becomes visibly lower; keep looking instead of treating
            # the first equal-y point as a hard stop.
        if direction == 0:
            for step in range(1, 4):
                candidate_frame = frame - step
                if candidate_frame < 0 or not _same_scene(candidate_frame, frame, cuts):
                    break
                candidate = merged.get(candidate_frame)
                if candidate is None or _normalized_ball_observation(candidate) is None:
                    continue
                if float(candidate[2]) >= float(value[2]) + 3.0:
                    direction = -1
                    break
        if direction == 0:
            continue
        if direction > 0:
            single_edge_frames.add(frame)
            # Include at most one additional single-phase frame in the same
            # direction; agreement frames are classified independently.
            for offset in range(1, single_limit):
                candidate_frame = frame + offset
                if candidate_frame >= n or candidate_frame in agreement_frames:
                    break
                candidate = primary[candidate_frame] if primary[candidate_frame] is not None else secondary[candidate_frame]
                other_candidate = secondary[candidate_frame] if primary[candidate_frame] is not None else primary[candidate_frame]
                if candidate is None or other_candidate is not None or float(candidate[2]) > edge_limit:
                    break
                single_edge_frames.add(candidate_frame)
        else:
            single_edge_frames.add(frame)

    for frame in dual_edge_frames:
        value = merged.get(frame)
        if value is not None and _normalized_ball_observation(value) is not None:
            merged[frame] = (1, value[1], value[2], "model_edge", min(0.35, float(value[4])))
    for frame in single_edge_frames - dual_edge_frames:
        value = merged.get(frame)
        if value is not None and _normalized_ball_observation(value) is not None:
            merged[frame] = (1, value[1], value[2], "single_phase_edge", min(0.22, float(value[4])))
    stats["dual_edge"] = len(dual_edge_frames)
    stats["single_edge"] = len(single_edge_frames - dual_edge_frames)

    # Large second differences identify detector switches and hit-adjacent
    # uncertainty.  Do not delete them here: a smash is allowed to accelerate
    # sharply.  Instead ask the sparse player-body pass to veto only points
    # actually embedded in a person box.
    coordinates = {
        frame: np.asarray(value[1:3], dtype=np.float64)
        for frame, value in merged.items()
        if _normalized_ball_observation(value) is not None
    }
    temporal_suspicious = set()
    for frame, point in coordinates.items():
        if frame - 1 not in coordinates or frame + 1 not in coordinates:
            continue
        previous = coordinates[frame - 1]
        following = coordinates[frame + 1]
        acceleration = float(np.linalg.norm(following - 2.0 * point + previous))
        maximum_step = max(
            float(np.linalg.norm(point - previous)),
            float(np.linalg.norm(following - point)),
        )
        if acceleration >= 45.0 or maximum_step >= 180.0:
            temporal_suspicious.update(
                candidate for candidate in range(frame - 2, frame + 3)
                if candidate in coordinates
            )

    # Mark only the single frame at each segment boundary.  A previous version
    # marked three frames, which sent legitimate post-hit flight points through
    # the body gate even when the phases agreed and no jump was present.
    coordinate_frames = sorted(coordinates)
    segments = []
    for frame in coordinate_frames:
        crosses_cut = bool(segments) and any(
            segments[-1][-1] < cut <= frame for cut in cuts
        )
        if not segments or frame != segments[-1][-1] + 1 or crosses_cut:
            segments.append([frame])
        else:
            segments[-1].append(frame)
    for segment in segments:
        if segment[0] > 0:
            temporal_suspicious.add(segment[0])
        if segment[-1] < n - 1:
            temporal_suspicious.add(segment[-1])
    suspicious_frames = low_support | temporal_suspicious
    stats["suspicious"] = len(suspicious_frames)
    stats["merged_visible"] = len(coordinates)
    return merged, suspicious_frames, stats


def find_dual_phase_coherent_motion_frames(
    primary_dict,
    secondary_dict,
    merged_dict,
    candidate_frames,
    frame_count,
    scene_cuts=None,
    excluded_frames=None,
    agreement_px=25.0,
    support_frames=2,
    min_run_frames=2,
    max_run_frames=8,
    min_median_step_px=12.0,
    min_path_px=80.0,
    min_spread_px=35.0,
    min_support_displacement_px=10.0,
    max_phase_velocity_delta_px=18.0,
    edge_y=35.0,
):
    """Protect short, fast trajectories independently seen by both phases.

    A real shuttle can overlap a player's detector box at contact.  The
    relaxed pose/body veto must therefore not erase a locally complete flight
    which both TrackNet temporal phases observed independently.  Protection
    is intentionally granted per *candidate run*, never per isolated point:

    * every run point is visible and agrees across both phases;
    * two consecutive agreeing observations exist immediately before and
      after the run (scene cuts or holes reject it);
    * the local path has clear displacement/speed and the two phases estimate
      similar frame-to-frame motion; and
    * edge-labelled points and caller-supplied top-interstitial candidates are
      excluded from the protected run.

    This narrow gate preserves contact/reversal sequences such as f317--321
    and f853--858 while static body locks, single-phase jumps (f563/f863), and
    top-edge interior branches (f700--715) remain eligible for veto.
    """
    n = max(0, int(frame_count))
    excluded = {int(frame) for frame in (excluded_frames or ())}
    stats = {
        "candidate_frames": 0,
        "candidate_runs": 0,
        "protected_frames": 0,
        "protected_runs": 0,
        "rejected_length": 0,
        "rejected_support": 0,
        "rejected_motion": 0,
        "rejected_phase_velocity": 0,
    }
    if n <= 0:
        return [], stats

    agreement_limit = max(0.0, float(agreement_px))
    # Two real observations on each side are the minimum evidence promised by
    # this gate; accepting one would turn a single jump into "coherent" motion.
    support = max(2, int(support_frames))
    minimum_run = max(1, int(min_run_frames))
    maximum_run = max(minimum_run, int(max_run_frames))
    median_step_limit = max(0.0, float(min_median_step_px))
    path_limit = max(0.0, float(min_path_px))
    spread_limit = max(0.0, float(min_spread_px))
    support_displacement_limit = max(0.0, float(min_support_displacement_px))
    phase_velocity_limit = max(0.0, float(max_phase_velocity_delta_px))
    edge_limit = max(0.0, float(edge_y))
    cuts = sorted({int(cut) for cut in (scene_cuts or ()) if 0 < int(cut) < n})

    primary = [
        _normalized_ball_observation((primary_dict or {}).get(frame))
        for frame in range(n)
    ]
    secondary = [
        _normalized_ball_observation((secondary_dict or {}).get(frame))
        for frame in range(n)
    ]
    merged = [
        _normalized_ball_observation((merged_dict or {}).get(frame))
        for frame in range(n)
    ]
    dual_agree = np.zeros(n, dtype=bool)
    for frame in range(n):
        if primary[frame] is None or secondary[frame] is None:
            continue
        dual_agree[frame] = (
            euclidean(primary[frame][1:3], secondary[frame][1:3])
            <= agreement_limit
        )

    eligible = []
    for frame in sorted({int(value) for value in (candidate_frames or ())}):
        if frame < 0 or frame >= n or frame in excluded or not dual_agree[frame]:
            continue
        value = merged[frame]
        if (
            value is None
            or str(value[3]) != "model"
            or float(value[2]) <= edge_limit
        ):
            continue
        eligible.append(frame)
    stats["candidate_frames"] = len(eligible)

    runs = []
    for frame in eligible:
        if (
            not runs
            or frame != runs[-1][-1] + 1
            or not _same_scene(runs[-1][-1], frame, cuts)
        ):
            runs.append([frame])
        else:
            runs[-1].append(frame)
    stats["candidate_runs"] = len(runs)

    protected = []
    for run in runs:
        if len(run) < minimum_run or len(run) > maximum_run:
            stats["rejected_length"] += 1
            continue
        left = list(range(run[0] - support, run[0]))
        right = list(range(run[-1] + 1, run[-1] + support + 1))
        local_frames = left + list(run) + right
        if (
            not local_frames
            or min(local_frames) < 0
            or max(local_frames) >= n
            or any(frame in excluded for frame in local_frames)
            or any(not _same_scene(run[0], frame, cuts) for frame in local_frames)
            or any(not dual_agree[frame] or merged[frame] is None for frame in local_frames)
        ):
            stats["rejected_support"] += 1
            continue

        points = np.asarray(
            [[merged[frame][1], merged[frame][2]] for frame in local_frames],
            dtype=np.float64,
        )
        primary_points = np.asarray(
            [[primary[frame][1], primary[frame][2]] for frame in local_frames],
            dtype=np.float64,
        )
        secondary_points = np.asarray(
            [[secondary[frame][1], secondary[frame][2]] for frame in local_frames],
            dtype=np.float64,
        )
        if not (
            np.isfinite(points).all()
            and np.isfinite(primary_points).all()
            and np.isfinite(secondary_points).all()
        ):
            stats["rejected_support"] += 1
            continue

        velocities = np.diff(points, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        center = np.median(points, axis=0)
        spread = float(np.max(np.linalg.norm(points - center, axis=1)))
        left_support_displacement = float(np.linalg.norm(points[support - 1] - points[0]))
        right_support_displacement = float(np.linalg.norm(points[-1] - points[-support]))
        if (
            len(speeds) == 0
            or float(np.median(speeds)) < median_step_limit
            or float(np.sum(speeds)) < path_limit
            or spread < spread_limit
            or left_support_displacement < support_displacement_limit
            or right_support_displacement < support_displacement_limit
        ):
            stats["rejected_motion"] += 1
            continue

        primary_velocity = np.diff(primary_points, axis=0)
        secondary_velocity = np.diff(secondary_points, axis=0)
        phase_velocity_delta = float(np.median(np.linalg.norm(
            primary_velocity - secondary_velocity,
            axis=1,
        )))
        if phase_velocity_delta > phase_velocity_limit:
            stats["rejected_phase_velocity"] += 1
            continue

        protected.extend(run)
        stats["protected_runs"] += 1

    protected = sorted(set(protected))
    stats["protected_frames"] = len(protected)
    return protected, stats


def veto_model_observations_in_player_boxes(
    ball_dict,
    pose_evidence,
    frames,
    box_margin_px=8.0,
    contact_protect_px=38.0,
    edge_protect_y=0.0,
    protected_frames=None,
):
    """Remove suspicious model points on a player while protecting racket contact.

    A small positive box margin catches TrackNet locks on a limb/head edge.
    Pose-estimated racket extensions are protected first so a real shuttle at
    the instant of contact is not erased by that broader body gate.
    """
    output = dict(ball_dict)
    protected = {int(value) for value in (protected_frames or ())}
    vetoed = []
    for frame in sorted({int(value) for value in (frames or ())}):
        if frame in protected:
            continue
        observation = _normalized_ball_observation(output.get(frame))
        if observation is None or str(observation[3]) != "model":
            continue
        point = (float(observation[1]), float(observation[2]))
        # A shuttle that is genuinely crossing the top image boundary is
        # often only a few pixels from the scoreboard.  Pose boxes can
        # overlap that edge after perspective projection; preserve the edge
        # observation and let the temporal/geometry stages arbitrate it.
        if float(edge_protect_y) > 0 and point[1] <= float(edge_protect_y):
            continue
        players = (pose_evidence or {}).get(frame, {}).get("players", [])
        inside = False
        for player in players:
            box = player.get("box")
            if box is None:
                continue
            protect_points = list(player.get("wrists", []))
            if player.get("contact") is not None:
                protect_points.append(player["contact"])
            if protect_points and min(
                euclidean(point, candidate) for candidate in protect_points
            ) <= max(0.0, float(contact_protect_px)):
                continue
            x1, y1, x2, y2 = map(float, box)
            margin = max(0.0, float(box_margin_px))
            if (
                x1 - margin <= point[0] <= x2 + margin
                and y1 - margin <= point[1] <= y2 + margin
            ):
                inside = True
                break
        if inside:
            output[frame] = (
                0, observation[1], observation[2], "player_veto", 0.0
            )
            vetoed.append(frame)
    return output, vetoed


def veto_isolated_phase_observations(
    ball_dict,
    primary_dict,
    secondary_dict,
    pose_evidence,
    candidate_frames,
    scene_cuts=None,
    support_window=5,
    max_bridge_residual_px=110.0,
    max_one_sided_step_px=95.0,
    top_edge_y=35.0,
    contact_protect_px=38.0,
):
    """Veto unsupported single-phase observations before Kalman smoothing.

    A half-window TrackNet pass can emit a lone body response at the first
    frame after a gap (e.g. ``(1031,427)`` or ``(656,753)``).  Treating that
    point as a fresh trajectory anchor lets Kalman forecasts paint a long
    false segment.  This gate only considers frames visible in exactly one
    phase and asks for a two-sided local bridge (or two coherent one-sided
    neighbors).  Top-edge lead-ins and pose-contact points are protected.

    The returned coordinates remain in the sidecar with a dedicated invisible
    provenance marker, so the caller can make the veto a hard smoother reset
    without confusing it with a detector miss.
    """
    output = dict(ball_dict or {})
    cuts = sorted({int(cut) for cut in (scene_cuts or ())})
    window = max(1, int(support_window))
    bridge_limit = max(1.0, float(max_bridge_residual_px))
    one_sided_limit = max(1.0, float(max_one_sided_step_px))
    edge_limit = max(0.0, float(top_edge_y))
    contact_limit = max(0.0, float(contact_protect_px))

    def same_scene(left, right):
        return _same_scene(left, right, cuts)

    def observation(mapping, frame):
        return _normalized_ball_observation((mapping or {}).get(int(frame)))

    def pose_contact(frame, point):
        evidence = (pose_evidence or {}).get(int(frame), {})
        players = list(evidence.get("players", []))
        players.extend(evidence.get("veto_players", []))
        for player in players:
            contacts = list(player.get("wrists", []))
            if player.get("contact") is not None:
                contacts.append(player["contact"])
            if any(
                candidate is not None
                and len(candidate) >= 2
                and euclidean(point, candidate) <= contact_limit
                for candidate in contacts
            ):
                return True
        return False

    def top_edge_supported(frame, point):
        if float(point[1]) > edge_limit:
            return False
        # Look through a couple of missing frames for a descending re-entry or
        # a preceding monotone ascent.  Edge aliases are already protected,
        # but this also covers a raw single-phase point before classification.
        for direction in (-1, 1):
            for distance in range(1, 4):
                other_frame = int(frame) + direction * distance
                if other_frame < 0 or not same_scene(frame, other_frame):
                    break
                other = _normalized_ball_observation(output.get(other_frame))
                if other is None:
                    continue
                if float(other[2]) >= float(point[1]) + 3.0:
                    return True
                break
        return False

    # Use the current post-veto output as support.  A point removed by an
    # earlier body/branch gate must not resurrect an isolated candidate.
    coordinates = {}
    for frame, value in output.items():
        normalized = _normalized_ball_observation(value)
        if normalized is not None:
            coordinates[int(frame)] = np.asarray(normalized[1:3], dtype=np.float64)
    # ``output`` is normally a dense ``range(frame_count)`` mapping produced
    # by phase fusion, but this helper is also used directly by diagnostics
    # and tests with sparse dictionaries.  ``len(output)`` is the number of
    # entries, not the largest frame id; using it as an upper bound silently
    # discards valid right-side support when keys are e.g. ``{2, 3, 4, 5}``.
    # Bound searches by the largest *visible* coordinate instead.
    max_coordinate_frame = max(coordinates, default=-1)

    vetoed = []
    stats = {"candidates": 0, "vetoed": 0, "bridge_supported": 0,
             "one_sided_supported": 0, "edge_protected": 0,
             "contact_protected": 0}
    for frame in sorted({int(value) for value in (candidate_frames or ())}):
        primary = observation(primary_dict, frame)
        secondary = observation(secondary_dict, frame)
        # Exactly one phase must support the point.  Agreement points are
        # intentionally left to the normal phase/body gates.
        if (primary is None) == (secondary is None):
            continue
        current = _normalized_ball_observation(output.get(frame))
        if current is None:
            continue
        stats["candidates"] += 1
        point = np.asarray(current[1:3], dtype=np.float64)
        if str(current[3]) in {"model_edge", "single_phase_edge"} or top_edge_supported(frame, point):
            stats["edge_protected"] += 1
            continue
        contact_near = pose_contact(frame, point)

        left = None
        right = None
        for distance in range(1, window + 1):
            candidate = frame - distance
            if candidate >= 0 and same_scene(frame, candidate):
                if candidate in coordinates:
                    left = (candidate, coordinates[candidate])
                    break
        for distance in range(1, window + 1):
            candidate = frame + distance
            if (
                candidate > max_coordinate_frame
                or not same_scene(frame, candidate)
            ):
                break
            if candidate in coordinates:
                right = (candidate, coordinates[candidate])
                break

        supported = False
        support_kind = ""
        contradictory = False
        if left is not None and right is not None and right[0] > left[0]:
            alpha = float(frame - left[0]) / float(right[0] - left[0])
            bridge = left[1] * (1.0 - alpha) + right[1] * alpha
            bridge_residual = float(np.linalg.norm(point - bridge))
            if bridge_residual <= bridge_limit:
                supported = True
                support_kind = "bridge"
            elif bridge_residual > 1.5 * bridge_limit:
                # A nearby pose wrist/body point must not override a clearly
                # contradictory two-sided trajectory (for example f448).
                contradictory = True
        else:
            side = left or right
            if side is not None:
                side_frame, side_point = side
                distance = abs(int(frame) - int(side_frame))
                # The candidate itself must connect to the nearest support
                # anchor.  The previous one-sided check only compared the
                # *support chain* behind that anchor, allowing a 400 px body
                # jump to masquerade as a coherent re-acquisition (f563).
                candidate_step = float(np.linalg.norm(point - side_point))
                contradictory = candidate_step > max(one_sided_limit, 220.0)
                if candidate_step <= max(one_sided_limit, 220.0):
                    # One-sided support needs at least two additional
                    # coherent points, not merely a nearby isolated anchor.
                    chain = []
                    direction = -1 if left is not None else 1
                    for offset in range(1, window + 1):
                        candidate = frame + direction * (distance + offset)
                        if (
                            candidate < 0
                            or candidate > max_coordinate_frame
                            or not same_scene(frame, candidate)
                        ):
                            break
                        if candidate in coordinates:
                            chain.append((candidate, coordinates[candidate]))
                        if len(chain) >= 2:
                            break
                    if len(chain) >= 2:
                        ordered = [side] + chain
                        steps = [
                            float(np.linalg.norm(ordered[index + 1][1] - ordered[index][1]))
                            for index in range(len(ordered) - 1)
                        ]
                        if all(step <= one_sided_limit for step in steps):
                            supported = True
                            support_kind = "one_sided"
        if supported:
            stats[f"{support_kind}_supported"] += 1
            continue
        if contact_near and not contradictory:
            stats["contact_protected"] += 1
            continue
        output[frame] = (0, current[1], current[2], "phase_isolated_veto", 0.0)
        vetoed.append(frame)

    stats["vetoed"] = len(vetoed)
    return output, sorted(set(vetoed)), stats


def find_top_exit_branch_frames(
    ball_dict,
    frame_count,
    scene_cuts=None,
    top_y=35.0,
    body_y=300.0,
    max_branch_frames=8,
):
    """Find short model branches that fall from the image top into a body.

    The official checkpoint occasionally follows a player's shirt/legs for a
    few frames immediately after a real shuttle reaches the top edge.  The
    two temporal phases can agree on that same false response, so phase
    disagreement alone cannot flag it.  This helper only selects frames for
    a subsequent pose-box veto; it never removes a point by itself.
    """
    n = max(0, int(frame_count))
    top_limit = float(top_y)
    body_limit = float(body_y)
    branch_limit = max(1, int(max_branch_frames))
    cuts = sorted({int(cut) for cut in (scene_cuts or ()) if 0 < int(cut) < n})
    coordinates = {}
    coordinate_sources = {}
    for frame in range(n):
        observation = _normalized_ball_observation(ball_dict.get(frame))
        # A dual-phase edge anchor can be the last trustworthy point before
        # the detector switches to a player/body branch.  Include it as the
        # branch origin, but keep single-phase edge points excluded.
        if observation is None or str(observation[3]) not in {"model", "model_edge"}:
            continue
        coordinates[frame] = (float(observation[1]), float(observation[2]))
        coordinate_sources[frame] = str(observation[3])

    def same_scene(left, right):
        return _same_scene(left, right, cuts)

    selected = set()
    for anchor in sorted(coordinates):
        if coordinates[anchor][1] > top_limit:
            continue
        edge_anchor = coordinate_sources.get(anchor) == "model_edge"
        edge_branch_confirmed = not edge_anchor
        # Look only a short distance after the top-edge anchor.  Missing
        # frames are tolerated because the shuttle may be fully out of frame;
        # a later body response is still a useful pose-veto target.
        for frame in range(anchor + 1, min(n, anchor + branch_limit + 1)):
            if not same_scene(anchor, frame):
                break
            point = coordinates.get(frame)
            if point is None:
                continue
            if edge_anchor and not edge_branch_confirmed:
                # A genuine edge re-entry normally continues smoothly from
                # y≈0 (e.g. f478→f486).  The false branch in this clip instead
                # jumps directly from the edge onto a player's body.  Require
                # that first ordinary model point to fall into the body zone
                # with a large vertical jump before applying a pose veto.
                if coordinate_sources.get(frame) != "model":
                    continue
                if (
                    point[1] < body_limit
                    or point[1] - coordinates[anchor][1] < 180.0
                ):
                    break
                edge_branch_confirmed = True
            if point[1] >= body_limit:
                selected.add(frame)
            # A second top-edge anchor closes the branch; do not select later
            # unrelated flight points in the same window.
            if point[1] <= top_limit and frame > anchor + 1:
                break
    return sorted(selected)


def find_top_edge_interstitial_branch_frames(
    ball_dict,
    frame_count,
    scene_cuts=None,
    edge_top_y=180.0,
    edge_core_y=35.0,
    min_anchor_frames=3,
    min_anchor_span_y=24.0,
    max_interstitial_gap=40,
    min_branch_residual_px=110.0,
):
    """Find model points stranded between two coherent top-edge runs.

    A high clear can leave the image at the top and re-enter many frames
    later.  During that unseen interval TrackNet sometimes follows a player
    or the scoreboard (for example f700--706/f715 in ``output.mp4``).  The
    temporal phases can agree on that branch, so ordinary phase disagreement
    and player-box gates are insufficient.  This detector looks for a much
    narrower signature:

    * a chronological run of at least three points monotonically approaching
      the top edge (the broad ``edge_top_y`` envelope defaults to 180 px);
    * a same-scene run of at least three points monotonically leaving it;
    * a bounded gap between those runs; and
    * both run endpoints reaching the strict ``edge_core_y`` zone (35 px by
      default; the production CLI keeps its 30 px core); and
    * a visible ordinary-model point in the gap which is more than
      ``min_branch_residual_px`` from the straight line joining the two edge
      anchors.

    The returned mapping is ``frame -> (left_anchor, right_anchor, residual)``
    where each anchor is ``(frame, x, y)``.  It is only a candidate set; call
    :func:`veto_top_edge_interstitial_branches` after pose evidence has been
    collected so wrist/contact protection can be applied.
    """
    n = max(0, int(frame_count))
    if n <= 0:
        return [], {}, {
            "anchor_runs": 0,
            "bridges": 0,
            "candidate_frames": 0,
        }
    top_limit = max(0.0, float(edge_top_y))
    core_limit = max(0.0, min(top_limit, float(edge_core_y)))
    minimum_frames = max(2, int(min_anchor_frames))
    minimum_span = max(0.0, float(min_anchor_span_y))
    maximum_gap = max(1, int(max_interstitial_gap))
    residual_limit = max(1.0, float(min_branch_residual_px))
    cuts = sorted({int(cut) for cut in (scene_cuts or ()) if 0 < int(cut) < n})

    def same_scene(left, right):
        return _same_scene(left, right, cuts)

    # Only detector observations can define an edge run.  Keeping the edge
    # provenance here also prevents a later classical/derived fill from
    # becoming an anchor for this veto.
    coordinates = {}
    for frame in range(n):
        observation = _normalized_ball_observation(ball_dict.get(frame))
        if observation is None:
            continue
        source = str(observation[3])
        if source not in {"model", "model_edge", "single_phase_edge"}:
            continue
        if float(observation[2]) > top_limit:
            continue
        coordinates[frame] = (
            float(observation[1]),
            float(observation[2]),
            source,
        )

    # Split into contiguous same-scene runs first.  A run can contain a small
    # amount of pixel jitter, but must have a clear net vertical direction.
    runs = []
    for frame in sorted(coordinates):
        if (
            not runs
            or frame != runs[-1][-1] + 1
            or not same_scene(runs[-1][-1], frame)
        ):
            runs.append([frame])
        else:
            runs[-1].append(frame)

    def monotone(run, direction):
        if len(run) < minimum_frames:
            return False
        ys = np.asarray([coordinates[frame][1] for frame in run], dtype=np.float64)
        if not np.isfinite(ys).all():
            return False
        span = float(ys[0] - ys[-1]) if direction == "up" else float(ys[-1] - ys[0])
        if span < minimum_span:
            return False
        diffs = np.diff(ys)
        # A one-pixel plateau is common at the heatmap boundary.  Permit it,
        # but reject a genuine reversal inside an anchor run.
        tolerance = 3.0
        if direction == "up":
            return bool(np.all(diffs <= tolerance))
        return bool(np.all(diffs >= -tolerance))

    ascending_runs = [run for run in runs if monotone(run, "up")]
    descending_runs = [run for run in runs if monotone(run, "down")]
    candidate_bridges = {}
    bridge_keys = set()
    for left_run in ascending_runs:
        left_frame = left_run[-1]
        left_xy = np.asarray(coordinates[left_frame][:2], dtype=np.float64)
        # The last point must genuinely touch the core top zone.  This avoids
        # treating an ordinary far-court arc as an unseen top-edge flight.
        if float(left_xy[1]) > core_limit:
            continue
        for right_run in descending_runs:
            right_frame = right_run[0]
            if right_frame <= left_frame:
                continue
            gap = right_frame - left_frame - 1
            if gap < 1 or gap > maximum_gap or not same_scene(left_frame, right_frame):
                continue
            right_xy = np.asarray(coordinates[right_frame][:2], dtype=np.float64)
            if float(right_xy[1]) > core_limit:
                continue
            # A pair of runs must have a real top-edge response on at least
            # one side.  ``edge_core_y`` is intentionally stricter than the
            # broad 110 px run envelope and protects normal court arcs.
            left_core = any(float(coordinates[f][1]) <= core_limit for f in left_run)
            right_core = any(float(coordinates[f][1]) <= core_limit for f in right_run)
            if not (left_core and right_core):
                continue
            span = float(right_frame - left_frame)
            if span <= 0:
                continue
            key = (left_frame, right_frame)
            if key in bridge_keys:
                continue
            bridge_keys.add(key)
            for frame in range(left_frame + 1, right_frame):
                value = _normalized_ball_observation(ball_dict.get(frame))
                if value is None or str(value[3]) != "model":
                    continue
                alpha = float(frame - left_frame) / span
                bridge = left_xy * (1.0 - alpha) + right_xy * alpha
                point = np.asarray(value[1:3], dtype=np.float64)
                residual = float(np.linalg.norm(point - bridge))
                if np.isfinite(point).all() and residual > residual_limit:
                    # If multiple compatible anchor pairs cover one frame,
                    # retain the strongest (largest residual) explanation.
                    old = candidate_bridges.get(frame)
                    if old is None or residual > float(old[2]):
                        candidate_bridges[frame] = (
                            (left_frame, float(left_xy[0]), float(left_xy[1])),
                            (right_frame, float(right_xy[0]), float(right_xy[1])),
                            residual,
                        )

    stats = {
        "anchor_runs": len(ascending_runs) + len(descending_runs),
        "ascending_runs": len(ascending_runs),
        "descending_runs": len(descending_runs),
        "bridges": len(bridge_keys),
        "candidate_frames": len(candidate_bridges),
    }
    return sorted(candidate_bridges), candidate_bridges, stats


def veto_top_edge_interstitial_branches(
    ball_dict,
    candidate_bridges,
    pose_evidence=None,
    protected_frames=None,
    edge_top_y=180.0,
    edge_core_y=35.0,
    contact_protect_px=38.0,
):
    """Remove anomalous interior points between coherent top-edge anchors.

    ``candidate_bridges`` is produced by
    :func:`find_top_edge_interstitial_branch_frames`.  Only ordinary model
    points are eligible; edge-labelled points, pose-contact points, and
    caller-protected lead-in frames remain untouched.  ``edge_top_y`` is only
    the broad discovery envelope; ordinary points are protected by the
    stricter ``edge_core_y`` threshold (or explicit contact/protected-frame
    evidence).  Vetoed coordinates are
    retained with ``source='phase_top_branch_veto'`` so the smoother can reset
    causally without treating the response as a detector miss from a separate
    stage.
    """
    output = dict(ball_dict or {})
    protected = {int(frame) for frame in (protected_frames or ())}
    # ``edge_top_y`` is deliberately broad: it lets the candidate search
    # include the first ordinary point after a top-edge excursion (which can
    # sit well below the heatmap boundary).  It must not, however, make every
    # ordinary point in that broad band immune to the veto.  Only points in
    # the strict core zone are genuine edge observations; an anomalous point
    # such as f715 (y≈176) must remain removable.
    top_limit = max(0.0, float(edge_top_y))
    core_limit = max(0.0, min(top_limit, float(edge_core_y)))
    contact_limit = max(0.0, float(contact_protect_px))
    candidate_bridges = candidate_bridges or {}
    vetoed = []
    stats = {
        "candidates": len(candidate_bridges),
        "vetoed": 0,
        "edge_protected": 0,
        "contact_protected": 0,
        "non_model_skipped": 0,
    }

    def near_contact(frame, point):
        evidence = (pose_evidence or {}).get(int(frame), {})
        players = list(evidence.get("players", []))
        players.extend(evidence.get("veto_players", []))
        for player in players:
            contacts = list(player.get("wrists", []) or [])
            if player.get("contact") is not None:
                contacts.append(player["contact"])
            if any(
                candidate is not None
                and len(candidate) >= 2
                and euclidean(point, candidate) <= contact_limit
                for candidate in contacts
            ):
                return True
        return False

    for frame in sorted(int(value) for value in candidate_bridges):
        observation = _normalized_ball_observation(output.get(frame))
        if observation is None:
            continue
        source = str(observation[3])
        point = (float(observation[1]), float(observation[2]))
        if frame in protected or source in {"model_edge", "single_phase_edge"} \
                or float(point[1]) <= core_limit:
            stats["edge_protected"] += 1
            continue
        if source != "model":
            stats["non_model_skipped"] += 1
            continue
        if near_contact(frame, point):
            stats["contact_protected"] += 1
            continue
        output[frame] = (
            0,
            observation[1],
            observation[2],
            "phase_top_branch_veto",
            0.0,
        )
        vetoed.append(frame)

    stats["vetoed"] = len(vetoed)
    return output, vetoed, stats


def find_top_entry_lead_frames(
    ball_dict,
    frame_count,
    scene_cuts=None,
    top_y=35.0,
    max_lead_frames=8,
    min_lead_points=3,
    min_vertical_step=8.0,
    max_step_px=220.0,
):
    """Find the short lead-in immediately before a dual-phase top exit.

    ``merge_tracknet_phase_observations`` labels a point ``model_edge`` only
    when both temporal phases support the top-edge response.  Pose boxes can
    nevertheless overlap the preceding two or three points (the far player
    and racket occupy a large fraction of that region).  This helper walks
    *backwards* from such an anchor and returns only a compact, contiguous,
    monotone image-y ascent.  It is deliberately narrower than
    :func:`find_top_exit_branch_frames`: single-phase edge points, ordinary
    model points, gaps, cuts, and reversals are never promoted.

    The returned frame ids are a protection set for body/background vetoes;
    callers must retain their ``model_edge`` provenance.  Since edge sources
    are excluded from ``MEASUREMENT_SOURCES`` in ``ball_tracking``, these
    frames cannot become hit evidence or create a rally on their own.
    """
    n = max(0, int(frame_count))
    if n <= 0:
        return []
    top_limit = max(0.0, float(top_y))
    lead_limit = max(1, int(max_lead_frames))
    required = max(2, min(lead_limit, int(min_lead_points)))
    vertical_step = max(0.0, float(min_vertical_step))
    speed_limit = max(vertical_step, float(max_step_px))
    cuts = sorted({int(cut) for cut in (scene_cuts or ()) if 0 < int(cut) < n})

    coordinates = {}
    for frame in range(n):
        observation = _normalized_ball_observation(ball_dict.get(frame))
        # A single-phase edge point is useful for reacquisition but is not
        # strong enough to protect a body-overlap branch.
        if observation is None or str(observation[3]) != "model_edge":
            continue
        coordinates[frame] = np.asarray(observation[1:3], dtype=np.float64)

    def same_scene(left, right):
        return _same_scene(left, right, cuts)

    selected = set()
    for anchor in sorted(coordinates):
        anchor_xy = coordinates[anchor]
        if float(anchor_xy[1]) > top_limit:
            continue
        lead = [anchor]
        current = anchor
        previous_velocity = None
        while len(lead) < lead_limit:
            previous_frame = current - 1
            if previous_frame not in coordinates or not same_scene(previous_frame, current):
                break
            previous_xy = coordinates[previous_frame]
            # Image y grows downwards, so an ascent has a negative dy.
            velocity = coordinates[current] - previous_xy
            step = float(np.linalg.norm(velocity))
            if velocity[1] > -vertical_step or step > speed_limit:
                break
            # Reject a sharp horizontal reversal; a gentle change at the
            # apex is fine, but a detector switch should not be protected.
            if previous_velocity is not None:
                prev_norm = float(np.linalg.norm(previous_velocity))
                if prev_norm > 1e-6 and float(np.dot(velocity, previous_velocity)) < 0.0:
                    break
            lead.append(previous_frame)
            previous_velocity = velocity
            current = previous_frame

        if len(lead) < required:
            continue
        # Require at least one lead point to be inside the image (rather than
        # a flat run of y≈0 banner responses) and a genuine vertical span.
        lead_y = [float(coordinates[frame][1]) for frame in lead]
        if max(lead_y) <= top_limit + vertical_step:
            continue
        if max(lead_y) - min(lead_y) < vertical_step:
            continue
        selected.update(lead)
    return sorted(selected)


def veto_top_exit_branch_points(
    ball_dict,
    pose_evidence,
    frames,
    contact_protect_px=38.0,
    protected_frames=None,
):
    """Veto top-exit branch points unless pose evidence supports contact.

    A branch can sit just outside a detector box (especially for the tiny
    far-side player), so a box-only gate may miss it.  Treat the branch as a
    high-precision negative cue, but preserve a point close to a detected
    wrist/contact location where a real smash or return is plausible.
    """
    output = dict(ball_dict)
    protected = {int(value) for value in (protected_frames or ())}
    vetoed = []
    tolerance = max(0.0, float(contact_protect_px))
    for frame in sorted({int(value) for value in (frames or ())}):
        if frame in protected:
            continue
        observation = _normalized_ball_observation(output.get(frame))
        if observation is None or str(observation[3]) != "model":
            continue
        point = (float(observation[1]), float(observation[2]))
        evidence = (pose_evidence or {}).get(frame, {})
        players = list(evidence.get("players", [])) + list(evidence.get("veto_players", []))
        contacts = []
        for player in players:
            contacts.extend(player.get("wrists", []) or [])
            if player.get("contact") is not None:
                contacts.append(player["contact"])
        near_contact = any(
            candidate is not None
            and len(candidate) >= 2
            and euclidean(point, candidate) <= tolerance
            for candidate in contacts
        )
        if near_contact:
            continue
        output[frame] = (0, observation[1], observation[2], "top_branch_veto", 0.0)
        vetoed.append(frame)
    return output, vetoed


def veto_short_unsupported_phase_branches(
    ball_dict,
    pose_evidence,
    suspicious_frames,
    scene_cuts=None,
    max_branch_frames=3,
    hard_jump_px=240.0,
    reversal_acceleration_px=45.0,
    static_branch_step_px=25.0,
    contact_protect_px=38.0,
    edge_protect_y=0.0,
    protected_frames=None,
):
    """Drop short detector branches that contradict the hitter direction.

    A single heatmap can jump from a real shuttle to a background object for
    two or three frames.  A global speed threshold would also erase genuine
    smashes, so this gate first splits a continuous run at a large jump or a
    direction reversal, then rejects only a short branch which is both
    unsupported by a player contact and headed the wrong way for that side.
    Tiny, nearly static branches embedded in a relaxed player box are removed
    as well.  Coordinates remain in the sidecar with an invisible provenance
    marker for auditing.
    """
    output = dict(ball_dict)
    suspicious = {int(frame) for frame in (suspicious_frames or ())}
    protected = {int(value) for value in (protected_frames or ())}
    cuts = sorted({int(cut) for cut in (scene_cuts or ())})
    coordinates = {}
    for frame, value in output.items():
        observation = _normalized_ball_observation(value)
        if observation is not None and str(observation[3]) == "model":
            coordinates[int(frame)] = np.asarray(observation[1:3], dtype=np.float64)

    def same_scene(left, right):
        return _same_scene(left, right, cuts)

    def pose_context(frame, point, include_relaxed=False):
        best = None
        evidence = (pose_evidence or {}).get(int(frame), {})
        player_list = evidence.get("players", [])
        if include_relaxed:
            player_list = evidence.get("veto_players", player_list)
        for player in player_list:
            box = player.get("box")
            if box is None:
                continue
            box_distance = _distance_to_box(point, box)
            contacts = list(player.get("wrists", []))
            if player.get("contact") is not None:
                contacts.append(player["contact"])
            contact_distance = min(
                (euclidean(point, candidate) for candidate in contacts),
                default=math.inf,
            )
            metric = min(box_distance, contact_distance)
            if best is None or metric < best[0]:
                best = (metric, str(player.get("side", "unknown")), box_distance, contact_distance)
        return best

    frames = sorted(coordinates)
    runs = []
    for frame in frames:
        if (
            not runs
            or frame != runs[-1][-1] + 1
            or not same_scene(runs[-1][-1], frame)
        ):
            runs.append([frame])
        else:
            runs[-1].append(frame)

    branch_vetoed = []
    static_vetoed = []
    for run in runs:
        branches = []
        start = 0
        start_reason = False
        for index in range(1, len(run)):
            previous = coordinates[run[index - 1]]
            current = coordinates[run[index]]
            velocity = current - previous
            jump = float(np.linalg.norm(velocity))
            split = jump >= float(hard_jump_px)
            if index >= 2:
                prior_velocity = coordinates[run[index - 1]] - coordinates[run[index - 2]]
                acceleration = float(np.linalg.norm(velocity - prior_velocity))
                reversal = float(np.dot(velocity, prior_velocity)) < 0.0
                split = split or (
                    acceleration >= float(reversal_acceleration_px) and reversal
                )
            if split:
                branches.append((run[start:index], start_reason))
                start = index
                start_reason = True
        branches.append((run[start:], start_reason))

        for branch, was_split in branches:
            if not was_split or len(branch) > max(1, int(max_branch_frames)):
                continue
            if not all(frame in suspicious for frame in branch):
                continue
            first = branch[0]
            if len(branch) >= 2:
                velocity = coordinates[branch[1]] - coordinates[branch[0]]
            else:
                previous_frames = [frame for frame in run if frame < first]
                velocity = (
                    coordinates[first] - coordinates[previous_frames[-1]]
                    if previous_frames else np.zeros(2, dtype=np.float64)
                )
            context = pose_context(first, coordinates[first])
            side = context[1] if context is not None else "unknown"
            # A far-side hitter sends the shuttle toward increasing image y;
            # a near-side hitter sends it toward decreasing y.  A short branch
            # going the opposite way is overwhelmingly a background switch.
            wrong_way = (
                (side == "far" and velocity[1] < -8.0)
                or (side == "near" and velocity[1] > 8.0)
            )
            unsupported = context is None or (
                context[2] > 42.0 and context[3] > 90.0
            )
            if wrong_way or unsupported:
                for frame in branch:
                    if frame in protected:
                        continue
                    observation = _normalized_ball_observation(output.get(frame))
                    if observation is None:
                        continue
                    if float(edge_protect_y) > 0 and float(observation[2]) <= float(edge_protect_y):
                        continue
                    output[frame] = (
                        0, observation[1], observation[2], "phase_branch_veto", 0.0
                    )
                    branch_vetoed.append(frame)

        # A separate conservative pass catches 2–3-frame static locks after a
        # missing frame (e.g. a player's shoe).  It does not touch a moving
        # high-clear branch.
        for index in range(len(run)):
            end = min(len(run), index + max(1, int(max_branch_frames)))
            branch = run[index:end]
            if len(branch) < 2 or not all(frame in suspicious for frame in branch):
                continue
            if any(
                float(np.linalg.norm(coordinates[branch[pos + 1]] - coordinates[branch[pos]]))
                > float(static_branch_step_px)
                for pos in range(len(branch) - 1)
            ):
                continue
            eligible = []
            for frame in branch:
                context = pose_context(frame, coordinates[frame], include_relaxed=True)
                if (
                    context is not None
                    # The relaxed phase-only pass may see a tiny far player
                    # whose box edge is 20–30 px away after interpolation;
                    # contact/wrist protection below keeps true racket hits.
                    and context[2] <= 32.0
                    and context[3] > float(contact_protect_px)
                ):
                    eligible.append(frame)
            if len(eligible) < 2:
                continue
            for frame in eligible:
                if frame in protected:
                    continue
                observation = _normalized_ball_observation(output.get(frame))
                if observation is None:
                    continue
                if float(edge_protect_y) > 0 and float(observation[2]) <= float(edge_protect_y):
                    continue
                output[frame] = (
                    0, observation[1], observation[2], "phase_static_veto", 0.0
                )
                static_vetoed.append(frame)

    return output, sorted(set(branch_vetoed + static_vetoed)), {
        "branch_vetoed": len(set(branch_vetoed)),
        "static_vetoed": len(set(static_vetoed)),
    }


def write_ball_tracking_csv(
    path,
    raw_ball,
    track,
    hit_events,
    rally_labels,
    detector_mode="tracknet",
    classical_track_ids=None,
):
    """Persist the post-processed provenance used by the renderer.

    Keeping this sidecar next to the rendered video makes it possible to
    audit which points were model observations, short-gap interpolation, or
    Kalman forecasts instead of treating every drawn pixel as ground truth.
    """
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    hits = {int(event.frame): event for event in hit_events}
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "Frame", "RawVisibility", "RawX", "RawY", "Visibility", "X", "Y",
            "RawSource", "RawConfidence", "Source", "Confidence",
            "ClassicalTrackID", "DetectorMode", "Hit", "HitScore",
            "HitUncertain", "HitSource", "HitTrackID", "HitX", "HitY",
            "HitBallObserved", "HitHitter", "HitHitterConfidence",
            "HitHitterSource", "RallyID", "HitRallyIndex",
            "HitVideoIndex", "ShotEndFrame", "ShotAvgSpeedMps",
            "ShotAvgSpeedKmh", "ShotDistanceM", "ShotObservedSeconds",
            "ShotElapsedSeconds", "ShotCoverage", "ShotSpeedQuality",
            "ShotSpeedMethod",
        ])
        for point in track:
            raw = raw_ball.get(int(point.frame), (0, 0, 0))
            is_render_only = str(point.source) == "offscreen_bridge"
            # A render-only offscreen bridge must never advertise an underlying
            # detector observation.  Keep RawVisibility/RawX/RawY explicitly
            # empty even when the source CSV contained a vetoed background
            # response at the same frame.
            if is_render_only:
                raw = (0, 0, 0, "missing", 0.0)
            event = None if is_render_only else hits.get(int(point.frame))
            raw_source = str(raw[3]) if len(raw) >= 4 else ("model" if int(raw[0]) else "missing")
            try:
                raw_confidence = float(raw[4]) if len(raw) >= 5 else (1.0 if int(raw[0]) else 0.0)
            except (TypeError, ValueError):
                raw_confidence = 0.0
            classical_track_id = "" if is_render_only else (classical_track_ids or {}).get(int(point.frame), "")
            rally_id = 0 if is_render_only else (
                int(rally_labels[point.frame]) if point.frame < len(rally_labels) else 0
            )
            writer.writerow([
                int(point.frame), int(raw[0]), int(raw[1]), int(raw[2]),
                int(point.visible), int(round(point.x)) if point.visible else 0,
                int(round(point.y)) if point.visible else 0,
                raw_source, f"{raw_confidence:.4f}", point.source,
                f"{float(point.confidence):.4f}", classical_track_id,
                detector_mode, int(event is not None),
                f"{float(event.score):.4f}" if event is not None else "",
                int(getattr(event, "uncertain", False)) if event is not None else 0,
                str(getattr(event, "source", "")) if event is not None else "",
                int(getattr(event, "track_id", -1)) if event is not None else "",
                int(round(event.x)) if event is not None else "",
                int(round(event.y)) if event is not None else "",
                int(getattr(event, "ball_observed", False)) if event is not None else 0,
                str(getattr(event, "hitter", "unknown")) if event is not None else "",
                f"{max(0.0, min(1.0, float(getattr(event, 'hitter_confidence', 0.0)))):.4f}"
                if event is not None else "",
                str(getattr(event, "hitter_source", "unknown")) if event is not None else "",
                rally_id,
                int(getattr(event, "rally_hit_index", 0)) if event is not None else "",
                int(getattr(event, "video_hit_index", 0)) if event is not None else "",
                int(getattr(event, "shot_end_frame", -1))
                if event is not None and int(getattr(event, "shot_end_frame", -1)) >= 0 else "",
                f"{float(event.shot_avg_speed_mps):.4f}"
                if event is not None and getattr(event, "shot_avg_speed_mps", None) is not None else "",
                f"{float(event.shot_avg_speed_mps) * 3.6:.3f}"
                if event is not None and getattr(event, "shot_avg_speed_mps", None) is not None else "",
                f"{float(getattr(event, 'shot_distance_m', 0.0)):.4f}"
                if event is not None and int(getattr(event, "shot_end_frame", -1)) >= 0 else "",
                f"{float(getattr(event, 'shot_observed_seconds', 0.0)):.4f}"
                if event is not None and int(getattr(event, "shot_end_frame", -1)) >= 0 else "",
                f"{float(getattr(event, 'shot_elapsed_seconds', 0.0)):.4f}"
                if event is not None and int(getattr(event, "shot_end_frame", -1)) >= 0 else "",
                f"{float(getattr(event, 'shot_coverage', 0.0)):.4f}"
                if event is not None and int(getattr(event, "shot_end_frame", -1)) >= 0 else "",
                str(getattr(event, "shot_speed_quality", "pending"))
                if event is not None else "",
                str(getattr(event, "shot_speed_method", "pending"))
                if event is not None else "",
            ])


def merge_hit_events(
    curvature_events,
    supplemental_events,
    min_separation_frames=8,
    same_track_separation=8,
    spatial_radius=120.0,
):
    """Evidence-aware temporal NMS with visual timing kept authoritative.

    Audio candidates are deliberately supplemental: they can confirm a nearby
    visual candidate and provide a hitter-side hint, but their impulse frame
    must not replace the trajectory turn used for the rendered marker.  The
    previous score-first NMS gave ``audio_ball`` a larger bonus than
    ``curvature`` and consequently shifted otherwise correct hits by several
    frames.
    """

    del spatial_radius  # A racket cannot make two contacts a few frames apart.
    candidates = list(curvature_events)
    if not candidates:
        # Audio is corroborating evidence only; it cannot manufacture an
        # event when no visual trajectory candidate survived validation.
        return []

    # Associate supplemental evidence without changing frame/x/y/source.
    # An audio impulse is useful for timing, but a loose nearest-body result
    # must never become a visual event's hitter identity.  Only a strict
    # virtual-racket contact association may transfer, and only when it has
    # one unambiguous visual neighbour in a tight timing window.  This avoids
    # copying a side from one of two adjacent false/duplicate turns.
    association_limit = max(1, min(3, int(min_separation_frames)))
    for supplemental in supplemental_events:
        supplemental_side = str(getattr(supplemental, "hitter", "unknown"))
        supplemental_source = str(getattr(supplemental, "hitter_source", "")).strip().lower()
        try:
            supplemental_confidence = float(getattr(supplemental, "hitter_confidence", 0.0))
        except (TypeError, ValueError):
            supplemental_confidence = 0.0
        if (
            supplemental_side not in {"near", "far"}
            or supplemental_source != "pose_contact"
            or not math.isfinite(supplemental_confidence)
            or supplemental_confidence < 0.63
        ):
            continue
        nearby = [
            event for event in candidates
            if abs(int(event.frame) - int(supplemental.frame)) <= association_limit
        ]
        if len(nearby) != 1:
            continue
        visual = nearby[0]
        if (
            str(getattr(visual, "hitter", "unknown")) == "unknown"
        ):
            visual.hitter = supplemental_side
            visual.hitter_confidence = supplemental_confidence
            visual.hitter_source = supplemental_source

    source_bonus = {
        "classical_piecewise": 0.18,
        # A top-edge exit is stricter than a curvature point at the same
        # frame: it keeps the last ordinary measured contact while the
        # primary phase's curvature derivative often lands on the following
        # edge-labelled point.  Prefer it so the later evidence validator
        # does not discard the whole contact for lacking right-side ordinary
        # observations.
        "edge_exit": 0.25,
        "curvature": 0.12,
        "classical_endpoint": 0.04,
    }
    time_limit = max(1, int(min_separation_frames))
    same_track_limit = max(time_limit, int(same_track_separation))
    selected = []
    ordered = sorted(
        candidates,
        key=lambda event: (
            float(event.score) + source_bonus.get(str(event.source), 0.0),
            float(event.score),
        ),
        reverse=True,
    )
    for candidate in ordered:
        duplicate = False
        for chosen in selected:
            delta = abs(int(candidate.frame) - int(chosen.frame))
            same_track = (
                int(getattr(candidate, "track_id", -1)) >= 0
                and int(getattr(candidate, "track_id", -1))
                == int(getattr(chosen, "track_id", -1))
            )
            if delta < time_limit or (same_track and delta < same_track_limit):
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    selected.sort(key=lambda event: event.frame)
    return selected


# Strict pose associations begin at 0.63.  A 0.75 anchor keeps a meaningful
# margin above that floor while retaining the reliable 0.75--0.80 contacts
# observed in distant broadcast views.
_POSE_CONTACT_ANCHOR_CONFIDENCE = 0.75
_RALLY_ALTERNATION_CONFIDENCE = 0.55
_RALLY_ALTERNATION_SOURCE = "rally_alternation_inferred"


def _hitter_confidence(event):
    """Return a finite normalized hitter-attribution confidence."""
    try:
        confidence = float(getattr(event, "hitter_confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence)) if math.isfinite(confidence) else 0.0


def _is_strong_pose_contact_anchor(event, minimum_confidence):
    """Whether an event can safely anchor a local rally-side inference."""
    return (
        str(getattr(event, "hitter", "unknown")).strip().lower() in {"near", "far"}
        and str(getattr(event, "hitter_source", "")).strip().lower() == "pose_contact"
        and _hitter_confidence(event) >= float(minimum_confidence)
    )


def infer_single_unknown_hitter_from_rally_anchors(
    hit_events,
    *,
    anchor_min_confidence=_POSE_CONTACT_ANCHOR_CONFIDENCE,
    min_gap_frames=10,
):
    """Apply a deliberately narrow badminton alternation consistency check.

    The near/far hit totals are *not* normalized.  An attribution is added
    only when one unknown event is directly bracketed, in the same segmented
    rally, by two high-confidence ``pose_contact`` events for the same side.
    That three-hit pattern has exactly one alternating explanation.  Any
    missed hit, duplicate turn, opposite-side anchor, rally boundary, or
    weak/legacy side hint remains untouched for video review.

    The return value is diagnostics for logs/tests; callers retain every raw
    trajectory event and can audit an inferred label through its source.
    """
    try:
        minimum_confidence = max(0.0, min(1.0, float(anchor_min_confidence)))
    except (TypeError, ValueError):
        minimum_confidence = _POSE_CONTACT_ANCHOR_CONFIDENCE
    try:
        minimum_gap = max(0, int(min_gap_frames))
    except (TypeError, ValueError):
        minimum_gap = 10

    stats = {
        "rallies": 0,
        "strong_pose_anchors": 0,
        "single_unknown_gaps": 0,
        "filled": 0,
        "skipped_opposite_anchors": 0,
        "skipped_close_events": 0,
        "skipped_collision_sources": 0,
        "near": 0,
        "far": 0,
        "unknown": 0,
        "rallies_over_balance_tolerance": 0,
    }
    grouped = {}
    for original_index, event in enumerate(hit_events or ()):
        try:
            rally_id = int(getattr(event, "rally", 0))
            frame = int(getattr(event, "frame", -1))
        except (TypeError, ValueError):
            continue
        if rally_id <= 0 or frame < 0:
            continue
        grouped.setdefault(rally_id, []).append((frame, original_index, event))

    stats["rallies"] = len(grouped)
    for grouped_events in grouped.values():
        grouped_events.sort(key=lambda item: (item[0], item[1]))
        events = [item[2] for item in grouped_events]
        stats["strong_pose_anchors"] += sum(
            _is_strong_pose_contact_anchor(event, minimum_confidence)
            for event in events
        )
        for previous, center, following in zip(events, events[1:], events[2:]):
            if str(getattr(center, "hitter", "unknown")).strip().lower() != "unknown":
                continue
            if not (
                _is_strong_pose_contact_anchor(previous, minimum_confidence)
                and _is_strong_pose_contact_anchor(following, minimum_confidence)
            ):
                continue
            stats["single_unknown_gaps"] += 1
            if any(
                str(getattr(event, "source", "")).strip().lower()
                in {"edge_exit", "classical_endpoint"}
                for event in (previous, center, following)
            ):
                # These endpoint-style candidates can be valid contacts, but
                # their timing is intentionally less stable than an ordinary
                # bidirectional turn.  Do not use them to establish parity.
                stats["skipped_collision_sources"] += 1
                continue
            try:
                left_gap = int(getattr(center, "frame")) - int(getattr(previous, "frame"))
                right_gap = int(getattr(following, "frame")) - int(getattr(center, "frame"))
            except (TypeError, ValueError):
                continue
            # A tiny separation usually means duplicate/ambiguous turn
            # candidates, rather than three physical badminton contacts.
            if left_gap <= minimum_gap or right_gap <= minimum_gap:
                stats["skipped_close_events"] += 1
                continue
            left_side = str(getattr(previous, "hitter", "unknown")).strip().lower()
            right_side = str(getattr(following, "hitter", "unknown")).strip().lower()
            if left_side != right_side:
                # With a single event in between, opposite anchors do not
                # yield a unique alternation explanation.
                stats["skipped_opposite_anchors"] += 1
                continue
            center.hitter = "far" if left_side == "near" else "near"
            center.hitter_confidence = _RALLY_ALTERNATION_CONFIDENCE
            center.hitter_source = _RALLY_ALTERNATION_SOURCE
            stats["filled"] += 1

        near_count = sum(
            str(getattr(event, "hitter", "unknown")).strip().lower() == "near"
            for event in events
        )
        far_count = sum(
            str(getattr(event, "hitter", "unknown")).strip().lower() == "far"
            for event in events
        )
        if abs(near_count - far_count) > 1:
            stats["rallies_over_balance_tolerance"] += 1

    for event in hit_events or ():
        side = str(getattr(event, "hitter", "unknown")).strip().lower()
        if side in {"near", "far"}:
            stats[side] += 1
        else:
            stats["unknown"] += 1
    return stats


def merge_visual_hit_candidates(
    reference_events,
    fused_events,
    min_separation_frames=8,
):
    """Union primary-reference and final-fused visual hit candidates.

    The primary TrackNet pass is useful when phase fusion has to suppress an
    isolated false positive, while the fused pass can recover a genuine turn
    seen consistently by both phases.  These are complementary candidate
    sources, not fallback alternatives: using ``reference or fused`` silently
    loses every fused-only contact as soon as the primary pass finds any hit.
    """

    return merge_hit_events(
        list(reference_events or ()) + list(fused_events or ()),
        [],
        min_separation_frames=min_separation_frames,
        same_track_separation=min_separation_frames,
    )


def has_confirmed_model_audio_evidence(
    event,
    event_point,
    audio_frames,
    audio_tolerance_frames,
):
    """Whether a low-score curvature turn is nevertheless a confirmed hit.

    A raw model point at the candidate frame, bidirectional trajectory support
    (already enforced by ``validate_hit_event_evidence``), and a nearby audio
    impulse are substantially stronger than a score just below the historic
    0.72 display threshold.  This helper affects only the ``HIT?`` label; it
    does not weaken event validation or create contacts from audio alone.
    """

    if (
        event_point is None
        or not bool(getattr(event_point, "visible", False))
        or not bool(getattr(event_point, "raw_visible", False))
        or str(getattr(event_point, "source", "")) != "model"
        or not bool(getattr(event, "ball_observed", False))
        or str(getattr(event, "source", "")).startswith("audio_")
    ):
        return False
    tolerance = max(0, int(audio_tolerance_frames))
    return any(
        abs(int(event.frame) - int(audio_frame)) <= tolerance
        for audio_frame in (audio_frames or ())
    )


def _quadratic_track_fit(points, frame0):
    frames = np.asarray([point.frame for point in points], dtype=np.float64)
    coordinates = np.asarray([[point.x, point.y] for point in points], dtype=np.float64)
    relative = frames - float(frame0)
    degree = min(2, len(points) - 1)
    coefficient_x = np.polyfit(relative, coordinates[:, 0], degree)
    coefficient_y = np.polyfit(relative, coordinates[:, 1], degree)
    predicted = np.column_stack((
        np.polyval(coefficient_x, relative),
        np.polyval(coefficient_y, relative),
    ))
    residual = coordinates - predicted
    sse = float(np.sum(residual * residual))
    rmse = math.sqrt(sse / max(1, len(points)))
    return coefficient_x, coefficient_y, sse, rmse


def _fit_position(fit, frame, frame0):
    coefficient_x, coefficient_y, _, _ = fit
    relative = float(frame) - float(frame0)
    return np.asarray([
        np.polyval(coefficient_x, relative),
        np.polyval(coefficient_y, relative),
    ], dtype=np.float64)


def _fit_velocity(fit, frame, frame0):
    coefficient_x, coefficient_y, _, _ = fit
    relative = float(frame) - float(frame0)
    return np.asarray([
        np.polyval(np.polyder(coefficient_x), relative),
        np.polyval(np.polyder(coefficient_y), relative),
    ], dtype=np.float64)


def _vector_angle_degrees(first, second):
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-6 or second_norm < 1e-6:
        return 0.0
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def detect_classical_visual_hits(
    classical_result,
    ball_track,
    contact_min_y,
    endpoint_gap_frames=48,
):
    """Detect classical-track contacts without calling parabolic apexes hits.

    Adjacent tracklets first pass a bidirectional quadratic continuity test.
    A continuous join is treated as one flight.  Within each joined flight a
    contact must make two quadratic fits substantially better than one fit and
    reverse the velocity sharply.  Low-altitude, non-edge endpoints remain
    explicit uncertain candidates when another flight resumes soon.
    """

    if classical_result is None:
        return [], set(), set()
    observations_by_track = {}
    for observation in classical_result.observations.values():
        observations_by_track.setdefault(int(observation.track_id), []).append(observation)
    for points in observations_by_track.values():
        points.sort(key=lambda point: point.frame)
    ranges = [
        item for item in sorted(
            classical_result.track_ranges,
            key=lambda item: (item.start_frame, item.end_frame),
        )
        if int(item.track_id) in observations_by_track
    ]
    if not ranges:
        return [], set(), set()

    stitched_end_ids = set()
    stitched_starts = set()
    for first, second in zip(ranges, ranges[1:]):
        left = observations_by_track[int(first.track_id)]
        right = observations_by_track[int(second.track_id)]
        gap = int(right[0].frame) - int(left[-1].frame)
        if gap <= 0 or gap > 18:
            continue
        origin = float(left[-1].frame)
        left_fit = _quadratic_track_fit(left[-min(9, len(left)):], origin)
        right_fit = _quadratic_track_fit(right[:min(9, len(right))], origin)
        prediction_error = float(np.linalg.norm(
            _fit_position(left_fit, right[0].frame, origin)
            - np.asarray([right[0].x, right[0].y], dtype=np.float64)
        ))
        reverse_error = float(np.linalg.norm(
            _fit_position(right_fit, left[-1].frame, origin)
            - np.asarray([left[-1].x, left[-1].y], dtype=np.float64)
        ))
        join_frame = 0.5 * (left[-1].frame + right[0].frame)
        angle = _vector_angle_degrees(
            _fit_velocity(left_fit, join_frame, origin),
            _fit_velocity(right_fit, join_frame, origin),
        )
        combined = left[-min(10, len(left)):] + right[:min(10, len(right))]
        combined_fit = _quadratic_track_fit(combined, join_frame)
        gate = max(45.0, 6.0 * gap)
        if (
            prediction_error <= gate
            and reverse_error <= 1.35 * gate
            and angle <= 28.0
            and combined_fit[3] <= 18.0
        ):
            stitched_end_ids.add(int(first.track_id))
            stitched_starts.add(int(right[0].frame))

    groups = []
    current_group = []
    for index, track_range in enumerate(ranges):
        points = observations_by_track[int(track_range.track_id)]
        current_group.extend(points)
        if int(track_range.track_id) not in stitched_end_ids:
            groups.append(current_group)
            current_group = []
    if current_group:
        groups.append(current_group)

    events = []
    support_frames = set()
    minimum_y = float(contact_min_y)
    for track_range in ranges:
        points = observations_by_track[int(track_range.track_id)]
        for point in (points[0], points[-1]):
            if float(point.y) >= minimum_y:
                support_frames.add(int(point.frame))

    for points in groups:
        if len(points) < 11:
            continue
        candidates = []
        for split_frame in range(points[0].frame + 5, points[-1].frame - 4):
            nearby = [
                point for point in points
                if abs(int(point.frame) - int(split_frame)) <= 13
            ]
            left = [point for point in nearby if point.frame <= split_frame - 1]
            right = [point for point in nearby if point.frame >= split_frame + 1]
            if len(left) < 5 or len(right) < 5:
                continue
            whole = _quadratic_track_fit(left + right, split_frame)
            before = _quadratic_track_fit(left, split_frame)
            after = _quadratic_track_fit(right, split_frame)
            scalar_count = 2 * (len(left) + len(right))
            whole_bic = scalar_count * math.log(
                max(whole[2] / scalar_count, 1e-9)
            ) + 6 * math.log(scalar_count)
            split_sse = before[2] + after[2]
            split_bic = scalar_count * math.log(
                max(split_sse / scalar_count, 1e-9)
            ) + 12 * math.log(scalar_count)
            delta_bic = float(whole_bic - split_bic)
            angle = _vector_angle_degrees(
                _fit_velocity(before, split_frame, split_frame),
                _fit_velocity(after, split_frame, split_frame),
            )
            nearest = min(points, key=lambda point: abs(point.frame - split_frame))
            if nearest.y < minimum_y or delta_bic < 50.0 or angle < 90.0:
                continue
            candidates.append((delta_bic, angle, int(split_frame), nearest))

        selected = []
        for candidate in sorted(candidates, reverse=True):
            if all(abs(candidate[2] - old[2]) >= 16 for old in selected):
                selected.append(candidate)
        for delta_bic, angle, frame, nearest in selected:
            if 0 <= frame < len(ball_track) and ball_track[frame].visible:
                point = ball_track[frame]
                x, y = float(point.x), float(point.y)
                observed = bool(point.raw_visible)
            else:
                x, y = float(nearest.x), float(nearest.y)
                observed = int(nearest.frame) == frame
            events.append(HitEvent(
                frame=frame,
                x=x,
                y=y,
                score=float(np.clip(0.68 + 0.001 * delta_bic + 0.0005 * angle, 0.72, 0.96)),
                speed=0.0,
                uncertain=True,
                source="classical_piecewise",
                track_id=int(nearest.track_id),
                ball_observed=observed,
            ))
            support_frames.add(frame)

    endpoint_by_track = {
        int(candidate.track_id): candidate
        for candidate in classical_result.hit_candidates
    }
    for current_range, next_range in zip(ranges, ranges[1:]):
        track_id = int(current_range.track_id)
        if track_id in stitched_end_ids:
            continue
        candidate = endpoint_by_track.get(track_id)
        gap = int(next_range.start_frame) - int(current_range.end_frame)
        if candidate is None or not (0 < gap <= max(1, int(endpoint_gap_frames))):
            continue
        frame = int(candidate.frame)
        if 0 <= frame < len(ball_track) and ball_track[frame].visible:
            point = ball_track[frame]
            x, y = float(point.x), float(point.y)
            observed = bool(point.raw_visible)
        else:
            x, y = float(candidate.x), float(candidate.y)
            observed = False
        events.append(HitEvent(
            frame=frame,
            x=x,
            y=y,
            score=min(0.76, 0.48 + 0.26 * float(candidate.score)),
            speed=float(candidate.speed),
            uncertain=True,
            source="classical_endpoint",
            track_id=track_id,
            ball_observed=observed,
        ))
        support_frames.add(frame)

    return merge_hit_events(events, [], min_separation_frames=8), support_frames, stitched_starts


def _estimate_ball_contact(track, frame, max_distance=18):
    """Return a nearby/interpolated ball point for an audio-timed contact."""

    allowed_sources = {"model", "classical", "interp"}
    index = int(frame)
    if 0 <= index < len(track):
        point = track[index]
        if point.visible and point.source in allowed_sources:
            return (
                (float(point.x), float(point.y)),
                bool(point.raw_visible),
                index,
            )
    left = None
    right = None
    limit = max(1, int(max_distance))
    for distance in range(1, limit + 1):
        left_index = index - distance
        if left is None and 0 <= left_index < len(track):
            point = track[left_index]
            if point.visible and point.source in allowed_sources:
                left = point
        right_index = index + distance
        if right is None and 0 <= right_index < len(track):
            point = track[right_index]
            if point.visible and point.source in allowed_sources:
                right = point
        if left is not None and right is not None:
            break
    if left is not None and right is not None:
        span = max(1, int(right.frame) - int(left.frame))
        alpha = float(index - int(left.frame)) / span
        return (
            (
                float(left.x) * (1.0 - alpha) + float(right.x) * alpha,
                float(left.y) * (1.0 - alpha) + float(right.y) * alpha,
            ),
            False,
            int(left.frame) if alpha < 0.5 else int(right.frame),
        )
    nearest = left if right is None else right
    if nearest is None:
        return None
    return ((float(nearest.x), float(nearest.y)), False, int(nearest.frame))


def _distance_to_box(point, box):
    x, y = map(float, point)
    x1, y1, x2, y2 = map(float, box)
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def _strict_hitter_from_pose_evidence(ball_xy, pose_evidence):
    """Associate a measured contact only with a clearly closest racket tip.

    A shuttle high above the court can be visually closer to the opposite
    player's body box than to the actual striker.  Body-box distance therefore
    remains a false-positive veto only; attribution requires the virtual tip
    of a visible arm and a meaningful two-side margin.
    """
    if pose_evidence is None or ball_xy is None:
        return None
    try:
        ball_x, ball_y = map(float, ball_xy)
    except (TypeError, ValueError):
        return None
    # Pose tracking can emit two overlapping person boxes for the same court
    # side.  They are not competing hitters: retain that side's closest
    # virtual racket tip before comparing near versus far.
    closest_by_side = {}
    for player in pose_evidence.get("players", []):
        side = str(player.get("side", ""))
        contact = player.get("contact")
        torso = player.get("torso")
        if side not in {"near", "far"} or contact is None:
            continue
        try:
            torso_value = float(torso)
            contact_x, contact_y = map(float, contact)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(torso_value) or torso_value < 4.0:
            continue
        distance = math.hypot(contact_x - ball_x, contact_y - ball_y) / torso_value
        if not math.isfinite(distance):
            continue
        previous = closest_by_side.get(side)
        if previous is None or distance < previous:
            closest_by_side[side] = distance
    candidates = [
        (distance, side)
        for side, distance in closest_by_side.items()
    ]
    if not candidates:
        return None
    candidates.sort()
    best_distance, best_side = candidates[0]
    second_distance = candidates[1][0] if len(candidates) > 1 else None
    maximum_distance = 0.75 if second_distance is None else 1.05
    if best_distance > maximum_distance:
        return None
    if second_distance is not None and second_distance - best_distance < 0.35:
        return None
    closeness = max(0.0, min(1.0, 1.0 - best_distance / maximum_distance))
    separation = (
        1.0 if second_distance is None
        else max(0.0, min(1.0, (second_distance - best_distance - 0.35) / 0.75))
    )
    confidence = min(0.94, 0.63 + 0.19 * closeness + 0.12 * separation)
    return best_side, float(confidence)


def apply_strict_pose_hitter_associations(hit_events, pose_evidence):
    """Attach direct hitter evidence to final visual/audio hit candidates.

    ``build_audio_hit_events`` may localize a subset of audio-supported
    contacts early, but the post-filter candidate list also contains ordinary
    curvature/endpoint hits.  Run the same strict virtual-racket association
    over *all* of those candidates before evidence validation and temporal
    merge, so a valid visual contact is not left unknown merely because audio
    localization did not create a separate event.

    Existing stronger ``pose_contact`` metadata is retained when the repeated
    frame-local association is weaker.  No box-distance or alternating-side
    hint can enter here: failures remain explicitly unknown.
    """
    stats = {
        "candidates": 0,
        "associated": 0,
        "updated": 0,
        "preserved_stronger": 0,
    }
    evidence_by_frame = pose_evidence or {}
    for event in hit_events or ():
        stats["candidates"] += 1
        try:
            frame = int(getattr(event, "frame"))
            ball_xy = (float(getattr(event, "x")), float(getattr(event, "y")))
        except (TypeError, ValueError, AttributeError):
            continue
        association = _strict_hitter_from_pose_evidence(
            ball_xy,
            evidence_by_frame.get(frame),
        )
        if association is None:
            continue
        side, confidence = association
        stats["associated"] += 1
        previous_side = str(getattr(event, "hitter", "unknown")).strip().lower()
        previous_source = str(getattr(event, "hitter_source", "")).strip().lower()
        previous_confidence = _hitter_confidence(event)
        # A second pass at the same frame can be slightly weaker because an
        # audio event uses an interpolated shuttle coordinate.  Do not reduce
        # a pre-existing direct pose association merely to make fields match.
        if (
            previous_source == "pose_contact"
            and previous_side in {"near", "far"}
            and previous_confidence > float(confidence)
        ):
            stats["preserved_stronger"] += 1
            continue
        if (
            previous_side != side
            or previous_source != "pose_contact"
            or abs(previous_confidence - float(confidence)) > 1e-9
        ):
            stats["updated"] += 1
        event.hitter = side
        event.hitter_confidence = float(confidence)
        event.hitter_source = "pose_contact"
    return stats


def _infer_audio_pose_contacts(
    model,
    video_path,
    frames,
    court_quad,
    midline_y,
    args,
    pose_imgsz=None,
    relaxed_player_filter=False,
):
    """Run a tiny pose pre-pass for hitter association and body vetoes.

    Pose coordinates are contextual evidence only.  They must never be used
    as the rendered shuttle/contact location when the ball is missing.
    """

    if model is None or not frames:
        return {}
    target_frames = sorted(set(int(frame) for frame in frames if int(frame) >= 0))
    if not target_frames:
        return {}
    target_set = set(target_frames)
    maximum_frame = target_frames[-1]
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return {}
    evidence = {}
    pending = []
    batch_size = max(1, int(args.audio_hit_pose_batch_size))
    relaxed_quad = None
    if relaxed_player_filter:
        # The tiny far/right player can have a foot just outside the fixed
        # floor quad.  For the phase-veto pass only, retain boxes intersecting
        # a modestly expanded court; ordinary hitter association keeps the
        # strict foot-in-court filter below.
        relaxed_quad = _expanded_ball_quad(
            court_quad,
            top_extension=180.0,
            side_padding=100.0,
            bottom_padding=60.0,
        )

    def process_batch(batch):
        if not batch:
            return
        predict_kwargs = {
            "source": [item[1] for item in batch],
            "classes": [0],
            "conf": args.conf_thres,
            "verbose": False,
            # The far player is tiny in a 1080p broadcast.  This pre-pass has
            # only a few dozen frames, so use a larger pose input than the
            # sparse CPU render loop when necessary.
            "imgsz": max(
                int(args.imgsz),
                int(args.audio_hit_pose_imgsz if pose_imgsz is None else pose_imgsz),
            ),
            "device": args.device if args.device else None,
        }
        if args.half:
            predict_kwargs["half"] = True
        try:
            results = model.predict(**predict_kwargs)
        except Exception as exc:
            print(f"[WARN] audio-hit pose localization failed: {exc}")
            return
        for (frame_number, image), result in zip(batch, results):
            frame_height, frame_width = image.shape[:2]
            boxes = (
                result.boxes.xyxy.cpu().numpy()
                if result.boxes is not None and len(result.boxes) else
                np.empty((0, 4), dtype=np.float32)
            )
            box_confidence = (
                result.boxes.conf.cpu().numpy()
                if result.boxes is not None and result.boxes.conf is not None else
                np.ones(len(boxes), dtype=np.float32)
            )
            keypoints = None
            keypoint_confidence = None
            if result.keypoints is not None and result.keypoints.xy is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                if result.keypoints.conf is not None:
                    keypoint_confidence = result.keypoints.conf.cpu().numpy()

            players = []
            veto_players = []
            active = None
            contacts_by_side = {}
            for person_index, box in enumerate(boxes):
                x1, y1, x2, y2 = map(float, box)
                width = x2 - x1
                height = y2 - y1
                if width < args.min_box_w or height < args.min_box_h:
                    continue
                foot = (0.5 * (x1 + x2), y2)
                foot_in_court = point_in_quad(foot, court_quad)
                if not foot_in_court:
                    if not relaxed_player_filter or relaxed_quad is None:
                        continue
                    corners = (
                        (x1, y1), (x2, y1), (x2, y2), (x1, y2)
                    )
                    if not any(point_in_quad(corner, relaxed_quad) for corner in corners):
                        continue
                side = "far" if foot[1] < float(midline_y) else "near"
                player = {
                    "box": (x1, y1, x2, y2),
                    "side": side,
                    "wrists": [],
                    "contact": None,
                    "torso": None,
                }
                best_arm = None
                if keypoints is not None and person_index < len(keypoints):
                    person_points = keypoints[person_index]
                    person_confidence = (
                        keypoint_confidence[person_index]
                        if keypoint_confidence is not None
                        and person_index < len(keypoint_confidence) else
                        np.ones(len(person_points), dtype=np.float32)
                    )
                    torso_points = []
                    for point_index in (5, 6, 11, 12):
                        if (
                            point_index < len(person_points)
                            and point_index < len(person_confidence)
                            and float(person_confidence[point_index]) >= 0.18
                        ):
                            torso_points.append(person_points[point_index].astype(np.float64))
                        else:
                            torso_points = []
                            break
                    if len(torso_points) == 4:
                        shoulder_center = 0.5 * (torso_points[0] + torso_points[1])
                        hip_center = 0.5 * (torso_points[2] + torso_points[3])
                        torso_length = float(np.linalg.norm(shoulder_center - hip_center))
                        if torso_length >= 4.0:
                            player["torso"] = torso_length
                    for shoulder_index, elbow_index, wrist_index in ((5, 7, 9), (6, 8, 10)):
                        if (
                            wrist_index >= len(person_points)
                            or min(
                                float(person_confidence[shoulder_index]),
                                float(person_confidence[elbow_index]),
                                float(person_confidence[wrist_index]),
                            ) < 0.18
                        ):
                            continue
                        shoulder = person_points[shoulder_index].astype(np.float64)
                        elbow = person_points[elbow_index].astype(np.float64)
                        wrist = person_points[wrist_index].astype(np.float64)
                        player["wrists"].append((float(wrist[0]), float(wrist[1])))
                        extension = float(np.linalg.norm(wrist - shoulder))
                        extension += 0.35 * float(np.linalg.norm(wrist - elbow))
                        extension += 0.20 * max(0.0, float(shoulder[1] - wrist[1]))
                        arm_score = extension / max(1.0, height)
                        arm_score += 0.05 * float(box_confidence[person_index])
                        contact = wrist + 0.35 * (wrist - elbow)
                        contact[0] = np.clip(contact[0], 0.0, frame_width - 1.0)
                        contact[1] = np.clip(contact[1], 0.0, frame_height - 1.0)
                        candidate = (
                            arm_score,
                            (float(contact[0]), float(contact[1])),
                            side,
                        )
                        if best_arm is None or candidate[0] > best_arm[0]:
                            best_arm = candidate
                if best_arm is not None:
                    player["contact"] = best_arm[1]
                veto_players.append(player)
                if foot_in_court:
                    players.append(player)
                if best_arm is not None and (active is None or best_arm[0] > active[0]):
                    active = best_arm
                if best_arm is not None:
                    previous_side = contacts_by_side.get(side)
                    if previous_side is None or best_arm[0] > previous_side[0]:
                        contacts_by_side[side] = best_arm
            # Retain boxes even when no arm keypoints passed the confidence
            # gate.  A shirt/head TrackNet false positive still needs the hard
            # player-body veto below; hitter/contact fields remain optional.
            if active is not None or players:
                evidence[int(frame_number)] = {
                    "contact": active[1] if active is not None else None,
                    "hitter": active[2] if active is not None else "unknown",
                    "contacts_by_side": {
                        side: value[1] for side, value in contacts_by_side.items()
                    },
                    "players": players,
                    "veto_players": veto_players,
                    "relaxed_player_filter": bool(relaxed_player_filter),
                }

    frame_index = 0
    try:
        while frame_index <= maximum_frame:
            ok, image = capture.read()
            if not ok:
                break
            if frame_index in target_set:
                pending.append((frame_index, image))
                if len(pending) >= batch_size:
                    process_batch(pending)
                    pending = []
            frame_index += 1
        process_batch(pending)
    finally:
        capture.release()
    return evidence


def build_audio_hit_events(
    video_path,
    fps,
    ball_track,
    scene_cuts,
    visual_support_frames,
    classical_track_ids,
    model,
    court_quad,
    midline_y,
    args,
):
    """Fuse broadband audio timing with a measured shuttle trajectory.

    Pose can identify the nearest player, but it is never allowed to invent a
    contact coordinate.  A candidate without a nearby ball estimate is
    discarded here and again by the final evidence validator.
    """

    if not args.audio_hit_detection:
        return [], [], False
    try:
        from audio_hit_detector import detect_audio_hit_candidates
    except Exception as exc:
        print(f"[WARN] audio-hit detector unavailable: {exc}")
        return [], [], False

    real_frames = [
        int(point.frame) for point in ball_track
        if point.visible and point.source in ("model", "classical")
    ]
    start_frame = 0
    stop_frame = len(ball_track)
    if real_frames:
        start_frame = max(0, min(real_frames) - int(round(2.0 * fps)))
        stop_frame = min(
            stop_frame,
            max(real_frames) + int(round(args.audio_hit_post_roll_seconds * fps)),
        )
    if scene_cuts:
        stop_frame = min(stop_frame, min(int(frame) for frame in scene_cuts))
    try:
        candidates, audio_available = detect_audio_hit_candidates(
            video_path,
            fps,
            start_frame=start_frame,
            stop_frame=stop_frame,
            score_threshold=args.audio_hit_score_threshold,
            nms_frames=args.audio_hit_nms_frames,
            return_audio_available=True,
        )
    except Exception as exc:
        print(f"[WARN] audio-hit detection failed: {exc}")
        return [], [], False

    if not audio_available:
        print("[INFO] source has no decodable audio; using visual hit evidence only")

    support = set(int(frame) for frame in visual_support_frames)
    selected = []
    for candidate in candidates:
        visually_supported = any(
            abs(int(candidate.frame) - frame) <= int(args.audio_hit_visual_tolerance)
            for frame in support
        )
        # Audio may confirm/refine a visual trajectory junction, but a loud
        # shoe/crowd impulse must not manufacture a hit by itself.
        if visually_supported:
            selected.append(candidate)

    pose_evidence = _infer_audio_pose_contacts(
        model,
        video_path,
        [candidate.frame for candidate in selected],
        court_quad,
        midline_y,
        args,
    ) if args.audio_hit_pose_fallback else {}

    events = []
    for candidate in selected:
        ball_estimate = _estimate_ball_contact(
            ball_track,
            candidate.frame,
            max_distance=args.audio_hit_ball_search_frames,
        )
        if ball_estimate is None:
            continue
        pose = pose_evidence.get(int(candidate.frame))
        hitter = "unknown"
        hitter_confidence = 0.0
        hitter_source = "unknown"
        association = _strict_hitter_from_pose_evidence(ball_estimate[0], pose)
        if association is not None:
            hitter, hitter_confidence = association
            hitter_source = "pose_contact"

        position, observed, source_frame = ball_estimate
        source = "audio_ball"
        track_id = int(classical_track_ids.get(int(source_frame), -1))

        normalized_score = float(np.clip(
            0.62 + 0.055 * (float(candidate.score) - args.audio_hit_score_threshold),
            0.62,
            0.94,
        ))
        events.append(HitEvent(
            frame=int(candidate.frame),
            x=float(position[0]),
            y=float(position[1]),
            score=normalized_score,
            speed=0.0,
            hitter=hitter,
            uncertain=True,
            source=source,
            track_id=track_id,
            ball_observed=bool(observed),
            hitter_confidence=float(hitter_confidence),
            hitter_source=hitter_source,
        ))
    return events, candidates, bool(audio_available)


def detect_scene_cuts(video_path, threshold=28.0, min_separation=8, return_frame_count=False):
    """Find hard camera cuts with a cheap low-resolution frame difference.

    The TrackNet/overlay court calibration is image-space.  A shot change
    therefore must break both the Kalman state and the drawn trail; otherwise
    the last point from the old camera is connected to the first point from
    the new camera.  We use a robust threshold (median + MAD) so ordinary
    player motion does not become a rally boundary.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ([], 0) if return_frame_count else []
    prev = None
    diffs = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diffs.append((frame_idx, float(cv2.absdiff(gray, prev).mean())))
        prev = gray
        frame_idx += 1
    cap.release()
    if not diffs:
        result = []
        return (result, frame_idx) if return_frame_count else result
    values = np.asarray([d for _, d in diffs], dtype=np.float32)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    # A hard cut in the supplied broadcast is ~48 while ordinary motion is
    # below ~18.  Keep an absolute floor for very static clips and a robust
    # relative term for noisy encodes.
    robust_limit = med + max(6.0, 6.0 * mad)
    limit = max(float(threshold), robust_limit, med * 2.4)
    candidates = [(i, d) for i, d in diffs if i > 1 and d >= limit]
    selected = []
    for i, d in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(abs(i - old) >= max(1, int(min_separation)) for old in selected):
            selected.append(i)
    result = sorted(selected)
    return (result, frame_idx) if return_frame_count else result


def _expanded_ball_quad(court_quad, top_extension=60.0, side_padding=40.0, bottom_padding=30.0):
    """Return the same conservative airborne-ball quad used by post-process."""
    quad = np.asarray(court_quad, dtype=np.float32).reshape(-1, 2)
    if len(quad) != 4:
        return quad
    out = quad.copy()
    axis = 0.5 * (quad[2] + quad[3]) - 0.5 * (quad[0] + quad[1])
    length = float(np.linalg.norm(axis))
    if length > 1e-6:
        axis /= length
        out[0] -= axis * max(0.0, float(top_extension))
        out[1] -= axis * max(0.0, float(top_extension))
        out[2] += axis * max(0.0, float(bottom_padding))
        out[3] += axis * max(0.0, float(bottom_padding))
    pad = max(0.0, float(side_padding))
    out[0, 0] -= pad
    out[3, 0] -= pad
    out[1, 0] += pad
    out[2, 0] += pad
    return out


def _point_in_box(point, box, margin=0.0):
    if point is None or box is None:
        return False
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = map(float, box)
    m = max(0.0, float(margin))
    return x1 - m <= x <= x2 + m and y1 - m <= y <= y2 + m


def _ball_overlaps_player(point, boxes, margin=8.0, player_points=None, foot_radius=0.0):
    """Veto obvious TrackNet responses inside a detected player body.

    On the supplied broadcast, the domain-mismatched checkpoint repeatedly
    fires on shirts/heads.  A true shuttle can be near a racket, but is rarely
    deep inside a person bounding box, so this is deliberately a soft,
    configurable veto rather than a hard global image mask.
    """
    if any(_point_in_box(point, box, margin) for box in (boxes or [])):
        return True
    if player_points and foot_radius > 0:
        return any(
            pt is not None and euclidean(point, pt) <= float(foot_radius)
            for pt in player_points
        )
    return False


def _near_pose_contact(point, contacts, tolerance=45.0):
    """Whether an event is close enough to a wrist/racket contact to keep it."""
    if point is None:
        return False
    limit = max(0.0, float(tolerance))
    return any(
        contact is not None
        and len(contact) >= 2
        and euclidean(point, contact) <= limit
        for contact in (contacts or ())
    )


def estimate_ball_center_shift(ball_dict, homography, court_quad, court_w_m, court_h_m):
    if len(ball_dict) == 0:
        return 0.0

    xs = []
    keys = sorted(ball_dict.keys())
    step = max(1, len(keys) // 2000)
    for i in keys[::step]:
        vis, bx, by = ball_dict[i][:3]
        if vis <= 0:
            continue
        img_pt = clamp_point_to_quad((bx, by), court_quad)
        m = to_court_m(img_pt, homography)
        if m is None:
            continue
        x = float(np.clip(m[0], 0.0, court_w_m))
        y = float(np.clip(m[1], 0.0, court_h_m))
        if 0.05 <= y <= (court_h_m - 0.05):
            xs.append(x)

    if len(xs) < 50:
        return 0.0

    center = court_w_m * 0.5
    median_x = float(np.median(np.array(xs)))
    shift = center - median_x
    max_shift = court_w_m * 0.25
    return float(np.clip(shift, -max_shift, max_shift))


def default_court_points(w, h):
    # Calibrated for common badminton broadcast perspective.
    return np.array(
        [
            [0.3667 * w, 0.4265 * h],  # TL
            [0.6385 * w, 0.4265 * h],  # TR
            [0.7490 * w, 0.9650 * h],  # BR
            [0.2540 * w, 0.9650 * h],  # BL
        ],
        dtype=np.float32,
    )


def select_four_points(frame):
    show = frame.copy()
    base = frame.copy()
    points = []

    def on_mouse(event, x, y, flags, param):
        del flags, param
        nonlocal show, points
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(show, (x, y), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                show,
                str(len(points)),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    win_name = "Click court corners: TL -> TR -> BR -> BL"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1400, 800)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        canvas = show.copy()
        tip = "Click TL -> TR -> BR -> BL | Enter=OK | r=Reset | q=Quit"
        cv2.putText(canvas, tip, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points = []
            show = base.copy()
        elif key == 13:
            if len(points) == 4:
                break
        elif key == ord("q") or key == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Canceled selecting court points.")

    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32)


def point_in_quad(point, quad):
    quad_int = quad.astype(np.int32)
    return cv2.pointPolygonTest(quad_int, (float(point[0]), float(point[1])), False) >= 0


def build_player_filter_quad(court_quad, top_inner_ratio=0.0):
    quad = court_quad.astype(np.float32).copy()
    if top_inner_ratio <= 0:
        return quad
    ratio = float(np.clip(top_inner_ratio, 0.0, 0.25))
    # Pull top edge inward to suppress audience/judges outside the far baseline.
    quad[0] = quad[0] + (quad[3] - quad[0]) * ratio
    quad[1] = quad[1] + (quad[2] - quad[1]) * ratio
    return quad


def build_detect_roi(court_quad, frame_w, frame_h, pad_x=28, pad_top=84, pad_bottom=18):
    xs = court_quad[:, 0]
    ys = court_quad[:, 1]
    x1 = int(max(0, math.floor(float(np.min(xs)) - pad_x)))
    x2 = int(min(frame_w, math.ceil(float(np.max(xs)) + pad_x)))
    y1 = int(max(0, math.floor(float(np.min(ys)) - pad_top)))
    y2 = int(min(frame_h, math.ceil(float(np.max(ys)) + pad_bottom)))
    if x2 <= x1:
        x1, x2 = 0, frame_w
    if y2 <= y1:
        y1, y2 = 0, frame_h
    return x1, y1, x2, y2


def estimate_foot_point(box_xyxy, keypoints=None, confs=None, ankle_min_conf=0.22, fallback_lift_ratio=0.0):
    x1, y1, x2, y2 = box_xyxy
    bh = max(1.0, float(y2 - y1))
    lift = float(np.clip(fallback_lift_ratio, 0.0, 0.28)) * bh
    fallback = (int((x1 + x2) / 2.0), int(y2 - lift))

    if keypoints is None or len(keypoints) < 17:
        return fallback

    ankle_pts = []
    for idx in (15, 16):
        if idx >= len(keypoints):
            continue
        if confs is not None and idx < len(confs) and float(confs[idx]) < ankle_min_conf:
            continue
        ankle_pts.append((float(keypoints[idx][0]), float(keypoints[idx][1])))

    if len(ankle_pts) == 0:
        return fallback
    if len(ankle_pts) == 1:
        ax, ay = ankle_pts[0]
    else:
        ax = 0.5 * (ankle_pts[0][0] + ankle_pts[1][0])
        ay = max(ankle_pts[0][1], ankle_pts[1][1])

    return (int(np.clip(ax, x1, x2)), int(np.clip(ay, y1, y2)))


def closest_point_on_segment(point, seg_a, seg_b):
    px, py = float(point[0]), float(point[1])
    ax, ay = float(seg_a[0]), float(seg_a[1])
    bx, by = float(seg_b[0]), float(seg_b[1])
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-9:
        return (ax, ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = float(np.clip(t, 0.0, 1.0))
    return (ax + t * abx, ay + t * aby)


def clamp_point_to_quad(point, quad):
    if point_in_quad(point, quad):
        return (float(point[0]), float(point[1]))
    best = None
    best_d = float("inf")
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        q = closest_point_on_segment(point, a, b)
        d = math.hypot(float(point[0]) - q[0], float(point[1]) - q[1])
        if d < best_d:
            best_d = d
            best = q
    return best if best is not None else (float(point[0]), float(point[1]))


def euclidean(p1, p2):
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def find_closest_candidate(candidates, ref_point, used_idx=None):
    if ref_point is None or len(candidates) == 0:
        return None

    best_i = None
    best_d = float("inf")
    for i, c in enumerate(candidates):
        if used_idx is not None and i in used_idx:
            continue
        d = euclidean(c["foot"], ref_point)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def update_track_memory(memory_dict, track_id, foot_xy, max_len=90):
    hist = memory_dict.get(track_id)
    if hist is None:
        hist = deque(maxlen=max_len)
        memory_dict[track_id] = hist
    hist.append((float(foot_xy[0]), float(foot_xy[1])))
    return hist


def is_static_top_track(history, top_y, min_frames=42, top_zone_px=18, top_ratio=0.88, static_move_px=30.0):
    if history is None or len(history) < min_frames:
        return False
    ys = [h[1] for h in history]
    top_hits = sum(1 for y in ys if y <= (top_y + top_zone_px))
    if (top_hits / float(len(history))) < top_ratio:
        return False

    xs = [h[0] for h in history]
    spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return spread <= static_move_px


def assign_players(
    candidates,
    prev_near_pt,
    prev_far_pt,
    prev_near_id=None,
    prev_far_id=None,
    far_lock_max_jump=110.0,
    far_max_up_jump=30.0,
    side_line_y=None,
    side_band_px=12.0,
):
    if len(candidates) == 0:
        return None, None

    candidates_sorted = sorted(candidates, key=lambda x: x["foot"][1])
    has_side_line = side_line_y is not None

    def pick_with_history(pool, prev_id, prev_pt, prefer_max_y):
        if len(pool) == 0:
            return None
        if prev_id is not None:
            by_id = next((c for c in pool if c["track_id"] == prev_id), None)
            if by_id is not None:
                return by_id
        if prev_pt is not None:
            idx = find_closest_candidate(pool, prev_pt)
            if idx is not None:
                return pool[idx]
        if prefer_max_y:
            return max(pool, key=lambda c: c["foot"][1])
        return min(pool, key=lambda c: c["foot"][1])

    if has_side_line:
        near_pool = [c for c in candidates_sorted if c["foot"][1] >= (side_line_y - side_band_px)]
        far_pool = [c for c in candidates_sorted if c["foot"][1] <= (side_line_y + side_band_px)]

        near_candidate = pick_with_history(near_pool, prev_near_id, prev_near_pt, prefer_max_y=True)
        far_candidate = pick_with_history(far_pool, prev_far_id, prev_far_pt, prefer_max_y=False)

        if near_candidate is not None and far_candidate is not None and near_candidate["track_id"] == far_candidate["track_id"]:
            # Prefer continuity on whichever side has stronger historical binding.
            keep_far = (prev_far_id is not None and far_candidate["track_id"] == prev_far_id)
            keep_near = (prev_near_id is not None and near_candidate["track_id"] == prev_near_id)
            if keep_far and not keep_near:
                remain = [c for c in near_pool if c["track_id"] != far_candidate["track_id"]]
                near_candidate = max(remain, key=lambda c: c["foot"][1]) if len(remain) > 0 else None
            elif keep_near and not keep_far:
                remain = [c for c in far_pool if c["track_id"] != near_candidate["track_id"]]
                far_candidate = min(remain, key=lambda c: c["foot"][1]) if len(remain) > 0 else None
            else:
                d_near = euclidean(near_candidate["foot"], prev_near_pt) if prev_near_pt is not None else float("inf")
                d_far = euclidean(far_candidate["foot"], prev_far_pt) if prev_far_pt is not None else float("inf")
                if d_far <= d_near:
                    remain = [c for c in near_pool if c["track_id"] != far_candidate["track_id"]]
                    near_candidate = max(remain, key=lambda c: c["foot"][1]) if len(remain) > 0 else None
                else:
                    remain = [c for c in far_pool if c["track_id"] != near_candidate["track_id"]]
                    far_candidate = min(remain, key=lambda c: c["foot"][1]) if len(remain) > 0 else None

        if far_candidate is not None and prev_far_pt is not None and (prev_far_id is None or far_candidate["track_id"] != prev_far_id):
            far_jump = euclidean(far_candidate["foot"], prev_far_pt)
            jumped_up_too_much = far_candidate["foot"][1] < (prev_far_pt[1] - far_max_up_jump)
            if jumped_up_too_much or far_jump > far_lock_max_jump:
                far_candidate = None

        return near_candidate, far_candidate

    if len(candidates_sorted) >= 2:
        near_candidate = candidates_sorted[-1]
        far_candidate = candidates_sorted[0]

        if prev_near_pt is not None and prev_far_pt is not None:
            best = None
            for i, near in enumerate(candidates_sorted):
                for j, far in enumerate(candidates_sorted):
                    if i == j:
                        continue
                    ny, fy = near["foot"][1], far["foot"][1]
                    sep = ny - fy
                    penalty = 0.0
                    if sep <= 0:
                        penalty += 1000.0
                    elif sep < 18:
                        penalty += (18 - sep) * 10.0
                    if prev_near_id is not None and near["track_id"] != prev_near_id:
                        penalty += 15.0
                    if prev_far_id is not None and far["track_id"] != prev_far_id:
                        penalty += 15.0
                    cost = euclidean(near["foot"], prev_near_pt) + euclidean(far["foot"], prev_far_pt) + penalty
                    if best is None or cost < best[0]:
                        best = (cost, near, far)
            if best is not None:
                near_candidate, far_candidate = best[1], best[2]

        ordered = sorted([near_candidate, far_candidate], key=lambda x: x["foot"][1])
        return ordered[-1], ordered[0]

    one = candidates_sorted[0]
    if prev_near_pt is None and prev_far_pt is None:
        return one, None
    if prev_near_pt is not None and prev_far_pt is not None:
        mid_y = 0.5 * (prev_near_pt[1] + prev_far_pt[1])
        if one["foot"][1] >= mid_y:
            return one, None
        return None, one
    if prev_near_pt is not None:
        return one, None
    return None, one


def catmull_rom_spline(points, steps=12):
    if len(points) < 2:
        return points[:]

    pts = [points[0]] + points[:] + [points[-1]]
    curve = []

    for i in range(1, len(pts) - 2):
        p0 = np.array(pts[i - 1], dtype=np.float32)
        p1 = np.array(pts[i], dtype=np.float32)
        p2 = np.array(pts[i + 1], dtype=np.float32)
        p3 = np.array(pts[i + 2], dtype=np.float32)

        for t in np.linspace(0, 1, steps, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            p = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            curve.append((int(p[0]), int(p[1])))

    curve.append(points[-1])
    return curve


def split_valid_segments(trail):
    segments = []
    current = []
    for p in trail:
        if p is None:
            if len(current) >= 2:
                segments.append(current)
            current = []
        else:
            current.append(p)
    if len(current) >= 2:
        segments.append(current)
    return segments


def draw_dashed_polyline(frame, points, color, thickness=3, dash_len=12, gap_len=10):
    if len(points) < 2:
        return

    seg_lens = [0.0]
    total = 0.0
    for i in range(1, len(points)):
        total += euclidean(points[i - 1], points[i])
        seg_lens.append(total)

    if total < 1.0:
        return

    cycle = dash_len + gap_len
    draw_from = 0.0

    while draw_from < total:
        draw_to = min(draw_from + dash_len, total)
        dash_points = []

        for i in range(1, len(points)):
            l1, l2 = seg_lens[i - 1], seg_lens[i]
            p1, p2 = points[i - 1], points[i]

            if l1 <= draw_from <= l2 and l2 > l1:
                r = (draw_from - l1) / (l2 - l1)
                dash_points.append((int(p1[0] + (p2[0] - p1[0]) * r), int(p1[1] + (p2[1] - p1[1]) * r)))

            if draw_from <= l2 <= draw_to:
                dash_points.append(p2)

            if l1 <= draw_to <= l2 and l2 > l1:
                r = (draw_to - l1) / (l2 - l1)
                dash_points.append((int(p1[0] + (p2[0] - p1[0]) * r), int(p1[1] + (p2[1] - p1[1]) * r)))
                break

        if len(dash_points) >= 2:
            cv2.polylines(frame, [np.array(dash_points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

        draw_from += cycle


def draw_trail(frame, trail, color, thickness=3, dash_len=12, gap_len=10, smooth_steps=12):
    for seg in split_valid_segments(list(trail)):
        if len(seg) < 2:
            continue
        smooth_seg = catmull_rom_spline(seg, steps=smooth_steps)
        draw_dashed_polyline(frame, smooth_seg, color, thickness, dash_len, gap_len)


def _split_ball_segments(trail):
    segments = []
    current = []
    for item in trail:
        if item is None:
            if len(current) >= 2:
                segments.append(current)
            current = []
            continue
        # item = (x, y, source, frame)
        if len(item) >= 4:
            current.append(item)
        else:
            current.append((item[0], item[1], "model", 0))
    if len(current) >= 2:
        segments.append(current)
    return segments


def draw_ball_trajectory(frame, trail, current=None, thickness=5):
    """Draw a neon, confidence-aware shuttle trail on the broadcast frame."""
    measurement_sources = {"model", "classical"}
    bridge_source = "offscreen_bridge"
    # Render-only top-edge hypotheses need to be visible on the purple
    # broadcast background, but must remain visually distinct from the bright
    # yellow measured-ball trail.  A cyan dashed stroke gives the viewer a
    # continuous cue without implying that every bridge sample came from the
    # detector.
    bridge_color = (255, 180, 40)  # BGR: cyan/blue, not measured yellow
    segments = _split_ball_segments(list(trail))
    if segments:
        glow = frame.copy()
        for segment in segments:
            for i in range(1, len(segment)):
                x0, y0, source0, _ = segment[i - 1]
                x1, y1, source1, _ = segment[i]
                alpha = (i + 1) / float(len(segment) + 1)
                is_bridge = source0 == bridge_source or source1 == bridge_source
                # A top-edge bridge is a render-only hypothesis.  Keep it
                # visibly weaker and dashed so it cannot be mistaken for a
                # measured shuttle location in the output video.
                if is_bridge:
                    frame0 = int(segment[i - 1][3])
                    frame1 = int(segment[i][3])
                    if ((frame0 + frame1) // 2) % 3 == 1:
                        continue
                    color = bridge_color
                    width = max(2, thickness - 1)
                elif source0 in measurement_sources and source1 in measurement_sources:
                    color = (0, int(120 + 135 * alpha), 255)
                    width = thickness
                else:
                    color = (int(80 + 80 * alpha), int(90 + 100 * alpha), 210)
                    width = max(1, thickness - 1)
                cv2.line(glow, (int(x0), int(y0)), (int(x1), int(y1)), color, width + (4 if is_bridge else 6), cv2.LINE_AA)
        cv2.addWeighted(glow, 0.28, frame, 0.72, 0, frame)
        for segment in segments:
            for i in range(1, len(segment)):
                x0, y0, source0, _ = segment[i - 1]
                x1, y1, source1, _ = segment[i]
                alpha = (i + 1) / float(len(segment) + 1)
                is_bridge = source0 == bridge_source or source1 == bridge_source
                if is_bridge:
                    frame0 = int(segment[i - 1][3])
                    frame1 = int(segment[i][3])
                    if ((frame0 + frame1) // 2) % 3 == 1:
                        continue
                    color = bridge_color
                    width = max(2, thickness - 1)
                else:
                    color = (
                        (0, int(120 + 135 * alpha), 255)
                        if source0 in measurement_sources and source1 in measurement_sources
                        else (80, 150, 220)
                    )
                    width = max(1, thickness - 1)
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), color, width, cv2.LINE_AA)

    if current is not None:
        x, y = int(round(current.x)), int(round(current.y))
        if str(getattr(current, "source", "")) == bridge_source:
            # Do not draw the bright measured-ball dot for an offscreen
            # hypothesis; a small muted ring is enough to show where the
            # dashed bridge currently passes through the frame.
            cv2.circle(frame, (x, y), 6, bridge_color, 2, cv2.LINE_AA)
            return
        # Glow + bright center makes the tiny shuttle visible on purple/green
        # broadcast backgrounds.
        glow = frame.copy()
        cv2.circle(glow, (x, y), 18, (0, 220, 255), 8, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.35, frame, 0.65, 0, frame)
        cv2.circle(frame, (x, y), 8, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_hit_marker(
    frame,
    event,
    frame_idx,
    hit_number=0,
    duration=15,
    uncertain=False,
    completed_shot_speed_kmh=None,
    completed_shot_method="",
):
    """Render a flashing hit ring; uncertain candidates are labeled ``HIT?``."""
    age = int(frame_idx - event.frame)
    if age < 0 or age > max(1, duration):
        return
    fade = max(0.0, 1.0 - age / float(max(1, duration)))
    pulse = 1.0 + 0.25 * math.sin(age * math.pi / 2.0)
    x, y = int(round(event.x)), int(round(event.y))
    radius = int((20 + age * 2) * pulse)
    glow = frame.copy()
    color = (0, 165, 255) if uncertain else (0, 180, 255)
    cv2.circle(glow, (x, y), radius + 8, color, 8, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.20 * fade, frame, 1.0 - 0.20 * fade, 0, frame)
    ring_color = (0, 190, 255) if uncertain else (0, 255, 255)
    cv2.circle(frame, (x, y), radius, ring_color, 3, cv2.LINE_AA)
    cv2.circle(frame, (x, y), max(5, radius // 3), (0, 0, 255), 2, cv2.LINE_AA)
    if uncertain:
        label = "HIT?" if hit_number <= 0 else f"HIT #{hit_number} ?"
    else:
        label = "HIT" if hit_number <= 0 else f"HIT #{hit_number}"
    label_x = x + radius + 8
    label_y = max(28, y - radius - 5)
    cv2.putText(frame, label, (label_x, label_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, ring_color, 2, cv2.LINE_AA)
    # The just-completed flight is known only when this new contact arrives.
    # Keep the marker label ASCII so it remains legible on systems without a
    # CJK OpenCV font; the stats panel below uses the normal Chinese renderer.
    if completed_shot_speed_kmh is not None and math.isfinite(float(completed_shot_speed_kmh)):
        speed_prefix = (
            "PREV EST" if str(completed_shot_method) == "perspective_proxy"
            else "PREV AVG"
        )
        cv2.putText(
            frame,
            f"{speed_prefix} {float(completed_shot_speed_kmh):.0f} km/h",
            (label_x, min(frame.shape[0] - 8, label_y + 28)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.52,
            (0, 235, 255),
            1,
            cv2.LINE_AA,
        )


def draw_ball_map_trail(frame, points, color=(0, 220, 255), thickness=2):
    """Draw a fading continuous trail on the mini court."""
    pts = [p for p in points if p is not None]
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        alpha = i / float(len(pts))
        c = tuple(int(v * (0.20 + 0.80 * alpha)) for v in color)
        cv2.line(frame, pts[i - 1], pts[i], c, max(1, int(thickness * (0.55 + alpha))), cv2.LINE_AA)


def make_player_heatmap(court_w_m, court_h_m, cell_m=0.20):
    """Create one persistent Mini Court occupancy grid in court metres."""
    cell = max(0.05, float(cell_m))
    grid_w = max(2, int(math.ceil(float(court_w_m) / cell)))
    grid_h = max(2, int(math.ceil(float(court_h_m) / cell)))
    return np.zeros((grid_h, grid_w), dtype=np.float32)


def add_player_heatmap_sample(heatmap, point_m, court_w_m, court_h_m, weight=1.0):
    """Accumulate one real player foot observation without synthesizing motion."""
    if heatmap is None or point_m is None:
        return False
    try:
        x, y = float(point_m[0]), float(point_m[1])
        grid_h, grid_w = heatmap.shape[:2]
    except (TypeError, ValueError, IndexError):
        return False
    if grid_h <= 0 or grid_w <= 0 or not np.isfinite([x, y]).all():
        return False
    width = max(float(court_w_m), 1e-6)
    height = max(float(court_h_m), 1e-6)
    # Coordinates have already passed player/court filtering.  The clip here
    # only handles tiny numerical overshoots at a baseline or side line.
    col = int(np.clip(math.floor(x / width * grid_w), 0, grid_w - 1))
    row = int(np.clip(math.floor(y / height * grid_h), 0, grid_h - 1))
    heatmap[row, col] += max(0.0, float(weight))
    return True


def draw_player_heatmap(
    frame,
    heatmap,
    top_left,
    size,
    color,
    alpha=0.34,
    sigma_cells=1.3,
):
    """Blend one colourized occupancy map into the Mini Court ROI."""
    if heatmap is None or heatmap.size == 0 or float(alpha) <= 0.0:
        return
    x0, y0 = map(int, top_left)
    width, height = map(int, size)
    if width <= 1 or height <= 1:
        return
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(frame_w, x0 + width + 1)
    y2 = min(frame_h, y0 + height + 1)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    heat = cv2.resize(
        heatmap.astype(np.float32),
        (roi.shape[1], roi.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    if float(sigma_cells) > 0.0:
        grid_h, grid_w = heatmap.shape[:2]
        sigma_x = max(0.1, float(sigma_cells) * roi.shape[1] / max(1, grid_w))
        sigma_y = max(0.1, float(sigma_cells) * roi.shape[0] / max(1, grid_h))
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    positive = heat[heat > 1e-6]
    if positive.size == 0:
        return
    scale = max(1e-6, float(np.percentile(positive, 95)))
    strength = np.clip(heat / scale, 0.0, 1.0) ** 0.70
    alpha_map = np.clip(float(alpha), 0.0, 0.85) * strength
    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    blended = (
        roi.astype(np.float32) * (1.0 - alpha_map[..., None])
        + tint * alpha_map[..., None]
    )
    roi[:] = np.clip(blended, 0, 255).astype(np.uint8)


def draw_alpha_rect(frame, top_left, bottom_right, color, alpha):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def get_zh_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            _FONT_CACHE[size] = ImageFont.truetype(path, size=size)
            return _FONT_CACHE[size]
    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def draw_text_cn(frame, text_items):
    if len(text_items) == 0:
        return

    h, w = frame.shape[:2]
    x1, y1 = w, h
    x2, y2 = 0, 0
    font_items = []
    for text, xy, size, bgr in text_items:
        font = get_zh_font(size)
        bbox = font.getbbox(text)
        tw = max(1, int(bbox[2] - bbox[0]))
        th = max(1, int(bbox[3] - bbox[1]))
        x, y = int(xy[0]), int(xy[1])
        x1 = min(x1, x)
        y1 = min(y1, y)
        x2 = max(x2, x + tw + 2)
        y2 = max(y2, y + th + 2)
        font_items.append((text, x, y, font, bgr))

    if x2 <= x1 or y2 <= y1:
        return

    pad = 6
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    roi = frame[y1:y2, x1:x2]
    pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for text, x, y, font, bgr in font_items:
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        draw.text((x - x1, y - y1), text, font=font, fill=rgb)
    roi[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


class MotionStats:
    def __init__(self):
        self.prev_m = None
        self.prev_frame_idx = None
        self.current_speed = 0.0
        self.current_speed_buffer = deque(maxlen=6)

        self.rally_distance = 0.0
        self.rally_time = 0.0
        self.rally_max_speed = 0.0
        self.rally_hits = 0

        self.total_distance = 0.0
        self.total_time = 0.0
        self.total_max_speed = 0.0

    @property
    def rally_avg_speed(self):
        return 0.0 if self.rally_time <= 1e-6 else self.rally_distance / self.rally_time

    @property
    def total_avg_speed(self):
        return 0.0 if self.total_time <= 1e-6 else self.total_distance / self.total_time

    def reset_rally(self):
        self.rally_distance = 0.0
        self.rally_time = 0.0
        self.rally_max_speed = 0.0
        self.rally_hits = 0

    def update(self, pt_m, frame_idx, fps):
        if pt_m is None:
            self.current_speed_buffer.append(0.0)
            self.current_speed = float(np.mean(self.current_speed_buffer))
            return

        inst_speed = 0.0
        if self.prev_m is not None and self.prev_frame_idx is not None and frame_idx > self.prev_frame_idx:
            dt = (frame_idx - self.prev_frame_idx) / max(fps, 1e-6)
            distance = euclidean(pt_m, self.prev_m)
            # Reject unrealistic jump caused by tracker ID switches.
            # Badminton players sprint at ~7 m/s; cap a touch above that.
            max_jump_m = 8.0 * dt + 0.05
            if distance <= max_jump_m:
                inst_speed = distance / max(dt, 1e-6)
                self.rally_distance += distance
                self.rally_time += dt
                self.rally_max_speed = max(self.rally_max_speed, inst_speed)
                self.total_distance += distance
                self.total_time += dt
                self.total_max_speed = max(self.total_max_speed, inst_speed)

        self.current_speed_buffer.append(inst_speed)
        self.current_speed = float(np.mean(self.current_speed_buffer))
        self.prev_m = pt_m
        self.prev_frame_idx = frame_idx


def to_court_m(point_xy, homography):
    if point_xy is None:
        return None
    src = np.array([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)[0][0]
    return (float(dst[0]), float(dst[1]))


def to_map_px(point_m, map_x, map_y, map_w, map_h, court_w_m, court_h_m):
    if point_m is None:
        return None
    x_norm = float(np.clip(point_m[0] / max(court_w_m, 1e-6), 0.0, 1.0))
    y_norm = float(np.clip(point_m[1] / max(court_h_m, 1e-6), 0.0, 1.0))
    return (int(map_x + x_norm * map_w), int(map_y + y_norm * map_h))


def enforce_half_court(point_m, side, court_w_m, court_h_m, far_half_expand=1.0):
    if point_m is None:
        return None
    x, y = float(point_m[0]), float(point_m[1])
    x = float(np.clip(x, 0.0, court_w_m))
    y = float(np.clip(y, 0.0, court_h_m))
    mid = court_h_m * 0.5
    if side == "far":
        # Fold lower-half leakage back to upper half to keep far player in top court,
        # while preserving vertical motion instead of flattening to the midline.
        y = float(mid - abs(mid - y))
        # Expand far-side activity range on mini court so it matches near-side visual span.
        scale = max(1.0, float(far_half_expand))
        y = float(np.clip(mid - (mid - y) * scale, 0.0, mid - 1e-3))
    if side == "near" and y < mid:
        y = float(mid + 1e-3)
    return (x, y)


def estimate_point_by_optical_flow(prev_gray, cur_gray, prev_pt, max_move=40.0):
    if prev_gray is None or cur_gray is None or prev_pt is None:
        return None

    p0 = np.array([[[float(prev_pt[0]), float(prev_pt[1])]]], dtype=np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        cur_gray,
        p0,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if p1 is None or st is None or st[0][0] != 1:
        return None

    nx, ny = float(p1[0][0][0]), float(p1[0][0][1])
    if math.hypot(nx - prev_pt[0], ny - prev_pt[1]) > max_move:
        return None
    return (int(nx), int(ny))


def draw_stats_panel(
    frame,
    rally_idx,
    far_stats,
    near_stats,
    pipeline_fps=0.0,
    rally_hit_count=0,
    total_hit_count=0,
    last_shot_speed_kmh=None,
    last_shot_method="",
):
    h, w = frame.shape[:2]
    x0, y0 = 12, 12
    panel_w = max(340, int(w * 0.33))

    min_line_h, max_line_h = 18, 24
    header_h = 158
    block_gap = 18
    line_h = int(np.clip((h - 80) / 22.0, min_line_h, max_line_h))
    block_h = 28 + 4 * line_h
    panel_h = header_h + block_h * 2 + block_gap + 18
    panel_h = int(np.clip(panel_h, 310, h - 18))
    draw_alpha_rect(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), 0.54)

    block1_y = y0 + header_h
    block2_y = block1_y + block_h + block_gap

    speed_text = "--"
    if last_shot_speed_kmh is not None:
        try:
            speed_value = float(last_shot_speed_kmh)
            if math.isfinite(speed_value):
                prefix = "~" if str(last_shot_method) == "perspective_proxy" else ""
                speed_text = f"{prefix}{speed_value:.0f} km/h"
        except (TypeError, ValueError):
            pass
    text_items = [
        (ZH_TITLE, (x0 + 12, y0 + 6), 30, (0, 210, 255)),
        (f"{ZH_RALLY}: {rally_idx}", (x0 + 12, y0 + 38), 26, (0, 175, 255)),
        (f"\u672c\u56de\u5408\u68c0\u6d4b\u51fb\u7403: {int(rally_hit_count)}", (x0 + 12, y0 + 64), 22, (245, 245, 245)),
        (f"\u89c6\u9891\u7d2f\u8ba1\u51fb\u7403: {int(total_hit_count)}", (x0 + 12, y0 + 88), 22, (245, 245, 245)),
        (f"\u6700\u8fd1\u6709\u6548\u7403\u901f: {speed_text}", (x0 + 12, y0 + 112), 22, (0, 225, 255)),
        (f"FPS: {pipeline_fps:.2f}", (x0 + 12, y0 + 138), 20, (190, 235, 255)),
    ]

    def add_block(y, title, stats, color):
        text_items.append((title, (x0 + 12, y), 26, color))
        rows = [
            f"\u5f53\u524d\u901f\u5ea6: {stats.current_speed:.2f} m/s",
            f"\u56de\u5408\u8ddd\u79bb: {stats.rally_distance:.2f} m",
            f"\u56de\u5408\u6700\u9ad8: {stats.rally_max_speed:.2f} m/s",
            f"\u603b\u8ddd\u79bb: {stats.total_distance:.2f} m",
        ]
        for i, row in enumerate(rows):
            text_items.append((row, (x0 + 12, y + 28 + i * line_h), 22, (240, 240, 240)))

    add_block(block1_y, ZH_TOP, far_stats, (0, 175, 255))
    add_block(block2_y, ZH_BOTTOM, near_stats, (255, 120, 220))
    draw_text_cn(frame, text_items)


def draw_mini_court(
    frame,
    far_m,
    near_m,
    ball_m,
    far_map_hist,
    near_map_hist,
    ball_map_hist,
    court_w_m,
    court_h_m,
    recent_hits=None,
    frame_idx=0,
    ball_source="model",
    far_heatmap=None,
    near_heatmap=None,
    draw_heatmap=True,
    heatmap_alpha=0.34,
    heatmap_sigma_cells=1.3,
    far_color=(0, 180, 255),
    near_color=(255, 110, 210),
):
    h, w = frame.shape[:2]
    panel_w = max(200, int(w * 0.18))
    panel_h = max(320, int(h * 0.62))
    px0, py0 = w - panel_w - 12, 12
    draw_alpha_rect(frame, (px0, py0), (px0 + panel_w, py0 + panel_h), (10, 16, 28), 0.64)
    cv2.putText(frame, "Mini Court", (px0 + 12, py0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)

    pad = 14
    cx0, cy0 = px0 + pad, py0 + 42
    cw, ch = panel_w - pad * 2, panel_h - 52
    if draw_heatmap:
        # Heat maps are layered below all court markings, trails, and current
        # player positions.  Each side is normalised independently so a
        # player with less movement remains visible.
        draw_player_heatmap(
            frame,
            far_heatmap,
            (cx0, cy0),
            (cw, ch),
            far_color,
            alpha=heatmap_alpha,
            sigma_cells=heatmap_sigma_cells,
        )
        draw_player_heatmap(
            frame,
            near_heatmap,
            (cx0, cy0),
            (cw, ch),
            near_color,
            alpha=heatmap_alpha,
            sigma_cells=heatmap_sigma_cells,
        )

    cv2.rectangle(frame, (cx0, cy0), (cx0 + cw, cy0 + ch), (70, 90, 130), 1, cv2.LINE_AA)

    # Main lines
    cv2.line(frame, (cx0 + cw // 2, cy0), (cx0 + cw // 2, cy0 + ch), (70, 90, 130), 1, cv2.LINE_AA)
    cv2.line(frame, (cx0, cy0 + ch // 2), (cx0 + cw, cy0 + ch // 2), (70, 90, 130), 1, cv2.LINE_AA)

    # Service lines (approx badminton doubles court markings)
    short_service = 1.98
    long_service = 0.76
    top_short_y = int(cy0 + (short_service / court_h_m) * ch)
    bot_short_y = int(cy0 + ((court_h_m - short_service) / court_h_m) * ch)
    top_long_y = int(cy0 + (long_service / court_h_m) * ch)
    bot_long_y = int(cy0 + ((court_h_m - long_service) / court_h_m) * ch)
    cv2.line(frame, (cx0, top_short_y), (cx0 + cw, top_short_y), (55, 75, 115), 1, cv2.LINE_AA)
    cv2.line(frame, (cx0, bot_short_y), (cx0 + cw, bot_short_y), (55, 75, 115), 1, cv2.LINE_AA)
    cv2.line(frame, (cx0, top_long_y), (cx0 + cw, top_long_y), (45, 65, 105), 1, cv2.LINE_AA)
    cv2.line(frame, (cx0, bot_long_y), (cx0 + cw, bot_long_y), (45, 65, 105), 1, cv2.LINE_AA)

    far_map = to_map_px(far_m, cx0, cy0, cw, ch, court_w_m, court_h_m)
    near_map = to_map_px(near_m, cx0, cy0, cw, ch, court_w_m, court_h_m)
    ball_map = to_map_px(ball_m, cx0, cy0, cw, ch, court_w_m, court_h_m)

    if far_map is not None:
        far_map_hist.append(far_map)
    if near_map is not None:
        near_map_hist.append(near_map)
    if ball_map is not None:
        ball_map_hist.append(ball_map)

    draw_ball_map_trail(frame, list(far_map_hist), (0, 180, 255), 2)
    draw_ball_map_trail(frame, list(near_map_hist), (255, 110, 210), 2)
    # A render-only top-edge bridge is deliberately cyan on the mini court.
    # It is not a measured shuttle point, so it remains distinct from the
    # solid yellow measured trail and hit rings while staying visible during a
    # long out-of-frame flight.
    bridge_ball = str(ball_source) == "offscreen_bridge"
    draw_ball_map_trail(
        frame,
        list(ball_map_hist),
        (255, 180, 40) if bridge_ball else (0, 245, 255),
        3 if bridge_ball else 4,
    )

    if far_map is not None:
        cv2.circle(frame, far_map, 5, (0, 210, 255), -1, cv2.LINE_AA)
    if near_map is not None:
        cv2.circle(frame, near_map, 5, (255, 140, 220), -1, cv2.LINE_AA)
    if ball_map is not None and bridge_ball:
        cv2.circle(frame, ball_map, 5, (255, 180, 40), 2, cv2.LINE_AA)
    elif ball_map is not None:
        glow = frame.copy()
        cv2.circle(glow, ball_map, 12, (0, 210, 255), 5, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.30, frame, 0.70, 0, frame)
        cv2.circle(frame, ball_map, 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, ball_map, 3, (255, 255, 255), -1, cv2.LINE_AA)

    # Keep a short history of hit rings on the mini court.  The event's map
    # coordinate is attached during run() after homography is known.
    for event in recent_hits or []:
        if event.map_xy is None:
            continue
        age = int(frame_idx - event.frame)
        if age < 0 or age > 18:
            continue
        fade = max(0.0, 1.0 - age / 18.0)
        radius = int(7 + age * 0.8)
        ring_color = (0, 190, 255) if getattr(event, "uncertain", False) else (0, 255, 255)
        cv2.circle(frame, event.map_xy, radius, ring_color, 2, cv2.LINE_AA)
        cv2.circle(frame, event.map_xy, max(3, radius // 2), (0, 80, 255), 1, cv2.LINE_AA)


def draw_skeleton(frame, keypoints, confs, color, min_conf=0.3):
    if keypoints is None:
        return

    # COCO-17 topology used by YOLO pose models.
    edges = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]

    def visible(i):
        if keypoints is None or i >= len(keypoints):
            return False
        if confs is None or i >= len(confs):
            return True
        return float(confs[i]) >= min_conf

    for i, j in edges:
        if visible(i) and visible(j):
            p1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            p2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

    for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        if visible(i):
            p = (int(keypoints[i][0]), int(keypoints[i][1]))
            cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)


def shift_player_for_draw(player, target_foot):
    if player is None or target_foot is None:
        return player
    src_foot = player.get("foot")
    if src_foot is None:
        return player
    dx = int(target_foot[0] - src_foot[0])
    dy = int(target_foot[1] - src_foot[1])
    if dx == 0 and dy == 0:
        return player

    out = dict(player)
    x1, y1, x2, y2 = out["box"]
    out["box"] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
    out["foot"] = (int(target_foot[0]), int(target_foot[1]))
    if "kpts" in out and out["kpts"] is not None:
        out["kpts"] = out["kpts"] + np.array([dx, dy], dtype=np.float32)
    return out


def draw_player(frame, player, label, color, draw_foot_point=True, draw_id=False, draw_pose=True):
    x1, y1, x2, y2 = map(int, player["box"])
    fx, fy = player["foot"]
    tid = player["track_id"]
    keypoints = player.get("kpts")
    kconf = player.get("kconf")

    if draw_pose and keypoints is not None:
        draw_skeleton(frame, keypoints, kconf, color)

    # Keep labels lightweight for long videos.
    if draw_id:
        text_x = max(8, x1)
        text_y = max(26, y1 - 8)
        cv2.putText(frame, f"id:{tid}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    if draw_foot_point:
        cv2.circle(frame, (fx, fy), 5, color, -1, cv2.LINE_AA)


def run(args):
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 30

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read first frame.")

    court_quad = parse_court_points(args.court_points)
    if court_quad is None:
        if args.select_court_points:
            court_quad = select_four_points(first_frame)
        else:
            raise ValueError(ZH_ERR_COURT)
    player_filter_quad = build_player_filter_quad(court_quad, args.top_inner_ratio)
    top_y = float(0.5 * (court_quad[0][1] + court_quad[1][1]))
    bot_y = float(0.5 * (court_quad[2][1] + court_quad[3][1]))
    midline_y_img = 0.5 * (top_y + bot_y)

    if args.detect_on_court_roi:
        detect_x1, detect_y1, detect_x2, detect_y2 = build_detect_roi(
            court_quad,
            w,
            h,
            pad_x=args.detect_roi_pad_x,
            pad_top=args.detect_roi_pad_top,
            pad_bottom=args.detect_roi_pad_bottom,
        )
    else:
        detect_x1, detect_y1, detect_x2, detect_y2 = 0, 0, w, h
    detect_area_ratio = ((detect_x2 - detect_x1) * (detect_y2 - detect_y1)) / max(1.0, float(w * h))

    dst = np.array(
        [[0.0, 0.0], [args.court_width_m, 0.0], [args.court_width_m, args.court_length_m], [0.0, args.court_length_m]],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(court_quad, dst)
    if homography is None:
        cap.release()
        raise RuntimeError("Failed to compute court homography.")

    cap.release()
    cap = cv2.VideoCapture(args.video_path)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    writer = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output writer: {args.output_path}")

    model = None if args.skip_player_detector else YOLO(args.yolo_model)

    near_trail = deque(maxlen=args.trail_len)
    far_trail = deque(maxlen=args.trail_len)
    near_map_hist = deque(maxlen=args.minimap_trail_len)
    far_map_hist = deque(maxlen=args.minimap_trail_len)
    ball_map_hist = deque(maxlen=args.ball_trail_len)
    ball_trail = deque(maxlen=args.ball_trail_len)
    # These grids intentionally receive only real YOLO foot detections.  The
    # visible player trails may use optical-flow/hold positions, which are
    # useful for continuity but would create false stationary heat spots.
    near_heatmap = make_player_heatmap(
        args.court_width_m, args.court_length_m, args.heatmap_cell_m
    )
    far_heatmap = make_player_heatmap(
        args.court_width_m, args.court_length_m, args.heatmap_cell_m
    )
    last_near_heatmap_frame = None
    last_far_heatmap_frame = None

    # Ball post-processing is performed once before rendering.  This keeps
    # Kalman state independent of YOLO detector cadence and lets hit/rally
    # events be rendered consistently on the main frame and mini court.
    ball_csv = args.ball_csv if args.ball_csv else infer_ball_csv(args.video_path)
    ball_csv_secondary = str(args.ball_csv_secondary or "").strip()
    tracknet_ball_primary = load_ball_dict(ball_csv)
    tracknet_ball_secondary = load_ball_dict(ball_csv_secondary)
    tracknet_ball_dict = tracknet_ball_primary
    if args.scene_cuts.strip():
        scene_cuts = []
        for token in args.scene_cuts.split(","):
            try:
                scene_cuts.append(int(token.strip()))
            except ValueError:
                continue
        scene_cuts = sorted(set(scene_cuts))
        decoded_frame_count = 0
    else:
        scene_cuts, decoded_frame_count = detect_scene_cuts(
            args.video_path,
            threshold=args.scene_cut_threshold,
            return_frame_count=True,
        )
    # Some broadcast MP4 containers advertise a stale frame count even though
    # the decoder stops earlier (the repository's ``output.mp4`` reports 1052
    # but yields 1001 frames).  The analysis path already uses the decodable
    # count below; keep the render diagnostics/progress denominator aligned as
    # well so a truncated tail is not presented as a ball-detection gap.
    if int(decoded_frame_count or 0) > 0:
        total_frames = int(decoded_frame_count)
    # A hard cut is a state boundary, not a seven-frame blind zone.  Reset on
    # the first frame of the new shot and allow that frame to initialize a new
    # state when processing beyond the cut is enabled.
    scene_reset_frames = [int(cut) for cut in scene_cuts]
    csv_frame_count = max(
        max(tracknet_ball_primary.keys(), default=-1),
        max(tracknet_ball_secondary.keys(), default=-1),
    ) + 1
    track_frame_count = int(decoded_frame_count or (csv_frame_count if csv_frame_count > 0 else total_frames))
    phase_fusion_stats = {}
    phase_suspicious_frames = set()
    phase_player_vetoed_frames = []
    phase_branch_veto_stats = {}
    top_branch_veto_frames = []
    top_entry_lead_frames = []
    top_interstitial_candidates = []
    top_interstitial_bridges = {}
    top_interstitial_stats = {}
    top_interstitial_veto_frames = []
    top_interstitial_veto_stats = {}
    dual_motion_protected_frames = []
    dual_motion_protection_stats = {}
    phase_protected_frames = []
    phase_isolated_veto_frames = []
    phase_isolated_veto_stats = {}
    early_player_veto_frames = []
    if tracknet_ball_secondary:
        (
            tracknet_ball_dict,
            phase_suspicious_frames,
            phase_fusion_stats,
        ) = merge_tracknet_phase_observations(
            tracknet_ball_primary,
            tracknet_ball_secondary,
            track_frame_count,
            agreement_px=args.tracknet_phase_agreement_px,
            bridge_search_frames=args.tracknet_phase_bridge_frames,
            bridge_residual_px=args.tracknet_phase_bridge_residual_px,
            scene_cuts=scene_cuts,
            edge_y_px=args.tracknet_phase_edge_protect_y,
            edge_lead_frames=3,
            single_edge_max_frames=2,
        )
        # Protect the measured lead-in immediately before a dual-phase
        # ``model_edge`` anchor.  This is intentionally computed from the
        # fused observations before any pose/body veto; edge/lead sources are
        # excluded from hit evidence downstream.
        top_entry_lead_frames = find_top_entry_lead_frames(
            tracknet_ball_dict,
            track_frame_count,
            scene_cuts=scene_cuts,
            top_y=args.tracknet_phase_edge_protect_y,
            max_lead_frames=8,
            min_lead_points=3,
        )
        (
            top_interstitial_candidates,
            top_interstitial_bridges,
            top_interstitial_stats,
        ) = find_top_edge_interstitial_branch_frames(
            tracknet_ball_dict,
            track_frame_count,
            scene_cuts=scene_cuts,
            edge_top_y=args.tracknet_phase_top_interstitial_edge_y,
            edge_core_y=args.tracknet_phase_edge_protect_y,
            min_anchor_frames=args.tracknet_phase_top_interstitial_anchor_frames,
            max_interstitial_gap=args.tracknet_phase_top_interstitial_gap_frames,
            min_branch_residual_px=args.tracknet_phase_top_interstitial_residual_px,
        )
        (
            dual_motion_protected_frames,
            dual_motion_protection_stats,
        ) = find_dual_phase_coherent_motion_frames(
            tracknet_ball_primary,
            tracknet_ball_secondary,
            tracknet_ball_dict,
            phase_suspicious_frames,
            track_frame_count,
            scene_cuts=scene_cuts,
            excluded_frames=top_interstitial_candidates,
            agreement_px=args.tracknet_phase_agreement_px,
            support_frames=2,
            min_run_frames=2,
            max_run_frames=8,
            min_median_step_px=12.0,
            min_path_px=80.0,
            min_spread_px=35.0,
            min_support_displacement_px=10.0,
            max_phase_velocity_delta_px=18.0,
            edge_y=args.tracknet_phase_edge_protect_y,
        )
        phase_protected_frames = sorted(set(
            top_entry_lead_frames + dual_motion_protected_frames
        ))
        phase_pose_evidence = {}
        if args.tracknet_phase_player_veto and model is not None:
            # A phase can agree on the same body response immediately after a
            # genuine top-edge crossing.  Mark that short branch separately so
            # it receives a wider body-box veto, while preserving the genuine
            # edge anchor itself.
            top_branch_candidates = find_top_exit_branch_frames(
                tracknet_ball_dict,
                track_frame_count,
                scene_cuts=scene_cuts,
                top_y=args.tracknet_phase_edge_protect_y,
                body_y=args.tracknet_top_branch_body_y,
                max_branch_frames=args.tracknet_top_branch_max_frames,
            ) if args.tracknet_top_branch_veto else []
            # Include the short clip lead-in as well.  The first TrackNet
            # response in this video is on the near player's leg/racket; it
            # is otherwise outside the sparse phase-boundary set and would
            # remain as a conspicuous false ball marker.
            early_candidates = [
                frame for frame in range(min(track_frame_count, 20))
                if (
                    _normalized_ball_observation(tracknet_ball_dict.get(frame))
                    is not None
                    and str(_normalized_ball_observation(tracknet_ball_dict.get(frame))[3]) == "model"
                )
            ]
            pose_frames = sorted(
                set(phase_suspicious_frames)
                | set(top_branch_candidates)
                | set(top_interstitial_candidates)
                | set(early_candidates)
            )
            phase_pose_evidence = _infer_audio_pose_contacts(
                model,
                args.video_path,
                pose_frames,
                court_quad,
                midline_y_img,
                args,
                pose_imgsz=args.tracknet_phase_pose_imgsz,
                relaxed_player_filter=True,
            )
            (
                tracknet_ball_dict,
                phase_branch_vetoed_frames,
                phase_branch_veto_stats,
            ) = veto_short_unsupported_phase_branches(
                tracknet_ball_dict,
                phase_pose_evidence,
                phase_suspicious_frames,
                scene_cuts=scene_cuts,
                max_branch_frames=args.tracknet_phase_max_branch_frames,
                contact_protect_px=args.tracknet_phase_contact_protect_px,
                edge_protect_y=args.tracknet_phase_edge_protect_y,
                protected_frames=phase_protected_frames,
            )
            (
                tracknet_ball_dict,
                phase_player_vetoed_frames,
            ) = veto_model_observations_in_player_boxes(
                tracknet_ball_dict,
                phase_pose_evidence,
                # Do not apply a broad body box to every segment boundary:
                # real post-contact flight points (for example f514) can be
                # inside a player's detector box while still following a
                # coherent 5+ frame trajectory.  Short branch classification
                # above is the high-precision scope for this hard veto.
                phase_branch_vetoed_frames,
                box_margin_px=args.tracknet_phase_player_box_margin,
                contact_protect_px=args.tracknet_phase_contact_protect_px,
                edge_protect_y=args.tracknet_phase_edge_protect_y,
                protected_frames=phase_protected_frames,
            )
            # The top-exit candidates are intentionally a separate pass: use
            # a larger margin to catch points just above a player's head while
            # keeping the wrist/contact protection active.
            tracknet_ball_dict, top_branch_veto_frames = veto_model_observations_in_player_boxes(
                tracknet_ball_dict,
                phase_pose_evidence,
                top_branch_candidates,
                box_margin_px=args.tracknet_top_branch_player_box_margin,
                contact_protect_px=args.tracknet_phase_contact_protect_px,
                edge_protect_y=args.tracknet_phase_edge_protect_y,
                protected_frames=phase_protected_frames,
            )
            tracknet_ball_dict, top_branch_contact_veto_frames = veto_top_exit_branch_points(
                tracknet_ball_dict,
                phase_pose_evidence,
                top_branch_candidates,
                contact_protect_px=args.tracknet_phase_contact_protect_px,
                protected_frames=phase_protected_frames,
            )
            tracknet_ball_dict, early_player_veto_frames = veto_model_observations_in_player_boxes(
                tracknet_ball_dict,
                phase_pose_evidence,
                early_candidates,
                box_margin_px=args.tracknet_phase_player_box_margin,
                contact_protect_px=args.tracknet_phase_contact_protect_px,
                edge_protect_y=args.tracknet_phase_edge_protect_y,
                protected_frames=phase_protected_frames,
            )
            top_branch_veto_frames = sorted(set(
                top_branch_veto_frames + top_branch_contact_veto_frames
            ))
            phase_player_vetoed_frames = sorted(set(
                phase_player_vetoed_frames
                + phase_branch_vetoed_frames
                + top_branch_veto_frames
                + early_player_veto_frames
            ) - set(phase_protected_frames))
        # This branch test is independent of the optional pose model.  When
        # pose is available its wrist/contact evidence protects a genuine
        # racket point; without pose, the strict two-sided edge bridge still
        # removes only the high-residual interior response.
        (
            tracknet_ball_dict,
            top_interstitial_veto_frames,
            top_interstitial_veto_stats,
        ) = veto_top_edge_interstitial_branches(
            tracknet_ball_dict,
            top_interstitial_bridges,
            pose_evidence=phase_pose_evidence,
            protected_frames=phase_protected_frames,
            edge_top_y=args.tracknet_phase_top_interstitial_edge_y,
            edge_core_y=args.tracknet_phase_edge_protect_y,
            contact_protect_px=args.tracknet_phase_contact_protect_px,
        )
        phase_player_vetoed_frames = sorted(set(
            phase_player_vetoed_frames + top_interstitial_veto_frames
        ) - set(phase_protected_frames))
        # A point seen by only one temporal phase can be a real reacquisition,
        # but an isolated point with no coherent bridge is usually a body
        # response.  Apply this high-precision veto after pose/branch gates so
        # a supported contact or top-edge lead remains protected.  It also
        # runs when the player detector is disabled, using trajectory support
        # alone rather than silently accepting a known single-phase outlier.
        isolated_candidates = {
            int(frame)
            for frame in phase_suspicious_frames
            if (
                int(frame) >= 10
                and not (scene_cuts and int(frame) >= int(scene_cuts[0]))
            )
        }
        (
            tracknet_ball_dict,
            phase_isolated_veto_frames,
            phase_isolated_veto_stats,
        ) = (
            veto_isolated_phase_observations(
                tracknet_ball_dict,
                tracknet_ball_primary,
                tracknet_ball_secondary,
                phase_pose_evidence,
                isolated_candidates,
                scene_cuts=scene_cuts,
                top_edge_y=args.tracknet_phase_edge_protect_y,
                contact_protect_px=args.tracknet_phase_contact_protect_px,
            )
        )
        phase_player_vetoed_frames = sorted(set(
            phase_player_vetoed_frames + phase_isolated_veto_frames
        ) - set(phase_protected_frames))
    phase_blocked_frames = {int(frame) for frame in phase_player_vetoed_frames}

    def apply_phase_blocks(observations):
        """Keep a phase/body veto from being reintroduced by classical fusion."""
        output = dict(observations)
        for frame in phase_blocked_frames:
            current = output.get(frame, (0, 0, 0))
            try:
                x, y = current[1], current[2]
            except (IndexError, TypeError):
                x, y = 0, 0
            output[frame] = (0, x, y, "phase_veto", 0.0)
        return output

    tracknet_ball_dict = apply_phase_blocks(tracknet_ball_dict)
    ball_dict = tracknet_ball_dict
    active_detector = "tracknet"
    classical_result = None
    classical_track_ids = {}
    classical_segments = []
    fusion_stats = {}
    gap_fill_stats = {}
    tracknet_longest_gap = 0
    # A phase/body veto is a hard temporal boundary for the smoother too.
    # Otherwise the zeroed observation can still be bridged by interpolation
    # or a short Kalman forecast and reappear as a plausible ball point.
    ball_reset_frames = sorted(set(scene_reset_frames) | phase_blocked_frames)
    ball_derivative_break_frames = sorted(set(scene_reset_frames) | phase_blocked_frames)
    ball_visual_break_frames = set()
    classical_render_top_extension = float(args.classical_top_extension)

    def prepare_ball_track(active_ball_dict, reset_frames, top_extension, static_window):
        """Apply the same ROI, Kalman and geometry stages to either detector."""
        active_quad = _expanded_ball_quad(
            court_quad,
            top_extension=top_extension,
            side_padding=args.ball_side_padding,
            bottom_padding=args.ball_bottom_padding,
        )
        # Keep a very small, temporally proven exception for ordinary points
        # immediately after a top-edge anchor.  The court quad remains strict
        # everywhere else; this mask only covers a monotone re-entry followed
        # by an in-quad confirmation (not a broad top/side padding).
        edge_reentry_frames = find_edge_reentry_frames(
            active_ball_dict,
            track_frame_count,
            active_quad,
            scene_cuts=scene_cuts,
            edge_y=args.tracknet_phase_edge_protect_y,
            max_frames=6,
            max_y=max(150.0, float(args.tracknet_phase_edge_protect_y) + 100.0),
            min_outside_frames=2,
            max_horizontal_step=70.0,
            max_vertical_step=80.0,
        )
        edge_reentry_frame_set = set(edge_reentry_frames)
        reset_frames = sorted(
            set(int(frame) for frame in reset_frames) | phase_blocked_frames
        )
        raw_track = smooth_ball_track(
            active_ball_dict,
            track_frame_count,
            fps,
            max_interp_gap=args.ball_max_interp_gap,
            max_predict_gap=args.ball_max_predict_gap,
            process_noise=args.ball_process_noise,
            reset_frames=reset_frames,
        )
        # Gate observations before Kalman so an off-court response cannot pull
        # interpolation or state toward a banner/player false positive.
        prefiltered = {}
        for frame_no, value in active_ball_dict.items():
            try:
                vis, bx, by = value[:3]
                point_xy = (float(bx), float(by))
                in_region = bool(vis) and point_in_quad(point_xy, active_quad)
                # The expanded quad is intentionally strict for ordinary
                # model/classical points, but a dual/single-phase edge label
                # can sit a few pixels outside the far-baseline endpoints
                # after perspective projection.  Dropping it here prevents
                # the smoother from ever seeing the re-entry anchor (notably
                # the x≈1338, y≈3 sequence around f574), which in turn makes
                # a real top-edge flight look like a long detector outage.
                # Keep the allowance a narrow segment corridor; it must not
                # turn the scoreboard into a valid ball ROI.
                source = str(value[3]) if len(value) >= 4 else "model"
                if (
                    not in_region
                    and bool(vis)
                    and source in EDGE_MEASUREMENT_SOURCES
                    and active_quad is not None
                    and float(by) <= float(np.max(active_quad[:2, 1])) + 18.0
                    and _point_near_top_edge(
                        point_xy,
                        active_quad,
                        perpendicular_tolerance=18.0,
                        side_tolerance=100.0,
                    )
                ):
                    in_region = True
                if not in_region and int(frame_no) in edge_reentry_frame_set:
                    in_region = True
            except (TypeError, ValueError, IndexError):
                in_region = False
            if (
                args.stop_ball_at_first_scene_cut
                and scene_cuts
                and int(frame_no) >= int(scene_cuts[0])
            ):
                in_region = False
            prefiltered[int(frame_no)] = value if in_region else (0, 0, 0)
        filtered_track = smooth_ball_track(
            prefiltered,
            track_frame_count,
            fps,
            max_interp_gap=args.ball_max_interp_gap,
            max_predict_gap=args.ball_max_predict_gap,
            process_noise=args.ball_process_noise,
            reset_frames=reset_frames,
        )
        filtered_track = filter_ball_track_geometry(
            filtered_track,
            court_polygon=court_quad,
            top_extension=top_extension,
            side_padding=args.ball_side_padding,
            bottom_padding=args.ball_bottom_padding,
            scene_cuts=scene_cuts,
            scene_cut_padding=0,
            static_window=static_window,
            static_disp=args.ball_static_disp,
            static_lock_window=args.ball_static_lock_window,
            static_lock_disp=args.ball_static_lock_disp,
            static_lock_timeout_window=args.ball_static_lock_timeout_window,
            static_lock_timeout_disp=args.ball_static_lock_timeout_disp,
            static_lock_min_y=args.ball_static_lock_min_y,
            static_lock_preroll=args.ball_static_lock_preroll,
            edge_reentry_frames=edge_reentry_frames,
        )
        return raw_track, prefiltered, filtered_track, active_quad

    raw_ball_track, prefiltered_ball_dict, ball_track, ball_quad = prepare_ball_track(
        tracknet_ball_dict,
        ball_reset_frames,
        args.ball_top_extension,
        args.ball_static_window,
    )
    # Keep a clean primary-phase trajectory as the hit-event reference.  Phase
    # fusion is intentionally more conservative for false-positive rejection;
    # using its missing/blocked points directly for curvature can otherwise
    # replace a proven contact with a later background turn.  The final fused
    # track still supplies the rendered shuttle and is used to validate the
    # reference event position.
    reference_hit_track = None
    reference_hit_events = []
    if tracknet_ball_secondary and tracknet_ball_primary:
        try:
            _, _, reference_hit_track, _ = prepare_ball_track(
                tracknet_ball_primary,
                ball_reset_frames,
                args.ball_top_extension,
                args.ball_static_window,
            )
            reference_hit_events = detect_hit_points(
                reference_hit_track,
                fps,
                homography=homography,
                min_separation_frames=args.hit_min_separation,
                min_speed_px=args.hit_min_speed_px,
                break_frames=ball_derivative_break_frames,
            )
        except Exception as exc:
            print(f"[WARN] primary hit-reference pass failed; using fused track: {exc}")
            reference_hit_track = None
            reference_hit_events = []
    tracknet_total_cov, tracknet_real_cov, tracknet_derived_cov = ball_coverage(ball_track)
    tracknet_real_points = sum(
        1 for point in ball_track
        if point.visible and point.source == "model"
    )
    gap_frame_count = (
        min(track_frame_count, int(scene_cuts[0]))
        if args.stop_ball_at_first_scene_cut and scene_cuts else
        track_frame_count
    )
    tracknet_longest_gap = longest_observation_gap(
        prefiltered_ball_dict,
        gap_frame_count,
    )

    run_classical = args.classical_ball_fallback and (
        tracknet_real_cov < args.classical_trigger_coverage
        or tracknet_longest_gap >= args.classical_trigger_gap_frames
    )
    if run_classical:
        try:
            # Lazy import keeps direct TrackNet-only use independent of the
            # optional CPU fallback module.
            from classical_ball_detector import detect_classical_ball

            classical_result = detect_classical_ball(
                args.video_path,
                court_quad,
                scene_cuts=scene_cuts,
                court_top_extension=args.classical_top_extension,
                high_flight_recall_enabled=args.classical_high_flight_recall,
                high_flight_top_extension=args.classical_high_flight_top_extension,
                court_side_padding=args.ball_side_padding,
                court_bottom_padding=args.ball_bottom_padding,
            )
            candidate_dict = {
                int(frame): (
                    1,
                    int(round(observation.x)),
                    int(round(observation.y)),
                    "classical",
                    float(observation.confidence),
                )
                for frame, observation in classical_result.observations.items()
                if 0 <= int(frame) < int(classical_result.frame_count)
                and np.isfinite([observation.x, observation.y, observation.confidence]).all()
            }
            candidate_track_ids = {
                int(frame): int(observation.track_id)
                for frame, observation in classical_result.observations.items()
                if int(frame) in candidate_dict
            }

            # Build boundaries from final observations rather than raw ranges:
            # temporal NMS may leave partially overlapping range metadata.
            previous_frame = None
            previous_track_id = None
            segment_start = None
            segment_end = None
            for frame in sorted(candidate_dict):
                track_id = candidate_track_ids[frame]
                begins_segment = (
                    previous_frame is None
                    or track_id != previous_track_id
                    or frame - previous_frame > 2
                )
                if begins_segment:
                    if segment_start is not None:
                        classical_segments.append(
                            (previous_track_id, segment_start, segment_end)
                        )
                    segment_start = frame
                segment_end = frame
                previous_frame = frame
                previous_track_id = track_id
            if segment_start is not None:
                classical_segments.append(
                    (previous_track_id, segment_start, segment_end)
                )

            enough_classical = (
                len(candidate_dict) >= args.classical_min_observations
                and len(classical_segments) >= args.classical_min_tracks
            )
            if enough_classical:
                track_frame_count = int(classical_result.frame_count or track_frame_count)
                segment_starts = [start for _, start, _ in classical_segments]
                ball_reset_frames = sorted(set(scene_reset_frames) | phase_blocked_frames)
                classical_render_top_extension = max(
                    float(args.classical_top_extension),
                    float(args.classical_high_flight_top_extension)
                    if args.classical_high_flight_recall else
                    float(args.classical_top_extension),
                )
                # A strong official model contributes many real points between
                # sparse classical anchors.  Fuse candidates before the single
                # Kalman pass instead of replacing an entire detector.  With a
                # weak/domain-mismatched model, keep the proven classical-only
                # precision fallback.
                use_fusion = (
                    tracknet_real_cov >= args.classical_trigger_coverage
                    and tracknet_real_points >= max(
                        args.classical_min_observations,
                        int(round(0.75 * len(candidate_dict))),
                    )
                )
                if use_fusion:
                    active_detector = "fused"
                    ball_dict, classical_track_ids, fusion_stats = fuse_ball_observations(
                        tracknet_ball_dict,
                        classical_result,
                        track_frame_count,
                        scene_cuts=scene_cuts,
                        agreement_px=args.ball_fusion_agreement_px,
                        bridge_search_frames=args.ball_fusion_bridge_frames,
                    )
                    ball_dict = apply_phase_blocks(ball_dict)
                    # The normal fusion runs before the final sparse
                    # phase/body veto.  Recover only a tightly bounded hole
                    # which still has two ordinary TrackNet anchors and a
                    # multi-point classical track; existing model points are
                    # never replaced.  ``classical_gap`` is renderable but
                    # intentionally excluded from hit/coverage evidence.
                    (
                        ball_dict,
                        gap_classical_ids,
                        gap_fill_stats,
                    ) = fill_tracknet_gaps_with_classical(
                        ball_dict,
                        classical_result,
                        track_frame_count,
                        scene_cuts=scene_cuts,
                    )
                    classical_track_ids.update(gap_classical_ids)
                    # Keep the first frame after an accepted derived run as a
                    # hard smoother boundary.  Otherwise the generic
                    # short-gap interpolator can turn the following model
                    # reacquisition into an ``interp`` point, accidentally
                    # making a classical-only recovery look like ordinary
                    # measured evidence.
                    gap_break_frames = {
                        int(frame) + 1
                        for frame in gap_classical_ids
                        if int(frame) + 1 < int(track_frame_count)
                    }
                    ball_reset_frames = sorted(
                        set(ball_reset_frames) | gap_break_frames
                    )
                    ball_derivative_break_frames = sorted(
                        set(ball_derivative_break_frames) | gap_break_frames
                    )
                    fusion_stats["classical_gap_fill"] = int(
                        gap_fill_stats.get("filled", 0)
                    )
                    fused_top_extension = max(
                        float(args.ball_top_extension),
                        float(classical_render_top_extension),
                    )
                    raw_ball_track, prefiltered_ball_dict, ball_track, ball_quad = prepare_ball_track(
                        ball_dict,
                        ball_reset_frames,
                        fused_top_extension,
                        args.ball_static_window,
                    )
                else:
                    active_detector = "classical"
                    ball_dict = apply_phase_blocks(candidate_dict)
                    classical_track_ids = candidate_track_ids
                    ball_visual_break_frames.update(segment_starts)
                    ball_derivative_break_frames = sorted(
                        set(scene_reset_frames + segment_starts)
                        | phase_blocked_frames
                    )
                    # The strict detector already rejects stationary
                    # components; no model-background gate is needed.
                    raw_ball_track, prefiltered_ball_dict, ball_track, ball_quad = prepare_ball_track(
                        ball_dict,
                        ball_reset_frames,
                        classical_render_top_extension,
                        0,
                    )
            else:
                classical_result = None
                classical_track_ids = {}
                classical_segments = []
        except Exception as exc:
            print(f"[WARN] classical ball fallback failed; keeping TrackNet: {exc}")
            classical_result = None
            classical_track_ids = {}
            classical_segments = []

    visual_support_frames = set()
    if active_detector == "classical":
        conservative_quad = _expanded_ball_quad(
            court_quad,
            top_extension=args.classical_top_extension,
            side_padding=args.ball_side_padding,
            bottom_padding=args.ball_bottom_padding,
        )
        contact_min_y = float(np.min(conservative_quad[:2, 1]))
        hit_events, visual_support_frames, stitched_starts = detect_classical_visual_hits(
            classical_result,
            ball_track,
            contact_min_y=contact_min_y,
            endpoint_gap_frames=args.hit_tracklet_gap_frames,
        )
        # A successfully stitched boundary is the same ballistic flight, so
        # do not flash/clear the neon trail at that join.
        ball_visual_break_frames.difference_update(stitched_starts)
        ball_derivative_break_frames = [
            frame for frame in ball_derivative_break_frames
            if frame not in stitched_starts
        ]
    else:
        fused_hit_events = detect_hit_points(
            ball_track,
            fps,
            homography=homography,
            min_separation_frames=args.hit_min_separation,
            min_speed_px=args.hit_min_speed_px,
            break_frames=ball_derivative_break_frames,
        )
        # A far-court high clear can reverse at the racket and immediately
        # leave above the image.  Its follow-up points are edge-labelled on
        # purpose and therefore cannot form a normal three-point curvature
        # candidate.  Add only the measured, one-sided edge-exit pattern; it
        # still has to pass the shared audio and pose/body evidence gates.
        fused_hit_events.extend(detect_top_edge_exit_hits(
            ball_track,
            break_frames=ball_derivative_break_frames,
        ))
        hit_events = merge_visual_hit_candidates(
            reference_hit_events,
            fused_hit_events,
            min_separation_frames=args.hit_min_separation,
        )
        visual_support_frames.update(event.frame for event in hit_events)

    audio_hit_events, raw_audio_candidates, audio_available = build_audio_hit_events(
        args.video_path,
        fps,
        ball_track,
        scene_cuts,
        visual_support_frames,
        classical_track_ids,
        model,
        court_quad,
        midline_y_img,
        args,
    )
    visual_event_frames = [int(event.frame) for event in hit_events]
    audio_hit_events = [
        event for event in audio_hit_events
        if any(
            abs(int(event.frame) - frame) <= int(args.audio_hit_visual_tolerance)
            for frame in visual_event_frames
        )
    ]
    # Validate before temporal NMS.  Otherwise a high-scoring pose/body false
    # positive could suppress a nearby trajectory-supported candidate and be
    # removed only afterwards, leaving no event at all.
    hit_candidates = list(hit_events) + list(audio_hit_events)
    hit_candidates = [
        event for event in hit_candidates
        if 0 <= int(event.frame) < len(ball_track)
        and point_in_quad((event.x, event.y), ball_quad)
    ]
    hit_pose_evidence = _infer_audio_pose_contacts(
        model,
        args.video_path,
        [event.frame for event in hit_candidates],
        court_quad,
        midline_y_img,
        args,
    ) if model is not None and hit_candidates else {}
    # Do not limit strict hitter attribution to the audio-localized subset.
    # At this point every final visual candidate has a frame-local pose sample
    # and a measured ball coordinate, so the association can be applied
    # consistently before validation/NMS chooses the rendered event.
    hit_pose_association_stats = apply_strict_pose_hitter_associations(
        hit_candidates,
        hit_pose_evidence,
    )
    hit_player_boxes = {
        int(frame): [
            player["box"] for player in evidence.get("players", [])
            if player.get("box") is not None
        ]
        for frame, evidence in hit_pose_evidence.items()
    }
    hit_player_contacts = {}
    for frame, evidence in hit_pose_evidence.items():
        contacts = []
        for player in evidence.get("players", []):
            contacts.extend(player.get("wrists", []))
            if player.get("contact") is not None:
                contacts.append(player["contact"])
        if contacts:
            hit_player_contacts[int(frame)] = contacts
    hit_candidates, hit_evidence_stats = validate_hit_event_evidence(
        ball_track,
        hit_candidates,
        audio_frames=[candidate.frame for candidate in raw_audio_candidates],
        player_boxes=hit_player_boxes,
        # Audio is an additional gate only when the source actually contains
        # a decodable audio track.  Requiring a nonexistent stream otherwise
        # rejects every valid trajectory event (for example silent coaching
        # clips) and turns an informative visual result into zero hits.
        require_audio=bool(args.audio_hit_detection and audio_available),
        audio_tolerance_frames=args.audio_hit_visual_tolerance,
        evidence_window_frames=min(18, max(6, args.audio_hit_ball_search_frames)),
        min_real_points=2,
        position_tolerance_px=90.0,
        player_margin_px=args.ball_player_veto_margin,
        player_contacts=hit_player_contacts,
        contact_tolerance_px=args.hit_contact_protect_px,
    )
    hit_events = merge_hit_events(
        [event for event in hit_candidates if not event.source.startswith("audio_")],
        [event for event in hit_candidates if event.source.startswith("audio_")],
        min_separation_frames=args.hit_min_separation,
        same_track_separation=args.hit_min_separation,
    )
    # A hit starts a new physical flight.  Keeping the whole pre-hit segment
    # in the same neon deque makes a valid rally look like a dense yellow
    # scribble, and it visually exaggerates a one-frame model switch at the
    # contact.  Clear only *after* a verified hit so the marker still has the
    # incoming trajectory as context while the next frame begins a clean
    # outgoing flight.  This is rendering-only: the full track, events and
    # sidecar remain unchanged.
    ball_visual_break_frames.update(
        int(event.frame) + 1
        for event in hit_events
        if int(event.frame) + 1 < int(track_frame_count)
    )
    # Surviving candidates have measured-ball + audio support by default.
    # Keep uncertainty only as provenance (e.g. a one-sided endpoint); weak
    # pose/body candidates have already been removed and cannot be rendered.
    for event in hit_events:
        event_point = ball_track[event.frame] if 0 <= event.frame < len(ball_track) else None
        low_score_but_confirmed = has_confirmed_model_audio_evidence(
            event,
            event_point,
            [candidate.frame for candidate in raw_audio_candidates],
            args.audio_hit_visual_tolerance,
        )
        event.uncertain = bool(
            getattr(event, "uncertain", False)
            or event.source.startswith("audio_")
            or args.skip_player_detector
            or event_point is None
            or event_point.source != "model"
            or event.source == "classical_endpoint"
            or (float(event.score) < 0.72 and not low_score_but_confirmed)
        )
    rally_labels, rally_segments = segment_rallies(
        ball_track,
        hit_events,
        fps,
        gap_frames=args.ball_rally_gap_frames,
        scene_cuts=scene_cuts,
        min_segment_points=args.ball_min_rally_points,
    )
    # Keep render-only top-edge bridges separate from the measured track used
    # above for hit validation and rally segmentation.  A bridge is a visual
    # cue for an out-of-frame flight, never evidence for a new hit/rally.
    render_ball_track = ball_track
    offscreen_bridge_count = 0
    if args.offscreen_bridge:
        render_ball_track = add_offscreen_bridges(
            ball_track,
            scene_cuts=scene_cuts,
            max_gap=args.offscreen_bridge_max_gap,
            min_gap=args.offscreen_bridge_min_gap,
            top_y=args.offscreen_bridge_top_y,
            velocity_window=args.offscreen_bridge_velocity_window,
            min_speed=args.offscreen_bridge_min_speed,
            min_vertical_speed=args.offscreen_bridge_min_vertical_speed,
            min_horizontal_speed=args.offscreen_bridge_min_horizontal_speed,
            confidence=args.offscreen_bridge_confidence,
            allow_mixed_edge_bridge=args.offscreen_bridge_mixed_edge_bridge,
            mixed_support_frames=args.offscreen_bridge_mixed_support_frames,
            mixed_speed_ratio=args.offscreen_bridge_mixed_speed_ratio,
        )
        offscreen_bridge_count = sum(
            1 for point in render_ball_track
            if point.source == "offscreen_bridge"
        )
    ball_x_shift = estimate_ball_center_shift(
        prefiltered_ball_dict,
        homography,
        court_quad,
        args.court_width_m,
        args.court_length_m,
    )
    for event in hit_events:
        map_m = to_court_m((event.x, event.y), homography)
        if map_m is not None:
            map_m = (
                float(np.clip(map_m[0] + ball_x_shift, 0.0, args.court_width_m)),
                float(np.clip(map_m[1], 0.0, args.court_length_m)),
            )
            event.map_xy = to_map_px(map_m, w - max(200, int(w * 0.18)) - 12 + 14, 12 + 42,
                                     max(200, int(w * 0.18)) - 28,
                                     max(320, int(h * 0.62)) - 52,
                                     args.court_width_m, args.court_length_m)
    tracking_csv = os.path.splitext(args.output_path)[0] + "_ball_tracking.csv"
    near_stats = MotionStats()
    far_stats = MotionStats()
    rally_idx = 1
    both_missing = 0
    prev_near_pt = None
    prev_far_pt = None
    prev_near_id = None
    prev_far_id = None
    near_missing_count = 0
    far_missing_count = 0
    in_rally = False
    prev_gray = None
    near_player_cache = None
    far_player_cache = None
    near_cache_age = 0
    far_cache_age = 0
    track_memory = {}
    offcourt_track_ids = set()
    last_ball_rally = 0
    last_ball_veto_boxes = []
    body_vetoed_hit_frames = set()
    # This set has already passed a stricter two-phase, two-sided-motion test
    # before the pose pre-pass.  Preserve it through the render-time body veto
    # as well; otherwise a real contact can be accepted for tracking and then
    # silently removed from the video/CSV solely because its ball center falls
    # inside the hitter's detector rectangle.
    render_hit_protected_frames = {
        int(frame) for frame in dual_motion_protected_frames
    }
    scene_cut_reset_frames = {int(cut) for cut in scene_cuts}
    # Event counters intentionally start at zero and advance only when the
    # playback reaches a surviving event.  Do not pre-aggregate the complete
    # video here: that was the source of a counter that started at 27 and then
    # fell as render-time body checks removed future events.
    live_hits_by_rally = {}
    video_hit_total = 0
    live_hit_number_by_frame = {}
    last_confirmed_event_by_rally = {}
    completed_shot_speed_by_event_frame = {}
    completed_shot_method_by_event_frame = {}
    # Keep the most recent usable estimate visible when the next flight is
    # genuinely unobservable; the panel calls this out as "recent valid" so
    # it is never mistaken for the current missing flight.
    last_valid_shot_speed_kmh = None
    last_valid_shot_method = ""

    frame_idx = 0
    detect_calls = 0
    t_start = time.perf_counter()
    print(f"[INFO] video={args.video_path}")
    print(f"[INFO] output={args.output_path}")
    print(f"[INFO] size={w}x{h}, fps={fps:.2f}, frames={total_frames}")
    print(f"[INFO] court_points={court_quad.astype(int).tolist()}")
    print(
        f"[INFO] detect_interval={max(1,args.detect_interval)}, "
        f"half={args.half}, "
        f"detect_roi={args.detect_on_court_roi}, "
        f"detect_area_ratio={detect_area_ratio:.3f}, "
        f"top_inner_ratio={args.top_inner_ratio:.3f}, "
        f"side_band_px={args.side_band_px:.1f}, "
        f"far_half_expand={args.far_half_expand:.2f}, "
        f"draw_pose_on_skipped={args.draw_pose_on_skipped}"
    )
    raw_cov, raw_real_cov, raw_derived_cov = ball_coverage(raw_ball_track)
    total_cov, real_cov, derived_cov = ball_coverage(ball_track)
    print(f"[INFO] scene_cuts={scene_cuts or 'none'}")
    if tracknet_ball_secondary:
        print(
            "[INFO] tracknet_phase_fusion="
            + ",".join(
                f"{key}:{value}"
                for key, value in sorted(phase_fusion_stats.items())
            )
            + f",player_vetoed:{len(phase_player_vetoed_frames)}"
            + f",isolated_vetoed:{len(phase_isolated_veto_frames)}"
            + f",top_entry_lead:{len(top_entry_lead_frames)}"
            + f",dual_motion_protected:{len(dual_motion_protected_frames)}"
            + f",top_interstitial_candidates:{len(top_interstitial_candidates)}"
            + f",top_interstitial_vetoed:{len(top_interstitial_veto_frames)}"
            + "," + ",".join(
                f"{key}:{value}"
                for key, value in sorted(phase_branch_veto_stats.items())
            )
        )
        if dual_motion_protection_stats:
            print(
                "[INFO] dual_motion_protection="
                + ",".join(
                    f"{key}:{value}"
                    for key, value in sorted(dual_motion_protection_stats.items())
                )
                + f",frames:{dual_motion_protected_frames}"
            )
        if top_interstitial_stats:
            print(
                "[INFO] top_interstitial="
                + ",".join(
                    f"{key}:{value}"
                    for key, value in sorted(top_interstitial_stats.items())
                )
                + ",veto_"
                + ",".join(
                    f"{key}:{value}"
                    for key, value in sorted(top_interstitial_veto_stats.items())
                )
            )
            # Keep the concrete frame ids in the diagnostic log as well as
            # aggregate counts.  Candidate runs are intentionally sparse,
            # so this remains compact and makes a parameter change (for
            # example 110→180 px) auditable without opening the sidecar.
            print(
                "[INFO] top_interstitial_frames="
                f"candidates:{top_interstitial_candidates},"
                f"vetoed:{top_interstitial_veto_frames}"
            )
    print(
        f"[INFO] active_ball_detector={active_detector}, "
        f"tracknet_filtered={tracknet_total_cov:.1%} "
        f"(real={tracknet_real_cov:.1%}, derived={tracknet_derived_cov:.1%}), "
        f"tracknet_longest_gap={tracknet_longest_gap}"
    )
    if active_detector == "classical":
        print(
            f"[INFO] classical_observations={len(classical_track_ids)}, "
            f"segments={len(classical_segments)}, "
            f"top_extension={classical_render_top_extension:.1f}px"
        )
    elif active_detector == "fused":
        print(
            "[INFO] ball_fusion="
            + ",".join(
                f"{key}:{value}" for key, value in sorted(fusion_stats.items())
            )
        )
    if args.audio_hit_detection:
        print(
            f"[INFO] audio_hit_candidates={len(raw_audio_candidates)}, "
            f"localized={len(audio_hit_events)}, "
            f"available={bool(audio_available)}, "
            f"strong_score={args.audio_hit_strong_score:.2f}"
        )
    if args.offscreen_bridge:
        print(f"[INFO] offscreen_render_bridges={offscreen_bridge_count}")
    print(
        "[INFO] hit_evidence="
        + ",".join(
            f"{key}:{value}" for key, value in sorted(hit_evidence_stats.items())
        )
    )
    print(
        "[INFO] hit_hitter_pose="
        + ",".join(
            f"{key}:{value}"
            for key, value in sorted(hit_pose_association_stats.items())
        )
    )
    print(f"[INFO] ball_csv={ball_csv or '<none>'}, "
          f"secondary={ball_csv_secondary or '<none>'}, "
          f"hits={len(hit_events)}, rallies={len(rally_segments)}, "
          f"coverage(filtered)={total_cov:.1%} (real={real_cov:.1%}, derived={derived_cov:.1%}), "
          f"raw={raw_cov:.1%}")
    print(f"[INFO] ball_tracking_sidecar={tracking_csv}")
    if args.draw_ball_on_minimap and ball_dict:
        if ball_csv:
            print(f"[INFO] ball_csv={ball_csv}")
        print(f"[INFO] ball_center_shift_x={ball_x_shift:.3f}m")
    else:
        print("[INFO] mini-court ball trail disabled")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ball_sample = render_ball_track[frame_idx] if frame_idx < len(render_ball_track) else BallPoint(frame_idx, 0, 0, False)
        if frame_idx in ball_visual_break_frames:
            # A new independent classical tracklet must never inherit the old
            # main-frame or mini-court line.  Player trails remain continuous.
            ball_trail.clear()
            ball_map_hist.clear()
        if frame_idx in scene_cut_reset_frames:
            # Do not connect trajectories across a camera change.  The player
            # trails are reset as well because their image-space coordinates
            # belong to the old calibration.
            ball_trail.clear()
            ball_map_hist.clear()
            near_map_hist.clear()
            far_map_hist.clear()
            near_heatmap.fill(0.0)
            far_heatmap.fill(0.0)
            last_near_heatmap_frame = None
            last_far_heatmap_frame = None
            near_trail.clear()
            far_trail.clear()
            prev_near_pt = None
            prev_far_pt = None
            prev_near_id = None
            prev_far_id = None
            near_player_cache = None
            far_player_cache = None
            last_ball_veto_boxes = []

        do_detect = (frame_idx == 0) or (frame_idx % max(1, args.detect_interval) == 0)
        if args.detect_ball_frames_only and not ball_sample.visible and frame_idx > 0:
            # CPU-friendly mode: player boxes are only needed on frames where
            # a TrackNet point could enter the rendered trail/hit decision.
            do_detect = False
        results = []
        roi_ox, roi_oy = 0, 0
        if do_detect and model is not None:
            detect_frame = frame[detect_y1:detect_y2, detect_x1:detect_x2]
            if detect_frame.size == 0:
                detect_frame = frame
            else:
                roi_ox, roi_oy = detect_x1, detect_y1
            track_kwargs = {
                "source": detect_frame,
                "persist": True,
                "tracker": args.tracker_cfg,
                "classes": [0],
                "conf": args.conf_thres,
                "verbose": False,
                "imgsz": args.imgsz,
                "device": args.device if args.device else None,
            }
            # Newer Ultralytics releases deprecate the explicit ``half=False``
            # keyword and warn on every sparse detector call.  Omit it for the
            # normal/full-precision path while preserving the existing GPU
            # ``--half`` opt-in for older environments.
            if args.half:
                track_kwargs["half"] = True
            results = model.track(**track_kwargs)
            detect_calls += 1

        candidates = []
        if len(results) > 0:
            r = results[0]
            if r.boxes is not None and r.boxes.id is not None and len(r.boxes.id) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)
                kpts_xy = None
                kpts_conf = None
                if args.draw_pose and r.keypoints is not None and r.keypoints.xy is not None:
                    kpts_xy = r.keypoints.xy.cpu().numpy()
                    if r.keypoints.conf is not None:
                        kpts_conf = r.keypoints.conf.cpu().numpy()

                for idx, (box, tid) in enumerate(zip(boxes, ids)):
                    x1, y1, x2, y2 = box
                    x1 += roi_ox
                    y1 += roi_oy
                    x2 += roi_ox
                    y2 += roi_oy
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw < args.min_box_w or bh < args.min_box_h:
                        continue

                    tid_i = int(tid)
                    if tid_i in offcourt_track_ids:
                        continue

                    person_kpts = None
                    person_kconf = None
                    if kpts_xy is not None and idx < len(kpts_xy):
                        person_kpts = kpts_xy[idx].copy()
                        person_kpts[:, 0] += roi_ox
                        person_kpts[:, 1] += roi_oy
                        if kpts_conf is not None and idx < len(kpts_conf):
                            person_kconf = kpts_conf[idx]

                    is_far_zone_hint = y2 <= (midline_y_img + args.far_zone_hint_margin)
                    foot_x, foot_y = estimate_foot_point(
                        (x1, y1, x2, y2),
                        keypoints=person_kpts,
                        confs=person_kconf,
                        ankle_min_conf=args.ankle_min_conf,
                        fallback_lift_ratio=(args.far_fallback_lift_ratio if is_far_zone_hint else 0.0),
                    )

                    hist = update_track_memory(track_memory, tid_i, (foot_x, foot_y), max_len=args.judge_hist_len)
                    if is_static_top_track(
                        hist,
                        top_y,
                        min_frames=args.judge_min_frames,
                        top_zone_px=args.judge_top_zone_px,
                        top_ratio=args.judge_top_ratio,
                        static_move_px=args.judge_static_move_px,
                    ):
                        offcourt_track_ids.add(tid_i)
                        continue

                    if not point_in_quad((foot_x, foot_y), court_quad):
                        continue
                    if not point_in_quad((foot_x, foot_y), player_filter_quad):
                        if prev_far_pt is None or euclidean((foot_x, foot_y), prev_far_pt) > args.top_inner_keep_dist:
                            continue

                    cand = {
                        "track_id": tid_i,
                        "box": (float(x1), float(y1), float(x2), float(y2)),
                        "foot": (foot_x, foot_y),
                    }
                    if person_kpts is not None:
                        cand["kpts"] = person_kpts
                        if person_kconf is not None:
                            cand["kconf"] = person_kconf
                    candidates.append(cand)

        near_player, far_player = assign_players(
            candidates,
            prev_near_pt,
            prev_far_pt,
            prev_near_id,
            prev_far_id,
            far_lock_max_jump=args.far_lock_max_jump,
            far_max_up_jump=args.far_max_up_jump,
            side_line_y=midline_y_img,
            side_band_px=args.side_band_px,
        )
        near_draw_pt = None
        far_draw_pt = None
        missing_step = 1 if (do_detect and model is not None) else 0

        if near_player is None and far_player is None:
            both_missing += missing_step
            near_missing_count += missing_step
            far_missing_count += missing_step

            if prev_near_pt is not None and near_missing_count <= args.hold_missing_frames:
                flow_near = estimate_point_by_optical_flow(prev_gray, cur_gray, prev_near_pt, args.max_flow_move)
                if flow_near is not None and point_in_quad(flow_near, court_quad):
                    prev_near_pt = flow_near
                near_draw_pt = prev_near_pt
            elif near_missing_count > args.reset_missing_frames:
                prev_near_pt = None
                prev_near_id = None

            if prev_far_pt is not None and far_missing_count <= args.hold_missing_frames:
                flow_far = estimate_point_by_optical_flow(prev_gray, cur_gray, prev_far_pt, args.max_flow_move)
                if flow_far is not None and point_in_quad(flow_far, court_quad):
                    prev_far_pt = flow_far
                far_draw_pt = prev_far_pt
            elif far_missing_count > args.reset_missing_frames:
                prev_far_pt = None
                prev_far_id = None

            near_trail.append(near_draw_pt)
            far_trail.append(far_draw_pt)
            near_stats.update(to_court_m(near_draw_pt, homography) if near_draw_pt is not None else None, frame_idx, fps)
            far_stats.update(to_court_m(far_draw_pt, homography) if far_draw_pt is not None else None, frame_idx, fps)
            if both_missing >= args.new_rally_missing_frames:
                in_rally = False
        else:
            if not in_rally:
                if frame_idx > 0:
                    rally_idx += 1
                    near_stats.reset_rally()
                    far_stats.reset_rally()
                in_rally = True
            both_missing = 0

            if near_player is not None:
                prev_near_pt = near_player["foot"]
                prev_near_id = near_player["track_id"]
                near_missing_count = 0
                near_draw_pt = near_player["foot"]
            else:
                near_missing_count += missing_step
                if prev_near_pt is not None and near_missing_count <= args.hold_missing_frames:
                    flow_near = estimate_point_by_optical_flow(prev_gray, cur_gray, prev_near_pt, args.max_flow_move)
                    if flow_near is not None and point_in_quad(flow_near, court_quad):
                        prev_near_pt = flow_near
                    near_draw_pt = prev_near_pt
                elif near_missing_count > args.reset_missing_frames:
                    prev_near_pt = None
                    prev_near_id = None

            if far_player is not None:
                prev_far_pt = far_player["foot"]
                prev_far_id = far_player["track_id"]
                far_missing_count = 0
                far_draw_pt = far_player["foot"]
            else:
                far_missing_count += missing_step
                if prev_far_pt is not None and far_missing_count <= args.hold_missing_frames:
                    flow_far = estimate_point_by_optical_flow(prev_gray, cur_gray, prev_far_pt, args.max_flow_move)
                    if flow_far is not None and point_in_quad(flow_far, court_quad):
                        prev_far_pt = flow_far
                    far_draw_pt = prev_far_pt
                elif far_missing_count > args.reset_missing_frames:
                    prev_far_pt = None
                    prev_far_id = None

            near_trail.append(near_draw_pt)
            far_trail.append(far_draw_pt)
            near_stats.update(to_court_m(near_draw_pt, homography) if near_draw_pt is not None else None, frame_idx, fps)
            far_stats.update(to_court_m(far_draw_pt, homography) if far_draw_pt is not None else None, frame_idx, fps)

        near_player_draw = near_player
        far_player_draw = far_player

        if near_player is not None:
            near_player_cache = near_player
            near_cache_age = 0
        else:
            near_cache_age += 1
            if (
                (not do_detect)
                and args.draw_pose_on_skipped
                and near_player_cache is not None
                and near_cache_age <= args.skip_pose_max_age
            ):
                if near_draw_pt is None:
                    near_draw_pt = prev_near_pt if prev_near_pt is not None else near_player_cache["foot"]
                near_player_draw = shift_player_for_draw(near_player_cache, near_draw_pt)

        if far_player is not None:
            far_player_cache = far_player
            far_cache_age = 0
        else:
            far_cache_age += 1
            if (
                (not do_detect)
                and args.draw_pose_on_skipped
                and far_player_cache is not None
                and far_cache_age <= args.skip_pose_max_age
            ):
                if far_draw_pt is None:
                    far_draw_pt = prev_far_pt if prev_far_pt is not None else far_player_cache["foot"]
                far_player_draw = shift_player_for_draw(far_player_cache, far_draw_pt)

        # Apply a frame-local player-box veto only to the rendered ball point.
        # TrackNet responses in the supplied checkpoint often land in a
        # player's shirt/head.  Keeping the raw point in `ball_track` retains
        # diagnostics and hit candidates, while preventing an implausible
        # point from contaminating the neon trail or mini-court history.
        veto_boxes = [c["box"] for c in candidates]
        if not veto_boxes:
            for cached in (near_player_cache, far_player_cache):
                if cached is not None and "box" in cached:
                    veto_boxes.append(cached["box"])
        if veto_boxes:
            last_ball_veto_boxes = list(veto_boxes)
        elif last_ball_veto_boxes:
            veto_boxes = last_ball_veto_boxes

        ball_draw_sample = ball_sample
        ball_veto_foot_radius = (
            args.ball_player_veto_foot_radius
            if str(ball_sample.source) == "model" else 0.0
        )
        if (
            not ball_sample.visible
            or _ball_overlaps_player(
                (ball_sample.x, ball_sample.y), veto_boxes,
                margin=args.ball_player_veto_margin,
                player_points=(near_draw_pt, far_draw_pt),
                foot_radius=ball_veto_foot_radius,
            )
        ):
            ball_draw_sample = None

        if args.draw_ball_trajectory and ball_draw_sample is not None:
            ball_trail.append((ball_draw_sample.x, ball_draw_sample.y,
                               ball_draw_sample.source, frame_idx))
        else:
            # A None entry prevents a line across a real visibility gap or a
            # player-box veto.
            ball_trail.append(None)

        # Defensive frame-local hard veto.  The sparse pre-pass normally
        # removes these before rally segmentation; this catches a player box
        # found only by the render pass and guarantees that a body-overlap
        # event cannot flash on either the main view or mini court.  Keep the
        # same wrist/contact exception used by the evidence validator: a real
        # shuttle is expected to overlap the hitter's box at racket contact.
        for event in hit_events:
            contact_protected = _near_pose_contact(
                (event.x, event.y),
                hit_player_contacts.get(int(event.frame), ()),
                tolerance=args.hit_contact_protect_px,
            )
            if (
                event.frame == frame_idx
                and int(event.frame) not in render_hit_protected_frames
                and not contact_protected
                and _ball_overlaps_player(
                    (event.x, event.y), veto_boxes,
                    margin=args.ball_player_veto_margin,
                    player_points=None,
                    foot_radius=0.0,
                )
            ):
                body_vetoed_hit_frames.add(int(event.frame))

        # Count only when playback reaches an event that already passed the
        # trajectory/audio evidence gate.  ``uncertain`` is a visual
        # confidence label, not a second rejection reason: by default HIT?
        # also advances the visible rally and video totals.
        for event in hit_events:
            if (
                int(event.frame) != frame_idx
                or int(event.frame) in body_vetoed_hit_frames
                or int(event.frame) in live_hit_number_by_frame
                or (
                    bool(getattr(event, "uncertain", False))
                    and not args.count_uncertain_hits
                )
            ):
                continue
            rally_key = int(getattr(event, "rally", 0))
            rally_count = live_hits_by_rally.get(rally_key, 0) + 1
            live_hits_by_rally[rally_key] = rally_count
            video_hit_total += 1
            event.rally_hit_index = rally_count
            event.video_hit_index = video_hit_total
            live_hit_number_by_frame[int(event.frame)] = video_hit_total

            # A completed ball flight is the interval from the last confirmed
            # contact in this rally to this one.  We intentionally calculate
            # it now (rather than before rendering) so the video never shows
            # a speed that depends on a future, later-vetoed hit.
            previous_event = last_confirmed_event_by_rally.get(rally_key)
            completed_speed_kmh = None
            completed_speed_method = "none"
            if previous_event is not None:
                shot = estimate_shot_average_speed(
                    ball_track,
                    previous_event.frame,
                    event.frame,
                    fps,
                    homography,
                    court_width_m=args.court_width_m,
                    court_length_m=args.court_length_m,
                    court_margin_m=args.shot_speed_court_margin_m,
                    min_coverage=args.shot_speed_min_coverage,
                    min_real_points=args.shot_speed_min_real_points,
                    min_segments=args.shot_speed_min_segments,
                    max_segment_speed_mps=args.shot_speed_max_segment_mps,
                    max_average_speed_mps=args.shot_speed_max_average_mps,
                    proxy_floor_y=top_y if args.shot_speed_perspective_proxy else None,
                    proxy_min_coverage=args.shot_speed_proxy_min_coverage,
                    proxy_min_real_points=args.shot_speed_proxy_min_real_points,
                    proxy_min_segments=args.shot_speed_proxy_min_segments,
                    proxy_max_scale_m_per_px=args.shot_speed_proxy_max_scale_m_per_px,
                    proxy_max_average_speed_mps=args.shot_speed_proxy_max_average_mps,
                )
                previous_event.shot_end_frame = int(event.frame)
                previous_event.shot_avg_speed_mps = shot.get("speed_mps")
                previous_event.shot_distance_m = float(shot.get("distance_m", 0.0))
                previous_event.shot_observed_seconds = float(shot.get("observed_seconds", 0.0))
                previous_event.shot_elapsed_seconds = float(shot.get("elapsed_seconds", 0.0))
                previous_event.shot_coverage = float(shot.get("coverage", 0.0))
                previous_event.shot_speed_quality = str(shot.get("quality", "unknown"))
                previous_event.shot_speed_method = str(shot.get("method", "none"))
                if shot.get("available") and shot.get("speed_kmh") is not None:
                    completed_speed_kmh = float(shot["speed_kmh"])
                    completed_speed_method = str(shot.get("method", "none"))
            completed_shot_speed_by_event_frame[int(event.frame)] = completed_speed_kmh
            completed_shot_method_by_event_frame[int(event.frame)] = completed_speed_method
            if completed_speed_kmh is not None:
                last_valid_shot_speed_kmh = completed_speed_kmh
                last_valid_shot_method = completed_speed_method
            last_confirmed_event_by_rally[rally_key] = event

        draw_trail(frame, near_trail, args.near_color, args.trail_thickness, args.dash_len, args.gap_len, args.smooth_steps)
        draw_trail(frame, far_trail, args.far_color, args.trail_thickness, args.dash_len, args.gap_len, args.smooth_steps)

        ball_m_for_map = None
        if ball_draw_sample is None:
            # Do not join two unrelated flight segments over a detector gap or
            # a player-box veto on the mini court.
            ball_map_hist.clear()
        if args.draw_ball_on_minimap and ball_draw_sample is not None:
            # Use the smoothed point directly.  Clamping the source point to
            # the floor quad makes airborne hits stick to the far baseline.
            ball_m_for_map = to_court_m((ball_draw_sample.x, ball_draw_sample.y), homography)
            if ball_m_for_map is not None:
                ball_m_for_map = (
                    float(np.clip(ball_m_for_map[0] + ball_x_shift, 0.0, args.court_width_m)),
                    float(np.clip(ball_m_for_map[1], 0.0, args.court_length_m)),
                )

        near_m_for_map = to_court_m(near_draw_pt, homography) if near_draw_pt is not None else None
        far_m_for_map = to_court_m(far_draw_pt, homography) if far_draw_pt is not None else None
        near_m_for_map = enforce_half_court(near_m_for_map, "near", args.court_width_m, args.court_length_m)
        far_m_for_map = enforce_half_court(
            far_m_for_map,
            "far",
            args.court_width_m,
            args.court_length_m,
            far_half_expand=args.far_half_expand,
        )

        # Heatmap samples deliberately use the current detector assignment,
        # never ``*_draw_pt``: the latter may be a held or optical-flow point
        # during a detector dropout.  Reject a side mismatch before the
        # Mini Court folding helper can turn it into a believable false spot.
        near_heatmap_m = None
        far_heatmap_m = None
        half_court_m = args.court_length_m * 0.5
        side_tolerance_m = 0.35
        if near_player is not None:
            raw_near_m = to_court_m(near_player["foot"], homography)
            if raw_near_m is not None and raw_near_m[1] >= half_court_m - side_tolerance_m:
                near_heatmap_m = enforce_half_court(
                    raw_near_m,
                    "near",
                    args.court_width_m,
                    args.court_length_m,
                )
        if far_player is not None:
            raw_far_m = to_court_m(far_player["foot"], homography)
            if raw_far_m is not None and raw_far_m[1] <= half_court_m + side_tolerance_m:
                far_heatmap_m = enforce_half_court(
                    raw_far_m,
                    "far",
                    args.court_width_m,
                    args.court_length_m,
                    far_half_expand=args.far_half_expand,
                )
        if args.draw_player_heatmap:
            sample_interval = max(1, int(args.heatmap_sample_interval))
            max_gap = max(
                1,
                int(args.heatmap_max_gap_frames),
                2 * max(1, int(args.detect_interval)),
            )
            if (
                near_heatmap_m is not None
                and (
                    last_near_heatmap_frame is None
                    or frame_idx - last_near_heatmap_frame >= sample_interval
                )
            ):
                sample_weight = 1 if last_near_heatmap_frame is None else min(
                    max_gap, max(1, frame_idx - last_near_heatmap_frame)
                )
                if add_player_heatmap_sample(
                    near_heatmap,
                    near_heatmap_m,
                    args.court_width_m,
                    args.court_length_m,
                    weight=sample_weight,
                ):
                    last_near_heatmap_frame = frame_idx
            if (
                far_heatmap_m is not None
                and (
                    last_far_heatmap_frame is None
                    or frame_idx - last_far_heatmap_frame >= sample_interval
                )
            ):
                sample_weight = 1 if last_far_heatmap_frame is None else min(
                    max_gap, max(1, frame_idx - last_far_heatmap_frame)
                )
                if add_player_heatmap_sample(
                    far_heatmap,
                    far_heatmap_m,
                    args.court_width_m,
                    args.court_length_m,
                    weight=sample_weight,
                ):
                    last_far_heatmap_frame = frame_idx

        recent_hits = [
            event for event in hit_events
            if event.frame not in body_vetoed_hit_frames
            and event.frame <= frame_idx <= event.frame + args.hit_marker_duration
        ]
        draw_mini_court(
            frame,
            far_m_for_map,
            near_m_for_map,
            ball_m_for_map,
            far_map_hist,
            near_map_hist,
            ball_map_hist,
            args.court_width_m,
            args.court_length_m,
            recent_hits=recent_hits,
            frame_idx=frame_idx,
            ball_source=str(getattr(ball_draw_sample, "source", "model"))
            if ball_draw_sample is not None else "model",
            far_heatmap=far_heatmap,
            near_heatmap=near_heatmap,
            draw_heatmap=args.draw_player_heatmap,
            heatmap_alpha=args.heatmap_alpha,
            heatmap_sigma_cells=args.heatmap_sigma_cells,
            far_color=args.far_color,
            near_color=args.near_color,
        )

        if near_player_draw is not None:
            draw_player(frame, near_player_draw, ZH_BOTTOM, args.near_color, args.draw_foot_point, args.draw_id, args.draw_pose)
        if far_player_draw is not None:
            draw_player(frame, far_player_draw, ZH_TOP, args.far_color, args.draw_foot_point, args.draw_id, args.draw_pose)

        # Draw the ball layer last so the neon trajectory and hit ring remain
        # visible when they cross a pose skeleton, racket, or player box.
        if args.draw_ball_trajectory:
            draw_ball_trajectory(frame, ball_trail, ball_draw_sample, args.ball_trail_thickness)
        for event in hit_events:
            if (
                event.frame not in body_vetoed_hit_frames
                and event.frame <= frame_idx <= event.frame + args.hit_marker_duration
            ):
                draw_hit_marker(
                    frame,
                    event,
                    frame_idx,
                    live_hit_number_by_frame.get(event.frame, 0),
                    args.hit_marker_duration,
                    uncertain=getattr(event, "uncertain", False),
                    completed_shot_speed_kmh=completed_shot_speed_by_event_frame.get(
                        int(event.frame)
                    ),
                    completed_shot_method=completed_shot_method_by_event_frame.get(
                        int(event.frame), ""
                    ),
                )

        # Ball-derived rally labels are more meaningful than a player detector
        # dropout.  Keep the old player fallback when no ball segment exists.
        ball_rally = rally_labels[frame_idx] if frame_idx < len(rally_labels) else 0
        rally_hits = 0
        if ball_rally > 0:
            if last_ball_rally != ball_rally:
                near_stats.reset_rally()
                far_stats.reset_rally()
                last_ball_rally = ball_rally
                # A new rally starts with no completed flight speed, even if
                # the previous rally ended immediately before this frame.
                last_valid_shot_speed_kmh = None
                last_valid_shot_method = ""
            rally_idx = ball_rally
            rally_hits = live_hits_by_rally.get(int(ball_rally), 0)
        near_stats.rally_hits = rally_hits
        far_stats.rally_hits = rally_hits

        ui_fps = frame_idx / max(time.perf_counter() - t_start, 1e-6)
        draw_stats_panel(
            frame,
            rally_idx,
            far_stats,
            near_stats,
            pipeline_fps=ui_fps,
            rally_hit_count=rally_hits,
            total_hit_count=video_hit_total,
            last_shot_speed_kmh=last_valid_shot_speed_kmh,
            last_shot_method=last_valid_shot_method,
        )

        # Optional debug: draw selected court quadrilateral.
        if args.draw_court_polygon:
            quad = court_quad.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [quad], True, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(frame)
        prev_gray = cur_gray
        frame_idx += 1

        if frame_idx % 200 == 0:
            print(f"[INFO] progress {frame_idx}/{total_frames}")

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

    cap.release()
    writer.release()
    # Do not leak a render-pass body veto into the sidecar/exported event set.
    if body_vetoed_hit_frames:
        hit_events = [
            event for event in hit_events
            if int(event.frame) not in body_vetoed_hit_frames
        ]
    # The renderer does not consume hitter identity, so apply the optional
    # rally-consistency check only after the final render-pass veto.  This
    # keeps the exported sidecar auditable: every inferred side is bracketed
    # by events that actually survive into the output.
    hitter_attribution_stats = infer_single_unknown_hitter_from_rally_anchors(
        hit_events,
    )
    write_ball_tracking_csv(
        tracking_csv,
        ball_dict,
        render_ball_track,
        hit_events,
        rally_labels,
        detector_mode=active_detector,
        classical_track_ids=classical_track_ids,
    )
    if frame_idx > 0:
        detect_ratio = detect_calls / float(frame_idx)
        print(f"[DONE] detect_calls={detect_calls}, detect_ratio={detect_ratio:.3f}")
    print(
        "[INFO] hitter_attribution="
        + ",".join(
            f"{key}:{value}"
            for key, value in sorted(hitter_attribution_stats.items())
        )
    )
    print("[DONE] finished")
    print(f"[DONE] output video: {args.output_path}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Player trajectory analytics overlay")
    p.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--ball_csv", type=str, default="")
    p.add_argument("--ball_csv_secondary", type=str, default="",
                   help="Second half-window TrackNet CSV for temporal-phase fusion")
    p.add_argument("--yolo_model", type=str, default="yolov8s-pose.pt")
    p.add_argument("--skip_player_detector", action="store_true",
                   help="Render ball/rally overlays without loading YOLO (CPU fallback)")
    p.add_argument("--tracker_cfg", type=str, default="bytetrack.yaml")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf_thres", type=float, default=0.18)
    p.add_argument("--detect_interval", type=int, default=1, help="Run detector every N frames (1=full precision)")
    p.add_argument("--detect_ball_frames_only", action="store_true",
                   help="On CPU, run YOLO only on frames with a candidate ball point")
    p.add_argument("--half", dest="half", action="store_true")
    p.add_argument("--no_half", dest="half", action="store_false")
    p.set_defaults(half=True)
    p.add_argument("--min_box_w", type=float, default=10.0)
    p.add_argument("--min_box_h", type=float, default=20.0)
    p.add_argument("--detect_on_court_roi", dest="detect_on_court_roi", action="store_true")
    p.add_argument("--no_detect_on_court_roi", dest="detect_on_court_roi", action="store_false")
    p.set_defaults(detect_on_court_roi=True)
    p.add_argument("--detect_roi_pad_x", type=int, default=28)
    p.add_argument("--detect_roi_pad_top", type=int, default=84)
    p.add_argument("--detect_roi_pad_bottom", type=int, default=18)
    p.add_argument("--top_inner_ratio", type=float, default=0.020, help="Suppress top off-court candidates")
    p.add_argument("--top_inner_keep_dist", type=float, default=100.0, help="Keep top candidate if close to previous far point")
    p.add_argument("--ankle_min_conf", type=float, default=0.22)
    p.add_argument("--far_fallback_lift_ratio", type=float, default=0.22, help="Lift far fallback foot when ankles are missing")
    p.add_argument("--far_zone_hint_margin", type=float, default=12.0)
    p.add_argument("--far_lock_max_jump", type=float, default=110.0)
    p.add_argument("--far_max_up_jump", type=float, default=30.0)
    p.add_argument("--side_band_px", type=float, default=16.0, help="Midline band for near/far side lock")
    p.add_argument("--far_half_expand", type=float, default=1.35, help="Expand far-side mini-court motion range")
    p.add_argument("--judge_hist_len", type=int, default=90)
    p.add_argument("--judge_min_frames", type=int, default=42)
    p.add_argument("--judge_top_zone_px", type=int, default=18)
    p.add_argument("--judge_top_ratio", type=float, default=0.88)
    p.add_argument("--judge_static_move_px", type=float, default=30.0)

    p.add_argument("--court_points", type=str, default="")
    p.add_argument("--select_court_points", dest="select_court_points", action="store_true")
    p.add_argument("--no_select_court_points", dest="select_court_points", action="store_false")
    p.set_defaults(select_court_points=True)
    p.add_argument("--court_width_m", type=float, default=6.1)
    p.add_argument("--court_length_m", type=float, default=13.4)

    p.add_argument("--trail_len", type=int, default=50)
    p.add_argument("--minimap_trail_len", type=int, default=120)
    # Seven seconds of history makes a complete rally look like a yellow
    # scribble even when every point is valid.  Roughly 2.4 seconds keeps the
    # current flight and its hit/re-entry context legible.
    p.add_argument("--ball_trail_len", type=int, default=60, help="Ball trail length in frames")
    p.add_argument("--ball_trail_thickness", type=int, default=5)
    p.add_argument("--trail_thickness", type=int, default=3)
    p.add_argument("--dash_len", type=int, default=12)
    p.add_argument("--gap_len", type=int, default=10)
    p.add_argument("--smooth_steps", type=int, default=12)
    p.add_argument("--near_color", type=lambda s: tuple(int(x) for x in s.split(",")), default=(255, 110, 210))
    p.add_argument("--far_color", type=lambda s: tuple(int(x) for x in s.split(",")), default=(0, 180, 255))
    p.add_argument("--draw_player_heatmap", dest="draw_player_heatmap", action="store_true",
                   help="Overlay real player-foot occupancy heatmaps on the Mini Court")
    p.add_argument("--no_draw_player_heatmap", dest="draw_player_heatmap", action="store_false",
                   help="Disable Mini Court player heatmaps")
    p.set_defaults(draw_player_heatmap=True)
    p.add_argument("--heatmap_cell_m", type=float, default=0.20,
                   help="Mini Court heatmap cell size in metres")
    p.add_argument("--heatmap_sigma_cells", type=float, default=1.3,
                   help="Gaussian smoothing radius in heatmap cells")
    p.add_argument("--heatmap_alpha", type=float, default=0.34,
                   help="Maximum Mini Court heatmap opacity")
    p.add_argument("--heatmap_sample_interval", type=int, default=2,
                   help="Use one real player-foot heatmap sample every N frames")
    p.add_argument("--heatmap_max_gap_frames", type=int, default=6,
                   help="Maximum temporal weight assigned after a heatmap detection gap")

    p.add_argument("--draw_id", action="store_true")
    p.add_argument("--draw_pose", dest="draw_pose", action="store_true")
    p.add_argument("--no_draw_pose", dest="draw_pose", action="store_false")
    p.set_defaults(draw_pose=True)
    p.add_argument("--draw_pose_on_skipped", dest="draw_pose_on_skipped", action="store_true")
    p.add_argument("--no_draw_pose_on_skipped", dest="draw_pose_on_skipped", action="store_false")
    p.set_defaults(draw_pose_on_skipped=True)
    p.add_argument("--skip_pose_max_age", type=int, default=5, help="Max skipped-frame age to reuse cached pose")

    p.add_argument("--draw_foot_point", dest="draw_foot_point", action="store_true")
    p.add_argument("--no_draw_foot_point", dest="draw_foot_point", action="store_false")
    p.set_defaults(draw_foot_point=True)

    p.add_argument("--draw_ball_on_minimap", dest="draw_ball_on_minimap", action="store_true")
    p.add_argument("--no_draw_ball_on_minimap", dest="draw_ball_on_minimap", action="store_false")
    p.set_defaults(draw_ball_on_minimap=True)
    p.add_argument("--draw_ball_trajectory", dest="draw_ball_trajectory", action="store_true")
    p.add_argument("--no_draw_ball_trajectory", dest="draw_ball_trajectory", action="store_false")
    p.set_defaults(draw_ball_trajectory=True)
    p.add_argument("--offscreen_bridge", dest="offscreen_bridge", action="store_true",
                   help="Draw a faint render-only bridge when the shuttle crosses the top edge")
    p.add_argument("--no_offscreen_bridge", dest="offscreen_bridge", action="store_false",
                   help="Disable render-only top-edge flight bridges")
    p.set_defaults(offscreen_bridge=True)
    p.add_argument("--offscreen_bridge_max_gap", type=int, default=40,
                   help="Maximum missing-frame span for a top-edge render bridge")
    p.add_argument("--offscreen_bridge_min_gap", type=int, default=4,
                   help="Minimum missing-frame span eligible for a top-edge bridge")
    p.add_argument("--offscreen_bridge_top_y", type=float, default=30.0,
                   help="Image y threshold for a top-edge bridge anchor")
    p.add_argument("--offscreen_bridge_velocity_window", type=int, default=4,
                   help="Real anchors used to estimate bridge endpoint velocity")
    p.add_argument("--offscreen_bridge_min_speed", type=float, default=5.0,
                   help="Minimum endpoint speed for a top-edge bridge")
    p.add_argument("--offscreen_bridge_min_vertical_speed", type=float, default=4.0,
                   help="Required upward/downward endpoint y speed for a bridge")
    p.add_argument("--offscreen_bridge_min_horizontal_speed", type=float, default=8.0,
                   help="Strong horizontal speed threshold for direction consistency")
    p.add_argument("--offscreen_bridge_confidence", type=float, default=0.12,
                   help="Low confidence written for render-only bridge points")
    p.add_argument("--offscreen_bridge_mixed_edge_bridge",
                   dest="offscreen_bridge_mixed_edge_bridge",
                   action="store_true",
                   help="Allow the strict ordinary-to-top-edge render bridge")
    p.add_argument("--no_offscreen_bridge_mixed_edge_bridge",
                   dest="offscreen_bridge_mixed_edge_bridge",
                   action="store_false",
                   help="Disable ordinary-to-top-edge render bridges")
    p.set_defaults(offscreen_bridge_mixed_edge_bridge=True)
    p.add_argument("--offscreen_bridge_mixed_support_frames", type=int, default=3,
                   help="Consecutive monotone anchors required for mixed edge bridge")
    p.add_argument("--offscreen_bridge_mixed_speed_ratio", type=float, default=0.25,
                   help="Minimum incoming/outgoing speed ratio for mixed bridge")
    p.add_argument("--classical_ball_fallback", dest="classical_ball_fallback", action="store_true",
                   help="Auto-use conservative CPU motion detection when TrackNet coverage is low")
    p.add_argument("--no_classical_ball_fallback", dest="classical_ball_fallback", action="store_false",
                   help="Disable the low-coverage classical ball detector")
    p.set_defaults(classical_ball_fallback=True)
    p.add_argument("--classical_trigger_coverage", type=float, default=0.35,
                   help="Fallback trigger based on filtered real TrackNet coverage")
    p.add_argument("--classical_trigger_gap_frames", type=int, default=12,
                   help="Also run classical fusion when any local real-detection gap reaches this length")
    p.add_argument("--classical_min_observations", type=int, default=30,
                   help="Minimum strict classical points required to replace TrackNet")
    p.add_argument("--classical_min_tracks", type=int, default=2,
                   help="Minimum strict classical flight segments required for replacement")
    p.add_argument("--classical_top_extension", type=float, default=120.0,
                   help="Conservative classical ROI above the far baseline")
    p.add_argument("--classical_high_flight_top_extension", type=float, default=480.0,
                   help="Second-pass ROI used to recover high clears near the image top")
    p.add_argument("--classical_high_flight_recall", dest="classical_high_flight_recall",
                   action="store_true", help="Enable the precision-gated high-clear pass")
    p.add_argument("--no_classical_high_flight_recall", dest="classical_high_flight_recall",
                   action="store_false", help="Disable the high-clear second pass")
    p.set_defaults(classical_high_flight_recall=True)
    p.add_argument("--ball_fusion_agreement_px", type=float, default=20.0,
                   help="Keep official TrackNet when both real detectors agree within this radius")
    p.add_argument("--ball_fusion_bridge_frames", type=int, default=6,
                   help="Maximum two-sided search used to arbitrate detector conflicts")
    p.add_argument("--tracknet_phase_agreement_px", type=float, default=25.0,
                   help="Maximum coordinate difference treated as two-phase agreement")
    p.add_argument("--tracknet_phase_bridge_frames", type=int, default=6,
                   help="Agreement-anchor search around a two-phase conflict group")
    p.add_argument("--tracknet_phase_bridge_residual_px", type=float, default=60.0,
                   help="Maximum residual for selecting one phase along an anchor bridge")
    p.add_argument("--tracknet_phase_pose_imgsz", type=int, default=960,
                   help="Sparse pose resolution for suspicious two-phase observations")
    p.add_argument("--tracknet_phase_player_box_margin", type=float, default=8.0,
                   help="Body-box expansion used by the pre-Kalman model-point veto")
    p.add_argument("--tracknet_phase_contact_protect_px", type=float, default=38.0,
                   help="Protect model points close to a pose-estimated racket contact")
    p.add_argument("--tracknet_phase_edge_protect_y", type=float, default=30.0,
                   help="Preserve model points at the top image edge during body/phase veto")
    p.add_argument("--tracknet_phase_top_interstitial_edge_y", type=float, default=180.0,
                   help="Broad top-edge envelope used to find interior branches between edge flights")
    p.add_argument("--tracknet_phase_top_interstitial_residual_px", type=float, default=110.0,
                   help="Minimum bridge residual for a top-edge interstitial branch veto")
    p.add_argument("--tracknet_phase_top_interstitial_gap_frames", type=int, default=40,
                   help="Maximum unseen span considered for a top-edge interstitial branch")
    p.add_argument("--tracknet_phase_top_interstitial_anchor_frames", type=int, default=3,
                   help="Consecutive monotone edge anchors required on each side")
    p.add_argument("--tracknet_top_branch_veto", dest="tracknet_top_branch_veto",
                   action="store_true",
                   help="Veto short post-top model branches that fall into a player box")
    p.add_argument("--no_tracknet_top_branch_veto", dest="tracknet_top_branch_veto",
                   action="store_false")
    p.set_defaults(tracknet_top_branch_veto=True)
    p.add_argument("--tracknet_top_branch_max_frames", type=int, default=8,
                   help="Frames after a top-edge anchor considered for body-branch veto")
    p.add_argument("--tracknet_top_branch_body_y", type=float, default=300.0,
                   help="Minimum y for a post-top branch to receive the wider body veto")
    p.add_argument("--tracknet_top_branch_player_box_margin", type=float, default=60.0,
                   help="Expanded player-box margin for post-top false branches")
    p.add_argument("--tracknet_phase_max_branch_frames", type=int, default=3,
                   help="Maximum short phase branch eligible for kinematic rejection")
    p.add_argument("--tracknet_phase_player_veto", dest="tracknet_phase_player_veto",
                   action="store_true",
                   help="Run sparse pose boxes before Kalman on suspicious phase points")
    p.add_argument("--no_tracknet_phase_player_veto", dest="tracknet_phase_player_veto",
                   action="store_false",
                   help="Disable sparse pre-Kalman body veto for phase fusion")
    p.set_defaults(tracknet_phase_player_veto=True)
    p.add_argument("--ball_max_interp_gap", type=int, default=8)
    p.add_argument("--ball_max_predict_gap", type=int, default=3)
    p.add_argument("--ball_process_noise", type=float, default=700.0)
    p.add_argument("--hit_min_separation", type=int, default=8)
    p.add_argument("--hit_min_speed_px", type=float, default=7.0)
    p.add_argument("--hit_tracklet_gap_frames", type=int, default=48,
                   help="Maximum gap for a low-altitude classical endpoint hit candidate")
    p.add_argument("--hit_marker_duration", type=int, default=15)
    p.add_argument("--count_uncertain_hits", dest="count_uncertain_hits", action="store_true",
                   help="Count evidence-validated lower-confidence HIT? events")
    p.add_argument("--no_count_uncertain_hits", dest="count_uncertain_hits", action="store_false",
                   help="Exclude lower-confidence HIT? events from counters and speed boundaries")
    p.set_defaults(count_uncertain_hits=True)
    p.add_argument("--shot_speed_min_coverage", type=float, default=0.15,
                   help="Minimum observed fraction needed for a projected shot-speed estimate")
    p.add_argument("--shot_speed_min_real_points", type=int, default=8,
                   help="Minimum model/classical ball points needed for a shot-speed estimate")
    p.add_argument("--shot_speed_min_segments", type=int, default=8,
                   help="Minimum adjacent trajectory segments needed for a shot-speed estimate")
    p.add_argument("--shot_speed_max_segment_mps", type=float, default=100.0,
                   help="Reject one projected ball segment faster than this as a geometry outlier")
    p.add_argument("--shot_speed_max_average_mps", type=float, default=40.0,
                   help="Reject an implausible court-plane average shuttle speed")
    p.add_argument("--shot_speed_court_margin_m", type=float, default=6.0,
                   help="Court-plane margin allowed before high-flight projections are excluded")
    p.add_argument("--shot_speed_perspective_proxy", dest="shot_speed_perspective_proxy",
                   action="store_true",
                   help="Use capped perspective motion when a high flight cannot map to the court floor")
    p.add_argument("--no_shot_speed_perspective_proxy", dest="shot_speed_perspective_proxy",
                   action="store_false",
                   help="Disable high-flight perspective speed estimates")
    p.set_defaults(shot_speed_perspective_proxy=True)
    p.add_argument("--shot_speed_proxy_min_coverage", type=float, default=0.20,
                   help="Minimum observed fraction for a capped perspective speed estimate")
    p.add_argument("--shot_speed_proxy_min_real_points", type=int, default=8,
                   help="Minimum real ball points for a capped perspective speed estimate")
    p.add_argument("--shot_speed_proxy_min_segments", type=int, default=8,
                   help="Minimum adjacent segments for a capped perspective speed estimate")
    p.add_argument("--shot_speed_proxy_max_scale_m_per_px", type=float, default=0.055,
                   help="Maximum local metre-per-pixel scale used above the far baseline")
    p.add_argument("--shot_speed_proxy_max_average_mps", type=float, default=55.0,
                   help="Reject an implausible capped-perspective average shuttle speed")
    p.add_argument("--audio_hit_detection", dest="audio_hit_detection", action="store_true",
                   help="Fuse broadband racket impulses with visual hit candidates")
    p.add_argument("--no_audio_hit_detection", dest="audio_hit_detection", action="store_false",
                   help="Disable audio-assisted hit timing")
    p.set_defaults(audio_hit_detection=True)
    p.add_argument("--audio_hit_score_threshold", type=float, default=1.25,
                   help="Weak audio threshold; weak cues require visual support")
    p.add_argument("--audio_hit_strong_score", type=float, default=3.0,
                   help="Diagnostic threshold for strong audio impulses; trajectory evidence is still required")
    p.add_argument("--audio_hit_nms_frames", type=float, default=8.0)
    p.add_argument("--audio_hit_visual_tolerance", type=int, default=8,
                   help="Audio/video frame tolerance; accommodates the decoder's short impulse delay")
    p.add_argument("--audio_hit_ball_search_frames", type=int, default=18)
    p.add_argument("--audio_hit_post_roll_seconds", type=float, default=1.35,
                   help="Stop accepting crowd/cut transients after the last ball observation")
    p.add_argument("--audio_hit_pose_fallback", dest="audio_hit_pose_fallback", action="store_true",
                   help="Use sparse pose inference for hitter association/body veto (never pose-only localization)")
    p.add_argument("--no_audio_hit_pose_fallback", dest="audio_hit_pose_fallback", action="store_false")
    p.set_defaults(audio_hit_pose_fallback=True)
    p.add_argument("--audio_hit_pose_batch_size", type=int, default=8)
    p.add_argument("--audio_hit_pose_imgsz", type=int, default=640)
    p.add_argument("--ball_rally_gap_frames", type=int, default=0,
                   help="Ball-gap rally boundary (0: automatic, five seconds)")
    p.add_argument("--ball_min_rally_points", type=int, default=3,
                   help="Minimum model/interpolated points before a gap starts a rally")
    p.add_argument("--ball_side_padding", type=float, default=40.0)
    p.add_argument("--ball_top_extension", type=float, default=480.0,
                   help="Airborne-ball extension above the far baseline (pixels)")
    p.add_argument("--ball_bottom_padding", type=float, default=30.0)
    p.add_argument("--ball_player_veto_margin", type=float, default=8.0,
                   help="Reject TrackNet points inside a player box by this margin")
    p.add_argument("--hit_contact_protect_px", type=float, default=45.0,
                   help="Allow a visual hit inside a player box when it is near a pose wrist/contact")
    p.add_argument("--ball_player_veto_foot_radius", type=float, default=0.0,
                   help="Optional model-only foot-radius veto (0 disables; body-box veto remains)")
    p.add_argument("--ball_static_window", type=int, default=20,
                   help="Raw-model window for suppressing a truly static response")
    p.add_argument("--ball_static_disp", type=float, default=8.0,
                   help="Maximum raw-model spread of a truly static response in pixels")
    p.add_argument("--ball_static_lock_window", type=int, default=16,
                   help="Mixed-source static-lock window used to remove stale player/net responses")
    p.add_argument("--ball_static_lock_disp", type=float, default=35.0,
                   help="Maximum mixed-source static-lock spread in pixels")
    p.add_argument("--ball_static_lock_timeout_window", type=int, default=4,
                   help="Short raw-model lock timeout (frames; 0 disables)")
    p.add_argument("--ball_static_lock_timeout_disp", type=float, default=10.0,
                   help="Maximum spread for the short raw-model lock timeout")
    p.add_argument("--ball_static_lock_min_y", type=float, default=80.0,
                   help="Do not apply short lock timeout above this image y")
    p.add_argument("--ball_static_lock_preroll", type=int, default=2,
                   help="Frames before a short lock to suppress when converging")
    p.add_argument("--scene_cut_threshold", type=float, default=28.0,
                   help="Low-resolution mean-difference threshold for hard cuts")
    p.add_argument("--scene_cuts", type=str, default="",
                   help="Optional comma-separated hard-cut frame indices (overrides auto detection)")
    p.add_argument("--stop_ball_at_first_scene_cut", dest="stop_ball_at_first_scene_cut",
                   action="store_true",
                   help="Stop ball analysis when fixed court calibration becomes invalid at the first hard cut")
    p.add_argument("--continue_ball_after_scene_cut", dest="stop_ball_at_first_scene_cut",
                   action="store_false",
                   help="Continue ball analysis after cuts when every shot uses the same court calibration")
    p.set_defaults(stop_ball_at_first_scene_cut=True)

    p.add_argument("--draw_court_polygon", action="store_true")

    p.add_argument("--new_rally_missing_frames", type=int, default=48)
    p.add_argument("--hold_missing_frames", type=int, default=10)
    p.add_argument("--reset_missing_frames", type=int, default=36)
    p.add_argument("--max_flow_move", type=float, default=45.0)
    p.add_argument("--max_frames", type=int, default=0)
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)
