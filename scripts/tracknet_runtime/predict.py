import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from test import predict_location, get_ensemble_weight, generate_inpaint_mask
from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from utils.general import *


def _count_decodable_frames(video_file):
    """Return the number of frames that OpenCV can actually decode.

    A few MP4 broadcasts in the wild advertise a larger ``FRAME_COUNT`` than
    the number of packets which can be read (for example, ``output.mp4``
    reports 1052 while its decoder stops at frame 1001).  The streaming
    dataset uses the metadata value for temporal-buffer sizing, so reconciling
    it before inference prevents the overlap tail branch from being skipped
    and keeps the CSV/video frame indices aligned.  ``grab`` avoids allocating
    full images and is considerably cheaper than a second inference pass.
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return 0
    count = 0
    while cap.grab():
        count += 1
    cap.release()
    return count


def _reconcile_stream_dataset_length(dataset, video_file):
    """Fix a stale container frame count on a streaming dataset in-place."""
    hinted = int(getattr(dataset, "video_len", 0) or 0)
    actual = _count_decodable_frames(video_file)
    if actual > 0 and actual != hinted:
        print(
            f"[WARN] container frame count {hinted} != decodable frames {actual}; "
            "using the decodable length for temporal buffering",
            flush=True,
        )
        dataset.video_len = actual
    return actual if actual > 0 else hinted


def _canonicalize_prediction_dict(pred_dict):
    """Sort and de-duplicate frame predictions before CSV/video writing.

    ``Video_IterableDataset`` pads its final sliding window.  On very short
    clips (and on containers whose advertised frame count is stale) that can
    yield a repeated frame id at the end of a batch.  The model output is
    still useful, but downstream writers index coordinates by decoded frame
    number; duplicate rows would shift every later coordinate.  Keep one
    record per frame, preferring a visible record over a padded miss, and fill
    genuinely absent ids with an explicit miss.
    """
    frames = list(pred_dict.get("Frame", []))
    if not frames:
        return pred_dict
    try:
        frame_ids = [int(frame) for frame in frames]
    except (TypeError, ValueError):
        return pred_dict
    n = max(frame_ids, default=-1) + 1
    if n <= 0:
        return pred_dict

    visibility = list(pred_dict.get("Visibility", []))
    selected = {}
    for idx, frame_id in enumerate(frame_ids):
        if frame_id < 0:
            continue
        current_vis = 0
        if idx < len(visibility):
            try:
                current_vis = int(float(visibility[idx]))
            except (TypeError, ValueError):
                current_vis = 0
        previous_idx = selected.get(frame_id)
        if previous_idx is None:
            selected[frame_id] = idx
            continue
        previous_vis = 0
        if previous_idx < len(visibility):
            try:
                previous_vis = int(float(visibility[previous_idx]))
            except (TypeError, ValueError):
                previous_vis = 0
        if current_vis > previous_vis:
            selected[frame_id] = idx

    canonical = dict(pred_dict)
    canonical["Frame"] = list(range(n))
    for key, default in (("X", 0), ("Y", 0), ("Visibility", 0)):
        values = list(pred_dict.get(key, []))
        canonical[key] = [
            values[selected[frame_id]] if frame_id in selected and selected[frame_id] < len(values) else default
            for frame_id in range(n)
        ]
    # Keep an aligned inpaint mask when one is present.  Empty masks are a
    # valid representation in the TrackNet-only path and should stay empty.
    mask = pred_dict.get("Inpaint_Mask")
    if mask is not None and len(mask) == len(frame_ids):
        values = list(mask)
        canonical["Inpaint_Mask"] = [
            values[selected[frame_id]] if frame_id in selected and selected[frame_id] < len(values) else 0
            for frame_id in range(n)
        ]
    return canonical


def suppress_static_prediction_locks(
    pred_dict,
    window=4,
    disp=10.0,
    min_y=80.0,
    preroll=2,
):
    """Veto short, compact TrackNet response locks before overlay fusion.

    A domain-mismatched heatmap can settle on a player's shoe or shirt for
    dozens of frames.  The old long-window filter only recognized the lock
    after it had already contaminated the causal state.  This pass is
    deliberately conservative: at least eight consecutive visible points
    below the top-flight zone must remain within one <=10 px fixed-centre
    corridor and have a very short cumulative path.  A couple of compact
    lead-in frames may be removed as well.  Coordinates are zeroed (rather
    than merely hidden in the renderer), so a half-window phase cannot
    reintroduce the stale point as a secondary observation.

    ``pred_dict`` is copied; non-coordinate metadata is preserved.  Set
    ``window <= 0`` or ``disp <= 0`` to disable the pass.
    """
    result = dict(pred_dict or {})
    frames = list(result.get("Frame", []))
    xs = list(result.get("X", []))
    ys = list(result.get("Y", []))
    vis = list(result.get("Visibility", []))
    count = min(len(frames), len(xs), len(ys), len(vis))
    timeout = max(0, int(window))
    radius = max(0.0, float(disp))
    y_limit = float(min_y)
    if count <= 0 or timeout < 2 or radius <= 0:
        return result

    try:
        frame_ids = [int(float(frames[i])) for i in range(count)]
    except (TypeError, ValueError):
        return result
    candidate = np.zeros(count, dtype=bool)
    for i in range(count):
        try:
            candidate[i] = (
                int(float(vis[i])) > 0
                and float(xs[i]) > 0
                and float(ys[i]) >= y_limit
                and np.isfinite([float(xs[i]), float(ys[i])]).all()
            )
        except (TypeError, ValueError):
            candidate[i] = False

    # Split on frame-id holes.  A missing packet is a separate flight and
    # should never be used to extend a static lock across a gap.
    runs = []
    run_start = None
    for i, flag in enumerate(candidate):
        contiguous = (
            run_start is not None
            and i > 0
            and frame_ids[i] == frame_ids[i - 1] + 1
        )
        if flag and run_start is None:
            run_start = i
        elif flag and run_start is not None and not contiguous:
            runs.append((run_start, i - 1))
            run_start = i
        elif not flag and run_start is not None:
            runs.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, count - 1))

    suppressed = set()
    # Four locally compact points are common around a real high-clear apex.
    # Confirm a lock over at least eight frames and one *fixed* centre before
    # removing it.  The old implementation then extended by per-frame step,
    # which could follow a smoothly moving shuttle indefinitely even though
    # it had travelled hundreds of pixels away from the original lock.
    confirm_window = max(8, 2 * timeout)
    confirm_radius = max(radius, 1.5 * radius)
    confirm_path = max(12.0, 1.5 * radius)
    for run_start, run_end in runs:
        if run_end - run_start + 1 < confirm_window:
            continue
        scan = run_start
        while scan + confirm_window - 1 <= run_end:
            lock = None
            for begin in range(scan, run_end - confirm_window + 2):
                end = begin + confirm_window - 1
                points = np.asarray(
                    [[float(xs[j]), float(ys[j])] for j in range(begin, end + 1)],
                    dtype=np.float64,
                )
                center = np.median(points, axis=0)
                spread = np.linalg.norm(points - center, axis=1)
                path = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
                if float(np.max(spread)) > radius or path > confirm_path:
                    continue
                lock = (begin, end, center)
                break
            if lock is None:
                break

            lock_start, lock_end, center = lock
            # Include only a lead-in which remains in the same fixed lock
            # corridor.  A merely small step toward a moving shuttle is not
            # enough evidence to suppress it.
            for _ in range(max(0, int(preroll))):
                lead = lock_start - 1
                if lead < run_start:
                    break
                distance = float(np.linalg.norm(
                    np.asarray([float(xs[lead]), float(ys[lead])], dtype=np.float64)
                    - center
                ))
                if distance > confirm_radius:
                    break
                lock_start = lead

            # Extend only while observations stay close to the original
            # centre.  This catches a persistent shoe/banner response without
            # swallowing a genuine slow trajectory after its apex.
            while lock_end + 1 <= run_end:
                candidate_i = lock_end + 1
                distance = float(np.linalg.norm(
                    np.asarray(
                        [float(xs[candidate_i]), float(ys[candidate_i])],
                        dtype=np.float64,
                    ) - center
                ))
                if distance > confirm_radius:
                    break
                lock_end = candidate_i
            suppressed.update(range(lock_start, lock_end + 1))
            scan = lock_end + 1

    if suppressed:
        result["X"] = list(xs)
        result["Y"] = list(ys)
        result["Visibility"] = list(vis)
        for i in suppressed:
            result["X"][i] = 0
            result["Y"][i] = 0
            result["Visibility"][i] = 0
    return result


def _axis_starts(length, tile, overlap):
    """Return deterministic, overlapping crop starts for one image axis."""
    if length <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    last = length - tile
    starts = list(range(0, last + 1, step))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _axis_starts_in_range(length, tile, overlap, start, end):
    """Return overlapping tile starts which cover ``[start, end]``.

    The original tiled path always started its grid at ``(0, 0)``.  That is
    wasteful for a broadcast frame: most of the top row is a scoreboard and
    audience, where the TrackNet checkpoint produces strong but incorrect
    responses.  Anchoring the grid to the court/ROI both reduces inference
    work and prevents those responses from entering candidate selection.
    """
    length = int(max(0, length))
    tile = int(max(1, min(tile, length if length else tile)))
    lo = int(np.clip(start, 0, max(0, length - tile)))
    hi = int(np.clip(end, lo + tile, length))
    if hi - lo <= tile:
        return [lo]
    step = max(1, int(round(tile * (1.0 - float(overlap)))))
    last = hi - tile
    starts = list(range(lo, last + 1, step))
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def _make_tiles(width, height, overlap=0.25, region=None):
    """Build source-image tiles which preserve the TrackNet aspect ratio.

    TrackNet was trained at 512x288.  A 1920x1080 broadcast frame shrinks a
    shuttle to only a few pixels when resized as a whole.  1024x576 crops keep
    the same aspect ratio while giving the model roughly 2x more shuttle
    pixels.  Overlap avoids losing a shuttle at a crop boundary.
    """
    tile_w = min(width, WIDTH * 2)
    tile_h = min(height, HEIGHT * 2)
    if region is None:
        xs = _axis_starts(width, tile_w, overlap)
        ys = _axis_starts(height, tile_h, overlap)
    else:
        rx1, ry1, rx2, ry2 = [int(round(v)) for v in region]
        xs = _axis_starts_in_range(width, tile_w, overlap, rx1, rx2)
        ys = _axis_starts_in_range(height, tile_h, overlap, ry1, ry2)
    return [(x, y, min(x + tile_w, width), min(y + tile_h, height))
            for y in ys for x in xs]


def _expanded_court_polygon(points, top_extension=120.0, side_padding=40.0,
                            bottom_padding=30.0):
    """Extend a court quad conservatively for airborne shuttle detections.

    ``points`` are ordered TL, TR, BR, BL.  A literal floor quad is too tight
    for a shuttle above the far baseline, while an axis-aligned rectangle
    admits the purple banner at the top of this broadcast.  We therefore move
    the two baselines along the court axis and add only a small horizontal
    margin.  The result stays a convex quadrilateral and can be passed to
    :func:`cv2.pointPolygonTest`.
    """
    quad = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(quad) != 4 or not np.isfinite(quad).all():
        return None
    top = 0.5 * (quad[0] + quad[1])
    bottom = 0.5 * (quad[2] + quad[3])
    direction = bottom - top
    length = float(np.linalg.norm(direction))
    expanded = quad.copy()
    if length > 1e-3:
        unit = direction / length
        expanded[0] -= unit * float(max(0.0, top_extension))
        expanded[1] -= unit * float(max(0.0, top_extension))
        expanded[2] += unit * float(max(0.0, bottom_padding))
        expanded[3] += unit * float(max(0.0, bottom_padding))
    pad = float(max(0.0, side_padding))
    if pad:
        expanded[0, 0] -= pad
        expanded[3, 0] -= pad
        expanded[1, 0] += pad
        expanded[2, 0] += pad
    return expanded


def _roi_from_court(court_polygon, width, height, extra=0.0):
    """Get a clipped image ROI around an expanded court polygon."""
    if court_polygon is None:
        return None
    poly = np.asarray(court_polygon, dtype=np.float32).reshape(-1, 2)
    if len(poly) != 4 or not np.isfinite(poly).all():
        return None
    margin = max(0.0, float(extra))
    x1 = int(np.floor(np.min(poly[:, 0]) - margin))
    y1 = int(np.floor(np.min(poly[:, 1]) - margin))
    x2 = int(np.ceil(np.max(poly[:, 0]) + margin))
    y2 = int(np.ceil(np.max(poly[:, 1]) + margin))
    x1 = int(np.clip(x1, 0, max(0, int(width) - 1)))
    y1 = int(np.clip(y1, 0, max(0, int(height) - 1)))
    x2 = int(np.clip(x2, x1 + 1, int(width)))
    y2 = int(np.clip(y2, y1 + 1, int(height)))
    return x1, y1, x2, y2


def _read_video_median(video_file, max_sample_num):
    """Estimate a median background without retaining the complete video."""
    cap = cv2.VideoCapture(video_file)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    sample_count = min(max(8, max_sample_num), n)
    indices = np.linspace(0, n - 1, sample_count, dtype=np.int32)
    frames = []
    for frame_i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        return None
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def _decode_tile_peak(heatmap, tile, min_score, roi, court_polygon=None):
    """Decode one heatmap into a sub-pixel-ish source-image candidate.

    The stock implementation thresholds at 0.2 and takes the largest contour.
    On high-resolution footage that discards weak but useful peaks.  We retain
    the strongest local component, use an intensity-weighted centroid, and
    apply a generous court ROI before temporal filtering.
    """
    peak = float(np.max(heatmap))
    if peak < min_score:
        return None
    # A relative threshold keeps the component compact when the peak is weak,
    # while the absolute threshold prevents the almost-uniform background from
    # becoming a detection.
    threshold = max(min_score, peak - max(0.025, peak * 0.20))
    mask = (heatmap >= threshold).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        py, px = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        cx_i, cy_i = float(px), float(py)
    else:
        py, px = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        label = int(labels[py, px])
        if label == 0:
            label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        ys, xs = np.where(labels == label)
        weights = np.maximum(heatmap[ys, xs] - threshold, 1e-4)
        cx_i = float(np.average(xs, weights=weights))
        cy_i = float(np.average(ys, weights=weights))

    # Responses exactly on a crop edge are usually padding/scene-boundary
    # artifacts.  A real shuttle near an edge is still present in a neighbor
    # crop because crops overlap, so dropping these responses is safe.
    edge_margin = 8.0
    if (cx_i < edge_margin or cx_i > WIDTH - 1 - edge_margin or
            cy_i < edge_margin or cy_i > HEIGHT - 1 - edge_margin):
        return None

    x1, y1, x2, y2 = tile
    src_x = x1 + cx_i * (x2 - x1) / WIDTH
    src_y = y1 + cy_i * (y2 - y1) / HEIGHT
    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        if not (rx1 <= src_x <= rx2 and ry1 <= src_y <= ry2):
            return None
    if court_polygon is not None and cv2.pointPolygonTest(
            court_polygon.astype(np.float32), (float(src_x), float(src_y)), False) < 0:
        return None
    # Local contrast is useful for ranking a model response against its broad
    # background.  Keep the raw peak too, since it is easier to interpret.
    baseline = float(np.percentile(heatmap, 50))
    return (src_x, src_y, peak, max(0.0, peak - baseline))


def _cluster_candidates(candidates, radius=45.0):
    """Merge duplicate responses from overlapping tiles/windows."""
    clusters = []
    for cand in candidates:
        if isinstance(cand, dict):
            x = cand.get("x", 0.0)
            y = cand.get("y", 0.0)
            peak = cand.get("peak", cand.get("score", 0.0))
            contrast = cand.get("contrast", 0.0)
        else:
            if len(cand) < 4:
                continue
            x, y, peak, contrast = cand[:4]
        try:
            x, y, peak, contrast = float(x), float(y), float(peak), float(contrast)
        except (TypeError, ValueError):
            continue
        if not np.isfinite([x, y, peak, contrast]).all():
            continue
        best = None
        best_dist = radius
        for i, cluster in enumerate(clusters):
            dx = x - cluster["x"]
            dy = y - cluster["y"]
            dist = float((dx * dx + dy * dy) ** 0.5)
            if dist < best_dist:
                best, best_dist = i, dist
        if best is None:
            clusters.append({"x": x, "y": y, "score": peak,
                             "contrast": contrast, "count": 1})
        else:
            c = clusters[best]
            weight = max(0.01, peak - 0.05)
            old_weight = max(0.01, c["score"] - 0.05)
            c["x"] = (c["x"] * old_weight + x * weight) / (old_weight + weight)
            c["y"] = (c["y"] * old_weight + y * weight) / (old_weight + weight)
            c["score"] += peak
            c["contrast"] += contrast
            c["count"] += 1
    for cluster in clusters:
        count = max(1, int(cluster.get("count", 1)))
        cluster["mean_score"] = float(cluster["score"]) / count
        cluster["mean_contrast"] = float(cluster["contrast"]) / count
    return clusters


def _find_static_candidate_frames(candidates_by_frame, window=12, disp=36.0):
    """Pre-mark long, compact candidate runs as static background responses.

    Doing this before temporal selection is important: a causal filter can
    otherwise draw the first ``window`` frames of a scoreboard lock before it
    has enough history to recognize the lock.  The returned per-frame lists
    contain static centers (rather than one frame-wide mask), so a genuine
    shuttle is preserved when a frame also contains a banner response.
    """
    n = len(candidates_by_frame)
    window = max(2, int(window))
    disp = max(1.0, float(disp))
    if n < window:
        return [[] for _ in range(n)]
    clustered = [_cluster_candidates(raw) for raw in candidates_by_frame]
    centers_by_frame = [[] for _ in range(n)]
    for start in range(0, n - window + 1):
        base = clustered[start]
        if not base:
            continue
        for candidate in base:
            center = np.array([candidate["x"], candidate["y"]], dtype=np.float32)
            points = [center]
            ok = True
            for frame_i in range(start + 1, start + window):
                nearby = [
                    np.array([other["x"], other["y"]], dtype=np.float32)
                    for other in clustered[frame_i]
                    if np.linalg.norm(np.array([other["x"], other["y"]], dtype=np.float32) - center) <= disp
                ]
                if not nearby:
                    ok = False
                    break
                # Use the closest response so a second unrelated blob cannot
                # make the run look static.
                points.append(min(nearby, key=lambda p: float(np.linalg.norm(p - center))))
            if not ok:
                continue
            points_arr = np.asarray(points, dtype=np.float32)
            median = np.median(points_arr, axis=0)
            if float(np.max(np.linalg.norm(points_arr - median, axis=1))) <= disp:
                for frame_i in range(start, start + window):
                    center_tuple = (float(median[0]), float(median[1]))
                    if not any(
                        float(np.linalg.norm(np.asarray(old, dtype=np.float32) - median)) <= disp * 0.5
                        for old in centers_by_frame[frame_i]
                    ):
                        centers_by_frame[frame_i].append(center_tuple)
                # Keep scanning: overlapping windows extend the mask over the
                # complete lock, including frames before/after the first hit.
                break
    return centers_by_frame


def _select_temporal_track(candidates_by_frame, width, height, min_score,
                           static_window=None, static_disp=None,
                           persistence_window=None, max_gap=None,
                           static_cooldown=None):
    """Choose a temporally coherent response and reject implausible jumps.

    This follows the history/velocity safeguards used by the other sport
    repositories: short gaps are tolerated, but a low-confidence jump across
    the broadcast frame is not allowed to become a trajectory.
    """
    result = {"Frame": [], "X": [], "Y": [], "Visibility": []}
    # Shuttle motion can be very fast, especially immediately after a hit.
    max_jump = max(180.0, float(os.environ.get("TRACKNET_MAX_JUMP", "360")))
    # A domain-mismatched TrackNet often locks onto a scoreboard/banner for
    # dozens of frames.  Reject a candidate whose accepted history remains in
    # a tiny image-space patch.  These knobs are environment-overridable so a
    # very static training clip can relax them without changing the API.
    static_window = int(static_window if static_window is not None else
                        os.environ.get("TRACKNET_STATIC_WINDOW", "12"))
    static_disp = float(static_disp if static_disp is not None else
                        os.environ.get("TRACKNET_STATIC_DISP", "36"))
    persistence_window = int(persistence_window if persistence_window is not None else
                             os.environ.get("TRACKNET_PERSISTENCE_WINDOW", "3"))
    max_gap = int(max_gap if max_gap is not None else
                  os.environ.get("TRACKNET_MAX_GAP", "5"))
    static_cooldown = int(static_cooldown if static_cooldown is not None else
                          os.environ.get("TRACKNET_STATIC_COOLDOWN", "0"))
    static_window = max(0, static_window)
    persistence_window = max(0, persistence_window)
    max_gap = max(0, max_gap)
    static_cooldown = max(0, static_cooldown)

    from collections import deque

    accepted_history = deque(maxlen=max(2, static_window))
    recent_candidates = deque(maxlen=max(1, persistence_window))
    # (center_x, center_y, expiry_frame) regions learned to be static.  A
    # cooldown avoids reacquiring the same banner after ``max_gap`` clears the
    # Kalman history; a real shuttle which later crosses the region can still
    # be accepted once the short cooldown expires.
    static_regions = []
    static_centers_by_frame = _find_static_candidate_frames(
        candidates_by_frame, window=static_window, disp=static_disp
    ) if static_window >= 2 else [[] for _ in candidates_by_frame]
    last = None
    velocity = np.zeros(2, dtype=np.float32)
    miss_count = 0

    def append_missing(frame_i):
        result["Frame"].append(frame_i)
        result["X"].append(0)
        result["Y"].append(0)
        result["Visibility"].append(0)

    for frame_i, raw in enumerate(candidates_by_frame):
        if static_regions:
            static_regions = [entry for entry in static_regions
                              if entry[2] < 0 or entry[2] >= frame_i]
        clusters = _cluster_candidates(raw)
        # Save a compact candidate snapshot before selection.  It is used as a
        # weak persistence bonus/gate below; a one-frame low-score blob in the
        # audience should not start an otherwise unsupported track.
        recent_snapshot = [(float(c["x"]), float(c["y"]),
                            float(c.get("mean_score", c["score"]))) for c in clusters]
        recent_candidates.append(recent_snapshot)
        if last is not None and any(
            float(np.linalg.norm(last - np.asarray(center, dtype=np.float32))) <= max(24.0, static_disp)
            for center in static_centers_by_frame[frame_i]
        ):
            # The causal state may have followed this lock before the
            # pre-pass identified it.  Drop that state now so a genuine,
            # distant candidate in the same frame can reacquire immediately.
            last = None
            velocity[:] = 0.0
            accepted_history.clear()
        chosen = None
        if clusters:
            scored = []
            for c in clusters:
                quality = float(c.get("mean_score", c["score"] / max(1, c["count"])))
                if quality < min_score:
                    continue
                point = np.array([c["x"], c["y"]], dtype=np.float32)
                if any(
                    float(np.linalg.norm(point - np.array([center_x, center_y], dtype=np.float32)))
                    <= max(24.0, static_disp * 1.5)
                    and (expiry < 0 or frame_i <= expiry)
                    for center_x, center_y, expiry in static_regions
                ):
                    continue
                if any(
                    float(np.linalg.norm(point - np.asarray(center, dtype=np.float32)))
                    <= max(24.0, static_disp)
                    for center in static_centers_by_frame[frame_i]
                ):
                    # Pre-pass suppression is candidate-specific, preserving
                    # a genuine moving shuttle if a frame also contains a
                    # persistent banner response.
                    continue
                persistence = 0
                if persistence_window > 1:
                    for previous in list(recent_candidates)[:-1]:
                        if any(float(np.linalg.norm(point - np.array([px, py]))) <= 55.0
                               for px, py, _ in previous):
                            persistence += 1

                # Once a history is long enough, a compact cluster is much
                # more likely to be a static overlay than a flying shuttle.
                # Clear the state when rejecting it, so the next real flight
                # may reacquire even if it is far from the banner.
                is_static = False
                static_center = None
                if static_window >= 2 and len(accepted_history) >= static_window:
                    hist = np.asarray(accepted_history, dtype=np.float32)
                    center = np.median(hist, axis=0)
                    spread = np.linalg.norm(hist - center, axis=1)
                    is_static = (float(np.max(spread)) <= static_disp and
                                 float(np.linalg.norm(point - center)) <= static_disp)
                    if is_static:
                        static_center = center
                if is_static:
                    if static_center is not None and static_cooldown > 0:
                        static_regions.append((float(static_center[0]), float(static_center[1]),
                                               frame_i + static_cooldown))
                    elif static_center is not None:
                        # expiry < 0 denotes a lock suppressed for the rest
                        # of this video; this is the safest default for a
                        # fixed scoreboard/banner location.
                        static_regions.append((float(static_center[0]), float(static_center[1]), -1))
                    continue

                dist = 0.0
                if last is not None:
                    predicted = last + velocity
                    dist = float(np.linalg.norm(point - predicted))
                    # A high-confidence response can legitimately jump at a
                    # smash, but weak responses should not teleport from the
                    # court into the banner or vice versa.
                    if dist > max_jump and quality < min_score + 0.10:
                        continue
                # Repeated tile support and temporal persistence are positive
                # evidence.  Distance is a soft cost, not a hard preference:
                # direction reversals at impact should remain selectable.
                rank = (1.8 * quality +
                        0.22 * float(c.get("mean_contrast", 0.0)) +
                        0.10 * min(int(c.get("count", 1)), 3) +
                        0.08 * min(persistence, 3) -
                        (0.55 * min(1.0, dist / max_jump) if last is not None else 0.0))
                # A weak, isolated candidate is held until it is corroborated
                # by another frame/tile.  Strong peaks and candidates near an
                # existing track remain usable immediately.
                if (last is None and quality < min_score + 0.10 and
                        int(c.get("count", 1)) < 2 and persistence == 0):
                    continue
                scored.append((rank, c, persistence, dist))
            if scored:
                scored.sort(key=lambda item: item[0], reverse=True)
                chosen = scored[0][1]

        if chosen is None:
            append_missing(frame_i)
            # Do not extrapolate indefinitely through a miss.
            velocity *= 0.5
            miss_count += 1
            if miss_count > max_gap:
                last = None
                velocity[:] = 0.0
                accepted_history.clear()
            continue

        point = np.array([chosen["x"], chosen["y"]], dtype=np.float32)
        if last is not None:
            velocity = 0.55 * velocity + 0.45 * (point - last)
            # A light EMA removes tile-to-tile jitter without smearing a hit.
            point = 0.70 * point + 0.30 * (last + velocity)
        last = point
        miss_count = 0
        if static_window > 0:
            accepted_history.append(point.copy())
        result["Frame"].append(frame_i)
        result["X"].append(int(np.clip(round(float(point[0])), 0, width - 1)))
        result["Y"].append(int(np.clip(round(float(point[1])), 0, height - 1)))
        result["Visibility"].append(1)
    return result


def run_high_res_tiles(video_file, model, device, seq_len, bg_mode, width, height,
                       max_sample_num=300, tile_overlap=0.25, tile_batch_size=2,
                       tile_min_score=0.15, roi=None, court_points=None,
                       court_top_extension=120.0, court_side_padding=40.0,
                       court_bottom_padding=30.0):
    """Run TrackNet on overlapping high-resolution crops.

    The model still receives its native 512x288 input; only the source crop is
    changed.  This makes the method compatible with the existing checkpoint and
    keeps the normal 512x288 path untouched for legacy videos.
    """
    median = _read_video_median(video_file, max_sample_num) if bg_mode else None
    court_polygon = (_expanded_court_polygon(
        court_points,
        top_extension=court_top_extension,
        side_padding=court_side_padding,
        bottom_padding=court_bottom_padding,
    ) if court_points is not None else None)
    court_roi = _roi_from_court(court_polygon, width, height)
    if roi is None:
        if court_roi is not None:
            # A court-derived ROI is substantially tighter than the old
            # y=0.27*H default and excludes the scoreboard/banner outright.
            roi = court_roi
        else:
            # Without manual court points retain a conservative lower-frame
            # ROI.  The supplied checkpoint is known to fire around y≈350 on
            # this broadcast; starting at 0.36H keeps that banner out.
            roi = (int(width * 0.10), int(height * 0.36),
                   int(width * 0.94), int(height * 0.99))
    else:
        roi = tuple(int(round(v)) for v in roi)
        if court_roi is not None:
            # Intersect user ROI with the court envelope.  The precise convex
            # mask below still handles corners; this only avoids evaluating
            # obviously irrelevant tiles.
            ix1 = max(roi[0], court_roi[0])
            iy1 = max(roi[1], court_roi[1])
            ix2 = min(roi[2], court_roi[2])
            iy2 = min(roi[3], court_roi[3])
            if ix2 > ix1 and iy2 > iy1:
                roi = (ix1, iy1, ix2, iy2)
            else:
                roi = court_roi
    roi = (
        int(np.clip(roi[0], 0, max(0, int(width) - 1))),
        int(np.clip(roi[1], 0, max(0, int(height) - 1))),
        int(np.clip(roi[2], 1, int(width))),
        int(np.clip(roi[3], 1, int(height))),
    )
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        roi = (0, 0, int(width), int(height))
    # Anchor the tile grid to the effective ROI instead of the full frame.
    # Leave a small source margin because peak decoding discards responses
    # exactly on a crop edge; the ROI/court mask still rejects the margin's
    # banner/background pixels.  For the supplied 1920x1080 court this remains
    # four tiles rather than nine.
    tile_margin = 16
    tile_region = (roi[0] - tile_margin, roi[1] - tile_margin,
                   roi[2] + tile_margin, roi[3] + tile_margin)
    tiles = _make_tiles(width, height, tile_overlap, region=tile_region)
    print(f'High-resolution tiled inference: {len(tiles)} tiles, ROI={roi}, '
          f'court_mask={court_polygon is not None}, '
          f'min_score={tile_min_score:.3f}')

    candidates_by_frame = []
    frame_count = 0
    cap = cv2.VideoCapture(video_file)
    window = []
    starts_processed = 0

    def process_window(start_i, frames):
        nonlocal starts_processed
        # Build a batch in modest chunks because the TrackNet U-Net has large
        # intermediate feature maps even though its parameter file is small.
        for tile_start in range(0, len(tiles), max(1, tile_batch_size)):
            tile_chunk = tiles[tile_start:tile_start + max(1, tile_batch_size)]
            batch = []
            for x1, y1, x2, y2 in tile_chunk:
                seq = []
                for frame in frames:
                    crop = frame[y1:y2, x1:x2]
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    small = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                    rgb_chw = np.moveaxis(small, -1, 0)
                    if bg_mode in ('subtract', 'subtract_concat'):
                        med_rgb = cv2.cvtColor(median[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                        diff = np.sum(np.abs(rgb.astype(np.int16) - med_rgb.astype(np.int16)), axis=2)
                        diff = np.clip(diff, 0, 255).astype(np.uint8)
                        diff_small = cv2.resize(diff, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                        if bg_mode == 'subtract':
                            seq.append(diff_small[None, ...])
                        else:
                            seq.append(np.concatenate([rgb_chw, diff_small[None, ...]], axis=0))
                    else:
                        seq.append(rgb_chw)
                if bg_mode == 'concat':
                    med_crop = median[y1:y2, x1:x2]
                    med_small = cv2.resize(cv2.cvtColor(med_crop, cv2.COLOR_BGR2RGB),
                                           (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                    med_chw = np.moveaxis(med_small, -1, 0)
                    inp = np.concatenate([med_chw] + seq, axis=0)
                else:
                    inp = np.concatenate(seq, axis=0)
                batch.append(inp.astype(np.float32) / 255.0)
            x = torch.from_numpy(np.stack(batch, axis=0)).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                y_pred = model(x).detach().cpu().numpy()
            for b, tile in enumerate(tile_chunk):
                for f in range(seq_len):
                    frame_i = start_i + f
                    if frame_i >= len(candidates_by_frame):
                        candidates_by_frame.extend([] for _ in range(frame_i - len(candidates_by_frame) + 1))
                    candidate = _decode_tile_peak(y_pred[b, f], tile, tile_min_score, roi, court_polygon)
                    if candidate is not None:
                        candidates_by_frame[frame_i].append(candidate)
        starts_processed += 1
        if starts_processed % 25 == 0:
            print(f'  tiled windows: {starts_processed}', flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        window.append(frame)
        if len(window) == seq_len:
            process_window(frame_count - seq_len, window)
            window.pop(0)
    cap.release()

    # Pad the tail exactly as the original dataset does, so every source frame
    # gets one CSV row.
    if frame_count == 0:
        return {"Frame": [], "X": [], "Y": [], "Visibility": [],
                "Img_scaler": (width / WIDTH, height / HEIGHT),
                "Img_shape": (width, height)}
    tail = window
    if not tail:
        # The normal loop leaves the last seq_len-1 frames in the window.
        cap = cv2.VideoCapture(video_file)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - seq_len + 1))
        tail = []
        while len(tail) < seq_len - 1:
            ok, frame = cap.read()
            if not ok:
                break
            tail.append(frame)
        cap.release()
    if tail:
        last = tail[-1]
        for offset in range(0, min(len(tail), seq_len - 1)):
            start_i = frame_count - len(tail) + offset
            if start_i >= frame_count:
                break
            remaining = tail[offset:]
            padded = remaining + [last] * (seq_len - len(remaining))
            if len(padded) == seq_len:
                process_window(start_i, padded)

    # Ensure all frames are represented even if a decoder reports a short tail.
    while len(candidates_by_frame) < frame_count:
        candidates_by_frame.append([])
    pred = _select_temporal_track(
        candidates_by_frame[:frame_count], width, height, tile_min_score,
        static_window=int(os.environ.get("TRACKNET_STATIC_WINDOW", "12")),
        static_disp=float(os.environ.get("TRACKNET_STATIC_DISP", "36")),
        persistence_window=int(os.environ.get("TRACKNET_PERSISTENCE_WINDOW", "3")),
        max_gap=int(os.environ.get("TRACKNET_MAX_GAP", "5")),
    )
    pred.update({"Img_scaler": (width / WIDTH, height / HEIGHT),
                 "Img_shape": (width, height),
                 "Inpaint_Mask": []})
    return pred


def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """ Predict coordinates from heatmap or inpainted coordinates. 

        Args:
            indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy()if torch.is_tensor(indices) else indices.numpy()
    
    # Transform input for heatmap prediction
    if y_pred is not None:
        thresh = float(os.environ.get("TRACKNET_VIS_THRESH", "0.2"))
        y_pred = y_pred > thresh
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred) # (N, L, H, W)
    
    # Transform input for coordinate prediction
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    # Predict from coordinate
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT* img_scaler[1]) 
                elif y_pred is not None:
                    # Predict from heatmap
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred, cy_pred = int(bbox_pred[0]+bbox_pred[2]/2), int(bbox_pred[1]+bbox_pred[3]/2)
                    cx_pred, cy_pred = int(cx_pred*img_scaler[0]), int(cy_pred*img_scaler[1])
                else:
                    raise ValueError('Invalid input')
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break
    
    return pred_dict    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='file path of the video')
    parser.add_argument('--tracknet_file', type=str, help='file path of the TrackNet model checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='file path of the InpaintNet model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--nonoverlap-offset', type=int, default=0,
                        help='first frame of non-overlap TrackNet windows (default: 0)')
    parser.add_argument('--max_sample_num', type=int, default=1800, help='maximum number of frames to sample for generating median image')
    parser.add_argument('--video_range', type=lambda splits: [int(s) for s in splits.split(',')], default=None, help='range of start second and end second of the video for generating median image')
    parser.add_argument('--save_dir', type=str, default='pred_result', help='directory to save the prediction result')
    parser.add_argument('--large_video', action='store_true', default=False, help='whether to process large video')
    parser.add_argument('--output_video', action='store_true', default=False, help='whether to output video with predicted trajectory')
    parser.add_argument('--traj_len', type=int, default=8, help='length of trajectory to draw on video')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='inference device')
    parser.add_argument('--high-res-tiles', action='store_true', default=False,
                        help='run overlapping 1024x576 crops for high-resolution input')
    parser.add_argument('--tile-overlap', type=float, default=0.25,
                        help='overlap ratio between high-resolution crops')
    parser.add_argument('--tile-batch-size', type=int, default=2,
                        help='number of crops per TrackNet forward pass')
    parser.add_argument('--tile-min-score', type=float,
                        default=float(os.environ.get('TRACKNET_TILE_MIN_SCORE', '0.15')),
                        help='minimum heatmap peak for tiled candidate decoding')
    parser.add_argument('--tracknet-roi', type=lambda s: [int(v) for v in s.split(',')], default=None,
                        help='optional x1,y1,x2,y2 ROI for tiled candidate filtering')
    parser.add_argument('--tracknet-court-points', type=lambda s: [float(v) for v in s.split(',')], default=None,
                        help='optional TL,TR,BR,BL court points (x1,y1,...); filters tiled candidates')
    parser.add_argument('--tracknet-court-top-extension', type=float, default=60.0,
                        help='pixels to extend the far baseline upward for airborne shuttles')
    parser.add_argument('--tracknet-court-side-padding', type=float, default=40.0,
                        help='horizontal pixels to pad the tiled court mask')
    parser.add_argument('--tracknet-court-bottom-padding', type=float, default=30.0,
                        help='pixels to extend the near baseline in the tiled court mask')
    args = parser.parse_args()

    if args.nonoverlap_offset < 0:
        parser.error('--nonoverlap-offset must be non-negative')
    if args.nonoverlap_offset and args.eval_mode != 'nonoverlap':
        parser.error('--nonoverlap-offset requires --eval_mode nonoverlap')
    if args.nonoverlap_offset and args.high_res_tiles:
        parser.error('--nonoverlap-offset is not supported with --high-res-tiles')

    num_workers = args.batch_size if args.batch_size <= 16 else 16
    video_file = args.video_file
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_range = args.video_range if args.video_range else None
    large_video = args.large_video
    out_csv_file = os.path.join(args.save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(args.save_dir, f'{video_name}.mp4')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f'Use device: {device}')

    # Load model
    tracknet_ckpt = torch.load(args.tracknet_file, map_location=device, weights_only=False)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    if args.nonoverlap_offset >= tracknet_seq_len:
        parser.error(
            f'--nonoverlap-offset must be smaller than TrackNet seq_len '
            f'({tracknet_seq_len})'
        )
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
    tracknet.load_state_dict(tracknet_ckpt['model'])

    if args.inpaintnet_file:
        inpaintnet_ckpt = torch.load(args.inpaintnet_file, map_location=device, weights_only=False)
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').to(device)
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
    else:
        inpaintnet = None

    cap = cv2.VideoCapture(args.video_file)
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)

    tracknet_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Inpaint_Mask':[],
                        'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}

    # Test on TrackNet
    tracknet.eval()
    seq_len = tracknet_seq_len
    if args.high_res_tiles and (w > WIDTH * 1.35 or h > HEIGHT * 1.35):
        cap.release()
        tracknet_pred_dict = run_high_res_tiles(
            video_file=args.video_file,
            model=tracknet,
            device=device,
            seq_len=seq_len,
            bg_mode=bg_mode,
            width=w,
            height=h,
            max_sample_num=args.max_sample_num,
            tile_overlap=max(0.0, min(0.75, args.tile_overlap)),
            tile_batch_size=max(1, args.tile_batch_size),
            tile_min_score=max(0.0, min(1.0, args.tile_min_score)),
            roi=tuple(args.tracknet_roi) if args.tracknet_roi and len(args.tracknet_roi) == 4 else None,
            court_points=(np.asarray(args.tracknet_court_points, dtype=np.float32).reshape(4, 2)
                          if args.tracknet_court_points and len(args.tracknet_court_points) == 8 else None),
            court_top_extension=max(0.0, args.tracknet_court_top_extension),
            court_side_padding=max(0.0, args.tracknet_court_side_padding),
            court_bottom_padding=max(0.0, args.tracknet_court_bottom_padding),
        )
    elif args.eval_mode == 'nonoverlap':
        # Create dataset with non-overlap sampling
        if large_video:
            dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=seq_len, bg_mode=bg_mode,
                                            max_sample_num=args.max_sample_num, video_range=video_range,
                                            start_offset=args.nonoverlap_offset)
            _reconcile_stream_dataset_length(dataset, video_file)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            print(f'Video length: {dataset.video_len}')
        else:
            # Sample all frames from video
            frame_list = generate_frames(args.video_file)
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap', bg_mode=bg_mode,
                                                 frame_arr=np.array(frame_list)[:, :, :, ::-1], padding=True,
                                                 start_offset=args.nonoverlap_offset)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().to(device)
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
            
            # Predict
            tmp_pred = predict(i, y_pred=y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])
    else:
        # Create dataset with overlap sampling for temporal ensemble
        if large_video:
            dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=1, bg_mode=bg_mode,
                                            max_sample_num=args.max_sample_num, video_range=video_range)
            _reconcile_stream_dataset_length(dataset, video_file)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            video_len = dataset.video_len
            print(f'Video length: {video_len}')
            
        else:
            # Sample all frames from video
            frame_list = generate_frames(args.video_file)
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='heatmap', bg_mode=bg_mode,
                                                 frame_arr=np.array(frame_list)[:, :, :, ::-1])
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            video_len = len(frame_list)
        
        # Init prediction buffer params
        num_sample, sample_count = video_len-seq_len+1, 0
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
        frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
        weight = get_ensemble_weight(seq_len, args.eval_mode)
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().to(device)
            b_size, seq_len = i.shape[0], i.shape[1]
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
            
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

            for b in range(b_size):
                if sample_count < buffer_size:
                    # Imcomplete buffer
                    y_pred = y_pred_buffer[batch_i+b, frame_i].sum(0) / (sample_count+1)
                else:
                    # General case
                    y_pred = (y_pred_buffer[batch_i+b, frame_i] * weight[:, None, None]).sum(0)
                
                ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                sample_count += 1

                if sample_count == num_sample:
                    # Last batch
                    y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                    for f in range(1, seq_len):
                        # Last input sequence
                        y_pred = y_pred_buffer[batch_i+b+f, frame_i].sum(0) / (seq_len-f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

            # Predict
            tmp_pred = predict(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

            # Update buffer, keep last predictions for ensemble in next iteration
            y_pred_buffer = y_pred_buffer[-buffer_size:]

    #assert video_len == len(tracknet_pred_dict['Frame']), 'Prediction length mismatch'
    # Remove short, compact body/background locks before an optional InpaintNet
    # pass and before writing the phase CSV.  Applying the same causal timeout
    # to both dual-phase runs prevents a stale primary/secondary response from
    # surviving fusion merely because one temporal window happened to persist.
    tracknet_pred_dict = suppress_static_prediction_locks(
        tracknet_pred_dict,
        window=int(os.environ.get("TRACKNET_STATIC_LOCK_WINDOW", "4")),
        disp=float(os.environ.get("TRACKNET_STATIC_LOCK_DISP", "10")),
        min_y=float(os.environ.get("TRACKNET_STATIC_LOCK_MIN_Y", "80")),
        preroll=int(os.environ.get("TRACKNET_STATIC_LOCK_PREROLL", "2")),
    )

    # Test on TrackNetV3 (TrackNet + InpaintNet)
    if inpaintnet is not None:
        inpaintnet.eval()
        seq_len = inpaintnet_seq_len
        tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h*0.05)
        inpaint_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

        if args.eval_mode == 'nonoverlap':
            # Create dataset with non-overlap sampling
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate', pred_dict=tracknet_pred_dict, padding=True)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.to(device), inpaint_mask.to(device)).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask) # replace predicted coordinates with inpainted coordinates
                
                # Thresholding
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.
                
                # Predict
                tmp_pred = predict(i, c_pred=coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
        else:
            # Create dataset with overlap sampling for temporal ensemble
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='coordinate', pred_dict=tracknet_pred_dict)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            weight = get_ensemble_weight(seq_len, args.eval_mode)

            # Init buffer params
            num_sample, sample_count = len(dataset), 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
            frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
            
            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                b_size = i.shape[0]
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.to(device), inpaint_mask.to(device)).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)
                
                # Thresholding
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.

                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)
                
                for b in range(b_size):
                    if sample_count < buffer_size:
                        # Imcomplete buffer
                        coor_inpaint = coor_inpaint_buffer[batch_i+b, frame_i].sum(0)
                        coor_inpaint /= (sample_count+1)
                    else:
                        # General case
                        coor_inpaint = (coor_inpaint_buffer[batch_i+b, frame_i] * weight[:, None]).sum(0)
                    
                    ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                    ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                    sample_count += 1

                    if sample_count == num_sample:
                        # Last input sequence
                        coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                        coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                        
                        for f in range(1, seq_len):
                            coor_inpaint = coor_inpaint_buffer[batch_i+b+f, frame_i].sum(0)
                            coor_inpaint /= (seq_len-f)
                            ensemble_i = torch.cat((ensemble_i, i[-1][f].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)

                # Thresholding
                th_mask = ((ensemble_coor_inpaint[:, :, 0] < COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < COOR_TH))
                ensemble_coor_inpaint[th_mask] = 0.

                # Predict
                tmp_pred = predict(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
                # Update buffer, keep last predictions for ensemble in next iteration
                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]
        

    # Write csv file
    pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict
    pred_dict = _canonicalize_prediction_dict(pred_dict)
    pred_dict = suppress_static_prediction_locks(
        pred_dict,
        window=int(os.environ.get("TRACKNET_STATIC_LOCK_WINDOW", "4")),
        disp=float(os.environ.get("TRACKNET_STATIC_LOCK_DISP", "10")),
        min_y=float(os.environ.get("TRACKNET_STATIC_LOCK_MIN_Y", "80")),
        preroll=int(os.environ.get("TRACKNET_STATIC_LOCK_PREROLL", "2")),
    )
    write_pred_csv(pred_dict, save_file=out_csv_file)

    # Write video with predicted coordinates
    if args.output_video:
        write_pred_video(video_file, pred_dict, save_file=out_video_file, traj_len=args.traj_len)

    print('Done.')
