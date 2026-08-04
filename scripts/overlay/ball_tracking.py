"""Temporal post-processing for TrackNet ball coordinates.

The TrackNet model produces a per-frame measurement, but a broadcast video
often contains short holes and occasional one-frame jumps.  This module keeps
the raw measurement separate from the derived trajectory and provides three
small, dependency-free building blocks used by the overlay renderer:

* a constant-velocity Kalman filter with short-gap interpolation;
* a conservative render-only bridge for top-edge out-of-frame flights;
* velocity/curvature based hit-point detection;
* rally segmentation from long ball gaps and camera cuts.

The implementation deliberately does not mark long Kalman predictions as real
observations.  They can be drawn faintly, but must not create synthetic hits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math
import numpy as np


# ``model_edge`` and ``single_phase_edge`` are detector observations which
# are useful for drawing/reacquiring a trajectory at the image boundary, but
# are deliberately *not* treated as ordinary hit evidence.  Keeping these
# labels separate is important: a top-edge point can be temporally real while
# still having no reliable image evidence for a racket contact.
#
# ``classical_gap`` is an even weaker provenance.  It is a CPU candidate which
# was admitted only to fill a *post-veto* TrackNet hole after a two-sided
# trajectory check.  It is useful for the rendered trail, but must never count
# towards measured coverage or generate a hit/rally event.  Keep it in the
# tracking set so the Kalman stage can preserve its coordinate and source.
REAL_MEASUREMENT_SOURCES = {"model", "classical"}
EDGE_MEASUREMENT_SOURCES = {"model_edge", "single_phase_edge"}
DERIVED_MEASUREMENT_SOURCES = {"classical_gap"}
TRACKING_MEASUREMENT_SOURCES = (
    REAL_MEASUREMENT_SOURCES
    | EDGE_MEASUREMENT_SOURCES
    | DERIVED_MEASUREMENT_SOURCES
)
MEASUREMENT_SOURCES = REAL_MEASUREMENT_SOURCES | {"interp"}


@dataclass
class BallPoint:
    frame: int
    x: float
    y: float
    visible: bool
    source: str = "missing"  # model / classical / classical_gap / interp / kalman / offscreen_bridge / missing
    raw_visible: bool = False
    confidence: float = 0.0


@dataclass
class HitEvent:
    frame: int
    x: float
    y: float
    score: float
    speed: float
    hitter: str = "unknown"
    rally: int = 0
    map_xy: Optional[Tuple[int, int]] = None
    uncertain: bool = False
    source: str = "curvature"
    track_id: int = -1
    ball_observed: bool = True
    # Playback/export metadata.  These are populated only after an event has
    # survived all frame-local checks in the renderer, so a later body veto
    # cannot make an already displayed counter run backwards.
    rally_hit_index: int = 0
    video_hit_index: int = 0
    # A shot belongs to the event that starts it.  Its result becomes known
    # when the next confirmed contact arrives, hence ``shot_end_frame`` is
    # -1 and the speed is ``None`` for an unfinished flight.
    shot_end_frame: int = -1
    shot_avg_speed_mps: Optional[float] = None
    shot_distance_m: float = 0.0
    shot_observed_seconds: float = 0.0
    shot_elapsed_seconds: float = 0.0
    shot_coverage: float = 0.0
    shot_speed_quality: str = "pending"
    shot_speed_method: str = "pending"
    # Hitter identity is independent from hit/trajectory quality.  A legacy
    # side hint may be useful for diagnostics, but downstream coaching must be
    # able to distinguish it from a strict pose-contact association.
    hitter_confidence: float = 0.0
    hitter_source: str = "unknown"


def _raw_arrays(ball_dict: Mapping[int, Tuple], frame_count: int):
    n = max(0, int(frame_count))
    xy = np.full((n, 2), np.nan, dtype=np.float32)
    observed = np.zeros(n, dtype=bool)
    observed_source = np.full(n, "model", dtype=object)
    observed_confidence = np.ones(n, dtype=np.float32)
    for i, value in ball_dict.items():
        if i < 0 or i >= n or value is None:
            continue
        vis, x, y = value[:3]
        if int(vis) > 0 and float(x) > 0 and float(y) > 0:
            xy[int(i)] = (float(x), float(y))
            observed[int(i)] = True
            if len(value) >= 4 and str(value[3]) in TRACKING_MEASUREMENT_SOURCES:
                observed_source[int(i)] = str(value[3])
            if len(value) >= 5:
                try:
                    observed_confidence[int(i)] = float(np.clip(float(value[4]), 0.0, 1.0))
                except (TypeError, ValueError):
                    pass
    return xy, observed, observed_source, observed_confidence


def _find_edge_gap_breaks(
    xy: np.ndarray,
    observed: np.ndarray,
    edge_y: float = 35.0,
    min_gap: int = 2,
    max_gap: int = 8,
    max_anchor_jump: float = 80.0,
) -> List[int]:
    """Return starts of short holes which likely cross the top image edge.

    TrackNet emits ``y≈0`` immediately before a high clear disappears.  If a
    later detection reappears deep in the court, ordinary linear interpolation
    and a few Kalman forecasts create a visually plausible but false line.
    A gap is considered an edge crossing when at least one real anchor is near
    the top edge and the other anchor is either also near that edge or differs
    by a substantial vertical amount.  Only the first missing frame is
    returned; callers use it as a causal reset boundary.

    The helper intentionally works on raw observations and has no court/player
    dependencies, making it safe for both the overlay and command-line tools.
    """
    n = min(len(xy), len(observed))
    if n <= 0:
        return []
    edge = max(0.0, float(edge_y))
    lo_gap = max(1, int(min_gap))
    hi_gap = max(lo_gap, int(max_gap))
    breaks: List[int] = []
    i = 0
    while i < n:
        if bool(observed[i]):
            i += 1
            continue
        start = i
        while i < n and not bool(observed[i]):
            i += 1
        end = i
        gap = end - start
        if gap > hi_gap or start <= 0 or end >= n:
            continue
        left = np.asarray(xy[start - 1], dtype=np.float64)
        right = np.asarray(xy[end], dtype=np.float64)
        if not np.isfinite([left, right]).all():
            continue
        left_top = float(left[1]) <= edge
        right_top = float(right[1]) <= edge
        vertical_jump = abs(float(right[1] - left[1]))
        if gap < lo_gap and vertical_jump < max(0.0, float(max_anchor_jump)):
            continue
        # One top anchor plus a sizeable vertical displacement is an
        # unambiguous leave/re-entry.  Two top anchors separated by >=2
        # frames are also treated as an unseen apex; there is no evidence to
        # choose a linear path between them.
        if (left_top or right_top) and (
            vertical_jump >= 70.0 or (left_top and right_top)
        ) or vertical_jump >= max(0.0, float(max_anchor_jump)):
            breaks.append(int(start))
    return breaks


def _interpolate_short_gaps(
    xy: np.ndarray,
    observed: np.ndarray,
    max_gap: int,
    break_frames: Optional[Sequence[int]] = None,
    observed_source: Optional[np.ndarray] = None,
):
    """Fill only bounded holes between two real observations."""
    n = len(observed)
    filled = xy.copy()
    source = np.full(n, "missing", dtype=object)
    if observed_source is None:
        source[observed] = "model"
    else:
        source[observed] = observed_source[observed]
    i = 0
    while i < n:
        if observed[i]:
            i += 1
            continue
        start = i
        while i < n and not observed[i]:
            i += 1
        end = i  # first observed index after the gap
        gap = end - start
        crosses_break = False
        if break_frames is not None:
            crosses_break = any(start <= int(frame) <= end for frame in break_frames)
        if start > 0 and end < n and gap <= max_gap and not crosses_break:
            p0, p1 = filled[start - 1], filled[end]
            for j in range(start, end):
                alpha = (j - (start - 1)) / float(end - (start - 1))
                filled[j] = p0 * (1.0 - alpha) + p1 * alpha
                source[j] = "interp"
    return filled, source


def smooth_ball_track(
    ball_dict: Mapping[int, Tuple],
    frame_count: int,
    fps: float,
    max_interp_gap: int = 8,
    max_predict_gap: int = 3,
    process_noise: float = 700.0,
    measurement_noise: float = 18.0,
    reset_frames: Optional[Sequence[int]] = None,
) -> List[BallPoint]:
    """Return a temporally smoothed track while preserving observation source.

    State is ``[x, y, vx, vy]`` in pixels/frame.  The relatively large process
    noise is intentional: a shuttle can reverse direction in a few frames and
    an over-confident low-noise filter visibly lags behind the hit.
    """
    n = max(0, int(frame_count))
    if n == 0:
        return []
    fps = max(float(fps), 1.0)
    raw_xy, observed, observed_source, observed_confidence = _raw_arrays(ball_dict, n)
    reset_set = set()
    if reset_frames is not None:
        for frame in reset_frames:
            try:
                reset_set.add(int(frame))
            except (TypeError, ValueError):
                continue
    # A short numerical hole at the top edge is commonly an actual
    # out-of-frame flight (for example y=3 -> [missing] -> y=427), not a
    # detector dropout that can safely be linearly filled.  Detect those
    # boundaries before interpolation and reset the causal state at the first
    # missing frame.  This keeps Kalman/interp output from masquerading as a
    # visible shuttle during a long high clear.
    edge_break_frames = _find_edge_gap_breaks(
        raw_xy,
        observed,
        edge_y=35.0,
        min_gap=2,
        max_gap=max(2, int(max_interp_gap)),
        max_anchor_jump=80.0,
    )
    reset_set.update(edge_break_frames)
    work_xy, source = _interpolate_short_gaps(
        raw_xy,
        observed,
        max(0, int(max_interp_gap)),
        break_frames=reset_set,
        observed_source=observed_source,
    )

    dt = 1.0
    transition = np.array(
        [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    q = float(max(process_noise, 1.0))
    # Continuous white-noise acceleration model, in pixel units.
    process = q * np.array(
        [[0.25, 0.0, 0.5, 0.0], [0.0, 0.25, 0.0, 0.5], [0.5, 0.0, 1.0, 0.0], [0.0, 0.5, 0.0, 1.0]],
        dtype=np.float32,
    )
    identity = np.eye(4, dtype=np.float32)

    state = None
    covariance = None
    last_measurement = -10**9
    # Do not forecast immediately after a hard reset/re-acquisition.  The
    # first point after an out-of-frame gap is an anchor, not evidence that the
    # shuttle remained visible in the following missing frames.
    prediction_warmup = True
    out: List[BallPoint] = []

    for i in range(n):
        if i in reset_set:
            state = None
            covariance = None
            last_measurement = -10**9
            prediction_warmup = True
        has_measurement = source[i] in (
            MEASUREMENT_SOURCES
            | EDGE_MEASUREMENT_SOURCES
            | DERIVED_MEASUREMENT_SOURCES
        ) and np.all(np.isfinite(work_xy[i]))
        if state is None:
            if not has_measurement:
                out.append(BallPoint(i, 0.0, 0.0, False, "missing", False, 0.0))
                continue
            state = np.array([work_xy[i, 0], work_xy[i, 1], 0.0, 0.0], dtype=np.float32)
            covariance = np.diag([measurement_noise**2, measurement_noise**2, 250.0, 250.0]).astype(np.float32)
            last_measurement = i
            prediction_warmup = True
            initial_confidence = float(observed_confidence[i]) if observed[i] else 0.55
            out.append(BallPoint(
                i, float(state[0]), float(state[1]), True, str(source[i]),
                bool(observed[i]), initial_confidence,
            ))
            continue

        state = transition @ state
        covariance = transition @ covariance @ transition.T + process

        accepted = False
        rejected_jump = False
        reacquired = False
        if has_measurement:
            z = work_xy[i].astype(np.float32)
            innovation = z - observation @ state
            innovation_norm = float(np.linalg.norm(innovation))
            # Gate only extreme jumps.  The shuttle is fast, so this is much
            # looser than a player-tracking gate.
            gate = max(180.0, 5.0 * math.sqrt(float(np.trace(covariance[:2, :2]))))
            if innovation_norm <= gate or source[i] == "interp":
                r = measurement_noise**2 * (4.0 if source[i] == "interp" else 1.0)
                meas_cov = np.eye(2, dtype=np.float32) * r
                innovation_cov = observation @ covariance @ observation.T + meas_cov
                gain = covariance @ observation.T @ np.linalg.inv(innovation_cov)
                state = state + gain @ innovation
                covariance = (identity - gain @ observation) @ covariance
                accepted = True
                last_measurement = i
            elif source[i] != "interp":
                # A large jump immediately after a top-edge anchor is more
                # often a body/background switch than a visible shuttle.  Do
                # not draw the stale Kalman state for this frame; wait for a
                # bounded re-acquisition instead.
                rejected_jump = True
                prediction_warmup = True

            # Once the bounded prediction window has expired, the old state
            # is no longer a meaningful gate for a newly observed shuttle.
            # Keeping that stale state used to reject every point after a
            # long TrackNet hole (or a high clear that left the frame), which
            # inflated a short detector miss into a much longer outage.  A
            # real measurement now starts a fresh track segment.  This is a
            # re-acquisition, not a Kalman prediction: provenance remains the
            # original model/classical source and no missing coordinate is
            # invented.
            if not accepted and i - last_measurement > max_predict_gap:
                state = np.array(
                    [work_xy[i, 0], work_xy[i, 1], 0.0, 0.0],
                    dtype=np.float32,
                )
                covariance = np.diag(
                    [measurement_noise**2, measurement_noise**2, 250.0, 250.0]
                ).astype(np.float32)
                accepted = True
                last_measurement = i
                reacquired = True

        frames_since_measurement = i - last_measurement
        if accepted:
            point_source = str(source[i])
            visible = True
            confidence = (
                float(observed_confidence[i])
                if point_source in REAL_MEASUREMENT_SOURCES
                else (
                    0.28
                    if point_source in EDGE_MEASUREMENT_SOURCES
                    else (0.40 if point_source in DERIVED_MEASUREMENT_SOURCES else 0.55)
                )
            )
            if observed[i] and not reacquired:
                prediction_warmup = False
            elif reacquired:
                prediction_warmup = True
        elif frames_since_measurement <= max_predict_gap and not prediction_warmup and not rejected_jump:
            point_source = "kalman"
            visible = True
            confidence = max(0.15, 0.45 - 0.10 * frames_since_measurement)
        else:
            point_source = "missing"
            visible = False
            confidence = 0.0
            # Do not let an old state drag the next rally across the frame.
            state[2:] *= 0.35

        out.append(BallPoint(i, float(state[0]), float(state[1]), visible, point_source, bool(observed[i]), confidence))

    return out


def add_offscreen_bridges(
    track: Sequence[BallPoint],
    scene_cuts: Optional[Sequence[int]] = None,
    max_gap: int = 40,
    min_gap: int = 4,
    top_y: float = 30.0,
    velocity_window: int = 4,
    min_speed: float = 5.0,
    min_vertical_speed: float = 4.0,
    min_horizontal_speed: float = 8.0,
    confidence: float = 0.12,
    allow_mixed_edge_bridge: bool = False,
    mixed_support_frames: int = 3,
    mixed_speed_ratio: float = 0.25,
) -> List[BallPoint]:
    """Add conservative, render-only bridges for top-edge flight gaps.

    A badminton shuttle often leaves the broadcast image at the top edge and
    re-enters a few frames later.  There is no observation to recover in that
    interval, but a short low-confidence line is useful for the visual trail.
    This helper deliberately does *not* alter the measured track used for hit
    detection or rally segmentation: callers should keep the returned list in
    a separate render path.

    A bridge is considered only when both sides are measured tracking
    observations (``model``/``classical`` or the low-confidence
    ``model_edge``/``single_phase_edge`` variants) from the same camera
    segment, the hole is at most ``max_gap`` frames, and the local motion
    approaches the top edge (negative image-y velocity) before the hole and
    leaves it afterwards (positive image-y velocity).  At least one anchor
    must satisfy ``y <= top_y``.  A weak horizontal reversal is tolerated
    around an apex, while a clear reversal is rejected as an inconsistent
    detector switch.  Edge anchors are used only to draw this render bridge;
    they remain excluded from hit/rally evidence by the source gates in this
    module.

    ``allow_mixed_edge_bridge`` enables a stricter special case for a normal
    TrackNet anchor followed (or preceded) by a phase-labelled top-edge
    anchor.  This is useful when the two temporal passes disagree only at the
    image boundary (for example the ordinary ``model`` point at frame 698
    and ``model_edge`` re-entry at frame 720).  The mixed path is deliberately
    more demanding than the ordinary/edge-symmetric path: both anchors must
    be at ``y <= top_y``, the three frames ending at the left anchor must move
    monotonically upwards, the three frames starting at the right anchor must
    move monotonically downwards, and every frame in the hole must be truly
    unobserved.  Horizontal velocity must keep its sign and the two fitted
    speeds may not differ by more than ``mixed_speed_ratio``.  These gates
    prevent a body point inside the gap from being hidden by a synthetic
    trail.

    Bridge points have ``source='offscreen_bridge'``, ``raw_visible=False``
    and a deliberately low confidence.  They therefore cannot become real
    evidence accidentally if a caller later computes coverage or hit events.
    Existing real observations are never overwritten; derived Kalman/interp
    points inside an accepted hole are replaced only in the returned render
    copy so the visible line is continuous and clearly marked as derived.
    """

    if not track:
        return []

    out = list(track)
    frame_to_index = {
        int(point.frame): index
        for index, point in enumerate(track)
        if point is not None
    }
    # A duplicate/non-monotone frame list is not expected from the smoother;
    # refusing to bridge it is safer than writing a point to the wrong row.
    if len(frame_to_index) != len(track):
        return out

    cuts = []
    cut_values = [scene_cuts] if isinstance(scene_cuts, (int, np.integer)) else (scene_cuts or ())
    for cut in cut_values:
        try:
            cuts.append(int(cut))
        except (TypeError, ValueError):
            continue
    cuts = sorted(set(cuts))
    max_hole = max(1, int(max_gap))
    min_hole = max(1, int(min_gap))
    window = max(2, int(velocity_window))
    top_limit = float(top_y)
    speed_limit = max(0.0, float(min_speed))
    vertical_limit = max(0.0, float(min_vertical_speed))
    horizontal_limit = max(0.0, float(min_horizontal_speed))
    mixed_support = max(3, int(mixed_support_frames))
    mixed_ratio = float(np.clip(float(mixed_speed_ratio), 0.0, 1.0))

    # Edge-labelled points are trustworthy enough to define where a shuttle
    # leaves/re-enters the image, but are intentionally not ordinary evidence.
    # Keep this anchor set separate from ``REAL_MEASUREMENT_SOURCES`` so
    # coverage/hit callers cannot accidentally inherit the relaxation.
    bridge_anchor_sources = TRACKING_MEASUREMENT_SOURCES
    real_indices = [
        index
        for index, point in enumerate(track)
        if point is not None
        and bool(point.visible)
        and str(point.source) in bridge_anchor_sources
        and np.isfinite([point.x, point.y]).all()
    ]
    if len(real_indices) < 2:
        return out

    def same_scene(first_frame: int, second_frame: int) -> bool:
        lo, hi = sorted((int(first_frame), int(second_frame)))
        return not any(lo < cut <= hi for cut in cuts)

    def local_velocity(anchor_position: int, direction: int):
        """Fit a local pixel/frame velocity on one side of an anchor."""
        anchor_frame = int(track[real_indices[anchor_position]].frame)
        if direction < 0:
            selected = real_indices[max(0, anchor_position - window + 1): anchor_position + 1]
        else:
            selected = real_indices[anchor_position: anchor_position + window]
        selected = [
            index for index in selected
            if same_scene(anchor_frame, int(track[index].frame))
        ]
        if len(selected) < 2:
            return None
        frames = np.asarray([float(track[index].frame) for index in selected], dtype=np.float64)
        points = np.asarray(
            [[float(track[index].x), float(track[index].y)] for index in selected],
            dtype=np.float64,
        )
        if not np.isfinite(frames).all() or not np.isfinite(points).all():
            return None
        # Do not estimate a direction across another long detector hole.  The
        # nearest two anchors are enough when the local window is sparse.
        if float(frames[-1] - frames[0]) > max(8.0, 2.0 * window):
            return None
        centered = frames - frames[0]
        if float(np.ptp(centered)) <= 0.0:
            return None
        try:
            vx, _ = np.polyfit(centered, points[:, 0], 1)
            vy, _ = np.polyfit(centered, points[:, 1], 1)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None
        velocity = np.asarray([vx, vy], dtype=np.float64)
        return velocity if np.isfinite(velocity).all() else None

    def mixed_side_velocity(anchor_index: int, direction: int):
        """Fit velocity on a contiguous, monotone three-frame side.

        ``direction=-1`` selects the frames ending at ``anchor_index``
        (approach to the top edge); ``direction=+1`` selects the frames
        starting at the anchor (descending re-entry).  We intentionally use
        frame-list positions rather than the sparse ``real_indices`` list so
        a missing frame cannot be silently skipped while satisfying the
        monotonicity gate.
        """
        if direction < 0:
            positions = list(range(anchor_index - mixed_support + 1, anchor_index + 1))
        else:
            positions = list(range(anchor_index, anchor_index + mixed_support))
        if positions[0] < 0 or positions[-1] >= len(track):
            return None
        selected = []
        for position in positions:
            point = track[position]
            if point is None or not bool(point.visible):
                return None
            if str(point.source) not in TRACKING_MEASUREMENT_SOURCES:
                return None
            if not np.isfinite([point.x, point.y]).all():
                return None
            if selected:
                previous = selected[-1]
                if int(point.frame) != int(previous.frame) + 1:
                    return None
                if not same_scene(int(previous.frame), int(point.frame)):
                    return None
            selected.append(point)
        points = np.asarray([[float(point.x), float(point.y)] for point in selected], dtype=np.float64)
        frames = np.asarray([float(point.frame) for point in selected], dtype=np.float64)
        y_delta = np.diff(points[:, 1])
        # Image coordinates grow downwards: an upward approach has decreasing
        # y, while a downward re-entry has increasing y.  A tiny 0.5 px
        # tolerance suppresses numerical ties without accepting a reversal.
        if direction < 0:
            if not np.all(y_delta <= -0.5):
                return None
        elif not np.all(y_delta >= 0.5):
            return None
        centered = frames - frames[0]
        try:
            vx, _ = np.polyfit(centered, points[:, 0], 1)
            vy, _ = np.polyfit(centered, points[:, 1], 1)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None
        velocity = np.asarray([vx, vy], dtype=np.float64)
        return velocity if np.isfinite(velocity).all() else None

    def hole_is_unobserved(left_frame: int, right_frame: int) -> bool:
        """Require every interior frame to be non-visible for mixed bridges."""
        for frame in range(int(left_frame) + 1, int(right_frame)):
            index = frame_to_index.get(frame)
            if index is None:
                return False
            point = track[index]
            # A visible Kalman/interpolation point is still a tracking claim;
            # do not overwrite it with a mixed render-only bridge.  Points
            # hidden by a geometry/phase veto have ``visible=False`` and are
            # intentionally eligible as part of the empty hole.
            if point is not None and bool(point.visible):
                return False
        return True

    for left_position, right_position in zip(
        range(len(real_indices) - 1), range(1, len(real_indices))
    ):
        left_index = real_indices[left_position]
        right_index = real_indices[right_position]
        left = track[left_index]
        right = track[right_index]
        left_frame = int(left.frame)
        right_frame = int(right.frame)
        hole = right_frame - left_frame - 1
        if hole < min_hole or hole > max_hole or right_frame <= left_frame:
            continue
        if any(left_frame < cut <= right_frame for cut in cuts):
            continue
        left_source = str(left.source)
        right_source = str(right.source)
        left_edge = left_source in EDGE_MEASUREMENT_SOURCES
        right_edge = right_source in EDGE_MEASUREMENT_SOURCES
        left_ordinary = left_source in REAL_MEASUREMENT_SOURCES
        right_ordinary = right_source in REAL_MEASUREMENT_SOURCES
        mixed = left_edge != right_edge and (
            (left_edge and right_ordinary) or (right_edge and left_ordinary)
        )

        if mixed:
            # Mixed ordinary↔edge connections are opt-in and use the strict
            # side/empty-hole gates below.  Derived ``classical_gap`` points
            # are not ordinary anchors and therefore cannot enter this path.
            if not allow_mixed_edge_bridge:
                continue
            if float(left.y) > top_limit or float(right.y) > top_limit:
                continue
            if not hole_is_unobserved(left_frame, right_frame):
                continue
            incoming = mixed_side_velocity(left_index, -1)
            outgoing = mixed_side_velocity(right_index, +1)
            if incoming is None or outgoing is None:
                continue
            incoming_speed = float(np.linalg.norm(incoming))
            outgoing_speed = float(np.linalg.norm(outgoing))
            if incoming_speed < speed_limit or outgoing_speed < speed_limit:
                continue
            # Keep the horizontal direction through the unseen apex.  A
            # near-zero component is accepted as direction-neutral; when both
            # components are meaningful their signs must agree.
            if (
                abs(float(incoming[0])) >= horizontal_limit
                and abs(float(outgoing[0])) >= horizontal_limit
                and float(incoming[0]) * float(outgoing[0]) < 0.0
            ):
                continue
            if (
                abs(float(incoming[0])) > 1e-6
                and abs(float(outgoing[0])) > 1e-6
                and float(incoming[0]) * float(outgoing[0]) < 0.0
            ):
                continue
            if min(incoming_speed, outgoing_speed) / max(incoming_speed, outgoing_speed) < mixed_ratio:
                continue
        else:
            # Do not connect a low-confidence top-edge anchor to an ordinary
            # model/classical point unless the strict mixed mode above has
            # explicitly admitted it.  Legacy ordinary↔ordinary bridges
            # remain available, while edge bridges require edge support on
            # both sides.
            if left_edge != right_edge:
                continue
            if min(float(left.y), float(right.y)) > top_limit:
                continue
            if left_edge and (
                float(left.y) > top_limit or float(right.y) > top_limit
            ):
                continue
            incoming = local_velocity(left_position, -1)
            outgoing = local_velocity(right_position, +1)
        if incoming is None or outgoing is None:
            continue
        incoming_speed = float(np.linalg.norm(incoming))
        outgoing_speed = float(np.linalg.norm(outgoing))
        if incoming_speed < speed_limit or outgoing_speed < speed_limit:
            continue

        # Image y grows downwards: a top-edge crossing approaches with a
        # negative y velocity and leaves with a positive one.  This rejects
        # the static player/background branch around the end of the rally.
        if incoming[1] > -vertical_limit or outgoing[1] < vertical_limit:
            continue

        # Permit the horizontal component to turn gently at the apex, but do
        # not connect two anchors whose strong x directions are opposite.
        if (
            abs(float(incoming[0])) >= horizontal_limit
            and abs(float(outgoing[0])) >= horizontal_limit
            and float(incoming[0]) * float(outgoing[0]) < 0.0
        ):
            continue

        span = float(right_frame - left_frame)
        left_xy = np.asarray([float(left.x), float(left.y)], dtype=np.float64)
        right_xy = np.asarray([float(right.x), float(right.y)], dtype=np.float64)
        if not np.isfinite([left_xy, right_xy]).all():
            continue

        # Linear interpolation is intentionally boring: it cannot overshoot
        # into the stands or create a spurious bounce.  The low confidence and
        # dedicated source make the segment visually distinct from measured
        # points.
        for frame in range(left_frame + 1, right_frame):
            index = frame_to_index.get(frame)
            if index is None:
                continue
            alpha = float(frame - left_frame) / span
            xy = left_xy * (1.0 - alpha) + right_xy * alpha
            previous = track[index]
            out[index] = BallPoint(
                frame=int(previous.frame),
                x=float(xy[0]),
                y=float(xy[1]),
                visible=True,
                source="offscreen_bridge",
                raw_visible=False,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
            )

    return out


def _expanded_court_polygon(
    court_polygon,
    top_extension: float = 0.0,
    side_padding: float = 0.0,
    bottom_padding: float = 0.0,
):
    """Build a modest image-space expansion of a TL/TR/BR/BL court quad.

    TrackNet reports the shuttle while it is above the far baseline, so using
    the floor quad literally would discard valid airborne points.  The
    expansion is deliberately conservative and remains a convex quadrilateral
    instead of replacing the court with a large axis-aligned rectangle.
    """
    if court_polygon is None:
        return None
    try:
        poly = np.asarray(court_polygon, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(poly) != 4 or not np.isfinite(poly).all():
        return None

    out = poly.copy()
    top_mid = 0.5 * (poly[0] + poly[1])
    bottom_mid = 0.5 * (poly[2] + poly[3])
    axis = bottom_mid - top_mid
    axis_len = float(np.linalg.norm(axis))
    if axis_len > 1e-6:
        axis /= axis_len
        # The first two vertices are the far baseline; move them opposite the
        # court's top-to-bottom direction to include airborne shuttles.
        out[0] -= axis * max(0.0, float(top_extension))
        out[1] -= axis * max(0.0, float(top_extension))
        # Likewise allow a small amount beyond the near baseline.
        out[2] += axis * max(0.0, float(bottom_padding))
        out[3] += axis * max(0.0, float(bottom_padding))

    # Broadcast quads are usually close to vertical in image space.  A small
    # horizontal expansion is safer than scaling around the centroid (which
    # would also enlarge the airborne region unexpectedly).
    pad = max(0.0, float(side_padding))
    if pad:
        out[0, 0] -= pad
        out[3, 0] -= pad
        out[1, 0] += pad
        out[2, 0] += pad
    return out


def _point_in_expanded_quad(point, polygon) -> bool:
    """Return whether an image point lies in a finite convex quad."""
    if polygon is None or point is None:
        return True
    try:
        poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError):
        return False
    if len(poly) != 4 or not np.isfinite([x, y]).all() or not np.isfinite(poly).all():
        return False
    # Implement the test without importing OpenCV in this dependency-light
    # module.  The sign of each edge cross-product is consistent for a convex
    # quad (clockwise or counter-clockwise).
    p = np.array([x, y], dtype=np.float32)
    cross = []
    for i in range(4):
        a = poly[i]
        b = poly[(i + 1) % 4]
        edge = b - a
        rel = p - a
        cross.append(float(edge[0] * rel[1] - edge[1] * rel[0]))
    eps = 1e-4
    return all(v >= -eps for v in cross) or all(v <= eps for v in cross)


def _point_near_top_edge(
    point,
    polygon,
    perpendicular_tolerance: float = 18.0,
    side_tolerance: float = 40.0,
) -> bool:
    """Allow a labelled edge observation just outside the expanded top edge.

    Kalman smoothing can overshoot a raw ``y=3`` point by a few pixels.  The
    ordinary court gate must remain strict for model/body responses, while a
    dual-phase edge observation may safely use a narrow corridor around the
    far-baseline segment.  This helper checks the segment projection rather
    than accepting an axis-aligned top strip.
    """
    if polygon is None or point is None:
        return False
    try:
        poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        p = np.asarray([float(point[0]), float(point[1])], dtype=np.float64)
    except (TypeError, ValueError, IndexError):
        return False
    if len(poly) != 4 or not np.isfinite(poly).all() or not np.isfinite(p).all():
        return False
    a, b = poly[0], poly[1]
    edge = b - a
    length_sq = float(np.dot(edge, edge))
    if length_sq <= 1e-9:
        return False
    t = float(np.dot(p - a, edge) / length_sq)
    # Permit a modest horizontal extension at either endpoint, but do not
    # turn the entire top of the frame into a valid ball region.
    edge_len = math.sqrt(length_sq)
    side = max(0.0, float(side_tolerance)) / max(edge_len, 1e-6)
    if t < -side or t > 1.0 + side:
        return False
    tolerance = max(0.0, float(perpendicular_tolerance))
    if t < 0.0 or t > 1.0:
        # Within the permitted endpoint extension use perpendicular distance
        # to the infinite line; Euclidean distance to the clamped endpoint
        # would incorrectly reject a point only a few pixels beyond TR/TL.
        cross = abs(float(edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])))
        return cross / max(edge_len, 1e-6) <= tolerance
    projection = a + t * edge
    return float(np.linalg.norm(p - projection)) <= tolerance


def filter_ball_track_geometry(
    track: Sequence[BallPoint],
    court_polygon=None,
    top_extension: float = 60.0,
    side_padding: float = 40.0,
    bottom_padding: float = 30.0,
    scene_cuts: Optional[Sequence[int]] = None,
    scene_cut_padding: int = 0,
    static_window: int = 20,
    static_disp: float = 8.0,
    static_lock_window: int = 16,
    static_lock_disp: float = 35.0,
    static_lock_timeout_window: int = 4,
    static_lock_timeout_disp: float = 10.0,
    static_lock_min_y: float = 80.0,
    static_lock_preroll: int = 2,
    edge_top_tolerance: float = 18.0,
    edge_side_tolerance: float = 100.0,
    edge_reentry_frames: Optional[Sequence[int]] = None,
) -> List[BallPoint]:
    """Reject geometrically implausible or persistent background detections.

    Parameters
    ----------
    track:
        Output of :func:`smooth_ball_track`.  The returned list has the same
        length and frame numbers.
    court_polygon:
        Optional image-space court quad in ``TL, TR, BR, BL`` order.  Points
        outside the expanded quad are marked invisible.  Their ``source`` and
        ``raw_visible`` fields are intentionally retained for diagnostics.
    top_extension, side_padding, bottom_padding:
        Pixel margins used only for the geometry gate.  Keep the top margin
        small (roughly 40--80 px for the supplied 1080p broadcast) so the
        scoreboard/banner is not accepted as a ball region.
    scene_cuts:
        Optional frame indices at hard cuts.  Points within
        ``scene_cut_padding`` frames of a cut are hidden because the previous
        camera's Kalman state is not meaningful after a cut.
    static_window, static_disp:
        A long run of *raw model observations* whose points stay within
        ``static_disp`` pixels is treated as a static background response.
        Interpolated/Kalman points and classical observations are excluded so
        a slow high-clear apex is not erased.  Set ``static_window <= 0`` to
        disable this test.
    static_lock_window, static_lock_disp:
        A second, broader lock gate for a mixed model/classical branch.  Some
        false responses move a few pixels or switch detector source while
        remaining on a player's shoe/net line, so the strict raw-model gate
        above does not catch them.  A lock is removed only after a relatively
        long run remains inside this small image-space radius.
    static_lock_timeout_window, static_lock_timeout_disp:
        Early timeout for a raw model lock.  A compact response below
        ``static_lock_min_y`` which persists for only a few frames is enough
        to identify a player's shoe/body; ``static_lock_preroll`` removes a
        small lead-in before the response becomes fully stationary.  The
        top-edge flight zone is intentionally excluded.
    edge_top_tolerance, edge_side_tolerance:
        Narrow corridor around the expanded far-baseline segment for
        ``model_edge``/``single_phase_edge`` points.  Ordinary model points do
        not receive this allowance.
    edge_reentry_frames:
        Optional, tightly pre-qualified frame ids immediately after a
        top-edge anchor.  These ordinary model points may sit just outside
        the expanded quad while the shuttle visibly re-enters; the caller
        must provide the mask from a temporal continuation test.  No global
        ROI widening is performed here.

    This function does not interpolate or invent points.  It only changes
    ``visible``; all coordinates and provenance fields remain available for
    logging and later threshold tuning.
    """
    if not track:
        return []

    expanded = _expanded_court_polygon(
        court_polygon,
        top_extension=top_extension,
        side_padding=side_padding,
        bottom_padding=bottom_padding,
    )
    n = len(track)
    keep = np.array([bool(p.visible) for p in track], dtype=bool)
    reentry_frames = {int(frame) for frame in (edge_reentry_frames or ())}

    # Geometry gate first.  It also removes Kalman predictions which drifted
    # outside the playing area after a long detector gap.
    if expanded is not None:
        for i, point in enumerate(track):
            if keep[i] and not _point_in_expanded_quad((point.x, point.y), expanded):
                edge_allowed = (
                    str(point.source) in EDGE_MEASUREMENT_SOURCES
                    and float(point.y) <= float(np.max(expanded[:2, 1])) + float(edge_top_tolerance)
                    and _point_near_top_edge(
                        (point.x, point.y),
                        expanded,
                        perpendicular_tolerance=edge_top_tolerance,
                        side_tolerance=edge_side_tolerance,
                    )
                )
                # A narrow temporal continuation mask can safely preserve a
                # few ordinary points just beyond the top-right/top-left
                # polygon edge.  It is intentionally independent of the
                # broad geometric margin: callers must prove an edge anchor,
                # monotone re-entry and a later in-quad confirmation before
                # placing a frame in this set.
                if (
                    int(point.frame) in reentry_frames
                    and str(point.source) == "model"
                ):
                    edge_allowed = True
                if not edge_allowed:
                    keep[i] = False

    # Invalidate a small neighborhood around hard cuts.  Do this after the
    # geometry gate so a caller can pass cuts discovered from the video even
    # when the post-cut camera uses a different court quad.
    if scene_cuts is not None and int(scene_cut_padding) > 0:
        radius = max(0, int(scene_cut_padding))
        for cut in scene_cuts:
            try:
                c = int(cut)
            except (TypeError, ValueError):
                continue
            lo = max(0, c - radius)
            hi = min(n, c + radius + 1)
            keep[lo:hi] = False

    # Remove long, nearly static *raw model* runs.  The previous implementation
    # tested the already smoothed path (including interpolation/Kalman) with an
    # 80 px radius.  That erased ordinary high clears around their apex.  A
    # real static lock is much tighter, and a classical observation in the
    # same window is independent evidence that the location is a shuttle.
    window = max(0, int(static_window))
    if window >= 2 and static_disp > 0:
        disp_limit = float(static_disp)
        static_candidates = np.asarray([
            bool(keep[i])
            and bool(point.raw_visible)
            and str(point.source) == "model"
            for i, point in enumerate(track)
        ], dtype=bool)
        for start in range(0, max(0, n - window + 1)):
            stop = start + window
            if any(
                keep[j]
                and track[j].raw_visible
                and str(track[j].source) == "classical"
                for j in range(start, stop)
            ):
                continue
            valid_window = np.flatnonzero(static_candidates[start:stop]) + start
            # Permit at most two detector holes inside a static lock; this
            # catches banner responses that blink for a frame without making
            # a short, genuinely sparse flight segment disappear.
            if len(valid_window) < max(2, window - 2):
                continue
            pts = np.asarray([[track[j].x, track[j].y] for j in valid_window], dtype=np.float32)
            if not np.isfinite(pts).all():
                continue
            center = np.median(pts, axis=0)
            spread = np.linalg.norm(pts - center, axis=1)
            if float(np.max(spread)) <= disp_limit:
                keep[valid_window] = False

    filtered: List[BallPoint] = []
    for i, point in enumerate(track):
        if keep[i] == bool(point.visible):
            filtered.append(point)
        else:
            # Preserve x/y/source/raw_visible/confidence so diagnostics can
            # distinguish a model point rejected by geometry from a true miss.
            filtered.append(
                BallPoint(
                    frame=int(point.frame),
                    x=float(point.x),
                    y=float(point.y),
                    visible=False,
                    source=str(point.source),
                    raw_visible=bool(point.raw_visible),
                    confidence=float(point.confidence),
                )
            )

    # Mixed-source static locks need a separate, deliberately conservative
    # pass.  The strict gate above intentionally excludes classical points,
    # but a classical fallback can also lock onto a player's shoe/net line
    # near the end of a rally.  Require nearly a full window of real points
    # and a wider-than-strict spread before hiding the run.  Derived points in
    # the same window are hidden too, otherwise a Kalman/interpolation tail
    # would keep drawing the stale lock.
    lock_window = max(0, int(static_lock_window))
    lock_disp = max(0.0, float(static_lock_disp))
    if lock_window >= 2 and lock_disp > 0:
        real_flags = np.asarray([
            bool(keep[i])
            and bool(point.visible)
            and str(point.source) in REAL_MEASUREMENT_SOURCES
            and np.isfinite([point.x, point.y]).all()
            for point in track
        ], dtype=bool)
        # Build genuinely contiguous real-observation runs.  The previous
        # implementation scanned every overlapping 20-frame window; a slow
        # but legitimate flight then satisfied many local windows in turn and
        # the union erased an entire rally.  Each lock below uses one fixed
        # center and extends only while *that same center* remains supported.
        runs = []
        run_start = None
        for index, flag in enumerate(real_flags):
            contiguous = (
                run_start is not None
                and index > 0
                and int(track[index].frame) == int(track[index - 1].frame) + 1
                and not any(
                    int(track[index - 1].frame) < int(cut) <= int(track[index].frame)
                    for cut in (scene_cuts or ())
                )
            )
            if flag and run_start is None:
                run_start = index
            elif flag and run_start is not None and not contiguous:
                runs.append((run_start, index - 1))
                run_start = index
            elif not flag and run_start is not None:
                runs.append((run_start, index - 1))
                run_start = None
        if run_start is not None:
            runs.append((run_start, n - 1))

        max_path = max(40.0, 2.0 * lock_disp)
        lock_radius = max(lock_disp, 1.5 * lock_disp)
        for run_start, run_end in runs:
            scan = run_start
            while scan + lock_window - 1 <= run_end:
                lock = None
                for begin in range(scan, run_end - lock_window + 2):
                    end = begin + lock_window
                    points = np.asarray(
                        [[track[j].x, track[j].y] for j in range(begin, end)],
                        dtype=np.float64,
                    )
                    if not np.isfinite(points).all():
                        continue
                    center = np.median(points, axis=0)
                    spread = np.linalg.norm(points - center, axis=1)
                    path = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
                    # A fixed-center compact run with a short cumulative path
                    # is a lock.  A moving apex may fit the radius locally but
                    # accumulates substantially more path length and survives.
                    if float(np.max(spread)) > lock_disp or path > max_path:
                        continue
                    lock = (begin, end - 1, center)
                    break
                if lock is None:
                    break
                lock_start, lock_end, center = lock
                # Extend only in the same fixed center corridor.  Do not
                # recompute the median for each overlapping window.
                while lock_end + 1 <= run_end:
                    candidate = lock_end + 1
                    distance = float(np.linalg.norm(
                        np.asarray([track[candidate].x, track[candidate].y], dtype=np.float64)
                        - center
                    ))
                    step = float(np.linalg.norm(
                        np.asarray([track[candidate].x, track[candidate].y], dtype=np.float64)
                        - np.asarray([track[lock_end].x, track[lock_end].y], dtype=np.float64)
                    ))
                    if distance > lock_radius or step > max(24.0, lock_disp * 1.5):
                        break
                    lock_end = candidate
                for index in range(lock_start, lock_end + 1):
                    point = filtered[index]
                    if not point.visible:
                        continue
                    distance = float(np.linalg.norm(
                        np.asarray([point.x, point.y], dtype=np.float64) - center
                    ))
                    if distance > lock_radius:
                        continue
                    filtered[index] = BallPoint(
                        frame=int(point.frame),
                        x=float(point.x),
                        y=float(point.y),
                        visible=False,
                        source=str(point.source),
                        raw_visible=bool(point.raw_visible),
                        confidence=float(point.confidence),
                    )
                scan = lock_end + 1
    # Fast raw-model timeout.  The mixed-source gate above intentionally uses
    # a long window and a generous radius to protect real slow flights.  A
    # separate short gate catches the characteristic 3--5 frame convergence
    # onto a player's shoe/torso before the long gate has enough history.  It
    # is restricted to the lower image region; model_edge points near y=0 and
    # ordinary high clears therefore remain eligible for rendering.
    timeout_window = max(0, int(static_lock_timeout_window))
    timeout_disp = max(0.0, float(static_lock_timeout_disp))
    timeout_min_y = float(static_lock_min_y)
    if timeout_window >= 2 and timeout_disp > 0:
        candidates = np.asarray([
            bool(point.visible)
            and bool(point.raw_visible)
            and str(point.source) == "model"
            and np.isfinite([point.x, point.y]).all()
            and float(point.y) >= timeout_min_y
            for point in track
        ], dtype=bool)
        cut_set = sorted({int(cut) for cut in (scene_cuts or ())})

        def cut_between(left_frame: int, right_frame: int) -> bool:
            return any(int(left_frame) < cut <= int(right_frame) for cut in cut_set)

        runs = []
        run_start = None
        for index, flag in enumerate(candidates):
            if flag and run_start is None:
                run_start = index
            elif not flag and run_start is not None:
                runs.append((run_start, index - 1))
                run_start = None
        if run_start is not None:
            runs.append((run_start, len(track) - 1))

        # A four-frame compact patch is not enough evidence by itself: a real
        # shuttle can move less than ten pixels for a few frames around an
        # apex.  Require a longer confirmation run and keep one fixed centre
        # while extending it.  The old implementation extended on *per-step*
        # distance, so any smooth flight was eventually swallowed after the
        # first compact patch (for example f116--148 and f813--823).
        confirm_window = max(6, 2 * timeout_window)
        confirm_radius = max(timeout_disp, timeout_disp * 1.5)
        confirm_path = max(12.0, timeout_disp * 1.5)
        for run_start, run_end in runs:
            if run_end - run_start + 1 < confirm_window:
                continue
            lock = None
            for begin in range(run_start, run_end - confirm_window + 2):
                end = begin + confirm_window - 1
                pts = np.asarray([[track[j].x, track[j].y]
                                  for j in range(begin, end + 1)], dtype=np.float64)
                if not np.isfinite(pts).all():
                    continue
                center = np.median(pts, axis=0)
                spread = np.linalg.norm(pts - center, axis=1)
                path = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
                if float(np.max(spread)) > timeout_disp or path > confirm_path:
                    continue
                lock = (begin, end, center)
                break
            if lock is None:
                continue
            lock_start, lock_end, center = lock
            # Include a short lead-in only when it remains in the same compact
            # corridor.  Never use a large per-step allowance here.
            for _ in range(max(0, int(static_lock_preroll))):
                candidate = lock_start - 1
                if candidate < run_start or cut_between(
                    int(track[candidate].frame), int(track[lock_start].frame)
                ):
                    break
                distance = float(np.linalg.norm(
                    np.asarray([track[candidate].x, track[candidate].y], dtype=np.float64)
                    - center
                ))
                if distance > confirm_radius:
                    break
                lock_start = candidate
            # Continue only while the *fixed* centre remains compact.  A
            # moving shuttle exits this corridor quickly; a shoe/net lock does
            # not, so its stale tail is removed without erasing flight motion.
            while lock_end < run_end:
                candidate = lock_end + 1
                if cut_between(int(track[lock_end].frame), int(track[candidate].frame)):
                    break
                distance = float(np.linalg.norm(
                    np.asarray([track[candidate].x, track[candidate].y], dtype=np.float64)
                    - center
                ))
                if distance > confirm_radius:
                    break
                lock_end = candidate
            for index in range(lock_start, lock_end + 1):
                point = filtered[index]
                if not point.visible:
                    continue
                filtered[index] = BallPoint(
                    frame=int(point.frame),
                    x=float(point.x),
                    y=float(point.y),
                    visible=False,
                    source=str(point.source),
                    raw_visible=bool(point.raw_visible),
                    confidence=float(point.confidence),
                )
    return filtered


def _valid_indices(track: Sequence[BallPoint], allow_kalman: bool = False) -> List[int]:
    allowed = set(MEASUREMENT_SOURCES)
    if allow_kalman:
        allowed.add("kalman")
    return [p.frame for p in track if 0 <= int(p.frame) < len(track)
            and p.visible and p.source in allowed]


def detect_hit_points(
    track: Sequence[BallPoint],
    fps: float,
    homography=None,
    player_positions: Optional[Mapping[int, Mapping[str, Tuple[float, float]]]] = None,
    min_separation_frames: int = 10,
    min_speed_px: float = 7.0,
    proximity_px: float = 260.0,
    break_frames: Optional[Sequence[int]] = None,
) -> List[HitEvent]:
    """Detect likely racket contacts from direction/curvature changes.

    A pure velocity sign flip is too brittle for broadcast footage.  We score
    local curvature, speed change, and direction reversal, then use a robust
    percentile threshold and a refractory window.  If player locations are
    available, a nearby-player bonus is applied rather than requiring strict
    proximity (the ball may be in front of the racket).
    """
    n = len(track)
    if n < 5:
        return []
    fps = max(float(fps), 1.0)
    positions = np.full((n, 2), np.nan, dtype=np.float32)
    usable = np.zeros(n, dtype=bool)
    # ``break_frames`` marks the first frame of a new independent flight
    # segment.  Classical detection is intentionally sparse and may produce
    # adjacent tracklets; derivatives must never span those boundaries.
    segment_starts = np.zeros(n, dtype=np.int32)
    for frame in break_frames or ():
        try:
            index = int(frame)
        except (TypeError, ValueError):
            continue
        if 0 < index < n:
            segment_starts[index] = 1
    segment_ids = np.cumsum(segment_starts)
    for p in track:
        if (0 <= int(p.frame) < n and p.visible and
                p.source in MEASUREMENT_SOURCES and
                p.source not in DERIVED_MEASUREMENT_SOURCES and
                np.isfinite([p.x, p.y]).all()):
            positions[p.frame] = (p.x, p.y)
            usable[p.frame] = True

    # Fill only tiny gaps for derivative computation; events still require the
    # center frame itself to be a real/interpolated point.
    for i in range(1, n - 1):
        if (
            not usable[i]
            and track[i].source not in DERIVED_MEASUREMENT_SOURCES
            and usable[i - 1]
            and usable[i + 1]
            and segment_ids[i - 1] == segment_ids[i] == segment_ids[i + 1]
        ):
            positions[i] = 0.5 * (positions[i - 1] + positions[i + 1])

    speed = np.zeros(n, dtype=np.float32)
    curvature = np.zeros(n, dtype=np.float32)
    reversal = np.zeros(n, dtype=np.float32)
    for i in range(1, n - 1):
        if not (segment_ids[i - 1] == segment_ids[i] == segment_ids[i + 1]):
            continue
        if not np.all(np.isfinite(positions[i - 1:i + 2])):
            continue
        v0 = positions[i] - positions[i - 1]
        v1 = positions[i + 1] - positions[i]
        s0, s1 = float(np.linalg.norm(v0)), float(np.linalg.norm(v1))
        speed[i] = 0.5 * (s0 + s1)
        curvature[i] = float(np.linalg.norm(v1 - v0))
        if s0 > 1e-3 and s1 > 1e-3:
            cosine = float(np.dot(v0, v1) / (s0 * s1))
            reversal[i] = max(0.0, -cosine)

    valid_curvature = curvature[(curvature > 0) & (speed >= min_speed_px)]
    if len(valid_curvature) == 0:
        return []
    # Dynamic threshold adapts to slow net exchanges and fast smashes.
    curvature_threshold = max(10.0, float(np.percentile(valid_curvature, 78)))
    # Do not let the 35th-percentile speed gate suppress a genuine sharp
    # reversal: Kalman lag and a one-frame contact can make the local speed at
    # the contact lower than the surrounding flight speed.  Keep an adaptive
    # floor for noisy tracks, but relax it to roughly 75% of P35.  The
    # curvature/local-maximum gate below still provides the precision filter.
    positive_speed = speed[speed > 0]
    if len(positive_speed) > 0:
        p35 = float(np.percentile(positive_speed, 35))
        # A compact defensive lift can almost stop at the racket before the
        # outgoing flight accelerates.  On the supplied broadcast f608 is
        # exactly this case: its 9.6 px/frame centre speed missed the former
        # 9.62 px/frame adaptive gate by rounding noise, despite a measured
        # direction turn and a nearby racket impulse.  Keep the absolute
        # floor, but use a slightly less aggressive percentile multiplier so
        # those real, low-speed contacts reach the later audio/body evidence
        # gates.  Curvature/local-maximum NMS still rejects ordinary smooth
        # flight points.
        speed_threshold = max(float(min_speed_px), 0.70 * p35)
    else:
        speed_threshold = float(min_speed_px)

    candidates: List[Tuple[HitEvent, int]] = []
    for i in range(2, n - 2):
        if not usable[i] or speed[i] < speed_threshold:
            continue
        local_indices = [
            j for j in range(max(0, i - 2), min(n, i + 3))
            if segment_ids[j] == segment_ids[i]
        ]
        local_max = max((float(curvature[j]) for j in local_indices), default=0.0)
        if curvature[i] < max(curvature_threshold, local_max * 0.72):
            continue
        # Score combines sharpness and direction change.  A local speed change
        # catches contacts where the direction does not fully reverse.
        speed_change = abs(float(speed[i + 1]) - float(speed[i - 1])) / max(float(speed[i]), 1.0)
        score = 0.55 * min(1.0, curvature[i] / max(curvature_threshold, 1.0))
        score += 0.30 * reversal[i] + 0.15 * min(1.0, speed_change)

        hitter = "unknown"
        if player_positions is not None and i in player_positions:
            choices = []
            for side, pt in player_positions[i].items():
                if pt is not None:
                    choices.append((math.hypot(float(pt[0]) - positions[i, 0], float(pt[1]) - positions[i, 1]), side))
            if choices:
                distance, hitter = min(choices)
                if distance <= proximity_px:
                    score += 0.20

        candidates.append((
            HitEvent(
                i,
                float(positions[i, 0]),
                float(positions[i, 1]),
                float(score),
                float(speed[i]),
                hitter=hitter,
                source="curvature",
            ),
            int(segment_ids[i]),
        ))

    # Non-maximum suppression in time: retain the strongest candidate around a
    # contact rather than flashing several rings over adjacent frames.
    candidates.sort(key=lambda item: item[0].score, reverse=True)
    selected: List[Tuple[HitEvent, int]] = []
    for candidate, segment_id in candidates:
        if all(
            segment_id != chosen_segment
            or abs(candidate.frame - chosen.frame)
            >= max(1, int(min_separation_frames))
            for chosen, chosen_segment in selected
        ):
            selected.append((candidate, segment_id))
    events = [event for event, _ in selected]
    events.sort(key=lambda h: h.frame)
    return events


def detect_top_edge_exit_hits(
    track: Sequence[BallPoint],
    min_incoming_points: int = 3,
    min_edge_points: int = 2,
    min_incoming_y_step: float = 4.0,
    min_edge_y_step: float = 10.0,
    break_frames: Optional[Sequence[int]] = None,
) -> List[HitEvent]:
    """Find a contact immediately before a verified top-edge exit.

    A high clear can be struck by the far player and leave the camera within
    one or two frames.  The ordinary TrackNet point at contact is followed by
    deliberately low-confidence ``model_edge`` points, so the regular
    three-point curvature detector has no valid right-hand derivative.  This
    helper recovers only the very specific measured pattern:

    ``ordinary downward approach -> ordinary contact -> >=2 upward edge pts``

    It emits the *last ordinary measured coordinate*, never the edge point or
    a synthetic bridge.  The caller still applies audio corroboration and the
    pose/body veto, which makes this a one-sided endpoint candidate rather
    than an audio-only event.
    """

    n = len(track)
    if n < max(1, int(min_incoming_points)) + max(1, int(min_edge_points)):
        return []

    blocked = set()
    for frame in break_frames or ():
        try:
            blocked.add(int(frame))
        except (TypeError, ValueError):
            continue

    incoming_required = max(2, int(min_incoming_points))
    edge_required = max(2, int(min_edge_points))
    incoming_step = max(0.0, float(min_incoming_y_step))
    edge_step = max(0.0, float(min_edge_y_step))
    candidates: List[HitEvent] = []

    for contact_frame in range(incoming_required - 1, n - edge_required):
        edge_start = contact_frame + 1
        if any(
            contact_frame < boundary <= edge_start
            for boundary in blocked
        ):
            continue
        incoming = track[contact_frame - incoming_required + 1:contact_frame + 1]
        edge = track[edge_start:edge_start + edge_required]
        if not all(
            point.visible
            and point.source in REAL_MEASUREMENT_SOURCES
            and np.isfinite([point.x, point.y]).all()
            for point in incoming
        ):
            continue
        if not all(
            point.visible
            and point.source in EDGE_MEASUREMENT_SOURCES
            and np.isfinite([point.x, point.y]).all()
            for point in edge
        ):
            continue

        incoming_y = np.diff([float(point.y) for point in incoming])
        # The shuttle must arrive at the player while descending in image
        # coordinates, then leave upward through the top boundary.  This
        # excludes a normal uninterrupted high-clear ascent.
        if not np.all(incoming_y >= incoming_step):
            continue
        outgoing_y = [float(point.y) for point in (incoming[-1:] + edge)]
        edge_y = np.diff(outgoing_y)
        if not np.all(edge_y <= -edge_step):
            continue

        contact = incoming[-1]
        incoming_speed = float(np.median(np.abs(incoming_y)))
        outgoing_speed = float(np.median(np.abs(edge_y)))
        score = float(np.clip(
            0.72 + 0.012 * min(15.0, incoming_speed)
            + 0.010 * min(18.0, outgoing_speed),
            0.72,
            0.96,
        ))
        candidates.append(HitEvent(
            frame=int(contact.frame),
            x=float(contact.x),
            y=float(contact.y),
            score=score,
            speed=max(incoming_speed, outgoing_speed),
            source="edge_exit",
            ball_observed=bool(contact.raw_visible),
        ))

    return candidates


def validate_hit_event_evidence(
    track: Sequence[BallPoint],
    events: Sequence[HitEvent],
    audio_frames: Optional[Sequence[int]] = None,
    player_boxes: Optional[Mapping[int, Sequence[Sequence[float]]]] = None,
    require_audio: bool = True,
    audio_tolerance_frames: int = 6,
    evidence_window_frames: int = 10,
    min_real_points: int = 2,
    position_tolerance_px: float = 90.0,
    player_margin_px: float = 8.0,
    player_contacts: Optional[Mapping[int, Sequence[Tuple[float, float]]]] = None,
    contact_tolerance_px: float = 45.0,
) -> Tuple[List[HitEvent], Dict[str, int]]:
    """Keep only hit events supported by the measured shuttle trajectory.

    Pose is deliberately *not* an event source here.  It may identify the
    hitter upstream, but a pose-only wrist/body coordinate must never become
    a rendered contact.  With audio enabled, every exported event also needs
    a nearby racket impulse.  This makes the default policy a true
    trajectory+audio intersection rather than a union of weak cues.

    Curvature and piecewise events require real shuttle observations on both
    sides of the contact.  Track endpoints may be one-sided, but still need at
    least ``min_real_points`` plus audio corroboration.  Any event inside a
    detected player box is rejected instead of merely being labelled
    uncertain; this specifically vetoes TrackNet responses on shirts/heads.

    Returns ``(accepted, rejection_counts)`` so callers can expose the reason
    counts in diagnostics and tests.
    """

    accepted: List[HitEvent] = []
    stats: Dict[str, int] = {"input": len(events), "accepted": 0}
    if not events:
        return accepted, stats

    n = len(track)
    audio = sorted({int(frame) for frame in (audio_frames or ())})
    audio_tolerance = max(0, int(audio_tolerance_frames))
    evidence_window = max(1, int(evidence_window_frames))
    required_points = max(2, int(min_real_points))
    position_tolerance = max(1.0, float(position_tolerance_px))
    player_margin = max(0.0, float(player_margin_px))

    def reject(reason: str) -> None:
        stats[reason] = stats.get(reason, 0) + 1

    def inside_player(point: Tuple[float, float], boxes) -> bool:
        x, y = map(float, point)
        for box in boxes or ():
            try:
                x1, y1, x2, y2 = map(float, box)
            except (TypeError, ValueError):
                continue
            if (
                x1 - player_margin <= x <= x2 + player_margin
                and y1 - player_margin <= y <= y2 + player_margin
            ):
                return True
        return False

    for event in events:
        frame = int(event.frame)
        source = str(getattr(event, "source", ""))
        if source == "audio_pose":
            reject("pose_only")
            continue
        if frame < 0 or frame >= n or not np.isfinite([event.x, event.y]).all():
            reject("invalid_frame_or_position")
            continue
        # A top-edge label is useful for drawing/reacquisition, but there is
        # no image evidence for a racket contact while the shuttle is at the
        # boundary (and an offscreen bridge is explicitly synthetic).  Do not
        # let a hit candidate generated from the independent primary phase
        # turn such a point back into a measured contact after phase fusion.
        event_point = track[frame]
        if str(getattr(event_point, "source", "")) in EDGE_MEASUREMENT_SOURCES \
                or str(getattr(event_point, "source", "")) in DERIVED_MEASUREMENT_SOURCES \
                or str(getattr(event_point, "source", "")) == "offscreen_bridge":
            reject("edge_or_derived_event")
            continue
        if inside_player((float(event.x), float(event.y)), (player_boxes or {}).get(frame, ())):
            contacts = (player_contacts or {}).get(frame, ())
            contact_near = any(
                math.hypot(
                    float(event.x) - float(contact[0]),
                    float(event.y) - float(contact[1]),
                ) <= max(0.0, float(contact_tolerance_px))
                for contact in contacts
                if contact is not None and len(contact) >= 2
            )
            if not contact_near:
                reject("player_overlap")
                continue

        has_audio = any(abs(frame - audio_frame) <= audio_tolerance for audio_frame in audio)
        if require_audio and not has_audio:
            reject("missing_audio")
            continue

        lo = max(0, frame - evidence_window)
        hi = min(n, frame + evidence_window + 1)
        real_points = [
            point for point in track[lo:hi]
            if (
                point.visible
                and point.source in REAL_MEASUREMENT_SOURCES
                and np.isfinite([point.x, point.y]).all()
            )
        ]
        if len(real_points) < required_points:
            reject("insufficient_real_points")
            continue

        before = [point for point in real_points if int(point.frame) < frame]
        after = [point for point in real_points if int(point.frame) > frame]
        strict_bidirectional = source in {"curvature", "classical_piecewise"}
        if strict_bidirectional and (not before or not after or len(real_points) < max(3, required_points)):
            reject("missing_bidirectional_trajectory")
            continue

        # Estimate the shuttle at the event time from real measurements.  A
        # bounded interpolation is preferred; a one-sided estimate is allowed
        # only for endpoint/audio events and remains tied to the closest real
        # observation.
        at_frame = [point for point in real_points if int(point.frame) == frame]
        if at_frame:
            estimate = np.asarray([at_frame[0].x, at_frame[0].y], dtype=np.float64)
        elif before and after:
            left = max(before, key=lambda point: int(point.frame))
            right = min(after, key=lambda point: int(point.frame))
            span = max(1, int(right.frame) - int(left.frame))
            alpha = float(frame - int(left.frame)) / float(span)
            estimate = np.asarray([
                float(left.x) * (1.0 - alpha) + float(right.x) * alpha,
                float(left.y) * (1.0 - alpha) + float(right.y) * alpha,
            ], dtype=np.float64)
        else:
            nearest = min(real_points, key=lambda point: abs(int(point.frame) - frame))
            estimate = np.asarray([nearest.x, nearest.y], dtype=np.float64)

        event_position = np.asarray([event.x, event.y], dtype=np.float64)
        if float(np.linalg.norm(event_position - estimate)) > position_tolerance:
            reject("trajectory_position_mismatch")
            continue

        # ``ball_observed`` is provenance, not caller-supplied confidence.
        event.ball_observed = bool(at_frame)
        accepted.append(event)

    accepted.sort(key=lambda event: int(event.frame))
    stats["accepted"] = len(accepted)
    return accepted, stats


def segment_rallies(
    track: Sequence[BallPoint],
    hits: Sequence[HitEvent],
    fps: float,
    gap_frames: Optional[int] = None,
    scene_cuts: Optional[Sequence[int]] = None,
    min_segment_points: int = 3,
    hard_boundaries: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[Tuple[int, int, int]]]:
    """Assign a rally number to each frame and return ``(start,end,id)``.

    Hard camera cuts are explicit boundaries even when a detector happens to
    emit a point on both sides of the cut.  Real/interpolated observations are
    clustered across detector holes, but a missing shuttle alone is weak
    evidence that play ended: high clears and occlusion commonly create gaps
    of several seconds.  With ``gap_frames`` unset/zero the fallback boundary
    is therefore five seconds.  Hit spacing is intentionally *not* a boundary,
    and short Kalman forecasts cannot create a synthetic rally by themselves.
    """
    n = len(track)
    if n == 0:
        return [], []
    visible_frames = set(_valid_indices(track, allow_kalman=False))
    # Audio/player-supported contacts can legitimately occur while the tiny
    # shuttle is occluded.  They extend rally activity but do not manufacture
    # trajectory coordinates.
    visible_frames.update(
        int(event.frame)
        for event in hits
        if 0 <= int(event.frame) < n
    )
    visible_frames = sorted(visible_frames)
    hard_boundary_set = set()
    for collection in (scene_cuts or (), hard_boundaries or ()):
        for boundary in collection:
            try:
                value = int(boundary)
            except (TypeError, ValueError):
                continue
            if 0 < value < n:
                hard_boundary_set.add(value)

    labels = [0] * n
    segments: List[Tuple[int, int, int]] = []
    if visible_frames:
        requested_gap = int(gap_frames or 0)
        max_gap = (
            requested_gap if requested_gap > 0
            else max(1, int(round(5.0 * max(float(fps), 1.0))))
        )
        clusters: List[List[int]] = [[visible_frames[0]]]
        for frame in visible_frames[1:]:
            previous = clusters[-1][-1]
            crosses_boundary = any(
                previous < boundary <= frame for boundary in hard_boundary_set
            )
            if frame - previous > max_gap or crosses_boundary:
                clusters.append([frame])
            else:
                clusters[-1].append(frame)

        min_points = max(1, int(min_segment_points))
        rally_id = 1
        for cluster in clusters:
            # A one-/two-point island is normally residual detector noise.
            if len(cluster) < min_points:
                continue
            start, end = cluster[0], cluster[-1]
            for frame in range(start, end + 1):
                labels[frame] = rally_id
            segments.append((start, end, rally_id))
            rally_id += 1

    for event in hits:
        if 0 <= event.frame < n:
            event.rally = labels[event.frame]
    return labels, segments


def _estimate_perspective_proxy_speed(
    by_frame: Mapping[int, BallPoint],
    start: int,
    end: int,
    fps: float,
    homography: np.ndarray,
    *,
    floor_y: float,
    min_coverage: float,
    min_real_points: int,
    min_segments: int,
    max_segment_speed_mps: float,
    max_average_speed_mps: float,
    max_scale_m_per_px: float,
) -> Optional[Dict[str, object]]:
    """Fallback speed from locally capped perspective scale.

    A ball above the far baseline cannot be safely projected to the *floor*:
    the ground homography approaches its horizon and makes a few image pixels
    look like tens of metres.  This fallback instead maps the image-space
    displacement through the homography Jacobian evaluated no higher than the
    far baseline.  It retains the camera's perspective scale without turning
    a high clear into an unbounded floor coordinate.

    It is intentionally returned as a distinct, lower-confidence method.
    Callers must label it as an estimate rather than a court-plane average.
    """
    if not math.isfinite(float(floor_y)):
        return None
    elapsed_seconds = float(end - start) / max(float(fps), 1e-6)
    if elapsed_seconds <= 0.0:
        return None

    def visible_xy(point: Optional[BallPoint]):
        if point is None or not point.visible:
            return None
        source = str(getattr(point, "source", ""))
        if source not in MEASUREMENT_SOURCES or source in DERIVED_MEASUREMENT_SOURCES:
            return None
        xy = np.asarray([float(point.x), float(point.y)], dtype=np.float64)
        return xy if np.isfinite(xy).all() else None

    def project(x: float, y: float):
        projected = homography @ np.asarray([x, y, 1.0], dtype=np.float64)
        denominator = float(projected[2])
        if not math.isfinite(denominator) or abs(denominator) <= 1e-9:
            return None
        court_xy = projected[:2] / denominator
        return court_xy if np.isfinite(court_xy).all() else None

    distance_m = 0.0
    observed_seconds = 0.0
    real_points = 0
    segments = 0
    previous_xy = None
    previous_frame = None
    max_segment_speed = max(1.0, float(max_segment_speed_mps))
    max_scale = max(1e-4, float(max_scale_m_per_px))

    for frame in range(start, end + 1):
        point = by_frame.get(frame)
        xy = visible_xy(point)
        if xy is None:
            previous_xy = None
            previous_frame = None
            continue
        if str(getattr(point, "source", "")) in REAL_MEASUREMENT_SOURCES:
            real_points += 1
        if previous_xy is not None and previous_frame == frame - 1:
            midpoint = 0.5 * (previous_xy + xy)
            # Clip the *Jacobian reference* to the far baseline, not the
            # observed point.  The image delta still includes vertical flight
            # motion while the metre-per-pixel scale stays bounded.
            ref_x = float(midpoint[0])
            ref_y = max(float(floor_y), float(midpoint[1]))
            base = project(ref_x, ref_y)
            x_step = project(ref_x + 1.0, ref_y)
            y_step = project(ref_x, ref_y + 1.0)
            if base is None or x_step is None or y_step is None:
                previous_xy = None
                previous_frame = None
                continue
            jacobian = np.column_stack((x_step - base, y_step - base))
            if not np.isfinite(jacobian).all():
                previous_xy = None
                previous_frame = None
                continue
            image_delta = xy - previous_xy
            image_distance = float(np.linalg.norm(image_delta))
            # Cap only the *magnitude* of a local image-to-metre conversion.
            # Clipping singular values would rotate/distort a purely
            # horizontal motion when the unused vertical direction is large.
            step_distance = min(
                float(np.linalg.norm(jacobian @ image_delta)),
                max_scale * image_distance,
            )
            step_seconds = 1.0 / max(float(fps), 1e-6)
            if math.isfinite(step_distance) and step_distance / step_seconds <= max_segment_speed:
                distance_m += step_distance
                observed_seconds += step_seconds
                segments += 1
            else:
                previous_xy = None
                previous_frame = None
                continue
        previous_xy = xy
        previous_frame = frame

    coverage = observed_seconds / max(elapsed_seconds, 1e-6)
    if (
        real_points < max(1, int(min_real_points))
        or segments < max(1, int(min_segments))
        or coverage < max(0.0, float(min_coverage))
        or observed_seconds <= 0.0
    ):
        return None
    speed_mps = distance_m / observed_seconds
    if not math.isfinite(speed_mps) or speed_mps > max(1.0, float(max_average_speed_mps)):
        return None
    return {
        "available": True,
        "speed_mps": float(speed_mps),
        "speed_kmh": float(speed_mps * 3.6),
        "distance_m": float(distance_m),
        "observed_seconds": float(observed_seconds),
        "elapsed_seconds": float(elapsed_seconds),
        "coverage": float(coverage),
        "real_points": int(real_points),
        "segments": int(segments),
        "quality": "proxy",
        "method": "perspective_proxy",
    }


def estimate_shot_average_speed(
    track: Sequence[BallPoint],
    start_frame: int,
    end_frame: int,
    fps: float,
    homography,
    *,
    court_width_m: Optional[float] = None,
    court_length_m: Optional[float] = None,
    court_margin_m: float = 6.0,
    min_coverage: float = 0.55,
    min_real_points: int = 2,
    min_segments: int = 3,
    max_segment_speed_mps: float = 160.0,
    max_average_speed_mps: float = 55.0,
    proxy_floor_y: Optional[float] = None,
    proxy_min_coverage: float = 0.20,
    proxy_min_real_points: int = 8,
    proxy_min_segments: int = 8,
    proxy_max_scale_m_per_px: float = 0.055,
    proxy_max_average_speed_mps: float = 55.0,
) -> Dict[str, object]:
    """Estimate the average shuttle speed for one confirmed flight.

    The estimate covers the interval from one confirmed contact to the next.
    Image positions are projected onto the calibrated court plane and only
    adjacent, measured/interpolated frames are accumulated.  In particular,
    Kalman forecasts, top-edge observations, render-only bridges, and a gap
    between valid points are never turned into a distance segment.

    A shuttle is airborne, while the supplied homography describes the court
    floor.  The result is therefore deliberately an *estimate of projected
    average speed*, not a radar-grade 3-D shuttle velocity.  When floor-plane
    coverage is insufficient, an optional perspective-capped fallback can
    still estimate a flight from its observed image motion.  Its returned
    ``method`` is ``perspective_proxy`` so callers can label it accordingly.
    """
    result: Dict[str, object] = {
        "available": False,
        "speed_mps": None,
        "speed_kmh": None,
        "distance_m": 0.0,
        "observed_seconds": 0.0,
        "elapsed_seconds": 0.0,
        "coverage": 0.0,
        "real_points": 0,
        "segments": 0,
        "quality": "invalid_window",
        "method": "none",
    }

    try:
        start = int(start_frame)
        end = int(end_frame)
        fps_value = max(float(fps), 1e-6)
        matrix = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        result["quality"] = "invalid_homography"
        return result
    if end <= start:
        return result
    if not np.isfinite(matrix).all():
        result["quality"] = "invalid_homography"
        return result

    elapsed_seconds = float(end - start) / fps_value
    result["elapsed_seconds"] = elapsed_seconds
    if elapsed_seconds <= 0.0:
        return result

    by_frame = {
        int(point.frame): point
        for point in track
        if start <= int(point.frame) <= end
    }
    margin = max(0.0, float(court_margin_m))
    width = None if court_width_m is None else max(0.0, float(court_width_m))
    length = None if court_length_m is None else max(0.0, float(court_length_m))

    def project(point: Optional[BallPoint]):
        if point is None or not point.visible:
            return None
        source = str(getattr(point, "source", ""))
        if source not in MEASUREMENT_SOURCES or source in DERIVED_MEASUREMENT_SOURCES:
            return None
        xy = np.asarray([float(point.x), float(point.y), 1.0], dtype=np.float64)
        if not np.isfinite(xy).all():
            return None
        projected = matrix @ xy
        denominator = float(projected[2])
        if not math.isfinite(denominator) or abs(denominator) <= 1e-9:
            return None
        court_xy = projected[:2] / denominator
        if not np.isfinite(court_xy).all():
            return None
        # Near the camera horizon an image-to-floor homography grows without
        # bound.  Excluding that zone prevents a single high-clear pixel from
        # producing hundreds of metres of fake distance.
        if width is not None and width > 0.0:
            if court_xy[0] < -margin or court_xy[0] > width + margin:
                return None
        if length is not None and length > 0.0:
            if court_xy[1] < -margin or court_xy[1] > length + margin:
                return None
        return court_xy

    distance_m = 0.0
    observed_seconds = 0.0
    real_points = 0
    segments = 0
    previous_xy = None
    previous_frame = None
    max_segment_speed = max(1.0, float(max_segment_speed_mps))

    for frame in range(start, end + 1):
        point = by_frame.get(frame)
        court_xy = project(point)
        if court_xy is None:
            previous_xy = None
            previous_frame = None
            continue

        if str(getattr(point, "source", "")) in REAL_MEASUREMENT_SOURCES:
            real_points += 1
        if previous_xy is not None and previous_frame == frame - 1:
            step_distance = float(np.linalg.norm(court_xy - previous_xy))
            step_seconds = 1.0 / fps_value
            if math.isfinite(step_distance) and step_distance / step_seconds <= max_segment_speed:
                distance_m += step_distance
                observed_seconds += step_seconds
                segments += 1
            else:
                # Do not let an implausible projected jump join the next
                # observation either.  It is usually a detector switch or a
                # high-flight point close to the homography horizon.
                previous_xy = None
                previous_frame = None
                continue
        previous_xy = court_xy
        previous_frame = frame

    coverage_ratio = observed_seconds / max(elapsed_seconds, 1e-6)
    result.update({
        "distance_m": float(distance_m),
        "observed_seconds": float(observed_seconds),
        "coverage": float(coverage_ratio),
        "real_points": int(real_points),
        "segments": int(segments),
    })
    if real_points < max(1, int(min_real_points)):
        result["quality"] = "insufficient_real_points"
    elif segments < max(1, int(min_segments)):
        result["quality"] = "insufficient_segments"
    elif coverage_ratio < max(0.0, float(min_coverage)):
        result["quality"] = "insufficient_coverage"
    elif observed_seconds <= 0.0:
        result["quality"] = "insufficient_segments"
    else:
        speed_mps = distance_m / observed_seconds
        if math.isfinite(speed_mps) and speed_mps <= max(1.0, float(max_average_speed_mps)):
            result.update({
                "available": True,
                "speed_mps": float(speed_mps),
                "speed_kmh": float(speed_mps * 3.6),
                "quality": "ok",
                "method": "court_plane",
            })
            return result
        result["quality"] = "average_speed_outlier"

    # A high clear above the far baseline has useful consecutive image motion
    # but cannot be projected directly onto the court floor.  Try the capped
    # perspective proxy only after the stricter court-plane estimate declines
    # the flight, and only when the caller supplied the calibrated baseline.
    if proxy_floor_y is not None:
        proxy = _estimate_perspective_proxy_speed(
            by_frame,
            start,
            end,
            fps_value,
            matrix,
            floor_y=float(proxy_floor_y),
            min_coverage=proxy_min_coverage,
            min_real_points=proxy_min_real_points,
            min_segments=proxy_min_segments,
            max_segment_speed_mps=max_segment_speed_mps,
            max_average_speed_mps=proxy_max_average_speed_mps,
            max_scale_m_per_px=proxy_max_scale_m_per_px,
        )
        if proxy is not None:
            proxy["primary_quality"] = result["quality"]
            return proxy
    return result


def coverage(track: Sequence[BallPoint]) -> Tuple[float, float, float]:
    """Return total, real-measurement and interpolated coverage percentages."""
    if not track:
        return 0.0, 0.0, 0.0
    n = float(len(track))
    real = sum(1 for p in track if p.visible and p.source in REAL_MEASUREMENT_SOURCES) / n
    interp = sum(1 for p in track if p.visible and p.source in ("interp", "kalman")) / n
    return real + interp, real, interp
