"""Conservative CPU fallback for detecting a badminton shuttle.

The learned TrackNet checkpoint used by this repository can be a poor domain
match for some 1080p broadcast footage.  This module provides an intentionally
high-precision fallback which needs only OpenCV and NumPy:

* three-frame motion differencing rejects persistent court/player texture;
* bright, low-saturation, small connected components form observations;
* constant-velocity greedy association builds short candidate trajectories;
* strict appearance and trajectory-smoothness gates remove almost all noise.

This is not intended to replace TrackNet.  It deliberately returns sparse
observations which can be fused with model detections and then passed through
the existing Kalman smoother.  Image-space court calibration is valid only for
one camera view, so processing stops at the first supplied scene cut.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ClassicalBallObservation:
    """One high-precision shuttle observation in source-video coordinates."""

    frame: int
    x: float
    y: float
    confidence: float
    source: str = "classical"
    track_id: int = -1
    quality: float = 0.0

    def as_ball_tuple(self) -> Tuple[int, int, int, str, float]:
        """Return the observation tuple consumed by the Kalman code.

        Source and confidence are retained so downstream rendering never
        mistakes a classical fallback point for a TrackNet model output.
        """

        return (
            1,
            int(round(self.x)),
            int(round(self.y)),
            self.source,
            float(self.confidence),
        )


@dataclass(frozen=True)
class ClassicalTrackRange:
    """Frame range and score of a trajectory which passed all strict gates."""

    track_id: int
    start_frame: int
    end_frame: int
    point_count: int
    score: float


@dataclass(frozen=True)
class ClassicalHitCandidate:
    """A weak hit candidate at the endpoint of a classical trajectory.

    A track ending is not proof of racket contact.  Callers should combine
    these candidates with the normal curvature/player-proximity hit detector.
    ``score`` is normalized to roughly ``[0, 1]`` and ``speed`` is pixels per
    frame at the final observed segment.
    """

    frame: int
    x: float
    y: float
    score: float
    speed: float
    track_id: int = -1


@dataclass(frozen=True)
class ClassicalBallDetectionResult:
    """Public result returned by :func:`detect_classical_ball`."""

    frame_count: int
    observations: Dict[int, ClassicalBallObservation]
    track_ranges: List[ClassicalTrackRange]
    hit_candidates: List[ClassicalHitCandidate]

    def to_ball_dict(self) -> Dict[int, Tuple[int, int, int, str, float]]:
        """Convert observations to the dictionary consumed by Kalman code."""

        return {
            int(frame): observation.as_ball_tuple()
            for frame, observation in self.observations.items()
        }


@dataclass(frozen=True)
class ClassicalBallDetectorConfig:
    """Thresholds for the conservative classical detector.

    Defaults are tuned for the repository's 1920x1080 ``output.mp4``.  Pixel
    thresholds may need scaling for substantially different resolutions.
    """

    # Expanded TL/TR/BR/BL court polygon.  A shuttle can be airborne above the
    # far baseline, hence the much larger top extension.
    court_top_extension: float = 120.0
    court_side_padding: float = 40.0
    court_bottom_padding: float = 30.0

    # Keep the conservative 120 px pass as the precision anchor, then scan a
    # second, taller ROI for ballistic flights which rise into the stands or
    # leave the image.  Set this switch to False for the original one-pass
    # behaviour.  The wider pass is intentionally not allowed to contribute
    # ordinary lower-court tracks, where moving player texture is abundant.
    high_flight_recall_enabled: bool = True
    high_flight_top_extension: float = 480.0
    high_flight_edge_margin: float = 24.0
    high_flight_abrupt_acceleration: float = 28.0
    high_flight_extension_frames: int = 40

    # A strict global mask is essential for seed precision, but it discards a
    # blurred shuttle as soon as its brightness/contrast drops.  Build a second
    # pool with looser per-pixel/component gates and expose it *only* to the
    # local ballistic extension step.  These candidates can never start a
    # trajectory by themselves, which keeps player/audience motion from
    # becoming a new ball track.
    conditioned_recall_enabled: bool = True
    conditioned_motion_threshold: int = 6
    conditioned_gray_threshold: int = 90
    conditioned_saturation_threshold: int = 165
    conditioned_component_max_area: int = 260
    conditioned_candidates_per_frame: int = 300
    conditioned_min_quality: float = 1.75
    conditioned_min_motion: float = 20.0
    conditioned_min_contrast: float = 8.0
    conditioned_dedup_radius: float = 8.0
    conditioned_max_misses: int = 5

    # Three-frame motion/appearance mask.
    motion_threshold: int = 12
    gray_threshold: int = 145
    saturation_threshold: int = 105

    # Connected-component shape gates.
    component_min_area: int = 3
    component_max_area: int = 150
    component_min_width: int = 2
    component_max_width: int = 28
    component_min_height: int = 2
    component_max_height: int = 28
    component_min_density: float = 0.22
    component_min_aspect: float = 0.18
    candidates_per_frame: int = 50

    # Greedy constant-velocity association.
    track_seed_quality: float = 1.35
    association_max_gap: int = 2
    active_track_cap: int = 200

    # Strict final trajectory gates.
    track_min_points: int = 8
    # The earlier 4.2 prototype admitted a long, smooth player-head track in
    # the supplied broadcast (frames 33-80).  4.45 removes that response while
    # retaining all seven higher-quality shuttle flight segments.
    track_min_rank: float = 4.45
    track_min_quality: float = 2.05
    track_min_aspect: float = 0.66
    track_min_density: float = 0.53
    track_min_contrast: float = 28.0
    track_min_fill: float = 0.62
    track_max_acceleration_p80: float = 10.0
    track_max_turn_p80: float = 36.0
    track_max_path_displacement: float = 2.5
    temporal_nms_overlap: float = 0.42


@dataclass(frozen=True)
class _Candidate:
    frame: int
    x: float
    y: float
    quality: float
    area: int
    width: int
    height: int
    density: float
    aspect: float
    motion: float
    contrast: float
    value: float


@dataclass
class _WorkingTrack:
    track_id: int
    points: List[_Candidate]
    last_frame: int
    velocity: Tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class _TrackMetrics:
    track: _WorkingTrack
    rank: float
    median_speed: float
    mean_quality: float
    mean_aspect: float
    mean_density: float
    mean_contrast: float
    spread: float
    fill: float
    acceleration_p80: float
    turn_p80: float
    path_displacement: float

    @property
    def start_frame(self) -> int:
        return self.track.points[0].frame

    @property
    def end_frame(self) -> int:
        return self.track.points[-1].frame

    @property
    def point_count(self) -> int:
        return len(self.track.points)


class _GreedyTracker:
    """Small constant-velocity tracker matching the tuned CPU prototype."""

    def __init__(self, config: ClassicalBallDetectorConfig):
        self.config = config
        self.tracks: List[_WorkingTrack] = []
        self.active: List[int] = []
        self.next_id = 0

    def update(self, frame: int, candidates: Sequence[_Candidate]) -> None:
        max_gap = max(1, int(self.config.association_max_gap))
        self.active = [
            index
            for index in self.active
            if frame - self.tracks[index].last_frame <= max_gap
        ]

        proposals: List[Tuple[float, int, int, int]] = []
        for track_index in self.active:
            track = self.tracks[track_index]
            gap = frame - track.last_frame
            if gap <= 0:
                continue
            vx, vy = track.velocity
            last = track.points[-1]
            predicted_x = last.x + vx * gap
            predicted_y = last.y + vy * gap
            gate = 25.0 + 25.0 * gap + min(60.0, math.hypot(vx, vy) * 0.7)
            for candidate_index, candidate in enumerate(candidates):
                distance = math.hypot(
                    candidate.x - predicted_x,
                    candidate.y - predicted_y,
                )
                if distance > gate:
                    continue
                area_cost = 0.12 * abs(
                    math.log((candidate.area + 1.0) / (last.area + 1.0))
                )
                cost = distance / gate + area_cost - 0.10 * candidate.quality
                proposals.append((cost, track_index, candidate_index, gap))

        proposals.sort()
        used_tracks = set()
        used_candidates = set()
        for _, track_index, candidate_index, gap in proposals:
            if track_index in used_tracks or candidate_index in used_candidates:
                continue
            track = self.tracks[track_index]
            candidate = candidates[candidate_index]
            old = track.points[-1]
            measured_velocity = (
                (candidate.x - old.x) / gap,
                (candidate.y - old.y) / gap,
            )
            track.velocity = (
                0.55 * track.velocity[0] + 0.45 * measured_velocity[0],
                0.55 * track.velocity[1] + 0.45 * measured_velocity[1],
            )
            track.points.append(candidate)
            track.last_frame = frame
            used_tracks.add(track_index)
            used_candidates.add(candidate_index)

        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in used_candidates:
                continue
            if candidate.quality < float(self.config.track_seed_quality):
                continue
            self.tracks.append(
                _WorkingTrack(
                    track_id=self.next_id,
                    points=[candidate],
                    last_frame=frame,
                )
            )
            self.active.append(len(self.tracks) - 1)
            self.next_id += 1

        cap = max(1, int(self.config.active_track_cap))
        if len(self.active) > cap:
            self.active = sorted(
                self.active,
                key=lambda index: (
                    len(self.tracks[index].points),
                    float(
                        np.mean(
                            [
                                point.quality
                                for point in self.tracks[index].points[-5:]
                            ]
                        )
                    ),
                ),
                reverse=True,
            )[:cap]


def _coerce_config(
    config: Optional[ClassicalBallDetectorConfig], overrides: Mapping[str, object]
) -> ClassicalBallDetectorConfig:
    result = config or ClassicalBallDetectorConfig()
    if not isinstance(result, ClassicalBallDetectorConfig):
        raise TypeError("config must be a ClassicalBallDetectorConfig")
    if not overrides:
        return result
    valid = {field.name for field in fields(ClassicalBallDetectorConfig)}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        raise TypeError(f"unknown detector option(s): {', '.join(unknown)}")
    return replace(result, **overrides)


def _first_scene_cut(scene_cuts: Optional[Iterable[int]]) -> Optional[int]:
    if scene_cuts is None:
        return None
    if isinstance(scene_cuts, (int, np.integer)):
        values = [int(scene_cuts)]
    else:
        values = []
        for value in scene_cuts:
            try:
                values.append(int(value))
            except (TypeError, ValueError):
                continue
    positive = [value for value in values if value >= 0]
    return min(positive) if positive else None


def _expanded_court_polygon(
    court_points: Sequence[Sequence[float]],
    config: ClassicalBallDetectorConfig,
) -> np.ndarray:
    try:
        polygon = np.asarray(court_points, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError) as exc:
        raise ValueError("court_points must contain TL,TR,BR,BL image points") from exc
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError("court_points must contain four finite TL,TR,BR,BL points")

    expanded = polygon.copy()
    top_midpoint = 0.5 * (polygon[0] + polygon[1])
    bottom_midpoint = 0.5 * (polygon[2] + polygon[3])
    court_axis = bottom_midpoint - top_midpoint
    length = float(np.linalg.norm(court_axis))
    if length > 1e-6:
        court_axis /= length
        expanded[:2] -= court_axis * max(0.0, float(config.court_top_extension))
        expanded[2:] += court_axis * max(0.0, float(config.court_bottom_padding))
    side = max(0.0, float(config.court_side_padding))
    expanded[[0, 3], 0] -= side
    expanded[[1, 2], 0] += side
    return expanded


def _court_mask_and_roi(
    shape: Tuple[int, int], polygon: np.ndarray
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    # Keep integer truncation to match the tuned prototype exactly.
    integer_polygon = polygon.astype(np.int32)
    cv2.fillConvexPoly(mask, integer_polygon, 255)
    x, y, roi_width, roi_height = cv2.boundingRect(integer_polygon)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + roi_width)
    y1 = min(height, y + roi_height)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("expanded court polygon lies outside the video frame")
    return mask[y0:y1, x0:x1], (x0, y0, x1, y1)


def _extract_candidates(
    frame: int,
    previous_gray: np.ndarray,
    current_bgr: np.ndarray,
    current_gray: np.ndarray,
    next_gray: np.ndarray,
    court_roi: np.ndarray,
    roi: Tuple[int, int, int, int],
    config: ClassicalBallDetectorConfig,
) -> List[_Candidate]:
    x0, y0, x1, y1 = roi
    previous = previous_gray[y0:y1, x0:x1]
    current = current_gray[y0:y1, x0:x1]
    following = next_gray[y0:y1, x0:x1]
    saturation = cv2.cvtColor(
        current_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV
    )[:, :, 1]

    # A valid transient must differ from both adjacent frames.  Using the
    # minimum rejects pixels which merely move into or out of a large player.
    motion = np.minimum(
        cv2.absdiff(current, previous),
        cv2.absdiff(current, following),
    )
    binary = (
        (motion > int(config.motion_threshold))
        & (current > int(config.gray_threshold))
        & (saturation < int(config.saturation_threshold))
        & (court_roi > 0)
    ).astype(np.uint8)

    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, 8
    )
    candidates: List[_Candidate] = []
    frame_height, frame_width = current_gray.shape[:2]
    for label in range(1, label_count):
        local_x, local_y, width, height, area = map(int, stats[label])
        if not (
            int(config.component_min_area)
            <= area
            <= int(config.component_max_area)
        ):
            continue
        if not (
            int(config.component_min_width)
            <= width
            <= int(config.component_max_width)
            and int(config.component_min_height)
            <= height
            <= int(config.component_max_height)
        ):
            continue
        density = area / float(width * height)
        aspect = min(width, height) / float(max(width, height))
        if density < float(config.component_min_density):
            continue
        if aspect < float(config.component_min_aspect):
            continue

        region = (
            labels[
                local_y : local_y + height,
                local_x : local_x + width,
            ]
            == label
        )
        component_motion = motion[
            local_y : local_y + height,
            local_x : local_x + width,
        ][region]
        component_value = current[
            local_y : local_y + height,
            local_x : local_x + width,
        ][region]
        mean_motion = float(component_motion.mean())
        mean_value = float(component_value.mean())

        global_x = local_x + x0
        global_y = local_y + y0
        patch_x0 = max(0, global_x - 4)
        patch_y0 = max(0, global_y - 4)
        patch_x1 = min(frame_width, global_x + width + 4)
        patch_y1 = min(frame_height, global_y + height + 4)
        local_background = float(
            np.median(current_gray[patch_y0:patch_y1, patch_x0:patch_x1])
        )
        contrast = mean_value - local_background
        shape = math.sqrt(max(0.0, aspect * density))
        quality = (
            0.023 * mean_motion
            + 0.014 * max(0.0, contrast)
            + 0.0025 * mean_value
            + shape
            - 0.004 * abs(area - 45)
        )
        centroid_x, centroid_y = map(float, centroids[label])
        candidates.append(
            _Candidate(
                frame=frame,
                x=centroid_x + x0,
                y=centroid_y + y0,
                quality=quality,
                area=area,
                width=width,
                height=height,
                density=density,
                aspect=aspect,
                motion=mean_motion,
                contrast=contrast,
                value=mean_value,
            )
        )

    candidates.sort(key=lambda candidate: candidate.quality, reverse=True)
    limit = max(1, int(config.candidates_per_frame))
    return candidates[:limit]


def _merge_conditioned_candidates(
    strict: Sequence[_Candidate],
    relaxed: Sequence[_Candidate],
    config: ClassicalBallDetectorConfig,
) -> List[_Candidate]:
    """Add weak candidates for local extension without weakening track seeds.

    The relaxed mask often represents a motion-blurred shuttle as a larger
    component.  It is therefore intentionally retained separately from the
    strict tracker and only merged into ``wide_candidates_by_frame``, which is
    queried inside a small physics-predicted gate.  Near-duplicate strict
    components win because their centroid and appearance are more reliable.
    """

    result = list(strict)
    radius = max(1.0, float(config.conditioned_dedup_radius))
    for candidate in relaxed:
        if (
            candidate.quality < float(config.conditioned_min_quality)
            or candidate.motion < float(config.conditioned_min_motion)
            or candidate.contrast < float(config.conditioned_min_contrast)
        ):
            continue
        if any(
            math.hypot(candidate.x - existing.x, candidate.y - existing.y)
            <= radius
            for existing in result
        ):
            continue
        result.append(candidate)
    return result


def _percentile80(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), 80))


def _measure_track(
    track: _WorkingTrack,
    config: ClassicalBallDetectorConfig,
) -> Optional[_TrackMetrics]:
    points = track.points
    # Keep enough samples for two velocity changes.  The final configurable
    # gate remains stricter by default, while recall-oriented high-flight
    # passes may retain a short three-/four-frame seed for later expansion.
    if len(points) < 3:
        return None

    speeds: List[float] = []
    velocities: List[np.ndarray] = []
    for previous, current in zip(points, points[1:]):
        dt = current.frame - previous.frame
        if dt <= 0:
            continue
        velocity = np.asarray(
            [(current.x - previous.x) / dt, (current.y - previous.y) / dt],
            dtype=np.float32,
        )
        velocities.append(velocity)
        speeds.append(float(np.linalg.norm(velocity)))
    if not speeds:
        return None

    xs = [point.x for point in points]
    ys = [point.y for point in points]
    spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
    median_speed = float(np.median(np.asarray(speeds, dtype=np.float32)))
    mean_quality = float(np.mean([point.quality for point in points]))
    mean_aspect = float(np.mean([point.aspect for point in points]))
    mean_density = float(np.mean([point.density for point in points]))
    mean_contrast = float(np.mean([point.contrast for point in points]))
    span = points[-1].frame - points[0].frame + 1
    fill = len(points) / float(max(1, span))
    rank = (
        mean_quality
        + 0.05 * min(len(points), 20)
        + 0.01 * min(spread, 100.0)
        - 0.03 * abs(median_speed - 12.0)
        + 0.3 * fill
    )

    # Loose trajectory prefilter used before the stricter appearance/motion
    # checks.  It also removes effectively stationary line/court responses.
    if not (2.5 <= median_speed <= 90.0):
        return None
    if spread < 25.0 or fill < 0.55:
        return None

    accelerations = [
        float(np.linalg.norm(current - previous))
        for previous, current in zip(velocities, velocities[1:])
    ]
    turns: List[float] = []
    for previous, current in zip(velocities, velocities[1:]):
        previous_norm = float(np.linalg.norm(previous))
        current_norm = float(np.linalg.norm(current))
        if previous_norm <= 1.0 or current_norm <= 1.0:
            continue
        cosine = float(
            np.dot(previous, current) / (previous_norm * current_norm)
        )
        turns.append(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))

    endpoint_displacement = math.hypot(
        points[-1].x - points[0].x,
        points[-1].y - points[0].y,
    )
    # This prototype metric sums per-frame velocity magnitudes.  Retaining the
    # exact definition keeps the tuned 2.5 threshold stable when a track has
    # an allowed two-frame association gap.
    velocity_path = sum(float(np.linalg.norm(velocity)) for velocity in velocities)
    path_displacement = velocity_path / max(1.0, endpoint_displacement)

    return _TrackMetrics(
        track=track,
        rank=rank,
        median_speed=median_speed,
        mean_quality=mean_quality,
        mean_aspect=mean_aspect,
        mean_density=mean_density,
        mean_contrast=mean_contrast,
        spread=spread,
        fill=fill,
        acceleration_p80=_percentile80(accelerations),
        turn_p80=_percentile80(turns),
        path_displacement=path_displacement,
    )


def _passes_strict_gates(
    metrics: _TrackMetrics,
    config: ClassicalBallDetectorConfig,
    stop_frame: Optional[int],
) -> bool:
    if stop_frame is not None and metrics.end_frame >= stop_frame:
        return False
    return bool(
        metrics.point_count >= int(config.track_min_points)
        and metrics.rank >= float(config.track_min_rank)
        and metrics.mean_quality >= float(config.track_min_quality)
        and metrics.mean_aspect >= float(config.track_min_aspect)
        and metrics.mean_density >= float(config.track_min_density)
        and metrics.mean_contrast >= float(config.track_min_contrast)
        and metrics.fill >= float(config.track_min_fill)
        and metrics.acceleration_p80
        <= float(config.track_max_acceleration_p80)
        and metrics.turn_p80 <= float(config.track_max_turn_p80)
        and metrics.path_displacement
        <= float(config.track_max_path_displacement)
    )


def _temporal_nms(
    tracks: Sequence[_TrackMetrics], config: ClassicalBallDetectorConfig
) -> List[_TrackMetrics]:
    selected: List[_TrackMetrics] = []
    overlap_threshold = max(0.0, float(config.temporal_nms_overlap))
    for candidate in sorted(tracks, key=lambda item: item.rank, reverse=True):
        candidate_span = candidate.end_frame - candidate.start_frame + 1
        reject = False
        for existing in selected:
            overlap = (
                min(candidate.end_frame, existing.end_frame)
                - max(candidate.start_frame, existing.start_frame)
                + 1
            )
            if overlap <= 0:
                continue
            existing_span = existing.end_frame - existing.start_frame + 1
            if overlap >= overlap_threshold * min(candidate_span, existing_span):
                reject = True
                break
        if not reject:
            selected.append(candidate)
    return selected


def _track_temporal_overlap(
    first: _TrackMetrics,
    second: _TrackMetrics,
) -> float:
    """Return temporal overlap relative to the shorter track span."""

    overlap = (
        min(first.end_frame, second.end_frame)
        - max(first.start_frame, second.start_frame)
        + 1
    )
    if overlap <= 0:
        return 0.0
    first_span = first.end_frame - first.start_frame + 1
    second_span = second.end_frame - second.start_frame + 1
    return overlap / float(max(1, min(first_span, second_span)))


def _passes_high_flight_edge_gates(
    metrics: _TrackMetrics,
    config: ClassicalBallDetectorConfig,
    stop_frame: Optional[int],
) -> bool:
    """Accept a short, smooth track which genuinely reaches the top edge.

    A shuttle travelling at broadcast speed may be visible for only five
    frames before leaving the image.  Such a track cannot pass the ordinary
    eight-point gate.  Requiring it to touch the image edge, however, permits
    tighter appearance and ballistic-motion gates.  These values recovered
    all five short/elongated flights in the reference clip without accepting
    a player trajectory.
    """

    if stop_frame is not None and metrics.end_frame >= stop_frame:
        return False
    minimum_y = min(point.y for point in metrics.track.points)
    return bool(
        metrics.point_count >= 5
        and minimum_y <= max(0.0, float(config.high_flight_edge_margin))
        and metrics.mean_quality >= 3.0
        and metrics.mean_aspect >= 0.50
        and metrics.mean_density >= 0.48
        and metrics.mean_contrast >= 35.0
        and metrics.fill >= 0.85
        and metrics.acceleration_p80 <= 10.5
        and metrics.turn_p80 <= 15.0
        and metrics.path_displacement <= 1.25
    )


def _clone_working_track(
    track: _WorkingTrack,
    points: Sequence[_Candidate],
    track_id: Optional[int] = None,
) -> _WorkingTrack:
    copied = list(points)
    velocity = (0.0, 0.0)
    if len(copied) >= 2:
        previous, current = copied[-2:]
        dt = current.frame - previous.frame
        if dt > 0:
            velocity = (
                (current.x - previous.x) / dt,
                (current.y - previous.y) / dt,
            )
    return _WorkingTrack(
        track_id=track.track_id if track_id is None else int(track_id),
        points=copied,
        last_frame=copied[-1].frame if copied else track.last_frame,
        velocity=velocity,
    )


def _trim_abrupt_tail(
    track: _WorkingTrack,
    config: ClassicalBallDetectorConfig,
) -> _WorkingTrack:
    """Stop a trusted high-flight track before a sudden player/body pickup."""

    points = track.points
    if len(points) < 5:
        return _clone_working_track(track, points)

    kept = list(points[:3])
    velocities: List[np.ndarray] = []
    for previous, current in zip(kept, kept[1:]):
        dt = current.frame - previous.frame
        velocities.append(
            np.asarray(
                [(current.x - previous.x) / dt, (current.y - previous.y) / dt],
                dtype=np.float32,
            )
        )

    acceleration_limit = max(
        1.0, float(config.high_flight_abrupt_acceleration)
    )
    for point in points[3:]:
        previous_point = kept[-1]
        dt = point.frame - previous_point.frame
        if dt <= 0:
            continue
        velocity = np.asarray(
            [
                (point.x - previous_point.x) / dt,
                (point.y - previous_point.y) / dt,
            ],
            dtype=np.float32,
        )
        previous_velocity = velocities[-1]
        acceleration = float(np.linalg.norm(velocity - previous_velocity))
        previous_speed = float(np.linalg.norm(previous_velocity))
        speed = float(np.linalg.norm(velocity))
        turn = 0.0
        if previous_speed > 1.0 and speed > 1.0:
            cosine = float(
                np.dot(previous_velocity, velocity) / (previous_speed * speed)
            )
            turn = math.degrees(
                math.acos(float(np.clip(cosine, -1.0, 1.0)))
            )

        # Do not treat the natural low-speed reversal at a parabolic apex as
        # a discontinuity.  A large acceleration, or a high-speed reversal,
        # is instead characteristic of the greedy tracker jumping to a body.
        if acceleration > acceleration_limit or (
            turn > 65.0 and previous_speed > 10.0 and speed > 10.0
        ):
            break
        kept.append(point)
        velocities.append(velocity)

    return _clone_working_track(track, kept)


def _predict_track_position(
    points: Sequence[_Candidate],
    frame: int,
) -> Tuple[float, float]:
    """Fit a local ballistic curve and extrapolate it to one nearby frame."""

    recent = list(points[-min(7, len(points)) :])
    frames = np.asarray([point.frame for point in recent], dtype=np.float64)
    origin = float(frames[-1])
    relative_frames = frames - origin
    degree = 2 if len(recent) >= 5 else 1
    target = float(frame) - origin
    x_coefficients = np.polyfit(
        relative_frames,
        np.asarray([point.x for point in recent], dtype=np.float64),
        degree,
    )
    y_coefficients = np.polyfit(
        relative_frames,
        np.asarray([point.y for point in recent], dtype=np.float64),
        degree,
    )
    return (
        float(np.polyval(x_coefficients, target)),
        float(np.polyval(y_coefficients, target)),
    )


def _extend_track_forward(
    points: Sequence[_Candidate],
    candidates_by_frame: Mapping[int, Sequence[_Candidate]],
    config: ClassicalBallDetectorConfig,
    frame_width: int,
    frame_height: int,
    stop_frame: Optional[int],
) -> List[_Candidate]:
    """Extend a time-increasing track with tightly gated nearby candidates."""

    result = list(points)
    if len(result) < 3:
        return result
    original_end = result[-1].frame
    misses = 0
    maximum_extension = max(0, int(config.high_flight_extension_frames))
    for frame in range(original_end + 1, original_end + maximum_extension + 1):
        if stop_frame is not None and frame >= stop_frame:
            break
        predicted_x, predicted_y = _predict_track_position(result, frame)
        if not (
            0.0 <= predicted_x < float(frame_width)
            and -20.0 <= predicted_y < float(frame_height - 5)
        ):
            break

        previous = result[-1]
        dt = frame - previous.frame
        if dt <= 0:
            continue
        before_previous = result[-2]
        previous_dt = previous.frame - before_previous.frame
        if previous_dt <= 0:
            break
        previous_velocity = np.asarray(
            [
                (previous.x - before_previous.x) / previous_dt,
                (previous.y - before_previous.y) / previous_dt,
            ],
            dtype=np.float32,
        )
        previous_speed = float(np.linalg.norm(previous_velocity))
        gate = min(
            30.0,
            10.0 + 0.30 * previous_speed + 4.0 * max(0, dt - 1),
        )

        choices: List[Tuple[float, _Candidate]] = []
        for candidate in candidates_by_frame.get(frame, ()):
            if candidate.quality < 1.25 or candidate.contrast < 10.0:
                continue
            distance = math.hypot(
                candidate.x - predicted_x,
                candidate.y - predicted_y,
            )
            if distance > gate:
                continue
            velocity = np.asarray(
                [
                    (candidate.x - previous.x) / dt,
                    (candidate.y - previous.y) / dt,
                ],
                dtype=np.float32,
            )
            acceleration = float(np.linalg.norm(velocity - previous_velocity))
            if acceleration > 13.0 + 2.0 * max(0, dt - 1):
                continue
            speed = float(np.linalg.norm(velocity))
            turn = 0.0
            if previous_speed > 3.0 and speed > 3.0:
                cosine = float(
                    np.dot(previous_velocity, velocity)
                    / (previous_speed * speed)
                )
                turn = math.degrees(
                    math.acos(float(np.clip(cosine, -1.0, 1.0)))
                )
            if turn > 38.0:
                continue
            cost = distance / gate + 0.035 * acceleration - 0.04 * candidate.quality
            choices.append((cost, candidate))

        if not choices:
            misses += 1
            if misses >= max(2, int(config.conditioned_max_misses)):
                break
            continue
        choices.sort(key=lambda item: item[0])
        result.append(choices[0][1])
        misses = 0
    return result


def _extend_track_bidirectionally(
    track: _WorkingTrack,
    candidates_by_frame: Mapping[int, Sequence[_Candidate]],
    config: ClassicalBallDetectorConfig,
    frame_width: int,
    frame_height: int,
    stop_frame: Optional[int],
) -> _WorkingTrack:
    """Extend both ends while preserving the tested forward predictor."""

    backward_points = [
        replace(point, frame=-point.frame) for point in reversed(track.points)
    ]
    reverse_candidates: Dict[int, List[_Candidate]] = {}
    first_frame = track.points[0].frame
    maximum_extension = max(0, int(config.high_flight_extension_frames))
    for original_frame in range(
        max(0, first_frame - maximum_extension), first_frame
    ):
        reverse_candidates[-original_frame] = [
            replace(candidate, frame=-candidate.frame)
            for candidate in candidates_by_frame.get(original_frame, ())
        ]
    backward_stop = None
    backward_extended = _extend_track_forward(
        backward_points,
        reverse_candidates,
        config,
        frame_width,
        frame_height,
        backward_stop,
    )
    chronological = [
        replace(point, frame=-point.frame)
        for point in reversed(backward_extended)
    ]
    fully_extended = _extend_track_forward(
        chronological,
        candidates_by_frame,
        config,
        frame_width,
        frame_height,
        stop_frame,
    )
    return _clone_working_track(track, fully_extended)


def _renumber_track_metrics(
    tracks: Sequence[_TrackMetrics],
) -> List[_TrackMetrics]:
    """Give merged tracks stable compact IDs in chronological order."""

    result: List[_TrackMetrics] = []
    ordered = sorted(
        tracks,
        key=lambda metrics: (
            metrics.start_frame,
            metrics.end_frame,
            -metrics.rank,
        ),
    )
    for track_id, metrics in enumerate(ordered):
        track = _clone_working_track(
            metrics.track,
            metrics.track.points,
            track_id=track_id,
        )
        result.append(replace(metrics, track=track))
    return result


def _deduplicate_track_frames(
    tracks: Sequence[_TrackMetrics],
    core_frames: Sequence[Sequence[int]],
    config: ClassicalBallDetectorConfig,
) -> List[_TrackMetrics]:
    """Keep one owner per frame, preferring measured cores over extensions."""

    core_sets = [set(frames) for frames in core_frames]
    winners: Dict[int, Tuple[Tuple[float, float, float], int]] = {}
    for track_index, metrics in enumerate(tracks):
        own_core = core_sets[track_index]
        for point in metrics.track.points:
            priority = (
                1.0 if point.frame in own_core else 0.0,
                float(metrics.rank),
                float(point.quality),
            )
            existing = winners.get(point.frame)
            if existing is None or priority > existing[0]:
                winners[point.frame] = (priority, track_index)

    result: List[_TrackMetrics] = []
    for track_index, metrics in enumerate(tracks):
        points = [
            point
            for point in metrics.track.points
            if winners.get(point.frame, ((0.0, 0.0, 0.0), -1))[1]
            == track_index
        ]
        if len(points) < 3:
            continue
        track = _clone_working_track(metrics.track, points)
        measured = _measure_track(track, config)
        if measured is not None:
            result.append(measured)
    return result


def _normalized_confidence(rank: float, point_quality: float) -> float:
    # Both terms are above hard acceptance thresholds.  Preserve a cautious
    # floor because classical detections are a fallback, not ground truth.
    rank_term = float(np.clip((rank - 4.45) / 1.35, 0.0, 1.0))
    quality_term = float(np.clip((point_quality - 1.35) / 3.0, 0.0, 1.0))
    return float(np.clip(0.52 + 0.24 * rank_term + 0.20 * quality_term, 0.52, 0.96))


def _build_result(
    frame_count: int,
    selected: Sequence[_TrackMetrics],
    endpoint_min_y: Optional[float] = None,
    frame_size: Optional[Tuple[int, int]] = None,
    endpoint_edge_margin: float = 0.0,
) -> ClassicalBallDetectionResult:
    observations: Dict[int, ClassicalBallObservation] = {}
    observation_rank: Dict[int, float] = {}
    track_ranges: List[ClassicalTrackRange] = []
    hit_candidates: List[ClassicalHitCandidate] = []

    for metrics in selected:
        track = metrics.track
        track_ranges.append(
            ClassicalTrackRange(
                track_id=track.track_id,
                start_frame=metrics.start_frame,
                end_frame=metrics.end_frame,
                point_count=metrics.point_count,
                score=float(metrics.rank),
            )
        )
        for point in track.points:
            if metrics.rank <= observation_rank.get(point.frame, -math.inf):
                continue
            observations[point.frame] = ClassicalBallObservation(
                frame=point.frame,
                x=float(point.x),
                y=float(point.y),
                confidence=_normalized_confidence(metrics.rank, point.quality),
                track_id=track.track_id,
                quality=float(point.quality),
            )
            observation_rank[point.frame] = metrics.rank

        endpoint = track.points[-1]
        if endpoint_min_y is not None and endpoint.y < float(endpoint_min_y):
            # A track ending while the shuttle is still above the conservative
            # ROI is a detector/visibility boundary, not racket contact.
            continue
        if frame_size is not None:
            frame_width, frame_height = frame_size
            margin = max(0.0, float(endpoint_edge_margin))
            if (
                endpoint.x <= margin
                or endpoint.x >= frame_width - 1.0 - margin
                or endpoint.y <= margin
                or endpoint.y >= frame_height - 1.0 - margin
            ):
                continue
        previous = track.points[-2]
        dt = max(1, endpoint.frame - previous.frame)
        endpoint_speed = math.hypot(
            endpoint.x - previous.x,
            endpoint.y - previous.y,
        ) / dt
        hit_candidates.append(
            ClassicalHitCandidate(
                frame=endpoint.frame,
                x=float(endpoint.x),
                y=float(endpoint.y),
                score=_normalized_confidence(metrics.rank, endpoint.quality),
                speed=float(endpoint_speed),
                track_id=track.track_id,
            )
        )

    ordered_observations = dict(sorted(observations.items()))
    track_ranges.sort(key=lambda item: (item.start_frame, item.end_frame))
    hit_candidates.sort(key=lambda item: item.frame)
    return ClassicalBallDetectionResult(
        frame_count=max(0, int(frame_count)),
        observations=ordered_observations,
        track_ranges=track_ranges,
        hit_candidates=hit_candidates,
    )


def detect_classical_ball(
    video_path: str,
    court_points: Sequence[Sequence[float]],
    scene_cuts: Optional[Iterable[int]] = None,
    config: Optional[ClassicalBallDetectorConfig] = None,
    **overrides: object,
) -> ClassicalBallDetectionResult:
    """Detect sparse shuttle trajectories using a CPU-only classical method.

    Args:
        video_path: Input video readable by OpenCV.
        court_points: Four image points ordered ``TL, TR, BR, BL``.
        scene_cuts: Optional cut-frame indices.  Fixed image-space calibration
            is stopped at the first cut; later frames are decoded only so that
            ``frame_count`` remains the actual readable frame count.
        config: Optional immutable detector configuration.
        **overrides: Convenience overrides for configuration field names.

    Returns:
        A :class:`ClassicalBallDetectionResult`.  Observations are deliberately
        sparse; feed ``result.to_ball_dict()`` to the normal Kalman smoother.
    """

    detector_config = _coerce_config(config, overrides)
    stop_frame = _first_scene_cut(scene_cuts)
    base_polygon = _expanded_court_polygon(court_points, detector_config)
    enhanced = bool(detector_config.high_flight_recall_enabled)
    wide_extension = max(
        float(detector_config.court_top_extension),
        float(detector_config.high_flight_top_extension),
    )
    wide_config = replace(
        detector_config,
        court_top_extension=wide_extension,
    )
    conditioned_config = replace(
        wide_config,
        motion_threshold=int(detector_config.conditioned_motion_threshold),
        gray_threshold=int(detector_config.conditioned_gray_threshold),
        saturation_threshold=int(detector_config.conditioned_saturation_threshold),
        component_min_area=2,
        component_max_area=int(detector_config.conditioned_component_max_area),
        component_min_width=1,
        component_max_width=40,
        component_min_height=1,
        component_max_height=40,
        component_min_density=0.10,
        component_min_aspect=0.08,
        candidates_per_frame=int(detector_config.conditioned_candidates_per_frame),
        track_seed_quality=0.0,
    )
    wide_polygon = _expanded_court_polygon(court_points, wide_config)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")

    decoded: List[Tuple[np.ndarray, np.ndarray]] = []
    frame_count = 0
    frame_width = 0
    frame_height = 0
    base_top_y = 0
    wide_candidates_by_frame: Dict[int, List[_Candidate]] = {}
    base_tracker = _GreedyTracker(detector_config)
    wide_tracker = _GreedyTracker(wide_config)
    try:
        for _ in range(3):
            ok, bgr = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            decoded.append((bgr, gray))
            frame_count += 1

        if not decoded:
            return ClassicalBallDetectionResult(0, {}, [], [])
        frame_height, frame_width = decoded[0][1].shape
        base_court_roi, base_roi = _court_mask_and_roi(
            (frame_height, frame_width), base_polygon
        )
        base_top_y = int(base_roi[1])
        if enhanced:
            wide_court_roi, wide_roi = _court_mask_and_roi(
                (frame_height, frame_width), wide_polygon
            )

        if len(decoded) == 3:
            previous, current, following = decoded
            current_index = 1
            while True:
                if stop_frame is None or current_index < stop_frame:
                    base_candidates = _extract_candidates(
                        current_index,
                        previous[1],
                        current[0],
                        current[1],
                        following[1],
                        base_court_roi,
                        base_roi,
                        detector_config,
                    )
                    base_tracker.update(current_index, base_candidates)
                    if enhanced:
                        wide_candidates = _extract_candidates(
                            current_index,
                            previous[1],
                            current[0],
                            current[1],
                            following[1],
                            wide_court_roi,
                            wide_roi,
                            wide_config,
                        )
                        extension_candidates = wide_candidates
                        if detector_config.conditioned_recall_enabled:
                            relaxed_candidates = _extract_candidates(
                                current_index,
                                previous[1],
                                current[0],
                                current[1],
                                following[1],
                                wide_court_roi,
                                wide_roi,
                                conditioned_config,
                            )
                            extension_candidates = _merge_conditioned_candidates(
                                wide_candidates,
                                relaxed_candidates,
                                detector_config,
                            )
                        wide_candidates_by_frame[current_index] = extension_candidates
                        wide_tracker.update(current_index, wide_candidates)

                ok, next_bgr = capture.read()
                if not ok:
                    break
                next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
                frame_count += 1
                previous, current, following = (
                    current,
                    following,
                    (next_bgr, next_gray),
                )
                current_index += 1
        else:
            # One-/two-frame clips cannot produce a three-frame difference.
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                frame_count += 1
    finally:
        capture.release()

    base_measured = [
        metrics
        for track in base_tracker.tracks
        if (metrics := _measure_track(track, detector_config)) is not None
    ]
    base_strict = [
        metrics
        for metrics in base_measured
        if _passes_strict_gates(metrics, detector_config, stop_frame)
    ]
    base_selected = _temporal_nms(base_strict, detector_config)
    if not enhanced:
        return _build_result(frame_count, base_selected)

    wide_measured = [
        metrics
        for track in wide_tracker.tracks
        if (metrics := _measure_track(track, wide_config)) is not None
    ]
    wide_strict = [
        metrics
        for metrics in wide_measured
        if _passes_strict_gates(metrics, wide_config, stop_frame)
    ]
    wide_selected = _temporal_nms(wide_strict, wide_config)

    # The precision anchor always comes from the original, conservative ROI.
    # The wide pass may only add a strict track that actually rose above that
    # ROI; this one condition removes the smooth player-back response in the
    # reference broadcast while retaining all six newly visible flights.
    high_flight_strict = [
        metrics
        for metrics in wide_selected
        if min(point.y for point in metrics.track.points) < float(base_top_y)
    ]

    seed_metrics: List[_TrackMetrics] = list(base_selected)
    seed_core_frames: List[Sequence[int]] = [
        tuple(point.frame for point in metrics.track.points)
        for metrics in base_selected
    ]

    for metrics in high_flight_strict:
        trimmed_track = _trim_abrupt_tail(metrics.track, wide_config)
        trimmed_metrics = _measure_track(trimmed_track, wide_config)
        if trimmed_metrics is None:
            continue
        # Avoid accepting the same physical flight twice if a custom caller
        # already expanded the conservative ROI beyond the normal 120 px.
        if any(
            _track_temporal_overlap(trimmed_metrics, existing) >= 0.42
            for existing in seed_metrics
        ):
            continue
        seed_metrics.append(trimmed_metrics)
        seed_core_frames.append(
            tuple(point.frame for point in trimmed_track.points)
        )

    edge_tracks = sorted(
        (
            metrics
            for metrics in wide_measured
            if _passes_high_flight_edge_gates(
                metrics, wide_config, stop_frame
            )
        ),
        key=lambda metrics: metrics.rank,
        reverse=True,
    )
    for metrics in edge_tracks:
        if any(
            _track_temporal_overlap(metrics, existing) >= 0.42
            for existing in seed_metrics
        ):
            continue
        trimmed_track = _trim_abrupt_tail(metrics.track, wide_config)
        trimmed_metrics = _measure_track(trimmed_track, wide_config)
        if trimmed_metrics is None:
            continue
        seed_metrics.append(trimmed_metrics)
        seed_core_frames.append(
            tuple(point.frame for point in trimmed_track.points)
        )

    all_core_frames = {
        frame for track_frames in seed_core_frames for frame in track_frames
    }
    expanded_metrics: List[_TrackMetrics] = []
    expanded_core_frames: List[Sequence[int]] = []
    for metrics, own_core_sequence in zip(seed_metrics, seed_core_frames):
        own_core = set(own_core_sequence)
        expanded_track = _extend_track_bidirectionally(
            metrics.track,
            wide_candidates_by_frame,
            wide_config,
            frame_width,
            frame_height,
            stop_frame,
        )
        # A direct observation from any trusted core beats an extrapolated
        # candidate from a neighbouring flight at the same frame.
        points = [
            point
            for point in expanded_track.points
            if point.frame in own_core or point.frame not in all_core_frames
        ]
        expanded_track = _clone_working_track(expanded_track, points)
        measured = _measure_track(expanded_track, wide_config)
        if measured is None:
            continue
        expanded_metrics.append(measured)
        expanded_core_frames.append(tuple(own_core_sequence))

    deduplicated = _deduplicate_track_frames(
        expanded_metrics,
        expanded_core_frames,
        wide_config,
    )
    selected = _renumber_track_metrics(deduplicated)
    return _build_result(
        frame_count,
        selected,
        endpoint_min_y=float(base_top_y),
        frame_size=(frame_width, frame_height),
        endpoint_edge_margin=float(detector_config.high_flight_edge_margin),
    )


__all__ = [
    "ClassicalBallDetectionResult",
    "ClassicalBallDetectorConfig",
    "ClassicalBallObservation",
    "ClassicalHitCandidate",
    "ClassicalTrackRange",
    "detect_classical_ball",
]
