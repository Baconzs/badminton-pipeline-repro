"""Audio transient cues for badminton racket-contact detection.

The shuttle can be invisible at contact, especially after a high clear.  A
short, broadband racket impulse is useful weak evidence in those holes.  This
module deliberately returns timing cues only: the overlay must still combine
them with the ball trajectory or player pose, and must label audio-only events
as uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import shutil
import subprocess

import numpy as np


@dataclass(frozen=True)
class AudioHitCandidate:
    frame: int
    score: float


def _resolve_ffmpeg() -> Optional[str]:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    return executable if executable else None


def _decode_mono_pcm(video_path: str, sample_rate: int) -> np.ndarray:
    executable = _resolve_ffmpeg()
    if not executable:
        return np.empty(0, dtype=np.float32)
    command = [
        executable,
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-map", "0:a:0?",
        "-vn", "-sn", "-dn",
        "-ac", "1",
        "-ar", str(int(sample_rate)),
        "-f", "s16le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except (OSError, ValueError):
        return np.empty(0, dtype=np.float32)
    if completed.returncode != 0 or len(completed.stdout) < 2:
        return np.empty(0, dtype=np.float32)
    usable_bytes = len(completed.stdout) - (len(completed.stdout) % 2)
    samples = np.frombuffer(
        completed.stdout[:usable_bytes], dtype="<i2"
    ).astype(np.float32)
    samples *= 1.0 / 32768.0
    return samples


def _robust_local_z(
    values: np.ndarray,
    radius: int = 100,
    guard: int = 5,
    floor: float = 0.04,
) -> np.ndarray:
    """Local median/MAD z-score, excluding the possible impact center."""

    values = np.asarray(values, dtype=np.float32)
    if len(values) < 3:
        return np.zeros_like(values)
    radius = max(2, min(int(radius), len(values) - 1))
    guard = max(0, min(int(guard), radius - 1))
    padded = np.pad(values, radius, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, 2 * radius + 1
    )
    context = np.concatenate(
        (
            windows[:, : radius - guard],
            windows[:, radius + guard + 1 :],
        ),
        axis=1,
    )
    median = np.median(context, axis=1)
    mad = np.median(np.abs(context - median[:, None]), axis=1)
    return (values - median) / (1.4826 * mad + max(1e-6, float(floor)))


def detect_audio_hit_candidates(
    video_path: str,
    video_fps: float,
    start_frame: int = 0,
    stop_frame: Optional[int] = None,
    score_threshold: float = 1.25,
    high_z_threshold: float = 2.25,
    high_ratio_threshold: float = 0.05,
    nms_frames: float = 8.0,
    sample_rate: int = 48000,
    return_audio_available: bool = False,
):
    """Return broadband audio impulses in source-video frame coordinates.

    A 21 ms spectral window is advanced every 5 ms.  Every feature is
    normalized against its surrounding half-second median/MAD, which makes
    the thresholds useful across quiet and crowd-heavy passages.  The caller
    supplies the visual activity range; this is important because applause or
    a broadcast cut after a rally can otherwise look like repeated impacts.
    """

    fps = max(1.0, float(video_fps))
    rate = max(16000, int(sample_rate))
    samples = _decode_mono_pcm(video_path, rate)

    # A source without an audio stream is common for exported coaching clips.
    # Keep that distinct from an audio stream which simply has no sufficiently
    # sharp impulses: callers need to retain visual-only hit detection in the
    # former case without weakening the audio-evidence rule in the latter.
    audio_available = len(samples) >= 2

    def finish(candidates: List[AudioHitCandidate]):
        if return_audio_available:
            return candidates, audio_available
        return candidates

    window_size = 1024
    hop = max(1, int(round(rate * 0.005)))
    if len(samples) < window_size:
        return finish([])
    samples -= float(np.median(samples))

    count = 1 + (len(samples) - window_size) // hop
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(count, window_size),
        strides=(samples.strides[0] * hop, samples.strides[0]),
        writeable=False,
    )
    window = np.hanning(window_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
    power = spectrum * spectrum
    frequency = np.fft.rfftfreq(window_size, 1.0 / rate)
    mid_band = (frequency >= 1500.0) & (frequency <= 14000.0)
    high_band = (frequency >= 4500.0) & (frequency <= 16000.0)
    total_band = (frequency >= 200.0) & (frequency <= 16000.0)

    mid_energy = power[:, mid_band].sum(axis=1)
    high_energy = power[:, high_band].sum(axis=1)
    high_ratio = high_energy / (power[:, total_band].sum(axis=1) + 1e-12)
    z_mid = _robust_local_z(np.log10(mid_energy + 1e-12))
    z_high = _robust_local_z(np.log10(high_energy + 1e-12))

    log_spectrum = np.log1p(spectrum)
    flux = np.zeros(count, dtype=np.float32)
    flux[1:] = np.maximum(
        log_spectrum[1:, mid_band] - log_spectrum[:-1, mid_band], 0.0
    ).mean(axis=1)
    z_flux = _robust_local_z(flux)

    difference = np.diff(frames, axis=1)
    peak = np.max(np.abs(difference), axis=1)
    rms = np.sqrt(np.mean(difference * difference, axis=1) + 1e-12)
    crest = peak / (rms + 1e-8)
    z_peak = _robust_local_z(np.log10(peak + 1e-7))
    z_crest = _robust_local_z(np.log10(crest + 1e-5))

    # The minimum is an AND-like broadband term: speech/crowd energy rising
    # in only one band cannot dominate the impact score.
    score = (
        0.55 * np.minimum(z_mid, z_high)
        + 0.25 * z_flux
        + 0.15 * z_peak
        + 0.05 * z_crest
    )
    center_sample = np.arange(count) * hop + window_size // 2
    video_frame = center_sample / float(rate) * fps
    first = max(0, int(start_frame))
    last = int(stop_frame) if stop_frame is not None else np.iinfo(np.int32).max
    valid = (
        (np.rint(video_frame) >= first)
        & (np.rint(video_frame) < last)
        & (score >= float(score_threshold))
        & (z_high >= float(high_z_threshold))
        & (high_ratio >= float(high_ratio_threshold))
    )

    order = np.flatnonzero(valid)
    order = order[np.argsort(score[order])[::-1]]
    selected: List[int] = []
    for index in order:
        if all(
            abs(float(video_frame[index] - video_frame[old]))
            >= max(1.0, float(nms_frames))
            for old in selected
        ):
            selected.append(int(index))
    selected.sort(key=lambda index: video_frame[index])
    return finish([
        AudioHitCandidate(
            frame=int(round(float(video_frame[index]))),
            score=float(score[index]),
        )
        for index in selected
    ])


__all__ = ["AudioHitCandidate", "detect_audio_hit_candidates"]
