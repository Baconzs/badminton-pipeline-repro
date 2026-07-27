#!/usr/bin/env python3
"""Smooth a TrackNet CSV and retain raw/interpolated provenance.

Example:
    python scripts/tools/smooth_ball_csv.py input_ball.csv output_ball_smoothed.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "overlay"))

from ball_tracking import filter_ball_track_geometry, smooth_ball_track  # noqa: E402


def read_csv(path):
    rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
    if not rows:
        return {}, 0
    frame_count = max(int(float(row.get("Frame", i))) for i, row in enumerate(rows)) + 1
    values = {}
    for i, row in enumerate(rows):
        frame = int(float(row.get("Frame", i)))
        try:
            values[frame] = (
                int(float(row.get("Visibility", 0))),
                int(float(row.get("X", 0))),
                int(float(row.get("Y", 0))),
            )
        except (TypeError, ValueError):
            values[frame] = (0, 0, 0)
    return values, frame_count


def write_csv(path, track):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Frame", "Visibility", "X", "Y", "Source", "RawVisibility", "Confidence"])
        for point in track:
            writer.writerow([
                point.frame,
                int(point.visible),
                int(round(point.x)) if point.visible else 0,
                int(round(point.y)) if point.visible else 0,
                point.source,
                int(point.raw_visible),
                f"{point.confidence:.4f}",
            ])


def main():
    parser = argparse.ArgumentParser(description="Kalman smooth TrackNet ball coordinates")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--max-interp-gap", type=int, default=8)
    parser.add_argument("--max-predict-gap", type=int, default=3)
    parser.add_argument(
        "--court-points",
        type=str,
        default="",
        help="Optional TL,TR,BR,BL court points: x1,y1,...,x4,y4",
    )
    parser.add_argument("--top-extension", type=float, default=60.0)
    parser.add_argument("--side-padding", type=float, default=40.0)
    parser.add_argument("--bottom-padding", type=float, default=30.0)
    parser.add_argument("--scene-cuts", type=str, default="")
    parser.add_argument("--static-window", type=int, default=12)
    parser.add_argument("--static-disp", type=float, default=80.0)
    args = parser.parse_args()

    values, frame_count = read_csv(args.input_csv)
    court_points = None
    if args.court_points:
        try:
            court_points = np.asarray(
                [float(v.strip()) for v in args.court_points.split(",")],
                dtype=np.float32,
            ).reshape(4, 2)
        except (TypeError, ValueError):
            raise SystemExit("--court-points must contain 8 numeric values")
    scene_cuts = []
    if args.scene_cuts:
        for token in args.scene_cuts.split(","):
            try:
                scene_cuts.append(int(token.strip()))
            except ValueError:
                continue
    track = smooth_ball_track(
        values,
        frame_count,
        args.fps,
        max_interp_gap=args.max_interp_gap,
        max_predict_gap=args.max_predict_gap,
        reset_frames=[
            frame
            for cut in scene_cuts
            for frame in range(max(0, cut - 3), cut + 4)
        ],
    )
    if court_points is not None or scene_cuts:
        track = filter_ball_track_geometry(
            track,
            court_polygon=court_points,
            top_extension=args.top_extension,
            side_padding=args.side_padding,
            bottom_padding=args.bottom_padding,
            scene_cuts=scene_cuts,
            scene_cut_padding=3,
            static_window=args.static_window,
            static_disp=args.static_disp,
        )
    write_csv(args.output_csv, track)
    real = sum(p.visible and p.source in ("model", "classical") for p in track)
    derived = sum(p.visible and p.source in ("interp", "kalman") for p in track)
    print(f"wrote {args.output_csv}: {len(track)} frames, real={real}, derived={derived}")


if __name__ == "__main__":
    main()
