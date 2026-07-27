# Badminton Match Video Analytics Pipeline

[简体中文](README.md) · **English**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-black.svg)](https://www.apple.com/macos/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ychenfen/badminton-pipeline-repro/pulls)

End-to-end pipeline that turns a regular badminton match video into an analytics overlay: **player trajectories, movement speed, accumulated distance, shuttle flight path**, plus optional bullet-time effects.

Built on **TrackNet (shuttle detection) + YOLOv8s-pose (player keypoints) + ByteTrack (ID tracking) + court homography**. Tested end-to-end on Apple Silicon (M4 Pro).

![demo](docs/images/demo.gif)

---

## Why this exists

Most "open-source badminton analyzer" projects either:
- only do shuttle detection (no player analytics)
- assume a top-down camera (real broadcasts are oblique)
- hard-code Windows paths and break on macOS / Linux
- have hardcoded thresholds that silently fail on real videos (e.g. 0% ball detection)

This repo is the **fully working, debugged, fully Mac-friendly** version of the chain, with every fix documented in [HANDOVER.md](HANDOVER.md).

## What you get

- **Real-time speed** per player (m/s, smoothed over 6 frames)
- **Per-rally distance + max speed** + **total distance** since video start
- **Mini-court trail visualization** for both players and the shuttle
- **Court perspective correction** — distances are in real meters, not pixels
- **Bullet-time freeze + slow motion** as cinematic finishing FX

## Architecture

```
Raw video.mp4
    │
    ▼  Step 1: TrackNet ──────────── shuttle detection (multi-frame heatmap model)
TrackNet visualization video + ball CSV
    │
    ▼  Step 2: Overlay ───────────── YOLOv8s-pose + ByteTrack + Homography
Overlay video with full analytics
    │
    ▼  Step 3: FX ───────────────── bullet-time freeze + slow-mo + virtual orbit
Final cinematic video
```

Each stage runs independently. Tweaking the panel, retuning thresholds, or adding new effects only needs the relevant stage to re-run.

---

## Quickstart (30-second sample)

### 1. Clone (Git LFS)

```bash
git lfs install     # brew install git-lfs first if missing
git clone https://github.com/ychenfen/badminton-pipeline-repro.git
cd badminton-pipeline-repro
```

The runner prefers the official TrackNetV3 checkpoint
(`weights/TrackNet_official_best.pt`, 130 MB). The legacy
`TrackNet_best.pt` is retained as a compatibility fallback. Large files are
pulled via Git LFS.

### 2. Install dependencies

```bash
python3 -m pip install --user --index-url https://pypi.org/simple \
    numpy opencv-python pandas Pillow torch ultralytics tqdm \
    pycocotools parse lap
```

`pycocotools`, `parse`, `lap` are hidden TrackNet/ByteTrack deps not in the original `requirements.txt`.

### 3. Mark 4 court corners (the only manual step)

```bash
python3 scripts/tools/select_court.py short.mp4
```

Click in order: **TL → TR → BR → BL** (the four corners of the court rectangle, **not** the net poles). Press `q` when done. The script prints a `--court_points "x1,y1,..."` string.

For the included `short.mp4`:

```
352,342,628,343,944,527,52,532
```

![court corners](docs/images/02_court_corners.jpg)

### 4. Run all three stages

```bash
./run_all_mac.sh \
  --input-video short.mp4 \
  --court-points "352,342,628,343,944,527,52,532" \
  --yolo-device mps
```

For this 1080p badminton broadcast, the runner uses standard TrackNet decoding
and two complementary non-overlap temporal phases by default. This is more
reliable than automatically switching to tiled inference on CUDA: the tiled
path can mistake players, the banner, or spectators for the shuttle during a
high clear. Pass `--tracknet-high-res` only to benchmark that experimental path;
`--tracknet-tile-overlap 0.25` controls its overlap.

In default `nonoverlap` mode the runner also executes a second TrackNet pass shifted
by half a sequence (offset 4 for the official 8-frame checkpoint), writing
`tracknet_official_result_phase_half/`. The overlay fuses the two phases per
frame, checks continuous conflict segments, and runs sparse pose/body rejection
before Kalman smoothing. Short human/background branches are suppressed while
audio remains corroborating evidence only. Use `--no-tracknet-dual-phase` to
disable this pass. It cannot recover frames where the shuttle genuinely leaves
the top of the image; those frames remain `missing` instead of being filled by a
background or player response.

The overlay uses the clean input frames, then applies short-gap interpolation,
a constant-velocity Kalman filter, court/static-background gating, and
camera-cut resets. It renders a bright shuttle trail, hit candidates, rally
IDs, and a Mini Court trail. If geometry-filtered real TrackNet coverage is
below 35%, the runner evaluates a CPU three-frame motion detector. It replaces
TrackNet only when it produces at least two strict tracks, 30 observations, and
more measured points than TrackNet. A conservative
120 px pass remains the precision anchor; a second 480 px, tightly gated pass
recovers high clears near the top of the image. Tracklet boundaries break only
the visible trail, while Kalman state hard-resets at camera cuts, so short
detector holes can still be interpolated without splitting one rally.
`--classical-trigger-coverage` adjusts this evaluation threshold;
`--no-classical-ball-fallback` disables the fallback.

A `*_ball_tracking.csv` sidecar records raw provenance/confidence, the active
detector, classical track IDs, smoothed provenance
(`model`/`classical`/`interp`/`kalman`/`missing`), hit coordinates, whether the
ball was observed on the hit frame, hit source, and rally ID. Bidirectional
quadratic stitching and piecewise fits reject ordinary ballistic apexes;
low-altitude endpoints provide visual candidates, while broadband audio may
only corroborate an existing trajectory turn/endpoint and cannot create a hit
by itself. Sparse pose inference labels the hitter and supplies a hard player-
body veto; wrist/body coordinates never become shuttle coordinates. Events
inside a player box, without nearby real shuttle measurements, or inconsistent
with the measured trajectory are removed before NMS. Camera cuts are
hard rally boundaries; the automatic fallback tolerates up to five seconds of
missing ball observations so a high clear does not split one rally.

`TRACKNET_VIS_THRESH=0.20` is the safe default for the official checkpoint on
this broadcast. Treat 0.15 as an A/B experiment only: weaker responses are much
more likely to jump onto a player or background.

### 5. Outputs

In `~/yumaoqiu_repro/`:
- `tracknet_official_result/short_ball.csv` — per-frame ball coordinates
- `tracknet_official_result_phase_half/short_ball.csv` — offset temporal-phase coordinates
- `end1_ball_tracking_official_fused.mp4` — analytics overlay video
- `end1_ball_tracking_official_fused_ball_tracking.csv` — auditable track/hit/rally sidecar
- `end1_ball_tracking_official_fused_fx.mp4` — final FX video

For broad player compatibility, open the sibling `_h264.mp4` files produced by
the runner (for example `end1_ball_tracking_official_fused_h264.mp4`).

A pre-rendered demo lives at `demo/short_overlay_demo.mp4`.

---

## What each stage does

### Step 1 — TrackNet (shuttle detection)

**The problem**: shuttlecocks are tiny (5-10 px), fast, and motion-blurred. Single-frame detectors (YOLO, etc.) miss them constantly.

**The trick**: the official checkpoint ingests 8 consecutive frames and
outputs 8 probability heatmaps. Cross-frame motion makes the blurred shuttle
visible to the network.

![tracknet output](docs/images/03_tracknet_output.jpg)

The CSV format:

```csv
Frame,Visibility,X,Y
0,1,455,202
1,1,455,202
4,1,481,122
```

### Step 2 — Overlay (player tracking + analytics)

90% of the codebase. Five jobs:

1. **YOLOv8s-pose** detects each player + 17 keypoints (ankle keypoints give precise foot positions)
2. **ByteTrack** maintains per-player IDs across frames so trajectories don't fragment
3. **Homography** warps the oblique court trapezoid into a top-down rectangle:

   ```
   Camera-view trapezoid              Top-down court
                                       (0, 0) ─── (6.1, 0)
      TL ────── TR                       │           │
       \        /        Homography      │           │
        \      /         ──────────►     │           │
         \    /                          │           │
      BL ────── BR                       │           │
                                       (0, 13.4)── (6.1, 13.4)
   ```

   `cv2.findHomography(court_quad, dst_rectangle)` gives a 3×3 matrix. Any pixel coordinate then projects to real-world meters, so distance and speed are camera-independent.

4. **MotionStats** per player: instant speed, rally distance, rally max speed, total distance
5. **Drawing**: stats panel (left), mini court (top-right), player skeletons + trails

![overlay full](docs/images/04_overlay_full.jpg)
![panel](docs/images/05_panel_close.jpg)
![mini court](docs/images/06_minicourt_close.jpg)

### Step 3 — FX (bullet-time)

Inspired by *The Matrix*. Lightweight monocular version:

- Pick "bullet moments" (manual / uniform / auto-detected motion peaks)
- Freeze 28 frames at each, virtual orbit camera does small rotation + zoom
- Follow with 40 frames slow motion (each frame repeated 6× with optional interpolation)
- Resume normal playback

Pure post-processing on the Step 2 output, no detection involved.

---

## Performance (MacBook Apple M4 Pro)

| Stage | CPU | MPS (GPU) |
|---|---|---|
| TrackNet (13344 frames) | ~3 hours | not yet supported, see [HANDOVER §10 P1.1](HANDOVER.md) |
| Overlay | ~30 min | ~10 min |
| FX | ~5 min | same |

TrackNet is the bottleneck. Adding MPS device support is task **P1.1** in the roadmap.

---

## FAQ

**Q: Ball completely undetected (Visibility all 0).**
A: Start with the runner's default `TRACKNET_VIS_THRESH=0.20`. If the official
checkpoint is still all-zero, verify that the intended weight and video are
being used before trying a documented A/B run at 0.15. See
[HANDOVER §6.1.4](HANDOVER.md).

**Q: Player speed reads 24 m/s (faster than Bolt).**
A: ID-switch jumps were being recorded as max speed. Already fixed with adaptive threshold `8.0 × dt + 0.05`. See [HANDOVER §6.2.7](HANDOVER.md).

**Q: Chinese characters render as boxes.**
A: macOS PingFang fallback already added. Verify `/System/Library/Fonts/PingFang.ttc` exists.

**Q: Court polygon doesn't align with the actual court.**
A: Re-run `select_court.py` and click in TL → TR → BR → BL order. Add `--draw_court_polygon` to overlay command to verify.

**Q: `ModuleNotFoundError: pycocotools / parse / lap`.**
A: Hidden deps not in `requirements.txt`. Install with the command in step 2 above.

More: [HANDOVER §8](HANDOVER.md).

---

## Roadmap

[HANDOVER.md §10](HANDOVER.md) lists 12 detailed tasks (P0-P3), each formatted as a self-contained brief (**background / goal / steps / files / verification / pitfalls**) so AI agents (Codex, Cursor, Claude) can pick any one and execute end-to-end.

```
P0.1 Inline defaults into run_all_mac.sh         15 min
P0.2 Clean up debug artifacts                    done
P0.3 Run full-length baseline                    3-4 h
P1.1 TrackNet on Apple MPS                       2-4 h
P1.2 Cache detections.json                       3-5 h
P1.3 YOLO frame-skip detection                   1-2 h
P2.1 Ball trajectory hole-filling (Kalman)       3-4 h
P2.2 Hit detection + auto rally split            4-6 h
P2.3 Mini Court visual polish                    1-2 h
P3.1 Gradio Web UI                               1-2 d
P3.2 Multi-camera angle support                  1-2 d
P3.3 JSON export + heatmap                       4-6 h
```

---

## Project layout

```
badminton-pipeline-repro/
├── README.md             # Chinese quick start
├── README_EN.md          # this file
├── HANDOVER.md           # full handover doc, 1500+ lines, AI-agent task pack
├── run_all_mac.sh        # macOS one-shot runner
├── run_all.ps1           # Windows runner
├── short.mp4             # 30s sample (LFS)
├── b13b2c0b...mp4        # 10:35 full match (LFS)
├── weights/              # model weights (LFS)
│   ├── TrackNet_official_best.pt  # official 8-frame/30-epoch checkpoint (default)
│   ├── TrackNet_best.pt           # legacy 4-frame/3-epoch fallback
│   └── yolov8s-pose.pt
├── demo/
│   └── short_overlay_demo.mp4
├── docs/images/          # README assets
└── scripts/
    ├── tracknet_runtime/    # Step 1
    ├── overlay/             # Step 2 (1340+ lines)
    ├── fx/                  # Step 3
    └── tools/               # debug helpers
```

---

## Key fixes vs upstream

| Issue | Symptom | Fix location |
|---|---|---|
| TrackNet 0% ball recall | CSV all-zero `Visibility` | `predict.py:35` threshold 0.5 → env var |
| Player speed 24 m/s | Wildly inflated panel numbers | `overlay_player_analytics.py:602` 1.2 m → `8×dt+0.05` |
| Chinese boxes on Mac | Font path was Windows-only | `overlay_player_analytics.py:18-27` font fallback |
| Stats panel cluttered | 7 metrics per player | Slimmed to 4 essential |
| Hidden deps | `ModuleNotFoundError` at runtime | Documented in install step |

Full changelog: [HANDOVER §9](HANDOVER.md).

---

## Citation

If this code helped your research, paper, or product, a star is the simplest way to say thanks.

```bibtex
@misc{badminton_pipeline_repro,
  author       = {ychenfen},
  title        = {Badminton Match Video Analytics Pipeline},
  year         = {2026},
  howpublished = {\url{https://github.com/ychenfen/badminton-pipeline-repro}}
}
```

---

## Credits

- TrackNet model: [TrackNetV3](https://github.com/qaz812345/TrackNetV3)
- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- Tracking: [ByteTrack](https://github.com/ifzhang/ByteTrack)
- Sample footage: YouTube channel POGBADMINTON

## License

Code: MIT. Model weights and sample videos follow their respective upstream licenses, intended for research and educational use only.

---

**Keywords**: badminton, sports analytics, video analytics, computer vision, object tracking, shuttle detection, player tracking, court homography, perspective correction, bullet time, TrackNet, YOLOv8, ByteTrack, OpenCV, PyTorch, Apple Silicon, macOS, M4 Pro, MPS, sports video AI, real-time analytics, 羽毛球, 视频分析, 图像识别, 运动分析, 计算机视觉, 目标检测, 多目标跟踪, 透视变换, 子弹时间.
