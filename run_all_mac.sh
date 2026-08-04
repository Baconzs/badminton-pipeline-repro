#!/usr/bin/env bash
set -euo pipefail

INPUT_VIDEO=""
WORK_ROOT="${HOME}/yumaoqiu_repro"
# COURT_POINTS="352,232,613,232,719,525,244,525"
COURT_POINTS="642,486,1275,484,1578,1003,350,1009"
MANUAL_COURT=0
PYTHON_BIN="python3"
FFMPEG_BIN_OVERRIDE="${FFMPEG_BIN_OVERRIDE:-}"
YOLO_DEVICE=""
TRACKNET_DEVICE="auto"
CONTINUE_BALL_AFTER_SCENE_CUT=0
TRACKNET_BATCH_SIZE=4
TRACKNET_MAX_SAMPLE_NUM=300
TRACKNET_TILE_OVERLAP=0.25
# TrackNet's standard heatmap decoder uses this candidate threshold.  Keep
# 0.20 as the safe default for the official checkpoint; lower values can make
# the largest connected component jump onto a player/banner response.
TRACKNET_VIS_THRESH="${TRACKNET_VIS_THRESH:-0.20}"
TRACKNET_COURT_TOP_EXTENSION=480
# The tiled high-resolution path is useful for controlled experiments, but on
# this broadcast it is less precise than the two standard temporal phases: it
# can lock onto a player or the banner while the shuttle is above the image.
# Keep it opt-in instead of changing a good result merely because CUDA exists.
TRACKNET_HIGH_RES="auto"
TRACKNET_EVAL_MODE="auto"
TRACKNET_WEIGHT_OVERRIDE=""
TRACKNET_DUAL_PHASE=1
TORCH_ACCEL_AVAILABLE=0
CLASSICAL_BALL_FALLBACK=1
CLASSICAL_TRIGGER_COVERAGE=0.35
# Bullet-time/slow-motion rendering is off by default for the Web UI, but the
# feature remains available as an explicit opt-in for the UI and command line.
ENABLE_BULLET_TIME_FX="${BADMINTON_ENABLE_BULLET_TIME_FX:-0}"

usage() {
  cat <<'EOF'
Usage:
  ./run_all_mac.sh --input-video <path_to_mp4> [options]

Required:
  --input-video PATH            Input original video (.mp4)

Options:
  --work-root PATH              Output root directory (default: ~/yumaoqiu_repro)
  --court-points STR            Fixed court points: x1,y1,x2,y2,x3,y3,x4,y4
  --manual-court                Enable manual court clicking (TL->TR->BR->BL)
  --python PATH                 Python executable (default: python3; also used for imageio-ffmpeg fallback)
  --yolo-device STR             YOLO device for overlay stage, e.g. mps/cpu (default: auto)
  --tracknet-device STR         TrackNet device: auto/cuda/mps/cpu (default: auto)
  --continue-ball-after-scene-cut
                                Continue ball analysis after hard cuts when all shots share one court calibration
  --tracknet-batch-size N       TrackNet inference batch size (default: 4, lower if OOM)
  --tracknet-max-sample-num N   Max frames sampled for median image (default: 300, lower if OOM)
  --tracknet-vis-thresh N       TrackNet heatmap candidate threshold (default: 0.20; env override)
  --tracknet-weight PATH        TrackNet checkpoint (default: official checkpoint when available)
  --tracknet-eval-mode MODE     nonoverlap/average/weight (default: nonoverlap + dual phase)
  --tracknet-dual-phase         Run half-window second phase for nonoverlap inference (default)
  --no-tracknet-dual-phase      Disable the second temporal phase
  --tracknet-tile-overlap N     Overlap ratio for high-resolution TrackNet crops (default: 0.25)
  --tracknet-court-top-extension N
                                TrackNet region above far baseline in pixels (default: 480)
  --tracknet-high-res            Force experimental high-resolution tiled TrackNet inference
  --no-tracknet-high-res         Disable tiled inference (default; recommended for this broadcast)
  --classical-ball-fallback      Fuse strict CPU motion tracks for weak/local TrackNet gaps (default)
  --no-classical-ball-fallback   Disable classical shuttle fusion/fallback
  --classical-trigger-coverage N Use classical-only fallback below this real TrackNet coverage (default: 0.35)
  --enable-bullet-time-fx      Enable the optional bullet-time/slow-motion FX (opt-in)
  --no-bullet-time-fx          Keep the normal-speed overlay (default)
  -h, --help                    Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-video)
      INPUT_VIDEO="${2:-}"; shift 2 ;;
    --work-root)
      WORK_ROOT="${2:-}"; shift 2 ;;
    --court-points)
      COURT_POINTS="${2:-}"; shift 2 ;;
    --manual-court)
      MANUAL_COURT=1; shift ;;
    --python)
      PYTHON_BIN="${2:-}"; shift 2 ;;
    --yolo-device)
      YOLO_DEVICE="${2:-}"; shift 2 ;;
    --tracknet-device)
      TRACKNET_DEVICE="${2:-}"; shift 2 ;;
    --continue-ball-after-scene-cut)
      CONTINUE_BALL_AFTER_SCENE_CUT=1; shift ;;
    --tracknet-batch-size)
      TRACKNET_BATCH_SIZE="${2:-}"; shift 2 ;;
    --tracknet-max-sample-num)
      TRACKNET_MAX_SAMPLE_NUM="${2:-}"; shift 2 ;;
    --tracknet-vis-thresh)
      TRACKNET_VIS_THRESH="${2:-}"; shift 2 ;;
    --tracknet-weight)
      TRACKNET_WEIGHT_OVERRIDE="${2:-}"; shift 2 ;;
    --tracknet-eval-mode)
      TRACKNET_EVAL_MODE="${2:-}"; shift 2 ;;
    --tracknet-dual-phase)
      TRACKNET_DUAL_PHASE=1; shift ;;
    --no-tracknet-dual-phase)
      TRACKNET_DUAL_PHASE=0; shift ;;
    --tracknet-tile-overlap)
      TRACKNET_TILE_OVERLAP="${2:-}"; shift 2 ;;
    --tracknet-court-top-extension)
      TRACKNET_COURT_TOP_EXTENSION="${2:-}"; shift 2 ;;
    --tracknet-high-res)
      TRACKNET_HIGH_RES=1; shift ;;
    --no-tracknet-high-res)
      TRACKNET_HIGH_RES=0; shift ;;
    --classical-ball-fallback)
      CLASSICAL_BALL_FALLBACK=1; shift ;;
    --no-classical-ball-fallback)
      CLASSICAL_BALL_FALLBACK=0; shift ;;
    --classical-trigger-coverage)
      CLASSICAL_TRIGGER_COVERAGE="${2:-}"; shift 2 ;;
    --enable-bullet-time-fx)
      ENABLE_BULLET_TIME_FX=1; shift ;;
    --no-bullet-time-fx)
      ENABLE_BULLET_TIME_FX=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ "${ENABLE_BULLET_TIME_FX}" != "0" && "${ENABLE_BULLET_TIME_FX}" != "1" ]]; then
  echo "[ERROR] Bullet-time FX switch must be 0 or 1: ${ENABLE_BULLET_TIME_FX}" >&2
  exit 1
fi

if [[ -z "${INPUT_VIDEO}" ]]; then
  echo "[ERROR] --input-video is required." >&2
  usage
  exit 1
fi

if [[ ! -f "${INPUT_VIDEO}" ]]; then
  echo "[ERROR] Input video not found: ${INPUT_VIDEO}" >&2
  exit 1
fi

if ! [[ "${TRACKNET_VIS_THRESH}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "[ERROR] --tracknet-vis-thresh must be a number in [0,1]: ${TRACKNET_VIS_THRESH}" >&2
  exit 1
fi
TRACKNET_VIS_THRESH_VALUE="${TRACKNET_VIS_THRESH}"
if ! awk -v value="${TRACKNET_VIS_THRESH_VALUE}" 'BEGIN { exit !(value >= 0 && value <= 1) }'; then
  echo "[ERROR] --tracknet-vis-thresh must be in [0,1]: ${TRACKNET_VIS_THRESH}" >&2
  exit 1
fi
export TRACKNET_VIS_THRESH

# On Linux workstations the system `python3` may not have the TrackNet/YOLO
# dependencies even though the project environment does.  Keep the explicit
# --python override authoritative, but automatically use a nearby conda
# interpreter when the default cannot import torch and ultralytics.
if [[ "${PYTHON_BIN}" == "python3" ]] && ! python3 -c 'import torch, ultralytics' >/dev/null 2>&1; then
  for candidate in "${CONDA_PREFIX:-}/bin/python" \
      "/data/miniconda3/envs/py311/bin/python"; do
    if [[ -x "${candidate}" ]] && "${candidate}" -c 'import torch, ultralytics' >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      echo "[INFO] Using project Python: ${PYTHON_BIN}"
      break
    fi
  done
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve accelerator availability once so TrackNet and the overlay use the
# same fallback policy.  A CUDA-enabled wheel is not sufficient by itself:
# torch.cuda.is_available() must also succeed with the installed driver.
TORCH_ACCEL_AVAILABLE="$(${PYTHON_BIN} -c 'import torch; print(int(torch.cuda.is_available() or (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available())))' 2>/dev/null || echo 0)"

TRACKNET_SCRIPT="${SCRIPT_DIR}/scripts/tracknet_runtime/predict.py"
OVERLAY_SCRIPT="${SCRIPT_DIR}/scripts/overlay/overlay_player_analytics.py"
FX_SCRIPT="${SCRIPT_DIR}/scripts/fx/video_fx_bullet_time.py"
if [[ -n "${TRACKNET_WEIGHT_OVERRIDE}" ]]; then
  TRACKNET_WEIGHT="${TRACKNET_WEIGHT_OVERRIDE}"
elif [[ -f "${SCRIPT_DIR}/weights/TrackNet_official_best.pt" ]]; then
  TRACKNET_WEIGHT="${SCRIPT_DIR}/weights/TrackNet_official_best.pt"
else
  TRACKNET_WEIGHT="${SCRIPT_DIR}/weights/TrackNet_best.pt"
  echo "[WARN] Official TrackNet checkpoint is unavailable; using legacy checkpoint: ${TRACKNET_WEIGHT}" >&2
fi
YOLO_WEIGHT="${SCRIPT_DIR}/weights/yolov8s-pose.pt"

required_files=(
  "${TRACKNET_SCRIPT}"
  "${OVERLAY_SCRIPT}"
  "${TRACKNET_WEIGHT}"
  "${YOLO_WEIGHT}"
)
if [[ "${ENABLE_BULLET_TIME_FX}" == "1" ]]; then
  required_files+=("${FX_SCRIPT}")
fi
for f in "${required_files[@]}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] Missing required file: ${f}" >&2
    exit 1
  fi
done

mkdir -p "${WORK_ROOT}"
TRACKNET_OUT_DIR="${WORK_ROOT}/tracknet_official_result"
TRACKNET_PHASE_OUT_DIR="${WORK_ROOT}/tracknet_official_result_phase_half"
mkdir -p "${TRACKNET_OUT_DIR}"

VIDEO_BASE="$(basename "${INPUT_VIDEO}")"
VIDEO_STEM="${VIDEO_BASE%.*}"
TRACKNET_VIDEO_RAW="${TRACKNET_OUT_DIR}/${VIDEO_STEM}.mp4"
TRACKNET_VIDEO_CANONICAL="${TRACKNET_OUT_DIR}/${VIDEO_STEM}_tracknetv3.mp4"
TRACKNET_CSV="${TRACKNET_OUT_DIR}/${VIDEO_STEM}_ball.csv"
TRACKNET_CSV_SECONDARY="${TRACKNET_PHASE_OUT_DIR}/${VIDEO_STEM}_ball.csv"
OVERLAY_OUT="${WORK_ROOT}/end1_ball_tracking_official_fused.mp4"
OVERLAY_TRACKING_CSV="${OVERLAY_OUT%.mp4}_ball_tracking.csv"
FX_OUT="${WORK_ROOT}/end1_ball_tracking_official_fused_fx.mp4"
FX_H264_OUT="${FX_OUT%.mp4}_h264.mp4"
OVERLAY_H264_OUT="${OVERLAY_OUT%.mp4}_h264.mp4"

resolve_ffmpeg_bin() {
  if [[ -n "${FFMPEG_BIN_OVERRIDE}" && -x "${FFMPEG_BIN_OVERRIDE}" ]]; then
    printf '%s\n' "${FFMPEG_BIN_OVERRIDE}"
    return 0
  fi

  if command -v ffmpeg >/dev/null 2>&1; then
    command -v ffmpeg
    return 0
  fi

  # Fall back to the static binary bundled by imageio-ffmpeg in the
  # selected Python environment when no system ffmpeg is installed.
  local imageio_ffmpeg_bin=""
  imageio_ffmpeg_bin="$(${PYTHON_BIN} -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
  if [[ -n "${imageio_ffmpeg_bin}" && -x "${imageio_ffmpeg_bin}" ]]; then
    printf '%s\n' "${imageio_ffmpeg_bin}"
    return 0
  fi

  return 1
}

transcode_to_h264() {
  local src="$1"
  local audio_src="${2:-}"
  local dst="${src%.mp4}_h264.mp4"
  local ffmpeg_bin=""
  ffmpeg_bin="$(resolve_ffmpeg_bin || true)"
  if [[ -z "${ffmpeg_bin}" ]]; then
    echo "[WARN] ffmpeg/imageio-ffmpeg not found; keeping mp4v source: ${src}" >&2
    return 0
  fi
  echo "[POST] Transcode ${src} -> H.264..."
  if [[ -n "${audio_src}" && -f "${audio_src}" ]]; then
    "${ffmpeg_bin}" -y -hide_banner -loglevel error \
      -i "${src}" -i "${audio_src}" \
      -map 0:v:0 -map 1:a? \
      -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart \
      -c:a aac -b:a 128k \
      "${dst}"
  else
    "${ffmpeg_bin}" -y -hide_banner -loglevel error \
      -i "${src}" \
      -map 0:v:0 -map 0:a? \
      -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart \
      -c:a aac -b:a 128k \
      "${dst}"
  fi
  echo "[OK] ${dst}"
}

echo "[STEP 1/3] TrackNet inference..."
echo "[INFO] TrackNet visibility threshold=${TRACKNET_VIS_THRESH} (candidate filter; temporal/geometry gates follow)"
if [[ "${TRACKNET_EVAL_MODE}" == "auto" ]]; then
  # Two offset non-overlap phases are both faster and more reliable for the
  # supplied 1080p badminton broadcast than one dense/weighted pass.  This
  # deliberately does not depend on CUDA, so a driver upgrade cannot silently
  # reduce recall or disable phase fusion.
  TRACKNET_EVAL_MODE="nonoverlap"
fi
tracknet_args=(
  "${TRACKNET_SCRIPT}"
  "--video_file" "${INPUT_VIDEO}"
  "--tracknet_file" "${TRACKNET_WEIGHT}"
  "--save_dir" "${TRACKNET_OUT_DIR}"
  "--output_video"
  "--device" "${TRACKNET_DEVICE}"
  "--large_video"
  "--batch_size" "${TRACKNET_BATCH_SIZE}"
  "--max_sample_num" "${TRACKNET_MAX_SAMPLE_NUM}"
  "--eval_mode" "${TRACKNET_EVAL_MODE}"
  "--tile-overlap" "${TRACKNET_TILE_OVERLAP}"
  "--tracknet-court-top-extension" "${TRACKNET_COURT_TOP_EXTENSION}"
)

if [[ "${TRACKNET_HIGH_RES}" == "auto" ]]; then
  TRACKNET_HIGH_RES=0
  echo "[INFO] Tiled TrackNet is experimental and disabled by default; use --tracknet-high-res to benchmark it explicitly."
fi
if [[ "${TRACKNET_HIGH_RES}" == "1" ]]; then
  tracknet_args+=("--high-res-tiles")
fi
# A manually selected quad is only available during the overlay stage.  Do
# not silently apply the stale/default quad to TrackNet in that mode: the
# predictor then falls back to its broad ROI, and the overlay applies the
# freshly selected geometry consistently to the rendered track.
if [[ ${MANUAL_COURT} -eq 0 && -n "${COURT_POINTS}" ]]; then
  tracknet_args+=("--tracknet-court-points" "${COURT_POINTS}")
fi
"${PYTHON_BIN}" "${tracknet_args[@]}"

# Non-overlap inference exposes each frame to only one temporal window.  A
# second pass shifted by half the checkpoint sequence length supplies an
# independent context for candidate-level fusion and body false-positive
# rejection.  Overlap/high-resolution modes already use different ensembles.
if [[ "${TRACKNET_DUAL_PHASE}" == "1" && "${TRACKNET_EVAL_MODE}" == "nonoverlap" && "${TRACKNET_HIGH_RES}" != "1" ]]; then
  TRACKNET_SEQ_LEN="$(${PYTHON_BIN} - "${TRACKNET_WEIGHT}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint["param_dict"]["seq_len"]))
PY
)"
  TRACKNET_PHASE_OFFSET=$((TRACKNET_SEQ_LEN / 2))
  if [[ "${TRACKNET_PHASE_OFFSET}" -gt 0 ]]; then
    mkdir -p "${TRACKNET_PHASE_OUT_DIR}"
    echo "[STEP 1b/3] TrackNet half-window phase (offset=${TRACKNET_PHASE_OFFSET})..."
    phase_args=(
      "${TRACKNET_SCRIPT}"
      "--video_file" "${INPUT_VIDEO}"
      "--tracknet_file" "${TRACKNET_WEIGHT}"
      "--save_dir" "${TRACKNET_PHASE_OUT_DIR}"
      "--device" "${TRACKNET_DEVICE}"
      "--large_video"
      "--batch_size" "${TRACKNET_BATCH_SIZE}"
      "--max_sample_num" "${TRACKNET_MAX_SAMPLE_NUM}"
      "--eval_mode" "nonoverlap"
      "--nonoverlap-offset" "${TRACKNET_PHASE_OFFSET}"
      "--tracknet-court-top-extension" "${TRACKNET_COURT_TOP_EXTENSION}"
      "--tracknet-court-side-padding" "40"
      "--tracknet-court-bottom-padding" "30"
    )
    if [[ ${MANUAL_COURT} -eq 0 && -n "${COURT_POINTS}" ]]; then
      phase_args+=("--tracknet-court-points" "${COURT_POINTS}")
    fi
    "${PYTHON_BIN}" "${phase_args[@]}"
    if [[ ! -f "${TRACKNET_CSV_SECONDARY}" ]]; then
      echo "[ERROR] Missing half-window TrackNet CSV: ${TRACKNET_CSV_SECONDARY}" >&2
      exit 1
    fi
    echo "[OK] ${TRACKNET_CSV_SECONDARY}"
  fi
else
  TRACKNET_CSV_SECONDARY=""
  if [[ "${TRACKNET_DUAL_PHASE}" == "1" ]]; then
    echo "[INFO] Dual-phase pass applies only to nonoverlap, non-tiled inference; using the primary TrackNet result."
  fi
fi

if [[ ! -f "${TRACKNET_VIDEO_RAW}" ]]; then
  echo "[ERROR] Missing TrackNet output video: ${TRACKNET_VIDEO_RAW}" >&2
  exit 1
fi
if [[ ! -f "${TRACKNET_CSV}" ]]; then
  echo "[ERROR] Missing TrackNet CSV: ${TRACKNET_CSV}" >&2
  exit 1
fi
cp -f "${TRACKNET_VIDEO_RAW}" "${TRACKNET_VIDEO_CANONICAL}"
echo "[OK] ${TRACKNET_VIDEO_CANONICAL}"
echo "[OK] ${TRACKNET_CSV}"
transcode_to_h264 "${TRACKNET_VIDEO_CANONICAL}"

echo "[STEP 2/3] Overlay rendering..."
overlay_args=(
  "${OVERLAY_SCRIPT}"
  # Render on the clean source frames.  The TrackNet diagnostic video already
  # contains its own dots/trail and would otherwise bake stale false-positive
  # markers underneath the Kalman-filtered overlay.
  "--video_path" "${INPUT_VIDEO}"
  "--output_path" "${OVERLAY_OUT}"
  "--ball_csv" "${TRACKNET_CSV}"
  "--yolo_model" "${YOLO_WEIGHT}"
  "--tracker_cfg" "bytetrack.yaml"
  "--detect_interval" "1"
  "--draw_ball_on_minimap"
  "--draw_ball_trajectory"
  "--draw_pose"
  "--classical_trigger_coverage" "${CLASSICAL_TRIGGER_COVERAGE}"
  "--ball_top_extension" "${TRACKNET_COURT_TOP_EXTENSION}"
)

if [[ "${CONTINUE_BALL_AFTER_SCENE_CUT}" == "1" ]]; then
  overlay_args+=("--continue_ball_after_scene_cut")
else
  overlay_args+=("--stop_ball_at_first_scene_cut")
fi

if [[ -n "${TRACKNET_CSV_SECONDARY}" ]]; then
  overlay_args+=("--ball_csv_secondary" "${TRACKNET_CSV_SECONDARY}")
fi

if [[ "${CLASSICAL_BALL_FALLBACK}" == "1" ]]; then
  overlay_args+=("--classical_ball_fallback")
else
  overlay_args+=("--no_classical_ball_fallback")
fi

if [[ ${MANUAL_COURT} -eq 1 ]]; then
  overlay_args+=("--select_court_points")
else
  overlay_args+=("--no_select_court_points" "--court_points" "${COURT_POINTS}")
fi

if [[ -n "${YOLO_DEVICE}" ]]; then
  overlay_args+=("--device" "${YOLO_DEVICE}")
fi

# A CPU-only workstation can still validate ball/hit/rally rendering without
# running pose inference on every 1080p frame.  Detect people only on frames
# that contain a geometry-qualified ball candidate; those boxes are sufficient
# to hard-veto obvious body-overlap ball and hit false positives.
if [[ "${TORCH_ACCEL_AVAILABLE}" != "1" || "${YOLO_DEVICE}" == "cpu" ]]; then
  [[ -z "${YOLO_DEVICE}" ]] && overlay_args+=("--device" "cpu")
  overlay_args+=("--imgsz" "320" "--no_half" "--no_draw_pose" "--detect_ball_frames_only")
  echo "[INFO] No usable CUDA/MPS overlay accelerator; enabling sparse CPU player validation."
fi

"${PYTHON_BIN}" "${overlay_args[@]}"

if [[ ! -f "${OVERLAY_OUT}" ]]; then
  echo "[ERROR] Missing overlay output: ${OVERLAY_OUT}" >&2
  exit 1
fi
echo "[OK] ${OVERLAY_OUT}"
if [[ -f "${OVERLAY_TRACKING_CSV}" ]]; then
  echo "[OK] ${OVERLAY_TRACKING_CSV}"
fi
transcode_to_h264 "${OVERLAY_OUT}" "${INPUT_VIDEO}"

if [[ "${ENABLE_BULLET_TIME_FX}" == "1" ]]; then
  echo "[STEP 3/3] Bullet-time FX (explicitly enabled)..."
  "${PYTHON_BIN}" "${FX_SCRIPT}" \
    --input "${OVERLAY_OUT}" \
    --output "${FX_OUT}"
else
  # Keep the historical output name so downstream consumers do not need to
  # special-case a temporarily disabled feature.  This is a byte-for-byte
  # normal-speed overlay copy; the FX script is not imported or executed.
  echo "[STEP 3/3] Bullet-time FX disabled; keeping normal-speed overlay."
  cp -f "${OVERLAY_OUT}" "${FX_OUT}"
  if [[ -f "${OVERLAY_H264_OUT}" ]]; then
    cp -f "${OVERLAY_H264_OUT}" "${FX_H264_OUT}"
  fi
fi

if [[ ! -f "${FX_OUT}" ]]; then
  echo "[ERROR] Missing final FX output: ${FX_OUT}" >&2
  exit 1
fi

if [[ "${ENABLE_BULLET_TIME_FX}" == "1" ]]; then
  transcode_to_h264 "${FX_OUT}"
fi

TRACKNET_H264_OUT="${TRACKNET_VIDEO_CANONICAL%.mp4}_h264.mp4"
echo "[DONE] Repro pipeline finished."
ls -lh "${TRACKNET_VIDEO_CANONICAL}" "${TRACKNET_CSV}" "${OVERLAY_OUT}" "${FX_OUT}"
[[ -f "${OVERLAY_TRACKING_CSV}" ]] && ls -lh "${OVERLAY_TRACKING_CSV}"
for f in "${TRACKNET_H264_OUT}" "${OVERLAY_H264_OUT}" "${FX_H264_OUT}"; do
  [[ -f "${f}" ]] && ls -lh "${f}"
done
