#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${BADMINTON_UI_PYTHON:-python3}"

# The repository can carry a tested BST checkout and checkpoint.  Configure
# that pair automatically so the normal one-command launcher also enables the
# optional stroke-type cards.  Explicit environment variables always win;
# setting only one of them is treated as an intentional external setup.
BUNDLED_BST_REPO="${SCRIPT_DIR}/third_party/BST-Badminton-Stroke-type-Transformer"
BUNDLED_BST_WEIGHTS=""
for candidate in \
  "${SCRIPT_DIR}/third_party/models/bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged.pt" \
  "${SCRIPT_DIR}/models/bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged.pt" \
  "${SCRIPT_DIR}/weights/bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged.pt"; do
  if [[ -f "${candidate}" ]]; then
    BUNDLED_BST_WEIGHTS="${candidate}"
    break
  fi
done
if [[ -z "${BADMINTON_BST_REPO:-}" && -z "${BST_REPO:-}" \
      && -z "${BADMINTON_BST_WEIGHTS:-}" && -z "${BST_WEIGHTS:-}" \
      && -d "${BUNDLED_BST_REPO}" && -n "${BUNDLED_BST_WEIGHTS}" ]]; then
  export BADMINTON_BST_REPO="${BUNDLED_BST_REPO}"
  export BADMINTON_BST_WEIGHTS="${BUNDLED_BST_WEIGHTS}"
  if [[ -z "${BADMINTON_BST_DATASET:-}" && -z "${BST_DATASET:-}" ]]; then
    export BADMINTON_BST_DATASET="shuttleset_25"
  fi
  if [[ -z "${BADMINTON_BST_MODEL:-}" && -z "${BST_MODEL:-}" ]]; then
    export BADMINTON_BST_MODEL="BST_0"
  fi
  if [[ -z "${BADMINTON_BST_POSE_STYLE:-}" && -z "${BST_POSE_STYLE:-}" ]]; then
    export BADMINTON_BST_POSE_STYLE="JnB_bone"
  fi
  if [[ -z "${BADMINTON_BST_SEQ_LEN:-}" && -z "${BST_SEQ_LEN:-}" ]]; then
    export BADMINTON_BST_SEQ_LEN="100"
  fi
  echo "[INFO] BST enabled from bundled repository/checkpoint."
elif [[ -n "${BADMINTON_BST_REPO:-}" || -n "${BST_REPO:-}" \
      || -n "${BADMINTON_BST_WEIGHTS:-}" || -n "${BST_WEIGHTS:-}" ]]; then
  echo "[INFO] Using explicit BST configuration from the environment."
else
  echo "[INFO] BST is not configured; the core Web UI will still run."
fi

if ! "${PYTHON_BIN}" -c 'import flask, cv2' >/dev/null 2>&1; then
  PROJECT_PYTHON="/data/miniconda3/envs/py311/bin/python"
  if [[ -x "${PROJECT_PYTHON}" ]] && "${PROJECT_PYTHON}" -c 'import flask, cv2' >/dev/null 2>&1; then
    PYTHON_BIN="${PROJECT_PYTHON}"
  else
    echo "[ERROR] Python environment needs Flask and OpenCV." >&2
    echo "        pip install -r requirements_repro.txt" >&2
    exit 1
  fi
fi

if [[ -n "${BADMINTON_BST_WEIGHTS:-}" ]] && ! "${PYTHON_BIN}" -c 'import positional_encodings, torchinfo' >/dev/null 2>&1; then
  echo "[WARN] BST checkpoint is configured, but optional Python packages are missing." >&2
  echo "       Install them with: ${PYTHON_BIN} -m pip install positional-encodings torchinfo" >&2
fi

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" -m web_app.server --python "${PYTHON_BIN}" "$@"
