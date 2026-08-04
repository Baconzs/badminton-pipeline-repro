#!/usr/bin/env bash
set -euo pipefail

# Remove historical Web UI runtime data without touching source code, model
# files, or the runtime directory itself.  The implementation lives in the
# standard-library Python block below so it works on both Linux and macOS,
# including the older Bash shipped with macOS.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${BADMINTON_RUNTIME_CLEAN_PYTHON:-python3}"
export SHUTTLEVISION_CLEAN_SCRIPT_DIR="${SCRIPT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "$@" <<'PY'
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


SCRIPT_DIR = Path(os.environ["SHUTTLEVISION_CLEAN_SCRIPT_DIR"]).resolve()
DEFAULT_ROOT = Path(
    os.environ.get("BADMINTON_UI_DATA_ROOT", str(SCRIPT_DIR / "runtime" / "web_ui"))
).expanduser()
TERMINAL_STATES = {"completed", "failed", "cancelled"}
HEX_ID = re.compile(r"^[0-9a-f]{8,64}$")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clean_runtime.sh",
        description="清理 ShuttleVision runtime/web_ui 中的历史任务、上传文件和封面。"
    )
    p.add_argument(
        "--runtime-root",
        default=str(DEFAULT_ROOT),
        help="runtime 数据根目录（默认: %(default)s；传 runtime 时会自动进入 runtime/web_ui）",
    )
    p.add_argument(
        "--older-than",
        type=float,
        default=0.0,
        metavar="DAYS",
        help="仅清理终态任务中超过 DAYS 天的记录（默认 0，清理全部终态历史）",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="清理所有 runtime 历史，包括仍处于活动状态的任务（实际删除必须同时使用 --force）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将清理的内容，不删除文件",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="跳过交互确认，适合脚本或定时任务",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="允许在检测到 Web UI/活动任务时继续；使用前应先停止服务",
    )
    return p


def resolve_root(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    path = path.resolve()
    # Accept both the documented data root and the convenient ``runtime``
    # spelling, but never silently target the repository root.
    if path.name != "web_ui" and (path / "web_ui").is_dir() and not (path / "jobs").exists():
        path = (path / "web_ui").resolve()
    forbidden = {Path("/"), SCRIPT_DIR, SCRIPT_DIR.parent, Path.home().resolve()}
    if path in forbidden:
        raise SystemExit(f"[ERROR] Refusing to clean an unsafe broad path: {path}")
    return path


def inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def human_size(value: int) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024 or unit == "TiB":
            return f"{number:.1f} {unit}" if unit != "B" else f"{int(number)} B"
        number /= 1024
    return f"{value} B"


def tree_size(path: Path) -> int:
    if path.is_symlink():
        try:
            return path.lstat().st_size
        except OSError:
            return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    if path.is_dir():
        for child in path.iterdir():
            total += tree_size(child)
    return total


def parse_timestamp(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


def job_timestamp(manifest: Path, data: Dict[str, object]) -> float:
    for key in ("updated_at", "created_at", "updated", "created"):
        parsed = parse_timestamp(data.get(key))
        if parsed is not None:
            return parsed
    try:
        return manifest.stat().st_mtime
    except OSError:
        return time.time()


def service_running() -> bool:
    # The bracket expression prevents pgrep from matching its own command.
    try:
        result = subprocess.run(
            ["pgrep", "-f", "[w]eb_app\.server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, OSError):
        return False


def add_candidate(
    candidates: Dict[Path, str], path: Path, reason: str, parent: Path
) -> None:
    if not path.exists() and not path.is_symlink():
        return
    resolved = path.resolve() if not path.is_symlink() else path.absolute()
    if resolved == parent.resolve() or not inside(resolved, parent):
        raise SystemExit(f"[ERROR] Refusing an unsafe cleanup target: {path}")
    candidates[path] = reason


def load_jobs(jobs_root: Path) -> Tuple[List[dict], List[str]]:
    jobs: List[dict] = []
    malformed: List[str] = []
    if not jobs_root.is_dir():
        return jobs, malformed
    for directory in sorted(jobs_root.iterdir()):
        if not directory.is_dir():
            malformed.append(str(directory))
            continue
        manifest = directory / "job.json"
        if not manifest.is_file():
            malformed.append(str(directory))
            continue
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            malformed.append(str(directory))
            continue
        if not isinstance(raw, dict):
            malformed.append(str(directory))
            continue
        job_id = str(raw.get("id", directory.name))
        status = raw.get("status")
        if job_id != directory.name or not HEX_ID.fullmatch(job_id) or not isinstance(status, str):
            malformed.append(str(directory))
            continue
        jobs.append(
            {
                "id": job_id,
                "status": status,
                "video_id": str(raw.get("video_id", "")),
                "timestamp": job_timestamp(manifest, raw),
                "path": directory,
            }
        )
    return jobs, malformed


def valid_video_manifest(path: Path) -> Optional[dict]:
    if path.suffix.lower() != ".json" or not HEX_ID.fullmatch(path.stem):
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict) or str(raw.get("id", "")) != path.stem:
        return None
    return raw


def candidate_plan(root: Path, args: argparse.Namespace) -> Tuple[Dict[Path, str], List[str], Set[str]]:
    jobs_root = root / "jobs"
    uploads_root = root / "uploads"
    videos_root = root / "videos"
    candidates: Dict[Path, str] = {}
    warnings: List[str] = []
    jobs, malformed = load_jobs(jobs_root)
    if malformed:
        warnings.append(
            f"发现 {len(malformed)} 个无法安全解析的 job 目录，默认保留；--all --force 才会清理。"
        )

    active = [job for job in jobs if job["status"] not in TERMINAL_STATES]
    if args.all:
        for job in jobs:
            add_candidate(candidates, job["path"], f"任务目录（status={job['status']}）", jobs_root)
        for path in malformed:
            add_candidate(candidates, Path(path), "无法解析的任务目录", jobs_root)
        if uploads_root.is_dir():
            for path in uploads_root.iterdir():
                add_candidate(candidates, path, "全部清理：上传目录", uploads_root)
        if videos_root.is_dir():
            for path in videos_root.iterdir():
                add_candidate(candidates, path, "全部清理：视频历史", videos_root)
        return candidates, warnings, {job["id"] for job in active}

    now = time.time()
    preserve_video_ids: Set[str] = set()
    for job in jobs:
        age_days = max(0.0, (now - float(job["timestamp"])) / 86400.0)
        eligible = job["status"] in TERMINAL_STATES and age_days >= args.older_than
        if eligible:
            add_candidate(
                candidates,
                job["path"],
                f"终态任务（status={job['status']}，{age_days:.1f} 天）",
                jobs_root,
            )
        else:
            if job["video_id"]:
                preserve_video_ids.add(job["video_id"])

    # Keep videos still referenced by an active, malformed, or age-retained
    # job.  Everything else in the runtime library is historical/orphaned.
    if videos_root.is_dir():
        for path in videos_root.iterdir():
            if path.is_file() and path.suffix.lower() == ".json":
                record = valid_video_manifest(path)
                if record is None:
                    warnings.append(f"无法解析的视频清单，默认保留：{path}")
                    continue
                video_id = path.stem
                if video_id in preserve_video_ids:
                    continue
                add_candidate(candidates, path, "无活动任务引用的视频清单", videos_root)
                poster = videos_root / f"{video_id}.jpg"
                if poster.exists():
                    add_candidate(candidates, poster, "对应视频封面", videos_root)
            elif (
                path.is_file()
                and path.suffix.lower() == ".jpg"
                and path.stem not in preserve_video_ids
                and not (videos_root / f"{path.stem}.json").is_file()
            ):
                add_candidate(candidates, path, "无清单引用的视频封面", videos_root)

    if uploads_root.is_dir():
        for path in uploads_root.iterdir():
            if path.name not in preserve_video_ids:
                add_candidate(candidates, path, "无活动任务引用的上传目录", uploads_root)

    return candidates, warnings, {job["id"] for job in active}


def confirm(candidates: Dict[Path, str], total: int, args: argparse.Namespace) -> bool:
    if args.dry_run or args.yes:
        return True
    if not sys.stdin.isatty():
        print("[ERROR] 非交互环境请加 --yes，或先用 --dry-run 查看清理范围。", file=sys.stderr)
        return False
    prompt = "输入 DELETE 确认删除以上 runtime 历史（其他内容不会触碰）： "
    try:
        return input(prompt).strip() == "DELETE"
    except EOFError:
        return False


def delete_candidates(candidates: Dict[Path, str], roots: Iterable[Path]) -> int:
    freed = 0
    root_list = [root.resolve() for root in roots]
    for path in sorted(candidates, key=lambda item: (len(item.parts), str(item))):
        if not path.exists() and not path.is_symlink():
            continue
        if not any(inside(path, root) and path.resolve() != root for root in root_list):
            raise SystemExit(f"[ERROR] Cleanup target moved outside allowed roots: {path}")
        freed += tree_size(path)
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    return freed


def main(argv: List[str]) -> int:
    args = parser().parse_args(argv)
    if args.older_than < 0:
        print("[ERROR] --older-than 不能为负数。", file=sys.stderr)
        return 2
    root = resolve_root(args.runtime_root)
    if not root.exists():
        print(f"[INFO] runtime 数据目录不存在，无需清理：{root}")
        return 0
    known_roots = [root / "jobs", root / "uploads", root / "videos"]
    if not any(path.exists() for path in known_roots):
        print(f"[ERROR] 未找到 jobs/uploads/videos，拒绝清理未知目录：{root}", file=sys.stderr)
        return 2

    running = service_running()
    candidates, warnings, active_ids = candidate_plan(root, args)
    if warnings:
        for warning in warnings:
            print(f"[WARN] {warning}")
    if active_ids:
        print(f"[INFO] 检测到 {len(active_ids)} 个非终态任务。")
    if running:
        print("[INFO] 检测到 Web UI 服务进程。")

    if (args.all or warnings) and not args.force and not args.dry_run:
        if args.all:
            print("[ERROR] --all 的实际删除必须显式加 --force。", file=sys.stderr)
        else:
            print("[ERROR] 存在无法安全解析的 runtime 记录；请先检查，确认后加 --force。", file=sys.stderr)
        return 2
    if (active_ids or running) and not args.force and not args.dry_run:
        print("[ERROR] 为避免与运行中的任务竞争，已拒绝实际删除。请先停止 Web UI；确认后可加 --force。", file=sys.stderr)
        return 2

    total = sum(tree_size(path) for path in candidates)
    mode = "全部历史" if args.all else f"终态历史（超过 {args.older_than:g} 天）"
    print(f"[PLAN] 根目录：{root}")
    print(f"[PLAN] 模式：{mode}")
    print(f"[PLAN] 待处理项目：{len(candidates)}，预计释放：{human_size(total)}")
    for path in sorted(candidates, key=str)[:40]:
        print(f"  - {path}  [{candidates[path]}]")
    if len(candidates) > 40:
        print(f"  ... 其余 {len(candidates) - 40} 项省略")

    if not candidates:
        print("[OK] 没有符合条件的历史数据。")
        return 0
    if args.dry_run:
        print("[DRY-RUN] 未删除任何文件。")
        return 0
    if not confirm(candidates, total, args):
        print("[CANCELLED] 未删除任何文件。")
        return 1

    freed = delete_candidates(candidates, known_roots)
    for directory in known_roots:
        directory.mkdir(parents=True, exist_ok=True)
    print(f"[OK] 已清理 {len(candidates)} 项，释放约 {human_size(freed)}。")
    if args.all:
        print(f"[WARN] --all 已包含活动任务；如 Web UI 正在运行，请重启服务后再提交新任务。")
    else:
        print(f"[OK] 已保留活动/未到期任务及其关联文件；数据根目录仍为：{root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
PY
