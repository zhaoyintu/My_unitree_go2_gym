#!/usr/bin/env python3
"""Convenience launcher for mjlab-trained Lite3 handstand deployment.

By default this script finds the newest checkpoint under
`logs/rsl_rl/lite3_handstand/<run>/model_*.pt` and forwards it to the
standalone MuJoCo deploy script.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "lite3_handstand"
UNDERLYING_DEPLOY = REPO_ROOT / "deploy_mujoco_viewer" / "deploy_mjlab_lite3_handstand.py"
PYTHON = sys.executable

_MODEL_RE = re.compile(r"model_(\d+)\.pt$")


def _checkpoint_iteration(path: Path) -> int:
    match = _MODEL_RE.match(path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _latest_run_dir(log_root: Path) -> Path:
    if not log_root.exists():
        raise FileNotFoundError(
            f"No Lite3 handstand log directory found: {log_root}\n"
            "Train first with: python scripts/train_go2.py Mjlab-Lite3-Handstand"
        )

    runs = [path for path in log_root.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run directories found under {log_root}")

    return sorted(runs, key=lambda path: path.name)[-1]


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = [
        path for path in run_dir.glob("model_*.pt")
        if _checkpoint_iteration(path) >= 0
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoints found under {run_dir}")

    return max(checkpoints, key=_checkpoint_iteration)


def resolve_policy_path(
    policy: Path | str | None,
    run: str | None,
    ckpt: str | None,
    log_root: Path | str = DEFAULT_LOG_ROOT,
) -> Path:
    """Resolve a policy path from explicit path, run name, or latest log run."""
    if policy is not None:
        policy_path = Path(policy).expanduser()
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")
        return policy_path.resolve()

    root = Path(log_root).expanduser()
    run_dir = root / run if run else _latest_run_dir(root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if ckpt is not None:
        ckpt_path = run_dir / ckpt
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path.resolve()

    return _latest_checkpoint(run_dir).resolve()


def build_deploy_command(policy: Path, args: argparse.Namespace) -> list[str]:
    """Build the command line for deploy_mjlab_lite3_handstand.py."""
    command = [
        PYTHON,
        str(UNDERLYING_DEPLOY),
        "--policy",
        str(policy),
        "--cmd",
        str(args.cmd[0]),
        str(args.cmd[1]),
        str(args.cmd[2]),
        "--duration",
        str(args.duration),
        "--video-fps",
        str(args.video_fps),
        "--video-width",
        str(args.video_width),
        "--video-height",
        str(args.video_height),
    ]

    if args.no_viewer:
        command.append("--no-viewer")
    if args.video is not None:
        command.extend(["--video", str(args.video)])
    if args.video_camera is not None:
        command.extend(["--video-camera", args.video_camera])

    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy the latest mjlab Lite3 handstand checkpoint in MuJoCo.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Explicit checkpoint path. Overrides --run/--ckpt.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run directory name under logs/rsl_rl/lite3_handstand.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint filename inside the selected run, e.g. model_5000.pt.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help="Root directory containing Lite3 handstand training runs.",
    )
    parser.add_argument(
        "--cmd",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("vx", "vy", "wz"),
        help="Constant velocity command passed to the policy.",
    )
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Write rollout MP4 to this path.",
    )
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-camera", type=str, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the underlying deploy command without running it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = resolve_policy_path(args.policy, args.run, args.ckpt, args.log_root)
    command = build_deploy_command(policy, args)

    print(f"[deploy] policy: {policy}")
    print("[deploy] command:")
    print(" ".join(command))

    if args.dry_run:
        return

    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
