#!/usr/bin/env python3
"""Train Unitree Go2 tasks using mjlab + RSL-RL.

Usage:
  python scripts/train_go2.py Mjlab-Go2-Handstand
  python scripts/train_go2.py Mjlab-Go2-Trot-Flat --env.scene.num_envs 4096
  python scripts/train_go2.py --list  # list available tasks

This script imports go2_mjlab to register all Go2 tasks with mjlab's registry,
then delegates to mjlab's training infrastructure.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so go2_mjlab is importable.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import go2_mjlab  # noqa: E402, F401 — registers Go2 tasks

# Now delegate to mjlab's train script.
# We patch sys.argv to pass through the task name and any overrides.
if __name__ == "__main__":
    from mjlab.scripts.train import main as mjlab_train_main

    mjlab_train_main()
