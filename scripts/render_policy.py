"""Render a trained policy rollout to mp4 (headless).

Loads a checkpoint, runs the policy in its play env under a fixed command
sequence (2 s stand, then forward at MEAS_VX), renders each step, writes a video.
Task is selectable via MEAS_TASK so it works for the 450-dim and 470-dim
(PhaseClock) contracts alike.

Usage:
  MEAS_TASK=<task-id> [MEAS_VX=0.5] [MEAS_SECS=12] \
    python scripts/render_policy.py <checkpoint.pt> [out.mp4]
"""
import ctypes
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
ctypes.CDLL("libEGL.so.1", mode=ctypes.RTLD_GLOBAL)

import imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import go2_mjlab  # noqa: F401
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK_ID = os.environ.get(
    "MEAS_TASK",
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15",
)
VX = float(os.environ.get("MEAS_VX", "0.5"))
SECS = float(os.environ.get("MEAS_SECS", "12"))
FPS = 30


def main():
    ckpt = Path(sys.argv[1]).expanduser().resolve()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("videos/policy.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] task={TASK_ID} vx={VX} ckpt={ckpt.name} device={device}")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
    agent_cfg = load_rl_cfg(TASK_ID)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(TASK_ID) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    twist = env.unwrapped.command_manager._terms["twist"]
    sim = env.unwrapped.sim
    step_dt = float(env.unwrapped.step_dt)

    obs, _ = env.reset()
    frames = []
    phases = [("stand", 2.0, (0.0, 0.0, 0.0)), (f"fwd {VX}", SECS, (VX, 0.0, 0.0))]
    for name, dur, cmd in phases:
        cmd_t = torch.tensor([cmd], device=device, dtype=torch.float32)
        for _ in range(int(round(dur / step_dt))):
            twist.vel_command_b[:] = cmd_t
            with torch.no_grad():
                a = policy(obs)
            obs, *_ = env.step(a)
            env.unwrapped._offline_renderer.update(sim.data, camera=None)
            frames.append(env.unwrapped._offline_renderer.render())
    imageio.mimsave(str(out), frames, fps=FPS)
    print(f"[INFO] saved {out}  ({len(frames)} frames @ {FPS}fps)")


if __name__ == "__main__":
    main()
