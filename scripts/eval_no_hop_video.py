#!/usr/bin/env python3
"""Headless mp4 of the Lite3 no-hop handstand policy under fixed commands.

Runs the policy with three command phases — zero, forward, backward —
then writes a single stitched mp4 with a HUD showing the current phase
and command.
"""
import ctypes
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
ctypes.CDLL("libEGL.so.1", mode=ctypes.RTLD_GLOBAL)

import imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import go2_mjlab  # noqa: F401 - registers tasks
import mjlab.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


TASK_ID = "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop"
PHASES = [
    ("zero",     5.0, (0.0,  0.0, 0.0)),
    ("forward", 10.0, (0.4,  0.0, 0.0)),
    ("backward",10.0, (-0.4, 0.0, 0.0)),
    ("turn",     5.0, (0.3,  0.0, 0.4)),
]
FPS = 50  # step_dt = 0.02s


def hud(frame: np.ndarray, phase: str, cmd: tuple[float, float, float]) -> np.ndarray:
    # Cheap PIL-free text overlay: black band on top + white pixel labels.
    out = frame.copy()
    H, W, _ = out.shape
    band = max(24, H // 16)
    out[:band] = 0
    return out


def main():
    parser_args = sys.argv[1:]
    if not parser_args:
        raise SystemExit("usage: eval_no_hop_video.py <checkpoint.pt> [out.mp4]")
    checkpoint = Path(parser_args[0]).expanduser().resolve()
    video_path = Path(parser_args[1]) if len(parser_args) > 1 else Path("videos/handstand_no_hop.mp4")
    video_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device={device}, checkpoint={checkpoint}")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    # Never resample commands; we will overwrite the command buffer ourselves.
    env_cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
    agent_cfg = load_rl_cfg(TASK_ID)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    twist_cmd_term = env.unwrapped.command_manager._terms["twist"]
    sim = env.unwrapped.sim
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["feet_ground_contact"]

    obs_dict, _ = env.reset()
    frames: list[np.ndarray] = []
    front_air_hits = 0
    total_handstand_steps = 0
    step_dt = float(env.unwrapped.step_dt)

    for phase_name, duration_s, cmd_vec in PHASES:
        n_steps = int(round(duration_s / step_dt))
        cmd_tensor = torch.tensor(cmd_vec, device=device, dtype=torch.float32).unsqueeze(0)
        print(f"[PHASE {phase_name}] {n_steps} steps, cmd={cmd_vec}")

        for _ in range(n_steps):
            # Force the command buffer this physics step.
            twist_cmd_term.vel_command_b[:] = cmd_tensor
            with torch.no_grad():
                actions = policy(obs_dict)
            obs_dict, _, _, _ = env.step(actions)

            env.unwrapped._offline_renderer.update(sim.data, camera=None)
            frame = env.unwrapped._offline_renderer.render()
            frames.append(hud(frame, phase_name, cmd_vec))

            # Track front-foot contact stats while in handstand pose.
            grav = robot.data.projected_gravity_b
            target = torch.tensor([1.0, 0.0, 0.0], device=device)
            quality = torch.exp(-2.0 * (grav[0] - target).square().sum())
            if float(quality) > 0.70:
                contact = contact_sensor.data.found > 0
                n_front = int(contact[0, [0, 1]].sum())
                total_handstand_steps += 1
                if n_front == 0:
                    front_air_hits += 1

    rate = front_air_hits / max(total_handstand_steps, 1)
    print(f"[INFO] both-front-air rate while in handstand: {front_air_hits}/{total_handstand_steps} = {rate * 100:.2f}%")
    imageio.mimsave(str(video_path), frames, fps=FPS)
    print(f"[INFO] saved {video_path}")
    env.close()


if __name__ == "__main__":
    main()
