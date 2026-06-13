"""Dump per-phase joint-angle traces for the Stride-V12 (v14) policy.

Loads the checkpoint, runs the same 4-phase command sequence as
``eval_no_hop_video.py``, and prints a compact summary of left-vs-right
joint values so we can see which leg is stuck folded.
"""
import ctypes
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
ctypes.CDLL("libEGL.so.1", mode=ctypes.RTLD_GLOBAL)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import go2_mjlab  # noqa: F401 - registers tasks
import mjlab.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


TASK_ID = "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V12"
PHASES = [
    ("zero",     250, (0.0,  0.0, 0.0)),
    ("forward",  500, (0.4,  0.0, 0.0)),
    ("backward", 500, (-0.4, 0.0, 0.0)),
    ("turn",     250, (0.3,  0.0, 0.4)),
]


def main():
    checkpoint = Path(sys.argv[1]).expanduser().resolve()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device={device}, checkpoint={checkpoint}")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
    agent_cfg = load_rl_cfg(TASK_ID)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    twist_cmd_term = env.unwrapped.command_manager._terms["twist"]
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["feet_ground_contact"]

    # Lite3 joint order in MJCF: FL_HipX/HipY/Knee, FR..., HL..., HR...
    joint_names = [
        "FL_HipX", "FL_HipY", "FL_Knee",
        "FR_HipX", "FR_HipY", "FR_Knee",
        "HL_HipX", "HL_HipY", "HL_Knee",
        "HR_HipX", "HR_HipY", "HR_Knee",
    ]

    obs, _ = env.reset()
    for phase_name, n_steps, cmd in PHASES:
        cmd_t = torch.tensor([cmd], device=device, dtype=torch.float32)
        joint_buf = []
        contact_buf = []
        for _ in range(n_steps):
            twist_cmd_term._command = cmd_t.clone()
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            joint_buf.append(robot.data.joint_pos[0].cpu().numpy().copy())
            contact_buf.append((contact_sensor.data.found[0] > 0).cpu().numpy().copy())
        joints = np.stack(joint_buf)  # [n_steps, 12]
        contacts = np.stack(contact_buf)  # [n_steps, 4]

        print(f"\n=== Phase {phase_name} (cmd={cmd}) ===")
        print(f"  {'joint':10s}  {'mean':>7s}  {'std':>6s}  {'min':>7s}  {'max':>7s}")
        for i, name in enumerate(joint_names):
            j = joints[:, i]
            print(f"  {name:10s}  {j.mean():+7.3f}  {j.std():6.3f}  {j.min():+7.3f}  {j.max():+7.3f}")

        # L-R asymmetry: FL vs FR, HL vs HR pair diffs
        print(f"\n  L-R differences (front):")
        for i_l, i_r, joint in [(1, 4, "HipY"), (2, 5, "Knee")]:
            d = (joints[:, i_l] - joints[:, i_r])
            print(f"    FL_{joint:5s} - FR_{joint:5s}: mean={d.mean():+.3f}, max|diff|={np.abs(d).max():.3f}")
        print(f"  L-R differences (rear):")
        for i_l, i_r, joint in [(7, 10, "HipY"), (8, 11, "Knee")]:
            d = (joints[:, i_l] - joints[:, i_r])
            print(f"    HL_{joint:5s} - HR_{joint:5s}: mean={d.mean():+.3f}, max|diff|={np.abs(d).max():.3f}")

        # Contact fraction
        print(f"\n  Contact fraction: FL={contacts[:,0].mean():.2f} FR={contacts[:,1].mean():.2f} HL={contacts[:,2].mean():.2f} HR={contacts[:,3].mean():.2f}")


if __name__ == "__main__":
    main()
