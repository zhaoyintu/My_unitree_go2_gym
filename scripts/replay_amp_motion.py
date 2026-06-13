"""Replay an AMP reference motion clip in the mjlab sim and render to mp4.

Sets the robot to each npz frame's root pose + joint positions, forwards the
sim (kinematics only — no policy, no physics integration), renders, and writes
a video.  Used to visually verify what a reference clip actually does (e.g.
whether the front paws alternate or move together).

Usage:
  python scripts/replay_amp_motion.py <motion.npz> [out.mp4] [loops]
"""
import ctypes
import os
import sys
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
from mjlab.tasks.registry import load_env_cfg

TASK_ID = "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15"
FPS = 30  # render slower than the 50 Hz clip so the gait is easy to see


def main():
    npz = Path(sys.argv[1]).expanduser().resolve()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("videos") / (npz.stem + "_replay.mp4")
    loops = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    out.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    d = np.load(npz)
    root_pos = d["root_pos"].astype(np.float32)              # (N,3)
    q_xyzw = d["root_quat_xyzw"].astype(np.float32)          # (N,4)
    jp = d["joint_pos"].astype(np.float32)                   # (N,12)
    jv = d["joint_vel"].astype(np.float32)                   # (N,12)
    N = root_pos.shape[0]
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    print(f"[INFO] {npz.name}: {N} frames ({N/50:.2f}s), device={device}")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env.reset()
    robot = env.scene["robot"]
    sim = env.sim
    eid = torch.arange(1, device=device)

    frames = []
    for _ in range(loops):
        for t in range(N):
            pose = torch.tensor(np.concatenate([root_pos[t], q_wxyz[t]])[None], device=device)
            robot.write_root_link_pose_to_sim(pose, eid)
            robot.write_joint_position_to_sim(torch.tensor(jp[t][None], device=device), env_ids=eid)
            robot.write_joint_velocity_to_sim(torch.tensor(jv[t][None], device=device), env_ids=eid)
            env.scene.write_data_to_sim()
            sim.forward()
            env.unwrapped._offline_renderer.update(sim.data, camera=None)
            frames.append(env.unwrapped._offline_renderer.render())
    imageio.mimsave(str(out), frames, fps=FPS)
    print(f"[INFO] saved {out}  ({len(frames)} frames @ {FPS}fps)")


if __name__ == "__main__":
    main()
