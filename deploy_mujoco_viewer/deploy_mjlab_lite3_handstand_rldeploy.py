"""MuJoCo deploy for Mjlab-Lite3-Handstand-RLDeploy checkpoints.

This is the RLDeploy-aligned Lite3 handstand viewer.  It differs from the
older `deploy_mjlab_lite3_handstand.py` by matching the real deploy loop:
motor torque PD at 1 kHz, 50 Hz policy, no reflected joint armature, and the
45 x 10 term-major actor input.

Example:
    python deploy_mujoco_viewer/deploy_mjlab_lite3_handstand_rldeploy.py \
        --policy logs/rsl_rl/lite3_handstand_rldeploy/<run>/model_9000.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import mujoco
import mujoco.viewer


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lite3_rldeploy_common import (  # noqa: E402
    ACTION_SCALE,
    CTRL_DT,
    DECIMATION,
    DEFAULT_JOINT_POS,
    FRAME_STACK,
    NUM_OBS,
    NUM_SINGLE_OBS,
    SIM_DT,
    TermHistory,
    actuator_ids,
    build_model,
    check_finite,
    compute_pd_torque,
    joint_state,
    policy_terms,
    print_status,
    record_video,
    reset_state,
)


class Actor(nn.Module):
    """Mjlab actor MLP plus EmpiricalNormalization checkpoint state."""

    def __init__(self, num_obs: int = NUM_OBS, num_actions: int = 12):
        super().__init__()
        layers = []
        in_dim = num_obs
        for hidden_dim in (512, 256, 128):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ELU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_actions))
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("mean", torch.zeros(1, num_obs))
        self.register_buffer("std", torch.ones(1, num_obs))
        self.eps = 1e-2

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "actor_state_dict" not in checkpoint:
            raise KeyError(
                f"checkpoint at {path} missing 'actor_state_dict'; got keys "
                f"{list(checkpoint.keys())[:6]}. This script expects an Mjlab "
                "actor checkpoint. For IsaacGym model_state_dict checkpoints, "
                "use deploy_isaacgym_lite3_handstand_rldeploy.py."
            )

        state_dict = checkpoint["actor_state_dict"]
        checkpoint_obs_tensor = state_dict.get("mlp.0.weight")
        if checkpoint_obs_tensor is None:
            checkpoint_obs_tensor = state_dict["obs_normalizer._mean"]
        checkpoint_obs_dim = checkpoint_obs_tensor.shape[-1]
        if checkpoint_obs_dim != NUM_OBS:
            hint = ""
            if checkpoint_obs_dim == 480:
                hint = (
                    " This looks like an old 48 x 10 IsaacGym checkpoint; "
                    "use deploy_mujoco_viewer/deploy_isaacgym_lite3_handstand.py."
                )
            raise RuntimeError(
                "Checkpoint actor obs dim "
                f"{checkpoint_obs_dim} does not match the {NUM_OBS}-dim "
                "actor for Mjlab-Lite3-Handstand-RLDeploy."
                f"{hint}"
            )

        mlp_state_dict = {
            key[len("mlp."):]: value
            for key, value in state_dict.items()
            if key.startswith("mlp.")
        }
        self.mlp.load_state_dict(mlp_state_dict)
        self.mean.copy_(state_dict["obs_normalizer._mean"].reshape(1, -1))
        self.std.copy_(state_dict["obs_normalizer._std"].reshape(1, -1))
        self.eval()

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        obs = torch.from_numpy(obs_np).float().unsqueeze(0)
        obs = (obs - self.mean) / (self.std + self.eps)
        return self.mlp(obs).squeeze(0).numpy()


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--cmd", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        metavar=("vx", "vy", "wz"),
                        help="Constant body-frame velocity command")
    parser.add_argument("--duration", type=float, default=120.0,
                        help="Rollout duration in seconds")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Run headless without launching a viewer")
    parser.add_argument("--no-realtime", action="store_true",
                        help="Do not pace GUI loop to real time")
    parser.add_argument("--null-policy", action="store_true",
                        help="Use zero action instead of the checkpoint policy")
    parser.add_argument("--summary", type=int, default=0, metavar="N",
                        help="Print N one-line control-step summaries and exit")
    parser.add_argument("--video", type=str, default=None,
                        help="Write rollout to this MP4 path")
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-camera", type=str, default=None)
    return parser


def main() -> None:
    args = _make_parser().parse_args()

    actor = Actor()
    actor.load(args.policy)
    print(f"[deploy] loaded Mjlab RLDeploy policy: {args.policy}")
    print(
        f"[deploy] actor_obs_dim={NUM_OBS}, single_obs={NUM_SINGLE_OBS}, "
        f"frame_stack={FRAME_STACK}, sim_dt={SIM_DT}, decimation={DECIMATION}"
    )
    print(f"[deploy] cmd={args.cmd}, policy_dt={CTRL_DT * 1000:.1f} ms")

    model = build_model()
    data = mujoco.MjData(model)
    reset_state(model, data)
    act_ids = actuator_ids(model)

    cmd = np.asarray(args.cmd, dtype=np.float64)
    last_action = np.zeros(12, dtype=np.float64)
    history = TermHistory()
    history.reset(policy_terms(model, data, cmd, last_action))

    def build_obs() -> np.ndarray:
        obs = history.build()
        if obs.shape != (NUM_OBS,):
            raise RuntimeError(f"obs shape {obs.shape} != ({NUM_OBS},)")
        return obs

    def step_once() -> dict[str, np.ndarray]:
        nonlocal last_action
        obs = build_obs()
        check_finite("obs", obs, t=data.time, base_z=data.qpos[2])
        if args.null_policy:
            action = np.zeros(12, dtype=np.float64)
        else:
            action = actor.act(obs).astype(np.float64)
        check_finite("action", action, t=data.time)

        target_q = action * ACTION_SCALE + DEFAULT_JOINT_POS
        check_finite("target_q", target_q, t=data.time)
        tau = np.zeros(12, dtype=np.float64)
        for _ in range(DECIMATION):
            qpos, qvel = joint_state(model, data)
            tau = compute_pd_torque(target_q, qpos, qvel)
            data.ctrl[act_ids] = tau
            mujoco.mj_step(model, data)

        last_action = action.astype(np.float64)
        history.push(policy_terms(model, data, cmd, last_action))
        return {"obs": obs, "action": action, "target_q": target_q, "tau": tau}

    n_steps = int(args.duration / CTRL_DT)

    if args.summary > 0:
        print("\nstep  time   base_z  max|a|  max|tau|")
        for i in range(args.summary):
            step_info = step_once()
            print(
                f"{i:4d}  {data.time:5.2f}  {data.qpos[2]:6.3f}  "
                f"{np.abs(step_info['action']).max():6.2f}  "
                f"{np.abs(step_info['tau']).max():7.2f}"
            )
        return

    if args.video:
        record_video(model, data, step_once, n_steps, args)
        return

    if args.no_viewer:
        for i in range(n_steps):
            step_once()
            if (i + 1) % int(round(1.0 / CTRL_DT)) == 0:
                print_status(data)
        return

    print("[viewer] SPACE = pause/resume, close window to exit")
    paused = [False]

    def key_callback(keycode: int) -> None:
        if keycode == 32:
            paused[0] = not paused[0]
            print(f"[viewer] {'paused' if paused[0] else 'resumed'}")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -10.0
        viewer.cam.azimuth = 90.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.4]

        i = 0
        while i < n_steps and viewer.is_running():
            t0 = time.time()
            if not paused[0]:
                step_once()
                i += 1
            viewer.sync()
            if not args.no_realtime:
                sleep_time = CTRL_DT - (time.time() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)


if __name__ == "__main__":
    main()
