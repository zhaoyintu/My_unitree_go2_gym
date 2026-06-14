"""MuJoCo sim2sim deploy for the Mjlab-Lite3-Handstand PhaseClock-GaitLock policy.

GaitLock is the phase-clock + hard contact-schedule variant that breaks the
high-frequency front-paw micro-shuffle (big alternating steps).  Its actor obs
is 470-dim — the standard 45x10 term-major RLDeploy input PLUS a gait-phase
clock term (sin/cos of 2*pi*t/cycle_time) appended as the LAST actor term and
stacked over the 10-frame history (so the clock contributes the final 2x10=20
values of the observation).

This viewer reuses the RLDeploy deploy loop (1 kHz PD, 50 Hz policy) from
``lite3_rldeploy_common`` and adds the runtime gait clock.  ANY real-robot
deployment of GaitLock must feed the same sin/cos(2*pi*t/cycle_time) clock — this
script is the sim2sim check of that contract.

Example:
    python deploy_mujoco_viewer/deploy_mjlab_lite3_handstand_gaitlock.py \
        --policy logs/rsl_rl/lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock/<run>/model_3400.pt \
        --cmd 0.5 0 0
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
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

# GaitLock appends a 2-dim (sin, cos) gait clock as the last actor term, stacked
# over the 10-frame history -> +20 values at the END of the term-major obs.
GAIT_CLOCK_DIM = 2
NUM_OBS_GAITLOCK = NUM_OBS + GAIT_CLOCK_DIM * FRAME_STACK  # 450 + 20 = 470
DEFAULT_CYCLE_TIME = 1.0  # must match the PhaseClock-GaitLock env (gait_clock cycle_time)


def gait_clock_vec(step_k: int, cycle_time: float) -> np.ndarray:
    """sin/cos of the gait phase at policy step ``step_k`` (matches the env's
    gait_clock: phase = (episode_length_buf * step_dt) % cycle_time / cycle_time)."""
    phase = ((step_k * CTRL_DT) % cycle_time) / cycle_time
    ang = 2.0 * np.pi * phase
    return np.array([np.sin(ang), np.cos(ang)], dtype=np.float32)


class Actor(nn.Module):
    """Mjlab actor MLP (470-dim input) + EmpiricalNormalization."""

    def __init__(self, num_obs: int = NUM_OBS_GAITLOCK, num_actions: int = 12):
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
                f"{list(checkpoint.keys())[:6]}."
            )
        state_dict = checkpoint["actor_state_dict"]
        obs_tensor = state_dict.get("mlp.0.weight")
        if obs_tensor is None:
            obs_tensor = state_dict["obs_normalizer._mean"]
        ckpt_obs_dim = obs_tensor.shape[-1]
        if ckpt_obs_dim != NUM_OBS_GAITLOCK:
            hint = ""
            if ckpt_obs_dim == NUM_OBS:
                hint = (
                    f" This is a {NUM_OBS}-dim (no gait clock) checkpoint — use "
                    "deploy_mjlab_lite3_handstand_rldeploy.py instead."
                )
            raise RuntimeError(
                f"Checkpoint actor obs dim {ckpt_obs_dim} != the "
                f"{NUM_OBS_GAITLOCK}-dim GaitLock actor.{hint}"
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
    parser.add_argument("--policy", type=str, required=True, help="Path to GaitLock .pt checkpoint")
    parser.add_argument("--cmd", nargs=3, type=float, default=[0.5, 0.0, 0.0],
                        metavar=("vx", "vy", "wz"),
                        help="Constant body-frame velocity command (default forward 0.5)")
    parser.add_argument("--cycle-time", type=float, default=DEFAULT_CYCLE_TIME,
                        help="Gait-clock cycle time (s); MUST match the trained env (GaitLock=1.0)")
    parser.add_argument("--duration", type=float, default=120.0, help="Rollout seconds")
    parser.add_argument("--no-viewer", action="store_true", help="Headless, no GUI")
    parser.add_argument("--no-realtime", action="store_true", help="Don't pace to real time")
    parser.add_argument("--null-policy", action="store_true", help="Zero action (debug)")
    parser.add_argument("--summary", type=int, default=0, metavar="N",
                        help="Print N one-line control-step summaries and exit")
    parser.add_argument("--video", type=str, default=None, help="Write rollout to this MP4 path")
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-camera", type=str, default=None)
    return parser


def main() -> None:
    args = _make_parser().parse_args()

    actor = Actor()
    actor.load(args.policy)
    print(f"[deploy] loaded GaitLock policy: {args.policy}")
    print(
        f"[deploy] actor_obs_dim={NUM_OBS_GAITLOCK} (={NUM_OBS}+{GAIT_CLOCK_DIM}x{FRAME_STACK} clock), "
        f"cycle_time={args.cycle_time}s, cmd={args.cmd}, policy_dt={CTRL_DT * 1000:.1f}ms"
    )

    model = build_model()
    data = mujoco.MjData(model)
    reset_state(model, data)
    act_ids = actuator_ids(model)

    cmd = np.asarray(args.cmd, dtype=np.float64)
    last_action = np.zeros(12, dtype=np.float64)
    history = TermHistory()
    history.reset(policy_terms(model, data, cmd, last_action))

    # Gait-clock history (last term, stacked over FRAME_STACK frames).
    step_k = 0
    clock_hist: deque[np.ndarray] = deque(maxlen=FRAME_STACK)
    for _ in range(FRAME_STACK):
        clock_hist.append(gait_clock_vec(step_k, args.cycle_time))

    def build_obs() -> np.ndarray:
        base = history.build()                                  # 450, term-major
        clock = np.concatenate(list(clock_hist), axis=0)        # 20, last term x10
        obs = np.concatenate([base, clock], axis=0).astype(np.float32)
        if obs.shape != (NUM_OBS_GAITLOCK,):
            raise RuntimeError(f"obs shape {obs.shape} != ({NUM_OBS_GAITLOCK},)")
        return obs

    def step_once() -> dict[str, np.ndarray]:
        nonlocal last_action, step_k
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
        step_k += 1
        history.push(policy_terms(model, data, cmd, last_action))
        clock_hist.append(gait_clock_vec(step_k, args.cycle_time))
        return {"obs": obs, "action": action, "target_q": target_q, "tau": tau}

    n_steps = int(args.duration / CTRL_DT)

    if args.summary > 0:
        print("\nstep  time   base_z  phase  max|a|  max|tau|")
        for i in range(args.summary):
            info = step_once()
            phase = ((step_k * CTRL_DT) % args.cycle_time) / args.cycle_time
            print(
                f"{i:4d}  {data.time:5.2f}  {data.qpos[2]:6.3f}  {phase:5.2f}  "
                f"{np.abs(info['action']).max():6.2f}  {np.abs(info['tau']).max():7.2f}"
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
