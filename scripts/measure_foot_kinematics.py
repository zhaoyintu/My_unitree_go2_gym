"""Headless front-paw kinematics measurement for the Lite3 handstand-walk.

Objectively answers "did the step get bigger / did the high-frequency contact
stop?" without relying on the phase-locked ``feet_clearance`` reward.

Loads each checkpoint into the SAME 450-dim Stride-V15 play env (the AMP policies
share the 450-dim actor contract; the 'amp' obs group is irrelevant to pure
inference), drives a fixed forward-walk command, and reports per FRONT foot
(FL, FR — the support/stepping paws in this front-paw handstand):

  * excursion  — world-frame foot-height p95-p5 (cm): how big the step is
  * z_std      — foot-height std (cm)
  * apex       — mean foot height while airborne (cm)
  * td_freq    — touchdown frequency (Hz): rising edges of ground contact /
                 second.  HIGH = micro-shuffle; LOW + large excursion = clean step
  * duty       — fraction of time in ground contact

Usage:
  python scripts/measure_foot_kinematics.py LABEL1=CKPT1 LABEL2=CKPT2 ...
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

TASK_ID = os.environ.get(
    "MEAS_TASK",
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15",
)
SETTLE = 150
MEASURE = 600  # 12 s @ 50 Hz
CMD = (float(os.environ.get("MEAS_VX", "0.5")), 0.0, 0.0)  # forward walk (MEAS_VX=0 -> stand)


def measure(env, robot, contact_sensor, fl_fr_ids, policy, twist_term, device, dt):
    cmd_t = torch.tensor([CMD], device=device, dtype=torch.float32)
    obs, _ = env.reset()
    for _ in range(SETTLE):
        twist_term._command = cmd_t.clone()
        with torch.no_grad():
            a = policy(obs)
        obs, *_ = env.step(a)
    zbuf, cbuf = [], []
    for _ in range(MEASURE):
        twist_term._command = cmd_t.clone()
        with torch.no_grad():
            a = policy(obs)
        obs, *_ = env.step(a)
        zbuf.append(robot.data.site_pos_w[0, fl_fr_ids, 2].cpu().numpy().copy())     # (2,) FL,FR
        cbuf.append((contact_sensor.data.found[0, :2] > 0).cpu().numpy().copy())     # (2,) FL,FR
    z = np.stack(zbuf)              # (T,2)
    con = np.stack(cbuf).astype(bool)
    dur = MEASURE * dt
    out = {}
    for i, nm in enumerate(["FL", "FR"]):
        zi, ci = z[:, i], con[:, i]
        rising = int(np.sum((~ci[:-1]) & ci[1:]))      # touchdown events
        exc = float(np.percentile(zi, 95) - np.percentile(zi, 5))
        apex = float(zi[~ci].mean()) if (~ci).any() else float("nan")
        out[nm] = dict(exc=exc, std=float(zi.std()), td_hz=rising / dur,
                       duty=float(ci.mean()), apex=apex, zmax=float(zi.max()))
    return out


def main():
    pairs = [a.split("=", 1) for a in sys.argv[1:]]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}  task={TASK_ID}")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
    agent_cfg = load_rl_cfg(TASK_ID)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(TASK_ID) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)

    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["feet_ground_contact"]
    twist_term = env.unwrapped.command_manager._terms["twist"]
    fl_fr_ids, _ = robot.find_sites(("FL", "FR"), preserve_order=True)
    dt = float(env.unwrapped.step_dt)

    results = {}
    for label, ckpt in pairs:
        ck = Path(ckpt).expanduser().resolve()
        runner.load(str(ck), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        results[label] = measure(env, robot, contact_sensor, fl_fr_ids, policy, twist_term, device, dt)
        print(f"[done] {label}  ({ck.name})")

    print(f"\n==== FRONT-PAW STEP @ cmd vx=0.5  ({MEASURE * dt:.0f}s, {MEASURE} steps) ====")
    print(f"{'policy':20s} {'foot':4s} {'excursion':>9s} {'z_std':>6s} {'apex':>6s} {'td_freq':>8s} {'duty':>5s}")
    for label, r in results.items():
        for nm in ["FL", "FR"]:
            d = r[nm]
            print(f"{label:20s} {nm:4s} {d['exc']*100:7.1f}cm {d['std']*100:5.1f} {d['apex']*100:5.1f} {d['td_hz']:6.2f}Hz {d['duty']:5.2f}")


if __name__ == "__main__":
    main()
