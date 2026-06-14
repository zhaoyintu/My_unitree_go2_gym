"""Measure leg-link self-collision clearance during a GaitLock rollout.

Runs the GaitLock policy in the RLDeploy MuJoCo sim and, each control step,
computes mj_geomDistance between leg collision geoms (mj_geomDistance ignores
contype/conaffinity/excludes — pure geometry), so it catches whether the
big-amplitude stepping makes leg links INTERPENETRATE.  The training/deploy sim
has self-collision DISABLED, so the policy could learn link-crossing poses that
break a real robot — this probe is the check.

Pair groups reported:
  * cross front/rear, same side (FL<->HL, FR<->HR)  -- the user's worry
  * cross left/right (FL<->FR, HL<->HR)
  * within-leg thigh<->shank (knee over-fold)       -- adjacent, info only

Usage:
  python deploy_mujoco_viewer/probe_gaitlock_self_collision.py \
      --policy <gaitlock model.pt> [--cmd 0.5 0 0] [--steps 600]
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import mujoco

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lite3_rldeploy_common import (  # noqa: E402
    ACTION_SCALE, CTRL_DT, DECIMATION, DEFAULT_JOINT_POS, FRAME_STACK,
    TermHistory, actuator_ids, build_model, compute_pd_torque, joint_state,
    policy_terms, reset_state,
)
from deploy_mjlab_lite3_handstand_gaitlock import (  # noqa: E402
    Actor, gait_clock_vec,
)

LEGS = ["FL", "FR", "HL", "HR"]
LINKS = ["THIGH", "SHANK", "FOOT"]


def _gid(model, name):
    try:
        return model.geom(name).id
    except Exception:
        return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--cmd", nargs=3, type=float, default=[0.5, 0.0, 0.0])
    ap.add_argument("--cycle-time", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--settle", type=int, default=100)
    args = ap.parse_args()

    actor = Actor()
    actor.load(args.policy)
    model = build_model()
    data = mujoco.MjData(model)
    reset_state(model, data)
    act_ids = actuator_ids(model)

    cmd = np.asarray(args.cmd, dtype=np.float64)
    last_action = np.zeros(12, dtype=np.float64)
    history = TermHistory()
    history.reset(policy_terms(model, data, cmd, last_action))
    clock = deque(maxlen=FRAME_STACK)
    step_k = 0
    for _ in range(FRAME_STACK):
        clock.append(gait_clock_vec(step_k, args.cycle_time))

    # geom id map (try a couple FOOT name variants)
    gid = {}
    for leg in LEGS:
        for lk in ("THIGH", "SHANK"):
            i = _gid(model, f"{leg}_{lk}_collision")
            if i >= 0:
                gid[(leg, lk)] = i
        for fn in (f"{leg}_FOOT_collision", f"{leg}_foot_collision", f"{leg}_FOOT"):
            i = _gid(model, fn)
            if i >= 0:
                gid[(leg, "FOOT")] = i
                break

    pairs = []  # (label, group, key1, key2)
    # cross front/rear same side
    for f_, r_ in (("FL", "HL"), ("FR", "HR")):
        for a in LINKS:
            for b in LINKS:
                if (f_, a) in gid and (r_, b) in gid:
                    pairs.append((f"{f_}_{a} <-> {r_}_{b}", "front-rear", (f_, a), (r_, b)))
    # cross left/right
    for l_, r_ in (("FL", "FR"), ("HL", "HR")):
        for a in LINKS:
            for b in LINKS:
                if (l_, a) in gid and (r_, b) in gid:
                    pairs.append((f"{l_}_{a} <-> {r_}_{b}", "left-right", (l_, a), (r_, b)))
    # within-leg thigh<->shank (adjacent; info only)
    for leg in LEGS:
        if (leg, "THIGH") in gid and (leg, "SHANK") in gid:
            pairs.append((f"{leg}_THIGH <-> {leg}_SHANK", "within-knee", (leg, "THIGH"), (leg, "SHANK")))

    mind = {p[0]: 1e9 for p in pairs}
    group = {p[0]: p[1] for p in pairs}
    interpen = {p[0]: 0 for p in pairs}
    ft = np.zeros(6, dtype=np.float64)
    knee_min = np.full(12, 1e9)
    knee_max = np.full(12, -1e9)

    for t in range(args.steps):
        obs = np.concatenate([history.build(), np.concatenate(list(clock))]).astype(np.float32)
        action = actor.act(obs).astype(np.float64)
        target_q = action * ACTION_SCALE + DEFAULT_JOINT_POS
        for _ in range(DECIMATION):
            qpos, qvel = joint_state(model, data)
            data.ctrl[act_ids] = compute_pd_torque(target_q, qpos, qvel)
            mujoco.mj_step(model, data)
        last_action = action
        step_k += 1
        history.push(policy_terms(model, data, cmd, last_action))
        clock.append(gait_clock_vec(step_k, args.cycle_time))
        if t >= args.settle:
            qpos, _ = joint_state(model, data)
            knee_min = np.minimum(knee_min, qpos)
            knee_max = np.maximum(knee_max, qpos)
            for label, _grp, k1, k2 in pairs:
                d = mujoco.mj_geomDistance(model, data, gid[k1], gid[k2], 1.0, ft)
                if d < mind[label]:
                    mind[label] = d
                if d < 0:
                    interpen[label] += 1

    n = max(1, args.steps - args.settle)
    print(f"=== GaitLock self-collision probe  (cmd={args.cmd}, {args.steps} steps) ===")
    for grp in ("front-rear", "left-right", "within-knee"):
        print(f"\n[{grp}]  min clearance over rollout (mm):")
        items = sorted([k for k in mind if group[k] == grp], key=lambda k: mind[k])
        for label in items[:8]:
            d = mind[label]
            pct = 100.0 * interpen[label] / n
            flag = "  *** INTERPENETRATE ***" if d < 0 else ("  (grazing)" if d < 0.005 else "")
            print(f"   {label:26s} min={d*1000:+8.2f} mm  interpen={pct:4.0f}% of steps{flag}")
    jn = ["FL_HipX","FL_HipY","FL_Knee","FR_HipX","FR_HipY","FR_Knee",
          "HL_HipX","HL_HipY","HL_Knee","HR_HipX","HR_HipY","HR_Knee"]
    print("\n[knee angle range used] (rad):")
    for i in (2, 5, 8, 11):
        print(f"   {jn[i]:9s} {knee_min[i]:+.2f} .. {knee_max[i]:+.2f}")


if __name__ == "__main__":
    main()
