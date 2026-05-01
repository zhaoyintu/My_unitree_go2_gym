"""End-to-end consistency check for the front-paw handstand-walking task.

After porting IsaacGym `GO2_Leggedstand` (the IsaacGym task that actually
trains a front-paw handstand walk) into mjlab `go2_handstand`, this script
checks that every coupled piece is in agreement:

    A. HANDSTAND_INIT_STATE  ⇒ normal four-paw stand (all feet on ground)
    B. The descire (target) pose at base_z=0.47 ⇒
         • projected_gravity_b == target_gravity == (+1, 0, 0)
         • front feet (stance: indices 0,1) on the ground
         • rear  feet (swing : indices 2,3) at world z ≈ 0.67
    C. env_cfgs.py reward foot-indices are wired to those roles:
         • handstand_contact / feet_air_time → (0, 1)  stance / FRONT
         • handstand_feet_on_air / handstand_feet_height_exp / feet_clearance
                                              → (2, 3)  swing  / REAR
         • handstand_feet_height_exp.target_height = 0.67
         • base_height.target_height            = 0.47

No mjlab import needed — env_cfgs.py is parsed with `ast`.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import mujoco

from verify_handstand_pose import (
    JOINT_NAMES, build_model, quat_rotate_inverse,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = REPO_ROOT / "go2_mjlab/config/go2_handstand/env_cfgs.py"
GO2_CONSTANTS = REPO_ROOT / "go2_mjlab/robots/go2_constants.py"

DESIRE_POSE = np.array([
    0.0, -0.7, -1.75,   # FR  stance (front)
    0.0, -0.7, -1.75,   # FL  stance (front)
    0.0,  0.8, -1.5,    # RL  swing  (rear)
    0.0,  0.8, -1.5,    # RR  swing  (rear)
])
INIT_POSE = np.array([
    # IsaacGym GO2_Leggedstand init: standing dog
    +0.1, 0.8, -1.5,    # FR  hip=+0.1 (R-side)
    -0.1, 0.8, -1.5,    # FL  hip=-0.1 (L-side)
    -0.1, 1.0, -1.5,    # RL  hip=-0.1 (L-side)
    +0.1, 1.0, -1.5,    # RR  hip=+0.1 (R-side)
])


def _parse_reward_kwargs(name: str) -> dict[str, object]:
    """Pull the params dict for a named reward entry out of env_cfgs.py."""
    src = ENV_CFG.read_text()
    tree = ast.parse(src)

    # Look for `rewards = { ... }` inside the env-config function and find
    # the entry whose key string == `name`.
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "rewards"
                    and isinstance(node.value, ast.Dict)):
                continue
            for k, v in zip(node.value.keys, node.value.values):
                if not (isinstance(k, ast.Constant) and k.value == name):
                    continue
                # v should be a `RewardTermCfg(..., params={...})` Call.
                if not isinstance(v, ast.Call):
                    return {}
                for kw in v.keywords:
                    if kw.arg == "params" and isinstance(kw.value, ast.Dict):
                        out: dict[str, object] = {}
                        for pk, pv in zip(kw.value.keys, kw.value.values):
                            if not isinstance(pk, ast.Constant):
                                continue
                            try:
                                out[pk.value] = ast.literal_eval(pv)
                            except ValueError:
                                pass
                        return out
    return {}


def _parse_init_pos_z() -> float:
    src = GO2_CONSTANTS.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "HANDSTAND_INIT_STATE"
                and isinstance(node.value, ast.Call)):
            for kw in node.value.keywords:
                if kw.arg == "pos" and isinstance(kw.value, ast.Tuple):
                    return float(ast.literal_eval(kw.value.elts[2]))
    raise RuntimeError("HANDSTAND_INIT_STATE.pos not found")


def place(model, data, pitch: float, base_z: float, joint_pose: np.ndarray):
    mujoco.mj_resetData(model, data)
    p = pitch
    data.qpos[0:3] = [0.0, 0.0, base_z]
    data.qpos[3:7] = [np.cos(p / 2), 0.0, np.sin(p / 2), 0.0]
    for jn, q in zip(JOINT_NAMES, joint_pose):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def feet_z(model, data) -> dict[str, float]:
    out = {}
    for fname in ("FR", "FL", "RL", "RR"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, fname)
        out[fname] = float(data.site_xpos[sid][2])
    return out


def main():
    ok = True
    model = build_model()
    data = mujoco.MjData(model)
    g_world = np.array([0.0, 0.0, -1.0])

    # ---- A. Init pose is normal four-paw stand --------------------------
    # IsaacGym init.pos = 0.42 starts the dog ~10cm above the ground (so
    # feet don't penetrate); physics then settles it.  Hold the init joint
    # angles with PD for 0.5 s and check the dog ends up on its four feet.
    init_z = _parse_init_pos_z()
    place(model, data, pitch=0.0, base_z=init_z, joint_pose=INIT_POSE)
    act_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
        for jn in JOINT_NAMES
    ])
    data.ctrl[act_ids] = INIT_POSE
    for _ in range(int(0.5 / 0.005)):
        mujoco.mj_step(model, data)

    fz = feet_z(model, data)
    quat = data.qpos[3:7]
    grav_b_init = quat_rotate_inverse(quat, g_world)
    trunk_z = float(data.qpos[2])
    print(f"A. HANDSTAND_INIT_STATE (pos.z = {init_z:.3f}, settled 0.5 s):")
    print(f"   trunk z = {trunk_z:.3f},  projected_gravity (body) = "
          f"({grav_b_init[0]:+.3f}, {grav_b_init[1]:+.3f}, {grav_b_init[2]:+.3f})")
    print(f"   FR/FL/RL/RR foot z = "
          f"{fz['FR']:+.3f} / {fz['FL']:+.3f} / {fz['RL']:+.3f} / {fz['RR']:+.3f}")

    upright = np.allclose(grav_b_init, (0.0, 0.0, -1.0), atol=0.05)
    all_on_ground = all(abs(fz[f]) < 0.04 for f in fz)
    if upright and all_on_ground:
        print("   ✓ dog settles upright with all four feet on the ground")
    else:
        if not upright:
            print("   ✗ body not upright after settle")
        if not all_on_ground:
            print("   ✗ at least one foot not on ground after settle")
        ok = False

    # ---- B. Desire (handstand) pose ------------------------------------
    place(model, data, pitch=+np.pi / 2, base_z=0.47, joint_pose=DESIRE_POSE)
    quat = data.qpos[3:7]
    grav_b = quat_rotate_inverse(quat, g_world)
    fz = feet_z(model, data)

    print("\nB. Descire pose at base_z=0.47, pitch=+π/2 (head DOWN):")
    print(f"   projected_gravity (body)  = "
          f"({grav_b[0]:+.3f}, {grav_b[1]:+.3f}, {grav_b[2]:+.3f})")
    print(f"   front (stance) FR/FL z   = {fz['FR']:+.3f} / {fz['FL']:+.3f}")
    print(f"   rear  (swing)  RL/RR z   = {fz['RL']:+.3f} / {fz['RR']:+.3f}")

    grav_ok = np.allclose(grav_b, (1.0, 0.0, 0.0), atol=2e-3)
    front_on_ground = max(abs(fz["FR"]), abs(fz["FL"])) < 0.03
    rear_at_target = abs(fz["RL"] - 0.67) < 0.05 and abs(fz["RR"] - 0.67) < 0.05
    if grav_ok:
        print("   ✓ gravity matches target (+1, 0, 0)")
    else:
        print("   ✗ gravity off")
        ok = False
    if front_on_ground:
        print("   ✓ stance (front) feet on ground")
    else:
        print("   ✗ stance feet not on ground")
        ok = False
    if rear_at_target:
        print("   ✓ swing (rear) feet near target world z = 0.67")
    else:
        print("   ✗ swing feet off target world z")
        ok = False

    # ---- C. env_cfgs.py reward wiring ----------------------------------
    print("\nC. env_cfgs.py reward foot_indices wiring:")

    expectations = [
        # (reward name, expected key, expected value)
        ("contact",                 "foot_indices", (0, 1)),
        ("feet_air_time",           "foot_indices", (0, 1)),
        ("handstand_feet_on_air",   "foot_indices", (2, 3)),
        ("handstand_feet_height_exp", "foot_indices", (2, 3)),
        ("feet_clearance",          "foot_indices", (2, 3)),
        ("handstand_feet_height_exp", "target_height", 0.67),
        ("base_height",             "target_height", 0.47),
        ("handstand_orientation",   "target_gravity", (1.0, 0.0, 0.0)),
    ]
    for reward_name, key, expected in expectations:
        kwargs = _parse_reward_kwargs(reward_name)
        actual = kwargs.get(key, "<missing>")
        match = actual == expected
        symbol = "✓" if match else "✗"
        print(f"   {symbol} rewards['{reward_name}'].params['{key}'] = {actual!r}  "
              f"(expected {expected!r})")
        if not match:
            ok = False

    print("\n" + "=" * 60)
    print("OVERALL:", "PASS ✓" if ok else "FAIL ✗")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
