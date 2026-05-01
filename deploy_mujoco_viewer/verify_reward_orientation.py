"""Post-fix verification for go2_handstand env_cfgs.py.

After flipping target_gravity from (-1, 0, 0) to (+1, 0, 0), confirm:
  1. The config file actually holds the new value at runtime.
  2. With the robot placed at the intended handstand pose (pitch=+π/2,
     head DOWN, front feet on ground), the projected_gravity OBSERVATION
     equals the target_gravity reward parameter.
  3. The handstand_orientation REWARD evaluates to ~0 at that pose
     (perfect-match → no penalty).
  4. With the robot at the OLD (wrong) target pose (pitch=-π/2, head UP),
     the same reward evaluates to ~4 (squared distance |(−1,0,0)−(+1,0,0)|² = 4),
     proving the new reward correctly distinguishes the two orientations.
"""
import ast
from pathlib import Path

import numpy as np
import mujoco

from verify_handstand_pose import (
    JOINT_NAMES, TARGET_JOINT_POS, build_model, quat_rotate_inverse,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = REPO_ROOT / "go2_mjlab/config/go2_handstand/env_cfgs.py"


def parse_target_gravity_from_source() -> tuple[float, float, float]:
    """Read the literal target_gravity tuple straight out of env_cfgs.py.

    Avoids importing mjlab (heavy + GPU-required), but still proves we read
    the same source the trainer reads.
    """
    src = ENV_CFG.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "target_gravity":
                    return tuple(float(c.value) for c in v.elts)
    raise RuntimeError("target_gravity not found in env_cfgs.py")


def place(model, data, pitch: float, base_z: float):
    mujoco.mj_resetData(model, data)
    p = pitch
    data.qpos[0:3] = [0.0, 0.0, base_z]
    data.qpos[3:7] = [np.cos(p / 2), 0.0, np.sin(p / 2), 0.0]
    for jn, q in zip(JOINT_NAMES, TARGET_JOINT_POS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def handstand_orientation_reward(grav_b: np.ndarray,
                                 target: tuple[float, float, float]) -> float:
    """Replicates go2_mjlab/mdp/rewards.py::handstand_orientation."""
    return float(np.sum((grav_b - np.asarray(target)) ** 2))


def main():
    ok = True

    # 1. Config sanity
    target_gravity = parse_target_gravity_from_source()
    expected = (1.0, 0.0, 0.0)
    print("1. env_cfgs.py target_gravity =", target_gravity)
    if target_gravity == expected:
        print("   ✓ matches the expected fix (+1, 0, 0)")
    else:
        print(f"   ✗ expected {expected}, got {target_gravity}")
        ok = False

    # 2 & 3. Place at intended pose and verify obs/reward agree.
    model = build_model()
    data = mujoco.MjData(model)
    g_world = np.array([0.0, 0.0, -1.0])

    place(model, data, pitch=+np.pi / 2, base_z=0.47)
    quat = data.qpos[3:7]
    grav_b = quat_rotate_inverse(quat, g_world)
    reward = handstand_orientation_reward(grav_b, target_gravity)
    front_z = (data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "FR")][2]
               + data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "FL")][2]) / 2

    print("\n2. INTENDED handstand pose (pitch=+π/2, head DOWN):")
    print(f"   projected_gravity (obs)  = "
          f"({grav_b[0]:+.3f}, {grav_b[1]:+.3f}, {grav_b[2]:+.3f})")
    print(f"   target_gravity (reward)  = {target_gravity}")
    if np.allclose(grav_b, target_gravity, atol=1e-3):
        print("   ✓ obs equals target — orientation aligned")
    else:
        print("   ✗ obs differs from target")
        ok = False
    print(f"   front feet avg z         = {front_z:+.3f} m   "
          f"({'落地 ✓' if abs(front_z) < 0.05 else 'not on ground ✗'})")

    print(f"\n3. handstand_orientation reward at this pose:")
    print(f"   square‖projected_gravity − target‖² = {reward:.4f}")
    if reward < 1e-3:
        print("   ✓ ≈ 0  (no penalty when at intended pose)")
    else:
        print("   ✗ should be ~0 at the intended pose")
        ok = False

    # 4. Place at the OLD wrong pose and verify the reward distinguishes them.
    place(model, data, pitch=-np.pi / 2, base_z=0.47)
    quat = data.qpos[3:7]
    grav_b_wrong = quat_rotate_inverse(quat, g_world)
    reward_wrong = handstand_orientation_reward(grav_b_wrong, target_gravity)

    print(f"\n4. OLD (wrong) pose (pitch=-π/2, head UP) — sanity check:")
    print(f"   projected_gravity (obs)  = "
          f"({grav_b_wrong[0]:+.3f}, {grav_b_wrong[1]:+.3f}, {grav_b_wrong[2]:+.3f})")
    print(f"   handstand_orientation reward = {reward_wrong:.4f}  "
          f"(expect ≈ 4.0 = ‖(−1)−(+1)‖²)")
    if abs(reward_wrong - 4.0) < 1e-3:
        print("   ✓ reward correctly penalizes head-up orientation")
    else:
        print("   ✗ unexpected value")
        ok = False

    print("\n" + ("=" * 60))
    print("OVERALL:", "PASS ✓" if ok else "FAIL ✗")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
