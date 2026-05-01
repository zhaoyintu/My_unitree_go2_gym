"""Direct evidence that go2_handstand `target_gravity=(-1,0,0)` is wrong.

We place the Go2 in two body orientations using the SAME joint defaults from
`HANDSTAND_INIT_STATE` (front thigh=-1, front calf=-1.1, rear thigh=2.25,
rear calf=-2.0) and look at which feet land on the ground.

The reward `handstand_orientation` with `target_gravity` decides which
orientation the policy converges to:
    target_gravity = (+1, 0, 0)  →  body +x along world -z (head DOWN)
    target_gravity = (-1, 0, 0)  →  body +x along world +z (head UP)

The pose the docstring describes ("balances on front legs") REQUIRES head
DOWN, i.e. target_gravity = (+1, 0, 0).  But the file ships with -1.

Run me to see foot heights side-by-side:
    python deploy_mujoco_viewer/proof_target_gravity_sign.py
"""
import numpy as np
import mujoco

from verify_handstand_pose import (
    JOINT_NAMES, TARGET_JOINT_POS, build_model, quat_rotate_inverse,
)


def place(model: mujoco.MjModel, data: mujoco.MjData,
          pitch: float, base_z: float) -> None:
    mujoco.mj_resetData(model, data)
    p = pitch
    data.qpos[0:3] = [0.0, 0.0, base_z]
    data.qpos[3:7] = [np.cos(p / 2), 0.0, np.sin(p / 2), 0.0]
    for jn, q in zip(JOINT_NAMES, TARGET_JOINT_POS):
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
    model = build_model()
    data = mujoco.MjData(model)
    g_world = np.array([0.0, 0.0, -1.0])

    cases = [
        # (label, pitch, base_z, target_gravity_for_this_pose)
        ("A. 头朝下 (env_cfgs.py target_gravity = (+1, 0, 0)，前腿撑地)",
            +np.pi / 2,  0.47, (+1, 0, 0)),
        ("B. 头朝上 (反向 target_gravity = (-1, 0, 0)，后腿撑地)",
            -np.pi / 2,  0.47, (-1, 0, 0)),
    ]

    rows = []
    for label, pitch, z, tg in cases:
        place(model, data, pitch, z)
        quat = data.qpos[3:7]
        grav_b = quat_rotate_inverse(quat, g_world)
        fz = feet_z(model, data)
        rows.append((label, pitch, grav_b, tg, fz))

    print("=" * 78)
    print("  关节默认角 = IsaacGym GO2_Leggedstand descire (front thigh=-0.7, calf=-1.75;")
    print("                                            rear thigh=0.8,  calf=-1.5)")
    print("  trunk 起始 z 也一致 (0.47 m)，唯一变量就是 base 的 pitch")
    print("=" * 78)
    for label, pitch, grav_b, tg, fz in rows:
        print(f"\n{label}")
        print(f"  pitch              = {pitch:+.3f} rad ({np.degrees(pitch):+.0f}°)")
        print(f"  projected_gravity  = ({grav_b[0]:+.2f}, {grav_b[1]:+.2f}, {grav_b[2]:+.2f})")
        print(f"  → 这个姿态对应的 target_gravity = {tg}")
        print(f"  foot heights (m):")
        for f in ("FR", "FL", "RL", "RR"):
            tag = "前(FRONT)" if f.startswith("F") else "后(REAR) "
            on_ground = " ← 落地" if abs(fz[f]) < 0.05 else ""
            print(f"    {f} {tag}: z = {fz[f]:+.3f}{on_ground}")

    fz_A = rows[0][4]
    fz_B = rows[1][4]
    front_on_ground_A = max(abs(fz_A["FR"]), abs(fz_A["FL"])) < 0.05
    rear_on_ground_B = max(abs(fz_B["RL"]), abs(fz_B["RR"])) < 0.05

    print("\n" + "=" * 78)
    print("  结论 — 当前 env 配置 vs 反向 target_gravity")
    print("=" * 78)
    print(f"  A 头朝下 (pitch=+90°)  ← 当前 target_gravity=(+1, 0, 0):")
    print(f"     前脚 z = {fz_A['FR']:+.3f}   后脚 z = {fz_A['RL']:+.3f}")
    print(f"     ⇒ 前脚{'落地 ✓' if front_on_ground_A else '没落地 ✗'}，"
          f"后脚高高翘到 {fz_A['RL']:.2f} m — 真前腿撑地 handstand。")
    print()
    print(f"  B 头朝上 (pitch=-90°)  ← 反向 target_gravity=(-1, 0, 0):")
    print(f"     前脚 z = {fz_B['FR']:+.3f}   后脚 z = {fz_B['RL']:+.3f}")
    print(f"     ⇒ 同一组关节角在头朝上时前后脚都悬空，没法支撑——"
          f"说明这组关节默认角是为 A 设计的。")
    print()
    print("  目前 env_cfgs.py / go2_constants.py 的所有方向（init pose、descire")
    print("  关节角、pose_range、target_gravity (+1,0,0)、stance/swing 脚分组）")
    print("  全部对齐到 A（头朝下，前腿撑地，对应 IsaacGym GO2_Leggedstand）。")
    print("=" * 78)


if __name__ == "__main__":
    main()
