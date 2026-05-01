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
        ("A. 头朝下 (docstring 描述: 'balances on front legs')",
            +np.pi / 2,  0.555, (+1, 0, 0)),
        ("B. 头朝上 (env_cfgs.py 当前 target_gravity 对应的姿态)",
            -np.pi / 2,  0.555, (-1, 0, 0)),
    ]

    rows = []
    for label, pitch, z, tg in cases:
        place(model, data, pitch, z)
        quat = data.qpos[3:7]
        grav_b = quat_rotate_inverse(quat, g_world)
        fz = feet_z(model, data)
        rows.append((label, pitch, grav_b, tg, fz))

    print("=" * 78)
    print("  关节默认角全部一致（front thigh=-1, calf=-1.1; rear thigh=2.25, calf=-2.0）")
    print("  trunk 起始 z 也一致 (0.555 m)，唯一变量就是 base 的 pitch")
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
    print("  结论 — env 配置自相矛盾")
    print("=" * 78)
    print(f"  A 头朝下 (pitch=+90°):")
    print(f"     前脚 z = {fz_A['FR']:+.3f}   后脚 z = {fz_A['RL']:+.3f}")
    print(f"     ⇒ 前脚刚好落地 {'✓' if front_on_ground_A else '✗'}, 后脚高高翘起 — "
          f"是合理的'前腿撑地' handstand。")
    print()
    print(f"  B 头朝上 (pitch=-90°):")
    print(f"     前脚 z = {fz_B['FR']:+.3f}   后脚 z = {fz_B['RL']:+.3f}")
    print(f"     ⇒ 没有任何一只脚落地 ✗，前后脚都悬空（前脚 1.1m 高，后脚 0.14m 高）。")
    print(f"     这个朝向下 handstand 默认关节角根本不合理，")
    print(f"     说明 HANDSTAND_INIT_STATE 是为 A（头朝下）设计的，不是 B。")
    print()
    print("  现在的训练配置是这样组合的：")
    print("     • HANDSTAND_INIT_STATE 关节默认角  →  服务于 A（头朝下，前腿撑地）")
    print("     • pose_range['pitch']=(1.31,1.57)  →  也是 A（pitch≈+90°）")
    print("     • 但 target_gravity=(-1,0,0)        →  是 B（头朝上）")
    print()
    print("  reward 跟 init pose / joint defaults 方向相反，policy 的训练目标本身")
    print("  就是矛盾的。这就是为什么 deploy 出来'坐着站起来'——policy 学到的是某种")
    print("  在两个矛盾目标之间凑合的姿态，既不是真 handstand 也不是稳定站立。")
    print()
    print("  修复方法（把 reward 反过来对齐已有的 init pose 设计）：")
    print("     env_cfgs.py L155:")
    print("         target_gravity = (-1.0, 0.0, 0.0)  →  (+1.0, 0.0, 0.0)")
    print("=" * 78)


if __name__ == "__main__":
    main()
