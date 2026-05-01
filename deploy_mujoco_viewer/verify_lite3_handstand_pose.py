"""Verify the front-paw handstand target pose for Lite3 in MuJoCo (no policy).

Lite3 differs from Go2 in joint conventions:
  * HipX axis = -x  (Go2 hip axis = +x)
  * HipY axis = -y  (Go2 thigh axis = +y)
  * Knee axis = -y  (Go2 calf axis = +y)
  * HipY range = [-2.67, 0.314] — the 0.314 rad (≈18°) upper limit
    means Lite3 cannot reach Go2's "thigh = -0.7" stance pose; the
    front leg's forward reach is much narrower.

Forward kinematics maximization (∂foot_x/∂HipY = ∂foot_x/∂Knee = 0)
gives the optimal extension: HipY at upper limit, HipY+Knee = π/2.
With the soft 0.9 limit factor applied that puts HipY = 0.283 and
Knee ≈ 1.288, yielding:
    - trunk_z target ≈ 0.44 m  (vs Go2 0.47)
    - rear foot world z ≈ 0.61 m (vs Go2 0.67)

Holds the pose with PD targeting the same joint defaults and prints
geometry diagnostics so we can confirm kinematics before training.

Usage:
    python deploy_mujoco_viewer/verify_lite3_handstand_pose.py
    python deploy_mujoco_viewer/verify_lite3_handstand_pose.py --no-viewer
"""
import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parents[1]
LITE3_XML = REPO_ROOT / "go2_mjlab" / "robots" / "xmls" / "lite3.xml"

# MJCF joint order: FL, FR, HL, HR (each: HipX, HipY, Knee).
JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]

# --- Default desire-pose values (overridable via CLI) ----------------------
# Front (stance): HipY pushed to 90% of upper limit (=0.283).  Knee is the
# main visual lever: 0.65 = nearly-straight leg, 1.288 = max foot forward
# reach (knee bent ~73°, shank vertical), 2.17 = deeply folded.
DEFAULT_FRONT_HIPY = 0.283
DEFAULT_FRONT_KNEE = 2.0     # ~65° interior angle — visually-natural fold

# Rear (swing) — Lite3 standing rest pose; same role as Go2's rear desire.
DEFAULT_REAR_HIPY = -0.8
DEFAULT_REAR_KNEE = 1.6

# Body vertical, head DOWN.  Pitch = +π/2 about world +y rotates body +x to
# world -z (head pointing into the floor).
INIT_PITCH = np.pi / 2

# Base height + rear-foot target are RECOMPUTED from joint angles via FK
# (see compute_fk_targets()).  These globals are overwritten in main().
INIT_BASE_Z = 0.44
TARGET_REAR_FOOT_Z = 0.61

TARGET_JOINT_POS = np.zeros(12)


def compute_fk_targets(front_hipy: float, front_knee: float,
                       rear_hipy: float, rear_knee: float
                       ) -> tuple[float, float]:
    """Return (base_z so front feet land on z=0, rear-foot world z)
    via 2-link forward kinematics with body pitched +π/2.

    Lite3 link lengths from MJCF: thigh = 0.20 m, shank = 0.21012 m.
    """
    L1, L2 = 0.20, 0.21012
    # Front foot body x (signed distance from trunk along body +x).
    front_foot_body_x = (
        0.1745
        + L1 * np.sin(front_hipy)
        + L2 * np.sin(front_hipy + front_knee)
    )
    # Rear foot body x (negative — rear shoulder is at body x = -0.1745).
    rear_foot_body_x = (
        -0.1745
        + L1 * np.sin(rear_hipy)
        + L2 * np.sin(rear_hipy + rear_knee)
    )
    # Body pitched +π/2 around world +y.  Body vector (bx, by, bz) maps
    # to world (bz, by, -bx).  Foot world z = base_z - foot_body_x.
    base_z = front_foot_body_x          # → front foot world z = 0
    rear_foot_world_z = base_z - rear_foot_body_x
    return float(base_z), float(rear_foot_world_z)

KP = np.full(12, 40.0)
KV = np.full(12, 1.0)
EFFORT = np.full(12, 30.0)


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(LITE3_XML))
    spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=300, height=300,
        rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3],
    )
    spec.add_material(
        name="grid", textures=["", "grid"],
        texrepeat=[5, 5], reflectance=0.2,
    )
    spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05], material="grid",
        contype=1, conaffinity=1, condim=3,
    )
    spec.worldbody.add_light(
        name="top", pos=[0, 0, 3], dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    # The lite3.xml ships bare <motor> actuators (one per joint, in
    # JOINT_NAMES order).  Re-purpose them in place into PD position
    # actuators so we can hold the pose with a target setpoint.
    name_to_idx = {a.name: i for i, a in enumerate(spec.actuators)}
    for jn in JOINT_NAMES:
        i = JOINT_NAMES.index(jn)
        # The motor names are FL_HipX, FL_HipY, etc — strip the trailing _joint.
        a = spec.actuators[name_to_idx[jn.removesuffix("_joint")]]
        a.dyntype = mujoco.mjtDyn.mjDYN_NONE
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.inheritrange = 1.0
        a.ctrllimited = True
        a.gainprm[0] = KP[i]
        a.biasprm[1] = -KP[i]
        a.biasprm[2] = -KV[i]
        a.forcelimited = True
        a.forcerange[:] = [-EFFORT[i], EFFORT[i]]
    return spec.compile()


def quat_rotate_inverse(q_wxyz, v):
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v + np.cross(qv, t) - w * t


def set_pose(model, data):
    mujoco.mj_resetData(model, data)
    p = INIT_PITCH
    data.qpos[0:3] = [0.0, 0.0, INIT_BASE_Z]
    data.qpos[3:7] = [np.cos(p / 2), 0.0, np.sin(p / 2), 0.0]
    for jn, q in zip(JOINT_NAMES, TARGET_JOINT_POS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def report(model, data, label):
    quat = data.qpos[3:7]
    grav_b = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
    print(f"\n{label}")
    print(f"  trunk z = {data.qpos[2]:.3f} m   "
          f"(target {INIT_BASE_Z:.3f})")
    print(f"  projected_gravity (body) = "
          f"({grav_b[0]:+.3f}, {grav_b[1]:+.3f}, {grav_b[2]:+.3f})   "
          f"target for 头朝下 handstand = (+1, 0, 0)")
    for fname in ["FL", "FR", "HL", "HR"]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, fname)
        x, y, z = data.site_xpos[sid]
        kind = "前 (前腿撑地, target z≈0)" if fname.startswith("F") \
               else f"后 (后腿翘起, target z≈{TARGET_REAR_FOOT_Z:.2f})"
        print(f"  {fname} foot {kind}: x={x:+.3f}  y={y:+.3f}  z={z:+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument(
        "--dynamic", action="store_true",
        help="Step physics (default: pose is held statically with mj_forward "
             "every frame — pose locks for visual inspection).",
    )
    parser.add_argument("--front-hipy", type=float, default=DEFAULT_FRONT_HIPY,
                        help="Front-leg HipY (range [-2.67, 0.314]). "
                             f"default={DEFAULT_FRONT_HIPY:.3f}")
    parser.add_argument("--front-knee", type=float, default=DEFAULT_FRONT_KNEE,
                        help="Front-leg Knee (range [0.524, 2.792]). "
                             f"default={DEFAULT_FRONT_KNEE:.3f}")
    parser.add_argument("--rear-hipy", type=float, default=DEFAULT_REAR_HIPY,
                        help=f"Rear-leg HipY. default={DEFAULT_REAR_HIPY:.3f}")
    parser.add_argument("--rear-knee", type=float, default=DEFAULT_REAR_KNEE,
                        help=f"Rear-leg Knee. default={DEFAULT_REAR_KNEE:.3f}")
    args = parser.parse_args()

    # Build joint vector + recompute base height / rear-foot target via FK.
    global TARGET_JOINT_POS, INIT_BASE_Z, TARGET_REAR_FOOT_Z
    TARGET_JOINT_POS = np.array([
        0.0, args.front_hipy, args.front_knee,
        0.0, args.front_hipy, args.front_knee,
        0.0, args.rear_hipy,  args.rear_knee,
        0.0, args.rear_hipy,  args.rear_knee,
    ])
    INIT_BASE_Z, TARGET_REAR_FOOT_Z = compute_fk_targets(
        args.front_hipy, args.front_knee, args.rear_hipy, args.rear_knee,
    )
    print(f"Front desire: HipY={args.front_hipy:+.3f}  Knee={args.front_knee:+.3f}")
    print(f"Rear  desire: HipY={args.rear_hipy:+.3f}  Knee={args.rear_knee:+.3f}")
    print(f"FK predicts: base_z={INIT_BASE_Z:.3f}  rear_foot_z={TARGET_REAR_FOOT_Z:.3f}")

    model = build_model()
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    set_pose(model, data)
    report(model, data, "[t=0] INITIAL POSE (just set, no physics yet):")

    act_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                          jn.removesuffix("_joint"))
        for jn in JOINT_NAMES
    ])
    data.ctrl[act_ids] = TARGET_JOINT_POS

    if args.no_viewer:
        n = int(args.duration / 0.005)
        for i in range(n):
            mujoco.mj_step(model, data)
            if (i + 1) % 200 == 0:  # every 1.0 s
                report(model, data, f"[t={data.time:.2f}s]")
        return

    if args.dynamic:
        print("\nLaunching viewer — DYNAMIC mode (PD-hold under gravity).")
    else:
        print("\nLaunching viewer — STATIC mode (pose locked, no physics).")
    print("SPACE = pause toggle (dynamic only),  R = reset pose,  close window to exit.")

    paused = [False]

    def key_callback(keycode):
        if keycode == 32:        # SPACE
            paused[0] = not paused[0]
            print(f"[viewer] {'paused' if paused[0] else 'resumed'}")
        elif keycode in (82, ord('r'), ord('R')):   # R
            set_pose(model, data)
            data.ctrl[act_ids] = TARGET_JOINT_POS
            print("[viewer] reset to initial pose")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as v:
        v.cam.distance = 1.8
        v.cam.azimuth = 90.0
        v.cam.elevation = -10.0
        v.cam.lookat[:] = [0.0, 0.0, 0.3]
        if args.dynamic:
            n = int(args.duration / 0.005)
            last_report_t = -1.0
            i = 0
            while i < n and v.is_running():
                t0 = time.time()
                if not paused[0]:
                    mujoco.mj_step(model, data)
                    if data.time - last_report_t > 1.0:
                        report(model, data, f"[t={data.time:.2f}s]")
                        last_report_t = data.time
                    i += 1
                v.sync()
                dt_left = 0.005 - (time.time() - t0)
                if dt_left > 0:
                    time.sleep(dt_left)
        else:
            # Static visualization — re-apply pose every frame, no stepping.
            # Camera can be rotated freely while the pose stays locked.
            while v.is_running():
                set_pose(model, data)
                v.sync()
                time.sleep(1.0 / 60)


if __name__ == "__main__":
    main()
