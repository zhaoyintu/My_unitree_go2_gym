"""Verify the front-paw handstand target pose in MuJoCo (no policy).

Places the Go2 in the desired pose:
  - Body vertical, HEAD DOWN (body +x aligned with world -z)
  - Front legs (FR, FL) extended down, supporting body weight on ground
  - Rear legs (RL, RR) tucked up in the air

Holds the pose with PD targeting the same joint defaults and prints geometry
diagnostics so we can confirm the kinematics before retraining anything.

Usage:
    python deploy_mujoco_viewer/verify_handstand_pose.py
    python deploy_mujoco_viewer/verify_handstand_pose.py --no-viewer
"""
import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parents[1]
GO2_XML = REPO_ROOT / "go2_mjlab" / "robots" / "xmls" / "go2.xml"

JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# Front legs reach forward-down to support body; rear legs tucked up.
# These are the same defaults already in HANDSTAND_INIT_STATE — the only
# thing that has to change for "前腿撑地" is the body orientation (pitch=+π/2,
# head DOWN), not the joint angles.
TARGET_JOINT_POS = np.array([
    0.0, -1.0, -1.1,    # FR  (front, support)
    0.0, -1.0, -1.1,    # FL  (front, support)
    0.0,  2.25, -2.0,   # RL  (rear, tucked)
    0.0,  2.25, -2.0,   # RR  (rear, tucked)
])

# Body vertical, head DOWN.  Pitch = +π/2 about world +y rotates body +x to
# world -z (head pointing into the floor).  This is what "前腿在地面，身体
# 直立" really means.
INIT_PITCH = np.pi / 2
# Body height tuned so front feet rest at z≈0 with the joint defaults above.
INIT_BASE_Z = 0.555

KP = np.full(12, 40.0)
KV = np.full(12, 1.0)
EFFORT = np.array([23.7, 23.7, 35.55] * 4)
ARMATURE = np.array([
    0.000111842 * 36, 0.000111842 * 36, 0.000111842 * 81,
] * 4)


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(GO2_XML))
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
    for i, jn in enumerate(JOINT_NAMES):
        spec.joint(jn).armature = ARMATURE[i]
        a = spec.add_actuator(name=jn, target=jn)
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
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
    print(f"  trunk z = {data.qpos[2]:.3f} m")
    print(f"  projected_gravity (body) = "
          f"({grav_b[0]:+.3f}, {grav_b[1]:+.3f}, {grav_b[2]:+.3f})   "
          f"target for 头朝下 handstand = (+1, 0, 0)")
    for fname in ["FR", "FL", "RL", "RR"]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, fname)
        x, y, z = data.site_xpos[sid]
        kind = "前 (前腿撑地)" if fname.startswith("F") else "后 (后腿翘起)"
        print(f"  {fname} foot {kind}: x={x:+.3f}  y={y:+.3f}  z={z:+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()

    model = build_model()
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    set_pose(model, data)
    report(model, data, "[t=0] INITIAL POSE (just set, no physics yet):")

    act_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
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

    print("\nLaunching viewer — observe whether the dog stands inverted on its")
    print("front paws.  SPACE = pause/resume,  R = reset pose,  close window to exit.")

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


if __name__ == "__main__":
    main()
