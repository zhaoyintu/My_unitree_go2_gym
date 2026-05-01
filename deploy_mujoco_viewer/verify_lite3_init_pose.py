"""Verify Lite3 HANDSTAND_INIT_STATE settles into a stable four-paw stand.

If the dog won't even stand without policy intervention, training is
fighting reset perturbations from frame 1 — that's a strong candidate
explanation for `alive ≈ 0.03` after 800 iter.

Drops the dog at HANDSTAND_INIT_STATE.pos with HANDSTAND_INIT_STATE.joint_pos,
holds the joints with the same Kp=40 / Kd=1 PD as training, and reports
trunk z + foot heights every 0.5 s.

Run:
    python deploy_mujoco_viewer/verify_lite3_init_pose.py            # GUI
    python deploy_mujoco_viewer/verify_lite3_init_pose.py --no-viewer
"""
import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parents[1]
LITE3_XML = REPO_ROOT / "go2_mjlab" / "robots" / "xmls" / "lite3.xml"

JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]
# go2_mjlab/robots/lite3_constants.py::HANDSTAND_INIT_STATE
INIT_JOINT_POS = np.array([
    +0.1, -0.8, 1.6,   # FL  L_HipX = +0.1
    -0.1, -0.8, 1.6,   # FR  R_HipX = -0.1
    +0.1, -0.8, 1.6,   # HL
    -0.1, -0.8, 1.6,   # HR
])
INIT_BASE_Z = 0.30      # HANDSTAND_INIT_STATE.pos.z
KP = 40.0
KV = 1.0
EFFORT = 30.0


def build_model():
    spec = mujoco.MjSpec.from_file(str(LITE3_XML))
    spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=300, height=300,
        rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3],
    )
    spec.add_material(name="grid", textures=["", "grid"],
                      texrepeat=[5, 5], reflectance=0.2)
    spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05], material="grid",
        contype=1, conaffinity=1, condim=3,
    )
    spec.worldbody.add_light(
        name="top", pos=[0, 0, 3], dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    # Convert <motor> to PD position actuators.
    name_to_idx = {a.name: i for i, a in enumerate(spec.actuators)}
    for jn in JOINT_NAMES:
        a = spec.actuators[name_to_idx[jn.removesuffix("_joint")]]
        a.dyntype = mujoco.mjtDyn.mjDYN_NONE
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.inheritrange = 1.0
        a.ctrllimited = True
        a.gainprm[0] = KP
        a.biasprm[1] = -KP
        a.biasprm[2] = -KV
        a.forcelimited = True
        a.forcerange[:] = [-EFFORT, EFFORT]
    return spec.compile()


def reset_init(model, data):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, INIT_BASE_Z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for jn, q in zip(JOINT_NAMES, INIT_JOINT_POS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def report(model, data, label):
    quat = data.qpos[3:7]
    w, x, y, z = quat
    qv = np.array([x, y, z])
    g = np.array([0.0, 0.0, -1.0])
    grav_b = g + np.cross(qv, 2.0 * np.cross(qv, g)) - w * 2.0 * np.cross(qv, g)
    print(f"\n{label}")
    print(f"  trunk z = {data.qpos[2]:.4f} m")
    print(f"  projected_gravity (body) = "
          f"({grav_b[0]:+.3f}, {grav_b[1]:+.3f}, {grav_b[2]:+.3f})")
    for fn in ("FL", "FR", "HL", "HR"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, fn)
        x, y, z = data.site_xpos[sid]
        print(f"  {fn} foot: x={x:+.3f}  y={y:+.3f}  z={z:+.4f}")
    # Joint deviation from INIT_JOINT_POS — should stay near zero under PD.
    actual = np.array([
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]]
        for jn in JOINT_NAMES
    ])
    dev = actual - INIT_JOINT_POS
    print(f"  joint deviation L1 = {np.abs(dev).sum():.4f}  max = {np.abs(dev).max():.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--duration", type=float, default=4.0)
    args = p.parse_args()

    model = build_model()
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    reset_init(model, data)
    report(model, data, "[t=0] INITIAL POSE (just set, no physics yet):")

    # Hold init pose with PD.
    act_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn.removesuffix("_joint"))
        for jn in JOINT_NAMES
    ])
    data.ctrl[act_ids] = INIT_JOINT_POS

    if args.no_viewer:
        n = int(args.duration / 0.005)
        for i in range(n):
            mujoco.mj_step(model, data)
            if (i + 1) % 100 == 0:  # every 0.5 s
                report(model, data, f"[t={data.time:.2f}s]")
        return

    print("\nLaunching viewer.  SPACE pause/resume,  R reset,  close window to exit.")
    paused = [False]

    def key_callback(keycode):
        if keycode == 32:
            paused[0] = not paused[0]
            print(f"[viewer] {'paused' if paused[0] else 'resumed'}")
        elif keycode in (82, ord('r'), ord('R')):
            reset_init(model, data)
            data.ctrl[act_ids] = INIT_JOINT_POS
            print("[viewer] reset")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as v:
        v.cam.distance = 1.5
        v.cam.azimuth = 90.0
        v.cam.elevation = -10.0
        v.cam.lookat[:] = [0.0, 0.0, 0.2]

        n = int(args.duration / 0.005)
        i = 0
        last_report = -1.0
        while i < n and v.is_running():
            t0 = time.time()
            if not paused[0]:
                mujoco.mj_step(model, data)
                if data.time - last_report > 0.5:
                    report(model, data, f"[t={data.time:.2f}s]")
                    last_report = data.time
                i += 1
            v.sync()
            dt_left = 0.005 - (time.time() - t0)
            if dt_left > 0:
                time.sleep(dt_left)


if __name__ == "__main__":
    main()
