"""Standalone MuJoCo deployment for the mjlab-trained Lite3 handstand policy.

Loads an rsl_rl checkpoint (.pt) trained with go2_mjlab/Mjlab-Lite3-Handstand
and runs it in plain MuJoCo. No mjlab dependency.

Required Python packages: torch, mujoco, numpy.

Example:
    python deploy_mujoco_viewer/deploy_mjlab_lite3_handstand.py \
        --policy /path/to/lite3_handstand/model_14999.pt
"""
import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import mujoco
import mujoco.viewer

# ---------------------------------------------------------------------------
# Config: must match training (go2_mjlab/config/lite3_handstand)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
LITE3_XML = REPO_ROOT / "go2_mjlab" / "robots" / "xmls" / "lite3.xml"

# Joint definition order in the MJCF: FL, FR, HL, HR (each: HipX, HipY, Knee).
# Important: this order must match what mjlab's joint_pos_rel /
# default_joint_offset / action mapping uses internally.
JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]

# HANDSTAND_INIT_STATE.joint_pos resolved per joint name (this is what
# joint_pos_rel subtracts and what target_q is offset by). Must match
# go2_mjlab/robots/lite3_constants.py::HANDSTAND_INIT_STATE — normal
# four-paw stand (the policy itself learns to flip head-down):
#   .*L_HipX_joint = +0.1, .*R_HipX_joint = -0.1
#   .*HipY_joint   = -0.8
#   .*Knee_joint   = +1.6
DEFAULT_JOINT_POS = np.array([
    +0.1, -0.8, 1.6,   # FL  HipX=+0.1 (L-side)
    -0.1, -0.8, 1.6,   # FR  HipX=-0.1 (R-side)
    +0.1, -0.8, 1.6,   # HL  HipX=+0.1 (L-side)
    -0.1, -0.8, 1.6,   # HR  HipX=-0.1 (R-side)
], dtype=np.float64)

# PD gains (Lite3 handstand uses Kp=40, Kd=1.0 — same as Go2 handstand).
KP = np.full(12, 40.0)
KV = np.full(12, 1.0)

# Effort limits (Nm) — Lite3 motor ctrlrange is ±30 N for all joints.
EFFORT_LIMITS = np.full(12, 30.0)

# Reflected rotor inertia (lite3_constants.py uses Go2 motor specs since
# Lite3 motor data isn't fully published; should be close enough).
ROTOR_INERTIA = 0.000111842
HIPX_GEAR = HIPY_GEAR = 6.0
KNEE_GEAR = 9.0
ARMATURE = np.array([
    ROTOR_INERTIA * HIPX_GEAR ** 2,    # HipX
    ROTOR_INERTIA * HIPY_GEAR ** 2,    # HipY
    ROTOR_INERTIA * KNEE_GEAR ** 2,    # Knee
] * 4)

# Action scale — per-joint (HipX uses smaller scale per deploy_lite3_recovery
# config and lite3_constants.py::LITE3_HANDSTAND_ACTION_SCALE).
ACTION_SCALE = np.array([
    0.125, 0.25, 0.25,   # FL: HipX, HipY, Knee
    0.125, 0.25, 0.25,   # FR
    0.125, 0.25, 0.25,   # HL
    0.125, 0.25, 0.25,   # HR
], dtype=np.float64)

# Observation: same layout as Go2 (lin_vel + ang_vel + grav + cmd + qpos + qvel + last_act).
NUM_SINGLE_OBS = 48          # 3 + 3 + 3 + 3 + 12 + 12 + 12
FRAME_STACK = 10
NUM_OBS = NUM_SINGLE_OBS * FRAME_STACK   # 480

# Sim config: timestep=0.005, decimation=4 → 50 Hz control (matches env_cfgs).
SIM_DT = 0.005
DECIMATION = 4
CTRL_DT = SIM_DT * DECIMATION

# Initial pose: matches HANDSTAND_INIT_STATE.pos in lite3_constants.py.
# Lite3 starts upright on four paws (TORSO body is anchored at z=0.30 in
# the MJCF; init z=0.30 matches that).
INIT_BASE_Z = 0.30
INIT_BASE_PITCH = 0.0


# ---------------------------------------------------------------------------
# Actor: MLP + EmpiricalNormalization (rsl_rl)
# ---------------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, num_obs=NUM_OBS, num_actions=12,
                 hidden_dims=(512, 256, 128)):
        super().__init__()
        layers = []
        in_dim = num_obs
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("mean", torch.zeros(1, num_obs))
        self.register_buffer("std", torch.ones(1, num_obs))
        self.eps = 1e-2

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["actor_state_dict"]
        mlp_sd = {k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")}
        self.mlp.load_state_dict(mlp_sd)
        self.mean.copy_(sd["obs_normalizer._mean"])
        self.std.copy_(sd["obs_normalizer._std"])
        self.eval()

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(obs_np).float().unsqueeze(0)
        x = (x - self.mean) / (self.std + self.eps)
        return self.mlp(x).squeeze(0).numpy()


# ---------------------------------------------------------------------------
# Build mujoco model: lite3.xml + ground + PD position actuators
# ---------------------------------------------------------------------------
def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(LITE3_XML))

    # Bigger offscreen framebuffer so `mujoco.Renderer` can produce 1080p+ frames.
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    # Skybox + ground texture/material.
    spec.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=300, height=300,
        rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3],
    )
    spec.add_material(
        name="grid", textures=["", "grid"],
        texrepeat=[5, 5], reflectance=0.2,
    )
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        material="grid",
        contype=1, conaffinity=1, condim=3,
    )
    spec.worldbody.add_light(
        name="top", pos=[0, 0, 3], dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )

    # Lite3's MJCF ships bare <motor> actuators (one per joint, named
    # FL_HipX, FL_HipY, ... — same joint name minus the "_joint" suffix).
    # Re-purpose them in place into PD position actuators so we can drive
    # joints with target setpoints (matches mjlab BuiltinPositionActuator).
    name_to_idx = {a.name: i for i, a in enumerate(spec.actuators)}
    for i, jn in enumerate(JOINT_NAMES):
        spec.joint(jn).armature = ARMATURE[i]
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
        a.forcerange[:] = [-EFFORT_LIMITS[i], EFFORT_LIMITS[i]]

    return spec.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vector v into body frame given body→world quat (w, x, y, z)."""
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v + np.cross(qv, t) - w * t


def reset_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    p = INIT_BASE_PITCH
    data.qpos[0:3] = [0.0, 0.0, INIT_BASE_Z]
    # Rotation about world y by `p`: q = (cos(p/2), 0, sin(p/2), 0).
    data.qpos[3:7] = [np.cos(p / 2), 0.0, np.sin(p / 2), 0.0]
    for jn, q in zip(JOINT_NAMES, DEFAULT_JOINT_POS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def joint_state(model, data) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.zeros(12)
    qvel = np.zeros(12)
    for i, jn in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qpos[i] = data.qpos[model.jnt_qposadr[jid]]
        qvel[i] = data.qvel[model.jnt_dofadr[jid]]
    return qpos, qvel


# ---------------------------------------------------------------------------
# Offscreen video recording (no GUI viewer)
# ---------------------------------------------------------------------------
def record_video(model, data, step_once, n_steps: int, args) -> None:
    """Run the rollout headless and write each control step as a frame to MP4."""
    import cv2

    out_path = Path(args.video).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(model, width=args.video_width, height=args.video_height)

    if args.video_camera:
        camera = args.video_camera
        update_lookat = lambda: None
    else:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 1.5
        cam.elevation = -10.0
        cam.azimuth = 90.0
        camera = cam
        def update_lookat():
            cam.lookat[:] = data.qpos[:3]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path), fourcc, float(args.video_fps),
        (args.video_width, args.video_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open MP4 writer for {out_path}")

    sim_dt_per_frame = 1.0 / args.video_fps
    next_capture = 0.0
    n_frames = 0

    print(f"[video] {out_path}  →  {args.video_width}×{args.video_height} @ {args.video_fps} fps")
    try:
        for i in range(n_steps):
            step_once()
            if data.time + 1e-9 >= next_capture:
                update_lookat()
                renderer.update_scene(data, camera=camera)
                frame_rgb = renderer.render()
                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                next_capture += sim_dt_per_frame
                n_frames += 1
            if (i + 1) % int(round(1.0 / CTRL_DT)) == 0:
                quat = data.qpos[3:7]
                grav_b = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
                print(f"  t={data.time:5.2f}s  z={data.qpos[2]:.3f}m  "
                      f"grav_b_x={grav_b[0]:+.2f}")
    finally:
        writer.release()
        renderer.close()
    print(f"[video] saved {out_path}  ({n_frames} frames, "
          f"{n_frames / args.video_fps:.1f}s)")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True,
                        help="Path to rsl_rl checkpoint .pt")
    parser.add_argument("--cmd", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        metavar=("vx", "vy", "wz"),
                        help="Constant velocity command in body frame")
    parser.add_argument("--duration", type=float, default=120.0,
                        help="Total wall-clock duration in seconds")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Run headless without launching a viewer")
    parser.add_argument("--realtime", action="store_true", default=True,
                        help="Pace the loop to wall clock")
    parser.add_argument("--video", type=str, default=None,
                        help="Write rollout to this MP4 path (forces headless)")
    parser.add_argument("--video-fps", type=int, default=50,
                        help="Video frame rate (default 50 = real-time at training control rate)")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-camera", type=str, default=None,
                        help="MJCF <camera> name to render from (default: free camera tracking trunk)")
    args = parser.parse_args()

    actor = Actor()
    actor.load(args.policy)
    print(f"[deploy] loaded policy: {args.policy}")
    print(f"[deploy] cmd = {args.cmd}, dt = {CTRL_DT * 1000:.1f} ms")

    model = build_model()
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    reset_state(model, data)

    # Cache actuator ids in JOINT_NAMES order (Lite3 actuators are named
    # without the "_joint" suffix — FL_HipX, not FL_HipX_joint).
    act_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                          jn.removesuffix("_joint"))
        for jn in JOINT_NAMES
    ])

    cmd = np.asarray(args.cmd, dtype=np.float64)
    last_action = np.zeros(12, dtype=np.float64)

    # Per-term history buffers (length=FRAME_STACK, oldest first).
    hist = {k: deque(maxlen=FRAME_STACK) for k in
            ("lin_vel", "ang_vel", "grav", "cmd",
             "joint_pos", "joint_vel", "act")}

    def make_terms():
        lin_vel = data.sensor("imu_lin_vel").data.copy()
        ang_vel = data.sensor("imu_ang_vel").data.copy()
        quat = data.qpos[3:7].copy()  # MuJoCo quat is (w, x, y, z)
        grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        qpos, qvel = joint_state(model, data)
        return [
            lin_vel.astype(np.float32),
            ang_vel.astype(np.float32),
            grav.astype(np.float32),
            cmd.astype(np.float32),
            (qpos - DEFAULT_JOINT_POS).astype(np.float32),
            qvel.astype(np.float32),
            last_action.astype(np.float32),
        ]

    # Prefill history (mjlab back-fills with first frame on first append).
    keys = list(hist.keys())
    initial = make_terms()
    for k, v in zip(keys, initial):
        for _ in range(FRAME_STACK):
            hist[k].append(v)

    def push_history():
        for k, v in zip(keys, make_terms()):
            hist[k].append(v)

    def build_obs() -> np.ndarray:
        return np.concatenate([
            np.concatenate(list(hist[k]), axis=0) for k in keys
        ], axis=0)

    n_steps = int(args.duration / CTRL_DT)

    def step_once():
        nonlocal last_action
        action = actor.act(build_obs())
        target_q = action * ACTION_SCALE + DEFAULT_JOINT_POS
        data.ctrl[act_ids] = target_q
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        last_action = action.astype(np.float64)
        push_history()

    if args.video:
        record_video(model, data, step_once, n_steps, args)
        return

    if args.no_viewer:
        for i in range(n_steps):
            step_once()
            if (i + 1) % int(1.0 / CTRL_DT) == 0:
                quat = data.qpos[3:7]
                grav_b = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
                print(f"  t={data.time:5.2f}s  z={data.qpos[2]:.3f}m  "
                      f"grav_b_x={grav_b[0]:+.2f} (target +1.0)")
        return

    print("[viewer] SPACE = pause/resume,  close window to exit")
    paused = [False]

    def key_callback(keycode):
        if keycode == 32:  # SPACE
            paused[0] = not paused[0]
            print(f"[viewer] {'paused' if paused[0] else 'resumed'}")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as v:
        v.cam.distance = 1.5
        v.cam.elevation = -10.0
        v.cam.azimuth = 90.0
        v.cam.lookat[:] = [0.0, 0.0, 0.4]

        i = 0
        while i < n_steps and v.is_running():
            t0 = time.time()
            if not paused[0]:
                step_once()
                i += 1
            v.sync()
            if args.realtime:
                dt_left = CTRL_DT - (time.time() - t0)
                if dt_left > 0:
                    time.sleep(dt_left)


if __name__ == "__main__":
    main()
