"""Standalone MuJoCo deployment for the mjlab-trained Go2 handstand policy.

Loads an rsl_rl checkpoint (.pt) trained with go2_mjlab/Mjlab-Go2-Handstand
and runs it in plain MuJoCo. No mjlab dependency.

Required Python packages: torch, mujoco, numpy.

Example:
    python deploy_mujoco_viewer/deploy_mjlab_handstand.py \
        --policy /mnt/d/go_mjlab_handstand/model_14999.pt
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
# Config: must match training (go2_mjlab/config/go2_handstand)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
GO2_XML = REPO_ROOT / "go2_mjlab" / "robots" / "xmls" / "go2.xml"

# Joint definition order in the MJCF (FR, FL, RL, RR — each: hip, thigh, calf).
JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# HANDSTAND_INIT_STATE.joint_pos resolved per joint name (this is what
# joint_pos_rel subtracts and what target_q is offset by).
DEFAULT_JOINT_POS = np.array([
    0.0, -1.00, -1.10,   # FR
    0.0, -1.00, -1.10,   # FL
    0.0,  2.25, -2.00,   # RL
    0.0,  2.25, -2.00,   # RR
], dtype=np.float64)

# PD gains (handstand uses Kp=40, Kd=1.0 for all 12 joints).
KP = np.full(12, 40.0)
KV = np.full(12, 1.0)

# Effort limits (Nm).
EFFORT_LIMITS = np.array([23.7, 23.7, 35.55] * 4)

# Reflected rotor inertia: rotor_inertia * gear_ratio^2.
ROTOR_INERTIA = 0.000111842
HIP_GEAR, KNEE_GEAR = 6.0, 9.0
ARMATURE = np.array([
    ROTOR_INERTIA * HIP_GEAR ** 2,    # hip
    ROTOR_INERTIA * HIP_GEAR ** 2,    # thigh
    ROTOR_INERTIA * KNEE_GEAR ** 2,   # calf
] * 4)

# Action / observation config (from go2_handstand env_cfgs.py).
ACTION_SCALE = 0.25
NUM_SINGLE_OBS = 48          # 3 + 3 + 3 + 3 + 12 + 12 + 12
FRAME_STACK = 10
NUM_OBS = NUM_SINGLE_OBS * FRAME_STACK   # 480

# Sim config: timestep=0.005, decimation=4 → 50 Hz control.
SIM_DT = 0.001
DECIMATION = 4
CTRL_DT = SIM_DT * DECIMATION

# Initial pose: handstand inverted on front legs (mid of training reset range).
INIT_BASE_Z = 0.40
INIT_BASE_PITCH = 1.4   # ≈ 80°


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
# Build mujoco model: go2.xml + ground + position actuators (Kp/Kd in spec)
# ---------------------------------------------------------------------------
def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(GO2_XML))

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

    # Joint armature + per-joint <position> actuator (matches mjlab BuiltinPositionActuator).
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
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str,
                        default="/mnt/d/go_mjlab_handstand/model_14999.pt",
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
    args = parser.parse_args()

    actor = Actor()
    actor.load(args.policy)
    print(f"[deploy] loaded policy: {args.policy}")
    print(f"[deploy] cmd = {args.cmd}, dt = {CTRL_DT * 1000:.1f} ms")

    model = build_model()
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    reset_state(model, data)

    # Cache actuator ids in JOINT_NAMES order.
    act_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
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
        # For each term, flatten chronological history (oldest→newest), then concat.
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

    if args.no_viewer:
        for i in range(n_steps):
            step_once()
            if (i + 1) % int(1.0 / CTRL_DT) == 0:
                quat = data.qpos[3:7]
                grav_b = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
                print(f"  t={data.time:5.2f}s  z={data.qpos[2]:.3f}m  "
                      f"grav_b_x={grav_b[0]:+.2f} (target -1.0)")
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
