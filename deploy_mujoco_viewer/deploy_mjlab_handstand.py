"""Standalone MuJoCo deployment for the mjlab-trained Go2 handstand policy.

Loads an rsl_rl checkpoint (.pt) trained with go2_mjlab/Mjlab-Go2-Handstand
and runs it in plain MuJoCo. No mjlab dependency.

Required Python packages: torch, mujoco, numpy.

Example:
    python deploy_mujoco_viewer/deploy_mjlab_handstand.py \
        --policy /mnt/d/go_mjlab_handstand/model_14999.pt

    # Drive vx / yaw live from a gamepad (needs `pip install pygame`):
    python deploy_mujoco_viewer/deploy_mjlab_handstand.py \
        --policy model_14999.pt --gamepad

    # First time on a new controller, find which axes the sticks map to:
    python deploy_mujoco_viewer/deploy_mjlab_handstand.py --gamepad --gamepad-test
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
# joint_pos_rel subtracts and what target_q is offset by). Must match
# go2_constants.py::HANDSTAND_INIT_STATE — IsaacGym GO2_Leggedstand init
# (normal four-paw stand; the policy itself learns to flip head-down).
DEFAULT_JOINT_POS = np.array([
    +0.1, 0.8, -1.5,   # FR  hip=+0.1 (R-side)
    -0.1, 0.8, -1.5,   # FL  hip=-0.1 (L-side)
    -0.1, 1.0, -1.5,   # RL  hip=-0.1 (L-side)
    +0.1, 1.0, -1.5,   # RR  hip=+0.1 (R-side)
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
SIM_DT = 0.005
DECIMATION = 4
CTRL_DT = SIM_DT * DECIMATION

# Initial pose: matches HANDSTAND_INIT_STATE.pos in go2_constants.py.
# Dog starts upright on four paws; the policy learns to flip itself.
INIT_BASE_Z = 0.42
INIT_BASE_PITCH = 0.0

# DIAGNOSTIC (--start-handstand): spawn ALREADY inverted in the front-paw
# handstand pose — body +x → world -z (so projected-gravity x ≈ +1, the trained
# target) at the env's desire joint angles. This tests whether the policy can
# HOLD a handstand (vs flip up from the upright init). DEFAULT_JOINT_POS — the
# obs/action offset — is left UNCHANGED so the observation frame still matches
# training; only the physical spawn state changes.
HANDSTAND_START_Z = 0.47          # front paws ≈ on ground, rear feet ≈ 0.67 (target)
HANDSTAND_START_PITCH = 1.5708   # +π/2 ≈ nose straight down
HANDSTAND_START_JOINT_POS = np.array([
    0.0, -0.7, -1.75,   # FR  front / stance (reaches forward-down to support)
    0.0, -0.7, -1.75,   # FL
    0.0,  0.8, -1.5,    # RL  rear / swing (rest pose)
    0.0,  0.8, -1.5,    # RR
], dtype=np.float64)


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
        if "0.weight" in mlp_sd:
            checkpoint_obs_dim = int(mlp_sd["0.weight"].shape[-1])
            expected_obs_dim = int(self.mlp[0].in_features)
            if checkpoint_obs_dim != expected_obs_dim:
                hint = ""
                if checkpoint_obs_dim == 450:
                    hint = (
                        " This checkpoint matches the 450-dim "
                        "Mjlab-Lite3-Handstand actor; use "
                        "deploy_mujoco_viewer/deploy_mjlab_lite3_handstand.py."
                    )
                raise RuntimeError(
                    f"Checkpoint actor obs dim {checkpoint_obs_dim} does not match "
                    f"this Go2 deploy script's {expected_obs_dim}-dim actor for "
                    f"Mjlab-Go2-Handstand.{hint}"
                )
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


def reset_state(model: mujoco.MjModel, data: mujoco.MjData,
                handstand: bool = False) -> None:
    mujoco.mj_resetData(model, data)
    if handstand:
        z, p, init_jpos = (HANDSTAND_START_Z, HANDSTAND_START_PITCH,
                           HANDSTAND_START_JOINT_POS)
    else:
        z, p, init_jpos = INIT_BASE_Z, INIT_BASE_PITCH, DEFAULT_JOINT_POS
    data.qpos[0:3] = [0.0, 0.0, z]
    # Rotation about world y by `p`: q = (cos(p/2), 0, sin(p/2), 0).
    data.qpos[3:7] = [np.cos(p / 2), 0.0, np.sin(p / 2), 0.0]
    for jn, q in zip(JOINT_NAMES, init_jpos):
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
# Optional gamepad / joystick teleop
# ---------------------------------------------------------------------------
class Gamepad:
    """Reads a gamepad via pygame, mirroring go2_rl_gym deploy_mujoco's
    `get_xbox_command` exactly:

        lx = axis(0)   ly = axis(1)   rx = axis(3)      # left X/Y, right X
        # hard deadzone 0.1 (value zeroed below it, NOT rescaled)
        cmd_x   = -ly * vx_max        # left stick  up    → forward
        cmd_y   = -lx * vy_max        # left stick  left  → +y (strafe)
        cmd_yaw = -rx * yaw_max       # right stick left  → +yaw (turn left)

    For the Go2 HANDSTAND the policy was trained with lin_vel_y = 0, so vy_max
    defaults to 0 (left-stick X disabled); vx_max / yaw_max default to the
    training range ±0.4. Axis indices vary by controller — use --gamepad-test
    to confirm. Each sign is flippable (--invert-*) because a head-down robot
    can feel mirrored to the operator.
    """

    def __init__(self, args):
        import os
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame  # lazy import: only required when --gamepad is used
        self._pg = pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No gamepad detected. Plug one in (or pair via Bluetooth) and "
                "check `ls /dev/input/js*`. (pygame found 0 joysticks.)"
            )
        self.js = pygame.joystick.Joystick(args.gamepad_id)
        self.js.init()
        self.axis_lx = args.axis_lx          # left stick X  → vy
        self.axis_ly = args.axis_ly          # left stick Y  → vx
        self.axis_rx = args.axis_rx          # right stick X → yaw
        self.vx_max = args.vx_max
        self.vy_max = args.vy_max
        self.yaw_max = args.yaw_max
        self.dz = args.gamepad_deadzone
        # default sign = -1 reproduces the reference's `-ly / -lx / -rx`.
        self.m_vx = 1.0 if args.invert_vx else -1.0
        self.m_vy = 1.0 if args.invert_vy else -1.0
        self.m_yaw = 1.0 if args.invert_yaw else -1.0
        self.reset_button = args.reset_button
        n_ax, n_btn = self.js.get_numaxes(), self.js.get_numbuttons()
        print(f"[gamepad] '{self.js.get_name()}'  ({n_ax} axes, {n_btn} buttons)")
        print(f"[gamepad] LY(axis{self.axis_ly})→vx[±{self.vx_max}]  "
              f"LX(axis{self.axis_lx})→vy[±{self.vy_max}]  "
              f"RX(axis{self.axis_rx})→yaw[±{self.yaw_max}]  deadzone={self.dz}")
        if max(self.axis_lx, self.axis_ly, self.axis_rx) >= n_ax:
            raise RuntimeError(
                f"axis index out of range; this pad has only {n_ax} axes. "
                f"Run --gamepad-test to find the right ones."
            )

    def read(self, cmd: np.ndarray) -> None:
        """Poll the pad and write [vx, vy, yaw] into cmd (in place)."""
        self._pg.event.pump()
        if self.reset_button >= 0 and self.js.get_button(self.reset_button):
            cmd[:] = 0.0
            return
        dz = self.dz
        lx = self.js.get_axis(self.axis_lx)
        ly = self.js.get_axis(self.axis_ly)
        rx = self.js.get_axis(self.axis_rx)
        if abs(lx) < dz: lx = 0.0
        if abs(ly) < dz: ly = 0.0
        if abs(rx) < dz: rx = 0.0
        cmd[0] = self.m_vx * ly * self.vx_max     # default -ly*vx_max
        cmd[1] = self.m_vy * lx * self.vy_max     # default -lx*vy_max (vy_max=0 ⇒ 0)
        cmd[2] = self.m_yaw * rx * self.yaw_max   # default -rx*yaw_max

    def debug_loop(self) -> None:
        """Print every axis/button live so the user can identify stick mappings."""
        print("[gamepad-test] move the sticks/buttons; Ctrl-C to quit.")
        try:
            while True:
                self._pg.event.pump()
                axes = "  ".join(
                    f"a{i}={self.js.get_axis(i):+.2f}"
                    for i in range(self.js.get_numaxes())
                )
                btns = [i for i in range(self.js.get_numbuttons())
                        if self.js.get_button(i)]
                print(f"  {axes}   btn={btns}        ", end="\r", flush=True)
                self._pg.time.wait(50)
        except KeyboardInterrupt:
            print("\n[gamepad-test] done.")


# ---------------------------------------------------------------------------
# Offscreen video recording (no GUI viewer)
# ---------------------------------------------------------------------------
def record_video(model, data, step_once, n_steps: int, args) -> None:
    """Run the rollout headless and write each control step as a frame to MP4.

    Uses `mujoco.Renderer` for offscreen rendering and `cv2.VideoWriter`
    (libopencv) for MP4 encoding — no extra binary deps beyond what's
    already in the conda env.
    """
    import cv2

    out_path = Path(args.video).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(model, width=args.video_width, height=args.video_height)

    if args.video_camera:
        camera = args.video_camera   # named MJCF camera
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

    # cv2 mp4v fourcc + RGB→BGR conversion is the cheapest cross-platform path.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path), fourcc, float(args.video_fps),
        (args.video_width, args.video_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open MP4 writer for {out_path}")

    # Capture one frame every 1/fps seconds of sim time so the recorded
    # video plays back in real time, regardless of the inner loop rate.
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
                frame_rgb = renderer.render()  # (H, W, 3) uint8
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
    parser.add_argument("--start-handstand", action="store_true",
                        help="DIAGNOSTIC: spawn already inverted in the handstand pose "
                             "to test whether the policy can HOLD it (vs flip from upright)")
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
    # Gamepad teleop (optional; needs `pip install pygame`).
    g = parser.add_argument_group("gamepad")
    g.add_argument("--gamepad", action="store_true",
                   help="Drive vx/yaw live from a gamepad each step (overrides --cmd)")
    g.add_argument("--gamepad-test", action="store_true",
                   help="Just print live axis/button values to find stick mappings, then exit")
    g.add_argument("--gamepad-id", type=int, default=0, help="pygame joystick index")
    # Axis mapping matches go2_rl_gym deploy_mujoco get_xbox_command (lx=0, ly=1, rx=3).
    g.add_argument("--axis-lx", type=int, default=0, help="left stick X axis  → vy")
    g.add_argument("--axis-ly", type=int, default=1, help="left stick Y axis  → vx (forward)")
    g.add_argument("--axis-rx", type=int, default=3, help="right stick X axis → yaw")
    g.add_argument("--vx-max", type=float, default=0.4, help="max |vx| (training range ±0.4)")
    g.add_argument("--vy-max", type=float, default=0.0,
                   help="max |vy|; 0 = disabled (handstand trained with lin_vel_y=0)")
    g.add_argument("--yaw-max", type=float, default=0.4, help="max |yaw| (training range ±0.4)")
    g.add_argument("--gamepad-deadzone", type=float, default=0.1,
                   help="hard stick deadzone (matches reference 0.1)")
    g.add_argument("--invert-vx", action="store_true", help="flip forward/back if reversed")
    g.add_argument("--invert-vy", action="store_true", help="flip strafe direction")
    g.add_argument("--invert-yaw", action="store_true", help="flip turn direction")
    g.add_argument("--reset-button", type=int, default=-1,
                   help="gamepad button index to zero the command (-1 = disabled)")
    args = parser.parse_args()

    # Axis-discovery mode: open the pad, print values, exit (no policy needed).
    if args.gamepad_test:
        Gamepad(args).debug_loop()
        return

    actor = Actor()
    actor.load(args.policy)
    print(f"[deploy] loaded policy: {args.policy}")
    print(f"[deploy] cmd = {args.cmd}, dt = {CTRL_DT * 1000:.1f} ms")

    model = build_model()
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    reset_state(model, data, handstand=args.start_handstand)

    # Cache actuator ids in JOINT_NAMES order.
    act_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
        for jn in JOINT_NAMES
    ])

    cmd = np.asarray(args.cmd, dtype=np.float64)
    last_action = np.zeros(12, dtype=np.float64)

    pad = Gamepad(args) if args.gamepad else None
    if pad is not None:
        print("[deploy] gamepad teleop active — left stick = forward/back, "
              "right stick = turn (vy locked to 0)")

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
    status_every = max(1, int(0.5 / CTRL_DT))   # ~2 Hz live command readout
    step_count = [0]

    def step_once():
        nonlocal last_action
        if pad is not None:
            pad.read(cmd)               # live gamepad → cmd (vx, 0, yaw)
        action = actor.act(build_obs())
        target_q = action * ACTION_SCALE + DEFAULT_JOINT_POS
        data.ctrl[act_ids] = target_q
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        last_action = action.astype(np.float64)
        push_history()
        step_count[0] += 1
        if pad is not None and step_count[0] % status_every == 0:
            print(f"[cmd] vx={cmd[0]:+.2f}  yaw={cmd[2]:+.2f}   ",
                  end="\r", flush=True)

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
                      f"grav_b_x={grav_b[0]:+.2f} (target -1.0)")
        return

    print("[viewer] SPACE = pause/resume,  close window to exit")
    paused = [False]

    def key_callback(keycode):
        # Arrow keys nudge the command live (keyboard fallback when no gamepad).
        if keycode == 32:  # SPACE
            paused[0] = not paused[0]
            print(f"[viewer] {'paused' if paused[0] else 'resumed'}")
        elif keycode == 265:  # UP — forward
            cmd[0] = float(np.clip(cmd[0] + 0.1, -args.vx_max, args.vx_max))
        elif keycode == 264:  # DOWN — back
            cmd[0] = float(np.clip(cmd[0] - 0.1, -args.vx_max, args.vx_max))
        elif keycode == 263:  # LEFT — turn left (+yaw)
            cmd[2] = float(np.clip(cmd[2] + 0.1, -args.yaw_max, args.yaw_max))
        elif keycode == 262:  # RIGHT — turn right (-yaw)
            cmd[2] = float(np.clip(cmd[2] - 0.1, -args.yaw_max, args.yaw_max))
        elif keycode == 82:  # R — zero command
            cmd[:] = 0.0
        if keycode in (265, 264, 263, 262, 82):
            print(f"[cmd] vx={cmd[0]:+.2f}  yaw={cmd[2]:+.2f}")

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
