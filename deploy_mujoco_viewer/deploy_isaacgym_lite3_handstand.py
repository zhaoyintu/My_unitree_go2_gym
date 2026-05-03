"""Standalone MuJoCo deployment for the IsaacGym-trained Lite3 handstand policy.

Loads an rsl_rl checkpoint (.pt) trained with `--task=lite3_handstand` and
runs it in plain MuJoCo.  Bypasses the standard `play.py` JIT-export step —
the script reads `model_state_dict` directly and rebuilds the actor MLP
in-place.

Differences vs `deploy_mjlab_lite3_handstand.py`:
  * Checkpoint format: rsl_rl `model_state_dict` (Actor + Critic + std)
    instead of mjlab's `actor_state_dict` with embedded EmpiricalNormalization.
  * No empirical normalizer.  IsaacGym applies fixed `obs_scales` from the
    config; the same scales are reproduced here.
  * No frame stack.  IsaacGym Lite3 uses single-frame 48-dim obs (mjlab uses
    450-dim 10-frame stack after removing actor base linear velocity).
  * Obs layout matches `Go2_legstand.compute_observations` —
    zeros(2) + stand_command(1) + ang_vel*scale + projected_gravity +
    commands*scales + (dof_pos - default)*scale + dof_vel*scale + last_action.
  * PD: Kp=30 / Kd=1 (matches `Lite3Cfg_Leggedstand.control` from
    lite3_push_recovery), not the 40/1 used by the mjlab task.

Required Python packages: torch, mujoco, numpy.

Example:
    python deploy_mujoco_viewer/deploy_isaacgym_lite3_handstand.py \
        --policy /path/to/logs/lite3_handstand/model_800.pt
"""
import argparse
import time
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

# PD gains: matches Lite3Cfg_Leggedstand.control (Kp=30, Kd=1) — that's the
# `lite3_push_recovery` value, kept for sim-to-real consistency on the Lite3
# deploy stack.  (Distinct from mjlab Lite3 handstand which uses Kp=40.)
KP = np.full(12, 30.0)
KV = np.full(12, 1.0)

# Effort limits (Nm) — Lite3 motor ctrlrange is ±30 N for all joints.
EFFORT_LIMITS = np.full(12, 30.0)

# Joint armature.  Training DR samples `joint_armature_range=[0.005, 0.015]`
# uniformly per-episode (cfg.domain_rand), so the policy has seen joints
# with effective inertia in that range.  We pick the centre of the range
# (0.01) for deploy — using the rotor_inertia*gear^2 derivation puts HipX/HipY
# at 0.004, BELOW the training min, which makes those joints more responsive
# than what the policy expects and contributes to overshoot in the handstand
# rollout.  Constant across joints to match training (DR samples per-joint
# but the variation is small inside [0.005, 0.015]).
ARMATURE = np.full(12, 0.01)

# Action scale — per-joint (HipX uses smaller scale per deploy_lite3_recovery
# config and lite3_constants.py::LITE3_HANDSTAND_ACTION_SCALE).
ACTION_SCALE = np.array([
    0.125, 0.25, 0.25,   # FL: HipX, HipY, Knee
    0.125, 0.25, 0.25,   # FR
    0.125, 0.25, 0.25,   # HL
    0.125, 0.25, 0.25,   # HR
], dtype=np.float64)

# Observation layout matches `Go2_legstand.compute_observations`:
#   zeros(2)              [0:2]
#   stand_command(1)      [2]
#   base_ang_vel * 0.25   [3:6]
#   projected_gravity     [6:9]   (already a unit vector, no scale)
#   commands[:3] * scale  [9:12]  scale = [lin_vel=2, lin_vel=2, ang_vel=0.25]
#   (q - q_default) * 1.0 [12:24]
#   dq * 0.05             [24:36]
#   last_action           [36:48]
NUM_OBS = 48          # frame_stack=1 in our IsaacGym cfg
# Observation scales — must match `Lite3Cfg_Leggedstand.normalization.obs_scales`.
OBS_SCALE_LIN_VEL = 2.0     # used for the lin_vel cmd channels
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05

# Sim config.  Training uses sim.dt=0.005 + decimation=4 → CTRL_DT=0.020s
# (50 Hz control loop).  We use a 5× finer sim.dt for higher physics
# fidelity, but compensate with decimation=20 so the policy still ticks at
# 50 Hz — matching training.  Any other combination breaks the deploy:
# a finer-than-training control loop drives policy errors at 5× speed and
# can NaN out an undertrained checkpoint within < 1 s.
SIM_DT = 0.001
DECIMATION = 20
CTRL_DT = SIM_DT * DECIMATION   # = 0.020 s = 50 Hz, matches training

# Initial pose: matches HANDSTAND_INIT_STATE.pos in lite3_constants.py.
# Lite3 starts upright on four paws (TORSO body is anchored at z=0.30 in
# the MJCF; init z=0.30 matches that).
INIT_BASE_Z = 0.30
INIT_BASE_PITCH = 0.0


# ---------------------------------------------------------------------------
# Actor: MLP from rsl_rl ActorCritic.actor (IsaacGym checkpoint format)
# ---------------------------------------------------------------------------
# The IsaacGym `ActorCritic` builds its actor as
# `nn.Sequential(Linear, ELU, Linear, ELU, Linear, ELU, Linear)`, so layers
# 0/2/4/6 are Linear, 1/3/5 are ELU.  State-dict keys for the actor look like
# `actor.0.weight`, `actor.0.bias`, ..., `actor.6.weight`, `actor.6.bias`
# (interleaved with non-parameterized activations).
#
# Unlike mjlab, IsaacGym does NOT keep an empirical observation normalizer in
# the checkpoint — the env applies fixed `obs_scales` directly inside
# `compute_observations`, so we replicate that pre-scaling at deploy time.
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
        self.actor = nn.Sequential(*layers)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in ckpt:
            raise KeyError(
                f"checkpoint at {path} missing 'model_state_dict'; got keys "
                f"{list(ckpt.keys())[:6]}.  This script expects an IsaacGym "
                f"rsl_rl ActorCritic checkpoint — for mjlab checkpoints use "
                f"deploy_mjlab_lite3_handstand.py instead."
            )
        sd = ckpt["model_state_dict"]
        actor_sd = {k[len("actor."):]: v for k, v in sd.items()
                    if k.startswith("actor.")}
        if not actor_sd:
            raise KeyError("no `actor.*` keys in model_state_dict")
        self.actor.load_state_dict(actor_sd)
        self.eval()

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(obs_np).float().unsqueeze(0)
        return self.actor(x).squeeze(0).numpy()


def _check_finite(name: str, arr: np.ndarray, **context) -> None:
    """Halt with a useful traceback if `arr` has NaN/Inf.

    NaN cascades silently in MuJoCo (you only get a generic CTRL warning a
    fraction of a second after the actual blow-up), so we trap it at the
    earliest point — right after the policy outputs an action or before
    we hand it to the simulator.  The dump tells you which slot of which
    quantity went bad and what the surrounding state looked like, so you
    can tell whether the network blew up on its own or was fed an OOD obs.
    """
    bad = ~np.isfinite(arr)
    if not bad.any():
        return
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    msg = [f"\n[deploy] non-finite {name}; halting deploy."]
    msg.append(f"  bad indices: {np.where(bad)[0].tolist()}")
    msg.append(f"  values:      {arr}")
    for k, v in context.items():
        msg.append(f"  {k}: {v}")
    raise FloatingPointError("\n".join(msg))


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
        # IMPORTANT: do NOT clamp ctrl to joint range.  IsaacGym's
        # `_compute_torques` uses the *unclamped* target setpoint (which
        # can be far outside the joint limit when the policy outputs
        # large actions), produces a large nominal torque, and then clips
        # the torque to the effort limit.  Setting ctrllimited=True here
        # would cause MuJoCo to clamp ctrl to joint range first, giving a
        # much smaller torque that doesn't match training.  Instead leave
        # ctrl unclamped and rely on forcelimited+forcerange below to
        # saturate the torque.  Empirically the trained Lite3 handstand
        # policy outputs raw action magnitudes up to ~110 in steady state,
        # so this divergence matters a lot.
        a.inheritrange = 0.0
        a.ctrllimited = False
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
    parser.add_argument("--trace", type=int, default=0, metavar="N",
                        help="Print obs/action/state for the first N control steps and exit. "
                             "Compares each value with what the policy was trained against.")
    parser.add_argument("--null-policy", action="store_true",
                        help="Replace the policy with a constant-zero output (target_q == default). "
                             "Useful for testing whether MuJoCo physics is stable at init pose without "
                             "any policy intervention.")
    parser.add_argument("--summary", type=int, default=0, metavar="N",
                        help="One-line-per-step summary trace (base_z, grav_b, max|a|, mean|dq|).")
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

    # IsaacGym Lite3 has frame_stack=1 — no observation history kept.

    def build_obs() -> np.ndarray:
        """48-dim observation matching `Go2_legstand.compute_observations`.

        IsaacGym applies fixed obs_scales inside the env; here we pre-scale
        the raw sensor values the same way so the policy sees the same
        distribution it saw during training.
        """
        ang_vel = data.sensor("imu_ang_vel").data
        quat = data.qpos[3:7]                          # MuJoCo (w, x, y, z)
        grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        qpos, qvel = joint_state(model, data)

        scaled_cmd = np.array([
            cmd[0] * OBS_SCALE_LIN_VEL,
            cmd[1] * OBS_SCALE_LIN_VEL,
            cmd[2] * OBS_SCALE_ANG_VEL,
        ], dtype=np.float32)

        obs = np.concatenate([
            np.zeros(2, dtype=np.float32),                          # zeros(2)
            np.zeros(1, dtype=np.float32),                          # stand_command (=0 by default)
            (ang_vel * OBS_SCALE_ANG_VEL).astype(np.float32),       # 3
            grav.astype(np.float32),                                # 3
            scaled_cmd,                                             # 3
            ((qpos - DEFAULT_JOINT_POS) * OBS_SCALE_DOF_POS).astype(np.float32),  # 12
            (qvel * OBS_SCALE_DOF_VEL).astype(np.float32),          # 12
            last_action.astype(np.float32),                         # 12
        ], axis=0)
        assert obs.shape == (NUM_OBS,), f"obs shape {obs.shape} != ({NUM_OBS},)"
        return obs

    n_steps = int(args.duration / CTRL_DT)

    def step_once():
        nonlocal last_action
        obs = build_obs()
        _check_finite("obs", obs, t=data.time, base_z=data.qpos[2])
        if args.null_policy:
            action = np.zeros(12, dtype=np.float64)
        else:
            action = actor.act(obs)
        _check_finite("action", action, t=data.time, last_action=last_action)
        target_q = action * ACTION_SCALE + DEFAULT_JOINT_POS
        _check_finite("target_q", target_q, t=data.time, action=action)
        data.ctrl[act_ids] = target_q
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        last_action = action.astype(np.float64)

    # ---- one-line summary trace: base_z, grav_b, max|a|, mean|dq| ----
    if args.summary > 0:
        print(f"\n=== MUJOCO SUMMARY first {args.summary} steps ===")
        print(f"  step  base_z  grav_b_x  grav_b_z  max|a|     mean|dq|")
        for i in range(args.summary):
            obs = build_obs()
            _check_finite("obs", obs, step=i, t=data.time)
            if args.null_policy:
                action = np.zeros(12)
            else:
                action = actor.act(obs)
            _check_finite("action", action, step=i, t=data.time)
            target_q = action * ACTION_SCALE + DEFAULT_JOINT_POS
            _check_finite("target_q", target_q, step=i, action=action)
            data.ctrl[act_ids] = target_q
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)
            last_action = action.astype(np.float64)
            quat = data.qpos[3:7]
            grav_b = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
            qpos, qvel = joint_state(model, data)
            print(f"  {i:4d}  {data.qpos[2]:6.3f}  {grav_b[0]:+.3f}    "
                  f"{grav_b[2]:+.3f}    {np.abs(action).max():6.2f}    "
                  f"{np.abs(qvel).mean():6.2f}")
        return

    # ---- diagnostic trace mode: dump obs/action/state for first N steps ----
    if args.trace > 0:
        np.set_printoptions(precision=4, suppress=True, linewidth=160)
        print(f"\n=== TRACE first {args.trace} control steps ===")
        print(f"SIM_DT={SIM_DT}  DECIMATION={DECIMATION}  CTRL_DT={CTRL_DT}")
        print(f"KP={KP[0]}  KV={KV[0]}  ACTION_SCALE[0:3]={ACTION_SCALE[:3]}")
        print(f"DEFAULT_JOINT_POS={DEFAULT_JOINT_POS}")
        for i in range(args.trace):
            obs = build_obs()
            ang_vel = data.sensor("imu_ang_vel").data.copy()
            quat = data.qpos[3:7].copy()
            grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
            qpos, qvel = joint_state(model, data)
            print(f"\n--- step {i}  t={data.time:.4f}s  base_z={data.qpos[2]:.4f}m"
                  f"  base_quat(wxyz)={quat} ---")
            print(f"  obs[0:3]   zeros + stand_cmd          : {obs[0:3]}")
            print(f"  obs[3:6]   ang_vel * 0.25  (raw={ang_vel}) : {obs[3:6]}")
            print(f"  obs[6:9]   projected_gravity (body)   : {obs[6:9]}")
            print(f"  obs[9:12]  scaled_cmd                  : {obs[9:12]}")
            print(f"  obs[12:24] (q-default)*1.0  raw_q={qpos}")
            print(f"             scaled                       : {obs[12:24]}")
            print(f"  obs[24:36] dq*0.05  raw_dq={qvel}")
            print(f"             scaled                       : {obs[24:36]}")
            print(f"  obs[36:48] last_action                  : {obs[36:48]}")
            if args.null_policy:
                action = np.zeros(12)
            else:
                action = actor.act(obs)
            target_q = action * ACTION_SCALE + DEFAULT_JOINT_POS
            print(f"  raw action (policy out, no scale)      : {action}")
            print(f"  target_q   = action*scale + default    : {target_q}")
            data.ctrl[act_ids] = target_q
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)
            last_action = action.astype(np.float64)
        print(f"\n=== TRACE done.  Final base_z={data.qpos[2]:.4f}m ===")
        return

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
