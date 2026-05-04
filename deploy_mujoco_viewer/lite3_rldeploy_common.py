"""Shared Lite3 RLDeploy-aligned MuJoCo deployment helpers.

The deployment contract here intentionally follows the C++/real-robot style
loop instead of the older viewer scripts:

* 1 ms MuJoCo step, 20 sim steps per policy step.
* Raw motor actuators receive PD torques, clipped to +/-30 Nm.
* Actor input is 45 values per frame, term-major stacked over 10 frames.
* Robot collision geoms match the Lite3 RLDeploy XML collision settings.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[1]
LITE3_XML = REPO_ROOT / "go2_mjlab" / "robots" / "xmls" / "lite3.xml"

JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]

DEFAULT_JOINT_POS = np.array([
    +0.1, -0.8, 1.6,
    -0.1, -0.8, 1.6,
    +0.1, -0.8, 1.6,
    -0.1, -0.8, 1.6,
], dtype=np.float64)

KP = np.full(12, 40.0, dtype=np.float64)
KD = np.full(12, 1.0, dtype=np.float64)
EFFORT_LIMITS = np.full(12, 30.0, dtype=np.float64)

ACTION_SCALE = np.array([
    0.125, 0.25, 0.25,
    0.125, 0.25, 0.25,
    0.125, 0.25, 0.25,
    0.125, 0.25, 0.25,
], dtype=np.float64)

ACTOR_TERM_NAMES = ("ang_vel", "grav", "cmd", "joint_pos", "joint_vel", "act")
TERM_DIMS = {
    "ang_vel": 3,
    "grav": 3,
    "cmd": 3,
    "joint_pos": 12,
    "joint_vel": 12,
    "act": 12,
}
NUM_SINGLE_OBS = 45
FRAME_STACK = 10
NUM_OBS = NUM_SINGLE_OBS * FRAME_STACK

SIM_DT = 0.001
DECIMATION = 20
CTRL_DT = SIM_DT * DECIMATION

INIT_BASE_Z = 0.30
INIT_BASE_PITCH = 0.0


def _set_array_field(field, values) -> None:
    for i, value in enumerate(values):
        field[i] = value


def apply_rldeploy_collision_cfg(spec: mujoco.MjSpec) -> None:
    """Apply Lite3 RLDeploy collision settings to collision geoms."""
    for geom in spec.geoms:
        if not geom.name.endswith("_collision"):
            geom.contype = 0
            geom.conaffinity = 0
            continue

        geom.contype = 0
        geom.conaffinity = 1
        geom.condim = 3
        geom.priority = 0
        _set_array_field(geom.friction, (1.0, 0.01, 0.01))
        _set_array_field(geom.solref, (0.005, 1.0))


def build_model() -> mujoco.MjModel:
    """Build the Lite3 model with RLDeploy collision and motor actuators."""
    spec = mujoco.MjSpec.from_file(str(LITE3_XML))
    apply_rldeploy_collision_cfg(spec)

    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    spec.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=300,
        height=300,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
    )
    spec.add_material(
        name="grid",
        textures=["", "grid"],
        texrepeat=[5, 5],
        reflectance=0.2,
    )
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        material="grid",
        contype=1,
        conaffinity=1,
        condim=3,
    )
    spec.worldbody.add_light(
        name="top",
        pos=[0, 0, 3],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )

    name_to_actuator = {a.name: a for a in spec.actuators}
    for i, joint_name in enumerate(JOINT_NAMES):
        joint = spec.joint(joint_name)
        joint.armature = 0.0
        actuator = name_to_actuator[joint_name.removesuffix("_joint")]
        actuator.ctrllimited = True
        actuator.ctrlrange[:] = [-EFFORT_LIMITS[i], EFFORT_LIMITS[i]]

    model = spec.compile()
    model.opt.timestep = SIM_DT
    model.opt.gravity[:] = [0.0, 0.0, -9.81]
    return model


def actuator_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = []
    for joint_name in JOINT_NAMES:
        actuator_name = joint_name.removesuffix("_joint")
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )
        if actuator_id < 0:
            raise KeyError(f"actuator {actuator_name!r} not found")
        ids.append(actuator_id)
    return np.asarray(ids, dtype=np.int32)


def reset_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    pitch = INIT_BASE_PITCH
    data.qpos[0:3] = [0.0, 0.0, INIT_BASE_Z]
    data.qpos[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
    for joint_name, q in zip(JOINT_NAMES, DEFAULT_JOINT_POS):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise KeyError(f"joint {joint_name!r} not found")
        data.qpos[model.jnt_qposadr[joint_id]] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def joint_state(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.zeros(12, dtype=np.float64)
    qvel = np.zeros(12, dtype=np.float64)
    for i, joint_name in enumerate(JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise KeyError(f"joint {joint_name!r} not found")
        qpos[i] = data.qpos[model.jnt_qposadr[joint_id]]
        qvel[i] = data.qvel[model.jnt_dofadr[joint_id]]
    return qpos, qvel


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into body frame for MuJoCo wxyz quats."""
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + np.cross(qv, t) - w * t


def policy_terms(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cmd: np.ndarray,
    last_action: np.ndarray,
) -> dict[str, np.ndarray]:
    ang_vel = data.sensor("imu_ang_vel").data.copy()
    quat = data.qpos[3:7].copy()
    grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
    qpos, qvel = joint_state(model, data)
    return {
        "ang_vel": ang_vel.astype(np.float32),
        "grav": grav.astype(np.float32),
        "cmd": cmd.astype(np.float32),
        "joint_pos": (qpos - DEFAULT_JOINT_POS).astype(np.float32),
        "joint_vel": qvel.astype(np.float32),
        "act": last_action.astype(np.float32),
    }


class TermHistory:
    """Term-major history stack matching Mjlab and rl_deploy input order."""

    def __init__(self, frame_stack: int = FRAME_STACK):
        if frame_stack < 1:
            raise ValueError(f"frame_stack must be >= 1, got {frame_stack}")
        self.frame_stack = frame_stack
        self.history = {
            name: deque(maxlen=frame_stack)
            for name in ACTOR_TERM_NAMES
        }

    def _validate(self, name: str, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        expected_shape = (TERM_DIMS[name],)
        if arr.shape != expected_shape:
            raise ValueError(f"{name} shape {arr.shape} != {expected_shape}")
        return arr

    def reset(self, terms: dict[str, np.ndarray]) -> None:
        for name in ACTOR_TERM_NAMES:
            value = self._validate(name, terms[name])
            self.history[name].clear()
            for _ in range(self.frame_stack):
                self.history[name].append(value.copy())

    def push(self, terms: dict[str, np.ndarray]) -> None:
        for name in ACTOR_TERM_NAMES:
            self.history[name].append(self._validate(name, terms[name]).copy())

    def build(self) -> np.ndarray:
        if any(len(self.history[name]) != self.frame_stack for name in ACTOR_TERM_NAMES):
            raise RuntimeError("TermHistory must be reset before build()")
        return np.concatenate([
            np.concatenate(list(self.history[name]), axis=0)
            for name in ACTOR_TERM_NAMES
        ], axis=0).astype(np.float32, copy=False)


def compute_pd_torque(
    target_q: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> np.ndarray:
    tau = KP * (target_q - qpos) + KD * (0.0 - qvel)
    return np.clip(tau, -EFFORT_LIMITS, EFFORT_LIMITS)


def check_finite(name: str, arr: np.ndarray, **context) -> None:
    bad = ~np.isfinite(arr)
    if not bad.any():
        return
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    lines = [f"\n[deploy] non-finite {name}; halting deploy."]
    lines.append(f"  bad indices: {np.where(bad)[0].tolist()}")
    lines.append(f"  values:      {arr}")
    for key, value in context.items():
        lines.append(f"  {key}: {value}")
    raise FloatingPointError("\n".join(lines))


def record_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step_once: Callable[[], dict[str, np.ndarray]],
    n_steps: int,
    args,
) -> None:
    import cv2

    out_path = Path(args.video).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, width=args.video_width, height=args.video_height)

    if args.video_camera:
        camera = args.video_camera

        def update_lookat() -> None:
            return None
    else:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 1.5
        cam.elevation = -10.0
        cam.azimuth = 90.0
        camera = cam

        def update_lookat() -> None:
            cam.lookat[:] = data.qpos[:3]

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.video_fps),
        (args.video_width, args.video_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open MP4 writer for {out_path}")

    sim_dt_per_frame = 1.0 / args.video_fps
    next_capture = 0.0
    n_frames = 0
    print(
        f"[video] {out_path}  {args.video_width}x{args.video_height} "
        f"@ {args.video_fps} fps"
    )
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
                print_status(data)
    finally:
        writer.release()
        renderer.close()
    print(f"[video] saved {out_path} ({n_frames} frames)")


def print_status(data: mujoco.MjData) -> None:
    quat = data.qpos[3:7]
    grav_b = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
    print(
        f"  t={data.time:5.2f}s  z={data.qpos[2]:.3f}m  "
        f"grav_b_x={grav_b[0]:+.2f} (target +1.0)"
    )
