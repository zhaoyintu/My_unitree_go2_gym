"""MuJoCo deploy/viewer for Mjlab-Lite3-Footstand checkpoints.

Same 45 x 10 term-major actor contract as the RLDeploy handstand viewer,
but with the footstand training parameters:

* DEFAULT_JOINT_POS = Lite3_stand init pose (front HipY -0.8, hind HipY
  -1.0, Knee 1.5, HipX FL/HL -0.1 FR/HR +0.1) — enters both the
  joint_pos observation offset and the action target.
* ACTION_SCALE = 0.25 uniform (the footstand task does NOT use the
  handstand 0.125 HipX scale).
* Dynamics matched to training: 200 Hz physics (dt 0.005, decimation 4)
  with the mjlab reflected-inertia joint armature.  PD torque is
  recomputed every physics step, which is equivalent to mjlab's builtin
  position actuator (Kp=40, Kd=1, |tau| <= 30 Nm).

Footstand target: projected gravity body-x = -1.0 (nose UP), trunk
world z = 0.52 m.

Example:
    python deploy_mujoco_viewer/deploy_mjlab_lite3_footstand.py \
        --policy logs/rsl_rl/lite3_footstand/<run>/model_XXXX.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lite3_rldeploy_common as C  # noqa: E402
from deploy_mjlab_lite3_handstand_rldeploy import Actor  # noqa: E402

# ---- Footstand overrides ----------------------------------------------------

# Lite3_stand default joint angles in MJCF order (FL, FR, HL, HR).
FOOTSTAND_DEFAULT_JOINT_POS = np.array([
    -0.1, -0.8, 1.5,   # FL
    +0.1, -0.8, 1.5,   # FR
    -0.1, -1.0, 1.5,   # HL
    +0.1, -1.0, 1.5,   # HR
], dtype=np.float64)

# policy_terms() and reset_state() read the module-global default pose.
C.DEFAULT_JOINT_POS = FOOTSTAND_DEFAULT_JOINT_POS

ACTION_SCALE = 0.25                       # uniform, per the footstand task
SIM_DT = 0.005                            # training timestep
DECIMATION = 4                            # training decimation -> 50 Hz policy
CTRL_DT = SIM_DT * DECIMATION

# mjlab BuiltinPositionActuator armature = reflected rotor inertia
# (rotor 0.000111842, gear 6/6/9 for HipX/HipY/Knee).
_ROTOR_INERTIA = 0.000111842
FOOTSTAND_ARMATURE = np.array(
    [_ROTOR_INERTIA * 6**2, _ROTOR_INERTIA * 6**2, _ROTOR_INERTIA * 9**2] * 4,
    dtype=np.float64,
)


def build_footstand_model() -> mujoco.MjModel:
    model = C.build_model()
    model.opt.timestep = SIM_DT
    for i, joint_name in enumerate(C.JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        model.dof_armature[model.jnt_dofadr[joint_id]] = FOOTSTAND_ARMATURE[i]
    return model


def print_status(data: mujoco.MjData) -> None:
    quat = data.qpos[3:7]
    grav_b = C.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
    print(
        f"  t={data.time:5.2f}s  z={data.qpos[2]:.3f}m  "
        f"grav_b_x={grav_b[0]:+.2f} (target -1.0)"
    )


# record_video() resolves print_status via the common module global.
C.print_status = print_status


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--cmd", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        metavar=("vx", "vy", "wz"),
                        help="Constant velocity command (world fwd, lateral, yaw)")
    parser.add_argument("--duration", type=float, default=120.0,
                        help="Rollout duration in seconds")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Run headless without launching a viewer")
    parser.add_argument("--no-realtime", action="store_true",
                        help="Do not pace GUI loop to real time")
    parser.add_argument("--null-policy", action="store_true",
                        help="Use zero action instead of the checkpoint policy")
    parser.add_argument("--summary", type=int, default=0, metavar="N",
                        help="Print N one-line control-step summaries and exit")
    parser.add_argument("--video", type=str, default=None,
                        help="Write rollout to this MP4 path")
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-camera", type=str, default=None)
    return parser


def main() -> None:
    args = _make_parser().parse_args()

    actor = Actor()
    actor.load(args.policy)
    print(f"[deploy] loaded Mjlab footstand policy: {args.policy}")
    print(
        f"[deploy] actor_obs_dim={C.NUM_OBS}, single_obs={C.NUM_SINGLE_OBS}, "
        f"frame_stack={C.FRAME_STACK}, sim_dt={SIM_DT}, decimation={DECIMATION}"
    )
    print(f"[deploy] cmd={args.cmd}, policy_dt={CTRL_DT * 1000:.1f} ms")

    model = build_footstand_model()
    data = mujoco.MjData(model)
    C.reset_state(model, data)
    act_ids = C.actuator_ids(model)

    cmd = np.asarray(args.cmd, dtype=np.float64)
    last_action = np.zeros(12, dtype=np.float64)
    history = C.TermHistory()
    history.reset(C.policy_terms(model, data, cmd, last_action))

    def build_obs() -> np.ndarray:
        obs = history.build()
        if obs.shape != (C.NUM_OBS,):
            raise RuntimeError(f"obs shape {obs.shape} != ({C.NUM_OBS},)")
        return obs

    def step_once() -> dict[str, np.ndarray]:
        nonlocal last_action
        obs = build_obs()
        C.check_finite("obs", obs, t=data.time, base_z=data.qpos[2])
        if args.null_policy:
            action = np.zeros(12, dtype=np.float64)
        else:
            action = actor.act(obs).astype(np.float64)
        C.check_finite("action", action, t=data.time)

        target_q = action * ACTION_SCALE + FOOTSTAND_DEFAULT_JOINT_POS
        C.check_finite("target_q", target_q, t=data.time)
        tau = np.zeros(12, dtype=np.float64)
        for _ in range(DECIMATION):
            qpos, qvel = C.joint_state(model, data)
            tau = C.compute_pd_torque(target_q, qpos, qvel)
            data.ctrl[act_ids] = tau
            mujoco.mj_step(model, data)

        last_action = action.astype(np.float64)
        history.push(C.policy_terms(model, data, cmd, last_action))
        return {"obs": obs, "action": action, "target_q": target_q, "tau": tau}

    n_steps = int(args.duration / CTRL_DT)

    if args.summary > 0:
        print("\nstep  time   base_z  grav_b_x  max|a|  max|tau|")
        for i in range(args.summary):
            step_info = step_once()
            grav_b = C.quat_rotate_inverse(
                data.qpos[3:7], np.array([0.0, 0.0, -1.0])
            )
            print(
                f"{i:4d}  {data.time:5.2f}  {data.qpos[2]:6.3f}  "
                f"{grav_b[0]:+8.2f}  "
                f"{np.abs(step_info['action']).max():6.2f}  "
                f"{np.abs(step_info['tau']).max():7.2f}"
            )
        return

    if args.video:
        C.record_video(model, data, step_once, n_steps, args)
        return

    if args.no_viewer:
        for i in range(n_steps):
            step_once()
            if (i + 1) % int(round(1.0 / CTRL_DT)) == 0:
                print_status(data)
        return

    print("[viewer] SPACE = pause/resume, close window to exit")
    paused = [False]

    def key_callback(keycode: int) -> None:
        if keycode == 32:
            paused[0] = not paused[0]
            print(f"[viewer] {'paused' if paused[0] else 'resumed'}")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -10.0
        viewer.cam.azimuth = 90.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.4]

        i = 0
        while i < n_steps and viewer.is_running():
            t0 = time.time()
            if not paused[0]:
                step_once()
                i += 1
            viewer.sync()
            if not args.no_realtime:
                sleep_time = CTRL_DT - (time.time() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)


if __name__ == "__main__":
    main()
