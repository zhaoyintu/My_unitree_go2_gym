"""MuJoCo deploy for mujoco_playground Lite3 handstand/footstand policies.

This script is for brax PPO policies trained in mujoco_playground
(Lite3Handstand / Lite3Footstand) and converted to TorchScript. The policy
contract differs from the IsaacGym/RLDeploy Lite3 scripts in this folder:

* Observation: a single 45-dim frame (no history), normalization baked into
  the TorchScript module:
  [local_linvel(3), gyro(3), gravity(3), qpos-default(12), qvel(12),
   last_action(12)].
* Action: INCREMENTAL position targets — ctrl += action * 0.25 each control
  step (the playground env integrates targets from the previous ctrl, it
  does not anchor on the default pose).
* Actuation: MuJoCo position actuators (Kp=40 in the XML, joint damping=1);
  no manual torque PD.

Example:
    python deploy_mujoco_viewer/deploy_playground_lite3_handstand.py \
        --policy logs/playground/lite3_handstand_playground.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import mujoco
import mujoco.viewer

def _default_xml() -> Path:
    """Find the playground Lite3 scene XML.

    Search order: bundled assets next to this script, then a sibling
    mujoco_playground checkout.
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "assets" / "lite3_playground" / "scene_deploy_flat_terrain.xml",
        script_dir / "assets" / "lite3_playground" / "scene_mjx_flat_terrain.xml",
        script_dir.parent.parent / "mujoco_playground" / "mujoco_playground"
        / "_src" / "locomotion" / "lite3" / "xmls" / "scene_deploy_flat_terrain.xml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


PLAYGROUND_LITE3_XML = _default_xml()

ACTION_SCALE = 0.25
SIM_DT = 0.004
DECIMATION = 5
CTRL_DT = SIM_DT * DECIMATION
NUM_OBS = 45
NUM_ACTIONS = 12


def build_model(xml_path: Path) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if abs(model.opt.timestep - SIM_DT) > 1e-9:
        raise RuntimeError(
            f"XML timestep {model.opt.timestep} != expected {SIM_DT}"
        )
    return model


def reset_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = model.keyframe("home").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def build_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    default_pose: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    imu_id = model.site("imu").id
    linvel = data.sensor("local_linvel").data
    gyro = data.sensor("gyro").data
    imu_xmat = data.site_xmat[imu_id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0.0, 0.0, -1.0])
    obs = np.concatenate([
        linvel,
        gyro,
        gravity,
        data.qpos[7:] - default_pose,
        data.qvel[6:],
        last_action,
    ]).astype(np.float32)
    if obs.shape != (NUM_OBS,):
        raise RuntimeError(f"obs shape {obs.shape} != ({NUM_OBS},)")
    return obs


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True,
                        help="Path to TorchScript .pt policy")
    parser.add_argument("--xml", type=str, default=str(PLAYGROUND_LITE3_XML),
                        help="Playground Lite3 scene XML")
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
    return parser


def main() -> None:
    args = _make_parser().parse_args()

    policy = torch.jit.load(args.policy, map_location="cpu")
    policy.eval()
    print(f"[deploy] loaded playground TorchScript policy: {args.policy}")
    print(
        f"[deploy] obs_dim={NUM_OBS} (single frame), action_scale="
        f"{ACTION_SCALE} (incremental), sim_dt={SIM_DT}, "
        f"decimation={DECIMATION}"
    )

    model = build_model(Path(args.xml))
    data = mujoco.MjData(model)
    reset_state(model, data)

    default_pose = model.keyframe("home").qpos[7:].copy()
    ctrl = model.keyframe("home").ctrl.copy()
    ctrl_low = model.actuator_ctrlrange[:, 0]
    ctrl_high = model.actuator_ctrlrange[:, 1]
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float64)

    @torch.no_grad()
    def act(obs_np: np.ndarray) -> np.ndarray:
        obs = torch.from_numpy(obs_np).float().unsqueeze(0)
        return policy(obs).squeeze(0).numpy().astype(np.float64)

    def step_once() -> dict[str, np.ndarray]:
        nonlocal ctrl, last_action
        obs = build_obs(model, data, default_pose, last_action)
        if not np.isfinite(obs).all():
            raise FloatingPointError(f"non-finite obs at t={data.time:.3f}")
        if args.null_policy:
            action = np.zeros(NUM_ACTIONS, dtype=np.float64)
        else:
            action = act(obs)

        ctrl = np.clip(ctrl + action * ACTION_SCALE, ctrl_low, ctrl_high)
        data.ctrl[:] = ctrl
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        last_action = action
        return {"obs": obs, "action": action, "ctrl": ctrl}

    n_steps = int(args.duration / CTRL_DT)

    if args.summary > 0:
        print("\nstep  time   base_z  imu_z   max|a|  max|ctrl_d|")
        imu_id = model.site("imu").id
        for i in range(args.summary):
            info = step_once()
            print(
                f"{i:4d}  {data.time:5.2f}  {data.qpos[2]:6.3f}  "
                f"{data.site_xpos[imu_id][2]:6.3f}  "
                f"{np.abs(info['action']).max():6.2f}  "
                f"{np.abs(info['action']).max() * ACTION_SCALE:7.3f}"
            )
        return

    if args.video:
        import mediapy as media

        renderer = mujoco.Renderer(
            model, height=args.video_height, width=args.video_width
        )
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance, cam.elevation, cam.azimuth = 2.0, -25.0, 135.0
        torso_id = model.body("TORSO").id
        frames = []
        render_every = max(1, int(round(1.0 / (CTRL_DT * args.video_fps))))
        for i in range(n_steps):
            step_once()
            if i % render_every == 0:
                cam.lookat[:] = [
                    data.xpos[torso_id][0], data.xpos[torso_id][1], 0.3,
                ]
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())
        media.write_video(args.video, frames, fps=args.video_fps)
        print(f"[video] saved {args.video} ({len(frames)} frames)")
        return

    if args.no_viewer:
        for i in range(n_steps):
            step_once()
            if (i + 1) % int(round(1.0 / CTRL_DT)) == 0:
                print(
                    f"t={data.time:6.2f}s  base_z={data.qpos[2]:.3f}  "
                    f"quat=({data.qpos[3]:+.2f} {data.qpos[4]:+.2f} "
                    f"{data.qpos[5]:+.2f} {data.qpos[6]:+.2f})"
                )
        return

    print("[viewer] SPACE = pause/resume, R = reset, close window to exit")
    paused = [False]

    def key_callback(keycode: int) -> None:
        if keycode == 32:
            paused[0] = not paused[0]
            print(f"[viewer] {'paused' if paused[0] else 'resumed'}")
        elif keycode in (ord("r"), ord("R")):
            nonlocal_reset()

    def nonlocal_reset() -> None:
        nonlocal ctrl, last_action
        reset_state(model, data)
        ctrl = model.keyframe("home").ctrl.copy()
        last_action = np.zeros(NUM_ACTIONS, dtype=np.float64)
        print("[viewer] reset")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth = 120.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.3]

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
