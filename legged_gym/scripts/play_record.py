"""Headless IsaacGym replay + MP4 recorder.

Loads a trained policy and rolls it out in the SAME IsaacGym sim that
produced the checkpoint, capturing frames via an offscreen camera sensor
(no X11 window — runs fine in WSL).  Use this to verify that a model
actually does the task in its training simulator before blaming the
deploy stack.

Mirrors `play.py` for env-cfg overrides (no noise, no DR push, no command
resampling, num_envs=1), then records `--frames` control steps to MP4.

Example:
    python legged_gym/scripts/play_record.py \
        --task=lite3_handstand_strict \
        --load_run=-1 --checkpoint=-1 \
        --frames=600 --video=/tmp/lite3_strict.mp4
"""
import os
import sys

# Strip the other locomotion repo from sys.path before any legged_gym import
# so we resolve to this repo (mirrors the smoke-test trick used elsewhere).
sys.path = [p for p in sys.path if 'go2_rl_gym-recovery' not in p]

import argparse

import numpy as np

import isaacgym  # noqa: F401  must precede torch import
from isaacgym import gymapi, gymutil
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401, F403  registers tasks
from legged_gym.utils import task_registry
import torch


def play_and_record(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Mirror play.py overrides for clean evaluation.
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.commands.resampling_time = 10000000.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_towards_goal = False
    env_cfg.env.test = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Optionally override the env's _reset_dofs / _reset_root_states so the
    # video starts from EXACTLY the default 4-paw stance.  Default training
    # reset randomises `dof_pos = default * rand(0.5, 1.5)` which can make
    # the start frame look like "the dog is already half-flipped".
    if args.force_default_init:
        import types
        from isaacgym import gymtorch

        def _reset_dofs_clean(self, env_ids):
            self.dof_pos[env_ids] = self.default_dof_pos
            self.dof_vel[env_ids] = 0.
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        def _reset_root_states_clean(self, env_ids):
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, 0:3] += self.env_origins[env_ids]
            # Zero velocities (training default uses rand(-0.5, 0.5)).
            self.root_states[env_ids, 7:13] = 0.
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        env._reset_dofs = types.MethodType(_reset_dofs_clean, env)
        env._reset_root_states = types.MethodType(_reset_root_states_clean, env)
        # Trigger reset on all envs so the patched versions take effect.
        env.reset_idx(torch.arange(env.num_envs, device=env.device))
        env.compute_observations()
        print("[play_record] forced exact default init pose")

    obs = env.get_observations()

    # Build runner with FRESH weights (no resume) so the standard log-dir
    # search isn't triggered, then manually load the ckpt the user gave us.
    train_cfg.runner.resume = False
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    if args.policy_path:
        ckpt = torch.load(args.policy_path, map_location=env.device,
                          weights_only=False)
        runner.alg.actor_critic.load_state_dict(ckpt["model_state_dict"])
        runner.alg.actor_critic.to(env.device)
        print(f"[play_record] loaded policy from {args.policy_path}")
    policy = runner.get_inference_policy(device=env.device)

    # Override commands to zero (front-paw stand stationary).
    env.commands[:, 0] = float(args.cmd_vx)
    env.commands[:, 1] = float(args.cmd_vy)
    env.commands[:, 2] = float(args.cmd_wz)

    # ------------------------------------------------------------------
    # Trace mode: run physics + policy for N steps, no rendering.
    # Uses whatever init pose the env's reset gave us (random in
    # [0.5, 1.5] * default per Lite3_handstand training cfg) — slightly
    # different from deploy's exact-default init but close enough to
    # compare action magnitudes and physics evolution.
    # ------------------------------------------------------------------
    if args.trace > 0:
        torch.set_printoptions(precision=4, sci_mode=False, linewidth=160)
        print(f"\n=== ISAACGYM TRACE first {args.trace} control steps ===")
        print(f"sim.dt={env.cfg.sim.dt}  decimation={env.cfg.control.decimation}  "
              f"ctrl_dt={env.cfg.sim.dt * env.cfg.control.decimation}")

        # Summary mode: only base_z, grav_b_x, max action magnitude per step.
        if args.trace_summary:
            print(f"  step  base_z  grav_b_x  grav_b_z  max|a|     mean|dq|  done")
            for i in range(args.trace):
                actions = policy(obs.detach())
                a = actions[0].detach().cpu().numpy()
                obs, _, rews, dones, infos = env.step(actions.detach())
                base_z = float(env.root_states[0, 2].item())
                grav_b = env.projected_gravity[0].cpu().numpy()
                qvel = env.dof_vel[0].cpu().numpy()
                done_flag = int(dones[0].item())
                print(f"  {i:4d}  {base_z:6.3f}  {grav_b[0]:+.3f}    "
                      f"{grav_b[2]:+.3f}    {np.abs(a).max():6.2f}    "
                      f"{np.abs(qvel).mean():6.2f}    {done_flag}")
            print(f"\n=== TRACE done.  Final base_z="
                  f"{env.root_states[0, 2].item():.4f}m ===")
            return
        for i in range(args.trace):
            quat = env.base_quat[0].cpu().numpy()
            qpos = env.dof_pos[0].cpu().numpy()
            qvel = env.dof_vel[0].cpu().numpy()
            ang_vel_raw = env.base_ang_vel[0].cpu().numpy() / env.obs_scales.ang_vel
            grav_b = env.projected_gravity[0].cpu().numpy()
            obs_np = obs[0].detach().cpu().numpy()

            print(f"\n--- step {i}  base_z={env.root_states[0, 2].item():.4f}m"
                  f"  base_quat(xyzw)={quat} ---")
            print(f"  obs[0:3]   zeros + stand_cmd          : {obs_np[0:3]}")
            print(f"  obs[3:6]   ang_vel * 0.25  raw={ang_vel_raw} : {obs_np[3:6]}")
            print(f"  obs[6:9]   projected_gravity (body)   : {obs_np[6:9]}")
            print(f"  obs[9:12]  scaled_cmd                  : {obs_np[9:12]}")
            print(f"  obs[12:24] (q-default)*1.0  raw_q={qpos}")
            print(f"             scaled                       : {obs_np[12:24]}")
            print(f"  obs[24:36] dq*0.05  raw_dq={qvel}")
            print(f"             scaled                       : {obs_np[24:36]}")
            print(f"  obs[36:48] last_action                  : {obs_np[36:48]}")

            actions = policy(obs.detach())
            action_np = actions[0].detach().cpu().numpy()
            scaled = action_np * env.action_scale_per_joint.cpu().numpy()
            target_q = scaled + env.default_dof_pos[0].cpu().numpy()
            print(f"  raw action (policy out, no scale)      : {action_np}")
            print(f"  target_q   = action*scale + default    : {target_q}")

            obs, _, rews, dones, infos = env.step(actions.detach())

        print(f"\n=== TRACE done.  Final base_z={env.root_states[0, 2].item():.4f}m ===")
        return

    # ------------------------------------------------------------------
    # Trajectory-replay mode: run IsaacGym physics headless, dump
    # (base_pos, base_quat, dof_pos) at each step, then re-render the
    # captured trajectory via MuJoCo's offscreen renderer.  Bypasses
    # IsaacGym's GL backend entirely (which can't initialise on headless
    # servers without an X11 display, e.g. WSL2 or DISPLAY-less ssh).
    # ------------------------------------------------------------------
    if args.render_via_mujoco:
        from pathlib import Path
        import cv2
        import mujoco

        repo_root = Path(__file__).resolve().parents[2]
        mjcf_path = repo_root / "go2_mjlab" / "robots" / "xmls" / "lite3.xml"
        if not mjcf_path.exists():
            raise FileNotFoundError(f"lite3 MJCF not found at {mjcf_path}")

        # MJCF joint order — must match IsaacGym dof_names (verified earlier).
        JOINT_NAMES_MJCF = [
            "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
            "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
            "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
            "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
        ]

        # Step 1: roll out in IsaacGym, dump physics state per frame.
        n_frames = int(args.frames)
        traj_pos = np.zeros((n_frames, 3), dtype=np.float64)
        traj_quat_xyzw = np.zeros((n_frames, 4), dtype=np.float64)  # IsaacGym order
        traj_dof = np.zeros((n_frames, 12), dtype=np.float64)
        print(f"[play_record] step 1/2: rolling out {n_frames} frames in "
              f"IsaacGym headless...")
        for step in range(n_frames):
            traj_pos[step] = env.root_states[0, 0:3].cpu().numpy()
            traj_quat_xyzw[step] = env.root_states[0, 3:7].cpu().numpy()
            traj_dof[step] = env.dof_pos[0].cpu().numpy()
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            if (step + 1) % int(round(args.video_fps)) == 0:
                qx, qy, qz, qw = traj_quat_xyzw[step]
                # body-frame gravity x-component as orientation probe.
                r00 = 1 - 2 * (qy * qy + qz * qz)
                r02 = 2 * (qx * qz - qy * qw)
                grav_b_x = -r02
                print(f"  step {step + 1:4d}  base_z={traj_pos[step, 2]:.3f}m  "
                      f"grav_b_x={grav_b_x:+.2f}")

        # Step 2: render the captured trajectory in MuJoCo.
        print(f"[play_record] step 2/2: rendering {n_frames} frames in "
              f"MuJoCo (offscreen)...")
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        data = mujoco.MjData(model)

        joint_qposadr = []
        for jn in JOINT_NAMES_MJCF:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise RuntimeError(f"joint {jn} not found in {mjcf_path}")
            joint_qposadr.append(int(model.jnt_qposadr[jid]))

        # Bigger offscreen framebuffer for high-res rendering.
        model.vis.global_.offwidth = args.video_width
        model.vis.global_.offheight = args.video_height

        renderer = mujoco.Renderer(
            model, height=args.video_height, width=args.video_width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 2.0
        cam.elevation = -15.0
        cam.azimuth = 90.0

        out_path = os.path.abspath(os.path.expanduser(args.video))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            out_path, fourcc, float(args.video_fps),
            (args.video_width, args.video_height))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open MP4 writer: {out_path}")

        for i in range(n_frames):
            # Free joint: qpos[0:3] = pos, qpos[3:7] = quat (MuJoCo wxyz).
            data.qpos[0:3] = traj_pos[i]
            qx, qy, qz, qw = traj_quat_xyzw[i]
            data.qpos[3:7] = [qw, qx, qy, qz]
            for j, qposadr in enumerate(joint_qposadr):
                data.qpos[qposadr] = traj_dof[i, j]
            mujoco.mj_forward(model, data)
            cam.lookat[:] = data.qpos[0:3]
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        renderer.close()
        print(f"[play_record] saved {out_path}  ({n_frames} frames @ "
              f"{args.video_fps} fps)")
        return

    # Offscreen camera attached to env 0 — chase view of the trunk.
    cam_props = gymapi.CameraProperties()
    cam_props.width = args.video_width
    cam_props.height = args.video_height
    cam_props.horizontal_fov = 75.0
    cam_props.use_collision_geometry = False
    cam_handle = env.gym.create_camera_sensor(env.envs[0], cam_props)
    if cam_handle == -1:
        raise RuntimeError(
            "create_camera_sensor returned -1 — offscreen rendering may need "
            "a working GL/EGL backend.  Try `unset DISPLAY` or run with "
            "`PYOPENGL_PLATFORM=egl`."
        )

    def update_camera():
        # Keep the camera 1.5 m to the side and 0.4 m up, looking at the trunk.
        root = env.root_states[0, 0:3].cpu().numpy()
        cam_pos = gymapi.Vec3(root[0] - 1.5, root[1] - 1.5, root[2] + 0.4)
        cam_target = gymapi.Vec3(root[0], root[1], root[2])
        env.gym.set_camera_location(
            cam_handle, env.envs[0], cam_pos, cam_target)

    update_camera()

    # MP4 writer.
    import cv2
    out_path = os.path.abspath(os.path.expanduser(args.video))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = float(args.video_fps)
    writer = cv2.VideoWriter(
        out_path, fourcc, fps, (args.video_width, args.video_height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open MP4 writer: {out_path}")

    print(f"[play_record] task={args.task}")
    print(f"[play_record] writing {args.frames} frames → {out_path}")

    for step in range(int(args.frames)):
        # Update camera to track trunk.
        update_camera()

        # Render all camera sensors so the offscreen buffer is up to date.
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        img = env.gym.get_camera_image(
            env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR)
        # IsaacGym returns RGBA bytes flattened — reshape and drop alpha.
        img = np.frombuffer(img, dtype=np.uint8).reshape(
            args.video_height, args.video_width, 4)[..., :3]
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # One control step.
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if (step + 1) % int(round(fps)) == 0:
            root = env.root_states[0]
            base_z = float(root[2].item())
            quat = root[3:7].cpu().numpy()  # IsaacGym (x, y, z, w)
            qx, qy, qz, qw = quat
            # body-frame gravity x-component as a quick "are we inverted?" probe.
            grav_world = np.array([0.0, 0.0, -1.0])
            # quat_apply_inverse: rotate by -q.  Manual since we don't import torch utils.
            r = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qz * qw), 2 * (qx * qz - qy * qw)],
                [2 * (qx * qy - qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qx * qw)],
                [2 * (qx * qz + qy * qw), 2 * (qy * qz - qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ])
            grav_body = r @ grav_world
            mean_rew = float(rews.mean().item())
            print(f"  step {step + 1:4d}  base_z={base_z:.3f}m  "
                  f"grav_b_x={grav_body[0]:+.2f}  rew={mean_rew:+.3f}")

    writer.release()
    print(f"[play_record] saved {out_path}")


def main():
    # gymutil.parse_arguments already provides --headless, --num_envs,
    # --rl_device, --seed, --compute_device_id, etc.  Don't re-declare them
    # here or they conflict and our defaults are silently dropped.  Only
    # add the args legged_gym + play_record actually need beyond gymutil's
    # built-ins.
    custom_parameters = [
        {"name": "--task", "type": str, "default": "lite3_handstand_strict",
         "help": "Task name (registered in legged_gym/envs/__init__.py)"},
        {"name": "--resume", "action": "store_true", "default": True,
         "help": "Resume from checkpoint (forced True for play_record)"},
        {"name": "--experiment_name", "type": str},
        {"name": "--run_name", "type": str},
        {"name": "--load_run", "type": str, "default": "-1"},
        {"name": "--checkpoint", "type": int, "default": -1},
        # legged_gym uses these via get_args; they're NOT in gymutil's
        # built-in set so we add them here.
        {"name": "--headless", "action": "store_true", "default": False,
         "help": "(forced True at runtime; flag kept for arg-compat only)"},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--num_envs", "type": int},
        {"name": "--seed", "type": int},
        {"name": "--max_iterations", "type": int},
        {"name": "--horovod", "action": "store_true", "default": False},
        # play_record-specific.
        {"name": "--policy_path", "type": str, "default": "",
         "help": "Direct path to a model_NNN.pt checkpoint (bypasses the "
                 "standard logs/<task>/<run>/ search)"},
        {"name": "--video", "type": str,
         "default": "/tmp/lite3_handstand.mp4",
         "help": "MP4 output path"},
        {"name": "--video_width", "type": int, "default": 1280},
        {"name": "--video_height", "type": int, "default": 720},
        {"name": "--video_fps", "type": int, "default": 50,
         "help": "Frames per second (matches training control rate)"},
        {"name": "--frames", "type": int, "default": 500,
         "help": "Number of control steps to record"},
        {"name": "--cmd_vx", "type": float, "default": 0.0},
        {"name": "--cmd_vy", "type": float, "default": 0.0},
        {"name": "--cmd_wz", "type": float, "default": 0.0},
        {"name": "--trace", "type": int, "default": 0,
         "help": "Print obs/action/state for first N control steps and exit. "
                 "Skips video.  Use to diff against the MuJoCo deploy --trace."},
        {"name": "--trace_summary", "action": "store_true", "default": False,
         "help": "With --trace, print only one line per step: base_z, grav_b, "
                 "max action, mean |dq|, done flag.  Lets you see if the "
                 "rollout stabilizes or diverges over many steps."},
        {"name": "--render_via_mujoco", "action": "store_true", "default": False,
         "help": "Run physics in IsaacGym headless, dump trajectory to "
                 "memory, then render with MuJoCo offscreen.  Use when "
                 "IsaacGym's GL backend can't init (typical on WSL2 or "
                 "DISPLAY-less ssh) — `create_camera_sensor returned -1`."},
        {"name": "--force_default_init", "action": "store_true", "default": False,
         "help": "Override training reset's `dof_pos = default * rand(0.5, 1.5)` "
                 "and `vel = rand(-0.5, 0.5)` so the rollout starts from EXACTLY "
                 "the 4-paw default pose at zero velocity.  Useful for clean "
                 "demo videos."},
    ]
    args = gymutil.parse_arguments(
        description="IsaacGym headless replay + MP4 recorder",
        custom_parameters=custom_parameters,
    )
    # Force headless ON regardless of gymutil's default (this script never
    # opens a viewer).  Disable graphics device only when we don't need
    # rendering — i.e. trace mode.  For video, leave graphics enabled so
    # camera sensors work, but still no viewer (no X11 window).
    args.headless = True
    # IsaacGym's GL backend is not used in trace or render_via_mujoco modes,
    # so disable the graphics device — avoids the create_camera_sensor and
    # X11 init paths that crash on headless servers.
    if args.trace > 0 or args.render_via_mujoco:
        args.graphics_device_id = -1
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    if args.num_envs is None:
        args.num_envs = 1
    if args.rl_device is None:
        args.rl_device = "cuda:0"
    play_and_record(args)


if __name__ == '__main__':
    main()
