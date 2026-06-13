"""Dump ground-contact fractions per sensor for a policy at cmd=0 (standing).

Reveals what the robot is resting on (trunk/thigh/calf/which feet) — used to
diagnose the from-scratch Minimal degenerate prone pose.
"""
import ctypes, os, sys, collections
from dataclasses import asdict
from pathlib import Path
os.environ["MUJOCO_GL"] = "egl"
ctypes.CDLL("libEGL.so.1", mode=ctypes.RTLD_GLOBAL)
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import go2_mjlab  # noqa
import mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK_ID = os.environ.get("MEAS_TASK", "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15")


def main():
    ckpt = Path(sys.argv[1]).expanduser().resolve()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 4
    env_cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
    agent_cfg = load_rl_cfg(TASK_ID)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(TASK_ID) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    twist = env.unwrapped.command_manager._terms["twist"]
    scene = env.unwrapped.scene
    cmd = torch.zeros((4, 3), device=device)
    obs, _ = env.reset()
    body = {}
    for nm in ["trunk_ground_touch", "thigh_ground_touch", "calf_ground_touch"]:
        try: body[nm] = scene[nm]
        except Exception: pass
    foot_acc, body_acc = [], collections.defaultdict(list)
    for t in range(400):
        twist.vel_command_b[:] = cmd
        with torch.no_grad(): a = policy(obs)
        obs, *_ = env.step(a)
        if t >= 150:
            foot_acc.append((scene["feet_ground_contact"].data.found > 0).float().mean(0).cpu().numpy())
            for nm, s in body.items():
                f = s.data.found
                body_acc[nm].append(float((f > 0).any(dim=-1).float().mean()) if f.ndim > 1 else float((f > 0).float().mean()))
    foot = np.stack(foot_acc).mean(0)
    print("CONTACT FRACTIONS (mean over last 250 steps, 4 envs):")
    print("  feet [FL,FR,HL,HR]:", np.round(foot, 2))
    for nm in body_acc:
        print(f"  {nm}: {np.mean(body_acc[nm]):.2f}")
    print("  root_z mean:", round(float(scene['robot'].data.root_link_pos_w[:,2].mean()), 3))


if __name__ == "__main__":
    main()
