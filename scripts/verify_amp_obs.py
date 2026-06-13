"""Cross-check: env amp_observations() == dataset amp_frames slice.

Sets each env to a different npz handstand frame (root pose + joint pos/vel),
forwards the sim, and compares the env's 43-dim AMP-obs against the dataset's
amp_frames[:, [7:49]++[2]].  The position terms (joint_pos, foot_pos_base,
root_z) and joint_vel MUST match tightly; base lin/ang vel are skipped (we do
not set a matching world-frame base velocity here, and AMP tolerates their
finite-difference mismatch anyway).

Run inside the mjlab env on a CUDA host (warp):
    PYTHONPATH=<mjlab-src> python scripts/verify_amp_obs.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import go2_mjlab  # noqa: F401  (registers tasks)
from go2_mjlab.amp.observations import amp_observations
from go2_mjlab.config.lite3_handstand_rldeploy.env_cfgs import (
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_env_cfg as amp_env_cfg,
)
from mjlab.envs import ManagerBasedRlEnv

NPZ = Path(go2_mjlab.__file__).parent / "amp_motions" / "lite3_handstand.npz"


def main():
    device = "cuda:0"
    data = np.load(NPZ)
    af = data["amp_frames"]  # (N, 61)
    frames = [0, 40, 80, 120, 160, af.shape[0] - 1]
    n = len(frames)

    cfg = amp_env_cfg(play=True)
    cfg.scene.num_envs = n
    env = ManagerBasedRlEnv(cfg, device=device)
    env.reset()
    robot = env.scene["robot"]

    # Build per-env state from the chosen frames.
    root_pos = torch.tensor(data["root_pos"][frames], dtype=torch.float32, device=device)
    q_xyzw = data["root_quat_xyzw"][frames]
    q_wxyz = torch.tensor(q_xyzw[:, [3, 0, 1, 2]], dtype=torch.float32, device=device)
    jp = torch.tensor(data["joint_pos"][frames], dtype=torch.float32, device=device)
    jv = torch.tensor(data["joint_vel"][frames], dtype=torch.float32, device=device)

    root_pose = torch.cat([root_pos, q_wxyz], dim=-1)  # (n, 7)
    env_ids = torch.arange(n, device=device)
    robot.write_root_link_pose_to_sim(root_pose, env_ids)
    robot.write_joint_position_to_sim(jp, env_ids=env_ids)
    robot.write_joint_velocity_to_sim(jv, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()

    amp = amp_observations(env).detach().cpu().numpy()  # (n, 43)

    # Expected slices from the dataset amp_frames.
    exp_jp = af[frames, 7:19]
    exp_foot = af[frames, 19:31]
    exp_jv = af[frames, 37:49]
    exp_z = af[frames, 2:3]

    def report(name, got, exp, sl):
        err = np.abs(got[:, sl] - exp).max()
        ok = err < 1e-3
        print(f"  {name:16s} slice {sl}: max|err|={err:.2e}  {'OK' if ok else 'MISMATCH'}")
        return ok

    print(f"amp_obs shape: {amp.shape} (expect ({n}, 43))")
    ok = True
    ok &= report("joint_pos", amp, exp_jp, slice(0, 12))
    ok &= report("foot_pos_base", amp, exp_foot, slice(12, 24))
    ok &= report("joint_vel", amp, exp_jv, slice(30, 42))
    ok &= report("root_z", amp, exp_z, slice(42, 43))
    # Show base lin/ang vel (not set to match; informational only).
    print(f"  base_lin_vel env vs data (informational): "
          f"{amp[0, 24:27]} vs {af[frames[0], 31:34]}")
    print("RESULT:", "ALL POSITION/JOINT-VEL TERMS MATCH ✓" if ok else "MISMATCH ✗")
    env.close()


if __name__ == "__main__":
    main()
