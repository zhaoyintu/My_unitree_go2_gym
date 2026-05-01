"""Empirical proof of action-vector / joint-vector alignment for Lite3.

Builds the registered Mjlab-Lite3-Handstand env, prints the static
mapping (joint id ↔ name ↔ default_joint_pos ↔ action_scale), and then
fires a one-hot action at each index and verifies which joint actually
moves. If the empirical "moved joint" matches the static "joint at this
index", action order is provably correct.

Run from a Python env where mjlab is importable (the same env you use
for training):

    python deploy_mujoco_viewer/audit_lite3_action_order.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import go2_mjlab  # registers tasks
from mjlab.tasks.registry import _TASK_REGISTRY
from mjlab.envs import ManagerBasedRlEnv


def main():
    cfg = _TASK_REGISTRY["Mjlab-Lite3-Handstand"].env_cfg
    # Cut env count to a small number so this is fast.
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg)
    asset = env.scene["robot"]

    # ---- Static mapping: action index → joint -----------------------------
    ids, names = asset.find_joints_by_actuator_names((".*",))
    default = asset.data.default_joint_pos[0]                  # [num_joints]
    action_term = env.action_manager._terms["joint_pos"]
    scale = action_term.scale[0].cpu().numpy()                  # [12]

    print("\n=== STATIC mapping (what mjlab thinks) =========================")
    print(f"{'action_idx':>10} | {'joint_id':>8} | {'joint_name':<22} | "
          f"{'default':>9} | {'scale':>6}")
    print("-" * 70)
    for action_idx, (jid, jname) in enumerate(zip(ids, names)):
        print(f"{action_idx:>10} | {jid:>8} | {jname:<22} | "
              f"{default[jid].item():+9.3f} | {scale[action_idx]:>6.3f}")

    # ---- Empirical mapping: which joint moves when action[i] = 1.0 --------
    # Reset, then for each action index, drive that single index for a few
    # steps and check which joint position changed the most.
    print("\n=== EMPIRICAL test (one-hot action; biggest joint Δ) ===========")
    print(f"{'action_idx':>10} | {'expected joint':<22} | "
          f"{'moved joint':<22} | {'OK?':>4}")
    print("-" * 70)

    obs, _ = env.reset()
    initial_q = asset.data.joint_pos[0].clone()

    all_ok = True
    for ai in range(12):
        env.reset()
        # Big positive step on action index ai, zero everything else.
        action = torch.zeros(env.num_envs, env.action_manager.total_action_dim,
                             device=env.device)
        action[:, ai] = 5.0
        # Run a few control steps so PD has time to actually move the joint.
        for _ in range(10):
            env.step(action)

        delta = (asset.data.joint_pos[0] - initial_q).cpu().numpy()
        moved_jid = int(delta.__abs__().argmax())
        moved_jname = asset.joint_names[moved_jid]

        expected_jname = names[ai]
        ok = (moved_jname == expected_jname)
        all_ok = all_ok and ok
        print(f"{ai:>10} | {expected_jname:<22} | "
              f"{moved_jname:<22} | {'✓' if ok else '✗ MISMATCH'}")

    print("\n" + ("=" * 70))
    print("RESULT:", "ACTION ORDER OK ✓" if all_ok else "ACTION ORDER BROKEN ✗")
    print("=" * 70)


if __name__ == "__main__":
    main()
