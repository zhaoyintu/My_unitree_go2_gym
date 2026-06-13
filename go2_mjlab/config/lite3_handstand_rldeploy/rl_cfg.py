"""RL runner config for the RL-deploy-aligned Lite3 handstand task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from go2_mjlab.config.lite3_handstand.rl_cfg import unitree_lite3_handstand_ppo_runner_cfg


def unitree_lite3_handstand_rldeploy_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy"
    return cfg


def unitree_lite3_handstand_rldeploy_dr_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_dr"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v2"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v3_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v2_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v3"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v4_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v3_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v4"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v5_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v4_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v5"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v6_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v5_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v6"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v7_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v6_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v7"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v8_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v7_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v8"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v9_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v8_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v9"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v10_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v9_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v10"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v11_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v9_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v11"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v12_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v11_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v12"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v13_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v12_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v13"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v14_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v13_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v14"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v14_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_step_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_step"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_no_default_pose"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_finetune_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Stage-2 finetune runner: resume the Stride-V15 checkpoint, add energy cost.

    ``experiment_name`` stays the Stride-V15 one so mjlab resolves
    ``load_run``/``load_checkpoint`` inside that experiment's log root.
    rsl_rl restores the iteration counter and trains ``max_iterations``
    ADDITIONAL iterations (14999 -> 17999).  Override the source via
    --agent.load-run / --agent.load-checkpoint when needed.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_ppo_runner_cfg()
    cfg.run_name = "finetune"
    cfg.resume = True
    cfg.load_run = "2026-05-20_15-15-26"      # stage-1 Stride-V15 run
    cfg.load_checkpoint = "model_14999.pt"
    cfg.max_iterations = 3_000
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_terrain_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Terrain-robustness finetune runner: resume Stride-V15, train on varied terrain.

    Keeps the Stride-V15 experiment_name so mjlab resolves load_run inside
    that experiment's log root; rsl_rl restores the iter counter and trains
    max_iterations ADDITIONAL iters (14999 -> 17999).  Override the source
    with --agent.load-run / --agent.load-checkpoint if needed.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_ppo_runner_cfg()
    cfg.run_name = "terrain"
    cfg.resume = True
    cfg.load_run = "2026-05-20_15-15-26"      # stage-1 Stride-V15 run
    cfg.load_checkpoint = "model_14999.pt"
    cfg.max_iterations = 3_000
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_rewardfix_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Experiment C runner — reward-only stride fix, RESUME the Stride-V15 ckpt.

    Obs unchanged (450-dim) so resume is valid.  Keeps the V15 experiment_name
    so load_run resolves; 3000 additional finetune iters (14999 -> 17999).
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_ppo_runner_cfg()
    cfg.run_name = "rewardfix"
    cfg.resume = True
    cfg.load_run = "2026-05-20_15-15-26"      # stage-1 Stride-V15 run
    cfg.load_checkpoint = "model_14999.pt"
    cfg.max_iterations = 3_000
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Experiment B runner — phase-clock obs, train FROM SCRATCH.

    The actor input grew (470-dim), so the 450-dim checkpoint cannot be
    resumed — train fresh.  Its own experiment_name keeps the run dir separate
    and avoids the resume/load_run path.  Full 15000-iter budget.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock"
    cfg.resume = False
    cfg.max_iterations = 15_000
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_robust_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Ground-robustness finetune runner: resume Stride-V15, flat + wide friction/pushes."""
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_ppo_runner_cfg()
    cfg.run_name = "robust"
    cfg.resume = True
    cfg.load_run = "2026-05-20_15-15-26"
    cfg.load_checkpoint = "model_14999.pt"
    cfg.max_iterations = 3_000
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_ppo_runner_cfg():
    """AMP finetune runner cfg: Stride-V15 PPO + AMP hyperparams, warm-start.

    Returns an RslRlAmpOnPolicyRunnerCfg (subclass of the PPO runner cfg) so the
    AMP fields survive asdict(cfg.agent) into the runner's train_cfg dict.
    Copies the V15 PPO/actor/critic fields verbatim, keeps the V15
    experiment_name (so load_run resolves there), and warm-starts from
    model_14999.pt (actor/critic obs stay 450-dim; the discriminator + AMP
    normalizer start fresh).
    """
    import dataclasses
    from pathlib import Path

    from go2_mjlab.amp.cfg import RslRlAmpOnPolicyRunnerCfg

    base = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_ppo_runner_cfg()
    cfg = RslRlAmpOnPolicyRunnerCfg()
    for f in dataclasses.fields(base):
        setattr(cfg, f.name, getattr(base, f.name))

    # AMP motion data — repo-relative so it resolves on any host.
    # rl_cfg.py: parents[0]=lite3_handstand_rldeploy, [1]=config, [2]=go2_mjlab.
    amp_npz = Path(__file__).resolve().parents[2] / "amp_motions" / "lite3_handstand.npz"
    cfg.amp_motion_files = (str(amp_npz),)
    cfg.amp_reward_coef = 2.0
    cfg.amp_task_reward_lerp = 0.5
    cfg.amp_discr_hidden_dims = (1024, 512)
    cfg.amp_num_preload_transitions = 2_000_000
    cfg.amp_replay_buffer_size = 1_000_000

    # Warm-start from the Stride-V15 stage-1 checkpoint.
    cfg.run_name = "amp"
    cfg.resume = True
    cfg.load_run = "2026-05-20_15-15-26"
    cfg.load_checkpoint = "model_14999.pt"
    cfg.max_iterations = 3_000
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_rewardfix_ppo_runner_cfg():
    """AMP+RewardFix finetune runner cfg.

    Identical to the pure-AMP runner cfg (same AMP hyperparams, same warm-start
    from Stride-V15 ``model_14999``) — only ``run_name`` differs so its logs land
    in a separate run dir.  The RewardFix reward relaxations live in the ENV cfg,
    not the runner; obs stays 450-dim so the warm-start is valid.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_ppo_runner_cfg()
    cfg.run_name = "amp_rewardfix"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_ppo_runner_cfg():
    """Minimal-reward + AMP, zero-command standing handstand — FROM SCRATCH.

    Same AMP machinery as the V15-AMP runner, but: (a) its OWN experiment_name
    (a fresh from-scratch run, NOT a warm-start of V15), (b) resume disabled,
    (c) full 15000-iter budget, (d) a stronger style coefficient (3.0, up from
    2.0) since AMP is now the SOLE front-paw gait shaper, with lerp 0.5 to keep
    the minimal task reward's authority over balance while training from scratch.
    """
    import dataclasses
    from pathlib import Path

    from go2_mjlab.amp.cfg import RslRlAmpOnPolicyRunnerCfg

    base = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_ppo_runner_cfg()
    cfg = RslRlAmpOnPolicyRunnerCfg()
    for f in dataclasses.fields(base):
        setattr(cfg, f.name, getattr(base, f.name))

    amp_npz = Path(__file__).resolve().parents[2] / "amp_motions" / "lite3_handstand.npz"
    cfg.amp_motion_files = (str(amp_npz),)
    cfg.amp_reward_coef = 3.0
    cfg.amp_task_reward_lerp = 0.5
    cfg.amp_discr_hidden_dims = (1024, 512)
    cfg.amp_num_preload_transitions = 2_000_000
    cfg.amp_replay_buffer_size = 1_000_000

    # FROM SCRATCH — own experiment dir, no warm-start.
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_minimal_amp_stand"
    cfg.run_name = "minimal_amp_stand"
    cfg.resume = False
    cfg.max_iterations = 15_000
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_ppo_runner_cfg():
    """Path B runner: same from-scratch 470-dim PhaseClock training, own experiment dir."""
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock"
    cfg.run_name = "phaseclock_gaitlock"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_resume_ppo_runner_cfg():
    """Minimal-reward + AMP standing handstand, WARM-STARTED from Stride-V15.

    Same minimal env + AMP machinery as the from-scratch variant (coef 3.0,
    lerp 0.5), but instead of training from scratch — which got STUCK lying prone
    because the minimal reward gives too weak a gradient to discover the kick-up
    into the handstand — this resumes the Stride-V15 stage-1 checkpoint, which
    already knows how to reach and hold the front-paw handstand (deployable: it
    learned the kick-up).  The minimal reward + AMP then reshape the gait toward
    a clean standing alternation while the obs stays 450-dim.  Keeps the V15
    experiment_name so ``load_run`` resolves; logged under run_name
    ``minimal_amp_stand_resume``; 3000 finetune iters (14999 -> 17999).
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15"
    cfg.run_name = "minimal_amp_stand_resume"
    cfg.resume = True
    cfg.load_run = "2026-05-20_15-15-26"
    cfg.load_checkpoint = "model_14999.pt"
    cfg.max_iterations = 3_000
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_v2_ppo_runner_cfg():
    """Minimal-v2 runner: from-scratch AMP, own experiment dir."""
    cfg = unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_minimal_amp_stand_v2"
    cfg.run_name = "minimal_amp_stand_v2"
    return cfg
