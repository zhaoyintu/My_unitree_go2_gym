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


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_step_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_step"
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_ppo_runner_cfg()
    cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_no_default_pose"
    return cfg
