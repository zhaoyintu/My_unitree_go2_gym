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
