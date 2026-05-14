"""Go2 mjlab training package.

Importing this package registers the local Go2 and Lite3 tasks with mjlab.
"""

from go2_mjlab.mdp import commands as mdp_commands  # noqa: F401 - register PhaseCommandCfg
from go2_mjlab.config.go2_trot.env_cfgs import unitree_go2_flat_env_cfg
from go2_mjlab.config.go2_trot.rl_cfg import unitree_go2_trot_ppo_runner_cfg
from go2_mjlab.config.go2_stairs.env_cfgs import unitree_go2_stairs_env_cfg
from go2_mjlab.config.go2_stairs.rl_cfg import unitree_go2_stairs_ppo_runner_cfg
from go2_mjlab.config.go2_jump.env_cfgs import unitree_go2_jump_env_cfg
from go2_mjlab.config.go2_jump.rl_cfg import unitree_go2_jump_ppo_runner_cfg
from go2_mjlab.config.go2_handstand.env_cfgs import unitree_go2_handstand_env_cfg
from go2_mjlab.config.go2_handstand.rl_cfg import unitree_go2_handstand_ppo_runner_cfg
from go2_mjlab.config.go2_leggedstand.env_cfgs import unitree_go2_leggedstand_env_cfg
from go2_mjlab.config.go2_leggedstand.rl_cfg import unitree_go2_leggedstand_ppo_runner_cfg
from go2_mjlab.config.go2_spring_jump.env_cfgs import unitree_go2_spring_jump_env_cfg
from go2_mjlab.config.go2_spring_jump.rl_cfg import unitree_go2_spring_jump_ppo_runner_cfg
from go2_mjlab.config.go2_backflip.env_cfgs import unitree_go2_backflip_env_cfg
from go2_mjlab.config.go2_backflip.rl_cfg import unitree_go2_backflip_ppo_runner_cfg
from go2_mjlab.config.lite3_handstand.env_cfgs import (
    unitree_lite3_handstand_env_cfg,
    unitree_lite3_handstand_robust_env_cfg,
)
from go2_mjlab.config.lite3_handstand.rl_cfg import (
    unitree_lite3_handstand_ppo_runner_cfg,
    unitree_lite3_handstand_robust_ppo_runner_cfg,
)
from go2_mjlab.config.lite3_handstand_rldeploy.env_cfgs import (
    unitree_lite3_handstand_rldeploy_dr_env_cfg,
    unitree_lite3_handstand_rldeploy_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v2_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v3_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v4_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v5_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v6_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_step_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_env_cfg,
)
from go2_mjlab.config.lite3_handstand_rldeploy.rl_cfg import (
    unitree_lite3_handstand_rldeploy_dr_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v2_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v3_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v4_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v5_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v6_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_step_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_ppo_runner_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

# Flat terrain velocity tracking (trot)
register_mjlab_task(
    "Mjlab-Go2-Trot-Flat",
    env_cfg=unitree_go2_flat_env_cfg(play=False),
    play_env_cfg=unitree_go2_flat_env_cfg(play=True),
    rl_cfg=unitree_go2_trot_ppo_runner_cfg(),
)

# Stairs terrain
register_mjlab_task(
    "Mjlab-Go2-Stairs",
    env_cfg=unitree_go2_stairs_env_cfg(play=False),
    play_env_cfg=unitree_go2_stairs_env_cfg(play=True),
    rl_cfg=unitree_go2_stairs_ppo_runner_cfg(),
)

# Jumping gait
register_mjlab_task(
    "Mjlab-Go2-Jump",
    env_cfg=unitree_go2_jump_env_cfg(play=False),
    play_env_cfg=unitree_go2_jump_env_cfg(play=True),
    rl_cfg=unitree_go2_jump_ppo_runner_cfg(),
)

# Handstand (inverted on front legs)
register_mjlab_task(
    "Mjlab-Go2-Handstand",
    env_cfg=unitree_go2_handstand_env_cfg(play=False),
    play_env_cfg=unitree_go2_handstand_env_cfg(play=True),
    rl_cfg=unitree_go2_handstand_ppo_runner_cfg(),
)

# Legged stand (standing on rear legs)
register_mjlab_task(
    "Mjlab-Go2-Leggedstand",
    env_cfg=unitree_go2_leggedstand_env_cfg(play=False),
    play_env_cfg=unitree_go2_leggedstand_env_cfg(play=True),
    rl_cfg=unitree_go2_leggedstand_ppo_runner_cfg(),
)

# Spring jump
register_mjlab_task(
    "Mjlab-Go2-Spring-Jump",
    env_cfg=unitree_go2_spring_jump_env_cfg(play=False),
    play_env_cfg=unitree_go2_spring_jump_env_cfg(play=True),
    rl_cfg=unitree_go2_spring_jump_ppo_runner_cfg(),
)

# Backflip
register_mjlab_task(
    "Mjlab-Go2-Backflip",
    env_cfg=unitree_go2_backflip_env_cfg(play=False),
    play_env_cfg=unitree_go2_backflip_env_cfg(play=True),
    rl_cfg=unitree_go2_backflip_ppo_runner_cfg(),
)

# Lite3 handstand (front-paw walk on DeepRobotics Lite3)
register_mjlab_task(
    "Mjlab-Lite3-Handstand",
    env_cfg=unitree_lite3_handstand_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_ppo_runner_cfg(),
)

# Lite3 handstand robust variant: stronger zero-command stance and pushes
register_mjlab_task(
    "Mjlab-Lite3-Handstand-Robust",
    env_cfg=unitree_lite3_handstand_robust_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_robust_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_robust_ppo_runner_cfg(),
)

# Lite3 handstand aligned with the C++ rl_deploy_handstand policy contract
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy",
    env_cfg=unitree_lite3_handstand_rldeploy_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_ppo_runner_cfg(),
)

# RLDeploy contract plus supported sim-to-real domain randomization
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-DR",
    env_cfg=unitree_lite3_handstand_rldeploy_dr_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_dr_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_dr_ppo_runner_cfg(),
)

# RobotLab-style handstand rewards/terminations with RLDeploy observations and DR
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_ppo_runner_cfg(),
)

# RobotLab task variant for isolating the effect of weaker default-pose shaping
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_ppo_runner_cfg(),
)

# Low-default-pose variant with stronger zero-command stance stability
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-ZeroStance",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_ppo_runner_cfg(),
)

# Low-default-pose variant with stronger zero-command quiet standing costs
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-QuietZeroStance",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_ppo_runner_cfg(),
)

# Quiet zero-stance variant that encourages alternating front-paw steps when moving
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-QuietStep",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_step_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_step_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_step_ppo_runner_cfg(),
)

# Quiet zero-stance variant with an unconditional anti-hop penalty
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_ppo_runner_cfg(),
)

# NoHop variant tuned for slower, larger-amplitude stepping gait
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_ppo_runner_cfg(),
)

# BigStep + single-stance contact bonus + stronger anti-hop to break out of the hop basin
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_ppo_runner_cfg(),
)

# Stride v4: reduce short-lift farm, amplify long-swing reward, slightly stiffer joints
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V2",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v2_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v2_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v2_ppo_runner_cfg(),
)

# Stride v5: explicit sinusoidal stepping rhythm to escape shuffle gait
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V3",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v3_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v3_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v3_ppo_runner_cfg(),
)

# Stride v6: disable contact mean reward so swing-encouraging signals can win
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V4",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v4_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v4_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v4_ppo_runner_cfg(),
)

# Stride v7: larger and slower swing target for visibly bigger strides
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V5",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v5_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v5_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v5_ppo_runner_cfg(),
)

# Stride v8: amplify swing rewards to push past the v7 plateau
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V6",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v6_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v6_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v6_ppo_runner_cfg(),
)

# RobotLab task variant for isolating the effect of removing default-pose shaping
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-NoDefaultPose",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_ppo_runner_cfg(),
)
