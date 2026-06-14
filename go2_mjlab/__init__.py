"""Go2 mjlab training package.

Importing this package registers the local Go2 and Lite3 tasks with mjlab.
"""

from go2_mjlab.mdp import commands as mdp_commands  # noqa: F401 - register PhaseCommandCfg
from go2_mjlab.config.go2_trot.env_cfgs import (
    unitree_go2_flat_env_cfg,
    unitree_go2_unified_env_cfg,
)
from go2_mjlab.config.go2_trot.rl_cfg import (
    unitree_go2_trot_ppo_runner_cfg,
    unitree_go2_unified_ppo_runner_cfg,
)
from go2_mjlab.config.go2_stairs.env_cfgs import unitree_go2_stairs_env_cfg
from go2_mjlab.config.go2_stairs.rl_cfg import unitree_go2_stairs_ppo_runner_cfg
from go2_mjlab.config.go2_jump.env_cfgs import unitree_go2_jump_env_cfg
from go2_mjlab.config.go2_jump.rl_cfg import unitree_go2_jump_ppo_runner_cfg
from go2_mjlab.config.go2_handstand.env_cfgs import (
    unitree_go2_handstand_env_cfg,
    unitree_go2_handstand_finetune_env_cfg,
)
from go2_mjlab.config.go2_handstand.rl_cfg import (
    unitree_go2_handstand_ppo_runner_cfg,
    unitree_go2_handstand_finetune_ppo_runner_cfg,
)
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
from go2_mjlab.config.lite3_footstand.env_cfgs import unitree_lite3_footstand_env_cfg
from go2_mjlab.config.lite3_footstand.rl_cfg import unitree_lite3_footstand_ppo_runner_cfg
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
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v7_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v8_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v9_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v10_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v11_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v12_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v13_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v14_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_finetune_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_terrain_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_rewardfix_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_robust_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_rewardfix_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_v2_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v2_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_2hz_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_lowlift10_env_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_lowlift5_env_cfg,
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
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v7_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v8_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v9_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v10_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v11_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v12_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v13_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v14_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_finetune_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_terrain_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_rewardfix_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_robust_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_rewardfix_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_resume_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_v2_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v2_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_2hz_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_lowlift10_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_lowlift5_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_step_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_robotlab_ppo_runner_cfg,
)
from go2_mjlab.amp import Lite3AmpOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

# Flat terrain velocity tracking (trot)
register_mjlab_task(
    "Mjlab-Go2-Trot-Flat",
    env_cfg=unitree_go2_flat_env_cfg(play=False),
    play_env_cfg=unitree_go2_flat_env_cfg(play=True),
    rl_cfg=unitree_go2_trot_ppo_runner_cfg(),
)

# Flat velocity tracking with reward ported 1:1 from UniLab go2_joystick_flat
# (for the flashsac / PPO / mjlab-PPO 3-way comparison under a unified recipe).
register_mjlab_task(
    "Mjlab-Go2-Velocity-Flat-Unified",
    env_cfg=unitree_go2_unified_env_cfg(play=False),
    play_env_cfg=unitree_go2_unified_env_cfg(play=True),
    rl_cfg=unitree_go2_unified_ppo_runner_cfg(),
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

# Handstand stage-2 energy finetune (mujoco_playground two-stage recipe)
register_mjlab_task(
    "Mjlab-Go2-Handstand-Finetune",
    env_cfg=unitree_go2_handstand_finetune_env_cfg(play=False),
    play_env_cfg=unitree_go2_handstand_finetune_env_cfg(play=True),
    rl_cfg=unitree_go2_handstand_finetune_ppo_runner_cfg(),
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

# Lite3 footstand (rear-leg standing), reward design from handstand_gym Lite3_stand
register_mjlab_task(
    "Mjlab-Lite3-Footstand",
    env_cfg=unitree_lite3_footstand_env_cfg(play=False),
    play_env_cfg=unitree_lite3_footstand_env_cfg(play=True),
    rl_cfg=unitree_lite3_footstand_ppo_runner_cfg(),
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

# Stride v9: retarget desired front-HipY to middle of range for stride headroom
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V7",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v7_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v7_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v7_ppo_runner_cfg(),
)

# Stride v10: aggressive 4-knob change to break the V7 plateau
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V8",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v8_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v8_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v8_ppo_runner_cfg(),
)

# Stride v11: relax rewards that oppose alternating stride
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V9",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v9_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v9_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v9_ppo_runner_cfg(),
)

# Stride v12: linear-with-cap swing-foot height reward (replace sin clearance)
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V10",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v10_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v10_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v10_ppo_runner_cfg(),
)

# Stride v13: restore default_pos pull to v10 strength (revert v11 audit's halving)
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V11",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v11_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v11_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v11_ppo_runner_cfg(),
)

# Stride v14: sharpen feet_clearance shape so "no lift" stops paying out
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V12",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v12_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v12_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v12_ppo_runner_cfg(),
)

# Stride v15: drop the rear-foot z target (over-constraint causing HL_Knee saturation)
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V13",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v13_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v13_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v13_ppo_runner_cfg(),
)

# Stride v16: scope default_pos to rear legs only — free front-leg stride
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V14",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v14_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v14_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v14_ppo_runner_cfg(),
)

# Stride v17: add front HipX to default_pos to fix front-leg twist
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_ppo_runner_cfg(),
)

# Stride-V15 stage-2 energy finetune (mujoco_playground two-stage recipe)
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15-Finetune",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_finetune_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_finetune_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_finetune_ppo_runner_cfg(),
)

# Stride-V15 terrain-robustness finetune (varied friction + mild rough/wavy ground, blind)
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15-Terrain",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_terrain_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_terrain_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_terrain_ppo_runner_cfg(),
)

# High-freq-contact fix, experiment C: reward-only stride fix (no obs change), resume
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15-RewardFix",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_rewardfix_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_rewardfix_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_rewardfix_ppo_runner_cfg(),
)

# High-freq-contact fix, experiment B: + gait-phase clock in obs (primary H1 fix), from scratch
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15-PhaseClock",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_ppo_runner_cfg(),
)

# Ground-robustness finetune (flat, NaN-safe): wide friction + strong disturbance pushes
# (replaces the heightfield-terrain finetune, which persistently NaN-ed in mujoco_warp)
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15-Robust",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_robust_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_robust_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_robust_ppo_runner_cfg(),
)

# AMP finetune: adversarial motion-prior style reward from the ahmp handstand mocap,
# vendored AMP module + custom Lite3AmpOnPolicyRunner (warm-starts the V15 checkpoint)
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15-AMP",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_ppo_runner_cfg(),
    runner_cls=Lite3AmpOnPolicyRunner,
)

# AMP + RewardFix: AMP style reward WITH the relaxed amplitude-suppressor
# rewards (same _apply_stride_reward_fixes as RewardFix/PhaseClock), so the
# style reward and task reward push the larger step the same direction.
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15-AMP-RewardFix",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_rewardfix_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_rewardfix_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_rewardfix_ppo_runner_cfg(),
    runner_cls=Lite3AmpOnPolicyRunner,
)

# Minimal-reward + AMP, zero-command STANDING handstand, trained FROM SCRATCH:
# strip the hand-crafted gait rewards (suspected cause of the 5.5 Hz shuffle) and
# let the front-paw alternation emerge purely from the AMP style reward.
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-Minimal-AMP-Stand",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_ppo_runner_cfg(),
    runner_cls=Lite3AmpOnPolicyRunner,
)

# Same minimal env, but WARM-STARTED from Stride-V15 (which already does the
# handstand kick-up) instead of from scratch (which got stuck lying prone).
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-Minimal-AMP-Stand-Resume",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_resume_ppo_runner_cfg(),
    runner_cls=Lite3AmpOnPolicyRunner,
)

# Minimal-v2: re-add the front-paw SUPPORT terms (zero_stance_contact +
# stance_air_penalty) that v1 dropped — v1 knelt on its calves to farm reward.
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-Minimal-AMP-Stand-V2",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_v2_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_v2_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_v2_ppo_runner_cfg(),
    runner_cls=Lite3AmpOnPolicyRunner,
)

# Path B: PhaseClock + hard phase-locked gait control — lock the gait to a slower
# clock (contact schedule -> frequency) and reward swing height (-> amplitude),
# now that the phase is observable. Standard PPO runner (no AMP), from scratch.
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15-PhaseClock-GaitLock",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_ppo_runner_cfg(),
)

# GaitLock-V2: + knee over-fold penalty (front knees were folding to the 2.79 rad
# limit, pressing thigh+shank together — a hardware self-collision). Warm-starts GaitLock.
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15-PhaseClock-GaitLock-V2",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v2_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v2_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v2_ppo_runner_cfg(),
)

# GaitLock-2Hz: V2 + gait clock 1.0->0.5s (design step freq 1Hz->2Hz/foot), to pull
# the actual ~3.9Hz cadence down toward the (more achievable) 2Hz. Warm-starts V2.
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15-PhaseClock-GaitLock-2Hz",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_2hz_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_2hz_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_2hz_ppo_runner_cfg(),
)

# GaitLock-V3: V2 + stronger contact_schedule (2.0->5.0) to enforce the 1Hz cadence
# and pull the actual step frequency down (the 2Hz/faster-clock attempt backfired).
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15-PhaseClock-GaitLock-V3",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_ppo_runner_cfg(),
)

# GaitLock-V3-LowLift10: keep ~4Hz cadence but CAP swing apex at ~10cm (tent reward).
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15-PhaseClock-GaitLock-V3-LowLift10",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_lowlift10_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_lowlift10_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_lowlift10_ppo_runner_cfg(),
)

# GaitLock-V3-LowLift5: same but CAP swing apex at ~5cm — low, fast (~4Hz) small steps.
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-NoHop-BigStep-Stride-V15-PhaseClock-GaitLock-V3-LowLift5",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_lowlift5_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_lowlift5_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_v3_lowlift5_ppo_runner_cfg(),
)

# RobotLab task variant for isolating the effect of removing default-pose shaping
register_mjlab_task(
    "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-NoDefaultPose",
    env_cfg=unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_env_cfg(play=False),
    play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_env_cfg(play=True),
    rl_cfg=unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_ppo_runner_cfg(),
)
