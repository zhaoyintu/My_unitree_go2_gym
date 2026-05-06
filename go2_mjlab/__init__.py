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
    unitree_lite3_handstand_rldeploy_robotlab_env_cfg,
)
from go2_mjlab.config.lite3_handstand_rldeploy.rl_cfg import (
    unitree_lite3_handstand_rldeploy_dr_ppo_runner_cfg,
    unitree_lite3_handstand_rldeploy_ppo_runner_cfg,
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
