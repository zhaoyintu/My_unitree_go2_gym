"""Soft-gated tracking Lite3 handstand reward variant.

This task keeps the static handstand shaping from
`lite3_handstand_mjlab_static_reward`, but restores large command-tracking
weights.  Tracking rewards remain multiplied by a soft handstand gate, so a
non-handstand posture cannot solve the task by sliding or wobbling.
"""

from legged_gym.envs.Lite3_Stand.Lite3_Handstand_MjlabStaticReward.Lite3_handstand_mjlab_static_reward_Config import (
    Lite3Cfg_MjlabStaticReward,
    Lite3CfgPPO_MjlabStaticReward,
)


class Lite3Cfg_MjlabGatedTrackingReward(Lite3Cfg_MjlabStaticReward):
    class rewards(Lite3Cfg_MjlabStaticReward.rewards):
        class scales(Lite3Cfg_MjlabStaticReward.rewards.scales):
            tracking_lin_vel = 2.5
            tracking_ang_vel = 2.5
            feet_air_time = 0.0
            feet_clearance = 0.0


class Lite3CfgPPO_MjlabGatedTrackingReward(Lite3CfgPPO_MjlabStaticReward):
    class runner(Lite3CfgPPO_MjlabStaticReward.runner):
        experiment_name = 'lite3_handstand_mjlab_gated_tracking_reward'
