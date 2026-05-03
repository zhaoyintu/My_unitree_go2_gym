"""Static-first Lite3 handstand reward variant for IsaacGym.

This task keeps the previous `lite3_handstand_mjlab_reward` task intact for
A/B comparison.  It inherits that task's IsaacGym DR/control/reset/PPO stack and
only changes reward scales plus a few reward shaping constants.
"""

from legged_gym.envs.Lite3_Stand.Lite3_Handstand_MjlabReward.Lite3_handstand_mjlab_reward_Config import (
    Lite3Cfg_MjlabReward,
    Lite3CfgPPO_MjlabReward,
)


class Lite3Cfg_MjlabStaticReward(Lite3Cfg_MjlabReward):
    class rewards(Lite3Cfg_MjlabReward.rewards):
        class scales(Lite3Cfg_MjlabReward.rewards.scales):
            # Static-first stage: make handstand pose rewards dominate before
            # command tracking is worth much.
            handstand_orientation = 2.0
            handstand_feet_on_air = 1.0
            handstand_feet_height_exp = 8.0
            base_height = 0.8
            tracking_lin_vel = 0.4
            tracking_ang_vel = 0.4
            contact = 0.8
            feet_air_time = 0.0
            feet_clearance = 0.0
            default_pos = -0.6
            default_pos_reward = 2.0

        rear_foot_height_target = 0.56
        rear_foot_lift_min = 0.08
        rear_foot_height_sharpness = 4.0
        orientation_sharpness = 2.0


class Lite3CfgPPO_MjlabStaticReward(Lite3CfgPPO_MjlabReward):
    class runner(Lite3CfgPPO_MjlabReward.runner):
        experiment_name = 'lite3_handstand_mjlab_static_reward'
