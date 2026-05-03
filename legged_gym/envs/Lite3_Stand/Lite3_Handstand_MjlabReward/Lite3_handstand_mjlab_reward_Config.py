"""Lite3 handstand task with mjlab-aligned rewards and IsaacGym DR.

This variant intentionally inherits the base IsaacGym Lite3 handstand task's
control, reset, PPO, terrain, and domain randomization settings.  Only reward
scales are changed to mirror `Mjlab-Lite3-Handstand` more closely.
"""

from legged_gym.envs.Lite3_Stand.Lite3_Handstand.Lite3_handstand_Config import (
    Lite3Cfg_Leggedstand,
    Lite3CfgPPO_Leggedstand,
)


class Lite3Cfg_MjlabReward(Lite3Cfg_Leggedstand):
    class rewards(Lite3Cfg_Leggedstand.rewards):
        class scales(Lite3Cfg_Leggedstand.rewards.scales):
            # Mjlab reward set.
            alive = 1.0
            handstand_orientation = -1.0
            handstand_feet_on_air = 0.4
            handstand_feet_height_exp = 5.0
            base_height = 1.5
            tracking_lin_vel = 2.5
            tracking_ang_vel = 2.5
            tracking_lin_vel_zero = -0.2
            tracking_ang_vel_zero = -0.2
            lin_vel_z = 0.2
            ang_vel_xy = 0.2
            symmetric_joints = -0.1
            contact = 0.3
            feet_air_time = 2.0
            feet_clearance = 0.4
            default_pos = -1.0
            default_pos_reward = 1.0
            default_hip_pos = -0.5
            action_rate = -0.05
            dof_acc = -2.5e-4
            base_contact = -2.0
            thigh_collision = -1.0
            shank_collision = -2.0

            # Disable base IsaacGym-only reward terms that are absent from the
            # mjlab Lite3 handstand reward recipe.
            termination = 0.0
            torques = 0.0
            dof_vel = 0.0
            collision = 0.0
            feet_stumble = 0.0
            ang_xz = 0.0


class Lite3CfgPPO_MjlabReward(Lite3CfgPPO_Leggedstand):
    class runner(Lite3CfgPPO_Leggedstand.runner):
        experiment_name = 'lite3_handstand_mjlab_reward'
