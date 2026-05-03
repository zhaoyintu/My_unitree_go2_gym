"""10-frame observation variant of the soft-gated Lite3 handstand task."""

from legged_gym.envs.Lite3_Stand.Lite3_Handstand_MjlabGatedTrackingReward.Lite3_handstand_mjlab_gated_tracking_reward_Config import (
    Lite3Cfg_MjlabGatedTrackingReward,
    Lite3CfgPPO_MjlabGatedTrackingReward,
)


class Lite3Cfg_MjlabGatedTrackingFrameStack(Lite3Cfg_MjlabGatedTrackingReward):
    class env(Lite3Cfg_MjlabGatedTrackingReward.env):
        frame_stack = 10
        c_frame_stack = 10
        num_observations = int(
            frame_stack * Lite3Cfg_MjlabGatedTrackingReward.env.num_single_obs
        )
        num_privileged_obs = int(
            c_frame_stack
            * Lite3Cfg_MjlabGatedTrackingReward.env.single_num_privileged_obs
        )


class Lite3CfgPPO_MjlabGatedTrackingFrameStack(Lite3CfgPPO_MjlabGatedTrackingReward):
    class algorithm(Lite3CfgPPO_MjlabGatedTrackingReward.algorithm):
        frame_stack = 10

    class runner(Lite3CfgPPO_MjlabGatedTrackingReward.runner):
        experiment_name = 'lite3_handstand_mjlab_gated_tracking_frame_stack'
