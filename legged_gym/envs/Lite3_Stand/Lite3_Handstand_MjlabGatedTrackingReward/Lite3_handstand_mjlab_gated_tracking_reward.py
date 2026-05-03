"""Lite3 handstand reward with high tracking behind a soft handstand gate."""

import torch

from legged_gym.envs.Lite3_Stand.Lite3_Handstand_MjlabStaticReward.Lite3_handstand_mjlab_static_reward import (
    Lite3_legstand_mjlab_static_reward,
)
from legged_gym.envs.Lite3_Stand.Lite3_Handstand_MjlabGatedTrackingReward.Lite3_handstand_mjlab_gated_tracking_reward_Config import (
    Lite3Cfg_MjlabGatedTrackingReward,
    Lite3CfgPPO_MjlabGatedTrackingReward,
)


class Lite3_legstand_mjlab_gated_tracking_reward(Lite3_legstand_mjlab_static_reward):
    """Static handstand shaping plus high command tracking after handstand."""

    def __init__(self, cfg: Lite3Cfg_MjlabGatedTrackingReward, sim_params,
                 physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _rear_feet_air_quality(self):
        contact = torch.norm(
            self.contact_forces[:, self.feet_name_reward_indices, :],
            dim=-1,
        ) > 1.0
        return (~contact).float().mean(dim=1)

    def _handstand_quality(self):
        pose_quality = torch.clamp(
            0.45 * self._orientation_quality()
            + 0.35 * self._rear_foot_lift_quality()
            + 0.20 * self._base_height_quality(),
            min=0.0,
            max=1.0,
        )
        rear_air_quality = self._rear_feet_air_quality()
        return pose_quality * (0.20 + 0.80 * rear_air_quality)
