"""Static-first Lite3 handstand reward functions.

This variant is intentionally separate from `lite3_handstand_mjlab_reward` so
the old mjlab-aligned reward can be trained as a baseline.  It keeps the same
IsaacGym environment, DR, control, reset, and collision split, but changes the
reward dynamics that were causing the policy to prefer tracking/base-height
local optima instead of entering the handstand basin.
"""

import torch

from legged_gym.envs.Lite3_Stand.Lite3_Handstand_MjlabReward.Lite3_handstand_mjlab_reward import (
    Lite3_legstand_mjlab_reward,
)
from legged_gym.envs.Lite3_Stand.Lite3_Handstand_MjlabStaticReward.Lite3_handstand_mjlab_static_reward_Config import (
    Lite3Cfg_MjlabStaticReward,
    Lite3CfgPPO_MjlabStaticReward,
)


class Lite3_legstand_mjlab_static_reward(Lite3_legstand_mjlab_reward):
    """Reward-only static handstand stage for A/B comparison."""

    def __init__(self, cfg: Lite3Cfg_MjlabStaticReward, sim_params,
                 physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _orientation_error(self):
        return torch.square(self.projected_gravity - self.target_gravity).sum(dim=1)

    def _orientation_quality(self):
        return torch.exp(
            -self._orientation_error() * self.cfg.rewards.orientation_sharpness
        )

    def _base_height_quality(self):
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights,
            dim=1,
        )
        return torch.exp(
            -torch.abs(base_height - self.cfg.rewards.base_height_target) * 5
        )

    def _rear_foot_heights(self):
        return self.feet_pos[:, :, 2]

    def _rear_foot_lift_quality(self):
        rear_height = torch.mean(self._rear_foot_heights(), dim=1)
        lift_range = (
            self.cfg.rewards.rear_foot_height_target
            - self.cfg.rewards.rear_foot_lift_min
        )
        return torch.clamp(
            (rear_height - self.cfg.rewards.rear_foot_lift_min) / lift_range,
            min=0.0,
            max=1.0,
        )

    def _handstand_quality(self):
        return torch.clamp(
            0.45 * self._orientation_quality()
            + 0.35 * self._rear_foot_lift_quality()
            + 0.20 * self._base_height_quality(),
            min=0.0,
            max=1.0,
        )

    def _reward_base_height(self):
        return self._base_height_quality() * (
            0.25 + 0.75 * self._orientation_quality()
        )

    def _reward_tracking_lin_vel(self):
        x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 2])
        y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return (
            torch.exp(-(x_error + y_error) / self.cfg.rewards.tracking_sigma)
            * self._handstand_quality()
        )

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] + self.base_ang_vel[:, 0])
        return (
            torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
            * self._handstand_quality()
        )

    def _reward_handstand_orientation(self):
        return self._orientation_quality()

    def _reward_handstand_feet_on_air(self):
        contact = torch.norm(
            self.contact_forces[:, self.feet_name_reward_indices, :],
            dim=-1,
        ) > 1.0
        return (~contact).float().mean(dim=1)

    def _reward_handstand_feet_height_exp(self):
        rear_heights = self._rear_foot_heights()
        height_error = torch.mean(
            torch.abs(rear_heights - self.cfg.rewards.rear_foot_height_target),
            dim=1,
        )
        target_quality = torch.exp(
            -height_error * self.cfg.rewards.rear_foot_height_sharpness
        )
        lift_quality = self._rear_foot_lift_quality()
        symmetry_quality = torch.exp(
            -torch.abs(rear_heights[:, 0] - rear_heights[:, 1]) * 5.0
        )
        return (
            0.35 * target_quality
            + 0.55 * lift_quality
            + 0.10 * symmetry_quality * lift_quality
        )

    def _reward_contact(self):
        contact = self.contact_forces[:, self.contact_foot_indices, 2] > 1.0
        return contact.float().mean(dim=1)
