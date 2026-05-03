"""Lite3 handstand env with mjlab-aligned reward functions.

The parent class supplies IsaacGym physics, reset, command sampling, control,
and domain randomization.  This subclass changes only reward behavior:

* opens the old batch-level handstand gate so shaping rewards are dense;
* fixes zero-command linear velocity from negative exp-reward to raw penalty;
* tracks all 12 joints for the desire-pose bonus;
* splits body-contact penalties into torso/thigh/shank groups;
* resets air-time contact state at episode boundaries.
"""

import torch

from legged_gym.envs.Lite3_Stand.Lite3_Handstand.Lite3_handstand import Lite3_legstand
from legged_gym.envs.Lite3_Stand.Lite3_Handstand_MjlabReward.Lite3_handstand_mjlab_reward_Config import (
    Lite3Cfg_MjlabReward,
    Lite3CfgPPO_MjlabReward,
)


class Lite3_legstand_mjlab_reward(Lite3_legstand):
    """Reward-only IsaacGym variant aligned with `Mjlab-Lite3-Handstand`."""

    def __init__(self, cfg: Lite3Cfg_MjlabReward, sim_params, physics_engine,
                 sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()

        def lookup(names) -> torch.Tensor:
            ids = [
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name)
                for name in names
            ]
            return torch.tensor(ids, dtype=torch.long, device=self.device)

        legs = ("FL", "FR", "HL", "HR")
        self.torso_contact_indices = lookup(["TORSO"])
        self.thigh_contact_indices = lookup([f"{leg}_THIGH" for leg in legs])
        self.shank_contact_indices = lookup([f"{leg}_SHANK" for leg in legs])

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            self.last_contacts[env_ids] = False

    def _body_contact_count(self, body_indices: torch.Tensor) -> torch.Tensor:
        return torch.sum(
            (torch.norm(self.contact_forces[:, body_indices, :], dim=-1) > 0.1).float(),
            dim=1,
        )

    def _reward_alive(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_base_contact(self):
        return self._body_contact_count(self.torso_contact_indices)

    def _reward_thigh_collision(self):
        return self._body_contact_count(self.thigh_contact_indices)

    def _reward_shank_collision(self):
        return self._body_contact_count(self.shank_contact_indices)

    def _reward_tracking_lin_vel(self):
        x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 2])
        y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-(x_error + y_error) / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_zero(self):
        x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 2])
        y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        cmd_near_zero = (torch.norm(self.commands[:, :2], dim=-1) < 0.1).float()
        return (x_error + y_error) * cmd_near_zero

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] + self.base_ang_vel[:, 0])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel_zero(self):
        ang_vel_error = torch.square(self.commands[:, 2] + self.base_ang_vel[:, 0])
        cmd_near_zero = (torch.abs(self.commands[:, 2]) < 0.1).float()
        return ang_vel_error * cmd_near_zero

    def _reward_default_pos_reward(self):
        err = self.dof_pos - self.descire_joint_pos
        return torch.exp(-torch.sum(torch.abs(err), dim=1))

    def _reward_feet_clearance(self):
        feet_height = self.rigid_state[:, self.contact_foot_indices, 2] - 0.02
        phase = self._get_phase()
        swing_mask = 1 - self._get_gait_phase()
        target_height = torch.abs(torch.sin(2 * torch.pi * phase)) * self.cfg.rewards.target_foot_height

        rew = torch.exp(-torch.abs(feet_height[:, 0] - target_height) * 10) * swing_mask[:, 0]
        rew += torch.exp(-torch.abs(feet_height[:, 1] - target_height) * 10) * swing_mask[:, 1]
        return rew

    def _reward_contact(self):
        contact = self.contact_forces[:, self.contact_foot_indices, 2] > 1.0
        return (torch.sum(contact, dim=1) == 1).float()

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.contact_foot_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        first_contact = (self.feet_air_time > 0.0).float() * contact_filt.float()
        self.feet_air_time += self.dt
        rew_air_time = torch.sum((self.feet_air_time - 0.4) * first_contact, dim=1)
        self.feet_air_time *= ~contact_filt
        self.last_contacts = contact
        return rew_air_time

    def _reward_symmetric_joints(self):
        dof = self.dof_pos.clone().view(self.num_envs, 4, int(self.num_dof / 4))
        dof[:, 1, 0] *= -1
        dof[:, 3, 0] *= -1
        err_front = torch.sum(torch.abs(dof[:, 0, :] - dof[:, 1, :]), dim=1)
        err_rear = torch.sum(torch.abs(dof[:, 2, :] - dof[:, 3, :]), dim=1)
        return 0.5 * (err_front + err_rear)
