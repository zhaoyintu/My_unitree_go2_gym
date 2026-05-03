"""Lite3 front-paw handstand env — Mjlab-aligned reward implementation.

Inherits the per-body collision split (TORSO/THIGH/SHANK) from
`Lite3_legstand_strict` but replaces all of strict's reward gating with
Mjlab's "always-on with cmd-driven moving mask" pattern.  Eleven reward
methods are overridden; one new method (`_reward_alive`) is added.

Why eleven overrides instead of fewer:

- Six handstand-mechanics rewards (`tracking_lin_vel`, `tracking_ang_vel`,
  `feet_clearance`, `contact`, `feet_air_time`, `symmetric_joints`,
  `ang_xz`) are gated by `rew_hanstand > 0.78` in the parent.  Mjlab's
  always-open quality gate is functionally "no gate", so each must be
  re-implemented sans gate.

- For the gait-coupled rewards (`feet_clearance`, `contact`,
  `feet_air_time`) we additionally add a `cmd_norm > 0.1` moving mask.
  Mjlab's gait rewards fire only when there's a non-zero command;
  without this mask, removing the rew_hanstand gate would let the gait
  rewards encourage walking-in-place at cmd=0 — exactly the
  high-frequency foot-tapping observed in MuJoCo deploy.  The cmd
  moving mask is what suppresses tapping in Mjlab.

- The pose-shape reward `default_pos_reward` and the cmd=0 stillness
  rewards `tracking_*_zero` are gated by `rew_hanstand > 0.6` (or 0.78)
  in pose / strict.  Mjlab fires them always.  The `tracking_*_zero`
  override here keeps only the cmd-magnitude mask.

Reward methods NOT overridden (using strict / base implementations):
  base_height, lin_vel_z, ang_vel_xy, torques, dof_acc, action_rate,
  default_pos, default_hip_pos, handstand_feet_on_air,
  handstand_orientation, handstand_feet_height_exp, base_contact,
  thigh_collision, shank_collision.

A note on `_reward_feet_clearance` and `_reward_feet_air_time`: both
have side effects (`self.feet_height` and `self.last_contacts /
self.feet_air_time` respectively) that the parent uses elsewhere.  The
overrides preserve the side-effect logic verbatim — only the trailing
gate expression is changed.
"""
import torch

from legged_gym.envs.Lite3_Stand.Lite3_Handstand_Strict.Lite3_handstand_strict import (
    Lite3_legstand_strict,
)
from legged_gym.envs.Lite3_Stand.Lite3_Handstand_Mjlab.Lite3_handstand_mjlab_Config import (
    Lite3Cfg_LeggedstandMjlab,
    Lite3CfgPPO_LeggedstandMjlab,
)


class Lite3_legstand_mjlab(Lite3_legstand_strict):
    """Lite3 handstand with Mjlab-aligned reward shape (always-on terms,
    cmd-driven moving masks, +1.0 alive baseline).
    """

    def __init__(self, cfg: Lite3Cfg_LeggedstandMjlab, sim_params, physics_engine,
                 sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ------------------------------------------------------------------
    # New: cold-start survival bonus (matches Mjlab's `alive`).
    # ------------------------------------------------------------------
    def _reward_alive(self):
        return torch.ones(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    # Always-on cmd-tracking (parent gates by rew_hanstand > 0.78).
    # ------------------------------------------------------------------
    def _reward_tracking_lin_vel(self):
        # body+x = world-z, body+z = world-x (head-down handstand frame).
        x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 2])
        y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-(x_error + y_error) / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # World yaw → body-frame -ang_vel_x (sign-flipped by inversion).
        ang_vel_error = torch.square(self.commands[:, 2] + self.base_ang_vel[:, 0])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    # ------------------------------------------------------------------
    # Cmd=0 stillness — keep the cmd-magnitude mask, drop rew_hanstand
    # gate.  Same raw-error² shape as strict (so a negative weight
    # actually penalises motion).
    # ------------------------------------------------------------------
    def _reward_tracking_lin_vel_zero(self):
        x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 2])
        y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return (x_error + y_error) * (
            torch.norm(self.commands[:, :2], dim=-1) < 0.1
        )

    def _reward_tracking_ang_vel_zero(self):
        ang_vel_error = torch.square(self.commands[:, 2] + self.base_ang_vel[:, 0])
        return ang_vel_error * (torch.abs(self.commands[:, 2]) < 0.1)

    # ------------------------------------------------------------------
    # Always-on pose-shape reward.  exp(-Σ|err|) over all 12 joints,
    # no gate.  Cfg weight (1.0) sets effective magnitude.
    # ------------------------------------------------------------------
    def _reward_default_pos_reward(self):
        err = self.dof_pos - self.descire_joint_pos        # [num_envs, 12]
        return torch.exp(-torch.sum(torch.abs(err), dim=1))

    # ------------------------------------------------------------------
    # Roll penalty — always on (Mjlab orientation reward already
    # handles the gravity-projection direction; ang_xz is a complement
    # that specifically penalises body roll).
    # ------------------------------------------------------------------
    def _reward_ang_xz(self):
        return torch.abs(self.base_euler_xyz[:, 0])

    # ------------------------------------------------------------------
    # Gait-coupled rewards: replace rew_hanstand gate with cmd-driven
    # moving mask.  Side effects (self.feet_height,
    # self.last_contacts, self.feet_air_time) preserved verbatim from
    # the parent.
    # ------------------------------------------------------------------
    def _reward_feet_clearance(self):
        self.feet_height = self.rigid_state[:, self.contact_foot_indices, 2] - 0.02
        left_feet_height = self.feet_height[:, 0]
        right_feet_height = self.feet_height[:, 1]
        swing_mask = (1 - self._get_gait_phase())
        phase = self._get_phase()
        target_height = torch.abs(torch.sin(2 * torch.pi * phase)) * self.cfg.rewards.target_foot_height
        rew = torch.exp(-torch.abs(left_feet_height - target_height) * 10) * swing_mask[:, 0]
        rew += torch.exp(-torch.abs(right_feet_height - target_height) * 10) * swing_mask[:, 1]
        moving_mask = torch.norm(self.commands[:, :2], dim=-1) > 0.1
        return rew * moving_mask

    def _reward_contact(self):
        contact = self.contact_forces[:, self.contact_foot_indices, 2] > 1
        moving_mask = torch.norm(self.commands[:, :2], dim=-1) > 0.1
        return (torch.sum(contact, dim=1) == 1) * moving_mask

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.contact_foot_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.4) * first_contact, dim=1)
        self.feet_air_time *= ~contact_filt
        moving_mask = torch.norm(self.commands[:, :2], dim=-1) > 0.1
        return rew_airTime * moving_mask

    # ------------------------------------------------------------------
    # Symmetric joints — always on (Mjlab parity).  Mirror-corrected
    # rear hip diff; weight stays at strict's -0.1.
    # ------------------------------------------------------------------
    def _reward_symmetric_joints(self):
        dof = self.dof_pos.clone().view(self.num_envs, 4, int(self.num_dof / 4))
        dof[:, 1, 0] *= -1
        dof[:, 3, 0] *= -1
        err = torch.sum(torch.abs(dof[:, 2, :] - dof[:, 3, :]), axis=1)
        return err
