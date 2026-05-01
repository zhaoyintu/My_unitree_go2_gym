"""Lite3 handstand env with stricter anti-kneeling guards.

Inherits from `Lite3_legstand` and adds:
  * Three per-body collision reward functions (TORSO/THIGH/SHANK) so each
    can be weighted independently — see config for weights.
  * Override `_reward_default_pos_reward` to track ALL 12 joints (parent
    only tracks rear 6), so the front legs receive direct angle-tracking
    pressure toward HipY=+0.283 / Knee=+2.0 once the gate fires.

The contact-index lookup runs in `_init_buffers` (after the parent has
created the actors), using `gym.find_actor_rigid_body_handle` for each
THIGH/SHANK/TORSO body name.
"""
import torch

from legged_gym.envs.base.base_task import BaseTask  # noqa: F401  (parent import chain)
from legged_gym.envs.Lite3_Stand.Lite3_Handstand.Lite3_handstand import Lite3_legstand
from legged_gym.envs.Lite3_Stand.Lite3_Handstand_Strict.Lite3_handstand_strict_Config import (
    Lite3Cfg_LeggedstandStrict,
    Lite3CfgPPO_LeggedstandStrict,
)


class Lite3_legstand_strict(Lite3_legstand):
    """Lite3 front-paw handstand with mjlab-aligned anti-kneeling rewards."""

    def __init__(self, cfg: Lite3Cfg_LeggedstandStrict, sim_params, physics_engine,
                 sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()

        # Lite3 URDF body naming convention is fixed: TORSO + four legs
        # (FL/FR/HL/HR) × {THIGH, SHANK, FOOT}.  Look up the rigid-body
        # handles by name — IsaacGym's actor API exposes only
        # `find_actor_rigid_body_handle(name)` (the parent's
        # `_create_envs` uses the same pattern at lines 794-810).
        def lookup(names) -> torch.Tensor:
            ids = [
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], n)
                for n in names
            ]
            return torch.tensor(ids, dtype=torch.long, device=self.device)

        legs = ("FL", "FR", "HL", "HR")
        self.thigh_contact_indices = lookup([f"{leg}_THIGH" for leg in legs])
        self.shank_contact_indices = lookup([f"{leg}_SHANK" for leg in legs])
        self.torso_contact_indices = lookup(["TORSO"])

    # ------------------------------------------------------------------
    # Per-body collision penalties (replace parent's combined `collision`)
    # ------------------------------------------------------------------
    def _reward_thigh_collision(self):
        return torch.sum(
            1.0 * (torch.norm(
                self.contact_forces[:, self.thigh_contact_indices, :], dim=-1
            ) > 0.1),
            dim=1,
        )

    def _reward_shank_collision(self):
        return torch.sum(
            1.0 * (torch.norm(
                self.contact_forces[:, self.shank_contact_indices, :], dim=-1
            ) > 0.1),
            dim=1,
        )

    def _reward_base_contact(self):
        return torch.sum(
            1.0 * (torch.norm(
                self.contact_forces[:, self.torso_contact_indices, :], dim=-1
            ) > 0.1),
            dim=1,
        )

    # ------------------------------------------------------------------
    # Override: track ALL 12 joints (parent slices `[:, 6:]` rear-only).
    # ------------------------------------------------------------------
    def _reward_default_pos_reward(self):
        err = self.dof_pos - self.descire_joint_pos        # [num_envs, 12]
        return torch.exp(-torch.sum(torch.abs(err), dim=1)) * (
            torch.mean(self.rew_hanstand) > 0.78
        )
