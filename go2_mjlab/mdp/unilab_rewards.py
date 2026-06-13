"""Reward functions ported 1:1 from UniLab go2_joystick_flat.

These replicate UniLab's reward formulas exactly so a mjlab Go2 task can be
compared against UniLab flashsac/PPO under an identical reward recipe.

UniLab reference: src/unilab/envs/locomotion/common/rewards.py + go2/joystick.py
- tracking_lin_vel : exp(-||cmd_xy - v_xy||^2 / sigma)          (xy only)
- tracking_ang_vel : exp(-(cmd_yaw - w_z)^2 / sigma)            (yaw only)
- ang_vel_xy       : sum(w_xy^2)
- similar_to_default: sum(|q - q_default|)                      (L1)
- swing_feet_z     : sum_i exp(-(z_i - 0.1)^2 / 0.01) * [phase_i >= 0.6] / 4
- contact          : sum_i [ (force_i>0.1) == (phase_i < 0.6) ] / 4

Gait clock matches UniLab (gait_frequency=2 -> cycle 0.5s): per-foot phase
offsets [0, 0.5, 0.5, 0] for feet ordered (FR, FL, RR, RL).

lin_vel_z, base_height reuse go2_mdp (already exact); action_rate reuses the
builtin mdp.action_rate_l2 (already exact).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")

# Per-foot trot phase offsets for feet ordered (FR, FL, RR, RL).
_FOOT_PHASE_OFFSET = (0.0, 0.5, 0.5, 0.0)


def _feet_phase(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Return (N, 4) per-foot phase in [0,1), matching UniLab feet_phase."""
    base = (env.episode_length_buf * env.step_dt) % 0.5 / 0.5  # (N,) == (t*2)%1
    offsets = torch.tensor(_FOOT_PHASE_OFFSET, device=env.device, dtype=base.dtype)
    return (base[:, None] + offsets[None, :]) % 1.0


def feet_phase_obs(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Per-foot trot phase in [0,1), shape (N, 4) — matches UniLab feet_phase obs."""
    return _feet_phase(env)


def tracking_lin_vel(
    env: "ManagerBasedRlEnv",
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    v = asset.data.root_link_lin_vel_b
    err = torch.sum(torch.square(command[:, :2] - v[:, :2]), dim=1)
    return torch.exp(-err / sigma)


def tracking_ang_vel(
    env: "ManagerBasedRlEnv",
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    w = asset.data.root_link_ang_vel_b
    err = torch.square(command[:, 2] - w[:, 2])
    return torch.exp(-err / sigma)


def ang_vel_xy(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    w = asset.data.root_link_ang_vel_b
    return torch.sum(torch.square(w[:, :2]), dim=1)


def similar_to_default(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff = asset.data.joint_pos - asset.data.default_joint_pos
    return torch.sum(torch.abs(diff), dim=1)


def swing_feet_z(
    env: "ManagerBasedRlEnv",
    target_height: float = 0.1,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Exp reward for swing-phase feet reaching target height (UniLab swing_feet_z)."""
    asset: Entity = env.scene[asset_cfg.name]
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (N, 4)
    is_swing = (_feet_phase(env) >= 0.6).float()
    height_err = torch.square(foot_z - target_height)
    swing_rew = torch.exp(-height_err / 0.01) * is_swing
    return torch.sum(swing_rew, dim=1) / foot_z.shape[1]


def contact_schedule(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    force_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward feet contact matching the trot schedule (UniLab contact)."""
    sensor: ContactSensor = env.scene[sensor_name]
    # contact force magnitude per foot -> boolean contact
    contact = sensor.data.force.norm(dim=-1) > force_threshold  # (N, 4)
    if contact.dim() == 3:  # (N, 4, slots) -> reduce
        contact = contact.any(dim=-1)
    should_contact = _feet_phase(env) < 0.6  # (N, 4) stance when phase<0.6
    matched = (contact == should_contact).float()
    return torch.sum(matched, dim=1) / matched.shape[1]
