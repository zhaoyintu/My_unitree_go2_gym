"""Env-side AMP observation — the 43-dim discriminator obs the policy produces.

MUST match the dataset's amp_frames slice exactly (see motion_loader.py):
    [joint_pos(12), foot_pos_base(12), base_lin_vel_b(3),
     base_ang_vel_b(3), joint_vel(12), root_z(1)]
in MJCF joint order (FL,FR,HL,HR × HipX,HipY,Knee) and foot order
(FL,FR,HL,HR), raw radians / rad·s⁻¹ / m / m·s⁻¹, velocities in BASE frame.

This is registered as the single term of a dedicated "amp" observation group
(single frame, no noise, no history).  The AMP runner reads obs["amp"] each
step; the actor/critic ignore it (obs_groups only routes "actor"/"critic").
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

_FOOT_SITES = ("FL", "FR", "HL", "HR")
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def amp_observations(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    foot_site_names: tuple[str, ...] = _FOOT_SITES,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    data = asset.data

    joint_pos = data.joint_pos                       # (B, 12) raw radians
    joint_vel = data.joint_vel                       # (B, 12) raw rad/s
    lin_b = data.root_link_lin_vel_b                 # (B, 3) base frame
    ang_b = data.root_link_ang_vel_b                 # (B, 3) base frame
    root_z = data.root_link_pos_w[:, 2:3]            # (B, 1) world height

    # Foot positions in BASE frame, foot order FL,FR,HL,HR (preserve_order!).
    site_ids, _ = asset.find_sites(foot_site_names, preserve_order=True)
    p_w = data.site_pos_w[:, site_ids]               # (B, 4, 3) world
    rel = p_w - data.root_link_pos_w.unsqueeze(1)    # (B, 4, 3)
    quat = data.root_link_quat_w.unsqueeze(1).expand(-1, len(site_ids), -1)
    foot_b = quat_apply_inverse(quat, rel).reshape(env.num_envs, -1)  # (B, 12)

    return torch.cat((joint_pos, foot_b, lin_b, ang_b, joint_vel, root_z), dim=-1)
