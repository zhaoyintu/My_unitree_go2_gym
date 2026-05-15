"""Custom reward functions for Unitree Go2 locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def lin_vel_z(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize z-axis base linear velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def base_height(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize base height deviation from target."""
    asset: Entity = env.scene[asset_cfg.name]
    base_z = asset.data.root_link_pos_w[:, 2]
    return torch.square(base_z - target_height)


def handstand_base_height(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward base height tracking for handstand tasks."""
    asset: Entity = env.scene[asset_cfg.name]
    base_z = asset.data.root_link_pos_w[:, 2]
    return torch.exp(-torch.abs(base_z - target_height) * 5)


class joint_acceleration:
    """Penalize joint accelerations via finite differences (matches IsaacGym formula).

    Unlike IsaacGym which operates on env.dof_vel directly, we go through
    asset.data.joint_vel. The result (unscaled squared difference, no dt
    division) matches IsaacGym's _reward_dof_acc exactly.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
        _, joint_names = asset.find_joints(cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).joint_names)
        self.num_joints = len(joint_names)
        self.last_joint_vel = torch.zeros(
            (env.num_envs, self.num_joints), device=env.device, dtype=torch.float32
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
        acc = joint_vel - self.last_joint_vel
        self.last_joint_vel = joint_vel.clone()
        return torch.sum(torch.square(acc), dim=1)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.last_joint_vel[env_ids] = 0.0


def trot_gait(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    cycle_time: float = 0.5,
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward for diagonal trot gait: (FR,RL) in phase, (FL,RR) opposite.

    Foot order from sensor: FR, FL, RR, RL.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0  # [B, 4]

    phase = (env.episode_length_buf * env.step_dt) % cycle_time / cycle_time
    stance_phase_0 = phase < 0.5
    stance_phase_1 = phase > 0.5

    trot_correct = (
        (contact[:, 0] == contact[:, 3])
        & (contact[:, 1] == contact[:, 2])
        & (contact[:, 0] == stance_phase_0)
        & (contact[:, 1] == stance_phase_1)
    ).float()

    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :3], dim=1)
    moving = (command_norm > 0.1).float()

    return trot_correct * moving


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint motion when no velocity command is given."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :3], dim=1)
    standing = (command_norm < 0.1).float()
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(joint_pos - default_joint_pos), dim=1)
    return deviation * standing


def contact_without_command(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward all 4 feet in contact when no velocity command is given."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0
    all_in_contact = (torch.sum(contact, dim=1) == 4).float()
    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :3], dim=1)
    standing = (command_norm < 0.1).float()
    return all_in_contact * standing


def default_hip_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize hip abduction deviation from zero (keep hips centered).

    Case-insensitive match on "hip" so this works for both Go2
    (`*_hip_joint`) and Lite3 (`*_HipX_joint`).  Callers should still
    pass an `asset_cfg.joint_names` regex that selects ONLY the
    abduction joints — for Lite3 this means `.*HipX_joint`, otherwise
    the HipY (= thigh) joint would also pass the lowercase substring.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _, joint_names = asset.find_joints(asset_cfg.joint_names)
    hip_mask = torch.tensor(
        ["hip" in name.lower() for name in joint_names],
        device=env.device,
        dtype=torch.float32,
    )
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    hip_deviation = torch.sum(torch.abs(joint_pos) * hip_mask.unsqueeze(0), dim=1)
    return hip_deviation


def tracking_linear_velocity_trot(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Linear velocity tracking gated on trot quality."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + z_error

    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0
    phase = (env.episode_length_buf * env.step_dt) % 0.5 / 0.5
    stance_phase_0 = phase < 0.5
    stance_phase_1 = phase > 0.5
    trot_mask = (
        (contact[:, 0] == contact[:, 3])
        & (contact[:, 1] == contact[:, 2])
        & (contact[:, 0] == stance_phase_0)
        & (contact[:, 1] == stance_phase_1)
    ).float()

    return torch.exp(-lin_vel_error / std**2) * (trot_mask > 0.7).float()


def tracking_angular_velocity_trot(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Angular velocity tracking gated on trot quality."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + xy_error

    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0
    phase = (env.episode_length_buf * env.step_dt) % 0.5 / 0.5
    stance_phase_0 = phase < 0.5
    stance_phase_1 = phase > 0.5
    trot_mask = (
        (contact[:, 0] == contact[:, 3])
        & (contact[:, 1] == contact[:, 2])
        & (contact[:, 0] == stance_phase_0)
        & (contact[:, 1] == stance_phase_1)
    ).float()

    return torch.exp(-ang_vel_error / std**2) * (trot_mask > 0.7).float()


def jump_gait(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    cycle_time: float = 0.5,
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward for jumping gait: all 4 feet synchronized (same phase)."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0  # [B, 4]

    phase = (env.episode_length_buf * env.step_dt) % cycle_time / cycle_time
    stance_phase = phase < 0.5

    all_same_phase = (
        (contact[:, 0] == contact[:, 1])
        & (contact[:, 1] == contact[:, 2])
        & (contact[:, 2] == contact[:, 3])
        & (contact[:, 0] == stance_phase)
    ).float()

    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :3], dim=1)
    moving = (command_norm > 0.1).float()

    return all_same_phase * moving


def alive(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Constant survival bonus."""
    return torch.ones(env.num_envs, device=env.device)


def lin_vel_x_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward body x-axis linear velocity (vertical pumping in handstand).

    Matches IsaacGym _reward_lin_vel_z: torch.square(base_lin_vel[:, 0]).
    Despite the name, this is a REWARD for body-x motion — it encourages
    the dynamic up/down motion needed to maintain handstand balance.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 0])


def handstand_lin_vel_z(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward low vertical body motion in front-paw handstand."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.exp(-torch.abs(asset.data.root_link_lin_vel_b[:, 0]) * 10)


def ang_vel_xy_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize body roll (x) and pitch (y) angular velocity.

    Matches IsaacGym _reward_ang_vel_xy: exp(-norm(|ang_vel[:, :2]|)).
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.exp(-torch.norm(torch.abs(asset.data.root_link_ang_vel_b[:, :2]), dim=1))


def handstand_ang_vel_yz(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward low non-yaw angular velocity during front-paw handstand."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.exp(-torch.norm(torch.abs(asset.data.root_link_ang_vel_b[:, 1:3]), dim=1))


def default_joint_penalty(
    env: ManagerBasedRlEnv,
    desire_joint_angles: list[float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from desired joint angles."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target = torch.tensor(desire_joint_angles, device=env.device, dtype=torch.float32)
    return torch.sum(torch.abs(joint_pos - target), dim=1)


# ---------------------------------------------------------------------------
# Handstand reward functions
# ---------------------------------------------------------------------------


def handstand_orientation(
    env: ManagerBasedRlEnv,
    target_gravity: tuple[float, float, float],
) -> torch.Tensor:
    """Reward matching projected gravity to target direction (inverted)."""
    target = torch.tensor(target_gravity, device=env.device, dtype=torch.float32)
    asset: Entity = env.scene["robot"]
    projected_gravity = asset.data.projected_gravity_b
    return torch.square(projected_gravity - target).sum(dim=1)


def handstand_feet_on_air(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_indices: tuple[int, ...] = (0, 1, 2, 3),
) -> torch.Tensor:
    """Reward selected (swing) feet being off the ground.

    `foot_indices` indexes the contact sensor's foot list (mjlab order:
    0=FR, 1=FL, 2=RR, 3=RL).  For front-paw handstand walking the swing
    legs are the rear ones, so pass (2, 3).
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0  # [B, 4]
    selected = contact[:, list(foot_indices)]
    return (~selected).float().prod(dim=1)


def handstand_feet_height(
    env: ManagerBasedRlEnv,
    target_height: float,
    foot_indices: tuple[int, ...] = (2, 3),
    foot_site_names: tuple[str, ...] = ("FR", "FL", "RR", "RL"),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward selected (swing) feet reaching `target_height` in WORLD z.

    Matches IsaacGym GO2_Leggedstand `_reward_handstand_feet_height_exp`,
    which evaluates `feet_pos[:, :, 2]` of `feet_name_reward` (= rear feet
    for the front-paw handstand task) against an absolute world-frame
    target (0.67 m).  `foot_indices` references positions in
    `foot_site_names`; default is the Go2 site naming, but other robots
    (e.g. Lite3 uses "FL", "FR", "HL", "HR") can override.
    """
    asset: Entity = env.scene[asset_cfg.name]
    site_ids, _ = asset.find_sites(foot_site_names)
    feet_z = asset.data.site_pos_w[:, site_ids, 2]  # [B, 4]
    selected = feet_z[:, list(foot_indices)]
    error = torch.abs(selected - target_height).sum(dim=1)
    return torch.exp(-error * 10)


def _handstand_orientation_quality(
    env: ManagerBasedRlEnv,
    target_gravity: tuple[float, float, float] = (1.0, 0.0, 0.0),
    sharpness: float = 2.0,
) -> torch.Tensor:
    target = torch.tensor(target_gravity, device=env.device, dtype=torch.float32)
    asset: Entity = env.scene["robot"]
    error = torch.square(asset.data.projected_gravity_b - target).sum(dim=1)
    return torch.exp(-error * sharpness)


def _handstand_base_height_quality(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    base_z = asset.data.root_link_pos_w[:, 2]
    return torch.exp(-torch.abs(base_z - target_height) * 5)


def _handstand_foot_heights(
    env: ManagerBasedRlEnv,
    foot_indices: tuple[int, ...],
    foot_site_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    site_ids, _ = asset.find_sites(foot_site_names)
    feet_z = asset.data.site_pos_w[:, site_ids, 2]
    return feet_z[:, list(foot_indices)]


def _handstand_rear_foot_lift_quality(
    env: ManagerBasedRlEnv,
    target_height: float = 0.56,
    rear_foot_lift_min: float = 0.08,
    foot_indices: tuple[int, ...] = (2, 3),
    foot_site_names: tuple[str, ...] = ("FL", "FR", "HL", "HR"),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    selected = _handstand_foot_heights(env, foot_indices, foot_site_names, asset_cfg)
    rear_height = torch.mean(selected, dim=1)
    lift_range = max(target_height - rear_foot_lift_min, 1e-6)
    return torch.clamp((rear_height - rear_foot_lift_min) / lift_range, min=0.0, max=1.0)


def _handstand_rear_air_quality(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_indices: tuple[int, ...] = (2, 3),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0
    selected = contact[:, list(foot_indices)]
    return (~selected).float().mean(dim=1)


def _handstand_stance_contact_quality(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_indices: tuple[int, ...] = (0, 1),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0
    selected = contact[:, list(foot_indices)]
    return selected.float().mean(dim=1)


def _handstand_body_clearance_quality(
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
) -> torch.Tensor:
    if not sensor_names:
        asset: Entity = env.scene["robot"]
        return torch.ones(env.num_envs, device=asset.data.root_link_pos_w.device)

    quality = None
    for sensor_name in sensor_names:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        found = contact_sensor.data.found > 0
        body_contact = found.reshape(found.shape[0], -1).any(dim=1)
        clear = (~body_contact).float()
        quality = clear if quality is None else quality * clear
    return quality


def _handstand_support_quality(
    env: ManagerBasedRlEnv,
    stance_sensor_name: str | None = None,
    stance_foot_indices: tuple[int, ...] | None = None,
    body_clearance_sensor_names: tuple[str, ...] = (),
) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    quality = torch.ones(env.num_envs, device=asset.data.root_link_pos_w.device)
    if stance_sensor_name is not None and stance_foot_indices is not None:
        quality = quality * _handstand_stance_contact_quality(
            env,
            sensor_name=stance_sensor_name,
            foot_indices=stance_foot_indices,
        )
    if body_clearance_sensor_names:
        quality = quality * _handstand_body_clearance_quality(env, body_clearance_sensor_names)
    return quality


def _handstand_soft_quality(
    env: ManagerBasedRlEnv,
    target_gravity: tuple[float, float, float] = (1.0, 0.0, 0.0),
    orientation_sharpness: float = 2.0,
    base_height_target: float = 0.39,
    rear_foot_target_height: float = 0.56,
    rear_foot_lift_min: float = 0.08,
    rear_foot_indices: tuple[int, ...] = (2, 3),
    foot_site_names: tuple[str, ...] = ("FL", "FR", "HL", "HR"),
    sensor_name: str | None = None,
    stance_foot_indices: tuple[int, ...] | None = None,
    body_clearance_sensor_names: tuple[str, ...] = (),
    support_floor: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Soft per-env handstand gate for RLDeploy reward terms.

    This keeps command tracking dense near the correct pose but prevents a
    prone/sliding policy from collecting full tracking reward before it has
    learned the front-paw handstand geometry.
    """
    orientation_quality = _handstand_orientation_quality(
        env, target_gravity=target_gravity, sharpness=orientation_sharpness
    )
    lift_quality = _handstand_rear_foot_lift_quality(
        env,
        target_height=rear_foot_target_height,
        rear_foot_lift_min=rear_foot_lift_min,
        foot_indices=rear_foot_indices,
        foot_site_names=foot_site_names,
        asset_cfg=asset_cfg,
    )
    base_height_quality = _handstand_base_height_quality(
        env, target_height=base_height_target, asset_cfg=asset_cfg
    )
    pose_quality = torch.clamp(
        0.45 * orientation_quality + 0.35 * lift_quality + 0.20 * base_height_quality,
        min=0.0,
        max=1.0,
    )
    if sensor_name is not None:
        rear_air_quality = _handstand_rear_air_quality(
            env, sensor_name=sensor_name, foot_indices=rear_foot_indices
        )
        pose_quality = pose_quality * (0.20 + 0.80 * rear_air_quality)
    support_quality = _handstand_support_quality(
        env,
        stance_sensor_name=sensor_name,
        stance_foot_indices=stance_foot_indices,
        body_clearance_sensor_names=body_clearance_sensor_names,
    )
    pose_quality = pose_quality * (
        support_floor + (1.0 - support_floor) * support_quality
    )
    return pose_quality


def handstand_orientation_exp(
    env: ManagerBasedRlEnv,
    target_gravity: tuple[float, float, float] = (1.0, 0.0, 0.0),
    sharpness: float = 2.0,
) -> torch.Tensor:
    """Positive handstand orientation reward with a smooth gradient."""
    return _handstand_orientation_quality(env, target_gravity=target_gravity, sharpness=sharpness)


def handstand_base_height_soft(
    env: ManagerBasedRlEnv,
    target_height: float,
    target_gravity: tuple[float, float, float] = (1.0, 0.0, 0.0),
    orientation_sharpness: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Base-height reward that is strongest near handstand orientation."""
    height_quality = _handstand_base_height_quality(env, target_height=target_height, asset_cfg=asset_cfg)
    orientation_quality = _handstand_orientation_quality(
        env, target_gravity=target_gravity, sharpness=orientation_sharpness
    )
    return height_quality * (0.25 + 0.75 * orientation_quality)


def handstand_rear_feet_height_static(
    env: ManagerBasedRlEnv,
    target_height: float,
    foot_indices: tuple[int, ...] = (2, 3),
    foot_site_names: tuple[str, ...] = ("FL", "FR", "HL", "HR"),
    rear_foot_lift_min: float = 0.08,
    rear_foot_height_sharpness: float = 4.0,
    stance_sensor_name: str | None = None,
    stance_foot_indices: tuple[int, ...] | None = None,
    body_clearance_sensor_names: tuple[str, ...] = (),
    support_floor: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Rear-foot height reward with lift progress before the exact target.

    The old `exp(-error * 10)` term is nearly zero while the rear feet are
    still low, which leaves very little gradient for escaping the prone local
    optimum.  This mirrors the IsaacGym static-first Lite3 recipe.
    """
    rear_heights = _handstand_foot_heights(env, foot_indices, foot_site_names, asset_cfg)
    height_error = torch.mean(torch.abs(rear_heights - target_height), dim=1)
    target_quality = torch.exp(-height_error * rear_foot_height_sharpness)
    lift_quality = _handstand_rear_foot_lift_quality(
        env,
        target_height=target_height,
        rear_foot_lift_min=rear_foot_lift_min,
        foot_indices=foot_indices,
        foot_site_names=foot_site_names,
        asset_cfg=asset_cfg,
    )
    symmetry_quality = torch.exp(-torch.abs(rear_heights[:, 0] - rear_heights[:, 1]) * 5.0)
    height_quality = 0.35 * target_quality + 0.55 * lift_quality + 0.10 * symmetry_quality * lift_quality
    support_quality = _handstand_support_quality(
        env,
        stance_sensor_name=stance_sensor_name,
        stance_foot_indices=stance_foot_indices,
        body_clearance_sensor_names=body_clearance_sensor_names,
    )
    return height_quality * (support_floor + (1.0 - support_floor) * support_quality)


def handstand_feet_height_l2_exp(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float,
    foot_indices: tuple[int, ...] = (2, 3),
    foot_site_names: tuple[str, ...] = ("FL", "FR", "HL", "HR"),
    stance_sensor_name: str | None = None,
    stance_foot_indices: tuple[int, ...] | None = None,
    body_clearance_sensor_names: tuple[str, ...] = (),
    support_floor: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """RobotLab-style selected-foot height reward with optional support gate."""
    feet_height = _handstand_foot_heights(env, foot_indices, foot_site_names, asset_cfg)
    feet_height_error = torch.sum(torch.square(feet_height - target_height), dim=1)
    height_reward = torch.exp(-feet_height_error / std**2)
    support_quality = _handstand_support_quality(
        env,
        stance_sensor_name=stance_sensor_name,
        stance_foot_indices=stance_foot_indices,
        body_clearance_sensor_names=body_clearance_sensor_names,
    )
    return height_reward * (support_floor + (1.0 - support_floor) * support_quality)


def ang_xz_penalty(
    env: ManagerBasedRlEnv,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Penalize roll (x) angular velocity during handstand. Gated by quality."""
    asset: Entity = env.scene["robot"]
    base_ang_vel = asset.data.root_link_ang_vel_b
    quality = _handstand_quality(env, target_height)
    return (torch.abs(base_ang_vel[:, 0]) + torch.abs(base_ang_vel[:, 2])) * (quality > 0.70).float()


def symmetric_joints(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward left-right joint symmetry."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [B, 12]
    dof = joint_pos.view(env.num_envs, 4, 3).clone()  # [B, 4 legs, 3 joints]
    # Negate right-side hip abduction to match left side
    dof[:, 1, 0] = -dof[:, 1, 0]
    dof[:, 3, 0] = -dof[:, 3, 0]
    err_front = torch.sum(torch.abs(dof[:, 0, :] - dof[:, 1, :]), dim=1)
    err_rear = torch.sum(torch.abs(dof[:, 2, :] - dof[:, 3, :]), dim=1)
    return 0.5 * (err_front + err_rear)


# ---------------------------------------------------------------------------
# Legged stand reward functions
# ---------------------------------------------------------------------------


def leggedstand_orientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward upright orientation for legged stand (gravity ~ [0,0,-1])."""
    target = torch.tensor((0.0, 0.0, -1.0), device=env.device, dtype=torch.float32)
    asset: Entity = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    return torch.square(projected_gravity - target).sum(dim=1)


def leggedstand_feet_on_air(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_indices: tuple[int, ...],
) -> torch.Tensor:
    """Reward selected feet (FR, FL) being off the ground."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0  # [B, 4]
    selected_contact = contact[:, list(foot_indices)]
    return (~selected_contact).float().prod(dim=1)


# ---------------------------------------------------------------------------
# Spring jump / Backflip shared reward functions
# ---------------------------------------------------------------------------


def upward_velocity(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Reward positive z-axis (upward) velocity during the jump phase."""
    asset: Entity = env.scene["robot"]
    z_vel = asset.data.root_link_lin_vel_b[:, 2]
    command = env.command_manager.get_command(command_name)
    phase_flag = command[:, 2]  # 1 = jump phase
    # Reward upward velocity during jump phase, before has_jumped
    rew = (z_vel > 0).float() * z_vel * (phase_flag == 1).float()
    return rew


def landing_position(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Reward landing near the starting position."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    # Use root position deviation from origin as landing error
    land_err = torch.norm(asset.data.root_link_pos_w[:, :2], dim=1)
    phase_flag = command[:, 2]
    # Only reward after jump phase
    return torch.exp(-land_err) * (phase_flag == 1).float()


def flight_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
) -> torch.Tensor:
    """Reward being in flight (no feet on ground during jump phase)."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0  # [B, 4]
    in_flight = (~contact).all(dim=1).float()
    command = env.command_manager.get_command(command_name)
    phase_flag = command[:, 2]
    return in_flight * (phase_flag == 1).float()


# ---------------------------------------------------------------------------
# Spring jump specific
# ---------------------------------------------------------------------------


def spring_jump_before_setting(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward holding default crouch pose before spring jump."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    dev = torch.sum(torch.abs(joint_pos - default_joint_pos), dim=1)
    command = env.command_manager.get_command("spring_jump")
    phase_flag = command[:, 2]
    return torch.exp(-dev / 2) * (phase_flag == 0).float()


# ---------------------------------------------------------------------------
# Backflip specific
# ---------------------------------------------------------------------------


def backflip_before_setting(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward holding default crouch pose before backflip."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    dev = torch.sum(torch.abs(joint_pos - default_joint_pos), dim=1)
    command = env.command_manager.get_command("backflip")
    phase_flag = command[:, 2]
    return torch.exp(-dev / 4) * (phase_flag == 0).float()


def angle_y_velocity(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Reward positive pitch (y-axis) angular velocity for backflip rotation."""
    asset: Entity = env.scene["robot"]
    ang_vel_y = asset.data.root_link_ang_vel_b[:, 1]  # pitch
    command = env.command_manager.get_command(command_name)
    phase_flag = command[:, 2]
    # Stronger reward during flight
    contact_sensor: ContactSensor = env.scene["feet_ground_contact"]
    contact = contact_sensor.data.found > 0
    in_flight = (~contact).all(dim=1).float()
    has_jumped = (in_flight == 0).float()  # simplified: back on ground

    rew = (ang_vel_y > 0).float() * ang_vel_y * (phase_flag == 1).float() * (in_flight == 0).float()
    rew += 3 * (ang_vel_y > 0).float() * ang_vel_y * in_flight
    return torch.clip(rew, max=20.0)


def flight_height(
    env: ManagerBasedRlEnv,
    target_height: float,
    sensor_name: str,
) -> torch.Tensor:
    """Reward being at target height during flight phase."""
    asset: Entity = env.scene["robot"]
    base_z = asset.data.root_link_pos_w[:, 2]
    height_error = base_z - target_height
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0
    in_flight = (~contact).all(dim=1).float()
    return torch.exp(-torch.abs(height_error) * 5) * in_flight * 6


# ---------------------------------------------------------------------------
# Handstand velocity tracking rewards
# ---------------------------------------------------------------------------

# Tracking sigma from IsaacGym go2_handstand config (0.25)
_HANDSTAND_TRACKING_SIGMA = 0.25


def _handstand_quality(env, target_height: float = 0.08) -> torch.Tensor:
    """Gate always open: return 2.0 so that quality > 0.70 is always true.

    The quality-gating architecture was ported from IsaacGym, where it uses
    torch.mean() as a scalar gate across all envs. In that setup the gate
    always opens once the robot can stand at the right height (standing
    default pose is stable). With handstand PD defaults the robot starts
    inverted and unstable — the gate only opens for ~2-3% of env-time even
    after 1000+ iterations, starving the policy of tracking/shaping signal.

    Removing the gate lets all rewards contribute from step one, matching
    the effective behavior of IsaacGym where the standing start means most
    envs pass the gate early. If needed, gating can be re-introduced as a
    per-env curriculum once basic balance is learned.
    """
    asset: Entity = env.scene["robot"]
    base_z = asset.data.root_link_pos_w[:, 2]
    return torch.full_like(base_z, 2.0)


def _handstand_standing_command_mask(
    env: ManagerBasedRlEnv,
    command_name: str | None,
    moving_threshold: float,
) -> torch.Tensor:
    if command_name is None:
        return torch.zeros(env.num_envs, device=env.device)
    command = env.command_manager.get_command(command_name)
    lin_near_zero = torch.norm(command[:, :2], dim=1) < moving_threshold
    yaw_near_zero = torch.abs(command[:, 2]) < moving_threshold
    return torch.logical_and(lin_near_zero, yaw_near_zero).float()


def _handstand_moving_command_mask(
    env: ManagerBasedRlEnv,
    command_name: str | None,
    moving_threshold: float,
) -> torch.Tensor:
    if command_name is None:
        return torch.ones(env.num_envs, device=env.device)
    return 1.0 - _handstand_standing_command_mask(env, command_name, moving_threshold)


def handstand_tracking_lin_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    tracking_sigma: float = _HANDSTAND_TRACKING_SIGMA,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Linear velocity tracking for FRONT-paw handstand walking.

    In this pose body +x = world -z (head DOWN) and body +z = world +x
    (forward in world).  `imu_lin_vel` returns lin_vel in body frame, so
    forward-world = lin_vel_b[2].  cmd_x is desired forward speed.
    Mirrors IsaacGym GO2_Leggedstand `_reward_tracking_lin_vel`:
        x_error = (cmd_x - lin_vel_b[2])^2
        y_error = (cmd_y - lin_vel_b[1])^2
    """
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    lin_vel_b = asset.data.root_link_lin_vel_b  # [B, 3]
    x_error = torch.square(command[:, 0] - lin_vel_b[:, 2])
    y_error = torch.square(command[:, 1] - lin_vel_b[:, 1])
    quality = _handstand_quality(env, target_height)
    return torch.exp(-(x_error + y_error) / tracking_sigma) * (quality > 0.70).float()


def handstand_tracking_ang_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    tracking_sigma: float = _HANDSTAND_TRACKING_SIGMA,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Yaw tracking for FRONT-paw handstand walking.

    body +x = world -z (head DOWN), so world yaw rate ω_z = -ang_vel_b[0].
    Mirrors IsaacGym GO2_Leggedstand `_reward_tracking_ang_vel`:
        ang_error = (cmd_yaw + ang_vel_b[0])^2
    """
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    ang_vel_b = asset.data.root_link_ang_vel_b  # [B, 3]
    ang_vel_error = torch.square(command[:, 2] + ang_vel_b[:, 0])
    quality = _handstand_quality(env, target_height)
    return torch.exp(-ang_vel_error / tracking_sigma) * (quality > 0.70).float()


def handstand_tracking_lin_vel_zero(
    env: ManagerBasedRlEnv,
    command_name: str,
    tracking_sigma: float = _HANDSTAND_TRACKING_SIGMA,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Penalize linear velocity when the command is near zero (head-down handstand)."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    lin_vel_b = asset.data.root_link_lin_vel_b
    x_error = torch.square(command[:, 0] - lin_vel_b[:, 2])
    y_error = torch.square(command[:, 1] - lin_vel_b[:, 1])
    quality = _handstand_quality(env, target_height)
    cmd_near_zero = (torch.norm(command[:, :2], dim=1) < 0.1).float()
    return (x_error + y_error) * (quality > 0.70).float() * cmd_near_zero


def handstand_tracking_ang_vel_zero(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Penalize angular velocity when the command is near zero (head-down handstand)."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    ang_vel_b = asset.data.root_link_ang_vel_b
    ang_vel_error = torch.square(command[:, 2] + ang_vel_b[:, 0])
    quality = _handstand_quality(env, target_height)
    cmd_near_zero = (torch.abs(command[:, 2]) < 0.1).float()
    return ang_vel_error * (quality > 0.70).float() * cmd_near_zero


def handstand_tracking_lin_vel_soft_gate(
    env: ManagerBasedRlEnv,
    command_name: str,
    tracking_sigma: float = _HANDSTAND_TRACKING_SIGMA,
    target_gravity: tuple[float, float, float] = (1.0, 0.0, 0.0),
    orientation_sharpness: float = 2.0,
    base_height_target: float = 0.39,
    rear_foot_target_height: float = 0.56,
    rear_foot_lift_min: float = 0.08,
    rear_foot_indices: tuple[int, ...] = (2, 3),
    foot_site_names: tuple[str, ...] = ("FL", "FR", "HL", "HR"),
    sensor_name: str | None = None,
    stance_foot_indices: tuple[int, ...] | None = None,
    body_clearance_sensor_names: tuple[str, ...] = (),
    support_floor: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Linear velocity tracking multiplied by soft handstand quality."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    lin_vel_b = asset.data.root_link_lin_vel_b
    x_error = torch.square(command[:, 0] - lin_vel_b[:, 2])
    y_error = torch.square(command[:, 1] - lin_vel_b[:, 1])
    quality = _handstand_soft_quality(
        env,
        target_gravity=target_gravity,
        orientation_sharpness=orientation_sharpness,
        base_height_target=base_height_target,
        rear_foot_target_height=rear_foot_target_height,
        rear_foot_lift_min=rear_foot_lift_min,
        rear_foot_indices=rear_foot_indices,
        foot_site_names=foot_site_names,
        sensor_name=sensor_name,
        stance_foot_indices=stance_foot_indices,
        body_clearance_sensor_names=body_clearance_sensor_names,
        support_floor=support_floor,
        asset_cfg=asset_cfg,
    )
    return torch.exp(-(x_error + y_error) / tracking_sigma) * quality


def handstand_tracking_ang_vel_soft_gate(
    env: ManagerBasedRlEnv,
    command_name: str,
    tracking_sigma: float = _HANDSTAND_TRACKING_SIGMA,
    target_gravity: tuple[float, float, float] = (1.0, 0.0, 0.0),
    orientation_sharpness: float = 2.0,
    base_height_target: float = 0.39,
    rear_foot_target_height: float = 0.56,
    rear_foot_lift_min: float = 0.08,
    rear_foot_indices: tuple[int, ...] = (2, 3),
    foot_site_names: tuple[str, ...] = ("FL", "FR", "HL", "HR"),
    sensor_name: str | None = None,
    stance_foot_indices: tuple[int, ...] | None = None,
    body_clearance_sensor_names: tuple[str, ...] = (),
    support_floor: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Yaw tracking multiplied by soft handstand quality."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    ang_vel_b = asset.data.root_link_ang_vel_b
    ang_vel_error = torch.square(command[:, 2] + ang_vel_b[:, 0])
    quality = _handstand_soft_quality(
        env,
        target_gravity=target_gravity,
        orientation_sharpness=orientation_sharpness,
        base_height_target=base_height_target,
        rear_foot_target_height=rear_foot_target_height,
        rear_foot_lift_min=rear_foot_lift_min,
        rear_foot_indices=rear_foot_indices,
        foot_site_names=foot_site_names,
        sensor_name=sensor_name,
        stance_foot_indices=stance_foot_indices,
        body_clearance_sensor_names=body_clearance_sensor_names,
        support_floor=support_floor,
        asset_cfg=asset_cfg,
    )
    return torch.exp(-ang_vel_error / tracking_sigma) * quality


def handstand_tracking_lin_vel_zero_soft_gate(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_gravity: tuple[float, float, float] = (1.0, 0.0, 0.0),
    orientation_sharpness: float = 2.0,
    base_height_target: float = 0.39,
    rear_foot_target_height: float = 0.56,
    rear_foot_lift_min: float = 0.08,
    rear_foot_indices: tuple[int, ...] = (2, 3),
    foot_site_names: tuple[str, ...] = ("FL", "FR", "HL", "HR"),
    sensor_name: str | None = None,
    stance_foot_indices: tuple[int, ...] | None = None,
    body_clearance_sensor_names: tuple[str, ...] = (),
    support_floor: float = 0.05,
    moving_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Zero-command linear velocity penalty behind the same soft gate."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    lin_vel_b = asset.data.root_link_lin_vel_b
    x_error = torch.square(command[:, 0] - lin_vel_b[:, 2])
    y_error = torch.square(command[:, 1] - lin_vel_b[:, 1])
    quality = _handstand_soft_quality(
        env,
        target_gravity=target_gravity,
        orientation_sharpness=orientation_sharpness,
        base_height_target=base_height_target,
        rear_foot_target_height=rear_foot_target_height,
        rear_foot_lift_min=rear_foot_lift_min,
        rear_foot_indices=rear_foot_indices,
        foot_site_names=foot_site_names,
        sensor_name=sensor_name,
        stance_foot_indices=stance_foot_indices,
        body_clearance_sensor_names=body_clearance_sensor_names,
        support_floor=support_floor,
        asset_cfg=asset_cfg,
    )
    cmd_near_zero = (torch.norm(command[:, :2], dim=1) < moving_threshold).float()
    return (x_error + y_error) * quality * cmd_near_zero


def handstand_tracking_ang_vel_zero_soft_gate(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_gravity: tuple[float, float, float] = (1.0, 0.0, 0.0),
    orientation_sharpness: float = 2.0,
    base_height_target: float = 0.39,
    rear_foot_target_height: float = 0.56,
    rear_foot_lift_min: float = 0.08,
    rear_foot_indices: tuple[int, ...] = (2, 3),
    foot_site_names: tuple[str, ...] = ("FL", "FR", "HL", "HR"),
    sensor_name: str | None = None,
    stance_foot_indices: tuple[int, ...] | None = None,
    body_clearance_sensor_names: tuple[str, ...] = (),
    support_floor: float = 0.05,
    moving_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Zero-command yaw penalty behind the same soft gate."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    ang_vel_b = asset.data.root_link_ang_vel_b
    ang_vel_error = torch.square(command[:, 2] + ang_vel_b[:, 0])
    quality = _handstand_soft_quality(
        env,
        target_gravity=target_gravity,
        orientation_sharpness=orientation_sharpness,
        base_height_target=base_height_target,
        rear_foot_target_height=rear_foot_target_height,
        rear_foot_lift_min=rear_foot_lift_min,
        rear_foot_indices=rear_foot_indices,
        foot_site_names=foot_site_names,
        sensor_name=sensor_name,
        stance_foot_indices=stance_foot_indices,
        body_clearance_sensor_names=body_clearance_sensor_names,
        support_floor=support_floor,
        asset_cfg=asset_cfg,
    )
    cmd_near_zero = (torch.abs(command[:, 2]) < moving_threshold).float()
    return ang_vel_error * quality * cmd_near_zero


# ---------------------------------------------------------------------------
# Handstand shaping rewards
# ---------------------------------------------------------------------------


def handstand_stance_contact_mean(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_indices: tuple[int, ...],
) -> torch.Tensor:
    """Reward indexed stance feet staying in ground contact."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0
    selected = contact[:, list(foot_indices)]
    return selected.float().mean(dim=1)


def handstand_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_indices: tuple[int, ...],
    command_name: str | None = None,
    moving_threshold: float = 0.1,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Reward exactly one of the indexed STANCE feet in ground contact.

    Mirrors IsaacGym `_reward_contact` which fires when the alternating
    stance foot is on the ground.  For front-paw handstand walking the
    stance feet are FRONT, so pass `foot_indices=(0, 1)`.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0  # [B, 4]
    selected = contact[:, list(foot_indices)]
    n_contact = torch.sum(selected, dim=1)
    quality = _handstand_quality(env, target_height)
    moving_mask = _handstand_moving_command_mask(env, command_name, moving_threshold)
    return (n_contact == 1).float() * (quality > 0.70).float() * moving_mask


def handstand_moving_no_stance_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_indices: tuple[int, ...],
    command_name: str,
    moving_threshold: float = 0.1,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Penalize moving handstand frames with no indexed stance foot planted."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0
    selected = contact[:, list(foot_indices)]
    n_contact = torch.sum(selected, dim=1)
    quality = _handstand_quality(env, target_height)
    moving_mask = _handstand_moving_command_mask(env, command_name, moving_threshold)
    return (n_contact == 0).float() * (quality > 0.70).float() * moving_mask


def handstand_stance_air_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_indices: tuple[int, ...],
    target_height: float = 0.08,
) -> torch.Tensor:
    """Anti-hop: penalize handstand frames with all stance feet airborne.

    Unlike ``handstand_moving_no_stance_contact`` this term has no
    command-based gate and fires whether the policy is commanded to move
    or stand. The orientation gate keeps the term silent while the body
    is collapsing or not yet in handstand pose.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0
    selected = contact[:, list(foot_indices)]
    n_contact = torch.sum(selected, dim=1)
    quality = _handstand_quality(env, target_height)
    return (n_contact == 0).float() * (quality > 0.70).float()


def handstand_moving_double_stance_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_indices: tuple[int, ...],
    command_name: str,
    moving_threshold: float = 0.1,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Weakly penalize moving handstand frames with all indexed stance feet planted."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0
    selected = contact[:, list(foot_indices)]
    n_contact = torch.sum(selected, dim=1)
    quality = _handstand_quality(env, target_height)
    moving_mask = _handstand_moving_command_mask(env, command_name, moving_threshold)
    return (n_contact == len(foot_indices)).float() * (quality > 0.70).float() * moving_mask


class handstand_feet_air_time:
    """Reward indexed STANCE feet for long air-time → ground-contact strides.

    Mirrors IsaacGym `_reward_feet_air_time` (defined on `contact_foot_indices`,
    which is the FRONT pair for the front-paw handstand walking task).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.foot_indices = cfg.params["foot_indices"]
        self.num_feet = len(self.foot_indices)
        self.air_time = torch.zeros(env.num_envs, self.num_feet, device=env.device)
        self.last_contacts = torch.zeros(env.num_envs, self.num_feet, device=env.device, dtype=torch.bool)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        foot_indices: tuple[int, ...],
        command_name: str | None = None,
        moving_threshold: float = 0.1,
        target_height: float = 0.08,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        contact = contact_sensor.data.found > 0  # [B, 4]
        selected = contact[:, list(foot_indices)]

        contact_filt = torch.logical_or(selected, self.last_contacts)
        first_contact = (self.air_time > 0.0).float() * contact_filt.float()
        self.air_time += env.step_dt
        rew = torch.sum((self.air_time - 0.4) * first_contact, dim=1)
        self.air_time = self.air_time * (~contact_filt).float()

        self.last_contacts = selected
        quality = _handstand_quality(env, target_height)
        moving_mask = _handstand_moving_command_mask(env, command_name, moving_threshold)
        return rew * (quality > 0.70).float() * moving_mask

    def reset(self, env_ids: torch.Tensor) -> None:
        self.air_time[env_ids] = 0.0
        self.last_contacts[env_ids] = False


def handstand_feet_clearance(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    foot_indices: tuple[int, ...] = (2, 3),
    foot_site_names: tuple[str, ...] = ("FR", "FL", "RR", "RL"),
    target_foot_height: float = 0.06,
    cycle_time: float = 1.6,
    command_name: str | None = None,
    moving_threshold: float = 0.1,
    target_height: float = 0.08,
    sharpness: float = 10.0,
) -> torch.Tensor:
    """Sinusoidal indexed-foot clearance reward.

    Rewards the two indexed feet tracking a |sin(2π·phase)|·target_foot_height
    world-z target during the swing half of the gait cycle. For front-paw
    handstand walking, pass the front stance pair `(0, 1)`. `foot_site_names`
    must list all four foot sites in the same order used elsewhere in the env
    (Go2 default "FR/FL/RR/RL"; Lite3 uses "FL/FR/HL/HR").

    ``sharpness`` controls the per-foot reward profile: the reward is
    ``exp(-sharpness · |z_world - z_target|)`` so larger values demand
    tighter tracking. Default 10.0 gives ~0.37 at 10 cm error; 20.0 needs
    half the error for the same reward.
    """
    assert len(foot_indices) == 2, "handstand_feet_clearance expects exactly two feet"
    asset: Entity = env.scene[asset_cfg.name]
    site_ids, _ = asset.find_sites(foot_site_names)
    feet_z = asset.data.site_pos_w[:, site_ids, 2]  # [B, 4]
    selected_z = feet_z[:, list(foot_indices)]  # [B, 2]

    phase = (env.episode_length_buf * env.step_dt) % cycle_time / cycle_time
    target = torch.abs(torch.sin(2 * torch.pi * phase)) * target_foot_height
    swing_mask_0 = (phase >= 0.5).float()  # foot 0 swings in second half
    swing_mask_1 = (phase < 0.5).float()   # foot 1 swings in first half

    rew = torch.exp(-torch.abs(selected_z[:, 0] - target) * sharpness) * swing_mask_0
    rew += torch.exp(-torch.abs(selected_z[:, 1] - target) * sharpness) * swing_mask_1

    quality = _handstand_quality(env, target_height)
    moving_mask = _handstand_moving_command_mask(env, command_name, moving_threshold)
    return rew * (quality > 0.70).float() * moving_mask


def handstand_stance_contact_zero(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_indices: tuple[int, ...],
    command_name: str,
    moving_threshold: float = 0.1,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Reward both stance feet staying planted when command is zero."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0
    selected = contact[:, list(foot_indices)]
    n_contact = torch.sum(selected, dim=1)
    quality = _handstand_quality(env, target_height)
    standing_mask = _handstand_standing_command_mask(env, command_name, moving_threshold)
    return (n_contact == len(foot_indices)).float() * (quality > 0.70).float() * standing_mask


def handstand_joint_vel_zero(
    env: ManagerBasedRlEnv,
    command_name: str,
    moving_threshold: float = 0.1,
    target_height: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint motion when the commanded handstand velocity is zero."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    quality = _handstand_quality(env, target_height)
    standing_mask = _handstand_standing_command_mask(env, command_name, moving_threshold)
    return torch.sum(torch.square(joint_vel), dim=1) * (quality > 0.70).float() * standing_mask


def handstand_default_pos_reward(
    env: ManagerBasedRlEnv,
    desire_joint_angles: list[float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Exponential reward for matching desire angles on ALL 12 joints.

    Front 6 (FR/FL) drive the stance pose (thigh=-0.7, calf=-1.75) so the
    front legs straighten into a true handstand instead of collapsing into
    a kneeling pose; rear 6 (RL/RR) drive the swing rest pose
    (thigh=0.8, calf=-1.5).  Gated on `_handstand_quality` so the bonus
    only fires once the body is roughly in handstand orientation.
    """
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [B, 12]
    target = torch.tensor(desire_joint_angles, device=env.device, dtype=torch.float32)
    dev = torch.sum(torch.abs(joint_pos - target), dim=1)
    quality = _handstand_quality(env, target_height)
    return torch.exp(-dev) * (quality > 0.70).float()


def handstand_torques(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Penalize total actuator force."""
    asset: Entity = env.scene["robot"]
    force = asset.data.actuator_force
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(torch.abs(force), dim=1)


def dof_pos_limits(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint positions that approach soft limits."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    upper = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    out_of_limits = -torch.clamp(joint_pos - lower, max=0.0)
    out_of_limits += torch.clamp(joint_pos - upper, min=0.0)
    return torch.sum(out_of_limits, dim=1)
