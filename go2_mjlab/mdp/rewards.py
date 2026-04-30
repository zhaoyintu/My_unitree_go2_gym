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


class joint_acceleration:
    """Penalize joint accelerations computed via finite differences."""

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
        acc = (joint_vel - self.last_joint_vel) / env.step_dt
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
    """Penalize hip abduction deviation from zero (keep hips centered)."""
    asset: Entity = env.scene[asset_cfg.name]
    _, joint_names = asset.find_joints(asset_cfg.joint_names)
    hip_mask = torch.tensor(
        ["hip" in name for name in joint_names],
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
    """Penalize body x-axis linear velocity (vertical in handstand)."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.exp(-torch.abs(asset.data.root_link_lin_vel_b[:, 0]) * 10.0)


def ang_vel_xy_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize body y/z (pitch/yaw) angular velocity."""
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
) -> torch.Tensor:
    """Reward all 4 feet being off the ground."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0
    return (~contact).float().prod(dim=1)


def handstand_feet_height(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward feet being above the body (headstand height)."""
    asset: Entity = env.scene[asset_cfg.name]
    trunk_z = asset.data.root_link_pos_w[:, 2]
    foot_site_names = ("FR", "FL", "RR", "RL")
    site_ids, _ = asset.find_sites(foot_site_names)
    feet_z = asset.data.site_pos_w[:, site_ids, 2]  # [B, 4]
    feet_above_trunk = feet_z - trunk_z.unsqueeze(1)
    error = torch.abs(feet_above_trunk - target_height).sum(dim=1)
    return torch.exp(-error * 10)


def ang_xz_penalty(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Penalize roll (x) angular velocity during handstand."""
    asset: Entity = env.scene["robot"]
    base_ang_vel = asset.data.root_link_ang_vel_b
    return torch.abs(base_ang_vel[:, 0]) + torch.abs(base_ang_vel[:, 2])


def symmetric_joints(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward left-right joint symmetry."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [B, 12]
    dof = joint_pos.view(env.num_envs, 4, 3)  # [B, 4 legs, 3 joints]
    # Negate right-side hip abduction to match left side
    dof[:, 1, 0] = -dof[:, 1, 0]
    dof[:, 3, 0] = -dof[:, 3, 0]
    err = torch.sum(torch.abs(dof[:, 0, :] - dof[:, 1, :]), dim=1)
    return err


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
