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
    """Reward body x-axis linear velocity (vertical pumping in handstand).

    Matches IsaacGym _reward_lin_vel_z: torch.square(base_lin_vel[:, 0]).
    Despite the name, this is a REWARD for body-x motion — it encourages
    the dynamic up/down motion needed to maintain handstand balance.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 0])


def ang_vel_xy_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize body roll (x) and pitch (y) angular velocity.

    Matches IsaacGym _reward_ang_vel_xy: exp(-norm(|ang_vel[:, :2]|)).
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.exp(-torch.norm(torch.abs(asset.data.root_link_ang_vel_b[:, :2]), dim=1))


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


# ---------------------------------------------------------------------------
# Handstand velocity tracking rewards
# ---------------------------------------------------------------------------

# Tracking sigma from IsaacGym go2_handstand config (0.25)
_HANDSTAND_TRACKING_SIGMA = 0.25


def _handstand_quality(env, target_height: float = 0.08) -> torch.Tensor:
    """Per-env handstand quality gate: exp(-|base_z - target| * 5).

    Returns shape [B]. Each env is gated independently — when an env's
    quality > 0.70, velocity tracking and shaping rewards activate for
    that env. This avoids the scalar-mean trap where freshly reset envs
    (base_z ≈ 0.40) drag down the mean and prevent gating for ALL envs.

    Target 0.08 matches the physically achievable equilibrium: with Kp≈40
    and 7kg robot mass, the front thighs sag ~0.17 rad under gravity,
    dropping the base from ~0.38m (kinematic) to ~0.07-0.09m.
    """
    asset: Entity = env.scene["robot"]
    base_z = asset.data.root_link_pos_w[:, 2]
    return torch.exp(-torch.abs(base_z - target_height) * 5)


def handstand_tracking_lin_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    tracking_sigma: float = _HANDSTAND_TRACKING_SIGMA,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Linear velocity tracking in handstand body frame.

    In handstand: body x = world up, body z = -world x.
    cmd_x tracks body z-axis velocity, cmd_y tracks body y-axis velocity.
    Gated by handstand quality > 70%.
    """
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    lin_vel_b = asset.data.root_link_lin_vel_b  # [B, 3]
    x_error = torch.square(command[:, 0] + lin_vel_b[:, 2])
    y_error = torch.square(command[:, 1] - lin_vel_b[:, 1])
    quality = _handstand_quality(env, target_height)
    return torch.exp(-(x_error + y_error) / tracking_sigma) * (quality > 0.70).float()


def handstand_tracking_ang_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    tracking_sigma: float = _HANDSTAND_TRACKING_SIGMA,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Angular velocity tracking in handstand.

    cmd_yaw tracks body x-axis angular velocity.
    Gated by handstand quality > 70%.
    """
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    ang_vel_b = asset.data.root_link_ang_vel_b  # [B, 3]
    ang_vel_error = torch.square(command[:, 2] - ang_vel_b[:, 0])
    quality = _handstand_quality(env, target_height)
    return torch.exp(-ang_vel_error / tracking_sigma) * (quality > 0.70).float()


def handstand_tracking_lin_vel_zero(
    env: ManagerBasedRlEnv,
    command_name: str,
    tracking_sigma: float = _HANDSTAND_TRACKING_SIGMA,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Penalize linear velocity when the command is near zero. Gated by handstand quality."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    lin_vel_b = asset.data.root_link_lin_vel_b
    x_error = torch.square(command[:, 0] + lin_vel_b[:, 2])
    y_error = torch.square(command[:, 1] - lin_vel_b[:, 1])
    quality = _handstand_quality(env, target_height)
    cmd_near_zero = (torch.norm(command[:, :2], dim=1) < 0.1).float()
    return torch.exp(-(x_error + y_error) / tracking_sigma) * (quality > 0.70).float() * cmd_near_zero


def handstand_tracking_ang_vel_zero(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Penalize angular velocity when the command is near zero. Gated by handstand quality."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    ang_vel_b = asset.data.root_link_ang_vel_b
    ang_vel_error = torch.square(command[:, 2] - ang_vel_b[:, 0])
    quality = _handstand_quality(env, target_height)
    cmd_near_zero = (torch.abs(command[:, 2]) < 0.1).float()
    return ang_vel_error * (quality > 0.70).float() * cmd_near_zero


# ---------------------------------------------------------------------------
# Handstand shaping rewards
# ---------------------------------------------------------------------------


def handstand_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_indices: tuple[int, ...],
    target_height: float = 0.08,
) -> torch.Tensor:
    """Reward exactly one rear foot in contact during handstand. Gated by quality."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0  # [B, 4]
    rear_contact = contact[:, list(foot_indices)]  # rear feet (RL, RR = indices 2, 3)
    n_contact = torch.sum(rear_contact, dim=1)
    quality = _handstand_quality(env, target_height)
    return (n_contact == 1).float() * (quality > 0.70).float()


class handstand_feet_air_time:
    """Reward rear feet (RL, RR) staying in the air during handstand.

    Tracks per-foot air time and rewards on first ground contact.
    Matches IsaacGym _reward_feet_air_time.
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
        target_height: float = 0.08,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        contact = contact_sensor.data.found > 0  # [B, 4]
        rear_contact = contact[:, list(foot_indices)]

        contact_filt = torch.logical_or(rear_contact, self.last_contacts)
        first_contact = (self.air_time > 0.0).float() * contact_filt.float()
        self.air_time += env.step_dt
        rew = torch.sum((self.air_time - 0.4) * first_contact, dim=1)
        self.air_time = self.air_time * (~contact_filt).float()

        self.last_contacts = rear_contact
        quality = _handstand_quality(env, target_height)
        return rew * (quality > 0.70).float()

    def reset(self, env_ids: torch.Tensor) -> None:
        self.air_time[env_ids] = 0.0
        self.last_contacts[env_ids] = False


def handstand_feet_clearance(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    target_foot_height: float = 0.06,
    cycle_time: float = 1.6,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Sinusoidal foot clearance reward for front feet during handstand.

    Rewards the two front feet (indices 0, 1: FR, FL) tracking a sinusoidal height
    target during their swing phase. Gated by handstand quality > 70%.
    """
    asset: Entity = env.scene[asset_cfg.name]
    site_ids, _ = asset.find_sites(("FR", "FL", "RR", "RL"))
    feet_z = asset.data.site_pos_w[:, site_ids, 2]  # [B, 4]

    phase = (env.episode_length_buf * env.step_dt) % cycle_time / cycle_time
    # Front feet swing when stance_phase_0 = False (phase > 0.5)
    swing_mask = phase > 0.5
    target = torch.abs(torch.sin(2 * torch.pi * phase)) * target_foot_height

    # Front feet only: indices 0, 1
    front_feet_z = feet_z[:, :2]
    rew = torch.exp(-torch.abs(front_feet_z[:, 0] - target) * 10) * swing_mask.float()
    rew += torch.exp(-torch.abs(front_feet_z[:, 1] - target) * 10) * swing_mask.float()

    quality = _handstand_quality(env, target_height)
    return rew * (quality > 0.70).float()


def handstand_default_pos_reward(
    env: ManagerBasedRlEnv,
    desire_joint_angles: list[float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    target_height: float = 0.08,
) -> torch.Tensor:
    """Exponential reward for matching desired joint angles, front 6 joints only.

    Gated by handstand quality > 70%.
    """
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [B, 12]
    target = torch.tensor(desire_joint_angles, device=env.device, dtype=torch.float32)
    # Front 6 joints (FR + FL: 3 joints each)
    front_dev = torch.sum(torch.abs(joint_pos[:, :6] - target[:6]), dim=1)
    quality = _handstand_quality(env, target_height)
    return torch.exp(-front_dev) * (quality > 0.70).float()


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
