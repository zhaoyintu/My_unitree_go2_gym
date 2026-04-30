"""Custom termination functions for Unitree Go2 locomotion tasks."""

import math

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor


def base_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 0.0,
) -> torch.Tensor:
    """Terminate when trunk/head touches the ground.

    When force_threshold > 0, uses net contact force (N). Otherwise uses
    binary contact detection (any touch triggers termination).
    """
    sensor: ContactSensor = env.scene[sensor_name]
    if force_threshold > 0 and sensor.data.force is not None:
        # force shape is (B, num_slots, 3) — take L2 norm per slot,
        # then check if any slot exceeds threshold (matching IsaacGym).
        force_mag = torch.norm(sensor.data.force, dim=-1)  # (B, num_slots)
        return (force_mag > force_threshold).any(dim=-1)
    assert sensor.data.found is not None
    return sensor.data.found.sum(dim=-1) > 0


def handstand_fell(
    env: ManagerBasedRlEnv,
    limit_angle: float,
) -> torch.Tensor:
    """Terminate when the robot falls during handstand (orientation exceeds limit)."""
    asset = env.scene["robot"]
    projected_gravity = asset.data.projected_gravity_b
    target = torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=torch.float32)
    cos_dist = torch.sum(projected_gravity * target, dim=1)
    return torch.acos(torch.clamp(cos_dist, -1.0, 1.0)) > limit_angle
