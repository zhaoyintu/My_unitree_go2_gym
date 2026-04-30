"""Custom termination functions for Unitree Go2 locomotion tasks."""

import math

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor


def base_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Terminate when the trunk or head touches the ground."""
    sensor: ContactSensor = env.scene[sensor_name]
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
