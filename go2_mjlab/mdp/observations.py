"""Custom observation functions for Unitree Go2 locomotion tasks."""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor


def gait_clock(
    env: ManagerBasedRlEnv,
    cycle_time: float = 0.5,
) -> torch.Tensor:
    """Sin/cos encoding of the gait phase clock."""
    phase = (env.episode_length_buf * env.step_dt) % cycle_time / cycle_time
    sin_phase = torch.sin(2 * torch.pi * phase)
    cos_phase = torch.cos(2 * torch.pi * phase)
    return torch.stack((sin_phase, cos_phase), dim=1)


def foot_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Binary foot contact mask from contact sensor."""
    sensor: ContactSensor = env.scene[sensor_name]
    return (sensor.data.found > 0).float()


def body_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Binary body contact indicator (e.g., trunk/head on ground)."""
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    if found.dim() > 1:
        return (found.sum(dim=-1) > 0).float()
    return (found > 0).float()
