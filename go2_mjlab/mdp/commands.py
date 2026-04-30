"""Custom command terms for Unitree Go2 specialized tasks.

Phase-based commands for spring jump and backflip tasks. The command is a 3-element
vector per environment: [x_vel_direction, 0, phase_flag] where phase_flag=0 means
setting/crouch phase and phase_flag=1 means jump/flip phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class PhaseCommand(CommandTerm):
    """Two-phase command: setting phase (0) then action phase (1).

    At resample time the command resets to phase 0. After a random delay the
    phase flag transitions to 1, triggering the action (jump/flip).
    """

    cfg: PhaseCommandCfg

    def __init__(self, cfg: PhaseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._cmd = torch.zeros(self.num_envs, 3, device=self.device)
        # Frame at which phase switches from 0 to 1
        self._switch_frame = torch.zeros(self.num_envs, device=self.device)
        self._resample_command(torch.arange(self.num_envs, device=self.device))

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._cmd[env_ids, :] = 0.0
        self._cmd[env_ids, 2] = 0.0  # phase_flag = 0 (setting)
        if self.cfg.x_vel_range is not None:
            self._cmd[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
                *self.cfg.x_vel_range
            )
        # Schedule phase switch at a random future frame
        r = torch.randint(
            self.cfg.switch_frame_range[0],
            self.cfg.switch_frame_range[1] + 1,
            (len(env_ids),),
            device=self.device,
        )
        self._switch_frame[env_ids] = r.float()

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        # Transition to phase 1 when episode_length reaches the switch frame
        switch_now = (self._env.episode_length_buf.float() >= self._switch_frame) & (self._cmd[:, 2] == 0)
        self._cmd[switch_now, 2] = 1.0


@dataclass(kw_only=True)
class PhaseCommandCfg(CommandTermCfg):
    """Configuration for a two-phase command term.

    x_vel_range: range for random x-velocity direction in phase 0.
        Set to None for tasks that don't need directional commands (backflip).
    switch_frame_range: frame range at which phase flag transitions from 0 to 1.
    """

    x_vel_range: tuple[float, float] | None = None
    switch_frame_range: tuple[int, int] = (50, 60)

    def build(self, env: ManagerBasedRlEnv) -> PhaseCommand:
        return PhaseCommand(self, env)
