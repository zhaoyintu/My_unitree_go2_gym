"""AMP runner config — extends RslRlOnPolicyRunnerCfg with AMP hyperparameters.

The AMP fields must be DATACLASS FIELDS (not ad-hoc attrs) so they survive
``asdict(cfg.agent)`` into the runner's ``train_cfg`` dict.  Tuple defaults use
``field(default_factory=...)`` to avoid mutable-default errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl.config import RslRlOnPolicyRunnerCfg


@dataclass
class RslRlAmpOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    # Path(s) to ahmp .npz motion files providing the expert AMP frames.
    amp_motion_files: tuple[str, ...] = field(default_factory=tuple)
    # Style-reward scale: r_style = coef * clamp(1 - 0.25*(D-1)^2, 0, inf).
    amp_reward_coef: float = 2.0
    # Blend with task reward: r = (1-lerp)*style + lerp*task.  Higher = more
    # task (finetune-friendly so the warm-started handstand is not wrecked).
    amp_task_reward_lerp: float = 0.5
    # Discriminator trunk hidden dims (rl_amp A1 default).
    amp_discr_hidden_dims: tuple[int, ...] = field(default_factory=lambda: (1024, 512))
    amp_num_preload_transitions: int = 2_000_000
    amp_replay_buffer_size: int = 1_000_000
    amp_grad_pen_lambda: float = 10.0
    amp_disc_weight_decay_trunk: float = 1.0e-3
    amp_disc_weight_decay_head: float = 1.0e-1
    amp_disc_learning_rate: float = 1.0e-3
