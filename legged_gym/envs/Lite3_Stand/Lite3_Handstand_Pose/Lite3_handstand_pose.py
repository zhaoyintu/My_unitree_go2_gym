"""Lite3 handstand env with pose-tracking reward shape.

Inherits from `Lite3_legstand_strict` (so we keep the per-body
collision penalties and 12-joint default_pos_reward override) and
applies three changes targeted at "policy reaches half-handstand but
joint pose drifts from the geometric desire":

  1. `_reward_default_pos_reward`: unconditional 0.2× baseline + 0.8×
     gated bonus (vs strict's 0/1 step gate).  Gate threshold 0.6
     instead of 0.78 so the bonus actually fires at base_z ≈ 0.34-0.44 m
     (vs strict's ≈ 0.365-0.415 m window which the policy never enters
     consistently).
  2. `_reward_tracking_lin_vel_zero` / `_reward_tracking_ang_vel_zero`:
     gate threshold 0.6.  These shape "stand still under cmd=0" and
     directly address the high-frequency foot-tapping observed at
     cmd=0 in MuJoCo deploy — at strict's 0.78 gate the penalty was
     dormant during the half-handstand state where tapping happens.

The other gated rewards (feet_clearance, contact, feet_air_time,
ang_xz, symmetric_joints) keep the parent's 0.78 gate intentionally:
those terms reward "handstand mechanics" and we want them to fire
only in a true handstand, not encourage faking the gate to collect
the bonus.
"""
import torch

from legged_gym.envs.Lite3_Stand.Lite3_Handstand_Strict.Lite3_handstand_strict import (
    Lite3_legstand_strict,
)
from legged_gym.envs.Lite3_Stand.Lite3_Handstand_Pose.Lite3_handstand_pose_Config import (
    Lite3Cfg_LeggedstandPose,
    Lite3CfgPPO_LeggedstandPose,
)


# Gate threshold for pose-tracking rewards.
# rew_hanstand = exp(-|base_z - 0.39| * 10).  Threshold 0.6 ↔ |base_z - 0.39|
# < 0.051 m, i.e. base_z in [0.339, 0.441] m.  Strict's 0.78 ↔ ±0.025 m
# (base_z in [0.365, 0.415] m), which the half-handstand policy never
# entered consistently → gated rewards stayed dormant.
GATE_THRESHOLD = 0.6


class Lite3_legstand_pose(Lite3_legstand_strict):
    """Lite3 front-paw handstand with pose-tracking reward shape."""

    def __init__(self, cfg: Lite3Cfg_LeggedstandPose, sim_params, physics_engine,
                 sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ------------------------------------------------------------------
    # Override: looser gate + unconditional baseline.
    # ------------------------------------------------------------------
    # Strict version returned `exp(-err) * (gate)` — zero when gate off.
    # Here we keep the same exp-shape but multiply by `(0.2 + 0.8 * gate)`
    # so the policy gets a 0.2× wallpaper signal toward desire pose even
    # before handstand is formed, with a 5× boost once gate fires.  Cfg
    # weight stays at 5.0, so effective per-step reward range:
    #   gate ON, err=0: 5.0 * 1.0 * 1.0 = 5.0
    #   gate OFF, err=0: 5.0 * 1.0 * 0.2 = 1.0
    def _reward_default_pos_reward(self):
        err = self.dof_pos - self.descire_joint_pos        # [num_envs, 12]
        base = torch.exp(-torch.sum(torch.abs(err), dim=1))
        gate = (self.rew_hanstand > GATE_THRESHOLD).float()
        return base * (0.2 + 0.8 * gate)

    # ------------------------------------------------------------------
    # Override: same raw-error² shape as strict (so a negative weight
    # really penalises motion under cmd=0), but with the looser gate.
    # ------------------------------------------------------------------
    def _reward_tracking_lin_vel_zero(self):
        x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 2])
        y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return (x_error + y_error) * (
            self.rew_hanstand > GATE_THRESHOLD
        ) * (torch.norm(self.commands[:, :2], dim=-1) < 0.1)

    # ------------------------------------------------------------------
    # Override: parent's `_reward_tracking_ang_vel_zero` already returns
    # raw `error²` (correct sign with negative weight) — we just need to
    # swap its 0.78 gate for our looser one.
    # ------------------------------------------------------------------
    def _reward_tracking_ang_vel_zero(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return ang_vel_error * (
            self.rew_hanstand > GATE_THRESHOLD
        ) * (torch.abs(self.commands[:, 2]) < 0.1)
