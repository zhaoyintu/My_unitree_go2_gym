"""Lite3 front-paw handstand — pose-tracking variant.

Builds on `lite3_handstand_strict`.  The strict task converged to a
non-canonical handstand: feet airborne but joint pose drifted from the
geometric desire (front HipY=+0.283 / Knee=+2.0; HipX=0).  Diagnosis
from training logs at iter 6087:

  rew_default_pos_reward = 0.005   ← gated +5.0 term essentially never fires
  rew_default_hip_pos    = -0.17   ← HipX drifting from 0
  rew_handstand_feet_height_exp = 3.36   ← feet ARE up
  rew_handstand_feet_on_air = 0.33  ← but only 1/3 of steps

Root cause: `rew_hanstand > 0.78` requires |base_z - 0.39| < 0.025 m.
The policy's "half-handstand" sits at base_z ≈ 0.32-0.36, never crossing
the gate, so the desire-pose reward is dormant and the policy converges
to whatever shape maximises the (ungated) handstand_feet_height_exp.

Three fixes:

1. Looser gate (0.78 → 0.6) on the three reward terms most directly
   responsible for *pose tracking* — `default_pos_reward`,
   `tracking_lin_vel_zero`, `tracking_ang_vel_zero`.  Gate at 0.6 means
   |base_z - 0.39| < ~0.05 m, comfortably within the half-handstand band
   so the desire-pose constraint actually fires during training.

   Other gated rewards (feet_clearance, contact, feet_air_time, etc.)
   keep the 0.78 gate so they only fire in proper handstand — these
   reward "handstand mechanics" and we don't want the policy faking
   them in a half-pose.

2. Unconditional baseline on `default_pos_reward` (0.2× when gate off,
   1.0× when on).  Even before handstand is formed, the policy gets a
   small "this is the target shape" signal.  Without this, the policy
   has to discover desire pose through random exploration.

3. Bump `default_hip_pos` weight 5× (-0.1 → -0.5).  Strict's value let
   HipX drift to ~0.085 rad on average; with -0.5 the HipX→0 pull is
   strong enough to dominate over the policy's "tuck-for-balance"
   tendency that handstand_feet_height_exp accidentally rewards.

Logs save to `logs/lite3_handstand_pose/` (separate from strict).
"""
from legged_gym.envs.Lite3_Stand.Lite3_Handstand_Strict.Lite3_handstand_strict_Config import (
    Lite3Cfg_LeggedstandStrict,
    Lite3CfgPPO_LeggedstandStrict,
)


class Lite3Cfg_LeggedstandPose(Lite3Cfg_LeggedstandStrict):
    class rewards(Lite3Cfg_LeggedstandStrict.rewards):
        class scales(Lite3Cfg_LeggedstandStrict.rewards.scales):
            # ---- Stronger HipX→0 pull (always active) ------------------
            # Strict had -0.1; observed HipX still ~0.085 rad off centre.
            # Bump 5× so the always-on penalty competes with handstand
            # mechanics rewards that don't constrain HipX directly.
            default_hip_pos = -0.5

            # default_pos_reward weight stays at 5.0 from strict — the env
            # override adds the unconditional baseline + lower gate.


class Lite3CfgPPO_LeggedstandPose(Lite3CfgPPO_LeggedstandStrict):
    class runner(Lite3CfgPPO_LeggedstandStrict.runner):
        # CRITICAL: separate logs from `lite3_handstand_strict`.  rsl_rl
        # writes checkpoints to `logs/{experiment_name}/{run_name}/`, so
        # changing experiment_name keeps the two task variants' weights
        # in distinct folders — no chance of accidentally resuming pose
        # checkpoint from the strict run or vice versa.
        experiment_name = 'lite3_handstand_pose'
