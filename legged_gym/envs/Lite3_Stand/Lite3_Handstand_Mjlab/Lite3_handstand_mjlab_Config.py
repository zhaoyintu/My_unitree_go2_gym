"""Lite3 front-paw handstand — Mjlab-aligned variant.

Mirrors the reward design of the working `feat/lite3-mjlab` branch's
`Mjlab-Lite3-Handstand` task as closely as the IsaacGym/legged_gym
framework allows.  Built to address the failure mode of the strict /
pose variants: tucked half-handstand at base_z ≈ 0.34 m with HipX
adducted ~17° inward.

Key alignments with Mjlab (not exhaustive — see env class for full
override list):

1. `default_pos` weight −0.05 → −1.0 (20× stronger).
   Mjlab uses −1.0 on the always-on 12-joint pose penalty.  The
   IsaacGym base task's −0.05 gives ~−0.015/step·joint pull, which is
   trivially absorbed by the always-active `handstand_feet_height_exp`
   (+5.0).  Bumping to −1.0 makes HipX adduction strictly suboptimal:
   the policy can no longer cheat its way to rear-foot height by
   collapsing the front-leg support polygon.

2. New `alive = +1.0/step` baseline reward.
   Mjlab provides a constant survival bonus so the value function is
   anchored to a positive baseline during cold-start, before any
   handstand mechanics rewards activate.  Without this, IsaacGym's
   policy enters PPO updates with mostly-negative advantage signals
   from the penalty terms, biasing toward minimum-action collapse.

3. TORSO contact: remove from termination, keep as `base_contact = -2.0`
   penalty (already in strict).  Mjlab treats falls as recoverable
   within an episode, accumulating gate-boundary recovery experience
   rather than wiping the env.  Empirically this is what lets the
   policy learn the second-by-second balance corrections needed for a
   stable handstand.

4. Drop all `rew_hanstand > 0.78` gates from handstand-mechanics
   rewards (tracking_*, contact, feet_clearance, feet_air_time, ang_xz,
   symmetric_joints) and from the pose-tracking reward
   (default_pos_reward).  Mjlab's "quality" gate is hard-coded to
   always-open; per-env gating in IsaacGym creates a cold-start trap
   where the gate fires too rarely for the gradient to stabilize.
   See env file for the moving-mask substitution on gait-coupled rewards
   (contact / feet_air_time / feet_clearance) — they now fire only
   when |cmd_xy| > 0.1, matching Mjlab's behaviour and preventing the
   foot-tapping seen in deploy under cmd=0.

5. `tracking_*_zero` weight −1.5 → −0.2 (matches Mjlab).
   With the rew_hanstand gate dropped, the stillness penalty is now
   always-active when cmd is near zero.  At -1.5 this would dominate
   the reward landscape from step 1 (before the policy can even hold a
   handstand); reducing to Mjlab's -0.2 keeps the cmd=0 signal honest
   without overwhelming the cold-start exploration.

6. `default_pos_reward = 1.0` (matches Mjlab cfg weight).
   Was 5.0 in strict/pose with a per-env gate that produced an erratic
   gradient.  At 1.0 always-on, the per-step contribution is
   exp(−Σ|err|), which scales smoothly with pose accuracy and never
   falls to zero.  Mjlab additionally multiplies by quality=2.0 in the
   formula (constant), which we omit here — relying on cfg weight to
   set magnitude.  If the always-on signal turns out too weak in
   practice, bump to 2.0.

DR is NOT touched.  The user explicitly noted that mjlab's reduced DR
is hard to port cleanly; we keep IsaacGym's DR (inherited from strict)
to retain the sim2real coverage that is already validated.

Logs save to `logs/lite3_handstand_mjlab/`.
"""
from legged_gym.envs.Lite3_Stand.Lite3_Handstand_Strict.Lite3_handstand_strict_Config import (
    Lite3Cfg_LeggedstandStrict,
    Lite3CfgPPO_LeggedstandStrict,
)


class Lite3Cfg_LeggedstandMjlab(Lite3Cfg_LeggedstandStrict):

    class asset(Lite3Cfg_LeggedstandStrict.asset):
        # Mjlab parity: TORSO contact is a penalty (`base_contact = -2.0`),
        # not a termination.  Letting the policy survive a partial fall
        # gives it the recovery experience needed to learn balance
        # corrections near the gate boundary; resetting on every TORSO
        # touch wipes that signal away.
        terminate_after_contacts_on = []

    class rewards(Lite3Cfg_LeggedstandStrict.rewards):

        class scales(Lite3Cfg_LeggedstandStrict.rewards.scales):
            # ---- Always-on cold-start baseline (new term) ---------------
            # Mjlab provides +1.0/step survival bonus.  Anchors the value
            # function to a positive baseline so the early-iter advantage
            # signal isn't dominated by penalty terms.
            alive = 1.0

            # ---- Heavy pull toward desire pose (20× strict's value) -----
            # The single biggest mechanical difference between Mjlab (works)
            # and strict/pose (tucks).  Mjlab: -1.0; strict: -0.05.
            default_pos = -1.0

            # ---- Mjlab-aligned scale for always-on pose-shape reward ---
            # Was 5.0 in strict (with per-env gate); 1.0 always-on matches
            # the Mjlab cfg weight.  Effective per-step contribution is
            # 1.0 * exp(−Σ|err|), which is non-zero throughout training.
            default_pos_reward = 1.0

            # ---- Cmd=0 stillness back to Mjlab magnitude --------------
            # Was -1.5 in strict with per-env gate; with the gate removed
            # in the env, -1.5 would over-dominate.  Mjlab uses -0.2.
            tracking_lin_vel_zero = -0.2
            tracking_ang_vel_zero = -0.2

            # ---- Match Mjlab's HipX-specific pull (strict uses -0.1) ----
            # Mjlab uses -0.5 on the always-on Σ|HipX| term in addition to
            # the 12-joint default_pos.  The two terms compound: HipX
            # deviation costs both the per-joint default_pos (-1.0) and
            # the HipX-specific default_hip_pos (-0.5).  Strict's -0.1 was
            # too weak even with default_pos at -0.05.
            default_hip_pos = -0.5

            # All other scales (handstand_feet_height_exp = 5.0,
            # handstand_orientation = -1.0, base_height = 1.0,
            # symmetric_joints = -0.1, etc.) inherited from strict.


class Lite3CfgPPO_LeggedstandMjlab(Lite3CfgPPO_LeggedstandStrict):
    class runner(Lite3CfgPPO_LeggedstandStrict.runner):
        # Separate logs from `lite3_handstand_strict` / `lite3_handstand_pose`
        # so checkpoints from the three reward designs never cross-pollute.
        experiment_name = 'lite3_handstand_mjlab'
