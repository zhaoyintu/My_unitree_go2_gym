"""Lite3 front-paw handstand with stricter anti-kneeling guards.

A/B variant of `lite3_handstand` that aligns reward weights with the mjlab
`Mjlab-Lite3-Handstand` task.  The base task lets the policy converge to a
"knees on shins" shortcut because:
  * the combined `collision` term (-1.0) lumps THIGH and SHANK contacts
    together — SHANK touching ground is the canonical kneeling signal and
    needs heavier weight than THIGH;
  * the `default_pos_reward` only tracks the rear 6 joints (`dof_pos[:, 6:]`)
    so the front legs have no direct angle-tracking signal; geometric
    rewards alone admit multiple poses, including kneeling.

This variant matches mjlab's `Lite3_handstand` rewards verbatim:
  * splits `collision` into TORSO/THIGH/SHANK with weights -2.0/-1.0/-2.0;
  * `default_pos_reward` tracks all 12 joints (override in env file).

See `docs/superpowers/specs/2026-05-01-lite3-handstand-isaacgym-design.md`
for the base design.
"""
from legged_gym.envs.Lite3_Stand.Lite3_Handstand.Lite3_handstand_Config import (
    Lite3Cfg_Leggedstand,
    Lite3CfgPPO_Leggedstand,
)


class Lite3Cfg_LeggedstandStrict(Lite3Cfg_Leggedstand):
    class rewards(Lite3Cfg_Leggedstand.rewards):
        # All non-scale fields (tracking_sigma, soft_dof_pos_limit, etc.)
        # are inherited from the base.

        class scales(Lite3Cfg_Leggedstand.rewards.scales):
            # Disable the parent's combined collision penalty; replaced
            # below by per-body weights.
            collision = 0.0
            # Per-body collision penalties (mjlab-aligned).  SHANK is the
            # kneeling signal and gets the heaviest negative weight.
            thigh_collision = -1.0
            shank_collision = -2.0
            # Trunk contact: termination already covers the catastrophic
            # case but a reward-side penalty produces a stronger gradient
            # near the failure boundary.
            base_contact = -2.0
            # Bump default_pos_reward from 0.5 → 1.0 to match mjlab; the
            # env override expands its scope from rear-6 to all 12 joints
            # so the front legs are now actively driven toward
            # HipY=+0.283 / Knee=+2.0 once the gate fires.
            default_pos_reward = 1.0


class Lite3CfgPPO_LeggedstandStrict(Lite3CfgPPO_Leggedstand):
    class runner(Lite3CfgPPO_Leggedstand.runner):
        experiment_name = 'lite3_handstand_strict'
