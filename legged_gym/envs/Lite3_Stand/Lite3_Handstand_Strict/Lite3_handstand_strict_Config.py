"""Lite3 front-paw handstand — strict variant.

Differs from the base `lite3_handstand` task in three orthogonal axes,
all aimed at producing a policy robust enough to clear the MuJoCo
sim2real validation step (and ultimately survive deployment on real
hardware):

1. Anti-kneeling rewards (mjlab-aligned).
   * Splits the combined `collision` (-1.0) term into per-body weights:
     TORSO=-2.0, THIGH=-1.0, SHANK=-2.0 — SHANK touching the ground is
     the canonical "kneeling on shins" signal so it gets the heaviest
     penalty.
   * The env file overrides `_reward_default_pos_reward` to track all
     12 joints (parent only tracks the rear 6), and we crank the weight
     1.0 → 5.0 so the policy gets a strong gradient toward the desire
     pose (HipY=+0.283 / Knee=+2.0 front, -0.8 / +1.6 rear) instead of
     finding a shortcut handstand with tucked-in front knees.

2. Tighter cmd=0 stillness (`tracking_*_zero` -0.2 → -1.5).
   * The base task's tracking_lin_vel_zero / tracking_ang_vel_zero have
     a 12× weaker weight than the positive tracking_* terms, so the
     policy ends up drifting / spinning when no command is given.
   * Bumping these to -1.5 makes "stand still under cmd=0" a first-class
     objective.

3. Wider domain randomisation.
   * Strict's first iteration (and the base task) trained the policy
     well in IsaacGym but it overshoots and falls in MuJoCo because it
     learned to rely on PhysX's specific joint-stop reflex behaviour.
   * Wider DR — joint_armature, friction, joint_friction/damping,
     motor_zero_offset — forces the policy to find a control law that
     works across a *band* of physics rather than one specific PhysX
     setting.  Goal: any policy that makes it through training is
     robust enough to transfer to MuJoCo without a sim2sim crash.

Note: `action_scale` and `hip_action_scale` are unchanged.  Reducing
them would invalidate any partially-trained checkpoint, forcing a fresh
restart.  Resuming an existing strict run with this updated config
remains valid — the policy will gradually shift to satisfy the new
reward shape and DR distribution.
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
            # ---- 1. Anti-kneeling: per-body collision splits ----------
            # Disable the parent's combined collision penalty; replaced
            # below by per-body weights.
            collision = 0.0
            thigh_collision = -1.0
            shank_collision = -2.0
            # Trunk contact: termination already covers the catastrophic
            # case but a reward-side penalty produces a stronger gradient
            # near the failure boundary.
            base_contact = -2.0

            # ---- 1. Anti-kneeling: stronger desire-pose tracking -------
            # Was 0.5 (rear-only via parent); env override extends it to
            # all 12 joints, and we bump the weight 1.0 → 5.0 so the
            # policy is meaningfully driven toward the geometric desire
            # (HipY=+0.283 / Knee=+2.0 front, -0.8 / +1.6 rear).  At
            # weight 1.0 the term contributed ~0.001 to per-episode
            # reward — effectively zero gradient.  At 5.0 it should
            # dominate over the negative `default_pos = -0.05` pull-back.
            default_pos_reward = 5.0

            # ---- 2. cmd=0 stillness ------------------------------------
            # Was -0.2 each.  Positive tracking_* are at +2.5, so the
            # 12.5× imbalance taught the policy to keep moving; bump to
            # -1.5 each.
            tracking_lin_vel_zero = -1.5
            tracking_ang_vel_zero = -1.5

    class domain_rand(Lite3Cfg_Leggedstand.domain_rand):
        # ---- 3. Moderately wider DR for sim2real robustness -----------
        # Initial strict v2 widened DR aggressively (friction 2×,
        # joint_damping 2.5× upper bound), which more than doubled the
        # iteration count to first-handstand and left action_std climbing
        # to 2.79 at iter 16k without converging.  Pull the most
        # aggressive expansions back partway — still wider than the base
        # but not drastically.
        joint_armature_range = [0.005, 0.020]    # was 0.025; base 0.015
        joint_friction_range = [0.01, 0.3]       # was 0.4;   base 0.2
        joint_damping_range = [0.0, 0.3]         # was 0.5;   base 0.2

        # Foot-ground friction: keep modest expansion.  Real surfaces
        # vary, but [0.15, 1.0] already covers most cases the dog will
        # see; previous [0.1, 1.5] made the policy hedge for a 15× spread
        # which is overkill.
        friction_range = [0.15, 1.0]             # was [0.1, 1.5]; base [0.2, 0.8]

        # Motor zero offset, PD gain multipliers, link mass: keep at the
        # slightly wider values — these don't slow training as much.
        motor_zero_offset_range = [-0.05, 0.05]   # base ±0.035
        stiffness_multiplier_range = [0.85, 1.15] # base ±10%
        damping_multiplier_range = [0.85, 1.15]
        multiplied_link_mass_range = [0.85, 1.15] # base [0.9, 1.1]


class Lite3CfgPPO_LeggedstandStrict(Lite3CfgPPO_Leggedstand):
    class runner(Lite3CfgPPO_Leggedstand.runner):
        experiment_name = 'lite3_handstand_strict'
