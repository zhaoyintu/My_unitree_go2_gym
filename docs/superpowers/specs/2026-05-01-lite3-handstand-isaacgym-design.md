# Lite3 Handstand — IsaacGym Task Design

**Date:** 2026-05-01
**Author:** zhaoyintu (with Claude)
**Status:** Approved — proceed to implementation plan

## Goal

Add a new IsaacGym (legged_gym) training task `lite3_handstand` that trains the
DeepRobotics Lite3 quadruped to do a **front-paw stand** (head down, front legs
supporting the body, hind end up). This complements the existing mjlab Lite3
handstand task and serves as a cross-framework A/B reference: both implement
the same task on the same robot, so divergent training behaviour points to a
framework- or implementation-level cause rather than a robot-or-task issue.

## Naming

The existing in-repo tasks use a confusing inverted convention:
- `GO2_Stand/GO2_Handstand/` — actually rear-leg stand
- `GO2_Stand/GO2_Leggedstand/` — actually front-paw stand (the "real" handstand)

This task uses the **corrected** convention: `lite3_handstand` means front-paw
stand. The implementation file lives at
`legged_gym/envs/Lite3_Stand/Lite3_Handstand/` (overwriting the existing
empty-shell directory that contained only stale `.pyc` files).

## Reference Sources

The design is a synthesis of three existing artifacts:

1. **`legged_gym/envs/GO2_Stand/GO2_Leggedstand/`** (in this repo) — provides
   the entire training recipe: 25-term reward set, observation space, command
   sampling, DR configuration, PPO hyperparameters. This is a known-working
   IsaacGym front-paw stand task.
2. **`go2_rl_gym-recovery` repo, `legged_gym/envs/lite3/lite3_config.py`
   (`Lite3Cfg` base class)** — provides Lite3-specific control conventions:
   - PD gains (Kp=30, Kd=1) — distinct from GO2_Leggedstand's Kp=40
   - Joint declaration order (FL/FR/HL/HR × HipX/HipY/Knee)
   - Action scale split (action_scale=0.25, hip_action_scale=0.125)
   - URDF/mesh paths, body-name conventions (TORSO/THIGH/SHANK/FOOT)
3. **`go2_mjlab/config/lite3_handstand/` and `go2_mjlab/robots/lite3_constants.py`** (in this repo) — provides
   the geometrically-derived Lite3 front-paw target pose. The mjlab side
   already worked through the constraint that Lite3's HipY upper limit
   (+0.314 rad) does not allow Go2's sign-flipped stance pose (HipY=+0.7).

## Scope

In scope:
- New IsaacGym task `lite3_handstand` registered in `legged_gym/envs/__init__.py`
- New config class `Lite3Cfg_Leggedstand` and PPO config `Lite3CfgPPO_Leggedstand`
- New env class `Lite3_legstand` (subclass of `BaseTask`)
- Lite3 URDF + mesh assets copied into `resources/robots/lite3/`
- Static import sanity check (no training run)
- Train command documented in the implementation plan; no README change in this task

Out of scope:
- Running training (left to user)
- Tuning differences from this baseline
- Sim-to-real deployment changes
- Cleaning up the stale `.pyc` files under `Lite3_Stand/Lite3_Leggedstand/`
  (left in place, harmless because not registered)

## File-Level Plan

```
My_unitree_go2_gym/
├── legged_gym/envs/
│   ├── Lite3_Stand/Lite3_Handstand/
│   │   ├── Lite3_handstand.py           # NEW — env class (port of Go2_legstand.py)
│   │   └── Lite3_handstand_Config.py    # NEW — config (port of Go2_legstand_Config.py)
│   └── __init__.py                      # MOD — add task_registry.register("lite3_handstand", ...)
├── resources/robots/lite3/              # NEW — copied from go2_rl_gym-recovery
│   ├── urdf/Lite3.urdf
│   └── meshes/{torso,hip,thigh,shank}.STL
└── docs/superpowers/specs/2026-05-01-lite3-handstand-isaacgym-design.md   # this file
```

## Config Spec (`Lite3_handstand_Config.py`)

### From `lite3_push_recovery` (Lite3 control consistency)

```python
class control(LeggedRobotCfg.control):
    stiffness = {'joint': 30.0}
    damping   = {'joint': 1.0}
    action_scale = 0.25
    hip_action_scale = 0.125    # HipX uses smaller scale; consumed by env

class asset(LeggedRobotCfg.asset):
    file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/lite3/urdf/Lite3.urdf'
    name = "lite3"
    foot_name = "FOOT"
    penalize_contacts_on       = ["THIGH", "SHANK"]
    terminate_after_contacts_on= ["TORSO"]
    feet_name_reward           = ["HL_FOOT", "HR_FOOT"]   # rear → in air
    contact_foot               = ["FL_FOOT", "FR_FOOT"]   # front → on ground
```

### Init / desire pose (from mjlab port — geometrically derived)

```python
class init_state(LeggedRobotCfg.init_state):
    pos = [0.0, 0.0, 0.30]
    default_joint_angles = {
        # 4-paw upright (sign-flipped vs Go2 because Lite3 axes are -x,-y,-y)
        '.*L_HipX_joint': 0.1,    # FL_HipX, HL_HipX
        '.*R_HipX_joint': -0.1,   # FR_HipX, HR_HipX
        '.*HipY_joint':  -0.8,
        '.*Knee_joint':   1.6,
    }
    # Front-paw stand target (head down, body inverted)
    descire_joint_angles = {
        # Front legs (FL, FR) — support contact
        'FL_HipX_joint': 0.0,  'FR_HipX_joint': 0.0,
        'FL_HipY_joint': 0.283, 'FR_HipY_joint': 0.283,  # 90% of upper limit
        'FL_Knee_joint': 2.0,   'FR_Knee_joint': 2.0,    # tight bend so elbow forms an acute physical angle
        # Rear legs (HL, HR) — held up in air
        'HL_HipX_joint': 0.0,  'HR_HipX_joint': 0.0,
        'HL_HipY_joint': -0.8, 'HR_HipY_joint': -0.8,
        'HL_Knee_joint': 1.6,  'HR_Knee_joint': 1.6,
    }

# In rewards/env consumption:
base_height_target = 0.39   # trunk world-z when stably inverted
rear_foot_target_z = 0.56   # used by feet_clearance & handstand_feet_height_exp
```

### From GO2_Leggedstand (verbatim)

- All 25 reward term scales (`tracking_lin_vel=2.5`, `tracking_ang_vel=2.5`,
  `tracking_lin_vel_zero=-0.2`, `tracking_ang_vel_zero=-0.2`, `lin_vel_z=0.2`,
  `ang_vel_xy=0.2`, `handstand_orientation=-1`, `torques=-2e-4`,
  `dof_acc=-2.5e-4`, `base_height=1.0`, `handstand_feet_on_air=0.4`,
  `collision=-1`, `action_rate=-0.05`, `default_pos=-0.05`,
  `default_hip_pos=-0.1`, `feet_clearance=0.4`, `ang_xz=-0.5`, `contact=0.3`,
  `feet_air_time=2.0`, `symmetric_joints=-0.1`, `handstand_feet_height_exp=5.0`,
  `default_pos_reward=0.5`, plus zero-scale terms `termination`, `dof_vel`,
  `feet_stumble`)
- Command ranges: `lin_vel_x=[-0.4, 0.4]`, `lin_vel_y=[0, 0]`,
  `ang_vel_yaw=[-0.4, 0.4]`, with 20% zero-cmd / 5% zero-xy sampling
- Domain randomization (proportional, so the ranges below stay the same even
  though our PD baseline is 30/1 instead of 40/1):
  - `stiffness_multiplier_range=[0.9, 1.1]`, `damping_multiplier_range=[0.9, 1.1]`
  - `motor_zero_offset_range=[-0.035, 0.035]`
  - `added_mass_range=[-1, 2]`, `link_mass_range=[0.9, 1.1]`
  - `friction_range=[0.2, 0.8]`, `restitution_range=[0.0, 0.3]`
  - `joint_friction_range=[0.01, 0.2]`, `joint_damping_range=[0.0, 0.2]`,
    `armature_range=[0.005, 0.015]`
  - `motor_obs_latency=[1, 3]`, `imu_obs_latency=[1, 3]`,
    `action_latency=[1, 3]` (sim steps)
  - `base_com_displacement=[-0.05, 0.05]` per axis
- Observation space (carry over the layout from `Go2_legstand.py:205-237`):
  - Policy obs: 48 dims = stand_command(2 zero + 2) + base_ang_vel(3) +
    projected_gravity(3) + commands[:3](3) + dof_pos_error(12) + dof_vel(12) +
    actions(12)
  - Critic obs: 89 dims = base_lin_vel(3) + policy obs(48) + DR info(31) +
    contact_mask(4)
- PPO (`Lite3CfgPPO_Leggedstand`) — copy `GO2CfgPPO_Leggedstand` verbatim
  (network sizes, lr, clip_param, entropy_coef, etc.)
- `experiment_name = "lite3_handstand"`

## Env Spec (`Lite3_handstand.py`)

Copy `Go2_legstand.py` and apply these mechanical substitutions throughout:

| Where | Go2 | Lite3 |
|---|---|---|
| Joint name patterns | `_hip_joint`, `_thigh_joint`, `_calf_joint` | `_HipX_joint`, `_HipY_joint`, `_Knee_joint` |
| Hip-pattern matching (e.g. `default_hip_pos` reward) | `"hip" in name` | `"HipX" in name` (or `name.lower().startswith(...)`) — Lite3 uses capital `H` |
| Foot body names | `FL_foot`, `FR_foot`, `RL_foot`, `RR_foot` | `FL_FOOT`, `FR_FOOT`, `HL_FOOT`, `HR_FOOT` |
| Termination body | `base` | `TORSO` |
| Penalty contact bodies | `thigh`, `calf` | `THIGH`, `SHANK` |
| Class names | `Go2_legstand`, `GO2Cfg_Leggedstand` | `Lite3_legstand`, `Lite3Cfg_Leggedstand` |
| `descire_joint_angles` consumption | Go2 dict | Lite3 dict (above) |
| `base_height_target` | 0.47 | 0.39 |
| `feet_clearance` target | (Go2 value) | 0.56 (rear foot world-z target) |

For `hip_action_scale`: in `_compute_torques` (or wherever `action_scale` is
applied), produce a per-joint scale array of length 12 — `0.125` for HipX
indices `[0, 3, 6, 9]`, `0.25` for the rest. Build this once in `_init_buffers`
from the joint-name list to avoid hardcoded indices, then `actions_scaled =
actions * self.action_scale_per_joint`.

## Task Registration

In `legged_gym/envs/__init__.py`:
```python
from legged_gym.envs.Lite3_Stand.Lite3_Handstand.Lite3_handstand import Lite3_legstand
from legged_gym.envs.Lite3_Stand.Lite3_Handstand.Lite3_handstand_Config import (
    Lite3Cfg_Leggedstand, Lite3CfgPPO_Leggedstand,
)
task_registry.register("lite3_handstand", Lite3_legstand,
                       Lite3Cfg_Leggedstand(), Lite3CfgPPO_Leggedstand())
```

## Validation

1. **Static import** — no training, just verify Python can import the module
   and resolve the URDF: `python -c "from legged_gym.envs.Lite3_Stand.Lite3_Handstand.Lite3_handstand_Config import Lite3Cfg_Leggedstand; c = Lite3Cfg_Leggedstand()"`
2. **URDF load smoke test** — `python -c "import isaacgym; ...; gym.load_asset(..., 'resources/robots/lite3/urdf/Lite3.urdf', ...)"` — optional, can skip if isaacgym headless setup is fragile.
3. **Train command works** — `python legged_gym/scripts/train.py --task=lite3_handstand --headless --max_iterations=2` runs at least 2 iterations without crashing. Left to user to invoke; design just provides the command.

Not in scope: full training, reward shape verification, sim-to-real check.

## Risks / Open Questions

- **HipY=+0.283 sits at 90% of the soft joint limit.** If
  `soft_joint_pos_limit` is enforced via a clamp before PD, the policy can
  never reach the desire angle exactly. GO2_Leggedstand uses `soft_dof_pos_limit
  = 0.9` (default), which keeps the reachable range above the desire by margin
  on Go2 (where the desire is far from limits). On Lite3 we may need to raise
  this to e.g. 0.95, or explicitly verify that 0.283 falls inside the soft
  range. To verify post-implementation; not blocking.
- **Reward coupling assumption.** GO2_Leggedstand rewards assume Go2 leg
  ordering (FL/FR/RL/RR). The env file uses `feet_indices` lookup for most
  reward terms, so the Lite3 ordering (FL/FR/HL/HR) should "just work" as long
  as `feet_name_reward` and `contact_foot` are correct — but `symmetric_joints`
  (left-right symmetry) needs verification that it pairs FL↔FR and HL↔HR
  rather than diagonals.
- **Action scale per-joint.** Adding `hip_action_scale` requires a small env
  change. If `Go2_legstand.py` doesn't already support per-joint scaling, the
  env code needs a one-line modification.

## Out-of-Repo Dependencies

- `go2_rl_gym-recovery/resources/robots/lite3/{urdf,meshes}/` — must be readable
  at design-doc time; we copy from there. Once copied, no further dep.
