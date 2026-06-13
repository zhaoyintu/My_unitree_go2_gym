"""DeepRobotics Lite3 constants for mjlab handstand training.

Mirrors `go2_constants.py` but for Lite3.  Differences worth keeping in
mind when reading reward / observation code from the Go2 side:

* Joint naming.  Go2 uses ``[FR/FL/RL/RR]_[hip/thigh/calf]_joint``; Lite3
  uses ``[FL/FR/HL/HR]_[HipX/HipY/Knee]_joint``.  HipX is roll/abduction
  (same role as Go2 ``hip``), HipY is the thigh joint, Knee is the calf.
* Joint axes.  Lite3 uses ``-x / -y / -y`` for HipX/HipY/Knee; Go2 uses
  ``+x / +y / +y``.  All three Lite3 joint values are sign-flipped from
  the Go2 equivalent (HipX vs hip, HipY vs thigh, Knee vs calf).
* Knee range.  ``[0.524, 2.792]`` is fully positive on Lite3 (vs Go2's
  fully negative ``[-2.7227, -0.83776]``); a Lite3 standing knee is
  ``+1.6`` whereas a Go2 standing calf is ``-1.5``.
* HipY range.  ``[-2.67, 0.314]`` on Lite3 — note the **upper bound of
  0.314 rad** (~18°).  Go2's front-leg "stance" handstand pose uses
  ``thigh = -0.7`` (which would map to ``HipY = +0.7`` on Lite3) — but
  that's outside the joint range.  The Lite3 handstand stance pose
  therefore can't be a sign-flip of Go2's; it must be re-derived.
* PD gains.  ``deploy_lite3_recovery.py`` uses Kp=30 / Kd=1; for
  handstand we mirror Go2's stiffer gains (Kp=40 / Kd=1 — same as
  IsaacGym GO2_Leggedstand) so the control authority is comparable.
* Action scale.  HipX uses 0.125, HipY/Knee use 0.25 (per the recovery
  deploy config).
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

LITE3_MJLAB_PATH = Path(__file__).resolve().parent

LITE3_XML: Path = LITE3_MJLAB_PATH / "xmls" / "lite3.xml"
assert LITE3_XML.exists(), f"Lite3 XML not found at {LITE3_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(LITE3_XML))


##
# Actuator config.
##

# Lite3 motor data — using the same motor model as Go2 (rotor inertia,
# velocity / effort limits) since Lite3 specs are not fully published.
# These values give roughly the same Kp*action_scale torque authority as
# the deploy_lite3_recovery.py PD controller.
ROTOR_INERTIA = 0.000111842

HIPX_GEAR_RATIO = 6
HIPY_GEAR_RATIO = 6
KNEE_GEAR_RATIO = 9

HIPX_ACTUATOR = ElectricActuator(
    reflected_inertia=reflected_inertia(ROTOR_INERTIA, HIPX_GEAR_RATIO),
    velocity_limit=30.1,
    effort_limit=30.0,
)
HIPY_ACTUATOR = ElectricActuator(
    reflected_inertia=reflected_inertia(ROTOR_INERTIA, HIPY_GEAR_RATIO),
    velocity_limit=30.1,
    effort_limit=30.0,
)
KNEE_ACTUATOR = ElectricActuator(
    reflected_inertia=reflected_inertia(ROTOR_INERTIA, KNEE_GEAR_RATIO),
    velocity_limit=20.06,
    effort_limit=30.0,
)

# Standard locomotion PD: matches deploy_lite3_recovery.py (Kp=30, Kd=1).
LITE3_HIPX_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*HipX_joint",),
    stiffness=30.0,
    damping=1.0,
    effort_limit=HIPX_ACTUATOR.effort_limit,
    armature=HIPX_ACTUATOR.reflected_inertia,
)
LITE3_HIPY_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*HipY_joint",),
    stiffness=30.0,
    damping=1.0,
    effort_limit=HIPY_ACTUATOR.effort_limit,
    armature=HIPY_ACTUATOR.reflected_inertia,
)
LITE3_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*Knee_joint",),
    stiffness=30.0,
    damping=1.0,
    effort_limit=KNEE_ACTUATOR.effort_limit,
    armature=KNEE_ACTUATOR.reflected_inertia,
)

# Handstand PD: Kp=40 matching Go2 handstand configuration.
# (Reverted from Kp=30 for an A/B experiment isolating whether the
# 5 sim2real DR events are the cause of failed early-iter learning.
# After the experiment, this can go back to 30 — the lower value
# matches the real-robot deploy config and may help sim-to-real.)
LITE3_HANDSTAND_HIPX_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*HipX_joint",),
    stiffness=40.0,
    damping=1.0,
    effort_limit=HIPX_ACTUATOR.effort_limit,
    armature=HIPX_ACTUATOR.reflected_inertia,
)
LITE3_HANDSTAND_HIPY_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*HipY_joint",),
    stiffness=40.0,
    damping=1.0,
    effort_limit=HIPY_ACTUATOR.effort_limit,
    armature=HIPY_ACTUATOR.reflected_inertia,
)
LITE3_HANDSTAND_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*Knee_joint",),
    stiffness=40.0,
    damping=1.0,
    effort_limit=KNEE_ACTUATOR.effort_limit,
    armature=KNEE_ACTUATOR.reflected_inertia,
)

# RLDeploy handstand PD: match the C++ `rl_deploy_handstand` command contract.
# The deploy MuJoCo XML has motor actuators and no reflected joint armature; the
# C++ loop computes tau = kp * (q_des - q) + kd * (0 - dq), clipped to 30 Nm.
LITE3_RLDEPLOY_HANDSTAND_HIPX_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*HipX_joint",),
    stiffness=40.0,
    damping=1.0,
    effort_limit=HIPX_ACTUATOR.effort_limit,
)
LITE3_RLDEPLOY_HANDSTAND_HIPY_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*HipY_joint",),
    stiffness=40.0,
    damping=1.0,
    effort_limit=HIPY_ACTUATOR.effort_limit,
)
LITE3_RLDEPLOY_HANDSTAND_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*Knee_joint",),
    stiffness=40.0,
    damping=1.0,
    effort_limit=KNEE_ACTUATOR.effort_limit,
)

##
# Keyframes.
##

# Sign convention check: Lite3 HipX axis is (-1,0,0).  Go2 hip axis is
# (+1,0,0).  Go2's L hip = -0.1, R hip = +0.1; sign-flipped on Lite3
# gives L HipX = +0.1, R HipX = -0.1.
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.30),
    joint_pos={
        ".*L_HipX_joint": 0.1,
        ".*R_HipX_joint": -0.1,
        ".*HipY_joint": -0.8,
        ".*Knee_joint": 1.6,
    },
    joint_vel={".*": 0.0},
)

# Handstand init = NORMAL four-paw stand.  The dog starts upright; the
# policy itself learns to flip onto its front paws (head down) and walk.
# (Same approach as `HANDSTAND_INIT_STATE` in go2_constants.py.)
HANDSTAND_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.30),
    joint_pos={
        ".*L_HipX_joint": 0.1,
        ".*R_HipX_joint": -0.1,
        ".*HipY_joint": -0.8,
        ".*Knee_joint": 1.6,
    },
    joint_vel={".*": 0.0},
)

# Footstand init = NORMAL four-paw stand with the handstand_gym
# `Lite3_stand` default joint angles (front HipY -0.8, hind HipY -1.0,
# Knee 1.5).  The source URDF and this MJCF share the same (-1,0,0)
# HipX axis, so the source values transfer directly (FL/HL = -0.1,
# FR/HR = +0.1).  Spawn z is 0.30 (not the source's 0.22, which relies
# on IsaacGym auto-depenetration); reset settles the robot onto its feet.
FOOTSTAND_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.30),
    joint_pos={
        ".*L_HipX_joint": -0.1,
        ".*R_HipX_joint": 0.1,
        "F[LR]_HipY_joint": -0.8,
        "H[LR]_HipY_joint": -1.0,
        ".*Knee_joint": 1.5,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Foot collision geom names: FL_FOOT_collision, FR_FOOT_collision,
# HL_FOOT_collision, HR_FOOT_collision.
_foot_regex = "^[FH][LR]_FOOT_collision$"

FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(_foot_regex,),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
    solimp=(0.9, 0.95, 0.023),
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    solref=(0.01, 1),
    condim={_foot_regex: 6, ".*_collision": 1},
    priority={_foot_regex: 1},
    friction={_foot_regex: (1, 5e-3, 5e-4)},
)

# Match the Lite3 XML used by `Lite3_rl_deploy`:
# collision geoms use contype=0/conaffinity=1, condim=3, solref=0.005 1,
# friction=1 0.01 0.01.  This disables robot self-collision while preserving
# terrain contact through the terrain geom's contype.
RLDEPLOY_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=0,
    friction=(1.0, 0.01, 0.01),
    solref=(0.005, 1),
)

# Softened-contact variant for the undulating-terrain finetune.  Same as
# RLDEPLOY_COLLISION but the contact time-constant is relaxed 0.005 s -> 0.02 s
# (and solimp made softer).  The stiff 5 ms contact explodes to NaN when the
# flat-trained handstand tumbles onto a heightfield slope or spawns slightly
# penetrating a wave crest; a 20 ms contact absorbs penetration smoothly so
# the episode just terminates instead of NaN-ing the whole batch.
RLDEPLOY_TERRAIN_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=0,
    friction=(1.0, 0.01, 0.01),
    solref=(0.02, 1),
    solimp=(0.9, 0.95, 0.01),
)

##
# Final config.
##

LITE3_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        LITE3_HIPX_ACTUATOR_CFG,
        LITE3_HIPY_ACTUATOR_CFG,
        LITE3_KNEE_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)

LITE3_HANDSTAND_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        LITE3_HANDSTAND_HIPX_ACTUATOR_CFG,
        LITE3_HANDSTAND_HIPY_ACTUATOR_CFG,
        LITE3_HANDSTAND_KNEE_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)

LITE3_RLDEPLOY_HANDSTAND_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        LITE3_RLDEPLOY_HANDSTAND_HIPX_ACTUATOR_CFG,
        LITE3_RLDEPLOY_HANDSTAND_HIPY_ACTUATOR_CFG,
        LITE3_RLDEPLOY_HANDSTAND_KNEE_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_lite3_robot_cfg() -> EntityCfg:
    """Lite3 with standard locomotion PD gains (Kp=30)."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=LITE3_ARTICULATION,
    )


def get_lite3_handstand_robot_cfg() -> EntityCfg:
    """Lite3 with stiffer PD gains (Kp=40) matching Go2 handstand."""
    return EntityCfg(
        init_state=HANDSTAND_INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=LITE3_HANDSTAND_ARTICULATION,
    )


def get_lite3_rldeploy_handstand_robot_cfg() -> EntityCfg:
    """Lite3 handstand robot with dynamics aligned to `rl_deploy_handstand`."""
    return EntityCfg(
        init_state=HANDSTAND_INIT_STATE,
        collisions=(RLDEPLOY_COLLISION,),
        spec_fn=get_spec,
        articulation=LITE3_RLDEPLOY_HANDSTAND_ARTICULATION,
    )


def get_lite3_terrain_handstand_robot_cfg() -> EntityCfg:
    """RLDeploy handstand robot with softened contact for undulating terrain."""
    return EntityCfg(
        init_state=HANDSTAND_INIT_STATE,
        collisions=(RLDEPLOY_TERRAIN_COLLISION,),
        spec_fn=get_spec,
        articulation=LITE3_RLDEPLOY_HANDSTAND_ARTICULATION,
    )


def get_lite3_footstand_robot_cfg() -> EntityCfg:
    """Lite3 for rear-leg footstand: Kp=40/Kd=1 PD (matching handstand_gym
    `Lite3_stand` control config) with the Lite3_stand init pose."""
    return EntityCfg(
        init_state=FOOTSTAND_INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=LITE3_HANDSTAND_ARTICULATION,
    )


# Deploy config (deploy_lite3_recovery.py): HipX gets 0.125, HipY/Knee
# get 0.25.  The narrower HipX action scale is because abduction is a
# higher-authority joint and a 0.25 step would be too aggressive.
LITE3_HANDSTAND_ACTION_SCALE: dict[str, float] = {
    ".*HipX_joint": 0.125,
    ".*HipY_joint": 0.25,
    ".*Knee_joint": 0.25,
}


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_lite3_robot_cfg())
    viewer.launch(robot.spec.compile())
