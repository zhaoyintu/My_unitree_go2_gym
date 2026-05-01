"""Unitree Go2 constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GO2_MJLAB_PATH = Path(__file__).resolve().parent

GO2_XML: Path = GO2_MJLAB_PATH / "xmls" / "go2.xml"
assert GO2_XML.exists(), f"Go2 XML not found at {GO2_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(GO2_XML))


##
# Actuator config.
##

# Rotor inertia (same as Go1, shared motor hardware).
ROTOR_INERTIA = 0.000111842

# Gearbox.
HIP_GEAR_RATIO = 6
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5

HIP_ACTUATOR = ElectricActuator(
    reflected_inertia=reflected_inertia(ROTOR_INERTIA, HIP_GEAR_RATIO),
    velocity_limit=30.1,
    effort_limit=23.7,
)
KNEE_ACTUATOR = ElectricActuator(
    reflected_inertia=reflected_inertia(ROTOR_INERTIA, KNEE_GEAR_RATIO),
    velocity_limit=20.06,
    effort_limit=35.55,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIP = HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_HIP = 2 * DAMPING_RATIO * HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_KNEE = KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_KNEE = 2 * DAMPING_RATIO * KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ

GO2_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
    stiffness=STIFFNESS_HIP,
    damping=DAMPING_HIP,
    effort_limit=HIP_ACTUATOR.effort_limit,
    armature=HIP_ACTUATOR.reflected_inertia,
)
GO2_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_calf_joint",),
    stiffness=STIFFNESS_KNEE,
    damping=DAMPING_KNEE,
    effort_limit=KNEE_ACTUATOR.effort_limit,
    armature=KNEE_ACTUATOR.reflected_inertia,
)

# Handstand PD gains: match IsaacGym Kp=40, Kd=1.0 for ALL 12 joints.
# IsaacGym PD: torque = 40 * (default + action*0.25 - q) - 1.0 * qvel
# We set stiffness and damping directly to match, bypassing the
# natural-frequency formula used for standard locomotion tasks.
GO2_HANDSTAND_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
    stiffness=40.0,
    damping=1.0,
    effort_limit=HIP_ACTUATOR.effort_limit,
    armature=HIP_ACTUATOR.reflected_inertia,
)
GO2_HANDSTAND_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_calf_joint",),
    stiffness=40.0,
    damping=1.0,
    effort_limit=KNEE_ACTUATOR.effort_limit,
    armature=KNEE_ACTUATOR.reflected_inertia,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.27),
    joint_pos={
        ".*thigh_joint": 0.9,
        ".*calf_joint": -1.8,
        ".*R_hip_joint": 0.1,
        ".*L_hip_joint": -0.1,
    },
    joint_vel={".*": 0.0},
)

# Handstand init = NORMAL four-paw stand. The dog starts upright; the
# policy itself learns to flip onto its front paws (head down) and walk.
# Mirrors IsaacGym GO2_Leggedstand init_state.default_joint_angles
# (the IsaacGym task that actually trains a front-paw handstand walk):
#     thigh: front=0.8, rear=1.0
#     calf:  -1.5 for all
#     hip:   |0.1| (sign convention follows mjlab MJCF below)
# Calf and thigh values map 1:1 between IsaacGym URDF and the MJCF —
# both use range [-2.7227, -0.83776] for the calf with axis (0,1,0).
HANDSTAND_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.42),
    joint_pos={
        ".*L_hip_joint": -0.1,
        ".*R_hip_joint": 0.1,
        "F[LR]_thigh_joint": 0.8,
        "R[LR]_thigh_joint": 1.0,
        ".*calf_joint": -1.5,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(_foot_regex,),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
    solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions.
# Foot collisions are given custom condim, friction.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    # Harden all collision geoms.
    solref=(0.01, 1),
    # Configure feet colliders. Other colliders are frictionless (condim=1).
    condim={_foot_regex: 6, ".*_collision": 1},
    priority={_foot_regex: 1},
    friction={_foot_regex: (1, 5e-3, 5e-4)},
)

##
# Final config.
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        GO2_HIP_ACTUATOR_CFG,
        GO2_KNEE_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)

GO2_HANDSTAND_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        GO2_HANDSTAND_HIP_ACTUATOR_CFG,
        GO2_HANDSTAND_KNEE_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_go2_robot_cfg() -> EntityCfg:
    """Get a fresh Go2 robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=GO2_ARTICULATION,
    )


def get_go2_handstand_robot_cfg() -> EntityCfg:
    """Get a Go2 robot config tuned for handstand (inverted on front legs).

    Uses higher init position (0.42), different default joint angles
    (front thighs at 0.8, rear thighs at 1.0, all calves at -1.5),
    and stiffer PD gains (Kp≈40 for hip/thigh, matching IsaacGym).
    """
    return EntityCfg(
        init_state=HANDSTAND_INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=GO2_HANDSTAND_ARTICULATION,
    )


GO2_ACTION_SCALE: dict[str, float] = {}
for a in GO2_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
        GO2_ACTION_SCALE[n] = 0.25 * e / s

# Handstand action scale: uniform 0.25 matching IsaacGym's control.action_scale.
# We do NOT use the formula 0.25 * effort_limit / stiffness because handstand PD
# gains were increased (Kp≈40) to match IsaacGym. The standard formula would
# auto-reduce action_scale to ~0.147 (since stiffness increased), cutting torque
# authority by 41% vs IsaacGym. Uniform 0.25 ensures Kp * action_scale = 10 Nm
# per unit action, matching IsaacGym exactly.
GO2_HANDSTAND_ACTION_SCALE: dict[str, float] = {
    ".*_hip_joint": 0.25,
    ".*_thigh_joint": 0.25,
    ".*_calf_joint": 0.25,
}


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_go2_robot_cfg())

    viewer.launch(robot.spec.compile())
