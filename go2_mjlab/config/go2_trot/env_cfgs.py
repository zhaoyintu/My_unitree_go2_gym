"""Unitree Go2 trot environment configuration."""

import math

from go2_mjlab.robots.go2_constants import (
    GO2_ACTION_SCALE,
    get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    TerrainHeightSensorCfg,
    RingPatternCfg,
)
from go2_mjlab import mdp as go2_mdp
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree Go2 flat terrain trot velocity configuration."""
    cfg = make_velocity_env_cfg()

    # Sim settings
    cfg.sim.mujoco.impratio = 10
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.njmax = 300
    cfg.sim.nconmax = None
    cfg.sim.contact_sensor_maxmatch = 64

    # Go2 robot entity
    cfg.scene.entities = {"robot": get_go2_robot_cfg()}

    # Viewer
    cfg.viewer.body_name = "trunk"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    # Wire raycast sensor frame
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            sensor.frame.name = "trunk"

    foot_names = ("FR", "FL", "RR", "RL")
    site_names = foot_names
    geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

    # Wire foot height scan
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in site_names
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

    # Contact sensors
    # Note: Go2 has single thigh collider per leg (no numbered suffixes)
    thigh_geom_names = tuple(f"{leg}_thigh_collision" for leg in foot_names)
    calf_geom_names = tuple(
        f"{leg}_calf_collision{i}" for leg in foot_names for i in (1, 2)
    )

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    thigh_ground_cfg = ContactSensorCfg(
        name="thigh_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot", pattern=thigh_geom_names),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    shank_ground_cfg = ContactSensorCfg(
        name="shank_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot", pattern=calf_geom_names),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    trunk_head_ground_cfg = ContactSensorCfg(
        name="trunk_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=("trunk_collision", "head_collision"),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        thigh_ground_cfg,
        shank_ground_cfg,
        trunk_head_ground_cfg,
    )

    # Flat terrain
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Remove terrain-dependent sensors and obs we don't need on flat
    remove_sensors = {"terrain_scan"}
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
    )
    if "height_scan" in cfg.observations["actor"].terms:
        del cfg.observations["actor"].terms["height_scan"]
    if "height_scan" in cfg.observations["critic"].terms:
        del cfg.observations["critic"].terms["height_scan"]

    # Action scale
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = GO2_ACTION_SCALE

    # ---- Gait clock observation ----
    cfg.observations["actor"].terms["gait_clock"] = ObservationTermCfg(
        func=go2_mdp.gait_clock,
        params={"cycle_time": 0.5},
    )
    cfg.observations["critic"].terms["gait_clock"] = ObservationTermCfg(
        func=go2_mdp.gait_clock,
        params={"cycle_time": 0.5},
    )

    # ---- Rewards ----
    # Remove base velocity task rewards we're replacing
    for rw in ("air_time", "body_ang_vel", "angular_momentum",
               "dof_pos_limits", "pose"):
        cfg.rewards.pop(rw, None)

    # Use Go2-trot-specific tracking rewards (gated on trot quality)
    cfg.rewards["track_linear_velocity"].func = go2_mdp.tracking_linear_velocity_trot
    cfg.rewards["track_linear_velocity"].params = {
        "command_name": "twist",
        "std": math.sqrt(0.25),
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot"),
    }
    cfg.rewards["track_angular_velocity"].func = go2_mdp.tracking_angular_velocity_trot
    cfg.rewards["track_angular_velocity"].params = {
        "command_name": "twist",
        "std": math.sqrt(0.5),
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot"),
    }

    # Go2-specific rewards
    cfg.rewards["lin_vel_z"] = RewardTermCfg(
        func=go2_mdp.lin_vel_z,
        weight=-2.0,
    )
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk",))},
    )
    cfg.rewards["trot"] = RewardTermCfg(
        func=go2_mdp.trot_gait,
        weight=0.8,
        params={
            "sensor_name": "feet_ground_contact",
            "cycle_time": 0.5,
            "command_name": "twist",
        },
    )
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=go2_mdp.stand_still,
        weight=-1.0,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )
    cfg.rewards["contact_without_command"] = RewardTermCfg(
        func=go2_mdp.contact_without_command,
        weight=1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
        },
    )
    cfg.rewards["default_hip_pos"] = RewardTermCfg(
        func=go2_mdp.default_hip_pos,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*hip_joint",))},
    )
    cfg.rewards["base_height"] = RewardTermCfg(
        func=go2_mdp.base_height,
        weight=-5.0,
        params={"target_height": 0.29},
    )
    cfg.rewards["joint_acceleration"] = RewardTermCfg(
        func=go2_mdp.joint_acceleration,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )

    # Adjust builtin reward weights
    cfg.rewards["upright"].weight = -2.0
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
    cfg.rewards["action_rate_l2"].weight = -0.01

    # Collision penalties
    cfg.rewards["thigh_collision"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "thigh_ground_touch"},
    )
    cfg.rewards["shank_collision"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "shank_ground_touch"},
    )
    cfg.rewards["trunk_head_collision"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "trunk_ground_touch"},
    )

    # Feet clearance
    cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
    cfg.rewards["foot_clearance"].weight = 0.1
    cfg.rewards["foot_swing_height"].weight = 0.0
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

    # ---- Domain Randomization ----
    # Foot friction (per-axis for condim 6)
    del cfg.events["foot_friction"]
    cfg.events["foot_friction_slide"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
            "operation": "abs",
            "axes": [0],
            "ranges": (0.2, 1.2),
            "shared_random": True,
        },
    )
    cfg.events["foot_friction_spin"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
            "operation": "abs",
            "distribution": "log_uniform",
            "axes": [1],
            "ranges": (1e-4, 2e-2),
            "shared_random": True,
        },
    )
    cfg.events["foot_friction_roll"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
            "operation": "abs",
            "distribution": "log_uniform",
            "axes": [2],
            "ranges": (1e-5, 5e-3),
            "shared_random": True,
        },
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

    # ---- Terminations ----
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )
    cfg.terminations["base_contact"] = TerminationTermCfg(
        func=go2_mdp.base_contact,
        params={"sensor_name": "trunk_ground_touch"},
    )

    # ---- Commands ----
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.resampling_time_range = (3.0, 8.0)

    # ---- Episode length ----
    cfg.episode_length_s = 24.0

    # ---- Curriculum ----
    cfg.curriculum.pop("terrain_levels", None)

    # ---- Play mode ----
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg
