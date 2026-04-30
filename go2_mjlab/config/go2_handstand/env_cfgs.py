"""Unitree Go2 handstand environment configuration.

Robot balances inverted on its front legs, base pointing upward.
Target projected gravity: (-1, 0, 0) in body frame (body x-axis aligned with world +z).

Ported from IsaacGym go2_handstand — matches reward scales, termination logic, and init state.
"""

import math

from go2_mjlab.robots.go2_constants import GO2_ACTION_SCALE, get_go2_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from go2_mjlab import mdp as go2_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def unitree_go2_handstand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree Go2 handstand configuration."""
    foot_names = ("FR", "FL", "RR", "RL")
    geom_names = tuple(f"{n}_foot_collision" for n in foot_names)

    # Sensors
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce",
        num_slots=1, track_air_time=True,
    )
    trunk_head_ground_cfg = ContactSensorCfg(
        name="trunk_ground_touch",
        primary=ContactMatch(
            mode="geom", entity="robot",
            pattern=("trunk_collision", "head_collision"),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="none", num_slots=1, history_length=4,
    )

    # Observations
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
    }
    critic_terms = {
        **actor_terms,
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
        "foot_contact": ObservationTermCfg(
            func=go2_mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms, concatenate_terms=True, enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms, concatenate_terms=True, enable_corruption=False,
        ),
    }

    # Actions
    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot", actuator_names=(".*",),
            scale=0.25, use_default_offset=True,
        )
    }

    # Rewards — ported from IsaacGym go2_handstand
    # Scales: https://github.com/.../Go2_handstand_Config.py class rewards.scales
    rewards = {
        "alive": RewardTermCfg(
            func=go2_mdp.alive, weight=1.0,
        ),
        "handstand_orientation": RewardTermCfg(
            func=go2_mdp.handstand_orientation, weight=-1.0,
            params={"target_gravity": (-1.0, 0.0, 0.0)},
        ),
        "handstand_feet_on_air": RewardTermCfg(
            func=go2_mdp.handstand_feet_on_air, weight=0.4,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "handstand_feet_height_exp": RewardTermCfg(
            func=go2_mdp.handstand_feet_height, weight=5.0,
            params={"target_height": 0.67, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",))},
        ),
        "base_height": RewardTermCfg(
            func=go2_mdp.base_height, weight=1.5,
            params={"target_height": 0.52},
        ),
        "lin_vel_z": RewardTermCfg(
            func=go2_mdp.lin_vel_x_penalty, weight=0.2,
        ),
        "ang_vel_xy": RewardTermCfg(
            func=go2_mdp.ang_vel_xy_penalty, weight=0.2,
        ),
        "ang_xz_penalty": RewardTermCfg(
            func=go2_mdp.ang_xz_penalty, weight=-0.5,
        ),
        "symmetric_joints": RewardTermCfg(
            func=go2_mdp.symmetric_joints, weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "action_rate_l2": RewardTermCfg(
            func=envs_mdp.action_rate_l2, weight=-0.05,
        ),
        "joint_acceleration": RewardTermCfg(
            func=go2_mdp.joint_acceleration, weight=-2.5e-7,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "base_contact": RewardTermCfg(
            func=go2_mdp.body_contact, weight=-2.0,
            params={"sensor_name": "trunk_ground_touch"},
        ),
        "default_joint_pos": RewardTermCfg(
            func=go2_mdp.default_joint_penalty, weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                # Joint order (MJCF): FR, FL, RL, RR
                "desire_joint_angles": [
                    0.0, 0.1, 0.8,    # FR: hip, thigh, calf
                    0.0, -0.1, 0.8,   # FL: hip, thigh, calf
                    0.0, -0.1, 2.25,  # RL: hip, thigh, calf
                    0.0, 0.1, 2.25,   # RR: hip, thigh, calf
                ],
            },
        ),
    }

    # Terminations — base contact only (matching IsaacGym)
    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "base_contact": TerminationTermCfg(
            func=go2_mdp.base_contact,
            params={"sensor_name": "trunk_ground_touch"},
        ),
    }

    # Events — match IsaacGym init: z=0.42, random velocities, joint randomization
    events = {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.13, 0.17), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
                },
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.5, 0.5),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_go2_robot_cfg()},
            sensors=(feet_ground_cfg, trunk_head_ground_cfg),
            num_envs=4096,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        metrics={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot", body_name="trunk",
            distance=1.5, elevation=-10.0, azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=35, njmax=1500,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=20.0,
    )
