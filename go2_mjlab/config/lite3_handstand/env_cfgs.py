"""DeepRobotics Lite3 front-paw handstand-walking environment.

Ported from `go2_handstand/env_cfgs.py`.  Same MDP / reward shape, but
adjusted for Lite3's joint names, axis sign conventions, joint-range
limits, and slightly different geometry.

Lite3-specific differences vs the Go2 version (see lite3_constants.py
for the full sign-convention writeup):

* Joint name pattern.  ``.*hip_joint`` → ``.*HipX_joint``,
  ``.*thigh_joint`` → ``.*HipY_joint``, ``.*calf_joint`` → ``.*Knee_joint``.
* MJCF joint order.  Lite3 is FL, FR, HL, HR (each: HipX, HipY, Knee).
  Foot-index role mapping is the same as Go2: (0, 1) = front feet =
  stance, (2, 3) = rear feet = swing.
* Body / geom names.  ``trunk`` → ``TORSO`` (no separate head body),
  ``[FR/FL/RL/RR]_thigh_collision`` → ``[FL/FR/HL/HR]_THIGH_collision``,
  calves use ``_SHANK_collision`` (single mesh per leg vs Go2's pair of
  cylinder geoms).
* Desire (handstand) pose.  Lite3 HipY range upper bound is 0.314 rad,
  so it cannot reach Go2's stance pose (thigh = -0.7 / HipY = +0.7).
  HipY is held at 90% of the soft upper limit (= 0.283).  Knee is
  chosen for visual quality rather than maximum foot reach: at the
  reach-maximizing Knee = π/2 − HipY ≈ 1.288 the front leg looks like
  "kneeling on a vertical shin" (interior knee angle ≈ 106°, looks
  like a right angle externally).  Knee = 2.0 instead gives a clearly
  bent leg (interior angle ≈ 65°, acute) with the front foot
  near-vertically below the front shoulder in world frame —
  recognizable handstand stance.  Rear swing rest pose stays at
  HipY = -0.8, Knee = +1.6 (Lite3 standing rest).
* Target heights.  base = 0.39 m (vs Go2 0.47), rear-foot world
  z = 0.56 m (vs Go2 0.67).  Both derived from forward kinematics with
  the chosen Knee = 2.0 desire — see
  ``deploy_mujoco_viewer/verify_lite3_handstand_pose.py``.
"""

from go2_mjlab.robots.lite3_constants import (
    LITE3_HANDSTAND_ACTION_SCALE,
    get_lite3_handstand_robot_cfg,
)
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
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.utils.noise import UniformNoiseCfg as Unoise


# ---- Lite3 handstand desire-pose constants ---------------------------------
# Front (FL, FR) stance — HipY at 90% of upper limit, Knee bent past π/2
# so the leg has a recognizable handstand fold (foot near-vertically
# below the shoulder in world frame).  See module docstring.
_FRONT_DESIRE = (0.0, 0.283, 2.0)
# Rear (HL, HR) swing rest pose — same as Lite3 normal standing.
_REAR_DESIRE = (0.0, -0.8, 1.6)
# Full 12-joint desire vector in MJCF order (FL, FR, HL, HR).
LITE3_DESIRE_JOINT_ANGLES = list(_FRONT_DESIRE * 2 + _REAR_DESIRE * 2)

# Geometric targets — verify_lite3_handstand_pose.py confirms feet land
# on z=0 with these values when the body is pitched +π/2 (head down).
LITE3_BASE_HEIGHT_TARGET = 0.39       # trunk world z during handstand
LITE3_REAR_FOOT_TARGET_Z = 0.56       # swing-rest rear foot world z


def unitree_lite3_handstand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Lite3 handstand configuration."""
    foot_names = ("FL", "FR", "HL", "HR")
    geom_names = tuple(f"{n}_FOOT_collision" for n in foot_names)

    # Sensors
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce",
        num_slots=1, track_air_time=True,
    )
    # Lite3 has no separate "head" body — TORSO is the only trunk geom.
    trunk_ground_cfg = ContactSensorCfg(
        name="trunk_ground_touch",
        primary=ContactMatch(
            mode="geom", entity="robot",
            pattern=("TORSO_collision",),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1, history_length=4,
    )
    thigh_ground_cfg = ContactSensorCfg(
        name="thigh_ground_touch",
        primary=ContactMatch(
            mode="geom", entity="robot",
            pattern=tuple(f"{n}_THIGH_collision" for n in foot_names),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="none", num_slots=1, history_length=4,
    )
    # Lite3 has a single SHANK collision mesh per leg (vs Go2's two cylinders).
    calf_ground_cfg = ContactSensorCfg(
        name="calf_ground_touch",
        primary=ContactMatch(
            mode="geom", entity="robot",
            pattern=tuple(f"{n}_SHANK_collision" for n in foot_names),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="none", num_slots=1, history_length=4,
    )

    # Observations — same dim layout as Go2 (48 per frame, history 10).
    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "commands": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "twist"},
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
            history_length=10,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms, concatenate_terms=True, enable_corruption=False,
        ),
    }

    # Actions
    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot", actuator_names=(".*",),
            scale=LITE3_HANDSTAND_ACTION_SCALE, use_default_offset=True,
        )
    }

    # Velocity commands — same range as Go2 handstand.
    commands = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.4, 0.4),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.4, 0.4),
            ),
            resampling_time_range=(5.0, 5.0),
            heading_command=False,
        ),
    }

    # ---- Rewards (matching go2_handstand) ----
    rewards = {
        "alive": RewardTermCfg(
            func=go2_mdp.alive, weight=1.0,
        ),
        "handstand_orientation": RewardTermCfg(
            func=go2_mdp.handstand_orientation, weight=-1.0,
            params={"target_gravity": (1.0, 0.0, 0.0)},
        ),
        "handstand_feet_on_air": RewardTermCfg(
            func=go2_mdp.handstand_feet_on_air, weight=0.4,
            params={
                "sensor_name": "feet_ground_contact",
                "foot_indices": (2, 3),   # HL, HR — rear feet (swing)
            },
        ),
        "handstand_feet_height_exp": RewardTermCfg(
            func=go2_mdp.handstand_feet_height, weight=5.0,
            params={
                "target_height": LITE3_REAR_FOOT_TARGET_Z,
                "foot_indices": (2, 3),
                "asset_cfg": SceneEntityCfg("robot", body_names=("TORSO",)),
            },
        ),
        "base_height": RewardTermCfg(
            func=go2_mdp.base_height, weight=1.5,
            params={"target_height": LITE3_BASE_HEIGHT_TARGET},
        ),
        "tracking_lin_vel": RewardTermCfg(
            func=go2_mdp.handstand_tracking_lin_vel, weight=2.5,
            params={"command_name": "twist"},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func=go2_mdp.handstand_tracking_ang_vel, weight=2.5,
            params={"command_name": "twist"},
        ),
        "tracking_lin_vel_zero": RewardTermCfg(
            func=go2_mdp.handstand_tracking_lin_vel_zero, weight=-0.2,
            params={"command_name": "twist"},
        ),
        "tracking_ang_vel_zero": RewardTermCfg(
            func=go2_mdp.handstand_tracking_ang_vel_zero, weight=-0.2,
            params={"command_name": "twist"},
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
        "contact": RewardTermCfg(
            func=go2_mdp.handstand_contact, weight=0.3,
            params={
                "sensor_name": "feet_ground_contact",
                "foot_indices": (0, 1),   # FL, FR — stance feet
            },
        ),
        "feet_air_time": RewardTermCfg(
            func=go2_mdp.handstand_feet_air_time, weight=2.0,
            params={
                "sensor_name": "feet_ground_contact",
                "foot_indices": (0, 1),   # FL, FR — stance feet
            },
        ),
        "feet_clearance": RewardTermCfg(
            func=go2_mdp.handstand_feet_clearance, weight=0.4,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("TORSO",)),
                "foot_indices": (2, 3),
                "target_foot_height": 0.06,
                "cycle_time": 1.6,
            },
        ),
        "default_pos": RewardTermCfg(
            func=go2_mdp.default_joint_penalty, weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "desire_joint_angles": LITE3_DESIRE_JOINT_ANGLES,
            },
        ),
        "default_pos_reward": RewardTermCfg(
            func=go2_mdp.handstand_default_pos_reward, weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "desire_joint_angles": LITE3_DESIRE_JOINT_ANGLES,
            },
        ),
        # Lite3's HipX joints play the role of Go2's hip_joint (abduction).
        "default_hip_pos": RewardTermCfg(
            func=go2_mdp.default_hip_pos, weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*HipX_joint",))},
        ),
        "action_rate_l2": RewardTermCfg(
            func=envs_mdp.action_rate_l2, weight=-0.05,
        ),
        "dof_acc": RewardTermCfg(
            func=go2_mdp.joint_acceleration, weight=-2.5e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "base_contact": RewardTermCfg(
            func=go2_mdp.body_contact, weight=-2.0,
            params={"sensor_name": "trunk_ground_touch"},
        ),
        "thigh_collision": RewardTermCfg(
            func=go2_mdp.body_contact, weight=-1.0,
            params={"sensor_name": "thigh_ground_touch"},
        ),
        "calf_collision": RewardTermCfg(
            func=go2_mdp.body_contact, weight=-2.0,
            params={"sensor_name": "calf_ground_touch"},
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "base_contact": TerminationTermCfg(
            func=go2_mdp.base_contact,
            params={"sensor_name": "trunk_ground_touch", "force_threshold": 1.0},
        ),
    }

    events = {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5), "y": (-0.5, 0.5),
                    "z": (-0.02, 0.02),
                    "yaw": (-3.14, 3.14),
                },
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
                "position_range": (-0.25, 0.25),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=envs_mdp.dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
                "operation": "abs",
                "ranges": (0.2, 0.8),
                "shared_random": True,
            },
        ),
        "base_mass": EventTermCfg(
            mode="startup",
            func=envs_mdp.dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("TORSO",)),
                "operation": "add",
                "ranges": (-1.0, 2.0),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=envs_mdp.dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "ranges": (-0.05, 0.05),
            },
        ),
        # Lite3 is lighter than Go2 (5.6 kg vs 12 kg), so the same ±20 N
        # force gives ~2x the acceleration.  Keep magnitudes for now and
        # tune after first training run.
        "push_robot": EventTermCfg(
            func=envs_mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(4.0, 8.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("TORSO",)),
                "force_range": (-20.0, 20.0),
                "torque_range": (-8.0, 8.0),
            },
        ),
    }

    if play:
        observations["actor"].enable_corruption = False
        events = {k: v for k, v in events.items() if k in ("reset_base", "reset_robot_joints")}
        for cmd_cfg in commands.values():
            if hasattr(cmd_cfg, 'resampling_time_range'):
                cmd_cfg.resampling_time_range = (100.0, 100.0)

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_lite3_handstand_robot_cfg()},
            sensors=(feet_ground_cfg, trunk_ground_cfg,
                     thigh_ground_cfg, calf_ground_cfg),
            num_envs=4096,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        metrics={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot", body_name="TORSO",
            distance=1.5, elevation=-10.0, azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=35, njmax=1500,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=20.0,
    )
