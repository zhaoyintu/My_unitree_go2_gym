"""Unitree Go2 handstand environment configuration.

Robot balances inverted on its front legs, base pointing upward.
Target projected gravity: (+1, 0, 0) in body frame (body x-axis aligned with world -z,
i.e. head pointing down to the floor; rear/butt pointing up).

Ported from IsaacGym go2_handstand — matches reward scales, termination logic, init state,
command structure, and domain randomization.
"""

from go2_mjlab.robots.go2_constants import (
    GO2_HANDSTAND_ACTION_SCALE,
    get_go2_handstand_robot_cfg,
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
        fields=("found", "force"), reduce="netforce", num_slots=1, history_length=4,
    )
    thigh_ground_cfg = ContactSensorCfg(
        name="thigh_ground_touch",
        primary=ContactMatch(
            mode="geom", entity="robot",
            pattern=(
                "FR_thigh_collision", "FL_thigh_collision",
                "RR_thigh_collision", "RL_thigh_collision",
            ),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="none", num_slots=1, history_length=4,
    )

    # Observations — includes commands for velocity tracking
    # IsaacGym: 48 dims = lin_vel(3) + ang_vel(3) + gravity(3) + commands(3)
    #                   + joint_pos(12) + joint_vel(12) + actions(12)
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
        # Override base_lin_vel with noiseless version for privileged critic
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
        "foot_contact": ObservationTermCfg(
            func=go2_mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }

    # IsaacGym uses frame_stack=10 for actor. Use 5 as a memory-efficient starting point.
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
            scale=GO2_HANDSTAND_ACTION_SCALE, use_default_offset=True,
        )
    }

    # ---- Velociy Commands ----
    # IsaacGym: lin_vel_x=[-0.4,0.4], lin_vel_y=0, ang_vel_yaw=[-0.4,0.4], resampling_time=5s
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

    # ---- Rewards (ported from IsaacGym GO2Cfg_Handstand.rewards.scales) ----
    rewards = {
        # Survival
        "alive": RewardTermCfg(
            func=go2_mdp.alive, weight=1.0,
        ),
        # Handstand quality
        "handstand_orientation": RewardTermCfg(
            func=go2_mdp.handstand_orientation, weight=-1.0,
            params={"target_gravity": (1.0, 0.0, 0.0)},
        ),
        "handstand_feet_on_air": RewardTermCfg(
            func=go2_mdp.handstand_feet_on_air, weight=0.4,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "handstand_feet_height_exp": RewardTermCfg(
            func=go2_mdp.handstand_feet_height, weight=5.0,
            params={
                "target_height": 0.67,
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
            },
        ),
        "base_height": RewardTermCfg(
            func=go2_mdp.base_height, weight=1.5,
            params={"target_height": 0.08},
        ),
        # Velocity tracking (gated by handstand quality > 70%)
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
        # Body velocity penalties
        "lin_vel_z": RewardTermCfg(
            func=go2_mdp.lin_vel_x_penalty, weight=0.2,
        ),
        "ang_vel_xy": RewardTermCfg(
            func=go2_mdp.ang_vel_xy_penalty, weight=0.2,
        ),
        # Orientation
        "ang_xz_penalty": RewardTermCfg(
            func=go2_mdp.ang_xz_penalty, weight=-0.5,
        ),
        # Symmetry
        "symmetric_joints": RewardTermCfg(
            func=go2_mdp.symmetric_joints, weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        # Contact shaping
        "contact": RewardTermCfg(
            func=go2_mdp.handstand_contact, weight=0.3,
            params={
                "sensor_name": "feet_ground_contact",
                "foot_indices": (2, 3),  # RL, RR — rear feet
            },
        ),
        "feet_air_time": RewardTermCfg(
            func=go2_mdp.handstand_feet_air_time, weight=2.0,
            params={
                "sensor_name": "feet_ground_contact",
                "foot_indices": (2, 3),  # RL, RR — rear feet
            },
        ),
        # Foot clearance (sinusoidal target for front feet)
        "feet_clearance": RewardTermCfg(
            func=go2_mdp.handstand_feet_clearance, weight=0.4,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
                "target_foot_height": 0.06,
                "cycle_time": 1.6,
            },
        ),
        # Default pose shaping — penalize deviation from reachable PD defaults.
        # These match HANDSTAND_INIT_STATE joint_pos, mapped from IsaacGym:
        #   IsaacGym calf=1.5 → MJCF calf=-1.1 (slightly flexed from max -0.84)
        #   IsaacGym calf=-1.75 → MJCF calf=-2.0 (folded)
        # MJCF joint order: FR, FL, RL, RR (each: hip, thigh, calf)
        "default_pos": RewardTermCfg(
            func=go2_mdp.default_joint_penalty, weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "desire_joint_angles": [
                    0.0, -1.0, -1.1,    # FR: hip, thigh (backward), calf (extended)
                    0.0, -1.0, -1.1,    # FL: hip, thigh (backward), calf (extended)
                    0.0, 2.25, -2.0,     # RL: hip, thigh (tucked up), calf (folded)
                    0.0, 2.25, -2.0,     # RR: hip, thigh (tucked up), calf (folded)
                ],
            },
        ),
        # Gated reward: reward matching handstand desire angles when quality high.
        # Desire rear thigh=2.25 (tucked up), front thigh=-1.0 (backward support).
        "default_pos_reward": RewardTermCfg(
            func=go2_mdp.handstand_default_pos_reward, weight=0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "desire_joint_angles": [
                    0.0, -1.0, -1.1,    # FR: hip, thigh (backward), calf (extended)
                    0.0, -1.0, -1.1,    # FL: hip, thigh (backward), calf (extended)
                    0.0, 2.25, -2.0,     # RL: hip, thigh (tucked up), calf (folded)
                    0.0, 2.25, -2.0,     # RR: hip, thigh (tucked up), calf (folded)
                ],
            },
        ),
        "default_hip_pos": RewardTermCfg(
            func=go2_mdp.default_hip_pos, weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*hip_joint",))},
        ),
        # Regularization
        "action_rate_l2": RewardTermCfg(
            func=envs_mdp.action_rate_l2, weight=-0.05,
        ),
        "dof_acc": RewardTermCfg(
            func=go2_mdp.joint_acceleration, weight=-2.5e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        # Collision
        "base_contact": RewardTermCfg(
            func=go2_mdp.body_contact, weight=-2.0,
            params={"sensor_name": "trunk_ground_touch"},
        ),
        "thigh_collision": RewardTermCfg(
            func=go2_mdp.body_contact, weight=-1.0,
            params={"sensor_name": "thigh_ground_touch"},
        ),
    }

    # Terminations — base contact only (matching IsaacGym)
    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "base_contact": TerminationTermCfg(
            func=go2_mdp.base_contact,
            params={"sensor_name": "trunk_ground_touch", "force_threshold": 1.0},
        ),
    }

    # Events — init state, joint randomization, domain randomization
    events = {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5), "y": (-0.5, 0.5),
                    # z: keep feet near ground. Kinematic base-to-foot is ~0.38m
                    # (pitch 85°), PD sag drops ~0.31m to settle at ~0.07m.
                    # Init at 0.40 puts feet ~2cm above ground.
                    "z": (-0.02, 0.02),
                    # Handstand-focused pitch init: 75°-90° forward. Narrower
                    # range than before — 57° was collapsing because front legs
                    # point backward at lower pitch angles.
                    "pitch": (1.31, 1.57),
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
                # IsaacGym: default_dof_pos * Uniform(0.5, 1.5). For front
                # thigh=-1.0 that's [-1.5,-0.5]. Match with ±0.25 additive
                # (keeps front calves within joint limit [-2.72, -0.84]).
                "position_range": (-0.25, 0.25),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        # Friction randomization (IsaacGym: friction_range [0.2, 0.8])
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
        # Base mass randomization (IsaacGym: added range [-1, 2] kg)
        "base_mass": EventTermCfg(
            mode="startup",
            func=envs_mdp.dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
                "operation": "add",
                "ranges": (-1.0, 2.0),
            },
        ),
        # Base COM offset (IsaacGym: [-0.05, 0.05])
        "base_com": EventTermCfg(
            mode="startup",
            func=envs_mdp.dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "ranges": (-0.05, 0.05),
            },
        ),
        # Push robot — gentle force-based perturbation. Velocity-based pushes
        # (±0.5 rad/s) are 10-20x too strong for a learning policy — they
        # immediately kill envs (episode length collapses to 0.4s). Force-based
        # pushes at ±20N provide gentle exploration without destabilizing.
        "push_robot": EventTermCfg(
            func=envs_mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(4.0, 8.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
                "force_range": (-20.0, 20.0),
                "torque_range": (-8.0, 8.0),
            },
        ),
    }

    # Play mode overrides
    if play:
        observations["actor"].enable_corruption = False
        events = {k: v for k, v in events.items() if k in ("reset_base", "reset_robot_joints")}
        for cmd_cfg in commands.values():
            if hasattr(cmd_cfg, 'resampling_time_range'):
                cmd_cfg.resampling_time_range = (100.0, 100.0)

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_go2_handstand_robot_cfg()},
            sensors=(feet_ground_cfg, trunk_head_ground_cfg, thigh_ground_cfg),
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
