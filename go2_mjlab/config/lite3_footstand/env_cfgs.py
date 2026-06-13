"""DeepRobotics Lite3 footstand (rear-leg standing) environment.

Reward design ported from handstand_gym `Lite3_stand`
(legged_gym/envs/lite3_stand/Lite3_handstand_Config.py).  Despite the
source file naming, that task is a FOOTSTAND: ``target_gravity =
(-1, 0, 0)`` pitches the nose UP, the FRONT feet (FL, FR) are rewarded
for being airborne, and the robot balances/walks on its hind legs.

Faithful-port notes (source → here):

* Rewards.  All 12 active terms carried over 1:1 — weights, kernels and
  the tracking gate (base-height reward > 0.8).  The source gated
  tracking on the BATCH MEAN of the height reward (a global curriculum
  switch artifact of IsaacGym buffers); here the gate is per-env.
* Desire pose.  HipX 0, front HipY -1.75, hind HipY -2.25, Knee 1.75 —
  used by the ``default_pos`` L1 penalty (`descire_joint_angles` in the
  source).  Same sign convention: the source URDF and our MJCF both use
  -x/-y/-y joint axes.
* Init pose.  Front HipY -0.8, hind HipY -1.0, Knee 1.5, HipX ∓0.1
  (see ``FOOTSTAND_INIT_STATE``).  Spawn z raised 0.22 → 0.30 because
  MuJoCo does not auto-depenetrate the IsaacGym crouch spawn.
* Control.  Kp=40 / Kd=1, action_scale 0.25 UNIFORM (the source uses
  0.25 for all joints — unlike the Lite3 handstand task's 0.125 HipX),
  sim dt 0.005, decimation 4, episode 20 s.
* Commands.  lin_vel_x/y ±0.8 m/s, yaw ±0.8 rad/s, resample every 5 s.
* DR.  Source choices preserved: friction randomization ON
  (0.5–1.25), base-mass randomization OFF, push every 15 s with
  ±0.5 m/s xy velocity.
* Termination.  TORSO ground contact (>1 N) or timeout — same as source
  (`terminate_after_contacts_on = ["TORSO"]`).
* Observations.  NOT from the source (which used a 48-dim single frame
  with gait-phase sin/cos for a disabled reward).  We keep the repo's
  Lite3 contract — 45-dim frame × 10-frame history (ang_vel, gravity,
  cmd, joint_pos, joint_vel, action) — so existing deploy tooling works.
"""

from go2_mjlab.robots.lite3_constants import get_lite3_footstand_robot_cfg
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


# ---- Lite3 footstand desire-pose constants ---------------------------------
# Source `descire_joint_angles`: the target pose the default_pos L1 penalty
# pulls toward.  Front legs tucked (HipY -1.75), hind legs folded under the
# body for support (HipY -2.25), all knees at 1.75.
_FRONT_DESIRE = (0.0, -1.75, 1.75)   # FL / FR (airborne in footstand)
_REAR_DESIRE = (0.0, -2.25, 1.75)    # HL / HR (support legs)
# Full 12-joint desire vector in MJCF order (FL, FR, HL, HR).
LITE3_FOOTSTAND_DESIRE_JOINT_ANGLES = list(_FRONT_DESIRE * 2 + _REAR_DESIRE * 2)

# Source rewards.base_height_target.
LITE3_FOOTSTAND_BASE_HEIGHT_TARGET = 0.52


def unitree_lite3_footstand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Lite3 footstand (rear-leg standing) configuration."""
    foot_names = ("FL", "FR", "HL", "HR")
    geom_names = tuple(f"{n}_FOOT_collision" for n in foot_names)

    # Sensors — same set as the Lite3 handstand task.
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce",
        num_slots=1, track_air_time=True,
    )
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
    calf_ground_cfg = ContactSensorCfg(
        name="calf_ground_touch",
        primary=ContactMatch(
            mode="geom", entity="robot",
            pattern=tuple(f"{n}_SHANK_collision" for n in foot_names),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="none", num_slots=1, history_length=4,
    )

    # Observations — repo Lite3 contract (45-dim frame, 10-frame history).
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

    # Actions — source action_scale = 0.25 uniform.
    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot", actuator_names=(".*",),
            scale=0.25, use_default_offset=True,
        )
    }

    # Velocity commands — source ranges ±0.8, resample 5 s.
    commands = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.8, 0.8),
                lin_vel_y=(-0.8, 0.8),
                ang_vel_z=(-0.8, 0.8),
            ),
            resampling_time_range=(5.0, 5.0),
            heading_command=False,
        ),
    }

    # ---- Rewards (1:1 port of Lite3_stand reward scales) ----
    rewards = {
        "footstand_orientation": RewardTermCfg(
            # exp(-||g_proj - target||^2), source weight 5.0.
            func=go2_mdp.handstand_orientation_exp, weight=5.0,
            params={"target_gravity": (-1.0, 0.0, 0.0), "sharpness": 1.0},
        ),
        "front_feet_on_air": RewardTermCfg(
            func=go2_mdp.handstand_feet_on_air, weight=0.4,
            params={
                "sensor_name": "feet_ground_contact",
                "foot_indices": (0, 1),   # FL, FR — front feet (airborne)
            },
        ),
        "base_height": RewardTermCfg(
            func=go2_mdp.footstand_base_height_exp, weight=0.6,
            params={"target_height": LITE3_FOOTSTAND_BASE_HEIGHT_TARGET},
        ),
        "tracking_lin_vel": RewardTermCfg(
            func=go2_mdp.footstand_tracking_lin_vel, weight=2.5,
            params={
                "command_name": "twist",
                "target_height": LITE3_FOOTSTAND_BASE_HEIGHT_TARGET,
            },
        ),
        "tracking_ang_vel": RewardTermCfg(
            func=go2_mdp.footstand_tracking_ang_vel, weight=2.5,
            params={
                "command_name": "twist",
                "target_height": LITE3_FOOTSTAND_BASE_HEIGHT_TARGET,
            },
        ),
        "lin_vel_z": RewardTermCfg(
            # Body x is the vertical axis when standing — same kernel as the
            # source `_reward_lin_vel_z` = exp(-|v_b_x| * 10).
            func=go2_mdp.handstand_lin_vel_z, weight=0.2,
        ),
        "ang_vel_xy": RewardTermCfg(
            # Source `_reward_ang_vel_xy` = exp(-||w_b[1:3]||).
            func=go2_mdp.handstand_ang_vel_yz, weight=0.2,
        ),
        "torques": RewardTermCfg(
            func=go2_mdp.handstand_torques, weight=-2.0e-4,
        ),
        "dof_acc": RewardTermCfg(
            # Same finite-difference (no dt) kernel as IsaacGym _reward_dof_acc.
            func=go2_mdp.joint_acceleration, weight=-5.5e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "action_rate_l2": RewardTermCfg(
            func=envs_mdp.action_rate_l2, weight=-0.01,
        ),
        "default_pos": RewardTermCfg(
            func=go2_mdp.default_joint_penalty, weight=-0.3,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "desire_joint_angles": LITE3_FOOTSTAND_DESIRE_JOINT_ANGLES,
            },
        ),
        # Source `collision` (-1.0 on THIGH+SHANK) split across the two sensors.
        "thigh_collision": RewardTermCfg(
            func=go2_mdp.body_contact, weight=-1.0,
            params={"sensor_name": "thigh_ground_touch"},
        ),
        "calf_collision": RewardTermCfg(
            func=go2_mdp.body_contact, weight=-1.0,
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
                # Source resets root lin/ang velocity uniformly in ±0.5.
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
        # Source friction_range = [0.5, 1.25] (IsaacGym scalar = slide).
        "foot_friction_slide": EventTermCfg(
            mode="startup",
            func=envs_mdp.dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
                "operation": "abs",
                "axes": [0],
                "ranges": (0.5, 1.25),
                "shared_random": True,
            },
        ),
        # Source push: every 15 s, xy velocity within ±0.5 m/s.
        "push_robot": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(15.0, 15.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                },
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
            entities={"robot": get_lite3_footstand_robot_cfg()},
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
