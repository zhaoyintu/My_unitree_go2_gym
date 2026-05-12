"""RL-deploy-aligned Lite3 handstand Mjlab environment config.

This task keeps the current Lite3 handstand reward shape, but pins the
policy contract and nominal MuJoCo dynamics to the C++ `rl_deploy_handstand`
runner:

  ang_vel, projected_gravity, cmd, joint_pos_rel, joint_vel, last_action

The actor therefore sees 45 values per frame and uses Mjlab's term-major
10-frame history flattening for a 450-dim policy input.  The critic keeps
base linear velocity and contact state as privileged training-only inputs.
"""

import math

from go2_mjlab import mdp as go2_mdp
from go2_mjlab.config.lite3_handstand.env_cfgs import unitree_lite3_handstand_env_cfg
from go2_mjlab.robots.lite3_constants import get_lite3_rldeploy_handstand_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import CurriculumTermCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


RLDEPLOY_ACTOR_TERM_ORDER = (
    "base_ang_vel",
    "projected_gravity",
    "commands",
    "joint_pos",
    "joint_vel",
    "actions",
)

RLDEPLOY_SINGLE_OBS_DIM = 45
RLDEPLOY_HISTORY_LENGTH = 10
RLDEPLOY_ACTOR_OBS_DIM = RLDEPLOY_SINGLE_OBS_DIM * RLDEPLOY_HISTORY_LENGTH
BODY_CLEARANCE_SENSOR_NAMES = (
    "trunk_ground_touch",
    "thigh_ground_touch",
    "calf_ground_touch",
)
ROBOTLAB_BODY_CLEARANCE_SENSOR_NAMES = (
    "trunk_ground_touch",
    "thigh_ground_touch",
)
LITE3_LINK_INERTIA_BODY_NAMES = (
    "FL_THIGH",
    "FL_SHANK",
    "FR_THIGH",
    "FR_SHANK",
    "HL_THIGH",
    "HL_SHANK",
    "HR_THIGH",
    "HR_SHANK",
)
LITE3_LINK_INERTIA_ALPHA_RANGE = (math.log(0.9) / 2.0, math.log(1.1) / 2.0)


def _rldeploy_actor_terms() -> dict[str, ObservationTermCfg]:
    terms = {
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
    return {name: terms[name] for name in RLDEPLOY_ACTOR_TERM_ORDER}


def unitree_lite3_handstand_rldeploy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Lite3 handstand task aligned with `rl_deploy_handstand`."""
    cfg = unitree_lite3_handstand_env_cfg(play=play)
    cfg.scene.entities["robot"] = get_lite3_rldeploy_handstand_robot_cfg()
    cfg.commands["twist"].rel_standing_envs = 0.25

    actor_terms = _rldeploy_actor_terms()
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

    if "foot_friction_slide" in cfg.events:
        cfg.events["foot_friction_slide"].params["ranges"] = (0.8, 1.2)
    cfg.events.pop("foot_friction_spin", None)
    cfg.events.pop("foot_friction_roll", None)

    rewards = cfg.rewards
    static_asset_cfg = SceneEntityCfg("robot", body_names=("TORSO",))
    soft_quality_params = {
        "target_gravity": (1.0, 0.0, 0.0),
        "orientation_sharpness": 2.0,
        "base_height_target": 0.39,
        "rear_foot_target_height": 0.56,
        "rear_foot_lift_min": 0.08,
        "rear_foot_indices": (2, 3),
        "foot_site_names": ("FL", "FR", "HL", "HR"),
        "sensor_name": "feet_ground_contact",
        "asset_cfg": static_asset_cfg,
    }

    # Static-first reward recipe: make the front-paw handstand pose pay before
    # command tracking can dominate.  The failed RLDeploy run earned high
    # tracking reward while prone because the old handstand gate was always open.
    rewards["handstand_orientation"].func = go2_mdp.handstand_orientation_exp
    rewards["handstand_orientation"].weight = 2.0
    rewards["handstand_orientation"].params = {
        "target_gravity": (1.0, 0.0, 0.0),
        "sharpness": 2.0,
    }
    rewards["handstand_feet_on_air"].weight = 1.0
    rewards["handstand_feet_height_exp"].func = go2_mdp.handstand_rear_feet_height_static
    rewards["handstand_feet_height_exp"].weight = 8.0
    rewards["handstand_feet_height_exp"].params = {
        "target_height": 0.56,
        "foot_indices": (2, 3),
        "foot_site_names": ("FL", "FR", "HL", "HR"),
        "rear_foot_lift_min": 0.08,
        "rear_foot_height_sharpness": 4.0,
        "asset_cfg": static_asset_cfg,
    }
    rewards["base_height"].func = go2_mdp.handstand_base_height_soft
    rewards["base_height"].weight = 0.8
    rewards["base_height"].params = {
        "target_height": 0.39,
        "target_gravity": (1.0, 0.0, 0.0),
        "orientation_sharpness": 2.0,
        "asset_cfg": static_asset_cfg,
    }
    rewards["tracking_lin_vel"].func = go2_mdp.handstand_tracking_lin_vel_soft_gate
    rewards["tracking_lin_vel"].params = {"command_name": "twist", **soft_quality_params}
    rewards["tracking_ang_vel"].func = go2_mdp.handstand_tracking_ang_vel_soft_gate
    rewards["tracking_ang_vel"].params = {"command_name": "twist", **soft_quality_params}
    rewards["tracking_lin_vel_zero"].func = go2_mdp.handstand_tracking_lin_vel_zero_soft_gate
    rewards["tracking_lin_vel_zero"].params = {"command_name": "twist", **soft_quality_params}
    rewards["tracking_ang_vel_zero"].func = go2_mdp.handstand_tracking_ang_vel_zero_soft_gate
    rewards["tracking_ang_vel_zero"].params = {"command_name": "twist", **soft_quality_params}
    rewards["contact"].func = go2_mdp.handstand_stance_contact_mean
    rewards["contact"].weight = 0.8
    rewards["contact"].params = {
        "sensor_name": "feet_ground_contact",
        "foot_indices": (0, 1),
    }
    rewards["feet_air_time"].weight = 0.0
    rewards["feet_clearance"].weight = 0.0
    rewards["default_pos"].weight = -0.6
    rewards["default_pos_reward"].weight = 2.0

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
            history_length=RLDEPLOY_HISTORY_LENGTH,
            flatten_history_dim=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    cfg.sim.mujoco.timestep = 0.001
    cfg.decimation = 20
    return cfg


def _add_rldeploy_support_gate_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    rewards = cfg.rewards
    support_params = {
        "stance_foot_indices": (0, 1),
        "body_clearance_sensor_names": BODY_CLEARANCE_SENSOR_NAMES,
    }
    for reward_name in (
        "tracking_lin_vel",
        "tracking_ang_vel",
        "tracking_lin_vel_zero",
        "tracking_ang_vel_zero",
    ):
        rewards[reward_name].params.update(support_params)

    rewards["handstand_feet_height_exp"].params.update(
        {
            "stance_sensor_name": "feet_ground_contact",
            "stance_foot_indices": (0, 1),
            "body_clearance_sensor_names": BODY_CLEARANCE_SENSOR_NAMES,
        }
    )
    rewards["contact"].weight = 2.0
    rewards["base_contact"].weight = -6.0
    rewards["thigh_collision"].weight = -4.0
    rewards["calf_collision"].weight = -4.0


def _add_rldeploy_sim2real_dr_events(cfg: ManagerBasedRlEnvCfg) -> None:
    """Add sim-to-real DR events supported by mjlab's stock event API."""
    actuator_asset_cfg = SceneEntityCfg("robot")
    joint_asset_cfg = SceneEntityCfg("robot", joint_names=(".*",))
    link_inertia_asset_cfg = SceneEntityCfg("robot", body_names=LITE3_LINK_INERTIA_BODY_NAMES)

    cfg.events["encoder_bias"].params["bias_range"] = (-0.02, 0.02)
    cfg.events.update(
        {
            "pd_gains": EventTermCfg(
                mode="startup",
                func=envs_mdp.dr.pd_gains,
                params={
                    "kp_range": (0.9, 1.1),
                    "kd_range": (0.9, 1.1),
                    "operation": "scale",
                    "asset_cfg": actuator_asset_cfg,
                },
            ),
            "motor_strength": EventTermCfg(
                mode="startup",
                func=envs_mdp.dr.effort_limits,
                params={
                    "effort_limit_range": (0.8, 1.2),
                    "operation": "scale",
                    "asset_cfg": actuator_asset_cfg,
                },
            ),
            "joint_friction": EventTermCfg(
                mode="startup",
                func=envs_mdp.dr.joint_friction,
                params={
                    "ranges": (0.01, 0.2),
                    "operation": "abs",
                    "asset_cfg": joint_asset_cfg,
                },
            ),
            "joint_damping": EventTermCfg(
                mode="startup",
                func=envs_mdp.dr.joint_damping,
                params={
                    "ranges": (0.0, 0.2),
                    "operation": "abs",
                    "asset_cfg": joint_asset_cfg,
                },
            ),
            "joint_armature": EventTermCfg(
                mode="startup",
                func=envs_mdp.dr.joint_armature,
                params={
                    "ranges": (0.005, 0.015),
                    "operation": "abs",
                    "asset_cfg": joint_asset_cfg,
                },
            ),
            "link_inertia": EventTermCfg(
                mode="startup",
                func=envs_mdp.dr.pseudo_inertia,
                params={
                    "alpha_range": LITE3_LINK_INERTIA_ALPHA_RANGE,
                    "asset_cfg": link_inertia_asset_cfg,
                },
            ),
        }
    )


def unitree_lite3_handstand_rldeploy_dr_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the RLDeploy Lite3 handstand task with additional sim-to-real DR."""
    cfg = unitree_lite3_handstand_rldeploy_env_cfg(play=play)
    _add_rldeploy_support_gate_rewards(cfg)
    if not play:
        _add_rldeploy_sim2real_dr_events(cfg)
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create a RobotLab-style Lite3 handstand task on the RLDeploy contract."""
    cfg = unitree_lite3_handstand_rldeploy_env_cfg(play=play)

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.rel_standing_envs = 0.25
    twist_cmd.ranges.lin_vel_x = (-0.4, 0.4)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.4, 0.4)

    static_asset_cfg = SceneEntityCfg("robot", body_names=("TORSO",))
    support_params = {
        "sensor_name": "feet_ground_contact",
        "stance_foot_indices": (0, 1),
        "body_clearance_sensor_names": ROBOTLAB_BODY_CLEARANCE_SENSOR_NAMES,
        "target_gravity": (1.0, 0.0, 0.0),
        "orientation_sharpness": 2.0,
        "base_height_target": 0.39,
        "rear_foot_target_height": 0.56,
        "rear_foot_lift_min": 0.08,
        "rear_foot_indices": (2, 3),
        "foot_site_names": ("FL", "FR", "HL", "HR"),
        "asset_cfg": static_asset_cfg,
        "support_floor": 0.05,
    }

    rewards = cfg.rewards
    rewards["handstand_orientation"].func = go2_mdp.handstand_orientation
    rewards["handstand_orientation"].weight = -1.0
    rewards["handstand_orientation"].params = {
        "target_gravity": (1.0, 0.0, 0.0),
    }
    rewards["handstand_feet_height_exp"].func = go2_mdp.handstand_feet_height_l2_exp
    rewards["handstand_feet_height_exp"].weight = 10.0
    rewards["handstand_feet_height_exp"].params = {
        "target_height": 0.56,
        "std": math.sqrt(0.25),
        "foot_indices": (2, 3),
        "foot_site_names": ("FL", "FR", "HL", "HR"),
        "stance_sensor_name": "feet_ground_contact",
        "stance_foot_indices": (0, 1),
        "body_clearance_sensor_names": ROBOTLAB_BODY_CLEARANCE_SENSOR_NAMES,
        "support_floor": 0.05,
        "asset_cfg": static_asset_cfg,
    }
    rewards["handstand_feet_on_air"].weight = 5.0
    rewards["handstand_feet_on_air"].params = {
        "sensor_name": "feet_ground_contact",
        "foot_indices": (2, 3),
    }
    rewards["base_height"].func = go2_mdp.handstand_base_height_soft
    rewards["base_height"].weight = 0.8
    rewards["base_height"].params = {
        "target_height": 0.39,
        "target_gravity": (1.0, 0.0, 0.0),
        "orientation_sharpness": 2.0,
        "asset_cfg": static_asset_cfg,
    }
    rewards["tracking_lin_vel"].func = go2_mdp.handstand_tracking_lin_vel_soft_gate
    rewards["tracking_lin_vel"].weight = 3.0
    rewards["tracking_lin_vel"].params = {"command_name": "twist", **support_params}
    rewards["tracking_ang_vel"].func = go2_mdp.handstand_tracking_ang_vel_soft_gate
    rewards["tracking_ang_vel"].weight = 1.5
    rewards["tracking_ang_vel"].params = {"command_name": "twist", **support_params}
    rewards["tracking_lin_vel_zero"].func = go2_mdp.handstand_tracking_lin_vel_zero_soft_gate
    rewards["tracking_lin_vel_zero"].weight = -0.4
    rewards["tracking_lin_vel_zero"].params = {"command_name": "twist", **support_params}
    rewards["tracking_ang_vel_zero"].func = go2_mdp.handstand_tracking_ang_vel_zero_soft_gate
    rewards["tracking_ang_vel_zero"].weight = -0.4
    rewards["tracking_ang_vel_zero"].params = {"command_name": "twist", **support_params}
    rewards["contact"].func = go2_mdp.handstand_stance_contact_mean
    rewards["contact"].weight = 2.0
    rewards["contact"].params = {
        "sensor_name": "feet_ground_contact",
        "foot_indices": (0, 1),
    }
    rewards["feet_air_time"].weight = 0.0
    rewards["feet_clearance"].weight = 0.0
    rewards["base_contact"].weight = -4.0
    rewards["thigh_collision"].weight = -2.0
    rewards["calf_collision"].weight = -2.0

    cfg.terminations["base_contact"] = TerminationTermCfg(
        func=go2_mdp.base_contact,
        params={"sensor_name": "trunk_ground_touch", "force_threshold": 1.0},
    )
    cfg.terminations["thigh_contact"] = TerminationTermCfg(
        func=go2_mdp.base_contact,
        params={"sensor_name": "thigh_ground_touch"},
    )
    # Lite3 shank collision can touch the ground with the nominal foot contact.
    # Keep it as a reward penalty, but do not use it as a hard termination.

    if play:
        cfg.curriculum = {}
        twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
        twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
        twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
    else:
        assert "base_mass" in cfg.events
        assert "base_com" in cfg.events
        cfg.events["encoder_bias"].params["bias_range"] = (-0.02, 0.02)
        _add_rldeploy_sim2real_dr_events(cfg)
        cfg.curriculum["command_vel"] = CurriculumTermCfg(
            func=velocity_mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {
                        "step": 0,
                        "lin_vel_x": (-0.4, 0.4),
                        "lin_vel_y": (0.0, 0.0),
                        "ang_vel_z": (-0.4, 0.4),
                    },
                    {
                        "step": 2000 * 24,
                        "lin_vel_x": (-0.6, 0.6),
                        "lin_vel_y": (-0.2, 0.2),
                        "ang_vel_z": (-0.6, 0.6),
                    },
                    {
                        "step": 5000 * 24,
                        "lin_vel_x": (-0.8, 0.8),
                        "lin_vel_y": (-0.5, 0.5),
                        "ang_vel_z": (-0.8, 0.8),
                    },
                    {
                        "step": 8000 * 24,
                        "lin_vel_x": (-1.0, 1.0),
                        "lin_vel_y": (-1.0, 1.0),
                        "ang_vel_z": (-1.0, 1.0),
                    },
                ],
            },
        )

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """RobotLab task variant with weaker desired-joint-pose shaping."""
    cfg = unitree_lite3_handstand_rldeploy_robotlab_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["default_pos"].weight = -0.15
    rewards["default_pos_reward"].weight = 0.4
    rewards["default_hip_pos"].weight = -0.1

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Low-default-pose variant with stronger zero-command stance stability."""
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["zero_stance_contact"] = RewardTermCfg(
        func=go2_mdp.handstand_stance_contact_zero,
        weight=1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "foot_indices": (0, 1),
            "command_name": "twist",
            "moving_threshold": 0.1,
        },
    )
    rewards["zero_joint_vel"] = RewardTermCfg(
        func=go2_mdp.handstand_joint_vel_zero,
        weight=-0.02,
        params={
            "command_name": "twist",
            "moving_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )
    rewards["lin_vel_z"].weight = 0.4
    rewards["action_rate_l2"].weight = -0.08

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Low-default-pose zero-stance variant with stronger quiet standing costs."""
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_env_cfg(play=play)

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.rel_standing_envs = 0.40

    rewards = cfg.rewards
    rewards["zero_stance_contact"].weight = 3.0
    rewards["zero_joint_vel"].weight = -0.08
    rewards["dof_acc"].weight = -1.0e-3
    rewards["action_rate_l2"].weight = -0.12
    rewards["lin_vel_z"].weight = 0.6

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Quiet zero-stance variant with an unconditional anti-hop penalty.

    Adds a single reward, ``stance_air_penalty``, that fires whenever both
    front (stance) feet are airborne while the body is in handstand pose.
    Applies in both zero-command and non-zero-command phases so the policy
    cannot satisfy velocity tracking by hopping.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["feet_air_time"].weight = 0.0
    rewards["stance_air_penalty"] = RewardTermCfg(
        func=go2_mdp.handstand_stance_air_penalty,
        weight=-2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "foot_indices": (0, 1),
            "target_height": 0.08,
        },
    )

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_step_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Quiet zero-stance variant that encourages alternating front-paw steps when moving."""
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_env_cfg(play=play)

    rewards = cfg.rewards
    moving_contact_params = {
        "sensor_name": "feet_ground_contact",
        "foot_indices": (0, 1),
        "command_name": "twist",
        "moving_threshold": 0.1,
    }
    rewards["contact"].weight = 0.0
    rewards["moving_single_stance_contact"] = RewardTermCfg(
        func=go2_mdp.handstand_contact,
        weight=2.5,
        params=moving_contact_params,
    )
    rewards["moving_both_front_feet_air"] = RewardTermCfg(
        func=go2_mdp.handstand_moving_no_stance_contact,
        weight=-2.0,
        params=moving_contact_params,
    )
    rewards["moving_double_front_contact"] = RewardTermCfg(
        func=go2_mdp.handstand_moving_double_stance_contact,
        weight=-0.5,
        params=moving_contact_params,
    )
    rewards["feet_air_time"].weight = 0.5
    rewards["feet_air_time"].params["command_name"] = "twist"
    rewards["feet_air_time"].params["moving_threshold"] = 0.1

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """RobotLab task variant with desired-joint-pose shaping disabled."""
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["default_pos"].weight = 0.0
    rewards["default_pos_reward"].weight = 0.0
    rewards["default_hip_pos"].weight = 0.0

    return cfg
