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
from go2_mjlab.robots.lite3_constants import (
    get_lite3_rldeploy_handstand_robot_cfg,
    get_lite3_terrain_handstand_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import CurriculumTermCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import flat, random_rough, wave_terrain
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


# ---- Mild varied terrain for blind handstand robustness --------------------
# The handstand policy has NO terrain-height perception (blind) — it treats
# the ground as a disturbance and relies on proprioception + DR.  So the
# terrain must be GENTLE and SMOOTH.
#
# IMPORTANT (NaN fix): the first attempt mixed in `random_rough`
# (HfRandomUniformTerrainCfg), which builds a GRID of independently random
# cell heights → near-vertical step discontinuities between cells.  A foot
# landing on such an edge with the stiff RLDeploy contact (solref 0.005)
# produces an exploding contact force → NaN (the run crashed at iter ~43
# with high thigh/calf-contact terminations).  Use only SMOOTH undulation
# (waves) whose gradual slopes the stiff contact handles cleanly, paired with
# the heightfield contact-robustness sim params set in the env cfg below.
# curriculum=False → every patch is a random draw, so all 4096 envs see
# varied ground from iteration 0 (we are finetuning, not learning from scratch).
LITE3_HANDSTAND_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    curriculum=False,
    sub_terrains={
        "flat": flat(proportion=0.6),
        "wave": wave_terrain(
            proportion=0.4, amplitude_range=(0.0, 0.015), num_waves=4,
        ),
    },
    add_lights=True,
)


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


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """NoHop variant tuned for slower, larger-amplitude stepping gait.

    Builds on the NoHop policy by:
      * Re-enabling ``feet_air_time`` (front feet, moving-only) so each
        swing leg is rewarded for staying airborne ≥0.4 s, pushing the
        policy toward longer strides instead of rapid micro-shuffle.
      * Strengthening ``dof_acc`` and ``action_rate_l2`` penalties to
        suppress the high-frequency twitches that the NoHop baseline
        learned for balance/locomotion.
      * Bumping ``stance_air_penalty`` to drop the ~14% moving-phase
        both-front-airborne rate observed in the NoHop policy.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["feet_air_time"].weight = 0.5
    rewards["feet_air_time"].params["command_name"] = "twist"
    rewards["feet_air_time"].params["moving_threshold"] = 0.1
    rewards["dof_acc"].weight = -2.0e-3
    rewards["action_rate_l2"].weight = -0.16
    rewards["stance_air_penalty"].weight = -3.0

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """BigStep variant that breaks out of the residual hop basin.

    BigStep v2 trained to high tracking (~2.0) but locked into a double-foot
    hop at ~15% both-front-airborne rate. ``feet_air_time`` (≥0.4 s swing)
    never paid out so it produced no gradient toward stepping. This variant
    adds an active pull toward single-stance gait phase and a stronger
    push out of hops:

      * ``single_stance_contact`` (+1.5): reward exactly one front foot
        planted while moving and in handstand. Same function the QuietStep
        attempt used, but at moderate weight without the toxic
        ``moving_double_stance`` penalty that broke that experiment.
      * ``stance_air_penalty`` -3.0 → -5.0: nearly double the cost of both
        front feet airborne.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["single_stance_contact"] = RewardTermCfg(
        func=go2_mdp.handstand_contact,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "foot_indices": (0, 1),
            "command_name": "twist",
            "moving_threshold": 0.1,
        },
    )
    rewards["stance_air_penalty"].weight = -5.0

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v2_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride v4: stretch the step amplitude, slow the step frequency.

    Stride v3 trained to 0% hop but the policy farmed ``single_stance_contact``
    by lifting front feet for only a couple of physics steps (~0.04 s),
    yielding a rapid micro-shuffle gait. This variant:

      * Reduces ``single_stance_contact`` (1.5 → 0.3) so the brief-lift
        farm is no longer the easy strategy.
      * Triples ``feet_air_time`` (0.5 → 1.5) so a ≥0.4 s single-leg swing
        is meaningfully more rewarding than a brief tiptoe.
      * Stiffens ``dof_acc`` (-2e-3 → -3e-3) so rapid joint motion pays
        more, nudging the policy toward fewer-but-larger leg cycles.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["single_stance_contact"].weight = 0.3
    rewards["feet_air_time"].weight = 1.5
    rewards["dof_acc"].weight = -3.0e-3

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v3_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride v5: force a 1 Hz sinusoidal stepping rhythm via feet_clearance.

    Stride v4 trained to a stable no-hop shuffle but the joints still
    moved at high frequency — visually a rapid drag rather than a stride.
    Free-form reward shaping (air_time threshold, single-stance bonus)
    failed to surface a long-swing gait. This variant re-enables the
    explicit ``handstand_feet_clearance`` sinusoid that forces each front
    foot to track a |sin(2π t)|·0.06 m world-z target during its half of
    a 1.0 s cycle, and pulls back ``single_stance_contact`` so the two
    signals don't fight.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v2_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["feet_clearance"].weight = 0.6
    rewards["feet_clearance"].params["cycle_time"] = 1.0
    rewards["feet_clearance"].params["command_name"] = "twist"
    rewards["feet_clearance"].params["moving_threshold"] = 0.1
    rewards["single_stance_contact"].weight = 0.1

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v4_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride v6: remove the ``contact`` mean reward that punishes stepping.

    Stride V3 inherited ``contact`` (``handstand_stance_contact_mean``) at
    weight +2.0 from the RobotLab base. That term scores 1.0 for both
    front feet planted and 0.5 for a single foot — so every swing forfeits
    a hefty per-step bonus, swamping the +1.5 ``feet_air_time`` and +0.6
    ``feet_clearance`` signals that should drive stepping. Anti-hop is
    fully covered by ``stance_air_penalty -5.0`` and quiet zero-command
    stance by ``zero_stance_contact +3.0``, so the ``contact`` term is
    redundant. This variant disables it so the swing-encouraging signals
    can dominate when the command is non-zero.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v3_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["contact"].weight = 0.0

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v5_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride v7: bigger and slower swing target.

    Stride V4 trained to a non-shuffle stepping gait but the lift looked
    visually small. The previous sinusoidal target was 0.06 m peak on a
    1.0 s cycle — about 6 cm foot lift at 1 Hz step frequency. This
    variant doubles the lift amplitude and slows the rhythm so each step
    looks like a deliberate stride:

      * ``target_foot_height`` 0.06 → 0.12 m (≈30° hip swing for Lite3)
      * ``cycle_time`` 1.0 → 1.5 s (0.75 s swing duration per foot)
      * ``feet_clearance.weight`` 0.6 → 1.0 (stronger pull on the target)
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v4_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["feet_clearance"].weight = 1.0
    rewards["feet_clearance"].params["target_foot_height"] = 0.12
    rewards["feet_clearance"].params["cycle_time"] = 1.5

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v6_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride v8: amplify swing rewards to push past the V5 plateau.

    Stride V5 trained to a working stepping gait but ``feet_clearance``
    stalled at ~0.36/1.0 — the policy was only achieving 30-40% of the
    sinusoidal target. Doubling the long-swing payouts gives the policy
    a stronger reason to invest joint energy in deeper, longer lifts:

      * ``feet_air_time`` 1.5 → 3.0 (≥0.4 s single-leg swings worth 2x)
      * ``feet_clearance`` 1.0 → 1.5 (sinusoidal tracking worth 1.5x)
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v5_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["feet_air_time"].weight = 3.0
    rewards["feet_clearance"].weight = 1.5

    return cfg


# Stride v7 lower-HipY desire pose: front legs sit at the middle of the
# HipY range instead of the upper limit, leaving ±0.2 rad of headroom on
# both sides so the policy can swing the front leg forward during a step.
# Rear stays at the Lite3 normal standing rest pose.
_STRIDE_FRONT_DESIRE_OPEN = (0.0, 0.1, 2.0)
_STRIDE_REAR_DESIRE = (0.0, -0.8, 1.6)
LITE3_STRIDE_OPEN_DESIRE_JOINT_ANGLES = list(
    _STRIDE_FRONT_DESIRE_OPEN * 2 + _STRIDE_REAR_DESIRE * 2
)


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v7_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride v9: open HipY headroom so the front legs can swing forward.

    Stride V6 plateaued because the desired HipY (0.283 rad) sits at 90%
    of the joint's upper limit. The default-pose rewards bias the policy
    toward this value, leaving only ~0.03 rad of forward swing range —
    feet can lift but not displace forward, producing a body-slides-past
    rapid-shuffle visual. This variant retargets desired HipY to 0.1 rad
    (middle of range) so ±0.2 rad of swing is available in both directions.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v6_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["default_pos"].params["desire_joint_angles"] = LITE3_STRIDE_OPEN_DESIRE_JOINT_ANGLES
    rewards["default_pos_reward"].params["desire_joint_angles"] = LITE3_STRIDE_OPEN_DESIRE_JOINT_ANGLES

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v8_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride v10: aggressive 4-knob change to break the V7 plateau.

    V7's HipY-open desire pose helped (got past V6's start fast) but
    feet_clearance still plateaued near 0.55. The remaining ceiling is
    a combination of three constraints all pulling the policy away from
    "swing harder":

      * ``dof_acc -3e-3`` and ``action_rate_l2 -0.16`` make rapid joint
        motion expensive — capping the speed of takeoff/landing.
      * ``feet_clearance`` sharpness 10 lets the policy stop tracking
        at ~7 cm error and still pocket 50% of the per-foot reward.
      * ``target_foot_height = 0.12 m`` is the visual ceiling — at 30 cm
        body height it reads as a modest lift.

    Relax the cost terms, tighten the tracking reward, raise the target:

      * ``dof_acc`` -3e-3 → -1e-3 (back to NoHop baseline)
      * ``action_rate_l2`` -0.16 → -0.10 (back to NoHop baseline)
      * ``feet_clearance`` sharpness 10 → 20 (half-error for same score)
      * ``feet_clearance.target_foot_height`` 0.12 → 0.15 m
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v7_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["dof_acc"].weight = -1.0e-3
    rewards["action_rate_l2"].weight = -0.10
    rewards["feet_clearance"].params["target_foot_height"] = 0.15
    rewards["feet_clearance"].params["sharpness"] = 20.0

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v9_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride v11: relax rewards that oppose alternating stride.

    The V8 audit (see ``2026-05-15-stride-v10-reward-audit.md``) flagged
    three reward terms that mildly fight the stepping objective. Rather
    than add yet another push reward, this variant turns those three
    counter-forces down:

      * ``symmetric_joints`` -0.1 → 0.0 — alternating gait is inherently
        asymmetric (one foot up, the other planted), so this penalty was
        constantly pricing the desired behaviour.
      * ``default_pos`` -0.15 → -0.05 and ``default_pos_reward`` 0.4 →
        0.15 — the front leg has to deviate from the static handstand
        target during every swing; halving the pull lets that happen
        cheaply.
      * ``lin_vel_z`` 0.6 → 0.2 — leaves a smaller anti-bob bias so the
        body can dip slightly into each stride, while still discouraging
        full hops (those are covered by ``stance_air_penalty``).
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v8_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["symmetric_joints"].weight = 0.0
    rewards["default_pos"].weight = -0.05
    rewards["default_pos_reward"].weight = 0.15
    rewards["lin_vel_z"].weight = 0.2

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v11_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride v13: restore the default_pose pull that v11 (Stride-V9) halved.

    v11's audit weakened ``default_pos`` (-0.15 → -0.05) and
    ``default_pos_reward`` (0.4 → 0.15) on the theory that they were
    pricing the per-step pose deviation needed for an alternating gait.
    v12's linear-cap clearance experiment then showed that with the pull
    so weak the policy drifts into extreme rear-leg folds — visually
    exposing the 28 mm structural overlap between THIGH and SHANK
    collision meshes (the meshes share the joint-housing geometry, see
    ``deploy_mujoco_viewer/probe_lite3_self_collision.py``). The lever
    we have without remodeling the collision geom is to stop the policy
    from contorting that far in the first place.

    Revert just the two ``default_pos*`` weights to the v10 (Stride-V8)
    values. Keep v11's other audit changes (``symmetric_joints`` 0,
    ``lin_vel_z`` 0.2). ``feet_clearance`` stays at 1.5 (inherited).
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v9_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["default_pos"].weight = -0.15
    rewards["default_pos_reward"].weight = 0.4

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v12_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride v14: sharpen feet_clearance shape so "no lift" stops paying out.

    v13 (Stride-V11) restored visual realism via strong default_pos but
    foot lift plateaued at ~0.30/1.5 — the Gaussian
    ``exp(-sharpness · |z - target|²)`` with sharpness 20 and target
    0.15 m still gives ~64% reward at z=0, so the policy farms the bulk
    of the reward without actually lifting.

    Tighten the same shape (no new reward term, clean comparison):

      * ``target_foot_height`` 0.15 → 0.25 m — visually larger swing.
      * ``sharpness``          20 → 50      — z=0 only earns 4.5%,
                                              z=0.25 earns 100%.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v11_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["feet_clearance"].params["target_foot_height"] = 0.25
    rewards["feet_clearance"].params["sharpness"] = 50.0

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v13_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride v15: drop the rear-foot z target — over-constraint.

    Probing v14 (Stride-V12) at iter 5500 revealed HL_Knee saturated at
    +3.31 rad (past MJCF limit 2.792), with std 0.001 — totally locked.
    The driver: ``handstand_feet_height_exp`` weight +10 commits the
    policy to placing rear feet at world z=0.56 m exactly. The policy
    found that maximally folding HL_Knee lifts the HL rear foot to
    ~0.6 m, satisfying the target via an extreme asymmetric pose. The
    9.0 reward gain trivially outweighs the -0.26 default_pos cost.

    But the rear-foot z target is **redundant** — handstand is already
    fully defined by:

      * ``handstand_orientation`` (-1.0): body pitched into inverted
      * ``base_height``           (+0.8): trunk world-z = 0.39 m
      * ``handstand_feet_on_air`` (+5.0): rear feet off the ground

    Pinning the rear foot to a specific world-z doesn't capture
    anything those three don't already cover, and it dominates the
    reward landscape so default_pos can't enforce a clean pose.

    Disable ``handstand_feet_height_exp`` entirely. The rear leg pose
    is now free for default_pos to pull to nominal (HipY=-0.8, Knee=1.6).
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v12_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["handstand_feet_height_exp"].weight = 0.0

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v14_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride v16: scope default_pos to rear legs only.

    v15 (Stride-V13) removes the rear-foot z target so default_pos can
    finally pull the rear legs back to nominal. But default_pos still
    covers ALL 12 joints — including the FRONT legs that need to
    deviate from nominal during a stride. Every swing pays the
    default_pos cost (front HipY + Knee deviate ~0.5 rad), fighting
    the feet_clearance reward that's trying to push the lift higher.

    Restrict default_pos and default_pos_reward to rear joints only.
    Front legs are now free to swing without paying the static-pose
    cost. Rear legs are still pulled hard to nominal (HipY=-0.8,
    Knee=1.6), which (combined with v15's removal of the rear-foot
    z target) should prevent HL_Knee from drifting to its limit.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v13_env_cfg(play=play)

    rear_only_asset_cfg = SceneEntityCfg(
        "robot",
        joint_names=("HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
                     "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"),
    )
    rear_desire = [0.0, -0.8, 1.6, 0.0, -0.8, 1.6]

    rewards = cfg.rewards
    rewards["default_pos"].params["asset_cfg"] = rear_only_asset_cfg
    rewards["default_pos"].params["desire_joint_angles"] = rear_desire
    rewards["default_pos_reward"].params["asset_cfg"] = rear_only_asset_cfg
    rewards["default_pos_reward"].params["desire_joint_angles"] = rear_desire

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride v17: pull front HipX back to 0 without re-constraining HipY/Knee.

    Stride V14 (v16) frees the front legs by removing them from
    default_pos. Front HipY/Knee can now swing freely for stride,
    BUT FL_HipX and FR_HipX both drift to ~+0.56 rad — visually
    the front legs are yawed 32 degrees in the SAME direction,
    making the handstand look twisted. ``default_hip_pos`` (-0.1)
    is too weak to dominate.

    Extend default_pos asset_cfg to also include FL_HipX and FR_HipX
    with desire 0 (mirror-symmetric centered). Front HipY and Knee
    remain unconstrained so swing is preserved.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v14_env_cfg(play=play)

    asset_cfg = SceneEntityCfg(
        "robot",
        joint_names=("FL_HipX_joint", "FR_HipX_joint",
                     "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
                     "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"),
    )
    desire = [0.0, 0.0,
              0.0, -0.8, 1.6,
              0.0, -0.8, 1.6]

    rewards = cfg.rewards
    rewards["default_pos"].params["asset_cfg"] = asset_cfg
    rewards["default_pos"].params["desire_joint_angles"] = desire
    rewards["default_pos_reward"].params["asset_cfg"] = asset_cfg
    rewards["default_pos_reward"].params["desire_joint_angles"] = desire

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v10_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride v12: replace sinusoidal clearance with linear-with-cap height reward.

    Stride V9 unblocked tracking but ``feet_clearance`` still plateaued at
    ~0.27/1.5. The Gaussian ``exp(-sharpness·|z - sin_target|²)`` shape
    pays ~64% reward at z=0 vs a 0.15 m peak target, so there is no
    gradient pressure against "barely lifting" — the policy converges to
    a low-amplitude shuffle that satisfies the rhythm but not the height.

    Replace the sin shape with a contact-gated linear-with-cap height
    reward:

      * ``feet_clearance`` weight 1.5 → 0.0 — disable the sinusoidal target.
      * ``swing_foot_height`` new term — when a front foot is airborne,
        reward ``clamp(z_world / 0.25, 0, 1)``. Zero reward at z=0, full
        reward at 25 cm lift. Sin rhythm is removed; stride frequency is
        left to ``feet_air_time`` and ``single_stance_contact``.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v9_env_cfg(play=play)

    rewards = cfg.rewards
    rewards["feet_clearance"].weight = 0.0
    rewards["swing_foot_height"] = RewardTermCfg(
        func=go2_mdp.handstand_swing_foot_height_linear,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "asset_cfg": SceneEntityCfg("robot", body_names=("TORSO",)),
            "foot_indices": (0, 1),
            "foot_site_names": ("FL", "FR", "HL", "HR"),
            "cap_height": 0.25,
            "command_name": "twist",
            "moving_threshold": 0.1,
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


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_finetune_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage-2 energy finetune of Stride-V15 (mujoco_playground recipe).

    Ports the Go1Handstand two-stage configuration from
    `mujoco_playground/go1_finetune_report.html`: restore the stage-1
    checkpoint and add a mechanical-power cost so residual hop/jitter —
    previously free — gets priced out while the learned handstand-walk
    keeps paying:

      * ``energy`` -0.003 — sum |qvel_i| * |tau_i| (playground stage-2
        value; both frameworks scale reward terms by dt so the weight
        transfers directly).

    ``dof_acc`` is NOT bumped: the playground finetune uses -2.5e-7 on
    sum(qacc^2), equivalent to -6.25e-4 in our per-policy-step delta-qvel
    units, and Stride-V15 already inherits -1.0e-3 (stronger).

    Train via resume — the runner cfg keeps the Stride-V15 experiment
    name and loads its stage-1 checkpoint.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_env_cfg(play=play)

    cfg.rewards["energy"] = RewardTermCfg(
        func=go2_mdp.energy, weight=-0.003,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_terrain_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride-V15 terrain-robustness finetune: varied friction + mild rough/wavy ground.

    Request: a finetune of the visually-best Stride-V15 handstand-walk that
    is robust to (a) varying ground friction and (b) gently undulating /
    uneven terrain — WITHOUT giving the policy any terrain perception
    (blind).  The actor observation is unchanged (450-dim RLDeploy
    contract), so this resumes directly from the Stride-V15 stage-1
    checkpoint.

    Changes vs Stride-V15:
      * Scene terrain plane → procedural generator
        (LITE3_HANDSTAND_TERRAINS_CFG: 40% flat, 35% random_rough <=3 cm,
         25% wave <=4 cm, curriculum off so every env sees varied ground).
      * ``foot_friction_slide`` range widened to (0.4, 1.25) (was the
        RLDeploy-pinned (0.8, 1.2)).  MuJoCo combines the two contacting
        geoms' friction, so randomizing the foot geom is equivalent to
        randomizing the effective ground-contact friction the policy feels
        — the terrain entity is not in scene.entities, so it cannot be
        targeted by the geom_friction DR event directly.

    Everything else (rewards, obs, DR, commands, PD) is inherited from
    Stride-V15 unchanged.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_env_cfg(play=play)

    # Softened-contact robot so a tumble / slight spawn-penetration on the
    # heightfield terminates the episode instead of NaN-ing the batch.
    cfg.scene.entities["robot"] = get_lite3_terrain_handstand_robot_cfg()

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=LITE3_HANDSTAND_TERRAINS_CFG,
    )
    # Let MuJoCo size the visual extent to the (much larger) terrain.
    cfg.scene.extent = None

    # Spawn near each patch origin so the front paws don't start buried in a
    # wave crest (a fixed-z spawn over varying surface height penetrates and
    # explodes the contact).  Tighten xy/z reset jitter; keep yaw full-range.
    if "reset_base" in cfg.events:
        cfg.events["reset_base"].params["pose_range"].update(
            {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.005, 0.005)}
        )

    # Heightfield contact-robustness sim params (mirrors go2_stairs, which
    # runs on generator terrain without NaN).  The base handstand uses
    # low solver iterations (10) tuned for a flat plane; on undulating
    # terrain that under-converges the stiff RLDeploy contact and explodes
    # to NaN.  Raise solver/CCD iterations, use the elliptic friction cone
    # with a high impedance ratio, and enlarge the contact/constraint pools.
    cfg.sim.mujoco.iterations = 100
    cfg.sim.mujoco.ls_iterations = 50
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.mujoco.impratio = 10
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 200
    cfg.sim.njmax = 4000

    if "foot_friction_slide" in cfg.events:
        cfg.events["foot_friction_slide"].params["ranges"] = (0.4, 1.25)

    return cfg


# ===========================================================================
# Stride high-frequency-contact fix (from the wf_efca09c1 multi-agent
# diagnosis).  Root cause H1: the rhythm-driving reward
# (handstand_feet_clearance, cycle_time 1.5 s) keys on an episode-time phase
# the actor CANNOT see — the RLDeploy actor obs has no clock term and 10
# frames (0.2 s) << 1.5 s cycle, so phase is unreconstructable.  A blind
# policy maximizes the phase-AVERAGED expectation → both front feet skim the
# ground and twitch (micro-shuffle).  Two experiments share one set of
# "remove the amplitude suppressors" reward fixes; they differ ONLY by whether
# the gait-phase clock is added to the observation, isolating H1.
# ===========================================================================

def _apply_stride_reward_fixes(cfg: ManagerBasedRlEnvCfg) -> None:
    """Reward-only stride fixes (no obs change) — safe to resume-finetune.

    * H2: feet_air_time is an UNCLAMPED barrier at 0.4 s (penalizes every
      sub-0.4 s touchdown; measured net-negative the whole V15 run).  Turn it
      into a positive ramp from an achievable 0.15 s swing, capped at 0.6 s so
      a one-leg perpetual hop can't farm it, and drop weight 3.0 → 1.0.
    * H4/H7/H8: dof_acc (-1e-3) and action_rate_l2 (-0.10) are monotone in
      joint speed and suppress the large-slow swing as much as the buzz.
      Relax one step toward baseline (do NOT add the energy term — it was the
      counterproductive finetune that did nothing).
    * H6: the tracking soft-gate folds front-foot contact into its support
      factor (support_floor 0.05) → lifting one front foot ~halves the
      velocity-tracking reward, a gradient toward keeping both feet planted.
      Raise support_floor to 0.6 so a swing only modestly dents tracking.
    """
    rewards = cfg.rewards

    # H2 — positive, capped air-time ramp from an achievable swing duration.
    rewards["feet_air_time"].weight = 1.0
    rewards["feet_air_time"].params.update(
        {"air_time_offset": 0.15, "air_time_cap": 0.6, "clamp_positive": True}
    )

    # H4/H7/H8 — relax amplitude-suppressing smoothness costs (one step).
    if "dof_acc" in rewards:
        rewards["dof_acc"].weight = -5.0e-4
    if "action_rate_l2" in rewards:
        rewards["action_rate_l2"].weight = -0.07
    # Ensure the counterproductive energy term is NOT present.
    rewards.pop("energy", None)

    # H6 — decouple velocity tracking from front-foot contact.
    for name in (
        "tracking_lin_vel",
        "tracking_ang_vel",
        "tracking_lin_vel_zero",
        "tracking_ang_vel_zero",
    ):
        if name in rewards and "support_floor" in rewards[name].params:
            rewards[name].params["support_floor"] = 0.6


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_rewardfix_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Experiment C — reward-only stride fix (no obs change), resume-finetune.

    Applies the "remove the amplitude suppressors" fixes (H2/H4/H6/H7/H8)
    WITHOUT touching the observation, so it resumes directly from the
    Stride-V15 stage-1 checkpoint (450-dim obs unchanged).  The diagnosis
    predicts this is INHERENTLY LIMITED: with no observable phase (H1
    unaddressed) the policy still cannot time which front foot to lift when,
    so it may grow amplitude but cannot fully escape the high-frequency
    shuffle.  This is the deliberate A/B control vs the phase-clock variant.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_env_cfg(play=play)
    _apply_stride_reward_fixes(cfg)
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Experiment B — add the gait-phase clock to the obs (H1 primary fix).

    Identical reward set to the reward-fix variant (C), PLUS the existing
    ``gait_clock`` sin/cos observation (cycle_time 1.5 s, matched to
    handstand_feet_clearance) appended to BOTH actor and critic groups —
    exactly as go2_trot wires it and as the IsaacGym source did.  This closes
    the loop: the rhythm reward now grades against a phase the actor can read,
    so the policy can deterministically schedule FL-up/FR-down vs
    FR-up/FL-down → one clean alternation per 1.5 s cycle instead of the
    phase-averaged buzz.

    The actor input grows 450 → 470 ((45+2)×10), so this CANNOT resume the
    450-dim checkpoint — it trains from scratch.  (Deploying this policy also
    requires feeding the same sin/cos(2π·t/1.5) clock at runtime.)
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_env_cfg(play=play)
    _apply_stride_reward_fixes(cfg)

    clock_term = ObservationTermCfg(
        func=go2_mdp.gait_clock,
        params={"cycle_time": 1.5},   # MUST match handstand_feet_clearance cycle
    )
    cfg.observations["actor"].terms["gait_clock"] = clock_term
    cfg.observations["critic"].terms["gait_clock"] = clock_term

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_robust_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride-V15 ground-robustness finetune (flat, NaN-safe).

    Replaces the abandoned heightfield-terrain finetune (undulating
    heightfield + the tumble-prone flat-trained handstand persistently
    NaN-ed in mujoco_warp: a non-foot link slamming a wave crest explodes
    the contact even after softening).  Instead of literal terrain geometry,
    this builds robustness to "varied / uneven ground" on a FLAT plane via:

      * Wide ground-friction randomization — ``foot_friction_slide`` range
        (0.4, 1.4) (was the RLDeploy-pinned (0.8, 1.2)).  MuJoCo combines the
        two contacting geoms' friction, so randomizing the foot geom varies
        the effective ground friction the policy feels.
      * Stronger, more frequent random pushes — emulate the impulse
        disturbances of stepping on an uneven / shifting surface: every
        2-4 s (was 4-8 s) with larger linear and angular velocity kicks.
      * Keeps V15's full sim2real DR (pd_gains, motor_strength, joint
        friction/damping/armature, link_inertia, base mass/com, encoder bias).

    Obs unchanged (450-dim), so this resumes the Stride-V15 stage-1 ckpt.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_env_cfg(play=play)

    if "foot_friction_slide" in cfg.events:
        cfg.events["foot_friction_slide"].params["ranges"] = (0.4, 1.4)

    if "push_robot" in cfg.events:
        cfg.events["push_robot"].interval_range_s = (2.0, 4.0)
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.3, 0.3),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.7, 0.7),
        }

    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride-V15 + AMP: adds an 'amp' observation group (the 43-dim
    discriminator AMP-obs) that the AMP runner reads each step.

    actor/critic obs are UNCHANGED (450-dim) — obs_groups routes only
    'actor'/'critic', so the networks ignore 'amp' and the policy contract
    stays warm-start-compatible with the Stride-V15 checkpoint.  The 'amp'
    group is single-frame, un-corrupted (the discriminator compares clean
    motion).  See go2_mjlab/amp/observations.py for the exact 43-dim layout,
    which matches the ahmp lite3_handstand.npz amp_frames slice.
    """
    # Lazy import so non-AMP tasks don't pull in torch/rsl_rl via the amp pkg.
    from go2_mjlab.amp.observations import amp_observations

    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_env_cfg(play=play)

    cfg.observations["amp"] = ObservationGroupCfg(
        terms={
            "amp_obs": ObservationTermCfg(
                func=amp_observations,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "foot_site_names": ("FL", "FR", "HL", "HR"),
                },
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
    )
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_amp_rewardfix_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stride-V15 + AMP + RewardFix: the AMP style reward AND the relaxed
    amplitude-suppressor rewards, pushing the SAME direction.

    The pure-AMP variant kept the base V15 reward set, whose full-strength
    amplitude suppressors (``feet_air_time`` 0.4 s penalty barrier @ 3.0,
    ``dof_acc`` -1e-3, ``action_rate_l2`` -0.1, tracking ``support_floor`` 0.05)
    directly OPPOSE the AMP style reward's pull toward the reference's ~0.3 m
    front-paw step — the two forces cancel and the step stays a micro-shuffle
    (observed: ``feet_clearance`` plateaus ~0.075, ``feet_air_time`` net
    negative).  This variant applies ``_apply_stride_reward_fixes`` (the exact
    same relaxations as the RewardFix / PhaseClock experiments) so the
    suppressors no longer fight the style reward — both the task reward and the
    discriminator now reward a larger, cleaner swing.

    actor/critic obs stay 450-dim (RewardFix changes NO observation), so this
    warm-starts the Stride-V15 stage-1 checkpoint exactly like the pure-AMP
    variant.  The single-frame, un-corrupted 'amp' observation group (43-dim)
    is added for the discriminator, identical to the pure-AMP env cfg.
    """
    # Lazy import so non-AMP tasks don't pull in torch/rsl_rl via the amp pkg.
    from go2_mjlab.amp.observations import amp_observations

    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_rewardfix_env_cfg(play=play)

    cfg.observations["amp"] = ObservationGroupCfg(
        terms={
            "amp_obs": ObservationTermCfg(
                func=amp_observations,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "foot_site_names": ("FL", "FR", "HL", "HR"),
                },
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
    )
    return cfg


def _apply_minimal_stand_reward(cfg: ManagerBasedRlEnvCfg) -> None:
    """Strip the V15 reward set to the minimal 'be inverted + don't fall + stay
    put' core, so the front-paw gait can EMERGE from AMP instead of from the
    hand-crafted gait-contact rewards (which produced the 5.5 Hz micro-shuffle).

    KEEP: alive, handstand_orientation (be inverted), handstand_feet_on_air
    (rear legs up), base_height (don't collapse), tracking_lin_vel/ang_vel (at
    zero command = stand still), base/thigh/calf collision (don't fall),
    lin_vel_z (anti vertical bounce), plus a TINY action_rate/dof_acc (sim2real
    smoothness only — much smaller than the V15 amplitude-suppressing values).

    DROP the whole gait-micro-management cluster (feet_air_time, feet_clearance,
    single/zero_stance_contact, stance_air_penalty, zero_joint_vel), the pose
    shaping (default_pos / default_pos_reward / default_hip_pos), the redundant
    tracking_*_zero, and ang_vel_xy (which would fight the reference's torso
    pitch).
    """
    rewards = cfg.rewards
    KEEP = {
        "alive", "handstand_orientation", "handstand_feet_on_air", "base_height",
        "tracking_lin_vel", "tracking_ang_vel", "base_contact", "thigh_collision",
        "calf_collision", "action_rate_l2", "dof_acc", "lin_vel_z",
    }
    for k in list(rewards.keys()):
        if k not in KEEP:
            rewards.pop(k)
    # Tiny smoothness only (these were amplitude suppressors at full strength).
    if "action_rate_l2" in rewards:
        rewards["action_rate_l2"].weight = -0.01
    if "dof_acc" in rewards:
        rewards["dof_acc"].weight = -5.0e-4
    # Fully decouple velocity tracking from front-foot contact (H6 gating).
    for nm in ("tracking_lin_vel", "tracking_ang_vel"):
        if nm in rewards and "support_floor" in rewards[nm].params:
            rewards[nm].params["support_floor"] = 1.0


def unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Minimal-reward + AMP, zero-command STANDING handstand, trained FROM SCRATCH.

    A clean diagnostic + delivery attempt after every reward/obs finetune left
    the front paws at a ~1 cm, ~5.5 Hz micro-shuffle.  Instead of micro-managing
    the front-paw contact pattern with hand-crafted rewards (the suspected cause
    of the shuffle attractor), this:

      * strips the reward to the minimal balance/safety core
        (``_apply_minimal_stand_reward``), so the ONLY front-paw gait signal is
        the AMP style reward imitating the ahmp ``lite3_handstand`` clip
        (verified to show large alternating front-paw steps);
      * fixes the command to ZERO (every env stands) — the reference is an
        in-place handstand, so standing removes the walk-vs-in-place conflict;
      * adds the 43-dim 'amp' observation group for the discriminator.

    actor/critic obs stay 450-dim; trained from scratch (its own experiment
    name, no warm-start) for the full iteration budget.  Either the front paws
    settle into a clean AMP-shaped alternation (reward was the culprit) or they
    still shuffle (the shuffle is physical) — both outcomes are decisive.
    """
    # Lazy import so non-AMP tasks don't pull in torch/rsl_rl via the amp pkg.
    from go2_mjlab.amp.observations import amp_observations

    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_env_cfg(play=play)
    _apply_minimal_stand_reward(cfg)

    # Zero command: every env stands; front-paw alternation must come from AMP.
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 1.0
    twist.ranges.lin_vel_x = (0.0, 0.0)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)

    cfg.observations["amp"] = ObservationGroupCfg(
        terms={
            "amp_obs": ObservationTermCfg(
                func=amp_observations,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "foot_site_names": ("FL", "FR", "HL", "HR"),
                },
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
    )
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_gaitlock_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Path B: PhaseClock + HARD phase-locked gait control (cadence + amplitude).

    PhaseClock proved the obs gait clock yields clean L/R alternation, but the
    cadence stayed ~7 Hz (the soft sin clearance reward never enforced the
    clock's slow cycle) and the step amplitude stayed small.  Now that the phase
    is OBSERVABLE, lock the gait to a slower clock with two enforcement terms:

      * clock ``cycle_time`` 1.5 -> 1.0 s everywhere (obs ``gait_clock`` on both
        actor+critic, and the ``feet_clearance`` phase) so the schedule below is
        followable from the observation;
      * (1) ``handstand_contact_schedule`` (+2.0): each front foot must be in
        ground contact during its clock STANCE half and airborne during its
        SWING half -> directly controls FREQUENCY (a faster-than-clock shuffle
        is penalised);
      * (2) ``handstand_swing_foot_height_linear`` (cap 0.12 m, +1.5): reward
        lifting the swing foot high -> directly controls AMPLITUDE;
      * drop the gameable soft-sin ``feet_clearance`` (weight 0).

    470-dim obs (clock in obs), trained from scratch like PhaseClock.  ``cycle_time``
    and the two weights are the knobs to tune if it falls (too slow) or stays
    fast (schedule too weak).
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_phaseclock_env_cfg(play=play)

    CYCLE = 1.0
    for grp in ("actor", "critic"):
        if "gait_clock" in cfg.observations[grp].terms:
            cfg.observations[grp].terms["gait_clock"].params["cycle_time"] = CYCLE
    # Drop the gameable soft-sin clearance; amplitude now comes from the linear term.
    if "feet_clearance" in cfg.rewards:
        cfg.rewards["feet_clearance"].weight = 0.0

    # (1) cadence lock — contact must follow the clock phase.
    cfg.rewards["contact_schedule"] = RewardTermCfg(
        func=go2_mdp.handstand_contact_schedule,
        weight=2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "foot_indices": (0, 1),
            "cycle_time": CYCLE,
            "command_name": "twist",
            "moving_threshold": 0.1,
        },
    )
    # (2) amplitude — reward lifting the swing foot high (linear to a 12 cm cap).
    cfg.rewards["swing_foot_height"] = RewardTermCfg(
        func=go2_mdp.handstand_swing_foot_height_linear,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "asset_cfg": SceneEntityCfg("robot", body_names=("TORSO",)),
            "foot_indices": (0, 1),
            "foot_site_names": ("FL", "FR", "HL", "HR"),
            "cap_height": 0.12,
            "command_name": "twist",
            "moving_threshold": 0.1,
        },
    )
    return cfg


def unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_v2_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Minimal-v2: minimal reward + AMP standing, but RE-ADD the front-paw SUPPORT
    terms that v1 wrongly dropped.

    v1 (from scratch) got stuck KNEELING on its calves (probe: feet contact
    [0,0,0,0], calf_ground_touch 1.0) — it farmed handstand_feet_on_air (+5) +
    tracking-still (+4.5) + alive with the front paws lifted.  A calf penalty
    can't fix this: the REAL V15 handstand ALSO has calf contact 1.0 (front shank
    rests near the planted paw), so calf contact does NOT distinguish the two.
    The true discriminator is FRONT-FOOT contact (real handstand 0.89 vs kneel
    0.0).  So re-add the two terms that force the front paws to be the support:
      * ``zero_stance_contact`` (+3): reward front feet in contact at zero command;
      * ``stance_air_penalty`` (-5): penalize both front feet airborne.
    These are support-structure terms (not gait-TIMING), so they don't reintroduce
    the high-freq shuffle, but they make kneeling (front feet up) ~-8 worse than
    the real front-paw stand.  Everything else stays minimal; obs 450-dim; AMP
    coef 3 / lerp 0.5; from scratch.
    """
    cfg = unitree_lite3_handstand_rldeploy_robotlab_minimal_amp_stand_env_cfg(play=play)
    base = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_no_hop_big_step_stride_v15_env_cfg(play=play)
    for term in ("zero_stance_contact", "stance_air_penalty"):
        if term in base.rewards:
            cfg.rewards[term] = base.rewards[term]
    return cfg
