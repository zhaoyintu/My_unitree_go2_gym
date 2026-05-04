"""RL-deploy-aligned Lite3 handstand Mjlab environment config.

This task keeps the current Lite3 handstand reward shape, but pins the
policy contract and nominal MuJoCo dynamics to the C++ `rl_deploy_handstand`
runner:

  ang_vel, projected_gravity, cmd, joint_pos_rel, joint_vel, last_action

The actor therefore sees 45 values per frame and uses Mjlab's term-major
10-frame history flattening for a 450-dim policy input.  The critic keeps
base linear velocity and contact state as privileged training-only inputs.
"""

from go2_mjlab import mdp as go2_mdp
from go2_mjlab.config.lite3_handstand.env_cfgs import unitree_lite3_handstand_env_cfg
from go2_mjlab.robots.lite3_constants import get_lite3_rldeploy_handstand_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
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
