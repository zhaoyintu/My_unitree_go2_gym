from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INIT = REPO_ROOT / "go2_mjlab/__init__.py"
ENV_CFG = REPO_ROOT / "go2_mjlab/config/lite3_handstand_rldeploy/env_cfgs.py"
RL_CFG = REPO_ROOT / "go2_mjlab/config/lite3_handstand_rldeploy/rl_cfg.py"
LITE3_CONSTANTS = REPO_ROOT / "go2_mjlab/robots/lite3_constants.py"
REWARDS = REPO_ROOT / "go2_mjlab/mdp/rewards.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _top_level_source(path: Path, name: str) -> str:
    source = _source(path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} not found in {path}")


class Lite3HandstandRLDeployStaticTest(unittest.TestCase):
    def test_registers_separate_rldeploy_task(self) -> None:
        source = _source(INIT)

        self.assertIn("unitree_lite3_handstand_rldeploy_env_cfg", source)
        self.assertIn("unitree_lite3_handstand_rldeploy_ppo_runner_cfg", source)
        self.assertIn('"Mjlab-Lite3-Handstand-RLDeploy"', source)
        self.assertIn("env_cfg=unitree_lite3_handstand_rldeploy_env_cfg(play=False)", source)
        self.assertIn("play_env_cfg=unitree_lite3_handstand_rldeploy_env_cfg(play=True)", source)

    def test_actor_observation_matches_rldeploy_contract(self) -> None:
        source = _source(ENV_CFG)
        actor_terms_source = _top_level_source(ENV_CFG, "_rldeploy_actor_terms")

        self.assertIn("RLDEPLOY_SINGLE_OBS_DIM = 45", source)
        self.assertIn("RLDEPLOY_HISTORY_LENGTH = 10", source)
        self.assertIn("RLDEPLOY_ACTOR_OBS_DIM = RLDEPLOY_SINGLE_OBS_DIM * RLDEPLOY_HISTORY_LENGTH", source)
        self.assertIn('"base_ang_vel"', actor_terms_source)
        self.assertIn('"projected_gravity"', actor_terms_source)
        self.assertIn('"commands"', actor_terms_source)
        self.assertIn('"joint_pos"', actor_terms_source)
        self.assertIn('"joint_vel"', actor_terms_source)
        self.assertIn('"actions"', actor_terms_source)
        self.assertNotIn('"base_lin_vel"', actor_terms_source)
        self.assertNotIn('"robot/imu_lin_vel"', actor_terms_source)

    def test_actor_history_uses_term_major_10_frame_stack(self) -> None:
        function_source = _top_level_source(ENV_CFG, "unitree_lite3_handstand_rldeploy_env_cfg")

        self.assertIn('cfg.observations = {', function_source)
        self.assertIn('"actor": ObservationGroupCfg(', function_source)
        self.assertIn("terms=actor_terms", function_source)
        self.assertIn("history_length=RLDEPLOY_HISTORY_LENGTH", function_source)
        self.assertIn("flatten_history_dim=True", function_source)

    def test_dynamics_match_rldeploy_contact_and_control_contract(self) -> None:
        constants_source = _source(LITE3_CONSTANTS)
        actuator_block = constants_source.split(
            "LITE3_RLDEPLOY_HANDSTAND_HIPX_ACTUATOR_CFG", 1
        )[1].split("##\n# Keyframes.", 1)[0]
        collision_block = constants_source.split("RLDEPLOY_COLLISION = CollisionCfg(", 1)[1].split(
            "##\n# Final config.", 1
        )[0]
        robot_cfg_source = _top_level_source(LITE3_CONSTANTS, "get_lite3_rldeploy_handstand_robot_cfg")

        self.assertIn("stiffness=40.0", actuator_block)
        self.assertIn("damping=1.0", actuator_block)
        self.assertIn("effort_limit=HIPX_ACTUATOR.effort_limit", actuator_block)
        self.assertNotIn("armature=", actuator_block)

        self.assertIn("contype=0", collision_block)
        self.assertIn("conaffinity=1", collision_block)
        self.assertIn("condim=3", collision_block)
        self.assertIn("priority=0", collision_block)
        self.assertIn("friction=(1.0, 0.01, 0.01)", collision_block)
        self.assertIn("solref=(0.005, 1)", collision_block)
        self.assertIn("collisions=(RLDEPLOY_COLLISION,)", robot_cfg_source)
        self.assertIn("articulation=LITE3_RLDEPLOY_HANDSTAND_ARTICULATION", robot_cfg_source)

    def test_env_uses_rldeploy_nominal_dynamics(self) -> None:
        function_source = _top_level_source(ENV_CFG, "unitree_lite3_handstand_rldeploy_env_cfg")

        self.assertIn('cfg.scene.entities["robot"] = get_lite3_rldeploy_handstand_robot_cfg()', function_source)
        self.assertIn('cfg.events["foot_friction_slide"].params["ranges"] = (0.8, 1.2)', function_source)
        self.assertIn('cfg.events.pop("foot_friction_spin", None)', function_source)
        self.assertIn('cfg.events.pop("foot_friction_roll", None)', function_source)
        self.assertIn("cfg.sim.mujoco.timestep = 0.001", function_source)
        self.assertIn("cfg.decimation = 20", function_source)

    def test_critic_keeps_base_linear_velocity_as_privileged_state(self) -> None:
        function_source = _top_level_source(ENV_CFG, "unitree_lite3_handstand_rldeploy_env_cfg")
        critic_source = function_source.split("critic_terms = {", 1)[1].split("cfg.observations = {", 1)[0]

        self.assertIn('"base_lin_vel"', critic_source)
        self.assertIn('"robot/imu_lin_vel"', critic_source)
        self.assertIn('"foot_contact"', critic_source)
        self.assertIn("go2_mdp.foot_contact", critic_source)

    def test_reward_recipe_prioritizes_handstand_before_tracking(self) -> None:
        function_source = _top_level_source(ENV_CFG, "unitree_lite3_handstand_rldeploy_env_cfg")

        self.assertIn('cfg.commands["twist"].rel_standing_envs = 0.25', function_source)
        self.assertIn('rewards["handstand_orientation"].func = go2_mdp.handstand_orientation_exp', function_source)
        self.assertIn('rewards["handstand_orientation"].weight = 2.0', function_source)
        self.assertIn('rewards["handstand_feet_on_air"].weight = 1.0', function_source)
        self.assertIn('rewards["handstand_feet_height_exp"].func = go2_mdp.handstand_rear_feet_height_static', function_source)
        self.assertIn('rewards["handstand_feet_height_exp"].weight = 8.0', function_source)
        self.assertIn('rewards["base_height"].func = go2_mdp.handstand_base_height_soft', function_source)
        self.assertIn('rewards["base_height"].weight = 0.8', function_source)
        self.assertIn('rewards["tracking_lin_vel"].func = go2_mdp.handstand_tracking_lin_vel_soft_gate', function_source)
        self.assertIn('rewards["tracking_ang_vel"].func = go2_mdp.handstand_tracking_ang_vel_soft_gate', function_source)
        self.assertIn('rewards["contact"].func = go2_mdp.handstand_stance_contact_mean', function_source)
        self.assertIn('rewards["contact"].weight = 0.8', function_source)
        self.assertIn('rewards["feet_air_time"].weight = 0.0', function_source)
        self.assertIn('rewards["feet_clearance"].weight = 0.0', function_source)
        self.assertIn('rewards["default_pos"].weight = -0.6', function_source)
        self.assertIn('rewards["default_pos_reward"].weight = 2.0', function_source)

    def test_soft_gated_tracking_does_not_use_always_open_quality_gate(self) -> None:
        source = _source(REWARDS)

        self.assertIn("def _handstand_soft_quality", source)
        for name in (
            "handstand_tracking_lin_vel_soft_gate",
            "handstand_tracking_ang_vel_soft_gate",
        ):
            function_source = _top_level_source(REWARDS, name)
            self.assertIn("_handstand_soft_quality(", function_source)
            self.assertNotIn("_handstand_quality(", function_source)
            self.assertNotIn("(quality > 0.70).float()", function_source)

    def test_runner_uses_separate_experiment_name(self) -> None:
        source = _source(RL_CFG)
        function_source = _top_level_source(
            RL_CFG,
            "unitree_lite3_handstand_rldeploy_ppo_runner_cfg",
        )

        self.assertIn("unitree_lite3_handstand_ppo_runner_cfg()", function_source)
        self.assertIn('cfg.experiment_name = "lite3_handstand_rldeploy"', function_source)
        self.assertIn("RslRlOnPolicyRunnerCfg", source)


if __name__ == "__main__":
    unittest.main()
