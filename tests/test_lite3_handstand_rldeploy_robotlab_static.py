from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INIT = REPO_ROOT / "go2_mjlab/__init__.py"
ENV_CFG = REPO_ROOT / "go2_mjlab/config/lite3_handstand_rldeploy/env_cfgs.py"
RL_CFG = REPO_ROOT / "go2_mjlab/config/lite3_handstand_rldeploy/rl_cfg.py"
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


class Lite3HandstandRLDeployRobotLabStaticTest(unittest.TestCase):
    def test_registers_separate_robotlab_task(self) -> None:
        source = _source(INIT)

        self.assertIn("unitree_lite3_handstand_rldeploy_robotlab_env_cfg", source)
        self.assertIn("unitree_lite3_handstand_rldeploy_robotlab_ppo_runner_cfg", source)
        self.assertIn('"Mjlab-Lite3-Handstand-RLDeploy-RobotLab"', source)
        self.assertIn(
            "env_cfg=unitree_lite3_handstand_rldeploy_robotlab_env_cfg(play=False)",
            source,
        )
        self.assertIn(
            "play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_env_cfg(play=True)",
            source,
        )

    def test_registers_separate_low_default_pose_task(self) -> None:
        source = _source(INIT)

        self.assertIn("unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_env_cfg", source)
        self.assertIn(
            "unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_ppo_runner_cfg",
            source,
        )
        self.assertIn('"Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose"', source)
        self.assertIn(
            "env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_env_cfg(play=False)",
            source,
        )
        self.assertIn(
            "play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_env_cfg(play=True)",
            source,
        )

    def test_registers_separate_no_default_pose_task(self) -> None:
        source = _source(INIT)

        self.assertIn("unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_env_cfg", source)
        self.assertIn(
            "unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_ppo_runner_cfg",
            source,
        )
        self.assertIn('"Mjlab-Lite3-Handstand-RLDeploy-RobotLab-NoDefaultPose"', source)
        self.assertIn(
            "env_cfg=unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_env_cfg(play=False)",
            source,
        )
        self.assertIn(
            "play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_env_cfg(play=True)",
            source,
        )

    def test_registers_separate_low_default_pose_zero_stance_task(self) -> None:
        source = _source(INIT)

        self.assertIn(
            "unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_env_cfg",
            source,
        )
        self.assertIn(
            "unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_ppo_runner_cfg",
            source,
        )
        self.assertIn('"Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-ZeroStance"', source)
        self.assertIn(
            "env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_env_cfg(play=False)",
            source,
        )
        self.assertIn(
            "play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_env_cfg(play=True)",
            source,
        )

    def test_registers_separate_low_default_pose_quiet_zero_stance_task(self) -> None:
        source = _source(INIT)

        self.assertIn(
            "unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_env_cfg",
            source,
        )
        self.assertIn(
            "unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_ppo_runner_cfg",
            source,
        )
        self.assertIn('"Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose-QuietZeroStance"', source)
        self.assertIn(
            "env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_env_cfg(play=False)",
            source,
        )
        self.assertIn(
            "play_env_cfg=unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_env_cfg(play=True)",
            source,
        )

    def test_runner_uses_separate_experiment_name(self) -> None:
        function_source = _top_level_source(
            RL_CFG,
            "unitree_lite3_handstand_rldeploy_robotlab_ppo_runner_cfg",
        )

        self.assertIn("unitree_lite3_handstand_rldeploy_ppo_runner_cfg()", function_source)
        self.assertIn(
            'cfg.experiment_name = "lite3_handstand_rldeploy_robotlab"',
            function_source,
        )

    def test_low_default_pose_runner_uses_separate_experiment_name(self) -> None:
        function_source = _top_level_source(
            RL_CFG,
            "unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_ppo_runner_cfg",
        )

        self.assertIn(
            "unitree_lite3_handstand_rldeploy_robotlab_ppo_runner_cfg()",
            function_source,
        )
        self.assertIn(
            'cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose"',
            function_source,
        )

    def test_no_default_pose_runner_uses_separate_experiment_name(self) -> None:
        function_source = _top_level_source(
            RL_CFG,
            "unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_ppo_runner_cfg",
        )

        self.assertIn(
            "unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_ppo_runner_cfg()",
            function_source,
        )
        self.assertIn(
            'cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_no_default_pose"',
            function_source,
        )

    def test_low_default_pose_zero_stance_runner_uses_separate_experiment_name(self) -> None:
        function_source = _top_level_source(
            RL_CFG,
            "unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_ppo_runner_cfg",
        )

        self.assertIn(
            "unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_ppo_runner_cfg()",
            function_source,
        )
        self.assertIn(
            'cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance"',
            function_source,
        )

    def test_low_default_pose_quiet_zero_stance_runner_uses_separate_experiment_name(self) -> None:
        function_source = _top_level_source(
            RL_CFG,
            "unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_ppo_runner_cfg",
        )

        self.assertIn(
            "unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_ppo_runner_cfg()",
            function_source,
        )
        self.assertIn(
            'cfg.experiment_name = "lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance"',
            function_source,
        )

    def test_robotlab_task_inherits_rldeploy_actor_contract(self) -> None:
        function_source = _top_level_source(
            ENV_CFG,
            "unitree_lite3_handstand_rldeploy_robotlab_env_cfg",
        )

        self.assertIn("cfg = unitree_lite3_handstand_rldeploy_env_cfg(play=play)", function_source)
        self.assertNotIn('"base_lin_vel"', function_source)
        self.assertNotIn('"robot/imu_lin_vel"', function_source)
        self.assertNotIn("ObservationGroupCfg", function_source)

    def test_robotlab_task_uses_large_command_range_with_curriculum(self) -> None:
        source = _source(ENV_CFG)
        function_source = _top_level_source(
            ENV_CFG,
            "unitree_lite3_handstand_rldeploy_robotlab_env_cfg",
        )

        self.assertIn("from mjlab.tasks.velocity import mdp as velocity_mdp", source)
        self.assertIn("CurriculumTermCfg", source)
        self.assertIn('twist_cmd = cfg.commands["twist"]', function_source)
        self.assertIn("assert isinstance(twist_cmd, UniformVelocityCommandCfg)", function_source)
        self.assertIn("twist_cmd.ranges.lin_vel_x = (-0.4, 0.4)", function_source)
        self.assertIn("twist_cmd.ranges.lin_vel_y = (0.0, 0.0)", function_source)
        self.assertIn("twist_cmd.ranges.ang_vel_z = (-0.4, 0.4)", function_source)
        self.assertIn('cfg.curriculum["command_vel"] = CurriculumTermCfg(', function_source)
        self.assertIn("func=velocity_mdp.commands_vel", function_source)
        self.assertIn('"step": 2000 * 24', function_source)
        self.assertIn('"step": 5000 * 24', function_source)
        self.assertIn('"step": 8000 * 24', function_source)
        self.assertIn('"lin_vel_x": (-1.0, 1.0)', function_source)
        self.assertIn('"lin_vel_y": (-1.0, 1.0)', function_source)
        self.assertIn('"ang_vel_z": (-1.0, 1.0)', function_source)
        self.assertIn("if play:", function_source)
        self.assertIn("cfg.curriculum = {}", function_source)

    def test_robotlab_task_reward_recipe_follows_robotlab_handstand(self) -> None:
        function_source = _top_level_source(
            ENV_CFG,
            "unitree_lite3_handstand_rldeploy_robotlab_env_cfg",
        )

        self.assertIn('rewards["handstand_orientation"].func = go2_mdp.handstand_orientation', function_source)
        self.assertIn('rewards["handstand_orientation"].weight = -1.0', function_source)
        self.assertIn(
            'rewards["handstand_feet_height_exp"].func = go2_mdp.handstand_feet_height_l2_exp',
            function_source,
        )
        self.assertIn('rewards["handstand_feet_height_exp"].weight = 10.0', function_source)
        self.assertIn('"target_height": 0.56', function_source)
        self.assertIn('"std": math.sqrt(0.25)', function_source)
        self.assertIn('"stance_sensor_name": "feet_ground_contact"', function_source)
        self.assertIn('"stance_foot_indices": (0, 1)', function_source)
        self.assertIn("ROBOTLAB_BODY_CLEARANCE_SENSOR_NAMES", _source(ENV_CFG))
        self.assertIn(
            '"body_clearance_sensor_names": ROBOTLAB_BODY_CLEARANCE_SENSOR_NAMES',
            function_source,
        )
        self.assertIn('rewards["handstand_feet_on_air"].weight = 5.0', function_source)
        self.assertIn('rewards["contact"].func = go2_mdp.handstand_stance_contact_mean', function_source)
        self.assertIn('rewards["contact"].weight = 2.0', function_source)
        self.assertIn('rewards["tracking_lin_vel"].weight = 3.0', function_source)
        self.assertIn('rewards["tracking_ang_vel"].weight = 1.5', function_source)
        self.assertIn('rewards["feet_air_time"].weight = 0.0', function_source)
        self.assertIn('rewards["feet_clearance"].weight = 0.0', function_source)
        self.assertNotIn('rewards["default_pos"].weight = -0.15', function_source)
        self.assertNotIn('rewards["default_pos_reward"].weight = 0.4', function_source)
        self.assertNotIn('rewards["default_hip_pos"].weight = -0.1', function_source)

    def test_low_default_pose_task_only_overrides_default_pose_rewards(self) -> None:
        function_source = _top_level_source(
            ENV_CFG,
            "unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_env_cfg",
        )

        self.assertIn(
            "cfg = unitree_lite3_handstand_rldeploy_robotlab_env_cfg(play=play)",
            function_source,
        )
        self.assertIn('rewards = cfg.rewards', function_source)
        self.assertIn('rewards["default_pos"].weight = -0.15', function_source)
        self.assertIn('rewards["default_pos_reward"].weight = 0.4', function_source)
        self.assertIn('rewards["default_hip_pos"].weight = -0.1', function_source)
        self.assertNotIn("_add_rldeploy_sim2real_dr_events(cfg)", function_source)
        self.assertNotIn('cfg.curriculum["command_vel"]', function_source)
        self.assertIn("return cfg", function_source)

    def test_no_default_pose_task_only_zeroes_default_pose_rewards(self) -> None:
        function_source = _top_level_source(
            ENV_CFG,
            "unitree_lite3_handstand_rldeploy_robotlab_no_default_pose_env_cfg",
        )

        self.assertIn(
            "cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_env_cfg(play=play)",
            function_source,
        )
        self.assertIn('rewards = cfg.rewards', function_source)
        self.assertIn('rewards["default_pos"].weight = 0.0', function_source)
        self.assertIn('rewards["default_pos_reward"].weight = 0.0', function_source)
        self.assertIn('rewards["default_hip_pos"].weight = 0.0', function_source)
        self.assertNotIn("_add_rldeploy_sim2real_dr_events(cfg)", function_source)
        self.assertNotIn('cfg.curriculum["command_vel"]', function_source)
        self.assertIn("return cfg", function_source)

    def test_low_default_pose_zero_stance_task_adds_zero_command_stability_rewards(self) -> None:
        source = _source(ENV_CFG)
        function_source = _top_level_source(
            ENV_CFG,
            "unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_env_cfg",
        )

        self.assertIn("RewardTermCfg", source)
        self.assertIn(
            "cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_env_cfg(play=play)",
            function_source,
        )
        self.assertIn('rewards = cfg.rewards', function_source)
        self.assertIn('rewards["zero_stance_contact"] = RewardTermCfg(', function_source)
        self.assertIn("func=go2_mdp.handstand_stance_contact_zero", function_source)
        self.assertIn("weight=1.0", function_source)
        self.assertIn('"sensor_name": "feet_ground_contact"', function_source)
        self.assertIn('"foot_indices": (0, 1)', function_source)
        self.assertIn('"command_name": "twist"', function_source)
        self.assertIn('"moving_threshold": 0.1', function_source)
        self.assertIn('rewards["zero_joint_vel"] = RewardTermCfg(', function_source)
        self.assertIn("func=go2_mdp.handstand_joint_vel_zero", function_source)
        self.assertIn("weight=-0.02", function_source)
        self.assertIn('SceneEntityCfg("robot", joint_names=(".*",))', function_source)
        self.assertIn('rewards["lin_vel_z"].weight = 0.4', function_source)
        self.assertIn('rewards["action_rate_l2"].weight = -0.08', function_source)
        self.assertNotIn("_add_rldeploy_sim2real_dr_events(cfg)", function_source)
        self.assertNotIn('cfg.curriculum["command_vel"]', function_source)
        self.assertIn("return cfg", function_source)

    def test_low_default_pose_quiet_zero_stance_task_strengthens_zero_command_stability(self) -> None:
        function_source = _top_level_source(
            ENV_CFG,
            "unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_quiet_zero_stance_env_cfg",
        )

        self.assertIn(
            "cfg = unitree_lite3_handstand_rldeploy_robotlab_low_default_pose_zero_stance_env_cfg(play=play)",
            function_source,
        )
        self.assertIn('twist_cmd = cfg.commands["twist"]', function_source)
        self.assertIn("assert isinstance(twist_cmd, UniformVelocityCommandCfg)", function_source)
        self.assertIn("twist_cmd.rel_standing_envs = 0.40", function_source)
        self.assertIn('rewards = cfg.rewards', function_source)
        self.assertIn('rewards["zero_stance_contact"].weight = 3.0', function_source)
        self.assertIn('rewards["zero_joint_vel"].weight = -0.08', function_source)
        self.assertIn('rewards["dof_acc"].weight = -1.0e-3', function_source)
        self.assertIn('rewards["action_rate_l2"].weight = -0.12', function_source)
        self.assertIn('rewards["lin_vel_z"].weight = 0.6', function_source)
        self.assertNotIn("_add_rldeploy_sim2real_dr_events(cfg)", function_source)
        self.assertNotIn('cfg.curriculum["command_vel"]', function_source)
        self.assertIn("return cfg", function_source)

    def test_robotlab_task_terminates_on_trunk_and_thigh_but_not_shank(self) -> None:
        function_source = _top_level_source(
            ENV_CFG,
            "unitree_lite3_handstand_rldeploy_robotlab_env_cfg",
        )

        self.assertIn('cfg.terminations["base_contact"] = TerminationTermCfg(', function_source)
        self.assertIn('"sensor_name": "trunk_ground_touch"', function_source)
        self.assertIn('cfg.terminations["thigh_contact"] = TerminationTermCfg(', function_source)
        self.assertIn('"sensor_name": "thigh_ground_touch"', function_source)
        self.assertIn(
            "# Lite3 shank collision can touch the ground with the nominal foot contact.",
            function_source,
        )
        self.assertNotIn('cfg.terminations["calf_contact"] = TerminationTermCfg(', function_source)
        self.assertNotIn('"sensor_name": "calf_ground_touch"', function_source)

    def test_robotlab_task_keeps_mass_com_and_adds_sim_to_real_dr(self) -> None:
        function_source = _top_level_source(
            ENV_CFG,
            "unitree_lite3_handstand_rldeploy_robotlab_env_cfg",
        )

        self.assertIn("_add_rldeploy_sim2real_dr_events(cfg)", function_source)
        self.assertIn('"base_mass" in cfg.events', function_source)
        self.assertIn('"base_com" in cfg.events', function_source)
        self.assertNotIn('cfg.events.pop("base_mass"', function_source)
        self.assertNotIn('cfg.events.pop("base_com"', function_source)
        self.assertIn('cfg.events["encoder_bias"].params["bias_range"] = (-0.02, 0.02)', function_source)

        helper_source = _top_level_source(ENV_CFG, "_add_rldeploy_sim2real_dr_events")
        for event_name in (
            '"pd_gains"',
            '"motor_strength"',
            '"joint_friction"',
            '"joint_damping"',
            '"joint_armature"',
            '"link_inertia"',
        ):
            self.assertIn(event_name, helper_source)

    def test_robotlab_height_reward_function_exists(self) -> None:
        reward_source = _top_level_source(REWARDS, "handstand_feet_height_l2_exp")

        self.assertIn("torch.square", reward_source)
        self.assertIn("torch.exp(-feet_height_error / std**2)", reward_source)
        self.assertIn("_handstand_support_quality(", reward_source)


if __name__ == "__main__":
    unittest.main()
