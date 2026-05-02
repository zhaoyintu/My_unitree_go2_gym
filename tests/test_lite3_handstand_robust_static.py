from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INIT = REPO_ROOT / "go2_mjlab/__init__.py"
LITE3_ENV_CFG = REPO_ROOT / "go2_mjlab/config/lite3_handstand/env_cfgs.py"
LITE3_RL_CFG = REPO_ROOT / "go2_mjlab/config/lite3_handstand/rl_cfg.py"
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


class Lite3HandstandRobustStaticTest(unittest.TestCase):
    def test_registers_separate_robust_task(self) -> None:
        source = _source(INIT)

        self.assertIn("unitree_lite3_handstand_robust_env_cfg", source)
        self.assertIn("unitree_lite3_handstand_robust_ppo_runner_cfg", source)
        self.assertIn('"Mjlab-Lite3-Handstand-Robust"', source)
        self.assertIn("env_cfg=unitree_lite3_handstand_robust_env_cfg(play=False)", source)
        self.assertIn("play_env_cfg=unitree_lite3_handstand_robust_env_cfg(play=True)", source)

    def test_robust_runner_uses_separate_experiment_name(self) -> None:
        source = _source(LITE3_RL_CFG)
        function_source = _top_level_source(
            LITE3_RL_CFG,
            "unitree_lite3_handstand_robust_ppo_runner_cfg",
        )

        self.assertIn("lite3_handstand_robust", function_source)
        self.assertIn("unitree_lite3_handstand_ppo_runner_cfg()", function_source)
        self.assertIn("def unitree_lite3_handstand_ppo_runner_cfg", source)

    def test_robust_task_adds_zero_command_standing_objectives(self) -> None:
        function_source = _top_level_source(
            LITE3_ENV_CFG,
            "unitree_lite3_handstand_robust_env_cfg",
        )

        self.assertIn("rel_standing_envs = 0.25", function_source)
        self.assertIn('rewards["tracking_lin_vel_zero"].weight = -0.8', function_source)
        self.assertIn('rewards["tracking_ang_vel_zero"].weight = -0.6', function_source)
        self.assertIn('"zero_stance_contact"', function_source)
        self.assertIn("go2_mdp.handstand_stance_contact_zero", function_source)
        self.assertIn('"zero_joint_vel"', function_source)
        self.assertIn("go2_mdp.handstand_joint_vel_zero", function_source)

    def test_robust_task_gates_walking_gait_rewards_by_command(self) -> None:
        function_source = _top_level_source(
            LITE3_ENV_CFG,
            "unitree_lite3_handstand_robust_env_cfg",
        )

        for reward_name in ("contact", "feet_air_time", "feet_clearance"):
            self.assertIn(f'rewards["{reward_name}"].params["command_name"] = "twist"', function_source)
            self.assertIn(f'rewards["{reward_name}"].params["moving_threshold"] = 0.1', function_source)

    def test_robust_task_strengthens_push_randomization(self) -> None:
        function_source = _top_level_source(
            LITE3_ENV_CFG,
            "unitree_lite3_handstand_robust_env_cfg",
        )

        self.assertIn('events["push_robot"].interval_range_s = (3.0, 6.0)', function_source)
        self.assertIn('"x": (-0.4, 0.4)', function_source)
        self.assertIn('"pitch": (-0.45, 0.45)', function_source)
        self.assertIn('"yaw": (-0.65, 0.65)', function_source)

    def test_reward_helpers_support_zero_command_stance(self) -> None:
        contact_source = _top_level_source(REWARDS, "handstand_contact")
        air_time_source = _top_level_source(REWARDS, "handstand_feet_air_time")
        clearance_source = _top_level_source(REWARDS, "handstand_feet_clearance")
        stance_source = _top_level_source(REWARDS, "handstand_stance_contact_zero")
        joint_vel_source = _top_level_source(REWARDS, "handstand_joint_vel_zero")

        self.assertIn("command_name: str | None = None", contact_source)
        self.assertIn("_handstand_moving_command_mask", contact_source)
        self.assertIn("command_name: str | None = None", air_time_source)
        self.assertIn("_handstand_moving_command_mask", air_time_source)
        self.assertIn("command_name: str | None = None", clearance_source)
        self.assertIn("_handstand_moving_command_mask", clearance_source)
        self.assertIn("_handstand_standing_command_mask", stance_source)
        self.assertIn("n_contact == len(foot_indices)", stance_source)
        self.assertIn("_handstand_standing_command_mask", joint_vel_source)
        self.assertIn("torch.sum(torch.square(joint_vel)", joint_vel_source)


if __name__ == "__main__":
    unittest.main()
