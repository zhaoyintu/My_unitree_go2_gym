from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_INIT = REPO_ROOT / "legged_gym/envs/__init__.py"
CFG = (
    REPO_ROOT
    / "legged_gym/envs/Lite3_Stand/Lite3_Handstand_MjlabReward"
    / "Lite3_handstand_mjlab_reward_Config.py"
)
ENV = (
    REPO_ROOT
    / "legged_gym/envs/Lite3_Stand/Lite3_Handstand_MjlabReward"
    / "Lite3_handstand_mjlab_reward.py"
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _top_level_source(path: Path, name: str) -> str:
    source = _source(path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} not found in {path}")


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = _source(path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(source, item) or ""
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


class Lite3IsaacGymMjlabRewardStaticTest(unittest.TestCase):
    def test_task_is_registered_as_reward_only_variant(self) -> None:
        source = _source(ENV_INIT)

        self.assertIn("Lite3_Handstand_MjlabReward", source)
        self.assertIn("Lite3_legstand_mjlab_reward", source)
        self.assertIn("Lite3Cfg_MjlabReward", source)
        self.assertIn('"lite3_handstand_mjlab_reward"', source)

    def test_config_overrides_rewards_without_overriding_domain_randomization(self) -> None:
        source = _source(CFG)
        config_source = _top_level_source(CFG, "Lite3Cfg_MjlabReward")

        self.assertIn("class rewards", config_source)
        self.assertNotIn("class domain_rand", config_source)
        self.assertNotIn("class control", config_source)
        self.assertNotIn("class init_state", config_source)
        self.assertIn("experiment_name = 'lite3_handstand_mjlab_reward'", source)

    def test_reward_scales_follow_mjlab_lite3_handstand(self) -> None:
        source = _source(CFG)

        expected_scales = (
            "alive = 1.0",
            "base_height = 1.5",
            "default_pos = -1.0",
            "default_pos_reward = 1.0",
            "default_hip_pos = -0.5",
            "collision = 0.0",
            "base_contact = -2.0",
            "thigh_collision = -1.0",
            "shank_collision = -2.0",
            "ang_xz = 0.0",
            "torques = 0.0",
        )
        for scale in expected_scales:
            self.assertIn(scale, source)

    def test_reward_functions_remove_batch_quality_gate_and_fix_zero_velocity_penalty(self) -> None:
        tracking_source = _method_source(ENV, "Lite3_legstand_mjlab_reward", "_reward_tracking_lin_vel")
        zero_source = _method_source(ENV, "Lite3_legstand_mjlab_reward", "_reward_tracking_lin_vel_zero")

        self.assertNotIn("torch.mean(self.rew_hanstand)", tracking_source)
        self.assertIn("torch.exp", tracking_source)
        self.assertNotIn("torch.exp", zero_source)
        self.assertIn("x_error + y_error", zero_source)
        self.assertIn("torch.norm(self.commands[:, :2]", zero_source)

    def test_pose_and_symmetry_rewards_cover_front_and_rear_pairs(self) -> None:
        pose_source = _method_source(ENV, "Lite3_legstand_mjlab_reward", "_reward_default_pos_reward")
        symmetry_source = _method_source(ENV, "Lite3_legstand_mjlab_reward", "_reward_symmetric_joints")

        self.assertNotIn("[:, 6:]", pose_source)
        self.assertIn("torch.exp(-torch.sum(torch.abs(err), dim=1))", pose_source)
        self.assertIn("err_front", symmetry_source)
        self.assertIn("err_rear", symmetry_source)
        self.assertIn("0.5 * (err_front + err_rear)", symmetry_source)

    def test_contact_rewards_are_split_and_air_time_state_resets(self) -> None:
        source = _source(ENV)
        reset_source = _method_source(ENV, "Lite3_legstand_mjlab_reward", "reset_idx")

        self.assertIn("def _reward_base_contact", source)
        self.assertIn("def _reward_thigh_collision", source)
        self.assertIn("def _reward_shank_collision", source)
        self.assertIn("self.last_contacts[env_ids] = False", reset_source)


if __name__ == "__main__":
    unittest.main()
