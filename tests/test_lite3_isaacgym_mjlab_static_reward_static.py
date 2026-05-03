from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_INIT = REPO_ROOT / "legged_gym/envs/__init__.py"
OLD_CFG = (
    REPO_ROOT
    / "legged_gym/envs/Lite3_Stand/Lite3_Handstand_MjlabReward"
    / "Lite3_handstand_mjlab_reward_Config.py"
)
OLD_ENV = (
    REPO_ROOT
    / "legged_gym/envs/Lite3_Stand/Lite3_Handstand_MjlabReward"
    / "Lite3_handstand_mjlab_reward.py"
)
CFG = (
    REPO_ROOT
    / "legged_gym/envs/Lite3_Stand/Lite3_Handstand_MjlabStaticReward"
    / "Lite3_handstand_mjlab_static_reward_Config.py"
)
ENV = (
    REPO_ROOT
    / "legged_gym/envs/Lite3_Stand/Lite3_Handstand_MjlabStaticReward"
    / "Lite3_handstand_mjlab_static_reward.py"
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


class Lite3IsaacGymMjlabStaticRewardStaticTest(unittest.TestCase):
    def test_static_reward_task_is_registered_without_replacing_old_task(self) -> None:
        source = _source(ENV_INIT)

        self.assertIn('"lite3_handstand_mjlab_reward"', source)
        self.assertIn('"lite3_handstand_mjlab_static_reward"', source)
        self.assertIn("Lite3_Handstand_MjlabStaticReward", source)
        self.assertIn("Lite3_legstand_mjlab_static_reward", source)

    def test_old_mjlab_reward_task_keeps_original_reward_shape(self) -> None:
        old_cfg = _source(OLD_CFG)
        old_env = _source(OLD_ENV)

        self.assertIn("handstand_orientation = -1.0", old_cfg)
        self.assertIn("tracking_lin_vel = 2.5", old_cfg)
        self.assertIn("feet_air_time = 2.0", old_cfg)
        self.assertNotIn("orientation_sharpness", old_cfg)
        self.assertNotIn("_handstand_quality", old_env)

    def test_static_config_inherits_old_task_and_only_changes_rewards(self) -> None:
        source = _source(CFG)
        config_source = _top_level_source(CFG, "Lite3Cfg_MjlabStaticReward")

        self.assertIn("Lite3Cfg_MjlabReward", source)
        self.assertIn("class rewards", config_source)
        self.assertNotIn("class domain_rand", config_source)
        self.assertNotIn("class control", config_source)
        self.assertNotIn("class init_state", config_source)
        self.assertIn("experiment_name = 'lite3_handstand_mjlab_static_reward'", source)

    def test_static_reward_scales_prioritize_handstand_before_tracking(self) -> None:
        source = _source(CFG)

        expected_scales = (
            "handstand_orientation = 2.0",
            "handstand_feet_on_air = 1.0",
            "handstand_feet_height_exp = 8.0",
            "tracking_lin_vel = 0.4",
            "tracking_ang_vel = 0.4",
            "feet_air_time = 0.0",
            "feet_clearance = 0.0",
            "default_pos_reward = 2.0",
            "rear_foot_height_target = 0.56",
            "orientation_sharpness = 2.0",
        )
        for scale in expected_scales:
            self.assertIn(scale, source)

    def test_static_reward_gates_tracking_and_densifies_rear_foot_rewards(self) -> None:
        tracking_source = _method_source(
            ENV,
            "Lite3_legstand_mjlab_static_reward",
            "_reward_tracking_lin_vel",
        )
        height_source = _method_source(
            ENV,
            "Lite3_legstand_mjlab_static_reward",
            "_reward_handstand_feet_height_exp",
        )
        air_source = _method_source(
            ENV,
            "Lite3_legstand_mjlab_static_reward",
            "_reward_handstand_feet_on_air",
        )
        contact_source = _method_source(
            ENV,
            "Lite3_legstand_mjlab_static_reward",
            "_reward_contact",
        )

        self.assertIn("* self._handstand_quality()", tracking_source)
        self.assertIn("0.55 * lift_quality", height_source)
        self.assertIn("0.35 * target_quality", height_source)
        self.assertIn("(~contact).float().mean(dim=1)", air_source)
        self.assertIn("contact.float().mean(dim=1)", contact_source)


if __name__ == "__main__":
    unittest.main()
