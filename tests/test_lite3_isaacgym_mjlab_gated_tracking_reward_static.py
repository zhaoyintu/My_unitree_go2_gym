from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_INIT = REPO_ROOT / "legged_gym/envs/__init__.py"
CFG = (
    REPO_ROOT
    / "legged_gym/envs/Lite3_Stand/Lite3_Handstand_MjlabGatedTrackingReward"
    / "Lite3_handstand_mjlab_gated_tracking_reward_Config.py"
)
ENV = (
    REPO_ROOT
    / "legged_gym/envs/Lite3_Stand/Lite3_Handstand_MjlabGatedTrackingReward"
    / "Lite3_handstand_mjlab_gated_tracking_reward.py"
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


class Lite3IsaacGymMjlabGatedTrackingRewardStaticTest(unittest.TestCase):
    def test_gated_tracking_task_is_registered_as_third_comparison_task(self) -> None:
        source = _source(ENV_INIT)

        self.assertIn('"lite3_handstand_mjlab_reward"', source)
        self.assertIn('"lite3_handstand_mjlab_static_reward"', source)
        self.assertIn('"lite3_handstand_mjlab_gated_tracking_reward"', source)
        self.assertIn("Lite3_Handstand_MjlabGatedTrackingReward", source)

    def test_gated_tracking_config_inherits_static_reward_task(self) -> None:
        source = _source(CFG)
        config_source = _top_level_source(CFG, "Lite3Cfg_MjlabGatedTrackingReward")

        self.assertIn("Lite3Cfg_MjlabStaticReward", source)
        self.assertIn("class rewards", config_source)
        self.assertNotIn("class domain_rand", config_source)
        self.assertNotIn("class control", config_source)
        self.assertNotIn("class init_state", config_source)
        self.assertIn(
            "experiment_name = 'lite3_handstand_mjlab_gated_tracking_reward'",
            source,
        )

    def test_gated_tracking_restores_large_tracking_weights(self) -> None:
        source = _source(CFG)

        self.assertIn("tracking_lin_vel = 2.5", source)
        self.assertIn("tracking_ang_vel = 2.5", source)
        self.assertIn("feet_air_time = 0.0", source)
        self.assertIn("feet_clearance = 0.0", source)

    def test_gated_tracking_gate_uses_rear_air_quality(self) -> None:
        source = _source(ENV)
        gate_source = _method_source(
            ENV,
            "Lite3_legstand_mjlab_gated_tracking_reward",
            "_handstand_quality",
        )
        air_source = _method_source(
            ENV,
            "Lite3_legstand_mjlab_gated_tracking_reward",
            "_rear_feet_air_quality",
        )

        self.assertIn("Lite3_legstand_mjlab_static_reward", source)
        self.assertIn("_rear_feet_air_quality", gate_source)
        self.assertIn("0.20 + 0.80 * rear_air_quality", gate_source)
        self.assertIn("(~contact).float().mean(dim=1)", air_source)


if __name__ == "__main__":
    unittest.main()
