from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LITE3_ENV_CFG = REPO_ROOT / "go2_mjlab/config/lite3_handstand/env_cfgs.py"
REWARDS = REPO_ROOT / "go2_mjlab/mdp/rewards.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_source(path: Path, function_name: str) -> str:
    source = _source(path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{function_name} not found in {path}")


class Lite3HandstandStaticTest(unittest.TestCase):
    def test_lite3_handstand_uses_handstand_specific_rewards(self) -> None:
        source = _source(LITE3_ENV_CFG)

        self.assertIn('func=go2_mdp.handstand_base_height', source)
        self.assertIn('func=go2_mdp.handstand_lin_vel_z', source)
        self.assertIn('func=go2_mdp.handstand_ang_vel_yz', source)
        self.assertNotIn('"ang_xz_penalty"', source)

    def test_lite3_actor_obs_excludes_unavailable_base_linear_velocity(self) -> None:
        source = _source(LITE3_ENV_CFG)
        actor_block = source.split("actor_terms = {", 1)[1].split("critic_terms = {", 1)[0]
        critic_block = source.split("critic_terms = {", 1)[1].split("observations = {", 1)[0]

        self.assertNotIn('"base_lin_vel"', actor_block)
        self.assertNotIn('"robot/imu_lin_vel"', actor_block)
        self.assertIn('"base_lin_vel"', critic_block)
        self.assertIn('"robot/imu_lin_vel"', critic_block)

    def test_lite3_feet_clearance_tracks_front_stance_feet(self) -> None:
        source = _source(LITE3_ENV_CFG)
        feet_clearance_block = source.split('"feet_clearance": RewardTermCfg(', 1)[1].split(
            '"default_pos": RewardTermCfg(', 1
        )[0]

        self.assertIn('"foot_indices": (0, 1)', feet_clearance_block)
        self.assertIn('"target_foot_height": 0.06', feet_clearance_block)

    def test_lite3_domain_randomization_uses_condim6_friction_and_base_body_com(self) -> None:
        source = _source(LITE3_ENV_CFG)

        self.assertIn('"foot_friction_slide"', source)
        self.assertIn('"axes": [0]', source)
        self.assertIn('"foot_friction_spin"', source)
        self.assertIn('"axes": [1]', source)
        self.assertIn('"distribution": "log_uniform"', source)
        self.assertIn('"foot_friction_roll"', source)
        self.assertIn('"axes": [2]', source)

        base_com_block = source.split('"base_com": EventTermCfg(', 1)[1].split(
            '"encoder_bias": EventTermCfg(', 1
        )[0]
        self.assertIn('SceneEntityCfg("robot", body_names=("TORSO",))', base_com_block)
        self.assertIn('"operation": "add"', base_com_block)
        self.assertIn('0: (-0.05, 0.05)', base_com_block)
        self.assertIn('1: (-0.05, 0.05)', base_com_block)
        self.assertIn('2: (-0.05, 0.05)', base_com_block)

    def test_lite3_domain_randomization_uses_velocity_push_and_encoder_bias(self) -> None:
        source = _source(LITE3_ENV_CFG)

        self.assertIn('func=envs_mdp.push_by_setting_velocity', source)
        self.assertIn('"velocity_range": {', source)
        self.assertIn('"encoder_bias": EventTermCfg(', source)
        self.assertIn('func=envs_mdp.dr.encoder_bias', source)
        self.assertIn('"bias_range": (-0.01, 0.01)', source)

    def test_tracking_lin_vel_zero_is_a_penalty_not_a_negative_reward(self) -> None:
        function_source = _function_source(REWARDS, "handstand_tracking_lin_vel_zero")

        self.assertNotIn("torch.exp", function_source)
        self.assertIn("x_error + y_error", function_source)
        self.assertIn("cmd_near_zero", function_source)

    def test_symmetric_joints_compares_front_and_rear_pairs_without_mutating_view(self) -> None:
        function_source = _function_source(REWARDS, "symmetric_joints")

        self.assertIn(".clone()", function_source)
        self.assertIn("err_front", function_source)
        self.assertIn("err_rear", function_source)
        self.assertIn("0.5 * (err_front + err_rear)", function_source)


if __name__ == "__main__":
    unittest.main()
