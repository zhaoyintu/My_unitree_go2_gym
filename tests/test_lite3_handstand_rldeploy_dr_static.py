from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INIT = REPO_ROOT / "go2_mjlab/__init__.py"
BASE_ENV_CFG = REPO_ROOT / "go2_mjlab/config/lite3_handstand/env_cfgs.py"
ENV_CFG = REPO_ROOT / "go2_mjlab/config/lite3_handstand_rldeploy/env_cfgs.py"
RL_CFG = REPO_ROOT / "go2_mjlab/config/lite3_handstand_rldeploy/rl_cfg.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _top_level_source(path: Path, name: str) -> str:
    source = _source(path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} not found in {path}")


class Lite3HandstandRLDeployDRStaticTest(unittest.TestCase):
    def test_registers_separate_rldeploy_dr_task(self) -> None:
        source = _source(INIT)

        self.assertIn("unitree_lite3_handstand_rldeploy_dr_env_cfg", source)
        self.assertIn("unitree_lite3_handstand_rldeploy_dr_ppo_runner_cfg", source)
        self.assertIn('"Mjlab-Lite3-Handstand-RLDeploy-DR"', source)
        self.assertIn("env_cfg=unitree_lite3_handstand_rldeploy_dr_env_cfg(play=False)", source)
        self.assertIn("play_env_cfg=unitree_lite3_handstand_rldeploy_dr_env_cfg(play=True)", source)

    def test_runner_uses_separate_experiment_name(self) -> None:
        source = _source(RL_CFG)
        function_source = _top_level_source(
            RL_CFG,
            "unitree_lite3_handstand_rldeploy_dr_ppo_runner_cfg",
        )

        self.assertIn("unitree_lite3_handstand_rldeploy_ppo_runner_cfg()", function_source)
        self.assertIn('cfg.experiment_name = "lite3_handstand_rldeploy_dr"', function_source)
        self.assertIn("RslRlOnPolicyRunnerCfg", source)

    def test_dr_task_inherits_rldeploy_policy_contract(self) -> None:
        function_source = _top_level_source(
            ENV_CFG,
            "unitree_lite3_handstand_rldeploy_dr_env_cfg",
        )

        self.assertIn("cfg = unitree_lite3_handstand_rldeploy_env_cfg(play=play)", function_source)
        self.assertNotIn('"base_lin_vel"', function_source)
        self.assertNotIn('"robot/imu_lin_vel"', function_source)
        self.assertNotIn("ObservationGroupCfg", function_source)

    def test_dr_task_adds_supported_sim_to_real_randomization(self) -> None:
        source = _source(ENV_CFG)
        helper_source = _top_level_source(ENV_CFG, "_add_rldeploy_sim2real_dr_events")

        self.assertIn("LITE3_LINK_INERTIA_ALPHA_RANGE", source)
        self.assertIn("math.log(0.9) / 2.0", source)
        self.assertIn("math.log(1.1) / 2.0", source)

        expected_events = {
            '"pd_gains"': "envs_mdp.dr.pd_gains",
            '"motor_strength"': "envs_mdp.dr.effort_limits",
            '"joint_friction"': "envs_mdp.dr.joint_friction",
            '"joint_damping"': "envs_mdp.dr.joint_damping",
            '"joint_armature"': "envs_mdp.dr.joint_armature",
            '"link_inertia"': "envs_mdp.dr.pseudo_inertia",
        }
        for event_name, dr_func in expected_events.items():
            self.assertIn(event_name, helper_source)
            self.assertIn(dr_func, helper_source)

        self.assertIn('"kp_range": (0.9, 1.1)', helper_source)
        self.assertIn('"kd_range": (0.9, 1.1)', helper_source)
        self.assertIn('"effort_limit_range": (0.8, 1.2)', helper_source)
        self.assertIn('"ranges": (0.01, 0.2)', helper_source)
        self.assertIn('"ranges": (0.0, 0.2)', helper_source)
        self.assertIn('"ranges": (0.005, 0.015)', helper_source)
        self.assertIn('"alpha_range": LITE3_LINK_INERTIA_ALPHA_RANGE', helper_source)
        self.assertIn('"bias_range"] = (-0.02, 0.02)', helper_source)
        self.assertIn('SceneEntityCfg("robot", joint_names=(".*",))', helper_source)
        self.assertIn(
            'SceneEntityCfg("robot", body_names=LITE3_LINK_INERTIA_BODY_NAMES)',
            helper_source,
        )

    def test_dr_task_keeps_current_gentle_base_push(self) -> None:
        base_source = _top_level_source(BASE_ENV_CFG, "unitree_lite3_handstand_env_cfg")
        helper_source = _top_level_source(ENV_CFG, "_add_rldeploy_sim2real_dr_events")

        self.assertIn('interval_range_s=(4.0, 8.0)', base_source)
        self.assertIn('"x": (-0.3, 0.3)', base_source)
        self.assertIn('"y": (-0.3, 0.3)', base_source)
        self.assertIn('"z": (-0.2, 0.2)', base_source)
        self.assertIn('"roll": (-0.35, 0.35)', base_source)
        self.assertIn('"pitch": (-0.35, 0.35)', base_source)
        self.assertIn('"yaw": (-0.5, 0.5)', base_source)
        self.assertNotIn('"push_robot"', helper_source)
        self.assertNotIn("push_by_setting_velocity", helper_source)
        self.assertNotIn("max_push_vel_xy", helper_source)
        self.assertNotIn('"x": (-1.0, 1.0)', helper_source)
        self.assertNotIn('"yaw": (-1.0, 1.0)', helper_source)


if __name__ == "__main__":
    unittest.main()
