from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON = REPO_ROOT / "deploy_mujoco_viewer/lite3_rldeploy_common.py"
MJLAB_DEPLOY = REPO_ROOT / "deploy_mujoco_viewer/deploy_mjlab_lite3_handstand_rldeploy.py"
ISAACGYM_DEPLOY = REPO_ROOT / "deploy_mujoco_viewer/deploy_isaacgym_lite3_handstand_rldeploy.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _top_level_source(path: Path, name: str) -> str:
    source = _source(path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} not found in {path}")


class Lite3RLDeployDeployStaticTest(unittest.TestCase):
    def test_common_dynamics_match_rldeploy_contract(self) -> None:
        source = _source(COMMON)
        build_model_source = _top_level_source(COMMON, "build_model")
        collision_source = _top_level_source(COMMON, "apply_rldeploy_collision_cfg")
        torque_source = _top_level_source(COMMON, "compute_pd_torque")

        self.assertIn("NUM_SINGLE_OBS = 45", source)
        self.assertIn("FRAME_STACK = 10", source)
        self.assertIn("NUM_OBS = NUM_SINGLE_OBS * FRAME_STACK", source)
        self.assertIn("SIM_DT = 0.001", source)
        self.assertIn("DECIMATION = 20", source)
        self.assertIn("CTRL_DT = SIM_DT * DECIMATION", source)
        self.assertIn("np.full(12, 40.0", source)
        self.assertIn("np.full(12, 1.0", source)
        self.assertIn("np.full(12, 30.0", source)

        self.assertIn("geom.contype = 0", collision_source)
        self.assertIn("geom.conaffinity = 1", collision_source)
        self.assertIn("geom.condim = 3", collision_source)
        self.assertIn("geom.priority = 0", collision_source)
        self.assertIn("_set_array_field(geom.friction, (1.0, 0.01, 0.01))", collision_source)
        self.assertIn("_set_array_field(geom.solref, (0.005, 1.0))", collision_source)

        self.assertIn("joint.armature = 0.0", build_model_source)
        self.assertNotIn("mjGAIN_FIXED", build_model_source)
        self.assertNotIn("mjBIAS_AFFINE", build_model_source)
        self.assertIn("model.opt.timestep = SIM_DT", build_model_source)

        self.assertIn("KP * (target_q - qpos) + KD * (0.0 - qvel)", torque_source)
        self.assertIn("np.clip", torque_source)

    def test_common_actor_observation_is_term_major_45_by_10(self) -> None:
        source = _source(COMMON)
        history_source = _top_level_source(COMMON, "TermHistory")
        terms_source = _top_level_source(COMMON, "policy_terms")

        self.assertIn('ACTOR_TERM_NAMES = ("ang_vel", "grav", "cmd", "joint_pos", "joint_vel", "act")', source)
        self.assertIn("return np.concatenate([", history_source)
        self.assertIn("for name in ACTOR_TERM_NAMES", history_source)
        self.assertIn("last_action.astype(np.float32)", terms_source)
        self.assertNotIn("imu_lin_vel", terms_source)
        self.assertNotIn("base_lin_vel", terms_source)

    def test_mjlab_rldeploy_script_uses_mjlab_checkpoint_format(self) -> None:
        source = _source(MJLAB_DEPLOY)
        actor_source = _top_level_source(MJLAB_DEPLOY, "Actor")

        self.assertIn("from lite3_rldeploy_common import", source)
        self.assertIn("actor_state_dict", actor_source)
        self.assertIn("obs_normalizer._mean", actor_source)
        self.assertIn("obs_normalizer._std", actor_source)
        self.assertIn("mlp.0.weight", actor_source)
        self.assertIn("checkpoint_obs_dim != NUM_OBS", actor_source)
        self.assertIn("Mjlab-Lite3-Handstand-RLDeploy", source)

    def test_isaacgym_rldeploy_script_uses_isaacgym_checkpoint_format(self) -> None:
        source = _source(ISAACGYM_DEPLOY)
        actor_source = _top_level_source(ISAACGYM_DEPLOY, "Actor")

        self.assertIn("from lite3_rldeploy_common import", source)
        self.assertIn("model_state_dict", actor_source)
        self.assertIn('k.startswith("actor.")', actor_source)
        self.assertIn("checkpoint_obs_dim != NUM_OBS", actor_source)
        self.assertIn("checkpoint_obs_dim == 480", actor_source)
        self.assertIn("deploy_isaacgym_lite3_handstand.py", actor_source)
        self.assertNotIn("obs_normalizer._mean", actor_source)

    def test_both_scripts_drive_motor_torque_not_position_setpoints(self) -> None:
        for path in (MJLAB_DEPLOY, ISAACGYM_DEPLOY):
            source = _source(path)

            self.assertIn("compute_pd_torque", source)
            self.assertIn("data.ctrl[act_ids] = tau", source)
            self.assertNotIn("data.ctrl[act_ids] = target_q", source)
            self.assertIn("TermHistory", source)


if __name__ == "__main__":
    unittest.main()
