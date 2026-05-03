from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_INIT = REPO_ROOT / "legged_gym/envs/__init__.py"
TASK_DIR = (
    REPO_ROOT
    / "legged_gym/envs/Lite3_Stand"
    / "Lite3_Handstand_MjlabGatedTrackingFrameStack"
)
CFG = TASK_DIR / "Lite3_handstand_mjlab_gated_tracking_frame_stack_Config.py"
ENV = TASK_DIR / "Lite3_handstand_mjlab_gated_tracking_frame_stack.py"


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


class Lite3IsaacGymMjlabGatedTrackingFrameStackStaticTest(unittest.TestCase):
    def test_frame_stack_task_is_registered_separately(self) -> None:
        source = _source(ENV_INIT)

        self.assertIn('"lite3_handstand_mjlab_gated_tracking_reward"', source)
        self.assertIn('"lite3_handstand_mjlab_gated_tracking_frame_stack"', source)
        self.assertIn("Lite3_Handstand_MjlabGatedTrackingFrameStack", source)

    def test_frame_stack_config_inherits_gated_tracking_reward(self) -> None:
        source = _source(CFG)
        config_source = _top_level_source(CFG, "Lite3Cfg_MjlabGatedTrackingFrameStack")

        self.assertIn("Lite3Cfg_MjlabGatedTrackingReward", source)
        self.assertIn("frame_stack = 10", config_source)
        self.assertIn("c_frame_stack = 10", config_source)
        self.assertIn("num_observations = int(frame_stack * num_single_obs)", config_source)
        self.assertIn(
            "num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)",
            config_source,
        )
        self.assertIn(
            "experiment_name = 'lite3_handstand_mjlab_gated_tracking_frame_stack'",
            source,
        )

    def test_frame_stack_env_stacks_actor_and_critic_observations(self) -> None:
        source = _source(ENV)
        class_source = _top_level_source(ENV, "Lite3_legstand_mjlab_gated_tracking_frame_stack")
        reset_source = _method_source(
            ENV,
            "Lite3_legstand_mjlab_gated_tracking_frame_stack",
            "reset_idx",
        )
        init_source = _method_source(
            ENV,
            "Lite3_legstand_mjlab_gated_tracking_frame_stack",
            "_init_buffers",
        )
        obs_source = _method_source(
            ENV,
            "Lite3_legstand_mjlab_gated_tracking_frame_stack",
            "compute_observations",
        )

        self.assertIn("Lite3_legstand_mjlab_gated_tracking_reward", source)
        self.assertIn("deque(maxlen=self.cfg.env.frame_stack)", init_source)
        self.assertIn("deque(maxlen=self.cfg.env.c_frame_stack)", init_source)
        self.assertIn("self.cfg.env.num_single_obs", init_source)
        self.assertIn("self.cfg.env.single_num_privileged_obs", init_source)
        self.assertIn("super().reset_idx(env_ids)", reset_source)
        self.assertIn("self.obs_history[i][env_ids] *= 0", reset_source)
        self.assertIn("self.critic_history[i][env_ids] *= 0", reset_source)
        self.assertIn("super().compute_observations()", obs_source)
        self.assertIn("single_obs = self.obs_buf.clone()", obs_source)
        self.assertIn("single_privileged_obs = self.privileged_obs_buf.clone()", obs_source)
        self.assertIn("self.obs_history.append(single_obs)", obs_source)
        self.assertIn("self.critic_history.append(single_privileged_obs)", obs_source)
        self.assertIn("torch.stack", obs_source)
        self.assertIn("torch.cat", obs_source)
        self.assertNotIn("class rewards", class_source)


if __name__ == "__main__":
    unittest.main()
