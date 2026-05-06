from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy_mujoco_viewer/deploy_isaacgym_lite3_handstand.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("deploy_isaacgym_lite3_handstand", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load module from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _actor_checkpoint(path: Path, obs_dim: int) -> None:
    torch.save(
        {
            "model_state_dict": {
                "actor.0.weight": torch.zeros(512, obs_dim),
                "actor.0.bias": torch.zeros(512),
                "actor.2.weight": torch.zeros(256, 512),
                "actor.2.bias": torch.zeros(256),
                "actor.4.weight": torch.zeros(128, 256),
                "actor.4.bias": torch.zeros(128),
                "actor.6.weight": torch.zeros(12, 128),
                "actor.6.bias": torch.zeros(12),
            }
        },
        path,
    )


class Lite3IsaacGymDeployModelTest(unittest.TestCase):
    def test_loads_frame_stack_checkpoint_by_rebuilding_actor_input_dim(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "frame_stack.pt"
            _actor_checkpoint(checkpoint, obs_dim=480)

            actor = module.Actor()
            actor.load(str(checkpoint))

        self.assertEqual(actor.num_obs, 480)
        self.assertEqual(actor.frame_stack, 10)
        self.assertEqual(actor.actor[0].in_features, 480)

    def test_observation_stack_matches_training_zero_prefill_order(self) -> None:
        module = _load_module()
        stack = module.ObservationStack(frame_stack=10)
        single_obs = np.arange(module.NUM_SINGLE_OBS, dtype=np.float32)

        obs = stack.build(single_obs)

        self.assertEqual(obs.shape, (480,))
        np.testing.assert_array_equal(obs[:9 * module.NUM_SINGLE_OBS], np.zeros(432, dtype=np.float32))
        np.testing.assert_array_equal(obs[-module.NUM_SINGLE_OBS:], single_obs)


if __name__ == "__main__":
    unittest.main()
