from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy_mujoco_viewer/deploy_mjlab_handstand.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("deploy_mjlab_handstand", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load module from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Go2MjlabDeployModelTest(unittest.TestCase):
    def test_lite3_checkpoint_reports_actionable_script_mismatch(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "lite3_450_obs.pt"
            torch.save(
                {
                    "actor_state_dict": {
                        "mlp.0.weight": torch.zeros(512, 450),
                        "obs_normalizer._mean": torch.zeros(1, 450),
                        "obs_normalizer._std": torch.ones(1, 450),
                    }
                },
                checkpoint,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "deploy_mjlab_lite3_handstand.py",
            ) as ctx:
                module.Actor().load(str(checkpoint))

        message = str(ctx.exception)
        self.assertIn("450", message)
        self.assertIn("480", message)
        self.assertIn("Mjlab-Lite3-Handstand", message)


if __name__ == "__main__":
    unittest.main()
