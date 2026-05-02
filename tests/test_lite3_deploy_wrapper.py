from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/deploy_lite3_handstand.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("deploy_lite3_handstand", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load module from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")


class Lite3DeployWrapperTest(unittest.TestCase):
    def test_selects_highest_iteration_from_latest_run(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "2026-05-01_10-00-00" / "model_900.pt")
            _touch(root / "2026-05-02_10-00-00" / "model_100.pt")
            _touch(root / "2026-05-02_10-00-00" / "model_2400.pt")

            selected = module.resolve_policy_path(
                policy=None,
                run=None,
                ckpt=None,
                log_root=root,
            )

            self.assertEqual(selected, root / "2026-05-02_10-00-00" / "model_2400.pt")

    def test_selects_checkpoint_from_named_run(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "run_a" / "model_100.pt")
            _touch(root / "run_b" / "model_400.pt")
            _touch(root / "run_b" / "model_800.pt")

            selected = module.resolve_policy_path(
                policy=None,
                run="run_b",
                ckpt=None,
                log_root=root,
            )

            self.assertEqual(selected, root / "run_b" / "model_800.pt")

    def test_explicit_policy_wins(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / "custom.pt"
            _touch(policy)
            _touch(root / "logs" / "run_a" / "model_100.pt")

            selected = module.resolve_policy_path(
                policy=policy,
                run="run_a",
                ckpt="model_100.pt",
                log_root=root / "logs",
            )

            self.assertEqual(selected, policy)

    def test_builds_underlying_deploy_command(self) -> None:
        module = _load_module()
        policy = Path("/tmp/model_100.pt")
        args = argparse.Namespace(
            cmd=[0.2, 0.0, -0.1],
            duration=12.5,
            no_viewer=True,
            video=Path("/tmp/out.mp4"),
            video_fps=30,
            video_width=640,
            video_height=480,
            video_camera="track",
        )

        command = module.build_deploy_command(policy, args)

        self.assertEqual(command[0], module.PYTHON)
        self.assertEqual(command[1], str(module.UNDERLYING_DEPLOY))
        self.assertIn("--policy", command)
        self.assertIn(str(policy), command)
        self.assertIn("--cmd", command)
        self.assertIn("0.2", command)
        self.assertIn("-0.1", command)
        self.assertIn("--no-viewer", command)
        self.assertIn("--video", command)
        self.assertIn("/tmp/out.mp4", command)
        self.assertIn("--video-camera", command)
        self.assertIn("track", command)


if __name__ == "__main__":
    unittest.main()
