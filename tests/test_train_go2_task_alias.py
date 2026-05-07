from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_GO2 = REPO_ROOT / "scripts/train_go2.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _top_level_source(path: Path, name: str) -> str:
    source = _source(path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} not found in {path}")


def _load_alias_function():
    namespace: dict[str, object] = {}
    exec(_top_level_source(TRAIN_GO2, "_apply_task_alias"), namespace)
    return namespace["_apply_task_alias"]


class TrainGo2TaskAliasTest(unittest.TestCase):
    def test_rewrites_g1_task_name_to_lite3_task_name(self) -> None:
        apply_task_alias = _load_alias_function()
        argv = [
            "scripts/train_go2.py",
            "Mjlab-G1-Handstand-RLDeploy-RobotLab-LowDefaultPose",
            "--agent.resume",
            "True",
            "--agent.max-iterations",
            "50000",
        ]

        apply_task_alias(argv)

        self.assertEqual(
            argv,
            [
                "scripts/train_go2.py",
                "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose",
                "--agent.resume",
                "True",
                "--agent.max-iterations",
                "50000",
            ],
        )

    def test_keeps_lite3_task_name_unchanged(self) -> None:
        apply_task_alias = _load_alias_function()
        argv = [
            "scripts/train_go2.py",
            "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose",
            "--agent.resume",
            "True",
        ]

        apply_task_alias(argv)

        self.assertEqual(
            argv[1],
            "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose",
        )

    def test_does_not_rewrite_options_before_task_name(self) -> None:
        apply_task_alias = _load_alias_function()
        argv = [
            "scripts/train_go2.py",
            "--seed",
            "1",
            "Mjlab-G1-Handstand-RLDeploy-RobotLab-LowDefaultPose",
        ]

        apply_task_alias(argv)

        self.assertEqual(argv[3], "Mjlab-Lite3-Handstand-RLDeploy-RobotLab-LowDefaultPose")


if __name__ == "__main__":
    unittest.main()
