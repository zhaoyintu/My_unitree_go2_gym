from __future__ import annotations

import types
import unittest
import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
REWARDS_PATH = REPO_ROOT / "go2_mjlab/mdp/rewards.py"


def _load_rewards_module():
    mjlab = types.ModuleType("mjlab")
    entity = types.ModuleType("mjlab.entity")
    reward_manager = types.ModuleType("mjlab.managers.reward_manager")
    scene_entity_config = types.ModuleType("mjlab.managers.scene_entity_config")
    sensor = types.ModuleType("mjlab.sensor")

    class Entity:
        pass

    class RewardTermCfg:
        pass

    class SceneEntityCfg:
        def __init__(self, name: str, **kwargs) -> None:
            self.name = name
            self.joint_ids = kwargs.get("joint_ids", slice(None))
            self.joint_names = kwargs.get("joint_names", (".*",))

    class ContactSensor:
        pass

    entity.Entity = Entity
    reward_manager.RewardTermCfg = RewardTermCfg
    scene_entity_config.SceneEntityCfg = SceneEntityCfg
    sensor.ContactSensor = ContactSensor

    old_modules = {
        name: sys.modules.get(name)
        for name in (
            "mjlab",
            "mjlab.entity",
            "mjlab.managers.reward_manager",
            "mjlab.managers.scene_entity_config",
            "mjlab.sensor",
        )
    }
    sys.modules.update(
        {
            "mjlab": mjlab,
            "mjlab.entity": entity,
            "mjlab.managers.reward_manager": reward_manager,
            "mjlab.managers.scene_entity_config": scene_entity_config,
            "mjlab.sensor": sensor,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location("lite3_rldeploy_rewards_under_test", REWARDS_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


rewards = _load_rewards_module()


class _FakeRobot:
    def __init__(
        self,
        *,
        projected_gravity: tuple[float, float, float],
        base_z: float,
        rear_foot_z: tuple[float, float],
        lin_vel_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ang_vel_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        site_pos_w = torch.zeros((1, 4, 3), dtype=torch.float32)
        site_pos_w[0, 2, 2] = rear_foot_z[0]
        site_pos_w[0, 3, 2] = rear_foot_z[1]
        self.data = types.SimpleNamespace(
            projected_gravity_b=torch.tensor([projected_gravity], dtype=torch.float32),
            root_link_pos_w=torch.tensor([[0.0, 0.0, base_z]], dtype=torch.float32),
            root_link_lin_vel_b=torch.tensor([lin_vel_b], dtype=torch.float32),
            root_link_ang_vel_b=torch.tensor([ang_vel_b], dtype=torch.float32),
            site_pos_w=site_pos_w,
        )

    def find_sites(self, names):
        return list(range(len(names))), list(names)


class _FakeEnv:
    def __init__(
        self,
        *,
        projected_gravity: tuple[float, float, float],
        base_z: float,
        rear_foot_z: tuple[float, float],
        rear_contact: tuple[bool, bool],
        command: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        robot = _FakeRobot(
            projected_gravity=projected_gravity,
            base_z=base_z,
            rear_foot_z=rear_foot_z,
        )
        contact = torch.tensor(
            [[False, False, rear_contact[0], rear_contact[1]]],
            dtype=torch.bool,
        )
        sensor = types.SimpleNamespace(data=types.SimpleNamespace(found=contact))
        self.scene = {"robot": robot, "feet_ground_contact": sensor}
        self.command_manager = types.SimpleNamespace(
            get_command=lambda _name: torch.tensor([command], dtype=torch.float32)
        )
        self.num_envs = 1
        self.device = torch.device("cpu")


class Lite3HandstandRLDeployRewardTest(unittest.TestCase):
    def test_soft_tracking_gate_suppresses_prone_tracking_loophole(self) -> None:
        prone_env = _FakeEnv(
            projected_gravity=(0.0, 0.0, -1.0),
            base_z=0.39,
            rear_foot_z=(0.03, 0.03),
            rear_contact=(True, True),
        )
        handstand_env = _FakeEnv(
            projected_gravity=(1.0, 0.0, 0.0),
            base_z=0.39,
            rear_foot_z=(0.56, 0.56),
            rear_contact=(False, False),
        )

        prone_reward = rewards.handstand_tracking_lin_vel_soft_gate(
            prone_env,
            command_name="twist",
            sensor_name="feet_ground_contact",
        )
        handstand_reward = rewards.handstand_tracking_lin_vel_soft_gate(
            handstand_env,
            command_name="twist",
            sensor_name="feet_ground_contact",
        )

        self.assertLess(float(prone_reward.item()), 0.05)
        self.assertGreater(float(handstand_reward.item()), 0.95)

    def test_static_rear_foot_height_reward_has_lift_gradient_before_target(self) -> None:
        low_env = _FakeEnv(
            projected_gravity=(1.0, 0.0, 0.0),
            base_z=0.39,
            rear_foot_z=(0.03, 0.03),
            rear_contact=(False, False),
        )
        half_lift_env = _FakeEnv(
            projected_gravity=(1.0, 0.0, 0.0),
            base_z=0.39,
            rear_foot_z=(0.30, 0.30),
            rear_contact=(False, False),
        )
        target_env = _FakeEnv(
            projected_gravity=(1.0, 0.0, 0.0),
            base_z=0.39,
            rear_foot_z=(0.56, 0.56),
            rear_contact=(False, False),
        )

        low_reward = rewards.handstand_rear_feet_height_static(low_env, target_height=0.56)
        half_lift_reward = rewards.handstand_rear_feet_height_static(half_lift_env, target_height=0.56)
        target_reward = rewards.handstand_rear_feet_height_static(target_env, target_height=0.56)

        self.assertLess(float(low_reward.item()), float(half_lift_reward.item()))
        self.assertLess(float(half_lift_reward.item()), float(target_reward.item()))


if __name__ == "__main__":
    unittest.main()
