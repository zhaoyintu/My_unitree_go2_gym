from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy_mujoco_viewer/deploy_mjlab_lite3_handstand.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("deploy_mjlab_lite3_handstand", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load module from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise AssertionError(f"Missing geom: {name}")
    return geom_id


def _actuator_id(model: mujoco.MjModel, name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id < 0:
        raise AssertionError(f"Missing actuator: {name}")
    return actuator_id


class Lite3DeployModelTest(unittest.TestCase):
    def test_build_model_matches_training_full_collision_cfg(self) -> None:
        module = _load_module()
        model = module.build_model()

        body_geom_id = _geom_id(model, "TORSO_collision")
        self.assertEqual(int(model.geom_contype[body_geom_id]), 1)
        self.assertEqual(int(model.geom_conaffinity[body_geom_id]), 1)
        self.assertEqual(int(model.geom_condim[body_geom_id]), 1)
        self.assertEqual(int(model.geom_priority[body_geom_id]), 0)
        self.assertEqual(model.geom_solref[body_geom_id].tolist(), [0.01, 1.0])

        shank_geom_id = _geom_id(model, "FL_SHANK_collision")
        self.assertEqual(int(model.geom_contype[shank_geom_id]), 1)
        self.assertEqual(int(model.geom_conaffinity[shank_geom_id]), 1)
        self.assertEqual(int(model.geom_condim[shank_geom_id]), 1)
        self.assertEqual(model.geom_solref[shank_geom_id].tolist(), [0.01, 1.0])

        foot_geom_id = _geom_id(model, "FL_FOOT_collision")
        self.assertEqual(int(model.geom_contype[foot_geom_id]), 1)
        self.assertEqual(int(model.geom_conaffinity[foot_geom_id]), 1)
        self.assertEqual(int(model.geom_condim[foot_geom_id]), 6)
        self.assertEqual(int(model.geom_priority[foot_geom_id]), 1)
        self.assertEqual(model.geom_friction[foot_geom_id].tolist(), [1.0, 0.005, 0.0005])
        self.assertEqual(model.geom_solref[foot_geom_id].tolist(), [0.01, 1.0])

        visual_geom_id = _geom_id(model, "TORSO_visual")
        self.assertEqual(int(model.geom_contype[visual_geom_id]), 0)
        self.assertEqual(int(model.geom_conaffinity[visual_geom_id]), 0)

    def test_position_actuators_allow_unclamped_targets_like_mjlab(self) -> None:
        module = _load_module()
        model = module.build_model()

        actuator_id = _actuator_id(model, "FL_HipY")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "FL_HipY_joint")
        joint_range = model.jnt_range[joint_id]
        expected_delta = module.EFFORT_LIMITS[1] / module.KP[1]

        self.assertEqual(int(model.actuator_ctrllimited[actuator_id]), 0)
        self.assertEqual(
            model.actuator_ctrlrange[actuator_id].tolist(),
            [joint_range[0] - expected_delta, joint_range[1] + expected_delta],
        )

    def test_model_uses_training_simulation_options(self) -> None:
        module = _load_module()
        model = module.build_model()

        self.assertEqual(model.opt.timestep, module.SIM_DT)
        self.assertEqual(model.opt.integrator, mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        self.assertEqual(model.opt.solver, mujoco.mjtSolver.mjSOL_NEWTON)
        self.assertEqual(model.opt.cone, mujoco.mjtCone.mjCONE_PYRAMIDAL)
        self.assertEqual(model.opt.jacobian, mujoco.mjtJacobian.mjJAC_AUTO)
        self.assertEqual(model.opt.iterations, 10)
        self.assertEqual(model.opt.ls_iterations, 20)
        self.assertEqual(model.opt.tolerance, 1e-8)
        self.assertEqual(model.opt.ls_tolerance, 0.01)


if __name__ == "__main__":
    unittest.main()
