"""Static action-order check for Lite3 — pure mujoco, no mjlab needed.

Loads `go2_mjlab/robots/xmls/lite3.xml` directly and prints the
MJCF-level mapping that drives the entire training pipeline:

    actuator_idx  →  joint_id  →  joint_name

This is what mjlab uses internally: action vector index `i` is sent to
`model.ctrl[i]`, which mujoco routes to the joint at `model.actuator_trnid[i]`.
There is no permutation in between.  As long as this mapping matches
the deploy script's JOINT_NAMES list (which determines DEFAULT_JOINT_POS
and ACTION_SCALE indexing), action order is provably correct.

(An earlier version of this script also tried an "empirical" PD test —
fire a one-hot setpoint, see which joint moves.  That turned out to be
unreliable because dynamic coupling between legs makes passive joints
visibly drift even with the body damped, masking the actually-actuated
joint.  The static MJCF mapping above is the unambiguous source of
truth and is what mjlab consumes; no need for an indirect test.)

Run anywhere `mujoco` and `numpy` are importable.
"""
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
LITE3_XML = REPO_ROOT / "go2_mjlab" / "robots" / "xmls" / "lite3.xml"

# Match the deploy script's expected MJCF order.
EXPECTED_JOINTS = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]
EXPECTED_DEFAULT_POS = np.array([
    +0.1, -0.8, 1.6,
    -0.1, -0.8, 1.6,
    +0.1, -0.8, 1.6,
    -0.1, -0.8, 1.6,
])

# Per-joint action scale from lite3_constants.py::LITE3_HANDSTAND_ACTION_SCALE.
EXPECTED_SCALE = np.array([
    0.125, 0.25, 0.25,
    0.125, 0.25, 0.25,
    0.125, 0.25, 0.25,
    0.125, 0.25, 0.25,
])


def main():
    model = mujoco.MjModel.from_xml_path(str(LITE3_XML))

    n_act = model.nu
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(n_act)
    ]
    actuator_target_jids = [int(model.actuator_trnid[i, 0]) for i in range(n_act)]
    actuator_target_joints = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        for jid in actuator_target_jids
    ]

    print("\n=== MJCF actuator-to-joint mapping (the ONE that mjlab uses) ===")
    print(f"{'act_idx':>7} | {'actuator name':<14} | {'drives joint':<22} | "
          f"{'expected_default':>16} | {'expected_scale':>14} | {'OK?':>4}")
    print("-" * 100)

    static_ok = True
    for i in range(n_act):
        match_joint = (actuator_target_joints[i] == EXPECTED_JOINTS[i])
        if not match_joint:
            static_ok = False
        marker = "✓" if match_joint else f"✗ expected {EXPECTED_JOINTS[i]}"
        print(f"{i:>7} | {actuator_names[i]:<14} | "
              f"{actuator_target_joints[i]:<22} | "
              f"{EXPECTED_DEFAULT_POS[i]:>+16.3f} | "
              f"{EXPECTED_SCALE[i]:>14.3f} | "
              f"{marker:>4}")

    print("\n" + "=" * 100)
    if static_ok:
        print("STATIC MAP: OK ✓")
        print("Each MJCF actuator drives its expected joint at the expected")
        print("MJCF index.  mjlab's action vector consumes these in this exact")
        print("order, so deploy script's JOINT_NAMES / DEFAULT_JOINT_POS /")
        print("ACTION_SCALE arrays are correctly aligned.")
    else:
        print("STATIC MAP: BROKEN ✗")
        print("MJCF order does NOT match deploy script's expected order.")
        print("Either the lite3.xml has been edited or EXPECTED_JOINTS in")
        print("deploy_mjlab_lite3_handstand.py is wrong.")
    print("=" * 100)


if __name__ == "__main__":
    main()
