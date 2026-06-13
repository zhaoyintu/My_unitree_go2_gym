"""Probe self-collision on Lite3 at the handstand nominal pose.

Temporarily rewrites lite3.xml in memory to enable self-collision
(contype=1 conaffinity=1 + drops the THIGH-SHANK and SHANK-FOOT excludes)
then forces the joint vector to the handstand desire pose. Steps the
model once and reports any contact pairs that are firing along with
their penetration depth.

The goal is to answer two questions before flipping self-collision on
in the trained model:
  1. Does the nominal handstand pose already interpenetrate (i.e. would
     enabling collision immediately destabilize the simulation)?
  2. If yes, which pairs and by how much?
"""
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
LITE3_XML = REPO_ROOT / "go2_mjlab" / "robots" / "xmls" / "lite3.xml"

JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]

# Front (handstand stance): HipY=0.283, Knee=2.0
# Rear (folded up):         HipY=-0.8,  Knee=1.6
DESIRE = [
    0.0, 0.283, 2.0,
    0.0, 0.283, 2.0,
    0.0, -0.8,  1.6,
    0.0, -0.8,  1.6,
]


def patch_xml(xml_text: str, drop_excludes: tuple[str, ...]) -> str:
    # Flip the collision-class default to allow self-collision.
    xml_text = re.sub(
        r'(<default class="collision">\s*<geom[^/]*contype=")0(" conaffinity=")1(")',
        r"\g<1>1\g<2>1\g<3>",
        xml_text,
    )

    # Drop selected excludes so the named body pairs can collide again.
    for body_pair in drop_excludes:
        body1, body2 = body_pair.split("|")
        xml_text = re.sub(
            rf'\s*<exclude body1="{body1}" body2="{body2}"/>',
            "",
            xml_text,
        )
    return xml_text


def main() -> None:
    raw = LITE3_XML.read_text()
    drop = (
        "HL_THIGH|HL_SHANK",
        "HR_THIGH|HR_SHANK",
        "FL_THIGH|FL_SHANK",
        "FR_THIGH|FR_SHANK",
    )
    patched = patch_xml(raw, drop)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Copy the meshes/asset folder next to the patched XML so includes work.
        out_xml = td_path / "lite3.xml"
        out_xml.write_text(patched)
        # mujoco needs assets — symlink the original asset folder.
        for entry in LITE3_XML.parent.iterdir():
            if entry.name == "lite3.xml":
                continue
            (td_path / entry.name).symlink_to(entry)

        model = mujoco.MjModel.from_xml_path(str(out_xml))
        data = mujoco.MjData(model)

        # Drive the joints to the handstand desire pose. The free joint sits
        # at the start of qpos (7 dofs) when the body is floating; we put the
        # base in handstand orientation (pitch +pi/2) to mimic the runtime.
        data.qpos[:3] = (0.0, 0.0, 0.39)
        # quaternion for +pi/2 pitch around body-y: w=cos(pi/4), y=sin(pi/4)
        data.qpos[3:7] = (np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0)

        for i, name in enumerate(JOINT_NAMES):
            jid = model.joint(name).qposadr[0]
            data.qpos[jid] = DESIRE[i]

        mujoco.mj_forward(model, data)

        ncon = data.ncon
        print(f"# contacts at handstand nominal pose: {ncon}")
        for c in range(ncon):
            contact = data.contact[c]
            g1, g2 = contact.geom1, contact.geom2
            n1 = model.geom(g1).name
            n2 = model.geom(g2).name
            depth = -contact.dist  # negative dist = penetration
            print(f"  {n1:30s}  <->  {n2:30s}  depth={depth*1000:+.2f} mm")

        # Also report the THIGH-SHANK closest-point distance per leg, even
        # when not in contact, so we know the headroom.
        for prefix in ("FL", "FR", "HL", "HR"):
            try:
                thigh_id = model.geom(f"{prefix}_THIGH_collision").id
                shank_id = model.geom(f"{prefix}_SHANK_collision").id
            except KeyError:
                continue
            from_to = np.zeros(6, dtype=np.float64)
            d = mujoco.mj_geomDistance(model, data, thigh_id, shank_id, 1.0, from_to)
            print(f"{prefix}_THIGH <-> {prefix}_SHANK distance: {d*1000:+.2f} mm")

        print()
        print("--- Knee sweep: HL leg, rear-handstand pose ---")
        hl_knee_qpos = model.joint("HL_Knee_joint").qposadr[0]
        for knee in np.arange(0.6, 2.5, 0.1):
            data.qpos[hl_knee_qpos] = knee
            mujoco.mj_forward(model, data)
            thigh_id = model.geom("HL_THIGH_collision").id
            shank_id = model.geom("HL_SHANK_collision").id
            ft = np.zeros(6, dtype=np.float64)
            d = mujoco.mj_geomDistance(model, data, thigh_id, shank_id, 1.0, ft)
            marker = "  <-- touch" if abs(d) < 0.005 else ("  <-- INTERFERE" if d < 0 else "")
            print(f"  Knee={knee:.2f}  distance={d*1000:+7.2f} mm{marker}")

        print()
        print("--- Knee sweep: FL leg, front-handstand pose ---")
        fl_knee_qpos = model.joint("FL_Knee_joint").qposadr[0]
        # restore HL to whatever, then sweep FL
        for knee in np.arange(0.6, 2.5, 0.1):
            data.qpos[fl_knee_qpos] = knee
            mujoco.mj_forward(model, data)
            thigh_id = model.geom("FL_THIGH_collision").id
            shank_id = model.geom("FL_SHANK_collision").id
            ft = np.zeros(6, dtype=np.float64)
            d = mujoco.mj_geomDistance(model, data, thigh_id, shank_id, 1.0, ft)
            marker = "  <-- touch" if abs(d) < 0.005 else ("  <-- INTERFERE" if d < 0 else "")
            print(f"  Knee={knee:.2f}  distance={d*1000:+7.2f} mm{marker}")


if __name__ == "__main__":
    main()
