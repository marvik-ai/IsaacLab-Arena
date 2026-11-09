# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build the Pinocchio-only URDF for the G1 with BrainCo Revo 2 hands.

The simulator keeps the Revo 2 mimic (``*_distal``) and locked (``*_tip``) joints, but the WBC
``RobotModel`` indexes Pinocchio's configuration vector by the joint-order yaml, which lists only the
actuated joints. Turning the mimic and locked joints into fixed joints gives a Pinocchio model whose
configuration vector has exactly the actuated joints, in URDF order.
"""

import os
import xml.etree.ElementTree as ET


def _is_locked(joint: ET.Element) -> bool:
    limit = joint.find("limit")
    if limit is None:
        return False
    lower, upper = limit.get("lower"), limit.get("upper")
    if lower is None or upper is None:
        return False
    return float(lower) == float(upper)


def _build_urdf(src_urdf: str, dst_urdf: str, fix_mimic: bool, relative_meshes: bool) -> list[str]:
    tree = ET.parse(src_urdf)
    root = tree.getroot()
    src_dir = os.path.dirname(os.path.abspath(src_urdf))
    dst_dir = os.path.dirname(os.path.abspath(dst_urdf))

    movable = []
    for joint in root.findall("joint"):
        is_mimic = joint.find("mimic") is not None
        if joint.get("type") in ("revolute", "continuous", "prismatic") and (
            (fix_mimic and is_mimic) or _is_locked(joint)
        ):
            joint.set("type", "fixed")
            for tag in ("axis", "limit", "mimic", "dynamics"):
                for child in joint.findall(tag):
                    joint.remove(child)
        if joint.get("type") != "fixed":
            movable.append(joint.get("name"))

    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename and "://" not in filename and not os.path.isabs(filename):
            mesh_path = os.path.normpath(os.path.join(src_dir, filename))
            mesh.set("filename", os.path.relpath(mesh_path, dst_dir) if relative_meshes else mesh_path)

    os.makedirs(dst_dir, exist_ok=True)
    tree.write(dst_urdf, xml_declaration=True, encoding="utf-8")
    return movable


def build_pinocchio_urdf(src_urdf: str, dst_urdf: str) -> list[str]:
    """Write a copy of ``src_urdf`` with mimic and locked joints fixed (for the WBC Pinocchio model).

    Mesh paths are rewritten to absolute paths (relative to ``src_urdf``) so the copy can live anywhere.

    Args:
        src_urdf: Path to the combined G1 + Revo 2 URDF.
        dst_urdf: Path to write the converted URDF to. Parent directories are created.

    Returns:
        Names of the joints that are still movable, in URDF order.
    """
    return _build_urdf(src_urdf, dst_urdf, fix_mimic=True, relative_meshes=False)


def build_sim_urdf(src_urdf: str, dst_urdf: str) -> list[str]:
    """Write a copy of ``src_urdf`` with only the locked ``*_tip`` joints fixed (input for the USD conversion).

    The mimic ``*_distal`` joints are kept as joints so the fingertips can bend; the whole-body controller commands
    them from their proximal joint (see ``revo2_coupling``). Fixing the tips also
    removes a quirk of the upstream URDF: the left tip joints are limited to ``lower=upper=1`` rad while the right
    ones are limited to 0, which would bend the left fingertips by one radian in the simulator.

    Mesh paths are rewritten relative to ``dst_urdf``, so the copy keeps working when it is written next to the meshes
    and the folder is later seen under a different path (for example inside a container).

    Returns:
        Names of the joints that are still movable, in URDF order.
    """
    return _build_urdf(src_urdf, dst_urdf, fix_mimic=False, relative_meshes=True)


def main() -> None:
    """Command line: ``revo2_urdf.py {sim,pinocchio} SRC_URDF DST_URDF`` (standard library only)."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("variant", choices=["sim", "pinocchio"])
    parser.add_argument("src_urdf")
    parser.add_argument("dst_urdf")
    args = parser.parse_args()
    build = build_sim_urdf if args.variant == "sim" else build_pinocchio_urdf
    movable = build(args.src_urdf, args.dst_urdf)
    print(f"Wrote {args.dst_urdf} with {len(movable)} movable joints.")


def check_joint_order(movable_joints: list[str], joints_order: dict[str, int]) -> None:
    """Raise if the movable joints do not match the WBC joint-order yaml (same names, same count)."""
    missing = sorted(set(joints_order) - set(movable_joints))
    extra = sorted(set(movable_joints) - set(joints_order))
    if missing or extra:
        raise ValueError(
            f"Pinocchio URDF joints do not match the WBC joint order. Missing from URDF: {missing}. "
            f"Not in joint order: {extra}."
        )


if __name__ == "__main__":
    main()
