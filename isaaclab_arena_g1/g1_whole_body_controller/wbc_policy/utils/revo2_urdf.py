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
    if limit is None or limit.get("lower") is None or limit.get("upper") is None:
        return False
    return float(limit.get("lower")) == float(limit.get("upper"))


def build_pinocchio_urdf(src_urdf: str, dst_urdf: str) -> list[str]:
    """Write a copy of ``src_urdf`` with mimic and locked joints fixed.

    Mesh paths are rewritten to absolute paths (relative to ``src_urdf``) so the copy can live anywhere.

    Args:
        src_urdf: Path to the combined G1 + Revo 2 URDF.
        dst_urdf: Path to write the converted URDF to. Parent directories are created.

    Returns:
        Names of the joints that are still movable, in URDF order.
    """
    tree = ET.parse(src_urdf)
    root = tree.getroot()
    src_dir = os.path.dirname(os.path.abspath(src_urdf))

    movable = []
    for joint in root.findall("joint"):
        if joint.get("type") in ("revolute", "continuous", "prismatic") and (
            joint.find("mimic") is not None or _is_locked(joint)
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
            mesh.set("filename", os.path.normpath(os.path.join(src_dir, filename)))

    os.makedirs(os.path.dirname(os.path.abspath(dst_urdf)), exist_ok=True)
    tree.write(dst_urdf, xml_declaration=True, encoding="utf-8")
    return movable


def check_joint_order(movable_joints: list[str], joints_order: dict[str, int]) -> None:
    """Raise if the movable joints do not match the WBC joint-order yaml (same names, same count)."""
    missing = sorted(set(joints_order) - set(movable_joints))
    extra = sorted(set(movable_joints) - set(joints_order))
    if missing or extra:
        raise ValueError(
            f"Pinocchio URDF joints do not match the WBC joint order. Missing from URDF: {missing}. "
            f"Not in joint order: {extra}."
        )
