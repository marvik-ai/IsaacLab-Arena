# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Coupling of the Revo 2 distal finger joints to their proximal joints.

The URDF declares each ``*_distal_joint`` as a mimic of the matching ``*_proximal_joint``. The USD converter only
records that for the Newton backend and removes it from the PhysX layer, so PhysX does not couple the joints. The
whole-body controller therefore commands the distal joints itself: ``distal_target = multiplier * proximal_position``,
using the measured proximal position so that a blocked proximal joint is not pushed back by its distal drive.
"""

import isaaclab_arena_g1.g1_env.g1_revo2_constants as revo2

_DISTAL_SUFFIX = "_distal_joint"
_PROXIMAL_SUFFIX = "_proximal_joint"


def distal_hand_joint_couplings(joint_names: list[str]) -> list[tuple[int, int, float]]:
    """Find every Revo 2 distal joint and the joint that drives it.

    Args:
        joint_names: Simulator joint names, in the simulator's order.

    Returns:
        ``(distal_index, proximal_index, multiplier)`` for each distal joint present. Empty for hands without
        distal joints (Dex3).

    Raises:
        ValueError: If a distal joint has no matching proximal joint.
    """
    couplings = []
    for distal_index, name in enumerate(joint_names):
        if not name.endswith(_DISTAL_SUFFIX):
            continue
        proximal_name = name[: -len(_DISTAL_SUFFIX)] + _PROXIMAL_SUFFIX
        if proximal_name not in joint_names:
            raise ValueError(f"Distal joint '{name}' has no matching proximal joint '{proximal_name}'.")
        multiplier = (
            revo2.REVO2_THUMB_DISTAL_MIMIC_MULTIPLIER
            if "_thumb_" in name
            else revo2.REVO2_FINGER_DISTAL_MIMIC_MULTIPLIER
        )
        couplings.append((distal_index, joint_names.index(proximal_name), multiplier))
    return couplings
