# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Constants for the BrainCo Revo 2 hands mounted on the G1 wrist.

Values come from ``g1_29dof_mode_15_brainco_hand.urdf`` (Unitree ``unitree_ros``,
``robots/g1_with_brainco_hand``). Each hand has six actuated joints. The ``*_distal`` joints
are URDF mimic joints of the matching ``*_proximal`` joint and the ``*_tip`` joints are locked,
so neither is commanded.
"""

# Mimic multipliers: distal = multiplier * proximal.
REVO2_FINGER_DISTAL_MIMIC_MULTIPLIER: float = 1.155
REVO2_THUMB_DISTAL_MIMIC_MULTIPLIER: float = 1.0

# Joint limits [lower, upper] in radians. Both sides share the same signs.
REVO2_THUMB_METACARPAL_LIMITS: list[float] = [0.0, 1.5184]
REVO2_THUMB_PROXIMAL_LIMITS: list[float] = [0.0, 1.0472]
REVO2_FINGER_PROXIMAL_LIMITS: list[float] = [0.0, 1.4661]

# Actuated joints per hand, without the side prefix, in the order used by the WBC joint-order yaml.
REVO2_ACTUATED_JOINT_SUFFIXES: list[str] = [
    "thumb_metacarpal_joint",
    "thumb_proximal_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
]

# Open and closed postures (radians) for the single-scalar grip command (Phase A).
# The thumb metacarpal is held in opposition for both postures. The opposition angle is a first
# guess (about 60% of the range) and must be tuned in the simulator against the apple grasp.
REVO2_THUMB_OPPOSITION_ANGLE: float = 0.9
REVO2_OPEN_POSTURE: dict[str, float] = {
    "thumb_metacarpal_joint": REVO2_THUMB_OPPOSITION_ANGLE,
    "thumb_proximal_joint": 0.0,
    "index_proximal_joint": 0.0,
    "middle_proximal_joint": 0.0,
    "ring_proximal_joint": 0.0,
    "pinky_proximal_joint": 0.0,
}
REVO2_CLOSED_POSTURE: dict[str, float] = {
    "thumb_metacarpal_joint": REVO2_THUMB_OPPOSITION_ANGLE,
    "thumb_proximal_joint": 0.9,
    "index_proximal_joint": 1.2,
    "middle_proximal_joint": 1.2,
    "ring_proximal_joint": 1.2,
    "pinky_proximal_joint": 1.2,
}
