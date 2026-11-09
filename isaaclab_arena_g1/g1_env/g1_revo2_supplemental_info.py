# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Supplemental information for the G1 with BrainCo Revo 2 hands (6 actuated joints per hand)."""

from dataclasses import dataclass, field
from typing import Literal

import isaaclab_arena_g1.g1_env.g1_revo2_constants as revo2
from isaaclab_arena_g1.g1_env.g1_supplemental_info import G1SupplementalInfo

_SIDES = ("left", "right")


def _hand_joints(side: str) -> list[str]:
    return [f"{side}_{suffix}" for suffix in revo2.REVO2_ACTUATED_JOINT_SUFFIXES]


def _revo2_joint_limits() -> dict[str, list[float]]:
    """Body joint limits from the base G1 plus the Revo 2 hand limits (Dex3 hand limits dropped)."""
    limits = {name: list(lim) for name, lim in G1SupplementalInfo().joint_limits.items() if "_hand_" not in name}
    for side in _SIDES:
        limits[f"{side}_thumb_metacarpal_joint"] = list(revo2.REVO2_THUMB_METACARPAL_LIMITS)
        limits[f"{side}_thumb_proximal_joint"] = list(revo2.REVO2_THUMB_PROXIMAL_LIMITS)
        for finger in ("index", "middle", "ring", "pinky"):
            limits[f"{side}_{finger}_proximal_joint"] = list(revo2.REVO2_FINGER_PROXIMAL_LIMITS)
    return limits


def _revo2_joint_groups() -> dict[str, dict[str, list[str]]]:
    """Base G1 groups with the hand groups replaced by the Revo 2 joints."""
    groups = {name: {k: list(v) for k, v in group.items()} for name, group in G1SupplementalInfo().joint_groups.items()}
    for side in _SIDES:
        groups[f"{side}_hand"] = {"joints": _hand_joints(side), "groups": []}
    return groups


def _posture(per_hand: dict[str, float]) -> dict[str, float]:
    return {f"{side}_{suffix}": value for side in _SIDES for suffix, value in per_hand.items()}


@dataclass
class G1Revo2SupplementalInfo(G1SupplementalInfo):
    """G1 supplemental information with Revo 2 hands.

    Mimic (``*_distal``) and locked (``*_tip``) hand joints are not listed here: they follow their
    proximal joint. ``waist_location`` mirrors the Dex3 variants in ``g1_supplemental_info``:

    - ``lower_body``: waist belongs to the lower body (default).
    - ``upper_body``: waist moves to the upper body.
    - ``lower_and_upper_body``: waist is in both groups.
    """

    waist_location: Literal["lower_body", "upper_body", "lower_and_upper_body"] = "lower_body"

    left_hand_actuated_joints: list[str] = field(default_factory=lambda: _hand_joints("left"))
    right_hand_actuated_joints: list[str] = field(default_factory=lambda: _hand_joints("right"))
    joint_limits: dict[str, list[float]] = field(default_factory=_revo2_joint_limits)
    joint_groups: dict[str, dict[str, list[str]]] = field(default_factory=_revo2_joint_groups)
    hand_open_posture: dict[str, float] | None = field(default_factory=lambda: _posture(revo2.REVO2_OPEN_POSTURE))
    hand_closed_posture: dict[str, float] | None = field(default_factory=lambda: _posture(revo2.REVO2_CLOSED_POSTURE))

    def __post_init__(self):
        if self.waist_location not in ("lower_body", "upper_body", "lower_and_upper_body"):
            raise ValueError(f"Invalid waist_location: {self.waist_location}")
        groups = dict(self.joint_groups)
        if self.waist_location == "upper_body":
            groups["lower_body"] = {"joints": [], "groups": ["legs"]}
            groups["upper_body_no_hands"] = {"joints": [], "groups": ["arms", "waist"]}
        elif self.waist_location == "lower_and_upper_body":
            groups["upper_body_no_hands"] = {"joints": [], "groups": ["arms", "waist"]}
        self.joint_groups = groups
