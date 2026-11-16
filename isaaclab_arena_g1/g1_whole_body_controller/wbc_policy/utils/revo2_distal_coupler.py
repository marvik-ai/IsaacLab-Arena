# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Kinematic coupling of the Revo 2 distal finger joints to their proximal joints.

The real hand drives each distal joint from the proximal one through a mechanism that cannot be bent apart by hand,
but PhysX does not couple the two joints (see ``revo2_coupling``). With separate drives, a fingertip pressed against
an object straightens the distal joint relative to the proximal one.

:class:`Revo2DistalCoupler` enforces ``q_distal = multiplier * q_proximal`` on every physics step by writing the
distal joint position and velocity. The proximal joint stays in charge: it is driven normally and the distal joint
follows it. The coupling is a position override, so a load on the fingertip is not passed on to the proximal drive and
the distal joint can exert more force than the real fingertip.

The coupler also keeps the actuated hand joints inside their limits. PhysX enforces joint limits softly, so a finger
pressed hard against something (for example the other hand) is pushed backwards past its zero position.
"""

from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

import warp as wp

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.revo2_coupling import distal_hand_joint_couplings

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

DISTAL_KINEMATIC_ENV_VAR = "ARENA_G1_REVO2_DISTAL_KINEMATIC"
"""Set to ``0`` to turn the kinematic coupling off and only drive the distal joints towards their target."""


class Revo2DistalCoupler:
    """Overwrites each distal joint with ``multiplier * proximal`` (position and velocity)."""

    def __init__(self, joint_names: list[str], device: torch.device | str):
        couplings = distal_hand_joint_couplings(joint_names)
        self._distal_ids = [distal for distal, _, _ in couplings]
        self._proximal_ids = [proximal for _, proximal, _ in couplings]
        self._multiplier = torch.tensor([multiplier for _, _, multiplier in couplings], device=device)
        self._limited_ids = sorted(
            {proximal for proximal in self._proximal_ids}
            | {i for i, name in enumerate(joint_names) if name.endswith("_thumb_metacarpal_joint")}
        )
        self._proximal_in_limited = [self._limited_ids.index(proximal) for proximal in self._proximal_ids]

    def apply(self, asset: Articulation) -> None:
        """Write the coupled distal joint state. Call it once per physics step, after the targets are set."""
        joint_pos = wp.to_torch(asset.data.joint_pos)
        joint_vel = wp.to_torch(asset.data.joint_vel)
        limits = wp.to_torch(asset.data.joint_pos_limits)[:, self._limited_ids]
        lower, upper = limits[..., 0], limits[..., 1]
        pos = joint_pos[:, self._limited_ids]
        vel = joint_vel[:, self._limited_ids]
        outside = (pos < lower) | (pos > upper)
        if outside.any():
            # Put joints that were pushed past a limit back on it and stop them moving further out.
            pushing_out = ((pos <= lower) & (vel < 0.0)) | ((pos >= upper) & (vel > 0.0))
            pos = pos.clamp(min=lower, max=upper)
            vel = torch.where(pushing_out, torch.zeros_like(vel), vel)
            asset.write_joint_position_to_sim_index(position=pos, joint_ids=self._limited_ids)
            asset.write_joint_velocity_to_sim_index(velocity=vel, joint_ids=self._limited_ids)
        proximal_pos = pos[:, self._proximal_in_limited]
        proximal_vel = vel[:, self._proximal_in_limited]
        distal_pos = self._multiplier * proximal_pos
        distal_vel = self._multiplier * proximal_vel
        asset.write_joint_position_to_sim_index(position=distal_pos, joint_ids=self._distal_ids)
        asset.write_joint_velocity_to_sim_index(velocity=distal_vel, joint_ids=self._distal_ids)


def create_revo2_distal_coupler(joint_names: list[str], device: torch.device | str) -> Revo2DistalCoupler | None:
    """Build the coupler, or return None for hands without distal joints (Dex3) or when it is switched off."""
    if os.environ.get(DISTAL_KINEMATIC_ENV_VAR, "1") == "0":
        return None
    if not distal_hand_joint_couplings(joint_names):
        return None
    return Revo2DistalCoupler(joint_names, device)
