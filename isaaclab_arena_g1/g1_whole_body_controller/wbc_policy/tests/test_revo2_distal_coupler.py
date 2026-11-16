# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The kinematic coupling of the Revo 2 distal joints to their proximal joints."""

import torch
from types import SimpleNamespace

import pytest
import warp as wp

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.revo2_distal_coupler import (
    DISTAL_KINEMATIC_ENV_VAR,
    create_revo2_distal_coupler,
)

wp.init()

_NAMES = [
    "torso_joint",
    "left_index_proximal_joint",
    "left_index_distal_joint",
    "left_thumb_proximal_joint",
    "left_thumb_distal_joint",
]


class _FakeAsset:
    """Records the joint writes of the coupler."""

    def __init__(self, joint_pos: list[float], joint_vel: list[float]):
        self.joint_names = _NAMES
        self.data = SimpleNamespace(
            joint_pos=wp.from_torch(torch.tensor([joint_pos], dtype=torch.float32)),
            joint_vel=wp.from_torch(torch.tensor([joint_vel], dtype=torch.float32)),
            joint_pos_limits=wp.from_torch(torch.tensor([[[-1.0, 1.0]] * 5], dtype=torch.float32)),
        )
        self.written = {}
        self.writes = []

    def write_joint_position_to_sim_index(self, *, position, joint_ids=None, env_ids=None, full_data=False):
        self.written["position"] = (position.clone(), list(joint_ids))
        self.writes.append(("position", list(joint_ids), position.clone()))

    def write_joint_velocity_to_sim_index(self, *, velocity, joint_ids=None, env_ids=None, full_data=False):
        self.written["velocity"] = (velocity.clone(), list(joint_ids))
        self.writes.append(("velocity", list(joint_ids), velocity.clone()))


def test_distal_joints_are_written_from_the_proximal_position_and_velocity():
    asset = _FakeAsset(joint_pos=[0.0, 0.5, 0.1, 0.4, 0.0], joint_vel=[0.0, 1.0, 0.0, 0.5, 0.0])
    create_revo2_distal_coupler(_NAMES, "cpu").apply(asset)
    position, position_ids = asset.written["position"]
    velocity, velocity_ids = asset.written["velocity"]
    assert position_ids == velocity_ids == [2, 4]
    assert position[0].tolist() == pytest.approx([1.155 * 0.5, 1.0 * 0.4])
    assert velocity[0].tolist() == pytest.approx([1.155 * 1.0, 1.0 * 0.5])


def test_a_proximal_joint_pushed_past_its_limit_is_put_back_and_stopped():
    asset = _FakeAsset(joint_pos=[0.0, -1.2, 0.0, 0.3, 0.0], joint_vel=[0.0, -2.0, 0.0, 0.5, 0.0])
    create_revo2_distal_coupler(_NAMES, "cpu").apply(asset)
    writes = {(kind, tuple(ids)): value[0].tolist() for kind, ids, value in asset.writes}
    assert writes[("position", (1, 3))] == pytest.approx([-1.0, 0.3])
    assert writes[("velocity", (1, 3))] == pytest.approx([0.0, 0.5])
    # The distal joints follow the corrected proximal position.
    assert writes[("position", (2, 4))] == pytest.approx([1.155 * -1.0, 0.3])


def test_joints_inside_their_limits_are_not_rewritten():
    asset = _FakeAsset(joint_pos=[0.0, 0.2, 0.0, 0.3, 0.0], joint_vel=[0.0] * 5)
    create_revo2_distal_coupler(_NAMES, "cpu").apply(asset)
    assert [ids for _, ids, _ in asset.writes] == [[2, 4], [2, 4]]


def test_hands_without_distal_joints_get_no_coupler():
    assert create_revo2_distal_coupler(["left_hand_thumb_0_joint", "left_hand_index_1_joint"], "cpu") is None


def test_the_coupler_can_be_switched_off(monkeypatch):
    monkeypatch.setenv(DISTAL_KINEMATIC_ENV_VAR, "0")
    assert create_revo2_distal_coupler(_NAMES, "cpu") is None


def test_the_coupler_is_on_by_default(monkeypatch):
    monkeypatch.delenv(DISTAL_KINEMATIC_ENV_VAR, raising=False)
    assert create_revo2_distal_coupler(_NAMES, "cpu") is not None
