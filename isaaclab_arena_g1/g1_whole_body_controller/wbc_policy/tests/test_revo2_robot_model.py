# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the G1 + BrainCo Revo 2 robot model pieces that do not need Isaac Sim."""

import numpy as np
import os
import yaml
from types import SimpleNamespace

import pytest

from isaaclab_arena_g1.g1_env.g1_revo2_supplemental_info import G1Revo2SupplementalInfo
from isaaclab_arena_g1.g1_env.g1_supplemental_info import G1SupplementalInfo
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.revo2_coupling import distal_hand_joint_couplings
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.revo2_urdf import (
    build_pinocchio_urdf,
    build_sim_urdf,
    check_joint_order,
)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../../../g1_env/config")

_SYNTHETIC_URDF = """<?xml version="1.0"?>
<robot name="t">
  <link name="base"/>
  <link name="prox"><visual><geometry><mesh filename="meshes/prox.STL"/></geometry></visual></link>
  <link name="dist"/>
  <link name="tip"/>
  <joint name="finger_proximal_joint" type="revolute">
    <parent link="base"/><child link="prox"/><axis xyz="0 1 0"/>
    <limit lower="0" upper="1.4" effort="1" velocity="1"/>
  </joint>
  <joint name="finger_distal_joint" type="revolute">
    <parent link="prox"/><child link="dist"/><axis xyz="0 1 0"/>
    <limit lower="0" upper="1.6" effort="1" velocity="1"/>
    <mimic joint="finger_proximal_joint" multiplier="1.155" offset="0"/>
  </joint>
  <joint name="finger_tip_joint" type="revolute">
    <parent link="dist"/><child link="tip"/><axis xyz="0 1 0"/>
    <limit lower="1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _joints_order() -> dict[str, int]:
    with open(os.path.join(_CONFIG_DIR, "loco_manip_g1_revo2_joints_order_41dof.yaml")) as f:
        return yaml.safe_load(f)


def test_joints_order_has_41_unique_contiguous_indices():
    order = _joints_order()
    assert sorted(order.values()) == list(range(41))


@pytest.mark.parametrize("waist_location", ["lower_body", "upper_body", "lower_and_upper_body"])
def test_revo2_supplemental_info_matches_joint_order(waist_location):
    info = G1Revo2SupplementalInfo(waist_location=waist_location)
    actuated = info.body_actuated_joints + info.left_hand_actuated_joints + info.right_hand_actuated_joints
    assert set(actuated) == set(_joints_order())
    assert set(info.joint_limits) == set(_joints_order())
    assert len(info.left_hand_actuated_joints) == len(info.right_hand_actuated_joints) == 6


def test_revo2_postures_are_within_joint_limits():
    info = G1Revo2SupplementalInfo()
    for posture in (info.hand_open_posture, info.hand_closed_posture):
        assert set(posture) == set(info.left_hand_actuated_joints + info.right_hand_actuated_joints)
        for name, value in posture.items():
            lower, upper = info.joint_limits[name]
            assert lower <= value <= upper, name


def test_dex3_supplemental_info_is_unchanged():
    info = G1SupplementalInfo()
    assert len(info.left_hand_actuated_joints) == 7
    assert info.hand_open_posture is None and info.hand_closed_posture is None
    assert not any("thumb_metacarpal" in name for name in info.joint_limits)


def test_build_pinocchio_urdf_fixes_mimic_and_locked_joints(tmp_path):
    src = tmp_path / "src" / "robot.urdf"
    src.parent.mkdir()
    src.write_text(_SYNTHETIC_URDF)
    dst = tmp_path / "out" / "robot.urdf"

    movable = build_pinocchio_urdf(str(src), str(dst))

    assert movable == ["finger_proximal_joint"]
    text = dst.read_text()
    assert 'type="fixed"' in text and "<mimic" not in text
    # Mesh paths are rewritten so the copy can live anywhere.
    assert str(src.parent / "meshes" / "prox.STL") in text


def test_build_sim_urdf_keeps_mimic_joints_and_uses_relative_meshes(tmp_path):
    src = tmp_path / "src" / "robot.urdf"
    src.parent.mkdir()
    src.write_text(_SYNTHETIC_URDF)
    dst = tmp_path / "src" / "robot_sim.urdf"

    movable = build_sim_urdf(str(src), str(dst))

    # The mimic distal joint stays (PhysX couples it); the locked tip joint is fixed.
    assert movable == ["finger_proximal_joint", "finger_distal_joint"]
    text = dst.read_text()
    assert "<mimic" in text
    assert 'filename="meshes/prox.STL"' in text


def test_distal_joints_follow_their_proximal_joint():
    names = ["torso", "left_index_proximal_joint", "left_index_distal_joint", "right_thumb_distal_joint"]
    names.append("right_thumb_proximal_joint")

    couplings = distal_hand_joint_couplings(names)

    assert couplings == [(2, 1, 1.155), (3, 4, 1.0)]


def test_hands_without_distal_joints_have_no_couplings():
    assert distal_hand_joint_couplings(["left_hand_thumb_0_joint", "left_hand_index_1_joint"]) == []


def test_distal_joint_without_proximal_joint_is_an_error():
    with pytest.raises(ValueError, match="no matching proximal joint"):
        distal_hand_joint_couplings(["left_index_distal_joint"])


def test_check_joint_order_reports_mismatch():
    check_joint_order(["a", "b"], {"a": 0, "b": 1})
    with pytest.raises(ValueError, match="Missing from URDF"):
        check_joint_order(["a"], {"a": 0, "b": 1})
    with pytest.raises(ValueError, match="Not in joint order"):
        check_joint_order(["a", "b", "c"], {"a": 0, "b": 1})


def _controller_hand_fn():
    """The hand-posture method, without building an IK solver (needs pinocchio, so skipped if missing)."""
    pytest.importorskip("pinocchio")
    from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.g1_wbc_upperbody_ik.g1_wbc_upperbody_controller import (
        G1WBCUpperbodyController,
    )

    info = G1Revo2SupplementalInfo()
    fake_self = SimpleNamespace(full_robot=SimpleNamespace(supplemental_info=info))
    return (lambda state, side: G1WBCUpperbodyController.get_side_hand_joint_pos(fake_self, state, side)), info


def test_revo2_hand_command_interpolates_between_open_and_closed():
    hand_pos, info = _controller_hand_fn()
    names = info.right_hand_actuated_joints
    open_q = np.array([info.hand_open_posture[n] for n in names])
    closed_q = np.array([info.hand_closed_posture[n] for n in names])

    np.testing.assert_allclose(hand_pos(0.0, "right"), open_q)
    np.testing.assert_allclose(hand_pos(1.0, "right"), closed_q)
    np.testing.assert_allclose(hand_pos(0.5, "right"), 0.5 * (open_q + closed_q))
    # Out-of-range commands are clipped, and the right hand is not sign-flipped (unlike Dex3).
    np.testing.assert_allclose(hand_pos(3.0, "right"), closed_q)
    np.testing.assert_allclose(
        hand_pos(-1.0, "left"), [info.hand_open_posture[n] for n in info.left_hand_actuated_joints]
    )
