# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Quest controller to Revo 2 grip scalar mapping."""

import pytest

from isaaclab_arena_g1.teleop.revo2_grip import grip_from_controller


def test_released_controller_is_open():
    assert grip_from_controller(0.0, 0.0) == 0.0
    assert grip_from_controller(0.04, 0.03) == 0.0


def test_fully_pressed_is_closed():
    assert grip_from_controller(1.0, 0.0) == 1.0
    assert grip_from_controller(0.0, 1.0) == 1.0


def test_larger_of_trigger_and_squeeze_wins_and_output_is_monotonic():
    assert grip_from_controller(0.5, 0.2) == grip_from_controller(0.2, 0.5)
    values = [grip_from_controller(t / 10, 0.0) for t in range(11)]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)


def test_midpoint_is_between_open_and_closed():
    assert grip_from_controller(0.5, 0.0) == pytest.approx(0.4737, abs=1e-3)
