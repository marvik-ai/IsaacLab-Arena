# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The Revo 2 distal joints are commanded from the measured position of their proximal joint."""

import numpy as np
import torch
from types import SimpleNamespace

import pytest
import warp as wp

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.run_policy import postprocess_actions

wp.init()

_JOINT_NAMES = [
    "left_index_proximal_joint",
    "left_index_distal_joint",
    "left_thumb_proximal_joint",
    "left_thumb_distal_joint",
]
_WBC_ORDER = {"left_index_proximal_joint": 0, "left_thumb_proximal_joint": 1}


def _process(commanded: list[float], measured: list[float]) -> torch.Tensor:
    robot_data = SimpleNamespace(
        joint_names=_JOINT_NAMES, joint_pos=wp.from_torch(torch.tensor([measured], dtype=torch.float32))
    )
    wbc_action = {"q": np.array([commanded], dtype=np.float32)}
    return postprocess_actions(wbc_action, robot_data, _WBC_ORDER, torch.device("cpu"))


def test_distal_follows_the_measured_proximal_position_not_the_command():
    # The index is commanded to 1.2 but is held at 0.5 by the object; the distal must follow 0.5, not 1.2.
    actions = _process(commanded=[1.2, 0.8], measured=[0.5, 0.0, 0.4, 0.0])
    assert actions[0, 0].item() == pytest.approx(1.2)  # proximal keeps the command
    assert actions[0, 1].item() == pytest.approx(1.155 * 0.5)
    assert actions[0, 3].item() == pytest.approx(1.0 * 0.4)  # thumb uses its own multiplier


def test_distal_target_is_never_negative_when_the_proximal_is_pushed_below_zero():
    actions = _process(commanded=[1.2, 0.8], measured=[-0.3, 0.0, -0.1, 0.0])
    assert actions[0, 1].item() == 0.0
    assert actions[0, 3].item() == 0.0
