# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Map Quest controller inputs to a single Revo 2 grip scalar in [0, 1]."""


def grip_from_controller(trigger: float, squeeze: float, deadzone: float = 0.05) -> float:
    """Grip closure from the trigger and the squeeze (grip) button: the larger one wins.

    Values below ``deadzone`` read as fully open, and the rest is rescaled so the output still spans 0..1.
    """
    value = max(float(trigger), float(squeeze))
    if value <= deadzone:
        return 0.0
    return min(1.0, (value - deadzone) / (1.0 - deadzone))
