# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""IsaacTeleop retargeter that outputs one grip scalar per controller (G1 with Revo 2 hands)."""

from isaacteleop.retargeting_engine.interface import BaseRetargeter, RetargeterIOType
from isaacteleop.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType, TensorGroupType
from isaacteleop.retargeting_engine.tensor_types import ControllerInput, ControllerInputIndex, FloatType

from isaaclab_arena_g1.teleop.revo2_grip import grip_from_controller


class Revo2GripRetargeter(BaseRetargeter):
    """Controller trigger/squeeze -> grip scalar in [0, 1]. Outputs 0 (open) when the controller is not tracked."""

    def __init__(self, side: str, name: str, deadzone: float = 0.05) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self._input_key = f"controller_{side}"
        self._deadzone = deadzone
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {self._input_key: OptionalType(ControllerInput())}

    def output_spec(self) -> RetargeterIOType:
        return {"grip": TensorGroupType("grip", [FloatType("grip")])}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        controller = inputs[self._input_key]
        if controller.is_none:
            outputs["grip"][0] = 0.0
            return
        trigger = controller[ControllerInputIndex.TRIGGER_VALUE]
        squeeze = controller[ControllerInputIndex.SQUEEZE_VALUE]
        outputs["grip"][0] = grip_from_controller(trigger, squeeze, self._deadzone)
