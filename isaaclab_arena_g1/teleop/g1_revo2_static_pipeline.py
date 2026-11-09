# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""IsaacTeleop pipeline for the G1 with Revo 2 hands (static manipulation): 23D output.

Same wrist SE3 retargeting as ``g1_pink_locomanipulation_pipeline``. The differences are:

- The hand is a single grip scalar per controller (``max(trigger, squeeze)``), see ``revo2_grip.py``. The Pink action
  turns it into Revo 2 joint targets.
- The locomotion command is fixed (zero velocity, hip height 0.72), so the thumbsticks do nothing.
- The 3 torso values are not connected to any input, so the reorderer leaves them at 0.

Action layout: [left_grip, right_grip, left_wrist(7), right_wrist(7), locomotion(4), torso_rpy(3)].
"""

def _build_g1_revo2_static_pipeline():
    """Build the IsaacTeleop retargeting pipeline for G1 WBC Pink with Revo 2 hands.

    Returns:
        OutputCombiner with a single "action" output (23D flattened tensor).
    """
    from isaacteleop.retargeters import (
        LocomotionFixedRootCmdRetargeter,
        LocomotionFixedRootCmdRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    from isaaclab_arena_g1.teleop.revo2_grip_retargeter import Revo2GripRetargeter

    controllers = ControllersSource(name="controllers")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    # -------------------------------------------------------------------------
    # SE3 Absolute Pose Retargeters (left and right wrists)
    # -------------------------------------------------------------------------
    # connect(): binds transformed left/right controller grip pose -> 7D ee_pose each.
    left_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=45.0,
        target_offset_pitch=180.0,
        target_offset_yaw=-90.0,
    )
    left_se3 = Se3AbsRetargeter(left_se3_cfg, name="left_ee_pose")
    connected_left_se3 = left_se3.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )

    right_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=-135.0,
        target_offset_pitch=0.0,
        target_offset_yaw=90.0,
    )
    right_se3 = Se3AbsRetargeter(right_se3_cfg, name="right_ee_pose")
    connected_right_se3 = right_se3.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    # -------------------------------------------------------------------------
    # Grip scalar per hand (trigger or squeeze, whichever is pressed more)
    # -------------------------------------------------------------------------
    left_grip = Revo2GripRetargeter("left", name="grip_left").connect(
        {"controller_left": transformed_controllers.output(ControllersSource.LEFT)}
    )
    right_grip = Revo2GripRetargeter("right", name="grip_right").connect(
        {"controller_right": transformed_controllers.output(ControllersSource.RIGHT)}
    )

    # -------------------------------------------------------------------------
    # Fixed root command: standing still, the thumbsticks are unused (static manipulation)
    # -------------------------------------------------------------------------
    locomotion = LocomotionFixedRootCmdRetargeter(LocomotionFixedRootCmdRetargeterConfig(hip_height=0.72), name="locomotion")
    connected_locomotion = locomotion.connect({})

    # -------------------------------------------------------------------------
    # TensorReorderer: 23D for G1 WBC Pink [20D above + torso_rpy(3) from ConstantRetargeter]
    # -------------------------------------------------------------------------
    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]
    left_hand_elements = ["l_grip"]
    right_hand_elements = ["r_grip"]
    locomotion_elements = ["loco_vel_x", "loco_vel_y", "loco_rot_vel_z", "loco_hip_height"]
    torso_elements = ["torso_x", "torso_y", "torso_z"]

    output_order = (
        ["l_grip", "r_grip"]  # left_gripper(1), right_gripper(1)
        + left_ee_elements
        + right_ee_elements
        + locomotion_elements
        + torso_elements  # 3 zeros
    )

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_grip": left_hand_elements,
            "right_grip": right_hand_elements,
            "locomotion": locomotion_elements,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_grip": "scalar",
            "right_grip": "scalar",
            "locomotion": "array",
        },
    )
    # connect(): binds each retargeter output to the reorderer; flattens to 23D action.
    # torso_rpy comes from ConstantRetargeter(output_dims=3, value=0.0).
    connected_reorderer = reorderer.connect({
        "left_ee_pose": connected_left_se3.output("ee_pose"),
        "right_ee_pose": connected_right_se3.output("ee_pose"),
        "left_grip": left_grip.output("grip"),
        "right_grip": right_grip.output("grip"),
        "locomotion": connected_locomotion.output("root_command"),
    })

    return OutputCombiner({"action": connected_reorderer.output("output")})
