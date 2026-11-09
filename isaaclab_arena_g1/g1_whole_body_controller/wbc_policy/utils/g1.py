# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import yaml
from typing import Literal

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

from isaaclab_arena_g1.g1_env.g1_revo2_supplemental_info import G1Revo2SupplementalInfo
from isaaclab_arena_g1.g1_env.g1_supplemental_info import (
    G1SupplementalInfo,
    G1SupplementalInfoWaistLowerAndUpperBody,
    G1SupplementalInfoWaistUpperBody,
)
from isaaclab_arena_g1.g1_env.robot_model import RobotModel
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.revo2_urdf import (
    build_pinocchio_urdf,
    check_joint_order,
)

# Environment variable with the path to the combined G1 + BrainCo Revo 2 URDF (kept outside the repo).
REVO2_URDF_ENV_VAR = "ARENA_G1_REVO2_URDF"
_REVO2_JOINTS_ORDER_PATH = os.path.join(
    os.path.dirname(__file__), "../../../g1_env/config/loco_manip_g1_revo2_joints_order_41dof.yaml"
)


def _instantiate_g1_revo2_robot_model(waist_location: str) -> RobotModel:
    """Build the WBC robot model for the G1 with Revo 2 hands from the URDF named by ``ARENA_G1_REVO2_URDF``."""
    src_urdf = os.environ.get(REVO2_URDF_ENV_VAR)
    if not src_urdf or not os.path.isfile(src_urdf):
        raise FileNotFoundError(
            f"hand_type='revo2' needs the combined G1 + Revo 2 URDF. Set {REVO2_URDF_ENV_VAR} to its path "
            f"(got {src_urdf!r})."
        )
    joints_order_path = os.path.normpath(_REVO2_JOINTS_ORDER_PATH)
    with open(joints_order_path) as f:
        joints_order = yaml.safe_load(f)

    # Pinocchio only sees the actuated joints: mimic and locked hand joints become fixed joints.
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "isaaclab_arena", "g1_revo2")
    pinocchio_urdf = os.path.join(cache_dir, "g1_29dof_revo2_pinocchio.urdf")
    movable_joints = build_pinocchio_urdf(src_urdf, pinocchio_urdf)
    check_joint_order(movable_joints, joints_order)

    return RobotModel(
        pinocchio_urdf,
        os.path.dirname(os.path.abspath(src_urdf)),
        supplemental_info=G1Revo2SupplementalInfo(waist_location=waist_location),
        joints_order_path=joints_order_path,
    )


def instantiate_g1_robot_model(
    waist_location: Literal["lower_body", "upper_body", "lower_and_upper_body"] = "lower_body",
    hand_type: Literal["dex3", "revo2"] = "dex3",
):
    """
    Instantiate a G1 robot model with configurable waist location, and summarize the supplemental info.

    Args:
        hand_type: Hand model on the G1: "dex3" (default, URDF from Nucleus) or "revo2" (BrainCo Revo 2, URDF
                   from the ``ARENA_G1_REVO2_URDF`` environment variable).
        waist_location: Whether to put waist in "lower_body" (default G1 behavior),
                        "upper_body" (waist controlled with arms/manipulation via IK),
                        or "lower_and_upper_body" (waist reference from arms/manipulation
                        via IK then passed to lower body policy)

    Returns:
        RobotModel: Configured G1 robot model
    """

    if hand_type == "revo2":
        return _instantiate_g1_revo2_robot_model(waist_location)
    assert hand_type == "dex3", f"Invalid hand_type: {hand_type}. Must be 'dex3' or 'revo2'"

    robot_model_config = {
        "asset_path": f"{ISAACLAB_NUCLEUS_DIR}/Arena/wbc_policy/robot_model/g1/",
        "urdf_path": f"{ISAACLAB_NUCLEUS_DIR}/Arena/wbc_policy/robot_model/g1/g1_29dof_with_hand.urdf",
    }

    asset_path_local = retrieve_file_path(robot_model_config["asset_path"], force_download=True)
    urdf_path_local = retrieve_file_path(robot_model_config["urdf_path"], force_download=True)

    assert waist_location in [
        "lower_body",
        "upper_body",
        "lower_and_upper_body",
    ], f"Invalid waist_location: {waist_location}. Must be 'lower_body' or 'upper_body' or 'lower_and_upper_body'"
    # Choose supplemental info based on waist location preference
    if waist_location == "lower_body":
        robot_model_supplemental_info = G1SupplementalInfo()
    elif waist_location == "upper_body":
        robot_model_supplemental_info = G1SupplementalInfoWaistUpperBody()
    elif waist_location == "lower_and_upper_body":
        robot_model_supplemental_info = G1SupplementalInfoWaistLowerAndUpperBody()

    robot_model = RobotModel(
        urdf_path_local,
        asset_path_local,
        supplemental_info=robot_model_supplemental_info,
    )
    return robot_model
