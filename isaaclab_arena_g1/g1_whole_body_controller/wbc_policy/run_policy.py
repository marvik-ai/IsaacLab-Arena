# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

import isaaclab.utils.math as math_utils
import warp as wp
from isaaclab.assets import ArticulationData

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.revo2_coupling import distal_hand_joint_couplings

_PASSIVE_HAND_JOINT_MARKERS = ("_distal_joint", "_tip_joint", "_thumb_tip")
"""Name fragments of Revo 2 hand joints that follow another joint (distal) or are locked (tip)."""


def is_passive_hand_joint(joint_name: str) -> bool:
    """Whether a simulator hand joint follows another joint or is locked, so it is not in the WBC joint order.

    Distal joints are still commanded: ``postprocess_actions`` sets them from the measured position of their
    proximal joint.
    """
    return any(marker in joint_name for marker in _PASSIVE_HAND_JOINT_MARKERS)


def convert_sim_joint_to_wbc_joint(
    sim_joint_data: np.ndarray, sim_joint_names: list[str], wbc_joints_order: dict[str, int]
) -> np.ndarray:
    """Convert sim joint observations to WBC joint observations.

    Args:
        sim_joint_data: Sim joint data in Lab's order
        sim_joint_names: Sim joint names in Lab's order
        wbc_joints_order: WBC joint order in policy config yaml

    Returns:
        WBC joint data in WBC joint order
    """
    num_joints = len(wbc_joints_order)
    num_envs = sim_joint_data.shape[0]
    wbc_joint_data = np.zeros((num_envs, num_joints))

    # Check if sim_joint_data is a numpy array, if not, convert from torch tensor to numpy
    if not isinstance(sim_joint_data, np.ndarray):
        sim_joint_data = sim_joint_data.cpu().numpy()

    for sim_joint_name in sim_joint_names:
        sim_joint_index = sim_joint_names.index(sim_joint_name)
        if sim_joint_name not in wbc_joints_order and is_passive_hand_joint(sim_joint_name):
            continue
        assert sim_joint_name in wbc_joints_order, f"Joint {sim_joint_name} not found in loco_manip_g1_joints_order"
        wbc_joint_index = wbc_joints_order[sim_joint_name]
        wbc_joint_data[:, wbc_joint_index] = sim_joint_data[:, sim_joint_index]
    return wbc_joint_data


def prepare_observations(
    num_envs: int, robot_data: ArticulationData, wbc_joints_order: dict[str, int]
) -> dict[str, np.ndarray]:
    """Prepare observations for the policy.

    Args:
        num_envs: Number of environments
        robot_data: Robot data
        wbc_joints_order: WBC joint order in policy config yaml

    Returns:
        Observations for the policy
        - q: Joint positions
        - dq: Joint velocities
        - ddq: Joint accelerations
        - floating_base_pose: Floating base pose
        - floating_base_vel: Floating base velocity
        - floating_base_acc: Floating base acceleration
        - torso_quat: Torso quaternion
        - torso_ang_vel: Torso angular velocity
    """
    # Get robot joint observations
    sim_joint_pos = wp.to_torch(robot_data.joint_pos).cpu().numpy()
    sim_joint_vel = wp.to_torch(robot_data.joint_vel).cpu().numpy()
    sim_default_joint_pos = wp.to_torch(robot_data.default_joint_pos).cpu().numpy()
    # The WBC observation has one column per joint in the WBC joint order. For Dex3 that equals the simulator joint
    # count; for Revo 2 the simulator also has passive distal joints that the WBC does not see.
    num_joints = len(wbc_joints_order)

    # Convert joints data from Lab's order to GR00T's order saved in config yaml
    wbc_joint_pos = np.zeros((num_envs, num_joints))
    wbc_joint_vel = np.zeros((num_envs, num_joints))
    wbc_joint_acc = np.zeros((num_envs, num_joints))
    wbc_joint_pos = convert_sim_joint_to_wbc_joint(sim_joint_pos, robot_data.joint_names, wbc_joints_order)
    wbc_joint_vel = convert_sim_joint_to_wbc_joint(sim_joint_vel, robot_data.joint_names, wbc_joints_order)
    wbc_default_joint_pos = convert_sim_joint_to_wbc_joint(
        sim_default_joint_pos, robot_data.joint_names, wbc_joints_order
    )

    # Prepare obs dict for WBC policy input to G1DecoupledWholeBodyPolicy class
    assert wbc_joint_pos.shape == wbc_joint_vel.shape == wbc_joint_acc.shape == (num_envs, num_joints)

    root_link_pos_w = wp.to_torch(robot_data.root_link_pos_w).cpu().numpy()
    root_link_quat_w_xyzw = wp.to_torch(robot_data.root_link_quat_w).cpu().numpy()
    root_link_quat_w_wxyz = np.concatenate((root_link_quat_w_xyzw[:, 3:4], root_link_quat_w_xyzw[:, :3]), axis=1)
    base_pose_w = np.concatenate((root_link_pos_w, root_link_quat_w_wxyz), axis=1)
    base_lin_vel_b = wp.to_torch(robot_data.root_link_lin_vel_b).cpu().numpy()
    base_ang_vel_b = wp.to_torch(robot_data.root_link_ang_vel_b).cpu().numpy()

    base_vel_b = np.concatenate((base_lin_vel_b, base_ang_vel_b), axis=1)
    # torso link in world frame
    torso_link_pose_w = wp.to_torch(robot_data.body_link_state_w)[:, robot_data.body_names.index("torso_link"), :]
    torso_link_quat_w_xyzw = torso_link_pose_w[:, 3:7]
    torso_link_quat_w_wxyz = torch.cat((torso_link_quat_w_xyzw[:, 3:4], torso_link_quat_w_xyzw[:, :3]), dim=1)
    torso_link_ang_vel_w = torso_link_pose_w[:, -3:]

    torso_link_ang_vel_b = math_utils.quat_apply_inverse(torso_link_quat_w_xyzw, torso_link_ang_vel_w)

    projected_gravity_b = wp.to_torch(robot_data.projected_gravity_b).cpu().numpy()
    # `mdp.base_ang_vel` returns `root_ang_vel_b` (COM frame). The AGILE recurrent policy
    # was trained with this exact quantity, so expose it separately from the link-frame
    # `floating_base_vel` that other WBC policies consume.
    root_ang_vel_b = wp.to_torch(robot_data.root_ang_vel_b).cpu().numpy()

    # Prepare obs tmers
    wbc_obs = {
        "q": wbc_joint_pos,
        "dq": wbc_joint_vel,
        "default_q": wbc_default_joint_pos,
        "ddq": np.zeros((num_envs, num_joints)),  # Not used by Standing Waist Height Policy
        "tau_est": np.zeros((num_envs, num_joints)),  # Not used by Standing Waist Height Policy
        "floating_base_pose": base_pose_w,  # wrt world frame, used to project gravity vector to local frame
        "floating_base_vel": base_vel_b,  # root_link_*_vel_b (link frame), used by Homie WBC
        "floating_base_acc": np.zeros((num_envs, 6)),  # Not used by Standing Waist Height Policy
        "projected_gravity_b": projected_gravity_b,  # gravity projected into base frame, used by AGILE recurrent
        "root_ang_vel_b": root_ang_vel_b,  # COM-frame angular velocity, used by AGILE recurrent
        "torso_quat": torso_link_quat_w_wxyz.cpu().numpy(),
        "torso_ang_vel": torso_link_ang_vel_b.cpu().numpy(),
    }
    return wbc_obs


def postprocess_actions(
    wbc_action: dict[str, np.ndarray],
    robot_data: ArticulationData,
    wbc_g1_joints_order: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """Postprocess actions for the policy."""
    num_envs = wbc_action["q"].shape[0]
    num_joints = len(robot_data.joint_names)
    processed_actions = torch.zeros((num_envs, num_joints), device=device)
    wbc_joints_pos_action = torch.from_numpy(wbc_action["q"])
    # Convert wbc gr00t joints order to Lab joints order
    for wbc_joint_name, wbc_joint_index in wbc_g1_joints_order.items():
        if wbc_joint_name not in robot_data.joint_names:
            print(f"Joint {wbc_joint_name} not found in asset")
            continue
        sim_joint_index = robot_data.joint_names.index(wbc_joint_name)
        processed_actions[:, sim_joint_index] = wbc_joints_pos_action[:, wbc_joint_index]
    # Hand joints that are not in the WBC joint order (Revo 2 distal joints) follow their proximal joint.
    # Revo 2 distal joints are not coupled in PhysX, so each one is driven to follow where its proximal joint actually
    # is (not where it is commanded to go). A proximal joint that is blocked by the object then does not get pushed
    # back past its limit by its own distal drive.
    measured = wp.to_torch(robot_data.joint_pos).to(device)
    for distal_index, proximal_index, multiplier in distal_hand_joint_couplings(robot_data.joint_names):
        processed_actions[:, distal_index] = multiplier * measured[:, proximal_index].clamp(min=0.0)
    return processed_actions
