# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab_arena_g1.g1_env.mdp.usd_instancing import deinstance_matching_prims

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_decoupled_wbc_joint_policy(env: ManagerBasedEnv, env_ids: torch.Tensor):
    # Reset lower body RL-based policy
    policy = env.action_manager.get_term("g1_action").get_wbc_policy
    policy.lower_body_policy.reset(env_ids)


def reset_decoupled_wbc_pink_policy(env: ManagerBasedEnv, env_ids: torch.Tensor):
    # Reset upper body IK solver
    env.action_manager.get_term("g1_action").upperbody_controller.body_ik_solver.initialize()
    env.action_manager.get_term("g1_action").upperbody_controller.in_warmup = True

    # Reset lower body RL-based policy
    policy = env.action_manager.get_term("g1_action").get_wbc_policy
    policy.lower_body_policy.reset(env_ids)

    # Reset P-controller
    env.action_manager.get_term("g1_action")._is_navigating = False
    env.action_manager.get_term("g1_action")._navigation_goal_reached = False
    env.action_manager.get_term("g1_action")._num_navigation_subgoals_reached = -1


def apply_high_friction_to_g1_fingers(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    material_path: str,
    static_friction: float,
    dynamic_friction: float,
    prim_name_markers: Sequence[str],
) -> None:
    """Bind a high-friction contact material to G1 hand/finger collision prims."""
    del env_ids
    from isaaclab.sim import bind_physics_material
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
    from pxr import UsdPhysics, UsdShade

    stage = env.sim.stage
    material_cfg = RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        friction_combine_mode="max",
    )
    material_cfg.func(material_path, material_cfg)

    bound_count = 0
    for env_prim_path in env.scene.env_prim_paths:
        robot_prim_path = f"{env_prim_path}/Robot"
        # Instanced links (e.g. the Revo 2 USD) hide their collision prims in instance proxies, which the
        # traversal below skips and where no material can be bound. No-op for assets that are not instanced.
        deinstance_matching_prims(stage, robot_prim_path, prim_name_markers)
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if not prim_path.startswith(robot_prim_path):
                continue
            prim_path_lower = prim_path.lower()
            if not any(marker in prim_path_lower for marker in prim_name_markers):
                continue
            applied_schemas = set(prim.GetAppliedSchemas())
            has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI) or "PhysicsCollisionAPI" in applied_schemas
            if not has_collision_api:
                continue

            bind_physics_material(prim_path, material_path, stage=stage)
            # bind_physics_material is decorated with apply_nested and returns None, so
            # verify the resulting direct physics material binding explicitly.
            material_binding_api = UsdShade.MaterialBindingAPI(prim)
            direct_binding = material_binding_api.GetDirectBinding("physics")
            if str(direct_binding.GetMaterialPath()) == material_path:
                bound_count += 1

    if bound_count == 0:
        warnings.warn(
            "apply_high_friction_to_g1_fingers: no G1 hand/finger collision prims were found; "
            "G1 hand friction may be unchanged.",
            stacklevel=1,
        )


def filter_revo2_thumb_palm_collisions(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Keep the Revo 2 thumbs from colliding with anything except the fingers of the same hand.

    With robot self-collisions on, the thumb sweeps into the palm and wrist links and stops. Filtering those pairs
    leaves the thumb-to-finger contact, which is the one that matters (the thumb must not pass through the index
    finger).
    """
    del env_ids
    import os

    from pxr import Usd, UsdPhysics

    finger_markers = ("_index_", "_middle_", "_ring_", "_pinky_")
    if os.environ.get("ARENA_G1_REVO2_THUMB_FINGER_COLLISIONS", "1") == "0":
        finger_markers = ()  # filter the thumb against the fingers too
    filtered_count = 0
    for env_prim_path in env.scene.env_prim_paths:
        robot_prim_path = f"{env_prim_path}/Robot"
        robot_prim = env.sim.stage.GetPrimAtPath(robot_prim_path)
        bodies = [
            prim
            for prim in Usd.PrimRange(robot_prim)
            if not prim.IsInstanceProxy() and "PhysicsRigidBodyAPI" in set(prim.GetAppliedSchemas())
        ]
        for side in ("left", "right"):
            thumbs = [p for p in bodies if f"{side}_thumb_" in p.GetName().lower()]
            # Same-hand finger links stay collidable with the thumb.
            keep_paths = {
                p.GetPath()
                for p in bodies
                if p.GetName().lower().startswith(side) and any(m in p.GetName().lower() for m in finger_markers)
            }
            for thumb in thumbs:
                targets = [p.GetPath() for p in bodies if p != thumb and p.GetPath() not in keep_paths]
                if not targets:
                    continue
                filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(thumb).CreateFilteredPairsRel()
                for target in targets:
                    filtered_pairs.AddTarget(target)
                filtered_count += len(targets)

    if filtered_count == 0:
        warnings.warn(
            "filter_revo2_thumb_palm_collisions: no thumb rigid bodies were found; thumb self-collisions unchanged.",
            stacklevel=1,
        )
