# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The Revo 2 thumb is filtered against everything except the fingers of the same hand (needs ``pxr``)."""

from types import SimpleNamespace

import pytest

pxr = pytest.importorskip("pxr")

from isaaclab_arena_g1.g1_env.mdp.g1_events import filter_revo2_thumb_palm_collisions  # noqa: E402

_BODIES = [
    "left_base_link",
    "left_thumb_metacarpal_Link",
    "left_thumb_proximal_Link",
    "left_index_proximal_Link",
    "left_pinky_distal_Link",
    "left_wrist_yaw_link",
    "right_thumb_proximal_Link",
    "right_index_proximal_Link",
]


def _filtered(stage, name):
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
    api = UsdPhysics.FilteredPairsAPI(prim)
    return {t.name for t in api.GetFilteredPairsRel().GetTargets()} if api else set()


def test_thumb_is_filtered_against_everything_but_same_hand_fingers():
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for path in ("/World", "/World/envs", "/World/envs/env_0", "/World/envs/env_0/Robot"):
        UsdGeom.Xform.Define(stage, path)
    for name in _BODIES:
        prim = UsdGeom.Xform.Define(stage, f"/World/envs/env_0/Robot/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
    env = SimpleNamespace(scene=SimpleNamespace(env_prim_paths=["/World/envs/env_0"]), sim=SimpleNamespace(stage=stage))

    filter_revo2_thumb_palm_collisions(env, None)

    assert _filtered(stage, "left_thumb_proximal_Link") == {
        "left_base_link",
        "left_thumb_metacarpal_Link",
        "left_wrist_yaw_link",
        "right_thumb_proximal_Link",
        "right_index_proximal_Link",
    }
    # Fingers and the palm themselves get no filter of their own.
    assert _filtered(stage, "left_index_proximal_Link") == set()
    assert "left_index_proximal_Link" not in _filtered(stage, "left_thumb_metacarpal_Link")
