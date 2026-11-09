# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test that de-instancing exposes collision prims hidden inside instance proxies (needs ``pxr``)."""

import pytest

pxr = pytest.importorskip("pxr")

from isaaclab_arena_g1.g1_env.mdp.usd_instancing import deinstance_matching_prims  # noqa: E402


def _stage_with_instanced_finger():
    """A robot with an instanceable finger link whose collision mesh lives inside the instance."""
    from pxr import Usd, UsdGeom, UsdPhysics

    proto_stage = Usd.Stage.CreateInMemory()
    proto = UsdGeom.Xform.Define(proto_stage, "/Proto").GetPrim()
    mesh = UsdGeom.Mesh.Define(proto_stage, "/Proto/mesh").GetPrim()
    UsdPhysics.CollisionAPI.Apply(mesh)
    proto_stage.SetDefaultPrim(proto)

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Robot")
    UsdGeom.Xform.Define(stage, "/Robot/torso")
    finger = UsdGeom.Xform.Define(stage, "/Robot/left_index_proximal_link").GetPrim()
    finger.GetReferences().AddReference(proto_stage.GetRootLayer().identifier)
    finger.SetInstanceable(True)
    other = UsdGeom.Xform.Define(stage, "/Robot/torso/body").GetPrim()
    other.GetReferences().AddReference(proto_stage.GetRootLayer().identifier)
    other.SetInstanceable(True)
    return stage


def _collision_prims(stage):
    return [p for p in stage.Traverse() if "PhysicsCollisionAPI" in set(p.GetAppliedSchemas())]


def test_collision_prims_in_instances_are_found_after_deinstancing_matching_prims():
    stage = _stage_with_instanced_finger()
    assert _collision_prims(stage) == []  # hidden inside instance proxies

    count = deinstance_matching_prims(stage, "/Robot", ("index",))

    assert count == 1
    assert [str(p.GetPath()) for p in _collision_prims(stage)] == ["/Robot/left_index_proximal_link/mesh"]
    # Instances whose path does not match a marker are left alone.
    assert stage.GetPrimAtPath("/Robot/torso/body").IsInstance()


def test_nothing_to_do_when_no_prim_matches():
    stage = _stage_with_instanced_finger()
    assert deinstance_matching_prims(stage, "/Robot", ("thumb",)) == 0
    assert deinstance_matching_prims(stage, "/Other", ("index",)) == 0
