# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for USD assets whose links are instanceable (for example the Revo 2 USD from the URDF converter)."""

from collections.abc import Sequence

# Instances can nest (an instanceable material inside an instanceable link), so repeat until nothing is left.
_MAX_PASSES = 5


def deinstance_matching_prims(stage, root_path: str, name_markers: Sequence[str]) -> int:
    """Turn off instancing on instance roots under ``root_path`` whose path contains one of ``name_markers``.

    Prims inside an instance are instance proxies: ``stage.Traverse()`` skips them, and USD does not allow authoring
    a material binding on them. Turning instancing off makes their children regular prims.

    Args:
        stage: The USD stage.
        root_path: Only prims under this path are considered.
        name_markers: Lower-case fragments to look for in the lower-cased prim path.

    Returns:
        The number of prims that were de-instanced.
    """
    total = 0
    for _ in range(_MAX_PASSES):
        instance_roots = [
            prim
            for prim in stage.Traverse()
            if prim.IsInstance()
            and str(prim.GetPath()).startswith(root_path)
            and any(marker in str(prim.GetPath()).lower() for marker in name_markers)
        ]
        if not instance_roots:
            break
        for prim in instance_roots:
            prim.SetInstanceable(False)
        total += len(instance_roots)
    return total
