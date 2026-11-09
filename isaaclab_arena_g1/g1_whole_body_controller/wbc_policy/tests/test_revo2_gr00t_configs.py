# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Consistency of the G1 + Revo 2 GR00T config files with each other and with the WBC joint order."""

import json
import os
import yaml

import pytest

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.revo2_coupling import distal_hand_joint_couplings

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
_EMBODIMENT_DIR = os.path.join(_REPO_ROOT, "isaaclab_arena_gr00t", "embodiments", "g1_revo2")
_JOINT_GROUPS = ["left_leg", "right_leg", "waist", "left_arm", "left_hand", "right_arm", "right_hand"]


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def policy_groups() -> dict[str, list[str]]:
    return _load_yaml(os.path.join(_EMBODIMENT_DIR, "gr00t_41dof_joint_space.yaml"))["joints"]


@pytest.fixture(scope="module")
def sim_joints() -> dict[str, int]:
    return _load_yaml(os.path.join(_EMBODIMENT_DIR, "51dof_joint_space.yaml"))["joints"]


@pytest.fixture(scope="module")
def modality() -> dict:
    with open(os.path.join(_EMBODIMENT_DIR, "modality.json")) as f:
        return json.load(f)


def test_modality_slices_match_the_policy_joint_groups(policy_groups, modality):
    assert list(policy_groups) == _JOINT_GROUPS
    expected_start = 0
    for section in ("state", "action"):
        expected_start = 0
        for group in _JOINT_GROUPS:
            entry = modality[section][group]
            assert entry["start"] == expected_start, f"{section}.{group}"
            assert entry["end"] - entry["start"] == len(policy_groups[group]), f"{section}.{group}"
            expected_start = entry["end"]
    assert expected_start == 41


def test_policy_groups_follow_the_wbc_joint_order(policy_groups):
    wbc_order = _load_yaml(
        os.path.join(_REPO_ROOT, "isaaclab_arena_g1", "g1_env", "config", "loco_manip_g1_revo2_joints_order_41dof.yaml")
    )
    policy_order = [name for group in _JOINT_GROUPS for name in policy_groups[group]]
    assert policy_order == sorted(wbc_order, key=wbc_order.get)


def test_sim_joint_order_covers_the_policy_joints_plus_the_distal_joints(policy_groups, sim_joints):
    assert sorted(sim_joints.values()) == list(range(len(sim_joints))) == list(range(51))
    policy_names = {name for names in policy_groups.values() for name in names}
    assert policy_names <= set(sim_joints)
    extra = set(sim_joints) - policy_names
    assert len(extra) == 10 and all(name.endswith("_distal_joint") for name in extra)
    # Every distal joint is paired with its proximal joint by the WBC coupling rule.
    names_in_sim_order = sorted(sim_joints, key=sim_joints.get)
    assert len(distal_hand_joint_couplings(names_in_sim_order)) == 10


@pytest.mark.parametrize(
    "config_path",
    [
        "isaaclab_arena_gr00t/lerobot/config/g1_revo2_static_apple_config.yaml",
        "isaaclab_arena_gr00t/policy/config/g1_revo2_static_apple_gr00t_closedloop_config.yaml",
    ],
)
def test_revo2_configs_point_at_existing_revo2_files(config_path):
    config = _load_yaml(os.path.join(_REPO_ROOT, config_path))
    path_keys = [key for key in config if key.endswith(("_path", "_template_path")) and key != "data_path"]
    assert path_keys
    for key in path_keys:
        value = config[key]
        if not value.startswith("isaaclab_arena_gr00t/"):
            continue
        assert os.path.isfile(os.path.join(_REPO_ROOT, value)), f"{key}: {value}"
        if "joint_space" in value or "modality.json" in value or "info.json" in value:
            assert "/g1_revo2/" in value and "43dof" not in value, f"{key}: {value}"
