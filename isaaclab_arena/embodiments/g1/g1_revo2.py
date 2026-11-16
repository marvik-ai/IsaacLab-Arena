# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1 embodiments with BrainCo Revo 2 hands instead of Dex3.

The robot USD is not shipped with Arena. Point ``ARENA_G1_REVO2_USD`` at the converted G1 + Revo 2 USD
(see ``docs/g1_revo2.md``) before constructing one of these embodiments.
"""

import os

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import isaaclab_arena_g1.g1_env.g1_revo2_constants as revo2
from isaaclab_arena.assets.register import register_asset

# The leading-underscore names are module-private in g1.py; reusing them keeps the Revo 2 embodiments identical to
# the Dex3 ones except for the hands.
from isaaclab_arena.embodiments.g1.g1 import (
    _DEFAULT_G1_CAMERA_OFFSET,
    G1_AGILE_CFG,
    G1EmbodimentBase,
    G1SceneCfg,
    G1WBCAgileJointActionCfg,
    G1WBCAgilePinkActionCfg,
    G1WBCJointEventCfg,
    G1WBCJointObservationsCfg,
    G1WBCPinkEventCfg,
    G1WBCPinkObservationsCfg,
    _remove_waist_from_pink_ik_action_config,
)
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_g1.g1_env.mdp import g1_events as g1_events_mdp

REVO2_USD_ENV_VAR = "ARENA_G1_REVO2_USD"
# Set to 1 to turn robot self-collisions on. They are off by default, as for the Dex3 robot: with them on, the GPU
# solver lets the robot fall over when the arms move.
REVO2_SELF_COLLISIONS_ENV_VAR = "ARENA_G1_REVO2_SELF_COLLISIONS"


def _make_g1_revo2_agile_cfg() -> ArticulationCfg:
    """G1 AGILE articulation with the Dex3 hand actuators replaced by Revo 2 ones.

    Hand gains follow the values tuned for these hands in the ``g1-apple-rl`` project: implicit PD with low
    stiffness and effort limits of 0.5 / 1.1 / 2.0 Nm for the thumb metacarpal / thumb proximal / finger proximal
    joints. The distal joints get the same gains as their proximal joint: the USD converter leaves them uncoupled
    in PhysX, so the whole-body controller commands them as ``multiplier * measured proximal position`` (see
    ``revo2_coupling``).
    """
    cfg = G1_AGILE_CFG.copy()
    cfg.actuators["hands"] = ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_thumb_metacarpal_joint",
            ".*_thumb_proximal_joint",
            ".*_(index|middle|ring|pinky)_proximal_joint",
        ],
        effort_limit_sim={
            ".*_thumb_metacarpal_joint": 0.5,
            ".*_thumb_proximal_joint": 1.1,
            ".*_(index|middle|ring|pinky)_proximal_joint": 2.0,
        },
        velocity_limit_sim=2.0,
        stiffness=20.0,
        damping=1.0,
    )
    cfg.actuators["hand_distal"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_thumb_distal_joint", ".*_(index|middle|ring|pinky)_distal_joint"],
        effort_limit_sim={
            ".*_thumb_distal_joint": 1.1,
            ".*_(index|middle|ring|pinky)_distal_joint": 2.0,
        },
        velocity_limit_sim=2.0,
        stiffness=20.0,
        damping=1.0,
    )
    cfg.init_state.joint_pos = {
        **cfg.init_state.joint_pos,
        ".*_thumb_metacarpal_joint": revo2.REVO2_THUMB_OPPOSITION_ANGLE,
    }
    return cfg


G1_REVO2_AGILE_CFG = _make_g1_revo2_agile_cfg()


@configclass
class G1Revo2AgileSceneCfg(G1SceneCfg):
    """G1 scene config with Revo 2 hands and the AGILE actuator gains."""

    robot: ArticulationCfg = G1_REVO2_AGILE_CFG.copy()


def _filter_thumb_palm_collisions(event_config) -> None:
    """Keep the thumbs from catching on the palm while self-collisions are on (see the event for details)."""
    if os.environ.get(REVO2_SELF_COLLISIONS_ENV_VAR, "0") == "0":
        return
    event_config.filter_thumb_palm_collisions = EventTerm(
        func=g1_events_mdp.filter_revo2_thumb_palm_collisions, mode="prestartup"
    )


def _set_revo2_usd_path(scene_config: G1Revo2AgileSceneCfg) -> None:
    usd_path = os.environ.get(REVO2_USD_ENV_VAR)
    if not usd_path or not os.path.isfile(usd_path):
        raise FileNotFoundError(
            f"The G1 + Revo 2 embodiments need the converted robot USD. Set {REVO2_USD_ENV_VAR} to a USD file "
            f"that is visible from where this runs (inside the container if you use Docker); got {usd_path!r}."
        )
    scene_config.robot.spawn.usd_path = usd_path
    # Fingers and thumb must collide with each other. PhysX still skips the pairs of directly jointed links.
    scene_config.robot.spawn.articulation_props.enabled_self_collisions = (
        os.environ.get(REVO2_SELF_COLLISIONS_ENV_VAR, "0") != "0"
    )


@register_asset
class G1Revo2WBCAgilePinkEmbodiment(G1EmbodimentBase):
    """G1 with Revo 2 hands, AGILE WBC and Pink IK upper-body control (teleoperation and data recording)."""

    name = "g1_revo2_wbc_agile_pink"
    link_prim_prefix = "Geometry/"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        camera_offset: Pose | None = _DEFAULT_G1_CAMERA_OFFSET,
        use_tiled_camera: bool = False,
        lock_waist: bool = False,
    ):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config = G1Revo2AgileSceneCfg()
        _set_revo2_usd_path(self.scene_config)
        self.action_config = G1WBCAgilePinkActionCfg()
        self.action_config.g1_action.hand_type = "revo2"
        if lock_waist:
            _remove_waist_from_pink_ik_action_config(self.action_config)
        self.observation_config = G1WBCPinkObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.observation_config.wbc.concatenate_terms = self.concatenate_observation_terms
        self.observation_config.action.concatenate_terms = self.concatenate_observation_terms
        self.event_config = G1WBCPinkEventCfg()
        _filter_thumb_palm_collisions(self.event_config)
        self.camera_config._is_tiled_camera = use_tiled_camera
        self.camera_config._camera_offset = camera_offset


@register_asset
class G1Revo2WBCAgileJointEmbodiment(G1EmbodimentBase):
    """G1 with Revo 2 hands, AGILE WBC and direct joint upper-body control (replay and GR00T closed loop)."""

    name = "g1_revo2_wbc_agile_joint"
    link_prim_prefix = "Geometry/"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        camera_offset: Pose | None = _DEFAULT_G1_CAMERA_OFFSET,
        use_tiled_camera: bool = True,
        lock_waist: bool = False,
    ):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config = G1Revo2AgileSceneCfg()
        _set_revo2_usd_path(self.scene_config)
        self.action_config = G1WBCAgileJointActionCfg()
        self.action_config.g1_action.hand_type = "revo2"
        self.observation_config = G1WBCJointObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.observation_config.wbc.concatenate_terms = self.concatenate_observation_terms
        self.event_config = G1WBCJointEventCfg()
        _filter_thumb_palm_collisions(self.event_config)
        self.camera_config._is_tiled_camera = use_tiled_camera
        self.camera_config._camera_offset = camera_offset
