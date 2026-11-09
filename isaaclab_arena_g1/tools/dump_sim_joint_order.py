# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write the simulator joint order of an Arena environment's robot to a joint-space yaml.

The GR00T conversion and closed-loop code index recorded states and actions by the simulator's joint order, so the
yaml has to come from the articulation itself. The script also takes one zero-action step and prints the action and
observation sizes, as a check that the embodiment is wired consistently.

Usage (inside the container, from the repository root), with the environment sub-command last::

    /isaac-sim/python.sh isaaclab_arena_g1/tools/dump_sim_joint_order.py \
        --output isaaclab_arena_gr00t/embodiments/g1_revo2/51dof_joint_space.yaml \
        galileo_g1_static_pick_and_place --embodiment g1_revo2_wbc_agile_joint
"""

import os
import torch

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

_YAML_HEADER = """# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Simulator joint order (joint name -> column index), written by isaaclab_arena_g1/tools/dump_sim_joint_order.py.
"""


def main():
    args_parser = get_isaaclab_arena_cli_parser()
    args_parser.add_argument("--output", type=str, required=True, help="Path of the joint-space yaml to write.")
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_cli = args_parser.parse_args()

        arena_builder = get_arena_builder_from_cli(args_cli)
        env, _ = arena_builder.make_registered_and_return_cfg()
        env.reset()

        joint_names = list(env.unwrapped.scene["robot"].data.joint_names)
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
        with open(args_cli.output, "w") as f:
            f.write(_YAML_HEADER)
            f.write("joints:\n")
            for index, name in enumerate(joint_names):
                f.write(f'  "{name}": {index}\n')
            f.write(f"\ntotal_joints: {len(joint_names)}\n")
        print(f"[dump_sim_joint_order] {len(joint_names)} joints written to {args_cli.output}")
        for index, name in enumerate(joint_names):
            print(f"[dump_sim_joint_order] {index:3d} {name}")

        action_dim = env.unwrapped.action_manager.total_action_dim
        print(f"[dump_sim_joint_order] total action dim: {action_dim}")
        print(f"[dump_sim_joint_order] observation dims: {env.unwrapped.observation_manager.group_obs_dim}")
        env.step(torch.zeros(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device))
        print("[dump_sim_joint_order] one zero-action step ran without errors")
        env.close()


if __name__ == "__main__":
    main()
