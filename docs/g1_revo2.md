# G1 with BrainCo Revo 2 hands

The `g1_revo2_wbc_agile_pink` (teleoperation, recording) and `g1_revo2_wbc_agile_joint` (replay, GR00T closed loop)
embodiments replace the Dex3 hands with BrainCo Revo 2 hands. The robot asset is not part of this repository.
This page covers getting the asset into the simulator and the first smoke test.

The hand has 6 actuated joints per side (`thumb_metacarpal`, `thumb_proximal`, and `index`, `middle`, `ring`,
`pinky` `_proximal`). The `*_distal` joints are mimic joints of their proximal joint, and the `*_tip` joints are
locked. The whole-body controller therefore sees 41 joints (29 body + 12 hand), while the simulator articulation has
51 (the 10 distal joints are the extra ones).

The USD converter records the mimic relation only for the Newton backend and removes it from the PhysX layer, so
PhysX does not couple the distal joints. The whole-body controller drives them instead, with
`distal_target = multiplier * proximal_position` (1.155 for the fingers, 1.0 for the thumb), and the action terms then
enforce `q_distal = multiplier * q_proximal` on every physics step (`revo2_distal_coupler.py`), so a fingertip pressed
against the apple cannot straighten the distal joint relative to the proximal one. The proximal joint stays in charge.
The coupling is a position override: it does not pass a load on the fingertip to the proximal drive. Set
`ARENA_G1_REVO2_DISTAL_KINEMATIC=0` to turn it off. Record and evaluate with the same setting, since the policy never
sees the distal joints.

## 1. Prepare the asset (host, no container)

The container mounts `~/.cache`, so the asset goes there. Nothing is copied into the repository.

```bash
ASSET_SRC=/home/minsky/projects/g1-apple-rl/assets/g1_with_brainco_hand
ASSET_DST=~/.cache/isaaclab_arena/g1_revo2/asset

mkdir -p "$ASSET_DST"
rsync -a --exclude=configuration --exclude='*.usd' "$ASSET_SRC"/ "$ASSET_DST"/

# Fix the locked tip joints, keep the mimic joints. Standard library only, any python3 works.
python3 isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/utils/revo2_urdf.py sim \
  "$ASSET_DST"/g1_29dof_mode_15_brainco_hand.urdf "$ASSET_DST"/g1_29dof_revo2_sim.urdf
```

Expected output: `Wrote ... with 51 movable joints.`

Why the tip joints are fixed: the upstream URDF limits the left `*_tip` joints to `lower=upper=1` rad and the right
ones to 0, so the left fingertips would be bent by one radian in the simulator.

## 2. Convert URDF to USD (inside the container)

Start the container yourself from the repository root (`./docker/run_docker.sh`). On the shared server, pin it to an
idle GPU and apply the headless fixes described in `docker/SHARED_SERVER_TELEOP.md`. Then:

```bash
/isaac-sim/python.sh submodules/IsaacLab/scripts/tools/convert_urdf.py \
  $HOME/.cache/isaaclab_arena/g1_revo2/asset/g1_29dof_revo2_sim.urdf \
  $HOME/.cache/isaaclab_arena/g1_revo2/usd \
  --headless
```

The second argument is an output **directory**, not a file name. The converter creates
`<dir>/g1_29dof_revo2_sim/g1_29dof_revo2_sim.usda`, which is the file to use, plus a `payloads/` folder next to it
that it needs. Do not move the `.usda` alone.

If the script rejects `--headless`, drop the flag. Leave `--merge-joints` and `--fix-base` off. The joint drive gains
it prints are overridden by the embodiment's actuators.

## 3. Point Arena at the files (inside the container)

```bash
export ARENA_G1_REVO2_USD=$HOME/.cache/isaaclab_arena/g1_revo2/usd/g1_29dof_revo2_sim/g1_29dof_revo2_sim.usda
export ARENA_G1_REVO2_URDF=$HOME/.cache/isaaclab_arena/g1_revo2/asset/g1_29dof_mode_15_brainco_hand.urdf
```

`ARENA_G1_REVO2_URDF` is the original combined URDF. The whole-body controller derives its own Pinocchio copy from it
(mimic and tip joints fixed) under `~/.cache/isaaclab_arena/g1_revo2/`.

Robot self-collisions are off for the Revo 2 embodiments, as for the Dex3 robot. With them on, the robot falls over
when the arms move on the GPU solver (a 35 cm arm swing tilts it by more than 80 degrees, against about 13 degrees with
them off). The thumb and fingers then pass through each other but still collide with objects.
`export ARENA_G1_REVO2_SELF_COLLISIONS=1` turns them on, at your own risk.

With self-collisions on, the thumb is filtered against everything except the fingers of the same hand.
`export ARENA_G1_REVO2_THUMB_FINGER_COLLISIONS=0` filters the thumb against the fingers too (finger-to-finger
collisions stay on), which is the way out if the thumb cannot close past the index finger.

## 4. Smoke test (inside the container)

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action \
  --num_steps 50 \
  galileo_g1_static_pick_and_place \
  --embodiment g1_revo2_wbc_agile_pink
```

There is no display flag on purpose: without `--viz kit` the run is headless, as in the other documented workflows.
Pass if the environment loads and steps without an exception. The arms may move to odd poses because a zero action
is not a valid wrist target; that is fine for this test.

| Symptom | Likely cause |
|---|---|
| `FileNotFoundError ... ARENA_G1_REVO2_USD` or `..._URDF` | Variable unset, or it points at a directory or a path not visible inside the container. It must be the `.usda` **file**. |
| `Joint ... not found in loco_manip_g1_joints_order` | A simulator joint is not in the WBC yaml and is not a passive distal joint. Send me the name. |
| `Joint ... not found in asset` (printed during stepping) | A name in `loco_manip_g1_revo2_joints_order_41dof.yaml` is missing from the USD. |
| Error about actuators matching no joints | The USD has no `*_distal_joint` joints. The current converter output has all 10. |
| Fingertips bend the wrong way or lag | The distal coupling (`revo2_coupling.py`) or the distal gains in `g1_revo2.py` need tuning. |
| Robot falls over or jitters | Hand mass and inertia changed the AGILE balance, or the hand actuator gains need tuning. |
| Warning that no finger collision prims were found | The fingertip friction markers do not match the USD prim names. |

Send me the full log of the run, including the first lines that list the articulation joints.

## 5. Dump the simulator joint order (inside the container)

The GR00T conversion and closed-loop code index recorded states and actions by the simulator's joint order, so the
joint-space yaml has to come from the articulation. This also takes one zero-action step with the **joint** embodiment
(replay and closed loop) and prints the action and observation sizes.

```bash
/isaac-sim/python.sh isaaclab_arena_g1/tools/dump_sim_joint_order.py \
  --output isaaclab_arena_gr00t/embodiments/g1_revo2/51dof_joint_space.yaml \
  galileo_g1_static_pick_and_place \
  --embodiment g1_revo2_wbc_agile_joint
```

Expected: 51 joints listed (29 body, 12 actuated hand joints and 10 distal joints), a total action dim of 58 (51 joint
targets + 7 commands), and the final line `one zero-action step ran without errors`.

## 6. GR00T files for the Revo 2 hands

These mirror the Dex3 files in `isaaclab_arena_gr00t/` and differ only in the hand joints:

| File | Purpose |
|---|---|
| `embodiments/g1_revo2/51dof_joint_space.yaml` | Simulator joint order (51 joints), written by step 5. Width of the recorded `robot_joint_pos` and `processed_actions`. |
| `embodiments/g1_revo2/gr00t_41dof_joint_space.yaml` | Policy joint groups (41 joints, no distal joints). Hand groups have 6 joints. |
| `embodiments/g1_revo2/modality.json` | State and action slices: `left_hand` 22:28, `right_arm` 28:35, `right_hand` 35:41. |
| `embodiments/g1_revo2/info.json` | LeRobot info template (feature shapes are filled in from the data). |
| `lerobot/config/g1_revo2_static_apple_config.yaml` | HDF5 to LeRobot conversion config. |
| `policy/config/g1_revo2_static_apple_gr00t_closedloop_config.yaml` | Closed-loop evaluation config. Set `model_path` to your checkpoint. |

The GR00T data config (`embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py`) only lists group names, not sizes, so
the Revo 2 configs reuse it.

The distal joints are never recorded as policy outputs. Their targets are derived from the proximal ones when the
action is applied, so a policy action has 41 + 7 values and the simulator action has 51 + 7 = 58.
