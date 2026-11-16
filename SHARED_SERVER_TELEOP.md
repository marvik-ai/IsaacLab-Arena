# Sim teleop on a shared, headless server

Workarounds for running the GR00T E2E course lesson
[Sim Teleop and WBC](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/sim-teleop-and-wbc.html)
on a remote GPU server that:

- is reached over SSH only (no display),
- runs other users' Isaac Sim livestream containers with `--net=host`,
- has GPUs shared with (and sometimes saturated by) those containers.

The XR client is the IWER emulator in desktop Chrome (or a Quest 3 browser), connecting over Tailscale.

## Summary

| Problem | Symptom | Workaround |
|---|---|---|
| No `DISPLAY` | `run_docker.sh` fails on `xhost` | Script change: guard `xhost` with a `DISPLAY` check |
| No display for Kit GUI / XR tab | `--viz kit` can't open a window | Drop `--viz kit`; XR runs headless and auto-starts |
| Port clash with other livestreams | 49100/tcp and 47998/udp already bound on host | Run Arena on a bridge network and remap ports (CloudXR, and optionally the WebRTC livestream) |
| WebRTC advertises container IP | Client: "sent WebRTC connectivity checks but received no response" | Set Media Address/Port in the client |
| Saturated GPU | Hang in `physx_manager.play()` after `Failed to get a valid attached USD stage id ...` | Pin the container to an idle GPU |
| Multi-GPU rendering | No video; `VK_ERROR_OUT_OF_DEVICE_MEMORY` in `createVulkanSharedTexture` | Disable multi-GPU rendering, pin CUDA |
| SSH drops | CloudXR exits; teleop gets `XR_ERROR_RUNTIME_UNAVAILABLE` | Run shells in `tmux` |

## 1. Headless operation

- **Script change:** upstream `docker/run_docker.sh` always runs `xhost +local:docker`, which fails over
  SSH without a display. It is now guarded:
  ```bash
  # before
  xhost +local:docker > /dev/null
  # after
  [ -n "$DISPLAY" ] && xhost +local:docker > /dev/null || true
  ```
- Launch `teleop.py` **without** `--viz kit`. With `--teleop_device openxr` and no visualizer, Isaac Lab
  uses `isaaclab.python.xr.openxr.headless.kit` and starts the XR session itself, so the Kit **XR tab**
  is not needed.
- Not available headless: the Kit *XR > Advanced Settings > Render Resolution* UI. For monitoring,
  use the WebRTC livestream described below.

## 2. Port isolation (bridge network)

CloudXR uses fixed ports, 49100/tcp (signaling) and 47998/udp (media). These are the same as Isaac Sim's
WebRTC livestream defaults, which other containers on the host already hold.

`run_docker.sh` has an opt-in bridge mode. Set both variables to replace `--net=host` with a bridge
network and two published ports:

| Env var | Host port (example) | Container port |
|---|---|---|
| `CXR_BRIDGE_PROXY_PORT` | `50500/tcp` | `48322` (WSS proxy, which forwards signaling to 49100 internally) |
| `CXR_BRIDGE_MEDIA_PORT` | `50501/udp` | `47998` (WebRTC media) |

Port 49100 does not need to be published, because the browser only talks to the WSS proxy.

Docker's published ports bypass `ufw`, so no extra firewall rules were needed.

### Client settings (Isaac Teleop web client)

Inside the bridge network, WebRTC advertises the container's internal address, which the client can't
reach. Override it in the client:

- **Connectivity Settings**: Server IP `<host tailscale IP>`, Port `50500`
- **Networking → Media Server Configuration**: Media Address `<host tailscale IP>`, Media Port `50501`
- Accept the self-signed cert at `https://<host IP>:50500/`. A new container generates a new cert, so
  accept it again after each restart.

## 3. GPU pinning

Symptoms we hit:

- The app hung forever in `physx_manager.play()` while PhysX ran on a GPU that another user's job kept
  at 100% utilization.
- With both GPUs visible, Kit enabled multi-GPU rendering. The XR shared texture for CloudXR then failed
  with `VK_ERROR_OUT_OF_DEVICE_MEMORY`, even though plenty of memory was free, and the client got no frames.

Fixes:

- `ARENA_GPUS=<idx>` in `run_docker.sh` replaces `--gpus=all` and sets `CUDA_VISIBLE_DEVICES`.
- Also export `CUDA_VISIBLE_DEVICES=<idx>` in **both** shells (CloudXR and teleop) and pass
  `--kit_args "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0"` to `teleop.py`.

Check `nvidia-smi` before starting and pick the least-utilized GPU.

## 4. Session robustness

- Order matters:
  1. Start CloudXR.
  2. Run `source ~/.cloudxr/run/cloudxr.env` in the teleop shell.
  3. Launch teleop.

  If you skip step 2, `XR_RUNTIME_JSON` falls back to the container default `/cloudxr/...` and teleop
  fails with `XR_ERROR_RUNTIME_UNAVAILABLE`. Check it with `echo $XR_RUNTIME_JSON`, which should point
  under `~/.cloudxr/`.
- If you kill teleop with `kill -9`, restart CloudXR as well.
- Run both container shells inside `tmux` so an SSH drop doesn't stop CloudXR.

## Full launch sequence

```bash
# Terminal 1 (host, inside tmux)
ARENA_GPUS=0 CXR_BRIDGE_PROXY_PORT=50500 CXR_BRIDGE_MEDIA_PORT=50501 ./docker/run_docker.sh -d ../datasets -m ../models -e ../eval
export CUDA_VISIBLE_DEVICES=0
python -m isaacteleop.cloudxr

# Terminal 2 (host, inside tmux), which attaches to the same container
ARENA_GPUS=0 CXR_BRIDGE_PROXY_PORT=50500 CXR_BRIDGE_MEDIA_PORT=50501 ./docker/run_docker.sh -d ../datasets -m ../models -e ../eval
export CUDA_VISIBLE_DEVICES=0
source ~/.cloudxr/run/cloudxr.env
python isaaclab_arena/scripts/imitation_learning/teleop.py \
  --device cpu \
  --kit_args "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0" \
  galileo_g1_static_pick_and_place \
  --object apple_01_objaverse_robolab \
  --destination clay_plates_hot3d_robolab \
  --teleop_device openxr 2>&1 | tee /tmp/teleop_run.log
```

Then, in the client: fill in the settings from section 2, click **CONNECT**, then **Play**.

## Optional: Isaac Sim WebRTC livestream

To watch the sim from the Isaac Sim WebRTC Streaming Client, publish two more ports. This works
alongside CloudXR: `teleop.py --teleop_device openxr` can stream to IWER and to the WebRTC client at
the same time, which gives you a monitoring view on the Mac. Both sides use the
same port number, because WebRTC advertises the port it listens on:

```bash
ARENA_GPUS=0 CXR_BRIDGE_PROXY_PORT=50500 CXR_BRIDGE_MEDIA_PORT=50501 \
  LIVESTREAM_SIGNAL_PORT=50510 LIVESTREAM_STREAM_PORT=50511 ./docker/run_docker.sh
```

Launch with `--livestream 2`, not `1`. Mode `1` hard-codes ports 49100/47998. Pass the ports in
`--kit_args`:

```bash
--livestream 2 --kit_args "--/exts/omni.kit.livestream.app/primaryStream/publicIp=<host IP> \
--/exts/omni.kit.livestream.app/primaryStream/signalPort=50510 \
--/exts/omni.kit.livestream.app/primaryStream/streamPort=50511 \
--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0"
```

Add these to the `teleop.py` command, in place of its `--kit_args`. In the WebRTC client, connect to
`<host IP>` on port `50510`. Wait until the scene has finished loading before you connect each client
(WebRTC and IWER). Connecting too early fails.

## Known limitations

- **IWER on a Mac:** about 8 FPS streaming while the server renders at 60 FPS. The network was a direct
  LAN path, so the likely bottleneck is the browser decoding the 4096×4032 packed stereo stream. Lowering
  the resolution in IWER's settings raised streaming to about 20–30 FPS. That's still below the render
  rate, but usable for trying the workflow. The client's lowest bitrate is 25 Mbps, and CloudXR has no
  server-side bitrate setting. The undocumented `NV_CXR_DOWNSAMPLE_AMOUNT=2` (set before
  `python -m isaacteleop.cloudxr`) gave no noticeable improvement.
- **IWER controls:** controllers are posed relative to the head, and the drag gizmo is hard to use.
  Typing Pos values (metres) and Rot values (radians) in the controller panels works better.
- **Head camera (`RobotHeadCam`) for the second viewport:** use `record_demos.py`, which works with
  `--device cpu --enable_cameras` and the livestream. `teleop.py` strips cameras in XR mode, so with
  `--enable_cameras` it fails with "scene entity 'robot_head_cam' does not exist". Patching that
  out only got as far as the camera's first image: with `--device cpu` it hangs, and with
  `--device cuda:0` it crashes with CUDA illegal memory access.
- **Benign warning:** `Failed to create XR anchor prim: ... '_get_backend_utils'`.
- **Fallback dataset:** for real data collection, use a headset, or skip recording and use the course's
  pre-collected dataset `nvidia/Arena-G1-Static-PickNPlace-Task`.
