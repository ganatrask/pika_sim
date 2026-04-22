# Pika Sim

MuJoCo simulation environments for robotic gripper pick-and-place tasks. Includes two gripper models — the [Agilex Pika](https://github.com/agilexrobotics) and the [Trossen WXAI](https://www.trossenrobotics.com/) — each with a 3-DOF Cartesian gantry, multi-camera rendering, sensor logging, and dataset recording for imitation learning.

## Requirements

- Python 3.10+
- MuJoCo >= 3.0

```bash
pip install mujoco numpy opencv-python h5py
```

## Simulations

### Pika Gripper (`pika_gripper_mujoco_sim/`)

Agilex Pika parallel-jaw gripper and dual-gripper sensor head.

| | |
|---|---|
| **Models** | Standalone gripper, pick-and-place environment, sensor head |
| **Gripper stroke** | 0–50 mm |
| **Cameras** | 5 (fisheye, RealSense D405, wrist cam, overhead, front) |
| **Meshes** | 16 STL files (gripper: 3, sensor head: 13) |
| **Dataset** | [ganatrask/pika_sim](https://huggingface.co/datasets/ganatrask/pika_sim) |

```bash
cd pika_gripper_mujoco_sim
python3 run_sim.py                        # Sensor head (default)
python3 run_sim.py --model gripper        # Standalone gripper
python3 pick_and_place.py --loop          # Randomized pick-and-place
python3 record_dataset.py --episodes 100  # Record dataset
python3 camera_viewer.py --depth          # Multi-camera view
```

### Trossen Gripper (`trossen_gripper_mujoco_sim/`)

Trossen WXAI parallel-jaw gripper with RealSense D405 camera mount.

| | |
|---|---|
| **Models** | Standalone gripper, pick-and-place environment |
| **Gripper stroke** | 0–44 mm (~88 mm total gap) |
| **Cameras** | 5 (RealSense D405, stereo left/right, overhead, front) |
| **Meshes** | 6 STL files |
| **Dataset** | [ganatrask/trossen_sim](https://huggingface.co/datasets/ganatrask/trossen_sim) |

```bash
cd trossen_gripper_mujoco_sim
python3 run_sim.py                        # Standalone gripper (default)
python3 run_sim.py --model pickplace      # Pick-and-place scene
python3 pick_and_place.py --loop          # Randomized pick-and-place
python3 record_dataset.py --episodes 100  # Record dataset
python3 camera_viewer.py --depth          # Multi-camera view
```

## Comparison

| | Pika | Trossen |
|---|---|---|
| Gripper type | Parallel jaw | Parallel jaw |
| Stroke | 50 mm | 44 mm (88 mm gap) |
| Gripper kp | 500 | 100 |
| Gantry kp | 5000 | 5000 |
| Sim timestep | 1 ms (pickplace), 2 ms (standalone) | 1 ms (pickplace), 2 ms (standalone) |
| Cube | 2 cm, 30 g | 2 cm, 30 g |
| Workspace | X: +/-0.20 m, Y: +/-0.10 m | X: +/-0.20 m, Y: +/-0.10 m |
| Sensor head model | Yes (16 mimic joints) | No |

## Dataset Format

Both simulations produce identical HDF5 dataset structures via `record_dataset.py`:

```
dataset/
├── episodes/
│   ├── episode_0000.h5
│   └── ...
└── videos/
    ├── wrist_cam_0000.mp4
    └── ...
```

Each episode contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `actions/ee_pose` | (T, 7) | Target end-effector pose [x, y, z, qw, qx, qy, qz] |
| `actions/gripper` | (T, 1) | Gripper command (0.0 = open, 1.0 = closed) |
| `observations/ee_pose` | (T, 7) | Actual end-effector pose |
| `observations/gripper_width` | (T, 1) | Gripper opening in meters |
| `observations/timestamp` | (T,) | Seconds since episode start |
| `videos/wrist_cam` | (T,) | Frame indices into companion MP4 |
| `env_state` | scalar | JSON metadata (cube poses, placement error) |
| `success` | scalar | True if placement error < 20 mm |

## Project Structure

```
sim/
├── pika_gripper_mujoco_sim/
│   ├── run_sim.py                  # Interactive viewer
│   ├── pick_and_place.py           # Pick-and-place demo
│   ├── record_dataset.py           # Dataset recorder
│   ├── camera_viewer.py            # Multi-camera rendering
│   ├── camera_tuner.py             # Camera parameter tuning
│   ├── inspect_episode.py          # Episode analysis
│   ├── pika_gripper.xml            # Standalone gripper
│   ├── pika_gripper_pickplace.xml  # Pick-and-place environment
│   ├── pika_sensor.xml             # Sensor head (dual gripper)
│   └── meshes/                     # 16 STL files
├── trossen_gripper_mujoco_sim/
│   ├── run_sim.py                  # Interactive viewer
│   ├── pick_and_place.py           # Pick-and-place demo
│   ├── record_dataset.py           # Dataset recorder
│   ├── camera_viewer.py            # Multi-camera rendering
│   ├── trossen_gripper.xml         # Standalone gripper
│   ├── trossen_gripper_pickplace.xml # Pick-and-place environment
│   └── meshes/                     # 6 STL files
└── CLAUDE.md
```
