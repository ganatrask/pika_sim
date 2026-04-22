---
license: apache-2.0
task_categories:
  - robotics
tags:
  - mujoco
  - simulation
  - pick-and-place
  - imitation-learning
  - gripper
pretty_name: Trossen Gripper Pick-and-Place Simulation Dataset
size_categories:
  - n<1K
---

# Trossen Gripper Pick-and-Place Simulation Dataset

Simulated pick-and-place episodes for imitation learning, generated in MuJoCo with a Trossen parallel-jaw gripper on a 3-DOF Cartesian gantry.

## Dataset Summary

| | |
|---|---|
| **Episodes** | 500 |
| **Steps per episode** | 145 |
| **Record frequency** | 20 Hz |
| **Simulator** | MuJoCo (timestep: 1ms) |
| **Task** | Randomized cube pick-and-place |
| **Success rate** | ~100% (placement error < 20mm) |

## Task Description

A parallel-jaw gripper mounted on a 3-DOF (x, y, z) Cartesian gantry picks up a 2cm cube (30g) from a randomized position on a table and places it at a randomized target location. Each episode follows a 13-point waypoint trajectory with smooth interpolation.

## File Structure

```
episodes/
  episode_0000.h5 ... episode_0499.h5
videos/
  wrist_cam_0000.mp4 ... wrist_cam_0499.mp4
```

## HDF5 Episode Format

Each `episode_NNNN.h5` contains:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `actions/ee_pose` | (145, 7) | float32 | Target end-effector pose [x, y, z, qw, qx, qy, qz] |
| `actions/gripper` | (145, 1) | float32 | Gripper command (0.0 = open, 1.0 = closed) |
| `observations/ee_pose` | (145, 7) | float32 | Actual end-effector pose [x, y, z, qw, qx, qy, qz] |
| `observations/gripper_width` | (145, 1) | float32 | Gripper opening in meters (0 - 0.088m) |
| `observations/timestamp` | (145,) | float64 | Seconds since episode start |
| `videos/wrist_cam` | (145,) | int64 | Frame indices into companion MP4 |
| `env_state` | scalar | JSON | Episode metadata (pick/place positions, placement error) |
| `success` | scalar | bool | True if placement error < 20mm |
| `episode_length` | scalar | int64 | Number of timesteps (145) |

## Conventions

| Convention | Value |
|---|---|
| Action frame | World (absolute) |
| Observation frame | World (absolute) |
| Quaternion format | wxyz (constant, no rotation DOF) |
| Gripper action | 0 = open, 1 = closed |

## Simulation Parameters

| Parameter | Value |
|---|---|
| Gantry workspace | X: ±0.25m, Y: ±0.15m, Z: -0.4m to 0m |
| Gripper stroke | 0 - 88mm |
| Cube size | 2cm, 30g |
| Gantry Kp | 5000 |
| Gripper Kp | 500 |

## Usage

```python
import h5py

with h5py.File("episodes/episode_0000.h5", "r") as ep:
    actions = ep["actions/ee_pose"][:]       # (145, 7)
    gripper = ep["actions/gripper"][:]       # (145, 1)
    obs_pose = ep["observations/ee_pose"][:] # (145, 7)
    success = ep["success"][()]              # bool
```

## Citation

If you use this dataset, please cite this repository.
