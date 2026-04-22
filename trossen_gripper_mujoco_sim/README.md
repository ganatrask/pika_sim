# Trossen Gripper Sim

MuJoCo simulation environment for the [Trossen WXAI](https://www.trossenrobotics.com/) parallel-jaw gripper. Includes a 3-DOF Cartesian pick-and-place demo with randomized object poses, multi-camera rendering, and full sensor logging (IMU, joint encoders, actuator forces).

Dataset: https://huggingface.co/datasets/ganatrask/trossen_sim

## Setup

```bash
pip install mujoco numpy opencv-python
```

Tested with Python 3.10+ and MuJoCo >= 3.0.

## Quick Start

```bash
cd sim/trossen_gripper_mujoco_sim

# Interactive gripper viewer
python3 run_sim.py                        # Standalone gripper (default)
python3 run_sim.py --model pickplace      # Pick-and-place scene
python3 run_sim.py --ctrl 0.022           # Fixed control (half-open)

# Pick-and-place with randomized cube positions
python3 pick_and_place.py                 # Single run with viewer
python3 pick_and_place.py --loop          # Continuous loop, re-randomizes each cycle
python3 pick_and_place.py --seed 42       # Reproducible randomization
python3 pick_and_place.py --headless      # No viewer, print final error

# Record dataset
python3 record_dataset.py --episodes 100  # Record 100 episodes
python3 record_dataset.py --episodes 10 --seed 42 --speed 1.0

# Multi-camera viewer (D405, stereo left/right, overhead, front)
python3 camera_viewer.py                  # All cameras
python3 camera_viewer.py --depth          # With depth maps
python3 camera_viewer.py --record         # Save to MP4
python3 camera_viewer.py --speed 0.5      # Slow motion
```

## Project Structure

```
trossen_gripper_mujoco_sim/
├── run_sim.py                        # Interactive viewer (gripper or pickplace)
├── pick_and_place.py                 # Randomized pick-and-place demo + sensor logging
├── camera_viewer.py                  # Multi-camera rendering (OpenCV)
├── record_dataset.py                 # Record episodes to HDF5 + MP4
├── trossen_gripper.xml               # Standalone gripper model
├── trossen_gripper_pickplace.xml     # Gripper + gantry + table + cube environment
├── meshes/                           # STL meshes (6 files)
└── my_dataset/                       # Recorded episodes (git-ignored)
    ├── episodes/                     # HDF5 files (episode_NNNN.h5)
    └── videos/                       # MP4 wrist camera recordings
```

## Models

### Trossen Gripper (`trossen_gripper.xml`)

Standalone Trossen WXAI parallel-jaw gripper for isolated testing. Symmetric carriage coupling via equality constraint — single actuator drives both fingers.

- Stroke: 0–44 mm (slide joint), ~88 mm total gap
- Actuator: position control, kp=50
- Timestep: 2 ms
- Convention: 0.0 = closed, 0.044 = fully open

### Pick-and-Place Environment (`trossen_gripper_pickplace.xml`)

Gripper mounted on a 3-DOF Cartesian gantry above a table with a 2 cm cube.

| Component | Details |
|-----------|---------|
| Gantry | X: +/-0.25 m, Y: +/-0.15 m, Z: -0.4 to 0 m |
| Actuators | Position control, kp=5000, force limit +/-50 N |
| Gripper | kp=100, kv=10, force limit +/-400 N |
| Cube | 2 cm, 30 g, free joint |
| Cameras | RealSense D405 (87 FOV), stereo left/right (87 FOV), overhead (60 FOV), front (50 FOV) |
| Sensors | Joint pos/vel, actuator forces, IMU (accel + gyro + quat), cube pose |
| Timestep | 1 ms |

## Pick-and-Place Details

Each cycle:
1. Cube is spawned at a random position on the table
2. A place target is chosen (min 8 cm from pick)
3. Waypoints are generated dynamically for the actual cube position
4. Smooth trajectory via Perlin's smoother-step interpolation

The green circle on the table marks the place target.

```
Workspace (reachable table area):
  X: [-0.20, +0.20] m
  Y: [-0.10, +0.10] m
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--loop` | Re-randomize and repeat after each cycle |
| `--headless` | Run without viewer, print final error |
| `--speed N` | Speed multiplier (0.5 = slow, 2.0 = fast) |
| `--seed N` | RNG seed for reproducibility |

### Sensor Output (50 Hz)

Each line logs:
- Joint positions and velocities (X, Y, Z, gripper)
- Actuator forces / efforts
- Gripper joint value (0–0.044) and gap distance (mm)
- IMU: accelerometer, gyroscope, orientation quaternion
- Cube world position

## Camera Viewer

Renders all cameras in a grid using OpenCV. Supports depth visualization and video recording.

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `Space` | Pause / resume |
| `1`–`5` | Toggle individual cameras |
| `R` | Reset with new random positions |

| Flag | Description |
|------|-------------|
| `--cameras realsense_d405 overhead` | Show specific cameras only |
| `--depth` | Render depth maps (colormap) |
| `--record` | Save composite to `camera_recording.mp4` |
| `--width N --height N` | Camera resolution (default 640x480) |

## Meshes

6 STL files for the Trossen WXAI gripper assembly:

| File | Description |
|------|-------------|
| `link_6.stl` | Gripper base / wrist link |
| `carriage_right.stl` | Right carriage slider |
| `carriage_left.stl` | Left carriage slider |
| `gripper_right.stl` | Right jaw |
| `gripper_left.stl` | Left jaw |
| `camera_mount_d405.stl` | RealSense D405 camera mount |
