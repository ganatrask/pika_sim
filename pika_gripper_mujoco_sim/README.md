# Pika Sim

MuJoCo simulation environment for the [Pika](https://github.com/agilexrobotics) gripper and sensor head. Includes a 3-DOF Cartesian pick-and-place demo with randomized object poses, multi-camera rendering, and full sensor logging (IMU, joint encoders, actuator forces).

Dataset: https://huggingface.co/datasets/ganatrask/pika_sim

## Setup

```bash
pip install mujoco numpy opencv-python
```

Tested with Python 3.10+ and MuJoCo >= 3.0.

## Quick Start

```bash
cd sim/mujoco

# Interactive gripper or sensor head viewer
python3 run_sim.py                    # Sensor head (default)
python3 run_sim.py --model gripper    # Parallel-jaw gripper

# Pick-and-place with randomized cube positions
python3 pick_and_place.py             # Single run with viewer
python3 pick_and_place.py --loop      # Continuous loop, re-randomizes each cycle
python3 pick_and_place.py --seed 42   # Reproducible randomization

# Multi-camera viewer (fisheye, RealSense D405, wrist cam, overhead, front)
python3 camera_viewer.py              # All cameras
python3 camera_viewer.py --depth      # With depth maps
python3 camera_viewer.py --record     # Save to MP4
```

## Project Structure

```
sim/
└── mujoco/
    ├── run_sim.py                  # Interactive viewer for gripper / sensor head
    ├── pick_and_place.py           # Randomized pick-and-place demo + sensor logging
    ├── camera_viewer.py            # Multi-camera rendering (OpenCV)
    ├── pika_gripper.xml            # Standalone gripper model
    ├── pika_gripper_pickplace.xml  # Gripper + gantry + table + cube environment
    ├── pika_sensor.xml             # Dual-gripper sensor head (16 mimic joints)
    └── meshes/                     # STL meshes (gripper: 3, sensor head: 13)
```

## Models

### Pika Gripper (`pika_gripper.xml`)

Standalone parallel-jaw gripper for isolated testing. Symmetric finger coupling via equality constraint — single actuator drives both fingers.

- Stroke: 0–50 mm (slide joint)
- Actuator: position control, kp=50
- Timestep: 2 ms

### Pick-and-Place Environment (`pika_gripper_pickplace.xml`)

Gripper mounted on a 3-DOF Cartesian gantry above a table with a 2 cm cube.

| Component | Details |
|-----------|---------|
| Gantry | X: +/-0.25 m, Y: +/-0.15 m, Z: 0–0.4 m |
| Actuators | Position control, kp=5000, force limit +/-50 N |
| Cube | 2 cm, 30 g, free joint |
| Cameras | Fisheye (140 FOV), RealSense D405 (64 FOV), wrist cam (80 FOV), overhead, front |
| Sensors | Joint pos/vel, actuator forces, IMU (accel + gyro + quat), cube pose |
| Timestep | 1 ms |

### Sensor Head (`pika_sensor.xml`)

Pika spatial data collection device with dual grippers driven by a single revolute joint through mechanical coupling (16 mimic joints via equality constraints).

- Center joint range: -0.5 to 1.2 rad
- Actuators: position (kp=50) + velocity (kv=5)
- Timestep: 2 ms

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
- Gripper angle (rad) and distance (mm)
- IMU: accelerometer, gyroscope, orientation quaternion
- Cube world position

## Camera Viewer

Renders all 5 cameras in a grid using OpenCV. Supports depth visualization and video recording.

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `Space` | Pause / resume |
| `1`–`5` | Toggle individual cameras |
| `R` | Reset with new random positions |

| Flag | Description |
|------|-------------|
| `--cameras fisheye wrist_cam` | Show specific cameras only |
| `--depth` | Render depth maps (colormap) |
| `--record` | Save composite to `camera_recording.mp4` |
| `--width N --height N` | Camera resolution (default 640x480) |

## Meshes

16 STL files exported from SolidWorks CAD models (via URDF on the `master` branch of [pika_ros](https://github.com/agilexrobotics/pika_ros)).

| Group | Files | Used by |
|-------|-------|---------|
| Gripper | `gripper_base_link`, `gripper_left_link`, `gripper_right_link` | `pika_gripper.xml`, `pika_gripper_pickplace.xml` |
| Sensor head | `base_link`, `center_link`, `left_link1/2`, `right_link1/2`, `left/right_hand_link`, `left/right_gripper_add_1/2` | `pika_sensor.xml` |
