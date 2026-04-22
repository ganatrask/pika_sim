#!/usr/bin/env python3
"""
Pika Gripper Pick-and-Place Demo (Dynamic / Randomized)

Reads the cube position from the simulation at runtime and generates
waypoints dynamically. Cube spawn and place target are randomized
within the reachable table area on each cycle.

Usage:
    python pick_and_place.py              # Run with viewer
    python pick_and_place.py --headless   # Run without viewer
    python pick_and_place.py --loop       # Loop continuously (re-randomizes each cycle)

Requirements:
    pip install mujoco numpy
"""

import argparse
import os
import time
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("Error: mujoco package not found.")
    print("Install with: pip install mujoco")
    exit(1)


# ============================================================
# Smooth trajectory generation
# ============================================================

def smoothstep(t):
    """Hermite smoothstep: smooth ease-in/ease-out [0,1] -> [0,1]"""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smoother_step(t):
    """Ken Perlin's smoother step: even smoother acceleration"""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a, b, t):
    """Linear interpolation with smooth t"""
    return a + (b - a) * smoother_step(t)


# ============================================================
# Constants
# ============================================================

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = -0.05

GRASP_Z = -0.30           # Slide Z for pads at cube height
HOVER_Z = -0.20            # Well above table for transit
HOME_Z = -0.05

# Reachable workspace on the table (gantry limits with margin)
WORKSPACE_X = (-0.20, 0.20)
WORKSPACE_Y = (-0.10, 0.10)
MIN_PICK_PLACE_DIST = 0.08  # Minimum distance between pick and place


def generate_waypoints(pick_x, pick_y, place_x, place_y):
    """Generate smooth pick-and-place waypoints for given pick/place locations."""
    return [
        # Start: home position, gripper open
        (0.0, 0.0, HOME_Z, GRIPPER_OPEN, 0.3, "Start at home"),

        # Approach: move above pick location
        (pick_x, pick_y, HOVER_Z, GRIPPER_OPEN, 0.8, "Above pick location"),

        # Descend to cube
        (pick_x, pick_y, GRASP_Z + 0.02, GRIPPER_OPEN, 0.5, "Descend near cube"),
        (pick_x, pick_y, GRASP_Z, GRIPPER_OPEN, 0.4, "At grasp height"),

        # Grasp — gripper snaps closed (step function, not interpolated)
        (pick_x, pick_y, GRASP_Z, GRIPPER_CLOSED, 0.4, "Grasp"),

        # Brief pause — settling the grip
        (pick_x, pick_y, GRASP_Z, GRIPPER_CLOSED, 0.2, "Grip settled"),

        # Lift
        (pick_x, pick_y, HOVER_Z, GRIPPER_CLOSED, 0.6, "Lifted to transit height"),

        # Transit
        (place_x, place_y, HOVER_Z, GRIPPER_CLOSED, 0.8, "Above place location"),

        # Descend to place
        (place_x, place_y, GRASP_Z + 0.03, GRIPPER_CLOSED, 0.5, "Descend to place"),
        (place_x, place_y, GRASP_Z, GRIPPER_CLOSED, 0.3, "At place height"),

        # Release
        (place_x, place_y, GRASP_Z, GRIPPER_OPEN, 0.4, "Release"),

        # Retract
        (place_x, place_y, HOVER_Z, GRIPPER_OPEN, 0.5, "Retracted up"),

        # Return home
        (0.0, 0.0, HOME_Z, GRIPPER_OPEN, 0.8, "Home"),
    ]


def random_table_pos(rng):
    """Random XY position within reachable table workspace."""
    x = rng.uniform(*WORKSPACE_X)
    y = rng.uniform(*WORKSPACE_Y)
    return x, y


def randomize_cube(model, data, rng):
    """Randomize cube position on table. Returns (pick_x, pick_y, place_x, place_y)."""
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    qpos_adr = model.jnt_qposadr[cube_jnt_id]

    # Random pick position
    pick_x, pick_y = random_table_pos(rng)

    # Random place position (ensure minimum distance from pick)
    for _ in range(50):
        place_x, place_y = random_table_pos(rng)
        dist = np.sqrt((place_x - pick_x)**2 + (place_y - pick_y)**2)
        if dist >= MIN_PICK_PLACE_DIST:
            break

    # Set cube position (free joint qpos: x, y, z, qw, qx, qy, qz)
    table_top_z = 0.21  # table pos.z(0.2) + half-height(0.01)
    cube_z = table_top_z + 0.01  # half cube size
    data.qpos[qpos_adr:qpos_adr + 3] = [pick_x, pick_y, cube_z]
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]  # identity quaternion
    data.qvel[model.jnt_dofadr[cube_jnt_id]:model.jnt_dofadr[cube_jnt_id] + 6] = 0

    # Update target zone visual to show place location
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    model.body_pos[target_body_id] = [place_x, place_y, table_top_z + 0.001]

    mujoco.mj_forward(model, data)
    return pick_x, pick_y, place_x, place_y


def read_cube_xy(model, data):
    """Read current cube XY position from simulation."""
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    pos = data.xpos[cube_body_id]
    return pos[0], pos[1]


class SensorLogger:
    """Logs IMU, joint positions, velocities, and efforts each step."""

    def __init__(self, model, log_rate_hz=50):
        self.model = model
        self.log_interval = 1.0 / log_rate_hz
        self.last_log_time = -1.0

        # Resolve sensor addresses once
        self._sensor_adr = {}
        self._sensor_dim = {}
        for i in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            self._sensor_adr[name] = model.sensor_adr[i]
            self._sensor_dim[name] = model.sensor_dim[i]

    def _read(self, data, name):
        adr = self._sensor_adr[name]
        dim = self._sensor_dim[name]
        return data.sensordata[adr:adr + dim].copy()

    def log(self, data):
        if data.time - self.last_log_time < self.log_interval:
            return
        self.last_log_time = data.time

        # Joint positions
        pos_x = self._read(data, "x_sensor")[0]
        pos_y = self._read(data, "y_sensor")[0]
        pos_z = self._read(data, "z_sensor")[0]
        pos_grip = self._read(data, "gripper_sensor")[0]

        # Joint velocities
        vel_x = self._read(data, "x_vel_sensor")[0]
        vel_y = self._read(data, "y_vel_sensor")[0]
        vel_z = self._read(data, "z_vel_sensor")[0]
        vel_grip = self._read(data, "gripper_vel_sensor")[0]

        # Actuator forces (effort)
        frc_x = self._read(data, "x_force_sensor")[0]
        frc_y = self._read(data, "y_force_sensor")[0]
        frc_z = self._read(data, "z_force_sensor")[0]
        frc_grip = self._read(data, "gripper_force_sensor")[0]

        # IMU
        accel = self._read(data, "imu_accel")
        gyro = self._read(data, "imu_gyro")
        orientation = self._read(data, "imu_orientation")

        # Gripper: convert slide (m) to angle (rad) and distance (m) like real hardware
        # Sim joint: 0 = fully open, -0.05 = fully closed
        # Real gripper: distance = finger gap (0.098m open, 0m closed), angle (0 open, 1.67 closed)
        # Left finger body is at Y=+0.08851, pad at Y_offset=-0.045 → pad Y = 0.04351
        # Right finger body at Y=-0.088529, pad at Y_offset=+0.045 → pad Y = -0.04353
        # Rest gap between pads = 0.04351 + 0.04353 = 0.08704m (~87mm)
        max_gap = 0.087  # finger pad gap when joint=0 (open)
        grip_distance = max_gap + pos_grip * 2  # pos_grip is negative when closing, *2 for symmetric
        grip_distance = max(0.0, grip_distance)
        grip_angle = (1.0 - grip_distance / max_gap) * 1.67  # 0 open → 1.67 closed

        # Cube position
        cube_pos = self._read(data, "cube_pos")

        print(
            f"  t={data.time:6.3f}"
            f"  | joints: x={pos_x:+.4f} y={pos_y:+.4f} z={pos_z:+.4f}"
            f"  | gripper: ang={grip_angle:.3f}rad dist={grip_distance*1000:.1f}mm"
            f" vel={vel_grip:+.4f}m/s frc={frc_grip:+.1f}N"
            f"  | IMU: acc=[{accel[0]:+.2f},{accel[1]:+.2f},{accel[2]:+.2f}]"
            f" gyr=[{gyro[0]:+.3f},{gyro[1]:+.3f},{gyro[2]:+.3f}]"
            f" quat=[{orientation[0]:.3f},{orientation[1]:+.3f},{orientation[2]:+.3f},{orientation[3]:+.3f}]"
            f"  | cube: [{cube_pos[0]:+.3f},{cube_pos[1]:+.3f},{cube_pos[2]:.3f}]"
        )


class SmoothController:
    """Smooth trajectory controller with interpolation between waypoints."""

    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.waypoint_idx = 0
        self.segment_start_time = 0.0
        self.started = False
        self.done = False

        # Precompute cumulative times
        self.segment_times = [w[4] for w in waypoints]
        self.total_time = sum(self.segment_times)

        # Previous state for first waypoint
        self.prev_state = np.array([
            waypoints[0][0], waypoints[0][1],
            waypoints[0][2], waypoints[0][3]
        ])

    def get_control(self, data):
        """Returns smoothly interpolated (x, y, z, gripper) control."""
        if self.done:
            w = self.waypoints[-1]
            return w[0], w[1], w[2], w[3]

        if not self.started:
            self.started = True
            self.segment_start_time = data.time
            print(f"\n  [{self.waypoints[0][5]}]")

        # Current waypoint target
        curr_wp = self.waypoints[self.waypoint_idx]
        target = np.array([curr_wp[0], curr_wp[1], curr_wp[2], curr_wp[3]])
        duration = curr_wp[4]

        # Progress through current segment
        elapsed = data.time - self.segment_start_time
        t = elapsed / duration if duration > 0 else 1.0

        if t >= 1.0:
            # Advance to next waypoint
            self.prev_state = target.copy()
            self.waypoint_idx += 1

            if self.waypoint_idx >= len(self.waypoints):
                self.done = True
                print(f"\n  Pick and place complete!")
                return target[0], target[1], target[2], target[3]

            self.segment_start_time = data.time
            next_wp = self.waypoints[self.waypoint_idx]
            print(f"  [{next_wp[5]}]")
            target = np.array([next_wp[0], next_wp[1], next_wp[2], next_wp[3]])
            t = 0.0

        # Smooth interpolation for xyz, step function for gripper
        smooth_t = smoother_step(t)
        xyz = self.prev_state[:3] + (target[:3] - self.prev_state[:3]) * smooth_t
        grip = target[3]  # gripper snaps to target immediately
        return xyz[0], xyz[1], xyz[2], grip


def new_cycle(model, data, rng, speed):
    """Randomize cube, read its position, generate waypoints, return controller."""
    pick_x, pick_y, place_x, place_y = randomize_cube(model, data, rng)
    waypoints = generate_waypoints(pick_x, pick_y, place_x, place_y)

    # Apply speed multiplier
    if speed != 1.0:
        waypoints = [(*w[:4], w[4] / speed, w[5]) for w in waypoints]

    print(f"\n  Cube at: ({pick_x:+.3f}, {pick_y:+.3f})")
    print(f"  Place target: ({place_x:+.3f}, {place_y:+.3f})")

    return SmoothController(waypoints), pick_x, pick_y, place_x, place_y


def main():
    parser = argparse.ArgumentParser(description="Pika Gripper Pick and Place (Randomized)")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--loop", action="store_true", help="Loop continuously with new random positions")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier (0.5=slow, 2.0=fast)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "pika_gripper_pickplace.xml")
    print(f"Loading: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    sensor_logger = SensorLogger(model, log_rate_hz=50)
    controller, pick_x, pick_y, place_x, place_y = new_cycle(model, data, rng, args.speed)

    print("\n=== Pika Gripper Pick-and-Place (Randomized) ===")
    print(f"  Speed: {args.speed:.1f}x | Total time: {controller.total_time:.1f}s")
    print("=" * 52)
    print("  Logging: joints(pos/vel/frc) + gripper(angle/dist/vel/frc) + IMU(accel/gyro/quat) + cube_pos")
    print("-" * 52)

    def step(model, data):
        nonlocal controller
        x, y, z, grip = controller.get_control(data)
        data.ctrl[0] = x
        data.ctrl[1] = y
        data.ctrl[2] = z
        data.ctrl[3] = grip
        sensor_logger.log(data)

    if args.headless:
        duration = controller.total_time + 2.0
        while data.time < duration:
            step(model, data)
            mujoco.mj_step(model, data)

        cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_pos = data.xpos[cube_id]
        print(f"\n  Final cube: ({cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f})")
        print(f"  Target:     ({place_x:.4f}, {place_y:.4f})")
        error = np.sqrt((cube_pos[0] - place_x)**2 + (cube_pos[1] - place_y)**2)
        print(f"  Error: {error*1000:.1f} mm")
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -25
            viewer.cam.distance = 0.7
            viewer.cam.lookat[:] = [0.0, 0.0, 0.25]

            while viewer.is_running():
                step(model, data)
                mujoco.mj_step(model, data)
                viewer.sync()

                if controller.done and args.loop:
                    time.sleep(1.0)
                    mujoco.mj_resetData(model, data)
                    controller, pick_x, pick_y, place_x, place_y = new_cycle(
                        model, data, rng, args.speed
                    )


if __name__ == "__main__":
    main()
