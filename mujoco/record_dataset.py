#!/usr/bin/env python3
"""
Record pick-and-place episodes to HDF5 dataset.

Dataset structure per episode:
    episode_NNNN.h5
    ├── actions/
    │   ├── ee_pose          (T, 7) float32   # target xyz + quat_wxyz
    │   └── gripper          (T, 1) float32   # 0.0=open, 1.0=close
    ├── observations/
    │   ├── ee_pose          (T, 7) float32   # current xyz + quat_wxyz
    │   ├── gripper_width    (T, 1) float32   # meters, 0.0-0.087
    │   └── timestamp        (T,)   float64   # seconds since episode start
    ├── videos/
    │   └── wrist_cam        (T,)   int64     # frame indices into wrist_cam.mp4
    ├── env_state            ()     object    # JSON: cube pose
    ├── success              ()     bool
    └── episode_length       ()     int64

Usage:
    python record_dataset.py --episodes 100 --out_dir ./dataset
    python record_dataset.py --episodes 10 --seed 42 --speed 1.0
"""

import argparse
import json
import os
import time
import numpy as np

try:
    import mujoco
except ImportError:
    print("Error: mujoco package not found. Install with: pip install mujoco")
    exit(1)

try:
    import h5py
except ImportError:
    print("Error: h5py package not found. Install with: pip install h5py")
    exit(1)

try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: opencv-python not found. Video recording disabled.")
    print("Install with: pip install opencv-python")

# Import from pick_and_place.py
from pick_and_place import (
    generate_waypoints, randomize_cube, SmoothController,
    GRIPPER_OPEN, GRIPPER_CLOSED,
)

# Constants
RECORD_HZ = 50
MAX_GAP = 0.087  # gripper max opening in meters
GRIPPER_QUAT_WXYZ = np.array([0.707107, 0.0, 0.707107, 0.0], dtype=np.float32)
SUCCESS_THRESHOLD = 0.02  # 20mm


class EpisodeRecorder:
    """Records one episode of pick-and-place data."""

    def __init__(self, model, record_hz=RECORD_HZ):
        self.model = model
        self.record_interval = 1.0 / record_hz
        self.last_record_time = -1.0
        self.start_time = None

        # Resolve sensor addresses
        self._sensor_adr = {}
        self._sensor_dim = {}
        for i in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            self._sensor_adr[name] = model.sensor_adr[i]
            self._sensor_dim[name] = model.sensor_dim[i]

        # Camera setup
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "realsense_d405")
        self.img_h = 480
        self.img_w = 640
        self.renderer = mujoco.Renderer(model, height=self.img_h, width=self.img_w)

        # Buffers
        self.actions_ee = []
        self.actions_gripper = []
        self.obs_ee = []
        self.obs_gripper_width = []
        self.obs_timestamp = []
        self.frames = []

    def _read_sensor(self, data, name):
        adr = self._sensor_adr[name]
        dim = self._sensor_dim[name]
        return data.sensordata[adr:adr + dim].copy()

    def _get_gripper_width(self, data):
        pos_grip = self._read_sensor(data, "gripper_sensor")[0]
        width = MAX_GAP + pos_grip * 2  # pos_grip is negative when closing
        return max(0.0, width)

    def _get_ee_pos(self, data):
        """Get end-effector world position from gantry joint sensors."""
        x = self._read_sensor(data, "x_sensor")[0]
        y = self._read_sensor(data, "y_sensor")[0]
        z = self._read_sensor(data, "z_sensor")[0]
        # Convert gantry joint positions to world xyz
        # Gantry base is at (0, 0, 0.75), gripper body offset (0, 0, -0.1)
        world_x = x
        world_y = y
        world_z = 0.75 + z - 0.1  # gantry_z base + slide + gripper offset
        return np.array([world_x, world_y, world_z], dtype=np.float32)

    def _render_frame(self, data):
        if cv2 is None:
            return None
        self.renderer.update_scene(data, camera=self.cam_id)
        return self.renderer.render().copy()  # (H, W, 3) uint8 RGB

    def record_step(self, data, ctrl_xyz, ctrl_grip):
        """Record a single timestep if enough time has passed."""
        if data.time - self.last_record_time < self.record_interval:
            return
        self.last_record_time = data.time

        if self.start_time is None:
            self.start_time = data.time

        # --- Actions ---
        # ee_pose action: target xyz + fixed quaternion
        action_ee = np.concatenate([
            np.array(ctrl_xyz, dtype=np.float32),
            GRIPPER_QUAT_WXYZ,
        ])
        self.actions_ee.append(action_ee)

        # gripper action: 0.0=open, 1.0=close (normalize from ctrl range)
        grip_normalized = np.clip(-ctrl_grip / 0.05, 0.0, 1.0)
        self.actions_gripper.append(np.array([grip_normalized], dtype=np.float32))

        # --- Observations ---
        ee_pos = self._get_ee_pos(data)
        obs_ee = np.concatenate([ee_pos, GRIPPER_QUAT_WXYZ])
        self.obs_ee.append(obs_ee)

        width = self._get_gripper_width(data)
        self.obs_gripper_width.append(np.array([width], dtype=np.float32))

        self.obs_timestamp.append(data.time - self.start_time)

        # --- Video frame ---
        frame = self._render_frame(data)
        if frame is not None:
            self.frames.append(frame)

    def get_episode_length(self):
        return len(self.obs_timestamp)

    def save(self, filepath, video_dir, episode_idx, env_state, success):
        """Save episode to HDF5 + MP4."""
        T = self.get_episode_length()
        if T == 0:
            print("  Warning: empty episode, skipping save")
            return False

        # Save video
        video_path = None
        if self.frames and cv2 is not None:
            video_path = os.path.join(video_dir, f"wrist_cam_{episode_idx:04d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, RECORD_HZ, (self.img_w, self.img_h))
            for frame in self.frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

        # Save HDF5
        with h5py.File(filepath, "w") as f:
            # actions/
            act = f.create_group("actions")
            act.create_dataset("ee_pose", data=np.array(self.actions_ee, dtype=np.float32))
            act.create_dataset("gripper", data=np.array(self.actions_gripper, dtype=np.float32))

            # observations/
            obs = f.create_group("observations")
            obs.create_dataset("ee_pose", data=np.array(self.obs_ee, dtype=np.float32))
            obs.create_dataset("gripper_width", data=np.array(self.obs_gripper_width, dtype=np.float32))
            obs.create_dataset("timestamp", data=np.array(self.obs_timestamp, dtype=np.float64))

            # videos/
            vid = f.create_group("videos")
            frame_indices = np.arange(T, dtype=np.int64)
            vid.create_dataset("wrist_cam", data=frame_indices)
            if video_path:
                vid["wrist_cam"].attrs["video_file"] = os.path.basename(video_path)

            # env_state (JSON)
            f.create_dataset("env_state", data=json.dumps(env_state))

            # success
            f.create_dataset("success", data=success)

            # episode_length
            f.create_dataset("episode_length", data=np.int64(T))

        return True


def run_episode(model, data, rng, speed):
    """Run one pick-and-place episode, return (recorder, env_state, success)."""
    # Randomize and get positions
    pick_x, pick_y, place_x, place_y = randomize_cube(model, data, rng)
    waypoints = generate_waypoints(pick_x, pick_y, place_x, place_y)

    if speed != 1.0:
        waypoints = [(*w[:4], w[4] / speed, w[5]) for w in waypoints]

    controller = SmoothController(waypoints)
    recorder = EpisodeRecorder(model, record_hz=RECORD_HZ)

    # Run simulation
    duration = controller.total_time + 1.0
    while data.time < duration:
        x, y, z, grip = controller.get_control(data)
        data.ctrl[0] = x
        data.ctrl[1] = y
        data.ctrl[2] = z
        data.ctrl[3] = grip

        recorder.record_step(data, [x, y, z], grip)
        mujoco.mj_step(model, data)

    # Check success
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_pos = data.xpos[cube_id].copy()
    error = np.sqrt((cube_pos[0] - place_x)**2 + (cube_pos[1] - place_y)**2)
    success = bool(error < SUCCESS_THRESHOLD)

    # Build env_state
    cube_quat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    qpos_adr = model.jnt_qposadr[cube_quat_id]
    cube_quat = data.qpos[qpos_adr + 3:qpos_adr + 7].tolist()

    env_state = {
        "pick_pos": [float(pick_x), float(pick_y)],
        "place_pos": [float(place_x), float(place_y)],
        "cube_final_pos": cube_pos.tolist(),
        "cube_final_quat_wxyz": cube_quat,
        "placement_error_mm": float(error * 1000),
        "gripper_max_width_m": MAX_GAP,
        "sim_timestep": model.opt.timestep,
        "record_hz": RECORD_HZ,
    }

    return recorder, env_state, success, error


def main():
    parser = argparse.ArgumentParser(description="Record pick-and-place dataset")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--out_dir", type=str, default="./dataset", help="Output directory")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Setup directories
    h5_dir = os.path.join(args.out_dir, "episodes")
    video_dir = os.path.join(args.out_dir, "videos")
    os.makedirs(h5_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "pika_gripper_pickplace.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print(f"Recording {args.episodes} episodes to {args.out_dir}")
    print(f"  Speed: {args.speed}x | Seed: {args.seed} | Record Hz: {RECORD_HZ}")
    print(f"  Video: {'enabled' if cv2 is not None else 'DISABLED (install opencv-python)'}")
    print("-" * 60)

    success_count = 0
    t_start = time.time()

    for ep in range(args.episodes):
        # Reset sim
        mujoco.mj_resetData(model, data)

        recorder, env_state, success, error = run_episode(model, data, rng, args.speed)

        # Save
        h5_path = os.path.join(h5_dir, f"episode_{ep:04d}.h5")
        recorder.save(h5_path, video_dir, ep, env_state, success)

        if success:
            success_count += 1

        T = recorder.get_episode_length()
        status = "OK" if success else "FAIL"
        print(f"  [{ep:4d}/{args.episodes}] T={T:4d} err={error*1000:5.1f}mm {status}"
              f"  pick=({env_state['pick_pos'][0]:+.2f},{env_state['pick_pos'][1]:+.2f})"
              f"  place=({env_state['place_pos'][0]:+.2f},{env_state['place_pos'][1]:+.2f})")

    elapsed = time.time() - t_start
    print("-" * 60)
    print(f"Done: {args.episodes} episodes in {elapsed:.1f}s ({elapsed/args.episodes:.2f}s/ep)")
    print(f"Success: {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
