#!/usr/bin/env python3
"""
Shared utilities for policy evaluation in MuJoCo simulation.

Provides:
- PikaSimEnv: MuJoCo environment wrapper for closed-loop eval
- HDF5 episode loading (pika sim format)
- Observation/action builders matching training conventions
- Metrics computation (open-loop RMSE, closed-loop placement error)
- Gripper configs for pika and trossen
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:
    raise ImportError("mujoco package not found. Install with: pip install mujoco")

try:
    import h5py
except ImportError:
    raise ImportError("h5py package not found. Install with: pip install h5py")

try:
    import cv2
except ImportError:
    cv2 = None

from robot_inference_client import get_client, InferenceObservation, InferenceAction

# Re-export for convenience
from pick_and_place import randomize_cube, read_cube_xy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GANTRY_BASE_Z = 0.75
GRIPPER_OFFSET_Z = 0.1
GRIPPER_QUAT_WXYZ = np.array([0.707107, 0.0, 0.707107, 0.0], dtype=np.float32)
SUCCESS_THRESHOLD = 0.02  # 20mm

CAMERA_MAP_MUJOCO = {
    "wrist": "realsense_d405",
    "overhead": "overhead",
    "front": "front",
}
CAMERA_MAP_CANONICAL = {
    "wrist": "wrist_cam",
    "overhead": "overhead",
    "front": "front",
}

GRIPPER_CONFIGS = {
    "pika": {
        "xml": "pika_gripper_pickplace.xml",
        "GRIPPER_OPEN": 0.0,
        "GRIPPER_CLOSED": -0.05,
        "MAX_GAP": 0.087,
        "QUAT_WXYZ": GRIPPER_QUAT_WXYZ,
        "gripper_base_gap": 0.087,  # width = MAX_GAP + pos_grip * 2 (record_dataset.py:101)
        "action_grip_to_ctrl": lambda g: -g * 0.05,       # [0,1] -> [-0.05, 0]
        "ctrl_to_norm": lambda c: np.clip(-c / 0.05, 0, 1),
    },
    "trossen": {
        "xml": "trossen_gripper_pickplace.xml",
        "GRIPPER_OPEN": 0.044,
        "GRIPPER_CLOSED": 0.0,
        "MAX_GAP": 0.088,
        "QUAT_WXYZ": GRIPPER_QUAT_WXYZ,
        "gripper_base_gap": 0.004,  # width = base_gap + pos_grip * 2
        "action_grip_to_ctrl": lambda g: (1 - g) * 0.044,  # [0,1] -> [0.044, 0]
        "ctrl_to_norm": lambda c: np.clip(1.0 - c / 0.044, 0, 1),
    },
}


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def ctrl_z_to_world_z(ctrl_z: float) -> float:
    """Gantry slide Z -> world Z."""
    return GANTRY_BASE_Z + ctrl_z - GRIPPER_OFFSET_Z


def world_z_to_ctrl_z(world_z: float) -> float:
    """World Z -> gantry slide Z."""
    return world_z - GANTRY_BASE_Z + GRIPPER_OFFSET_Z


# ---------------------------------------------------------------------------
# Sensor reading helpers
# ---------------------------------------------------------------------------

def _resolve_sensor_addrs(model):
    """Build sensor name -> (address, dim) mapping."""
    addrs = {}
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        addrs[name] = (model.sensor_adr[i], model.sensor_dim[i])
    return addrs


def _read_sensor(data, sensor_addrs, name):
    adr, dim = sensor_addrs[name]
    return data.sensordata[adr:adr + dim].copy()


def _get_ee_pos(data, sensor_addrs) -> np.ndarray:
    """Read x/y/z sensors -> world frame (3,)."""
    x = _read_sensor(data, sensor_addrs, "x_sensor")[0]
    y = _read_sensor(data, sensor_addrs, "y_sensor")[0]
    z = _read_sensor(data, sensor_addrs, "z_sensor")[0]
    world_z = ctrl_z_to_world_z(z)
    return np.array([x, y, world_z], dtype=np.float32)


def _get_gripper_width(data, sensor_addrs, base_gap: float, max_gap: float) -> float:
    """Read gripper sensor -> width in meters."""
    pos_grip = _read_sensor(data, sensor_addrs, "gripper_sensor")[0]
    width = base_gap + pos_grip * 2
    return float(np.clip(width, 0.0, max_gap))


# ---------------------------------------------------------------------------
# qpos / action / observation builders
# ---------------------------------------------------------------------------

def build_qpos(ee_pose_7: np.ndarray, gripper_width: float, max_gap: float) -> np.ndarray:
    """Concat ee_pose (7) + normalized gripper -> (8,). grip_norm: 1=open, 0=closed."""
    grip_norm = gripper_width / max_gap
    return np.concatenate([ee_pose_7, [grip_norm]]).astype(np.float32)


def build_action(ee_pose_7: np.ndarray, gripper_01: float) -> np.ndarray:
    """Concat action ee_pose (7) + gripper -> (8,). gripper: 0=open, 1=closed."""
    return np.concatenate([ee_pose_7, [gripper_01]]).astype(np.float32)


def build_observation(qpos_8d: np.ndarray, images_dict: dict, prompt: str = "") -> InferenceObservation:
    """Build InferenceObservation for the server."""
    return InferenceObservation(
        qpos=qpos_8d,
        images=images_dict,
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# HDF5 episode loading
# ---------------------------------------------------------------------------

def load_episode(h5_path: Path) -> dict:
    """Load pika/trossen sim HDF5 episode. Returns dict with numpy arrays."""
    with h5py.File(h5_path, "r") as f:
        return {
            "actions_ee": f["actions/ee_pose"][()],
            "actions_grip": f["actions/gripper"][()],
            "obs_ee": f["observations/ee_pose"][()],
            "obs_grip_width": f["observations/gripper_width"][()],
            "obs_timestamp": f["observations/timestamp"][()],
            "env_state": json.loads(f["env_state"][()]),
            "success": bool(f["success"][()]),
            "episode_length": int(f["episode_length"][()]),
            "video_frames": f["videos/wrist_cam"][()],
        }


def episode_gt_actions(ep: dict) -> np.ndarray:
    """Ground truth actions as (T, 8)."""
    return np.concatenate([ep["actions_ee"], ep["actions_grip"]], axis=1)


def episode_gt_qpos(ep: dict, max_gap: float) -> np.ndarray:
    """Ground truth qpos as (T, 8). grip_norm: 1=open."""
    grip_norm = ep["obs_grip_width"] / max_gap  # (T, 1)
    return np.concatenate([ep["obs_ee"], grip_norm], axis=1)


def load_episode_video(dataset_dir, episode_idx: int) -> np.ndarray:
    """Load MP4 video frames -> (T, H, W, 3) uint8 RGB."""
    if cv2 is None:
        raise ImportError("opencv-python required for video loading")
    video_path = Path(dataset_dir) / "videos" / f"wrist_cam_{episode_idx:04d}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames, dtype=np.uint8)


def list_episodes(dataset_dir: Path) -> list[Path]:
    """Sorted glob of episode_*.h5 files."""
    return sorted((Path(dataset_dir) / "episodes").glob("episode_*.h5"))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class OpenLoopMetrics:
    rmse_xyz: float
    rmse_orientation: float
    rmse_overall: float
    gripper_accuracy: float
    per_joint_rmse: np.ndarray


@dataclass
class ClosedLoopMetrics:
    success: bool
    placement_error_mm: float
    pick_pos: tuple[float, float]
    place_pos: tuple[float, float]


def compute_action_rmse(preds: np.ndarray, gts: np.ndarray) -> OpenLoopMetrics:
    """Compare (T, 8) predicted vs ground truth actions."""
    per_joint = np.sqrt(np.mean((preds - gts) ** 2, axis=0))
    return OpenLoopMetrics(
        rmse_xyz=float(np.sqrt(np.mean((preds[:, :3] - gts[:, :3]) ** 2))),
        rmse_orientation=float(np.sqrt(np.mean((preds[:, 3:7] - gts[:, 3:7]) ** 2))),
        rmse_overall=float(np.sqrt(np.mean((preds - gts) ** 2))),
        gripper_accuracy=float(np.mean((preds[:, 7] > 0.5) == (gts[:, 7] > 0.5))),
        per_joint_rmse=per_joint,
    )


def compute_placement_metrics(cube_xy, target_xy, pick_pos) -> ClosedLoopMetrics:
    error_m = float(np.linalg.norm(np.array(cube_xy) - np.array(target_xy)))
    return ClosedLoopMetrics(
        success=error_m < SUCCESS_THRESHOLD,
        placement_error_mm=error_m * 1000,
        pick_pos=tuple(pick_pos),
        place_pos=tuple(target_xy),
    )


def compute_success_rates(results: list[ClosedLoopMetrics]) -> dict:
    """Multi-threshold success rates."""
    errors_mm = np.array([r.placement_error_mm for r in results])
    return {
        "success_rate_5mm": float(np.mean(errors_mm < 5)),
        "success_rate_20mm": float(np.mean(errors_mm < 20)),
        "success_rate_50mm": float(np.mean(errors_mm < 50)),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_open_loop_summary(results: list[OpenLoopMetrics]):
    """Formatted table of per-episode + aggregate metrics."""
    print("\n" + "=" * 70)
    print("OPEN-LOOP EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n{'Ep':>4s}  {'RMSE_xyz':>9s}  {'RMSE_ori':>9s}  {'RMSE_all':>9s}  {'Grip_acc':>9s}")
    print("-" * 50)
    for i, m in enumerate(results):
        print(f"{i:4d}  {m.rmse_xyz:9.5f}  {m.rmse_orientation:9.5f}  "
              f"{m.rmse_overall:9.5f}  {m.gripper_accuracy:9.3f}")

    # Aggregated
    n = len(results)
    mean_xyz = np.mean([m.rmse_xyz for m in results])
    mean_ori = np.mean([m.rmse_orientation for m in results])
    mean_all = np.mean([m.rmse_overall for m in results])
    mean_grip = np.mean([m.gripper_accuracy for m in results])
    print("-" * 50)
    print(f"{'AVG':>4s}  {mean_xyz:9.5f}  {mean_ori:9.5f}  "
          f"{mean_all:9.5f}  {mean_grip:9.3f}")
    print(f"\nEpisodes: {n}")
    print("Gripper accuracy: binary match at threshold 0.5 "
          "(1=closed in action space)")


def print_closed_loop_summary(results: list[ClosedLoopMetrics]):
    """Success rates at 5/20/50mm, mean/std placement error."""
    print("\n" + "=" * 70)
    print("CLOSED-LOOP EVALUATION RESULTS")
    print("=" * 70)

    errors = [r.placement_error_mm for r in results]
    n = len(results)
    rates = compute_success_rates(results)

    print(f"\nEpisodes: {n}")
    print(f"Success rate (<5mm):  {rates['success_rate_5mm']:.1%}")
    print(f"Success rate (<20mm): {rates['success_rate_20mm']:.1%}")
    print(f"Success rate (<50mm): {rates['success_rate_50mm']:.1%}")
    print(f"Mean placement error: {np.mean(errors):.1f} mm  "
          f"(std: {np.std(errors):.1f} mm)")

    print(f"\n{'Ep':>4s}  {'Error_mm':>9s}  {'Success':>7s}  "
          f"{'Pick':>18s}  {'Place':>18s}")
    print("-" * 65)
    for i, r in enumerate(results):
        ok = "OK" if r.success else "FAIL"
        print(f"{i:4d}  {r.placement_error_mm:9.1f}  {ok:>7s}  "
              f"({r.pick_pos[0]:+.3f},{r.pick_pos[1]:+.3f})  "
              f"({r.place_pos[0]:+.3f},{r.place_pos[1]:+.3f})")


def results_to_json_open_loop(results: list[OpenLoopMetrics], backend: str,
                              gripper: str, prompt: str, **extra) -> dict:
    """JSON-serializable output for open-loop eval."""
    from datetime import datetime
    n = len(results)
    return {
        "eval_type": "open_loop",
        "backend": backend,
        "gripper": gripper,
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "num_episodes": n,
        **extra,
        "aggregated": {
            "mean_rmse_xyz": float(np.mean([m.rmse_xyz for m in results])),
            "mean_rmse_orientation": float(np.mean([m.rmse_orientation for m in results])),
            "mean_rmse_overall": float(np.mean([m.rmse_overall for m in results])),
            "mean_gripper_accuracy": float(np.mean([m.gripper_accuracy for m in results])),
        },
        "per_episode": [
            {
                "episode_idx": i,
                "rmse_xyz": float(m.rmse_xyz),
                "rmse_orientation": float(m.rmse_orientation),
                "rmse_overall": float(m.rmse_overall),
                "gripper_accuracy": float(m.gripper_accuracy),
                "per_joint_rmse": m.per_joint_rmse.tolist(),
            }
            for i, m in enumerate(results)
        ],
    }


def results_to_json_closed_loop(results: list[ClosedLoopMetrics], backend: str,
                                gripper: str, prompt: str, seed: int,
                                horizon: int, **extra) -> dict:
    """JSON-serializable output for closed-loop eval."""
    from datetime import datetime
    errors_mm = [r.placement_error_mm for r in results]
    rates = compute_success_rates(results)
    return {
        "eval_type": "closed_loop",
        "backend": backend,
        "gripper": gripper,
        "prompt": prompt,
        "seed": seed,
        "horizon": horizon,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "num_episodes": len(results),
        **extra,
        "aggregated": {
            **rates,
            "mean_placement_error_mm": float(np.mean(errors_mm)),
            "std_placement_error_mm": float(np.std(errors_mm)),
        },
        "per_episode": [
            {
                "episode_idx": i,
                "success": r.success,
                "placement_error_mm": r.placement_error_mm,
                "pick_pos": list(r.pick_pos),
                "place_pos": list(r.place_pos),
            }
            for i, r in enumerate(results)
        ],
    }


# ---------------------------------------------------------------------------
# Video saving utility
# ---------------------------------------------------------------------------

def save_video(frames: list[np.ndarray], path: str, fps: int = 20):
    """Save list of RGB frames to MP4."""
    if cv2 is None:
        raise ImportError("opencv-python required for video saving")
    if not frames:
        return
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


# ---------------------------------------------------------------------------
# PikaSimEnv — MuJoCo environment for closed-loop evaluation
# ---------------------------------------------------------------------------

class PikaSimEnv:
    """MuJoCo environment for closed-loop policy evaluation."""

    def __init__(self, gripper: str = "pika", cameras: tuple = ("wrist",),
                 img_w: int = 640, img_h: int = 480):
        cfg = GRIPPER_CONFIGS[gripper]
        self.cfg = cfg
        self.max_gap = cfg["MAX_GAP"]
        self.base_gap = cfg["gripper_base_gap"]
        self.action_grip_to_ctrl = cfg["action_grip_to_ctrl"]

        # Load model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(script_dir, cfg["xml"])
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Renderer
        self.renderer = mujoco.Renderer(self.model, img_h, img_w)

        # Sensor addresses
        self._sensor_addrs = _resolve_sensor_addrs(self.model)

        # Camera setup
        self.cameras = list(cameras)
        self.camera_map_mujoco = CAMERA_MAP_MUJOCO
        self.camera_map_canonical = CAMERA_MAP_CANONICAL

    def reset(self, rng) -> tuple[float, float, float, float]:
        """Reset sim, randomize cube. Returns (pick_x, pick_y, place_x, place_y)."""
        mujoco.mj_resetData(self.model, self.data)
        return randomize_cube(self.model, self.data, rng)

    def get_qpos(self) -> np.ndarray:
        """Read sensors -> 8D qpos [x,y,z, qw,qx,qy,qz, grip_norm].
        grip_norm: 1=open, 0=closed (matches training obs convention).
        """
        ee_pos = _get_ee_pos(self.data, self._sensor_addrs)
        ee_pose = np.concatenate([ee_pos, GRIPPER_QUAT_WXYZ])
        width = _get_gripper_width(self.data, self._sensor_addrs,
                                   self.base_gap, self.max_gap)
        grip_norm = width / self.max_gap
        return np.concatenate([ee_pose, [grip_norm]]).astype(np.float32)

    def render_cameras(self) -> dict[str, np.ndarray]:
        """Render configured cameras. Returns {canonical_name: (H,W,3) uint8}.
        No flip — matches record_dataset.py training data.
        """
        images = {}
        for short_name in self.cameras:
            mujoco_name = self.camera_map_mujoco[short_name]
            canonical = self.camera_map_canonical[short_name]
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, mujoco_name)
            self.renderer.update_scene(self.data, camera=cam_id)
            images[canonical] = self.renderer.render().copy()
        return images

    def apply_action(self, action_8d: np.ndarray):
        """Map 8D action to sim ctrl, step sim 50x (20Hz policy / 1kHz sim).
        action_8d: [x, y, z, qw, qx, qy, qz, grip]
          - x, y passthrough
          - z: world -> gantry slide
          - grip: 0=open, 1=closed -> ctrl value
          - quat (3:7) ignored (constant, no rotation actuator)
        """
        self.data.ctrl[0] = action_8d[0]
        self.data.ctrl[1] = action_8d[1]
        self.data.ctrl[2] = world_z_to_ctrl_z(action_8d[2])
        self.data.ctrl[3] = self.action_grip_to_ctrl(action_8d[7])
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

    def compute_placement_error(self, place_x: float, place_y: float) -> float:
        """Cube-to-target XY distance in meters."""
        cube_x, cube_y = read_cube_xy(self.model, self.data)
        return float(np.linalg.norm(
            np.array([cube_x, cube_y]) - np.array([place_x, place_y])
        ))

    def close(self):
        self.renderer.close()
