#!/usr/bin/env python3
"""
Pika Gripper Camera Viewer

Renders all on-board and scene cameras during the pick-and-place demo.
Shows the fisheye, RealSense D405, and scene cameras in separate OpenCV windows.

Usage:
    python camera_viewer.py                  # All cameras
    python camera_viewer.py --cameras fisheye realsense_d405   # Specific cameras
    python camera_viewer.py --depth          # Include depth rendering
    python camera_viewer.py --speed 0.5      # Slow motion
    python camera_viewer.py --record         # Save frames to video

Requirements:
    pip install mujoco numpy opencv-python
"""

import argparse
import os
import time
import numpy as np

try:
    import mujoco
except ImportError:
    print("Error: mujoco not found. Install with: pip install mujoco")
    exit(1)

try:
    import cv2
except ImportError:
    print("Error: opencv not found. Install with: pip install opencv-python")
    exit(1)

# Import the smooth controller from pick_and_place
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pick_and_place import SmoothController, smoother_step, lerp, generate_waypoints, randomize_cube


def render_camera(model, data, renderer, camera_name, width=640, height=480, depth=False):
    """Render a named camera and return the RGB image (and optionally depth)."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        return None, None

    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render()

    depth_img = None
    if depth:
        renderer.enable_depth_rendering(True)
        renderer.update_scene(data, camera=cam_id)
        raw_depth = renderer.render()
        renderer.enable_depth_rendering(False)
        # Normalize depth for visualization
        depth_img = raw_depth.copy()
        valid = depth_img < 10.0  # ignore far plane
        if valid.any():
            dmin, dmax = depth_img[valid].min(), depth_img[valid].max()
            if dmax > dmin:
                depth_img = np.clip((depth_img - dmin) / (dmax - dmin), 0, 1)
            else:
                depth_img = np.zeros_like(depth_img)
        depth_img = (depth_img * 255).astype(np.uint8)
        depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_TURBO)

    return rgb, depth_img


def add_label(img, text, pos=(10, 25), scale=0.6, color=(255, 255, 255)):
    """Add text label to image."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
    return img


def main():
    parser = argparse.ArgumentParser(description="Pika Gripper Camera Viewer")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Camera names to show (default: all)")
    parser.add_argument("--depth", action="store_true",
                        help="Show depth maps for all cameras")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (0.5=slow, 2.0=fast)")
    parser.add_argument("--record", action="store_true",
                        help="Record composite view to video file")
    parser.add_argument("--width", type=int, default=640,
                        help="Camera render width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Camera render height (default: 480)")
    args = parser.parse_args()

    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "pika_gripper_pickplace.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    rng = np.random.default_rng()

    # Discover cameras
    all_cameras = []
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name:
            all_cameras.append(name)

    cameras = args.cameras if args.cameras else all_cameras
    cameras = [c for c in cameras if c in all_cameras]

    if not cameras:
        print(f"No valid cameras found. Available: {all_cameras}")
        return

    print(f"\n=== Pika Camera Viewer ===")
    print(f"  Cameras: {', '.join(cameras)}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Depth: {'ON' if args.depth else 'OFF'}")
    print(f"  Speed: {args.speed:.1f}x")
    print(f"  Controls: [Q] quit | [SPACE] pause | [1-{len(cameras)}] toggle camera")
    print(f"{'=' * 30}\n")

    # Create renderer
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    # Video writer
    writer = None
    if args.record:
        cols = min(len(cameras), 3)
        rows = (len(cameras) + cols - 1) // cols
        total_w = cols * args.width
        total_h = rows * args.height
        if args.depth:
            total_w *= 2
        outfile = os.path.join(script_dir, "camera_recording.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(outfile, fourcc, 30, (total_w, total_h))
        print(f"  Recording to: {outfile}")

    pick_x, pick_y, place_x, place_y = randomize_cube(model, data, rng)
    waypoints = generate_waypoints(pick_x, pick_y, place_x, place_y)
    if args.speed != 1.0:
        waypoints = [(*w[:4], w[4] / args.speed, w[5]) for w in waypoints]
    controller = SmoothController(waypoints)
    print(f"  Cube at: ({pick_x:+.3f}, {pick_y:+.3f}) -> Place: ({place_x:+.3f}, {place_y:+.3f})")
    paused = False
    camera_visible = {c: True for c in cameras}
    render_every = 3  # render every N sim steps (30fps at 0.001 timestep = every 33 steps)
    step_count = 0

    try:
        while True:
            # Step simulation
            if not paused:
                x, y, z, grip = controller.get_control(data)
                data.ctrl[0] = x
                data.ctrl[1] = y
                data.ctrl[2] = z
                data.ctrl[3] = grip
                mujoco.mj_step(model, data)
                step_count += 1

            # Render cameras at ~30fps
            if step_count % render_every != 0 and not paused:
                continue

            # Collect frames
            frames = []
            for cam_name in cameras:
                if not camera_visible[cam_name]:
                    # Show blank with name
                    blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    add_label(blank, f"{cam_name} [hidden]", color=(100, 100, 100))
                    frames.append(blank)
                    continue

                rgb, depth_map = render_camera(
                    model, data, renderer, cam_name,
                    args.width, args.height, args.depth
                )

                if rgb is not None:
                    # Convert RGB to BGR for OpenCV
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    # Flip vertically (MuJoCo renders upside down)
                    bgr = cv2.flip(bgr, 0)

                    # Add camera name label
                    label = cam_name
                    if cam_name == "fisheye":
                        label = "Fisheye (170° FOV, 640x480)"
                    elif cam_name == "realsense_d405":
                        label = "RealSense D405 (64° FOV, 640x480)"
                    elif cam_name == "overhead":
                        label = "Overhead"
                    elif cam_name == "front":
                        label = "Front View"
                    elif cam_name == "wrist_cam":
                        label = "Wrist Camera"
                    add_label(bgr, label)

                    # Add timestamp
                    add_label(bgr, f"t={data.time:.2f}s", pos=(args.width - 140, 25),
                              scale=0.5, color=(200, 200, 200))

                    if args.depth and depth_map is not None:
                        depth_bgr = cv2.flip(depth_map, 0)
                        add_label(depth_bgr, f"{cam_name} DEPTH")
                        bgr = np.hstack([bgr, depth_bgr])

                    frames.append(bgr)
                else:
                    blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    add_label(blank, f"{cam_name} [not found]")
                    frames.append(blank)

            # Arrange in grid
            if frames:
                cols = min(len(frames), 3)
                rows = (len(frames) + cols - 1) // cols

                # Pad to fill grid
                frame_h, frame_w = frames[0].shape[:2]
                while len(frames) < rows * cols:
                    frames.append(np.zeros((frame_h, frame_w, 3), dtype=np.uint8))

                grid_rows = []
                for r in range(rows):
                    row_frames = frames[r * cols:(r + 1) * cols]
                    # Ensure all same size
                    row_frames = [cv2.resize(f, (frame_w, frame_h)) for f in row_frames]
                    grid_rows.append(np.hstack(row_frames))

                composite = np.vstack(grid_rows)

                # Scale down if too large for screen
                max_screen_w = 1920
                max_screen_h = 1080
                scale = min(max_screen_w / composite.shape[1],
                            max_screen_h / composite.shape[0], 1.0)
                if scale < 1.0:
                    new_w = int(composite.shape[1] * scale)
                    new_h = int(composite.shape[0] * scale)
                    composite = cv2.resize(composite, (new_w, new_h))

                cv2.imshow("Pika Camera Viewer", composite)

                if writer:
                    writer.write(cv2.resize(composite, (
                        cols * args.width * (2 if args.depth else 1),
                        rows * args.height)))

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # Space = pause
                paused = not paused
                print(f"  {'PAUSED' if paused else 'RESUMED'} at t={data.time:.2f}s")
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if idx < len(cameras):
                    cam = cameras[idx]
                    camera_visible[cam] = not camera_visible[cam]
                    print(f"  {cam}: {'visible' if camera_visible[cam] else 'hidden'}")
            elif key == ord('r'):  # Reset
                mujoco.mj_resetData(model, data)
                pick_x, pick_y, place_x, place_y = randomize_cube(model, data, rng)
                waypoints = generate_waypoints(pick_x, pick_y, place_x, place_y)
                if args.speed != 1.0:
                    waypoints = [(*w[:4], w[4] / args.speed, w[5]) for w in waypoints]
                controller = SmoothController(waypoints)
                step_count = 0
                print(f"  RESET -> Cube: ({pick_x:+.3f}, {pick_y:+.3f}) Place: ({place_x:+.3f}, {place_y:+.3f})")

            # Loop if done
            if controller.done:
                time.sleep(1.0)
                mujoco.mj_resetData(model, data)
                pick_x, pick_y, place_x, place_y = randomize_cube(model, data, rng)
                waypoints = generate_waypoints(pick_x, pick_y, place_x, place_y)
                if args.speed != 1.0:
                    waypoints = [(*w[:4], w[4] / args.speed, w[5]) for w in waypoints]
                controller = SmoothController(waypoints)
                step_count = 0

    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.release()
            print(f"\n  Video saved.")
        cv2.destroyAllWindows()
        renderer.close()
        print("  Viewer closed.")


if __name__ == "__main__":
    main()
