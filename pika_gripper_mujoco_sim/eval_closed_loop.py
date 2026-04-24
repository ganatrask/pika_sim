#!/usr/bin/env python3
"""
Closed-loop policy evaluation: run a trained policy in MuJoCo simulation
with randomized pick-and-place tasks.

The eval script IS the client — it imports robot_inference_client as a library
and uses it to talk to the inference server over ZMQ.

Usage:
    python eval_closed_loop.py \
        --gripper pika \
        --backend act \
        --url tcp://gpu-box:5556 \
        --episodes 50 \
        --seed 42 \
        --cameras wrist \
        --prompt "pick up the cube and place it on the target"
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from robot_inference_client import get_client

from eval_common import (
    GRIPPER_CONFIGS,
    ClosedLoopMetrics,
    PikaSimEnv,
    build_observation,
    print_closed_loop_summary,
    results_to_json_closed_loop,
    save_video,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Closed-loop policy evaluation (MuJoCo sim)")
    parser.add_argument("--gripper", type=str, default="pika",
                        choices=list(GRIPPER_CONFIGS.keys()),
                        help="Gripper type")
    parser.add_argument("--backend", type=str, default="act",
                        help="Inference backend (act, openpi, cosmos)")
    parser.add_argument("--url", type=str, default="tcp://localhost:5556",
                        help="Inference server URL")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of eval episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility")
    parser.add_argument("--horizon", type=int, default=180,
                        help="Steps per episode (180 x 50ms = 9s)")
    parser.add_argument("--cameras", type=str, default="wrist",
                        help="Comma-separated short camera names")
    parser.add_argument("--prompt", type=str, default="",
                        help="Task language instruction")
    parser.add_argument("--save-video", action="store_true",
                        help="Record wrist cam MP4 per episode")
    parser.add_argument("--video-dir", type=str, default="./eval_videos",
                        help="Directory for saved videos")
    parser.add_argument("--output", type=str, default="results_closed_loop.json",
                        help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    cameras = [c.strip() for c in args.cameras.split(",")]

    print(f"Closed-loop eval: {args.episodes} episodes, seed={args.seed}")
    print(f"  Gripper: {args.gripper} | Backend: {args.backend} | URL: {args.url}")
    print(f"  Horizon: {args.horizon} steps | Cameras: {cameras}")
    if args.prompt:
        print(f"  Prompt: {args.prompt!r}")
    print("-" * 60)

    # Setup sim environment
    env = PikaSimEnv(gripper=args.gripper, cameras=tuple(cameras))

    # Connect to inference server
    client_kwargs = {}
    if args.backend == "act":
        client_kwargs["url"] = args.url
    elif args.backend in ("openpi", "openpi_remote"):
        client_kwargs["host"] = args.url.split("://")[-1].split(":")[0]
        port = args.url.split(":")[-1]
        if port.isdigit():
            client_kwargs["port"] = int(port)
    elif args.backend == "cosmos":
        client_kwargs["url"] = args.url

    client = get_client(args.backend, **client_kwargs)
    client.connect()

    # Warmup — discard first inference (GPU JIT, CUDA kernel compile)
    print("Warming up inference server...")
    dummy_qpos = np.zeros(8, dtype=np.float32)
    from eval_common import CAMERA_MAP_CANONICAL
    dummy_images = {
        CAMERA_MAP_CANONICAL[c]: np.zeros((480, 640, 3), dtype=np.uint8)
        for c in cameras
    }
    dummy_obs = build_observation(dummy_qpos, dummy_images)
    warmup_result = client.get_action(dummy_obs)
    assert warmup_result.action.shape == (8,), \
        f"Expected (8,) action, got {warmup_result.action.shape}"
    client.reset()
    print("Warmup done.")

    # Evaluate episodes
    rng = np.random.default_rng(args.seed)
    results = []
    t_start = time.time()

    for ep_idx in range(args.episodes):
        pick_x, pick_y, place_x, place_y = env.reset(rng)
        client.reset()
        video_frames = []

        for step in range(args.horizon):
            qpos = env.get_qpos()
            images = env.render_cameras()
            obs = build_observation(qpos, images, args.prompt)

            result = client.get_action(obs)
            action = result.action
            if action.ndim == 2:
                action = action[0]

            env.apply_action(action)

            if args.save_video:
                wrist_img = images.get("wrist_cam")
                if wrist_img is not None:
                    video_frames.append(wrist_img)

        # Compute placement error
        error_m = env.compute_placement_error(place_x, place_y)
        metrics = ClosedLoopMetrics(
            success=error_m < 0.02,
            placement_error_mm=error_m * 1000,
            pick_pos=(pick_x, pick_y),
            place_pos=(place_x, place_y),
        )
        results.append(metrics)

        status = "OK" if metrics.success else "FAIL"
        print(f"  [{ep_idx:4d}/{args.episodes}] "
              f"err={metrics.placement_error_mm:6.1f}mm {status}  "
              f"pick=({pick_x:+.3f},{pick_y:+.3f})  "
              f"place=({place_x:+.3f},{place_y:+.3f})")

        if args.save_video and video_frames:
            video_path = os.path.join(args.video_dir,
                                      f"episode_{ep_idx:04d}.mp4")
            save_video(video_frames, video_path)

    elapsed = time.time() - t_start
    env.close()
    client.close()

    # Summary
    print_closed_loop_summary(results)
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/args.episodes:.2f}s/ep)")

    # Save JSON
    output = results_to_json_closed_loop(
        results, args.backend, args.gripper, args.prompt,
        args.seed, args.horizon)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
