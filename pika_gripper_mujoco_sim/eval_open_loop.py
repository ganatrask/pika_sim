#!/usr/bin/env python3
"""
Open-loop policy evaluation: replay recorded HDF5 episodes through the
inference server and compare predicted actions to ground-truth.

No MuJoCo stepping — purely measures prediction accuracy.

Usage:
    python eval_open_loop.py \
        --dataset ./dataset \
        --backend act \
        --url tcp://gpu-box:5556 \
        --gripper pika \
        --max-episodes 20 \
        --output results_open_loop.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from robot_inference_client import get_client

from eval_common import (
    GRIPPER_CONFIGS,
    build_action,
    build_observation,
    build_qpos,
    compute_action_rmse,
    episode_gt_actions,
    list_episodes,
    load_episode,
    load_episode_video,
    print_open_loop_summary,
    results_to_json_open_loop,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open-loop policy evaluation (HDF5 replay)")
    parser.add_argument("--dataset", type=str, default="./dataset",
                        help="Path to dataset directory")
    parser.add_argument("--backend", type=str, default="act",
                        help="Inference backend (act, openpi, cosmos)")
    parser.add_argument("--url", type=str, default="tcp://localhost:5556",
                        help="Inference server URL")
    parser.add_argument("--gripper", type=str, default="pika",
                        choices=list(GRIPPER_CONFIGS.keys()),
                        help="Gripper type")
    parser.add_argument("--prompt", type=str, default="",
                        help="Task language instruction")
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="Max episodes to evaluate (0=all)")
    parser.add_argument("--output", type=str, default="results_open_loop.json",
                        help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = GRIPPER_CONFIGS[args.gripper]
    max_gap = cfg["MAX_GAP"]
    dataset_dir = Path(args.dataset)

    # Discover episodes
    episodes = list_episodes(dataset_dir)
    if not episodes:
        print(f"No episodes found in {dataset_dir / 'episodes'}")
        sys.exit(1)
    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]

    print(f"Open-loop eval: {len(episodes)} episodes from {dataset_dir}")
    print(f"  Backend: {args.backend} | URL: {args.url} | Gripper: {args.gripper}")
    if args.prompt:
        print(f"  Prompt: {args.prompt!r}")
    print("-" * 60)

    # Connect to inference server
    client_kwargs = {}
    if args.backend == "act":
        client_kwargs["url"] = args.url
    elif args.backend in ("openpi", "openpi_remote"):
        # Parse host:port from URL
        client_kwargs["host"] = args.url.split("://")[-1].split(":")[0]
        port = args.url.split(":")[-1]
        if port.isdigit():
            client_kwargs["port"] = int(port)
    elif args.backend == "cosmos":
        client_kwargs["url"] = args.url

    client = get_client(args.backend, **client_kwargs)
    client.connect()

    # Warmup — discard first inference
    print("Warming up inference server...")
    dummy_obs = build_observation(
        np.zeros(8, dtype=np.float32),
        {"wrist_cam": np.zeros((480, 640, 3), dtype=np.uint8)},
    )
    warmup_result = client.get_action(dummy_obs)
    assert warmup_result.action.shape == (8,), \
        f"Expected (8,) action, got {warmup_result.action.shape}"
    client.reset()
    print("Warmup done.")

    # Evaluate episodes
    all_metrics = []
    t_start = time.time()

    for ep_idx, h5_path in enumerate(episodes):
        client.reset()

        # Load episode data
        ep = load_episode(h5_path)
        T = ep["episode_length"]

        # Load video frames
        try:
            frames = load_episode_video(dataset_dir, ep_idx)
        except FileNotFoundError as e:
            print(f"  [{ep_idx}] Skipping: {e}")
            continue

        predictions = []
        ground_truth = []

        for t in range(T):
            # Build observation from saved data
            qpos = build_qpos(
                ep["obs_ee"][t],
                ep["obs_grip_width"][t, 0],
                max_gap,
            )

            # Get video frame
            frame_idx = ep["video_frames"][t]
            if frame_idx >= len(frames):
                break
            image = frames[frame_idx]

            obs = build_observation(
                qpos_8d=qpos,
                images_dict={"wrist_cam": image},
                prompt=args.prompt,
            )

            # Query server
            result = client.get_action(obs)
            action = result.action
            if action.ndim == 2:
                action = action[0]

            predictions.append(action)
            ground_truth.append(build_action(
                ep["actions_ee"][t],
                ep["actions_grip"][t, 0],
            ))

        if not predictions:
            print(f"  [{ep_idx}] No predictions, skipping")
            continue

        metrics = compute_action_rmse(
            np.array(predictions),
            np.array(ground_truth),
        )
        all_metrics.append(metrics)

        print(f"  [{ep_idx:4d}/{len(episodes)}] T={T:4d}  "
              f"RMSE_xyz={metrics.rmse_xyz:.5f}  "
              f"RMSE_all={metrics.rmse_overall:.5f}  "
              f"grip_acc={metrics.gripper_accuracy:.3f}")

    elapsed = time.time() - t_start
    client.close()

    if not all_metrics:
        print("No episodes evaluated successfully.")
        sys.exit(1)

    # Summary
    print_open_loop_summary(all_metrics)
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/len(all_metrics):.2f}s/ep)")

    # Save JSON
    output = results_to_json_open_loop(
        all_metrics, args.backend, args.gripper, args.prompt)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
