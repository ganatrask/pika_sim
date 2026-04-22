#!/usr/bin/env python3
"""
Trossen Gripper MuJoCo Simulation Launcher

Usage:
    python run_sim.py                   # Launch standalone gripper (default)
    python run_sim.py --model pickplace # Launch pick-and-place scene
    python run_sim.py --ctrl 0.022      # Fixed control (half-open)

Requirements:
    pip install mujoco
"""

import argparse
import os
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("Error: mujoco package not found.")
    print("Install with: pip install mujoco")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Trossen Gripper MuJoCo Simulation")
    parser.add_argument(
        "--model",
        choices=["gripper", "pickplace"],
        default="gripper",
        help="Which model to simulate (default: gripper)",
    )
    parser.add_argument(
        "--ctrl",
        type=float,
        default=None,
        help="Fixed gripper control input (0=closed, 0.044=open)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.model == "pickplace":
        model_path = os.path.join(script_dir, "trossen_gripper_pickplace.xml")
    else:
        model_path = os.path.join(script_dir, "trossen_gripper.xml")

    print(f"Loading model: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print(f"\nTrossen {args.model} simulation ready.")
    print("Controls:")
    print("  - Right-click drag: rotate view")
    print("  - Scroll: zoom")
    print("  - Double-click: track body")
    print("  - Ctrl+Right-click: apply perturbation force")
    print("  - Space: pause/unpause")
    print("  - Backspace: reset")

    if args.ctrl is not None:
        def controller(model, data):
            if args.model == "pickplace":
                data.ctrl[3] = args.ctrl  # gripper is actuator index 3
            else:
                data.ctrl[0] = args.ctrl

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                controller(model, data)
                mujoco.mj_step(model, data)
                viewer.sync()
    else:
        # Sinusoidal open/close demo
        def controller(model, data):
            t = data.time
            # Oscillate gripper between closed (0) and open (0.044)
            val = 0.022 + 0.022 * np.sin(t * 2.0)
            if args.model == "pickplace":
                data.ctrl[3] = val
            else:
                data.ctrl[0] = val

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                controller(model, data)
                mujoco.mj_step(model, data)
                viewer.sync()


if __name__ == "__main__":
    main()
