#!/usr/bin/env python3
"""
Pika MuJoCo Simulation Launcher

Usage:
    python run_sim.py                  # Launch sensor head sim (default)
    python run_sim.py --model gripper  # Launch gripper sim
    python run_sim.py --model sensor   # Launch sensor head sim

Requirements:
    pip install mujoco mujoco-viewer-py
    # OR just: pip install mujoco (includes built-in viewer since v3.0)
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
    parser = argparse.ArgumentParser(description="Pika MuJoCo Simulation")
    parser.add_argument(
        "--model",
        choices=["sensor", "gripper"],
        default="sensor",
        help="Which model to simulate (default: sensor)",
    )
    parser.add_argument(
        "--ctrl",
        type=float,
        default=None,
        help="Fixed control input for center_joint (rad) or gripper opening (m)",
    )
    args = parser.parse_args()

    # Resolve model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.model == "sensor":
        model_path = os.path.join(script_dir, "pika_sensor.xml")
    else:
        model_path = os.path.join(script_dir, "pika_gripper.xml")

    print(f"Loading model: {model_path}")

    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Launch interactive viewer
    print(f"\nPika {args.model} simulation ready.")
    print("Controls:")
    print("  - Right-click drag: rotate view")
    print("  - Scroll: zoom")
    print("  - Double-click: track body")
    print("  - Ctrl+Right-click: apply perturbation force")
    print("  - Space: pause/unpause")
    print("  - Backspace: reset")

    if args.ctrl is not None:
        # Run with fixed control
        def controller(model, data):
            data.ctrl[0] = args.ctrl

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                controller(model, data)
                mujoco.mj_step(model, data)
                viewer.sync()
    else:
        # Run with sinusoidal demo motion
        def controller(model, data):
            t = data.time
            if args.model == "sensor":
                # Oscillate center_joint between -0.3 and 1.0 rad
                data.ctrl[0] = 0.35 + 0.65 * np.sin(t * 1.5)
            else:
                # Oscillate gripper between closed and open
                data.ctrl[0] = -0.025 + 0.025 * np.sin(t * 2.0)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                controller(model, data)
                mujoco.mj_step(model, data)
                viewer.sync()


if __name__ == "__main__":
    main()
