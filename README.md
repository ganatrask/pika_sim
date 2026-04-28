# Pika Sim

MuJoCo simulation environments for robotic gripper pick-and-place tasks. Includes two gripper models — the [Agilex Pika](https://github.com/agilexrobotics) and the [Trossen WXAI](https://www.trossenrobotics.com/) — each with a 3-DOF Cartesian gantry, multi-camera rendering, sensor logging, and dataset recording for imitation learning.

## Environment Setup

### Option A: conda (recommended)

```bash
# Python 3.11+ required by robot-inference-client
conda create -n gripper_sim python=3.11 -y
conda activate gripper_sim

# Install sim + eval dependencies
pip install -e ".[eval]"

# Install robot-inference client (needed for eval scripts)
pip install -e "../robot-inference/client[act]"
```

### Option B: uv

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate

uv pip install -e ".[eval]"
uv pip install -e "../robot-inference/client[act]"
```

### Verify installation

```bash
python -c "import mujoco, h5py, cv2, robot_inference_client; print('All dependencies OK')"
```

### Directory layout

Both repos should be cloned as siblings:

```
your_workspace/
├── pika_sim/             # this repo (git@github.com:ganatrask/pika_sim.git)
│   ├── pyproject.toml
│   ├── pika_gripper_mujoco_sim/
│   └── trossen_gripper_mujoco_sim/
└── robot-inference/      # inference SDK (client + server)
    └── client/           # pip install -e "client[act]"
```

The eval scripts also auto-discover `robot_inference_client` from sibling
directories without pip-installing it — but `pip install` is recommended for
a clean reproducible setup.

## Simulations

### Pika Gripper (`pika_gripper_mujoco_sim/`)

Agilex Pika parallel-jaw gripper and dual-gripper sensor head.

| | |
|---|---|
| **Models** | Standalone gripper, pick-and-place environment, sensor head |
| **Gripper stroke** | 0–50 mm |
| **Cameras** | 5 (fisheye, RealSense D405, wrist cam, overhead, front) |
| **Meshes** | 16 STL files (gripper: 3, sensor head: 13) |
| **Dataset** | [ganatrask/pika_sim](https://huggingface.co/datasets/ganatrask/pika_sim) |

```bash
cd pika_gripper_mujoco_sim
python3 run_sim.py                        # Sensor head (default)
python3 run_sim.py --model gripper        # Standalone gripper
python3 pick_and_place.py --loop          # Randomized pick-and-place
python3 record_dataset.py --episodes 100  # Record dataset
python3 camera_viewer.py --depth          # Multi-camera view
```

### Trossen Gripper (`trossen_gripper_mujoco_sim/`)

Trossen WXAI parallel-jaw gripper with RealSense D405 camera mount.

| | |
|---|---|
| **Models** | Standalone gripper, pick-and-place environment |
| **Gripper stroke** | 0–44 mm (~88 mm total gap) |
| **Cameras** | 5 (RealSense D405, stereo left/right, overhead, front) |
| **Meshes** | 6 STL files |
| **Dataset** | [ganatrask/trossen_sim](https://huggingface.co/datasets/ganatrask/trossen_sim) |

```bash
cd trossen_gripper_mujoco_sim
python3 run_sim.py                        # Standalone gripper (default)
python3 run_sim.py --model pickplace      # Pick-and-place scene
python3 pick_and_place.py --loop          # Randomized pick-and-place
python3 record_dataset.py --episodes 100  # Record dataset
python3 camera_viewer.py --depth          # Multi-camera view
```

## Comparison

| | Pika | Trossen |
|---|---|---|
| Gripper type | Parallel jaw | Parallel jaw |
| Stroke | 50 mm | 44 mm (88 mm gap) |
| Gripper kp | 500 | 100 |
| Gantry kp | 5000 | 5000 |
| Sim timestep | 1 ms (pickplace), 2 ms (standalone) | 1 ms (pickplace), 2 ms (standalone) |
| Cube | 2 cm, 30 g | 2 cm, 30 g |
| Workspace | X: +/-0.20 m, Y: +/-0.10 m | X: +/-0.20 m, Y: +/-0.10 m |
| Sensor head model | Yes (16 mimic joints) | No |

## Dataset Format

Both simulations produce identical HDF5 dataset structures via `record_dataset.py`:

```
dataset/
├── episodes/
│   ├── episode_0000.h5
│   └── ...
└── videos/
    ├── wrist_cam_0000.mp4
    └── ...
```

Each episode contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `actions/ee_pose` | (T, 7) | Target end-effector pose [x, y, z, qw, qx, qy, qz] |
| `actions/gripper` | (T, 1) | Gripper command (0.0 = open, 1.0 = closed) |
| `observations/ee_pose` | (T, 7) | Actual end-effector pose |
| `observations/gripper_width` | (T, 1) | Gripper opening in meters |
| `observations/timestamp` | (T,) | Seconds since episode start |
| `videos/wrist_cam` | (T,) | Frame indices into companion MP4 |
| `env_state` | scalar | JSON metadata (cube poses, placement error) |
| `success` | scalar | True if placement error < 20 mm |

## Policy Evaluation

Evaluate trained ACT VAE policies against the MuJoCo simulation. Requires a running inference server on a GPU machine.

### Architecture

```
┌──────────────────────────────┐       ┌──────────────────────────────┐
│  This machine (eval)         │       │  GPU machine (inference)     │
│                              │       │                              │
│  MuJoCo sim                  │  ZMQ  │  Trained ACT checkpoint      │
│    ├─ render camera → image ─┼──────>│    ├─ normalize qpos         │
│    ├─ read sensors  → qpos  ─┼──────>│    ├─ model forward pass     │
│    └─ apply action <─────────┼───────┤    └─ denormalize action     │
│                              │       │                              │
│  eval_closed_loop.py         │       │  robot-inference server      │
└──────────────────────────────┘       └──────────────────────────────┘
```

### Start inference server (GPU machine)

```bash
cd robot-inference/server
uv run --extra act serve --backend act \
    --checkpoint /path/to/act_vae_pick_and_place_pika_sim_v0 \
    --checkpoint-step 25000 \
    --port 5556
```

### Run evaluation (eval machine)

```bash
cd pika_gripper_mujoco_sim

# Open-loop: replay recorded episodes, compare predictions to ground truth
python eval_open_loop.py \
    --dataset ./dataset \
    --backend act \
    --url tcp://<gpu-machine>:5556 \
    --gripper pika \
    --max-episodes 20

# Closed-loop: policy drives the MuJoCo sim
python eval_closed_loop.py \
    --gripper pika \
    --backend act \
    --url tcp://<gpu-machine>:5556 \
    --episodes 50 --seed 42 \
    --save-video
```

### Eval outputs

- `results_open_loop.json` — per-episode RMSE (position, orientation, gripper accuracy)
- `results_closed_loop.json` — success rates at 5/20/50 mm, placement error per episode
- `eval_videos/` — optional MP4 recordings of wrist camera per episode

### Cross-gripper sweep

```bash
for gripper in pika trossen; do
    for version in v0 v1; do
        python eval_closed_loop.py \
            --gripper $gripper --backend act \
            --url tcp://<gpu-machine>:5556 \
            --episodes 50 --seed 42 \
            --output "results_${gripper}_${version}.json"
    done
done
```

## Full Eval Matrix (`run_all_evals.sh`)

`run_all_evals.sh` automates running every model, checkpoint step, and gripper combination end-to-end. For each combo it starts an inference server, runs 50 closed-loop episodes, kills the server, and moves on.

### Prerequisites

- Checkpoints in `sim_checkpoints/` (git-ignored), organized as:
  ```
  sim_checkpoints/
  ├── act_vae_pick_and_place_pika_sim_v0/
  │   ├── policy_step_5000.ckpt
  │   ├── policy_step_10000.ckpt
  │   └── ...
  ├── act_vae_pick_and_place_trossen_sim_v1/
  └── ...
  ```
- `robot-inference` server and client repos at their expected paths
- `model-playground` venv with torch + ML deps

### Models in the matrix

| Short name | Checkpoint directory | Training data |
|---|---|---|
| `pika_v0` | `act_vae_pick_and_place_pika_sim_v0` | Pika-only |
| `pika_v1` | `act_vae_pick_and_place_pika_sim_v1` | Pika-only |
| `trossen_v0` | `act_vae_pick_and_place_trossen_sim_v0` | Trossen-only |
| `trossen_v1` | `act_vae_pick_and_place_trossen_sim_v1` | Trossen-only |
| `mix_95t_5p_v0` | `act_vae_pick_and_place_95_trossen_5_pika_sim_v0` | 95% Trossen + 5% Pika |
| `mix_95t_5p_v1` | `act_vae_pick_and_place_95_trossen_5_pika_sim_v1` | 95% Trossen + 5% Pika |

Each model is evaluated at steps `5000 10000 15000 20000 25000` (mix models also include `last`), against both `pika` and `trossen` grippers — 52 combinations total.

### Usage

```bash
# Preview what would run (no execution)
./run_all_evals.sh --dry-run

# Run everything
./run_all_evals.sh

# Resume after interruption (skips runs that already have results.json)
./run_all_evals.sh --resume

# Only run combos matching a filter
./run_all_evals.sh --filter "pika_v0"
./run_all_evals.sh --filter "trossen"
./run_all_evals.sh --filter "step_25000"
```

### What each run does

1. Starts the inference server with the checkpoint + step
2. Waits for the server to be ready (up to 5 min for CUDA init)
3. Runs `eval_closed_loop.py` (50 episodes, seed 701, horizon 180)
4. Kills the server
5. Writes `note.txt` with run metadata

### Output structure

Results go to `pika_gripper_mujoco_sim/eval_runs_matrix/`:

```
eval_runs_matrix/
├── pika_v0__step_5000__pika/
│   ├── results.json        # Success rates, placement errors, failure breakdown
│   ├── note.txt            # Run metadata (model, step, gripper, timing)
│   ├── server.log          # Inference server stdout/stderr
│   ├── eval.log            # eval_closed_loop.py output
│   └── videos/             # Per-episode wrist cam MP4s (if --save-video)
├── pika_v0__step_5000__trossen/
├── ...
├── master_log.txt          # One-line status per run (OK/FAIL + summary)
└── summary.csv             # Aggregated table of all results
```

### Reading results

The `summary.csv` contains one row per run with success rates and error metrics:

```
model,step,gripper,success_50mm,success_20mm,success_5mm,mean_error_mm,std_error_mm,pick_fail,drop,inaccurate
pika_v0,5000,pika,0.820,0.640,0.120,18.3,22.1,0.080,0.040,0.040
...
```

Key metrics:
- **success@50mm / @20mm / @5mm** — fraction of episodes where cube placement error is below threshold
- **mean_error_mm** — average placement error across all episodes
- **pick_fail** — fraction where the gripper failed to pick up the cube
- **drop** — fraction where the cube was picked but dropped before placement
- **inaccurate** — fraction where the cube was placed but outside the 50mm threshold

## Project Structure

```
sim/
├── pyproject.toml                  # Dependencies (pip install -e ".[eval]")
├── pika_gripper_mujoco_sim/
│   ├── run_sim.py                  # Interactive viewer
│   ├── pick_and_place.py           # Pick-and-place demo
│   ├── record_dataset.py           # Dataset recorder
│   ├── camera_viewer.py            # Multi-camera rendering
│   ├── camera_tuner.py             # Camera parameter tuning
│   ├── inspect_episode.py          # Episode analysis
│   ├── eval_common.py              # Eval shared utilities (PikaSimEnv, metrics)
│   ├── eval_open_loop.py           # Open-loop evaluation
│   ├── eval_closed_loop.py         # Closed-loop evaluation
│   ├── pika_gripper.xml            # Standalone gripper
│   ├── pika_gripper_pickplace.xml  # Pick-and-place environment
│   ├── pika_sensor.xml             # Sensor head (dual gripper)
│   └── meshes/                     # 16 STL files
├── trossen_gripper_mujoco_sim/
│   ├── run_sim.py                  # Interactive viewer
│   ├── pick_and_place.py           # Pick-and-place demo
│   ├── record_dataset.py           # Dataset recorder
│   ├── camera_viewer.py            # Multi-camera rendering
│   ├── trossen_gripper.xml         # Standalone gripper
│   ├── trossen_gripper_pickplace.xml # Pick-and-place environment
│   └── meshes/                     # 6 STL files
├── run_all_evals.sh                # Run all model×checkpoint×gripper eval combinations
├── sim_checkpoints/                # Trained checkpoints (git-ignored)
└── docs/                           # Implementation plans
```

