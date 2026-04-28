#!/usr/bin/env bash
# =============================================================================
# run_all_evals.sh — Run all 52 model×checkpoint×gripper eval combinations
#
# Usage:
#   ./run_all_evals.sh                  # Run all 52 combinations
#   ./run_all_evals.sh --resume         # Skip already-completed runs
#   ./run_all_evals.sh --dry-run        # Print what would run without executing
#   ./run_all_evals.sh --filter "pika"  # Only run combos matching filter
#
# Each run:
#   1. Starts the inference server with the given checkpoint
#   2. Waits for it to become ready
#   3. Runs eval_closed_loop.py (50 episodes, seed 701)
#   4. Kills the server
#   5. Writes a note.txt with run metadata
#
# Results go to eval_runs/<run_name>/ with descriptive directory names.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_PKG_DIR="${SCRIPT_DIR}/pika_gripper_mujoco_sim"
CHECKPOINT_BASE="${SCRIPT_DIR}/sim_checkpoints"
DATASETS_DIR="/home/shyam/projects/cc/model-playground/datasets"
ROBOT_INFERENCE_SRC="/home/shyam/projects/cc/robot-inference/server/src"
ROBOT_INFERENCE_CLIENT_SRC="/home/shyam/projects/cc/robot-inference/client/src"
ROBOT_INFERENCE_SERVER_DIR="/home/shyam/projects/cc/robot-inference/server"
MODEL_PLAYGROUND_DIR="/home/shyam/projects/cc/model-playground"
EVAL_BASE_DIR="${SIM_PKG_DIR}/eval_runs_matrix"
PORT=5556
URL="tcp://localhost:${PORT}"
SEED=701
EPISODES=50
HORIZON=180
SERVER_STARTUP_TIMEOUT=300  # seconds to wait for server (model+CUDA init can take ~2-3min)
SERVER_PID=""

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
RESUME=false
DRY_RUN=false
FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)  RESUME=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --filter)  FILTER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Model definitions: (short_name, checkpoint_dir, available_steps)
# ---------------------------------------------------------------------------
# Steps available for each model (derived from ls of checkpoint dirs)
# All models have: 5000 10000 15000 20000 25000
# Mix v1 also has: last

declare -A MODEL_DIRS
MODEL_DIRS=(
    ["pika_v0"]="act_vae_pick_and_place_pika_sim_v0"
    ["pika_v1"]="act_vae_pick_and_place_pika_sim_v1"
    ["trossen_v0"]="act_vae_pick_and_place_trossen_sim_v0"
    ["trossen_v1"]="act_vae_pick_and_place_trossen_sim_v1"
    ["mix_95t_5p_v0"]="act_vae_pick_and_place_95_trossen_5_pika_sim_v0"
    ["mix_95t_5p_v1"]="act_vae_pick_and_place_95_trossen_5_pika_sim_v1"
)

# Steps per model (space-separated)
declare -A MODEL_STEPS
MODEL_STEPS=(
    ["pika_v0"]="5000 10000 15000 20000 25000"
    ["pika_v1"]="5000 10000 15000 20000 25000"
    ["trossen_v0"]="5000 10000 15000 20000 25000"
    ["trossen_v1"]="5000 10000 15000 20000 25000"
    ["mix_95t_5p_v0"]="5000 10000 15000 20000 25000 last"
    ["mix_95t_5p_v1"]="5000 10000 15000 20000 25000 last"
)

GRIPPERS=("pika" "trossen")

# Ordered model list for deterministic iteration
MODEL_ORDER=("pika_v0" "pika_v1" "trossen_v0" "trossen_v1" "mix_95t_5p_v0" "mix_95t_5p_v1")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Cleaning up: killing server (PID $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_server() {
    local checkpoint_path="$1"
    local step="$2"

    local actual_step="$step"
    if [[ "$step" == "last" ]]; then
        # Server expects policy_step_<N>.ckpt — symlink policy_last.ckpt
        # as policy_step_99999.ckpt so the server can load it
        if [[ -f "${checkpoint_path}/policy_last.ckpt" ]]; then
            ln -sf policy_last.ckpt "${checkpoint_path}/policy_step_99999.ckpt"
            actual_step=99999
        else
            log "ERROR: policy_last.ckpt not found in $checkpoint_path"
            return 1
        fi
    fi
    local step_arg="--checkpoint-step $actual_step"

    log "Starting server: checkpoint=$checkpoint_path step=$step"
    # Use model-playground venv which has torch + all ML deps
    PYTHONPATH="$ROBOT_INFERENCE_SRC:$MODEL_PLAYGROUND_DIR" \
        "$MODEL_PLAYGROUND_DIR/.venv/bin/python" -m robot_inference_server.cli \
        --backend act \
        --checkpoint "$checkpoint_path" \
        $step_arg \
        --datasets-dir "$DATASETS_DIR" \
        --port "$PORT" \
        &>"${CURRENT_RUN_DIR}/server.log" &
    SERVER_PID=$!
    log "Server started (PID $SERVER_PID)"
}

wait_for_server() {
    log "Waiting for server on port $PORT..."
    local elapsed=0
    while true; do
        # Check if server process died
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "ERROR: Server process died"
            if [[ -f "${CURRENT_RUN_DIR}/server.log" ]]; then
                log "Last 20 lines of server log:"
                tail -20 "${CURRENT_RUN_DIR}/server.log"
            fi
            return 1
        fi
        # Check if port is open using bash built-in
        if (echo > /dev/tcp/localhost/$PORT) 2>/dev/null; then
            break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        if [[ $elapsed -ge $SERVER_STARTUP_TIMEOUT ]]; then
            log "ERROR: Server failed to start within ${SERVER_STARTUP_TIMEOUT}s"
            if [[ -f "${CURRENT_RUN_DIR}/server.log" ]]; then
                log "Last 20 lines of server log:"
                tail -20 "${CURRENT_RUN_DIR}/server.log"
            fi
            return 1
        fi
    done
    # Extra wait for server to finish initialization after port opens
    sleep 3
    log "Server is ready (took ${elapsed}s)"
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping server (PID $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

run_is_complete() {
    local run_dir="$1"
    [[ -f "${run_dir}/results.json" ]]
}

# ---------------------------------------------------------------------------
# Build the eval matrix
# ---------------------------------------------------------------------------

declare -a RUN_NAMES=()
declare -A RUN_MODEL=()
declare -A RUN_STEP=()
declare -A RUN_GRIPPER=()

for model in "${MODEL_ORDER[@]}"; do
    for step in ${MODEL_STEPS[$model]}; do
        for gripper in "${GRIPPERS[@]}"; do
            name="${model}__step_${step}__${gripper}"
            RUN_NAMES+=("$name")
            RUN_MODEL["$name"]="$model"
            RUN_STEP["$name"]="$step"
            RUN_GRIPPER["$name"]="$gripper"
        done
    done
done

TOTAL=${#RUN_NAMES[@]}
log "Total combinations: $TOTAL"

# ---------------------------------------------------------------------------
# Filter and count
# ---------------------------------------------------------------------------

declare -a FILTERED_RUNS=()
for name in "${RUN_NAMES[@]}"; do
    if [[ -n "$FILTER" && "$name" != *"$FILTER"* ]]; then
        continue
    fi
    if $RESUME && run_is_complete "${EVAL_BASE_DIR}/${name}"; then
        continue
    fi
    FILTERED_RUNS+=("$name")
done

if $RESUME; then
    log "Resume mode: ${#FILTERED_RUNS[@]} remaining out of $TOTAL"
fi
if [[ -n "$FILTER" ]]; then
    log "Filter '$FILTER': ${#FILTERED_RUNS[@]} runs match"
fi

# ---------------------------------------------------------------------------
# Dry run — just print the matrix
# ---------------------------------------------------------------------------

if $DRY_RUN; then
    echo ""
    printf "%-45s %-10s %-8s %s\n" "RUN NAME" "STEP" "GRIPPER" "CHECKPOINT"
    printf "%s\n" "$(printf '=%.0s' {1..100})"
    for name in "${FILTERED_RUNS[@]}"; do
        model="${RUN_MODEL[$name]}"
        step="${RUN_STEP[$name]}"
        gripper="${RUN_GRIPPER[$name]}"
        ckpt="${CHECKPOINT_BASE}/${MODEL_DIRS[$model]}"
        printf "%-45s %-10s %-8s %s\n" "$name" "$step" "$gripper" "$(basename "$ckpt")"
    done
    echo ""
    echo "Total: ${#FILTERED_RUNS[@]} runs"
    exit 0
fi

# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

mkdir -p "$EVAL_BASE_DIR"

# Master log
MASTER_LOG="${EVAL_BASE_DIR}/master_log.txt"
echo "Eval run started: $(date)" >> "$MASTER_LOG"
echo "Total runs: ${#FILTERED_RUNS[@]}" >> "$MASTER_LOG"
echo "---" >> "$MASTER_LOG"

COMPLETED=0
FAILED=0
SKIPPED=0

cd "$SIM_PKG_DIR"

for i in "${!FILTERED_RUNS[@]}"; do
    name="${FILTERED_RUNS[$i]}"
    model="${RUN_MODEL[$name]}"
    step="${RUN_STEP[$name]}"
    gripper="${RUN_GRIPPER[$name]}"
    checkpoint_path="${CHECKPOINT_BASE}/${MODEL_DIRS[$model]}"
    run_num=$((i + 1))

    CURRENT_RUN_DIR="${EVAL_BASE_DIR}/${name}"

    echo ""
    log "=========================================================="
    log "[$run_num/${#FILTERED_RUNS[@]}] $name"
    log "  Model:      $model"
    log "  Checkpoint:  $(basename "$checkpoint_path")"
    log "  Step:        $step"
    log "  Gripper:     $gripper"
    log "=========================================================="

    # Skip if already done (extra safety)
    if run_is_complete "$CURRENT_RUN_DIR"; then
        log "SKIP: already complete"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    mkdir -p "$CURRENT_RUN_DIR"

    # Write note.txt
    cat > "${CURRENT_RUN_DIR}/note.txt" <<EOF
model: ${model}
checkpoint: ${checkpoint_path}
step: ${step}
gripper: ${gripper}
seed: ${SEED}
episodes: ${EPISODES}
horizon: ${HORIZON}
started: $(date -Iseconds)
EOF

    # Start server
    if ! start_server "$checkpoint_path" "$step"; then
        log "FAILED: could not start server"
        echo "FAIL $name — server start failed" >> "$MASTER_LOG"
        FAILED=$((FAILED + 1))
        stop_server
        continue
    fi

    # Wait for server to be ready
    if ! wait_for_server; then
        log "FAILED: server not ready"
        echo "FAIL $name — server timeout" >> "$MASTER_LOG"
        FAILED=$((FAILED + 1))
        stop_server
        continue
    fi

    # Run eval
    log "Running eval_closed_loop.py..."
    if PYTHONPATH="$ROBOT_INFERENCE_CLIENT_SRC:${PYTHONPATH:-}" \
       "$MODEL_PLAYGROUND_DIR/.venv/bin/python" eval_closed_loop.py \
        --gripper "$gripper" \
        --backend act \
        --url "$URL" \
        --episodes "$EPISODES" \
        --seed "$SEED" \
        --horizon "$HORIZON" \
        --save-video \
        --eval-dir "$CURRENT_RUN_DIR" \
        2>&1 | tee "${CURRENT_RUN_DIR}/eval.log"; then

        # eval_closed_loop creates a numbered subdir (eval_001/).
        # Move its contents up to our named directory for cleaner structure.
        eval_subdir=$(find "$CURRENT_RUN_DIR" -maxdepth 1 -name "eval_*" -type d | head -1)
        if [[ -n "$eval_subdir" && -f "${eval_subdir}/results.json" ]]; then
            mv "${eval_subdir}/results.json" "${CURRENT_RUN_DIR}/results.json"
            if [[ -d "${eval_subdir}/videos" ]]; then
                mv "${eval_subdir}/videos" "${CURRENT_RUN_DIR}/videos" 2>/dev/null || true
            fi
            rmdir "$eval_subdir" 2>/dev/null || rm -rf "$eval_subdir"
        fi

        # Extract summary from results
        if [[ -f "${CURRENT_RUN_DIR}/results.json" ]]; then
            summary=$(python3 -c "
import json, sys
with open('${CURRENT_RUN_DIR}/results.json') as f:
    r = json.load(f)
a = r.get('aggregated', {})
print(f\"success@50mm={a.get('success_rate_50mm',0):.0%}  \"
      f\"success@20mm={a.get('success_rate_20mm',0):.0%}  \"
      f\"mean_err={a.get('mean_placement_error_mm',0):.1f}mm  \"
      f\"pick_fail={a.get('failure_rates',{}).get('pick_fail',0):.0%}\")
" 2>/dev/null || echo "could not parse results")

            log "DONE: $summary"
            echo "OK   $name — $summary" >> "$MASTER_LOG"
            # Append completion time to note
            echo "completed: $(date -Iseconds)" >> "${CURRENT_RUN_DIR}/note.txt"
            echo "result: $summary" >> "${CURRENT_RUN_DIR}/note.txt"
            COMPLETED=$((COMPLETED + 1))
        else
            log "FAILED: no results.json produced"
            echo "FAIL $name — no results.json" >> "$MASTER_LOG"
            FAILED=$((FAILED + 1))
        fi
    else
        log "FAILED: eval script exited with error"
        echo "FAIL $name — eval error" >> "$MASTER_LOG"
        FAILED=$((FAILED + 1))
    fi

    # Stop server before next run
    stop_server
    sleep 2
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

echo ""
log "=========================================================="
log "ALL DONE"
log "  Completed: $COMPLETED"
log "  Failed:    $FAILED"
log "  Skipped:   $SKIPPED"
log "  Total:     ${#FILTERED_RUNS[@]}"
log "  Results:   $EVAL_BASE_DIR"
log "=========================================================="

echo "---" >> "$MASTER_LOG"
echo "Finished: $(date)  completed=$COMPLETED failed=$FAILED skipped=$SKIPPED" >> "$MASTER_LOG"

# Generate summary table
log "Generating summary table..."
python3 - "$EVAL_BASE_DIR" <<'PYEOF'
import json, os, sys

base_dir = sys.argv[1]
rows = []

for name in sorted(os.listdir(base_dir)):
    results_path = os.path.join(base_dir, name, "results.json")
    if not os.path.isfile(results_path):
        continue
    with open(results_path) as f:
        r = json.load(f)
    a = r.get("aggregated", {})
    fr = a.get("failure_rates", {})
    # Parse run name: model__step_N__gripper
    parts = name.split("__")
    model = parts[0] if len(parts) >= 1 else name
    step = parts[1].replace("step_", "") if len(parts) >= 2 else "?"
    gripper = parts[2] if len(parts) >= 3 else "?"
    rows.append({
        "model": model,
        "step": step,
        "gripper": gripper,
        "s50": a.get("success_rate_50mm", 0),
        "s20": a.get("success_rate_20mm", 0),
        "s5": a.get("success_rate_5mm", 0),
        "err": a.get("mean_placement_error_mm", 0),
        "std": a.get("std_placement_error_mm", 0),
        "pf": fr.get("pick_fail", 0),
        "drop": fr.get("drop", 0),
        "inacc": fr.get("inaccurate", 0),
    })

if not rows:
    print("No results found.")
    sys.exit(0)

# Print table
header = f"{'Model':<18s} {'Step':>6s} {'Grip':>7s} | {'@50mm':>6s} {'@20mm':>6s} {'@5mm':>6s} | {'MeanErr':>8s} {'StdErr':>8s} | {'PickF':>6s} {'Drop':>6s} {'Inacc':>6s}"
sep = "-" * len(header)
print(f"\n{sep}")
print(header)
print(sep)
for r in rows:
    print(f"{r['model']:<18s} {r['step']:>6s} {r['gripper']:>7s} | "
          f"{r['s50']:>5.0%} {r['s20']:>5.0%} {r['s5']:>5.0%} | "
          f"{r['err']:>7.1f}m {r['std']:>7.1f}m | "
          f"{r['pf']:>5.0%} {r['drop']:>5.0%} {r['inacc']:>5.0%}")
print(sep)

# Save as CSV
csv_path = os.path.join(base_dir, "summary.csv")
with open(csv_path, "w") as f:
    f.write("model,step,gripper,success_50mm,success_20mm,success_5mm,mean_error_mm,std_error_mm,pick_fail,drop,inaccurate\n")
    for r in rows:
        f.write(f"{r['model']},{r['step']},{r['gripper']},{r['s50']:.3f},{r['s20']:.3f},{r['s5']:.3f},{r['err']:.1f},{r['std']:.1f},{r['pf']:.3f},{r['drop']:.3f},{r['inacc']:.3f}\n")
print(f"\nSummary CSV saved to: {csv_path}")
PYEOF
