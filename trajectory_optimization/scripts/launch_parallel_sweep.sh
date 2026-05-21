#!/usr/bin/env bash
# Register a wandb sweep and run N single-threaded agents in parallel.
#
# Usage (from the scripts/ directory):
#   ./launch_parallel_sweep.sh <sweep.yaml> <N>
#     <sweep.yaml>  path to the wandb sweep config
#     <N>           number of parallel agents (each pinned to 1 cpu thread)
#
# Each run is cpu + single-env (~1 core), so throughput comes from running N agents
# at once. OMP_NUM_THREADS=1 keeps every torch process to one thread so they don't thrash.
# The script blocks while the agents run; press Ctrl-C to stop them all (including the
# train.py children they spawn). Per-worker output is tee'd to logs/agent_<i>.log.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <sweep.yaml> <N>" >&2
  exit 1
fi

sweep_yaml="$1"
n="$2"

if [ ! -f "$sweep_yaml" ]; then
  echo "error: sweep config '$sweep_yaml' not found" >&2
  exit 1
fi
if [ ! -f train.py ]; then
  echo "error: run this from the scripts/ directory (train.py not found in cwd)" >&2
  exit 1
fi

# recursively terminate a process and all of its descendants (uv -> wandb -> uv -> python)
kill_tree() {
  local pid="$1"
  local child
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    kill_tree "$child"
  done
  kill -TERM "$pid" 2>/dev/null || true
}

pids=()
cleanup() {
  trap - INT TERM  # don't re-enter while we tear down
  echo
  echo "stopping ${#pids[@]} agents..."
  for pid in "${pids[@]}"; do
    kill_tree "$pid"
  done
  wait 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

# register the sweep and recover the "entity/project/sweep_id" path from its output
sweep_output="$(uv run wandb sweep "$sweep_yaml" 2>&1)"
echo "$sweep_output"
sweep_id="$(echo "$sweep_output" | grep -oP 'wandb agent \K[\w./-]+' | tail -1 || true)"

if [ -z "$sweep_id" ]; then
  echo "error: could not parse the sweep id from wandb's output" >&2
  exit 1
fi

echo "launching $n agents for sweep: $sweep_id"
mkdir -p logs
for i in $(seq 1 "$n"); do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run wandb agent "$sweep_id" \
    > "logs/agent_${i}.log" 2>&1 &
  pids+=("$!")
  echo "  agent $i started (pid $!) -> logs/agent_${i}.log"
done

echo "running $n agents; press Ctrl-C to stop them all. tail with: tail -f logs/agent_1.log"
wait
