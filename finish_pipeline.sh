#!/bin/bash
# Overnight pipeline: wait for paper-config training -> benchmark -> commit -> push.
# Logs to RL_outputs/finish-pipeline.log
set -u
cd /home/jarvis/projects/loader_learning
LOG=RL_outputs/finish-pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "pipeline started, waiting for training PID $(cat RL_outputs/train.pid)"
while ps -p "$(cat RL_outputs/train.pid)" >/dev/null 2>&1; do sleep 300; done
say "training process exited"

FINAL=$(ls -t RL_outputs/*/stage_7_final.zip 2>/dev/null | head -1)
if [ -z "$FINAL" ]; then
  say "WARNING: no stage_7_final.zip found (training may have crashed); loader_critic holds the last saved stage"
else
  say "stage 7 final checkpoint: $FINAL"
fi
say "loader_critic mtime: $(stat -c %y loader_critic 2>/dev/null || echo missing)"

say "running final obstacle benchmark (slack=10000)"
OBS_SLACK=10000 ACADOS_SOURCE_DIR=/home/jarvis/projects/acados \
  LD_LIBRARY_PATH=/home/jarvis/projects/acados/lib \
  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  .venv-demo/bin/python benchmark_mpc.py >> "$LOG" 2>&1
BENCH_JSON=$(ls -t RL_outputs/mpc-bench-current-*.json | head -1)
say "benchmark saved: $BENCH_JSON"

mkdir -p benchmarks
cp "$BENCH_JSON" benchmarks/paper-config-final.json
SUMMARY=$(python3 -c "
import json
d = json.load(open('benchmarks/paper-config-final.json'))
print('convergence %s, collisions %s/20, collision-free %s, converge median %ss' % (
    d['convergence_rate'], d['collisions'], d['collision_free_rate'],
    d['converge_time_median_s']))")
say "benchmark summary: $SUMMARY"

git add teach_loader.py loader_navigation_rl/loader_goal_env.py benchmarks/
git commit -m "Retrain with paper configuration (Eq.9 p=0.25 cost, [48,96,144,96,48] nets)

Motivation: with the repo's sqrt cost and [96,96,96] nets, the fully
trained (35M steps) critic scored WORSE on the obstacle benchmark than
a mid-curriculum snapshot (5% vs 40% convergence), suggesting the late
curriculum stages over-constrained the cost landscape.

Changes:
- compute_reward: c = ||We||_p with p = 0.25 (paper Eq. 9) instead of
  (sum w*|e|)^(1/2).
- net_arch pi/qf: [48,96,144,96,48] (paper Sec. V-A). Note ~1.5x
  activation memory vs [96,96,96]: batch lowered 200k -> 150k on 6GB.

Overnight benchmark result (20 fixed obstacle scenarios, slack=10000):
$SUMMARY
Full data: benchmarks/paper-config-final.json
Baseline comparisons: repo-config final critic 5%, stage-4 snapshot 40%,
no-critic quadratic baseline 10%." >> "$LOG" 2>&1
say "committed: $(git log --oneline -1)"

git push fork fix/training-correctness-and-mpc-compat >> "$LOG" 2>&1 \
  && say "pushed to fork" || say "PUSH FAILED - see log"
say "pipeline done"
