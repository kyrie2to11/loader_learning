#!/bin/bash
# Finish pipeline v2: wait for a FROZEN training PID -> benchmark -> commit -> push.
# Usage: ./finish_pipeline.sh <pid> <label>
set -u
cd /home/jarvis/projects/loader_learning
TRAIN_PID="${1:?need pid}"
LABEL="${2:-run}"
START_TS=$(date +%s)
LOG=RL_outputs/finish-pipeline-$LABEL.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "pipeline v2: waiting for PID $TRAIN_PID (label=$LABEL)"
while ps -p "$TRAIN_PID" >/dev/null 2>&1; do sleep 300; done
say "training process exited"

FINAL=$(find RL_outputs -name stage_7_final.zip -newermt "@$START_TS" 2>/dev/null | head -1)
if [ -z "$FINAL" ]; then
  say "WARNING: no stage_7_final.zip newer than pipeline start; training likely crashed"
else
  say "final checkpoint: $FINAL"
fi

say "running benchmark (with obstacles, slack=10000) on final loader_critic"
OBS_SLACK=10000 ACADOS_SOURCE_DIR=/home/jarvis/projects/acados \
  LD_LIBRARY_PATH=/home/jarvis/projects/acados/lib \
  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  .venv-demo/bin/python benchmark_mpc.py >>"$LOG" 2>&1
say "running benchmark (obstacle-free)"
NO_OBS=1 OBS_SLACK=10000 ACADOS_SOURCE_DIR=/home/jarvis/projects/acados \
  LD_LIBRARY_PATH=/home/jarvis/projects/acados/lib \
  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  .venv-demo/bin/python benchmark_mpc.py >>"$LOG" 2>&1

B1=$(ls -t RL_outputs/mpc-bench-current-*.json | grep -v noobs | head -1)
B2=$(ls -t RL_outputs/mpc-bench-current-noobs-*.json 2>/dev/null | head -1)
mkdir -p benchmarks
[ -n "$B1" ] && cp "$B1" "benchmarks/lambdagp-$LABEL-with-obstacles.json"
[ -n "$B2" ] && cp "$B2" "benchmarks/lambdagp-$LABEL-no-obstacles.json"
SUM1=$(python3 -c "import json;d=json.load(open('benchmarks/lambdagp-$LABEL-with-obstacles.json'));print(d['convergence_rate'],'conv,',d['collisions'],'collisions')" 2>/dev/null || echo n/a)
SUM2=$(python3 -c "import json;d=json.load(open('benchmarks/lambdagp-$LABEL-no-obstacles.json'));print(d['convergence_rate'],'conv')" 2>/dev/null || echo n/a)
say "with-obstacles: $SUM1 | obstacle-free: $SUM2"

git add teach_loader.py loader_navigation_rl/loader_goal_env.py benchmarks/ loader_critic loader_actor 2>/dev/null
git commit -m "Experiment B: lambda_gp=1e-4 gradient penalty (repo config, batch 120k)

Paper Sec. IV-D attributes MPC optimization-landscape smoothness to the
critic gradient penalty; repo ships lambda_gp=0. This run enables it
(1e-4) with the repo cost/network otherwise. Batch 120k: the penalty's
create_graph double-backward plus ~1GB fixed CUDA overhead exceed 6GB
at larger batches (200k/160k OOMed on the first update).

Benchmarks (20 scenarios, slack=10000):
- with obstacles: $SUM1
- obstacle-free:  $SUM2
Baselines: repo-config-no-penalty 5% (obstacles) / 60% (obstacle-free)." >>"$LOG" 2>&1
say "committed: $(git log --oneline -1)"

git push fork fix/training-correctness-and-mpc-compat >>"$LOG" 2>&1 &&
  say "pushed to fork" || say "PUSH FAILED"
say "pipeline done"
