"""Headless MPC obstacle-avoidance benchmark.

Runs the same MPCActor as the interactive demo (N=20, 3 obstacles) against
fixed goal+obstacle scenarios, counts success (position < 0.1 m within the
time limit, using the paper's convergence criterion), and records the
minimum distance to each obstacle. No Qt, no GPU.

Run from the repo root with the demo venv:
    ACADOS_SOURCE_DIR=/home/jarvis/projects/acados \
    LD_LIBRARY_PATH=/home/jarvis/projects/acados/lib \
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
    .venv-demo/bin/python benchmark_mpc.py [critic_path]
"""

import json
import os
import sys
import time

import numpy as np
import torch

from mpc_solvers.acados_sqp_solver import AcadosSQPSolver
from test_loader_mpc import MPCActor

MACHINE_RADIUS = 0.8
HORIZON_S = 30.0
DT = 0.2
N_STEPS = int(HORIZON_S / DT)

# Fixed scenarios: (goal [x, y, theta], obstacles as (x, y, r) list).
# Obstacles sit between the origin (start pose) and the goal, forcing a detour.
SCENARIOS = [
    (np.array([8.0, 0.0, 0.0]), [(4.0, 0.0, 1.0)]),
    (np.array([-8.0, 0.0, np.pi]), [(-4.0, 0.0, 1.0)]),
    (np.array([0.0, 8.0, np.pi / 2]), [(0.0, 4.0, 1.0)]),
    (np.array([0.0, -8.0, -np.pi / 2]), [(0.0, -4.0, 1.0)]),
    (np.array([6.0, 6.0, np.pi / 4]), [(3.0, 3.0, 1.5)]),
    (np.array([-6.0, 6.0, 3 * np.pi / 4]), [(-3.0, 3.0, 1.5)]),
    (np.array([6.0, -6.0, -np.pi / 4]), [(3.0, -3.0, 1.5)]),
    (np.array([-6.0, -6.0, -3 * np.pi / 4]), [(-3.0, -3.0, 1.5)]),
    (np.array([7.0, 3.0, 0.3]), [(3.5, 1.5, 1.0), (5.5, 2.5, 0.7)]),
    (np.array([3.0, 7.0, 1.2]), [(1.5, 3.5, 1.0), (2.5, 5.5, 0.7)]),
    (np.array([8.0, 0.0, np.pi / 2]), [(3.0, 0.0, 1.0), (6.0, 0.0, 1.0)]),
    (np.array([0.0, 9.0, 0.0]), [(0.0, 3.0, 1.2), (0.0, 6.0, 1.2)]),
    (np.array([7.0, -3.0, -0.5]), [(3.0, -1.0, 1.2)]),
    (np.array([-7.0, 3.0, 2.5]), [(-3.0, 1.5, 1.2)]),
    (np.array([5.0, 8.0, 1.9]), [(2.0, 4.5, 1.0)]),
    (np.array([-5.0, -8.0, -1.5]), [(-2.5, -4.0, 1.0)]),
    (np.array([9.0, 1.0, 0.0]), [(4.5, 0.2, 0.8), (7.0, -0.5, 0.6)]),
    (np.array([1.0, 9.0, np.pi / 2]), [(0.2, 4.5, 0.8), (-0.5, 7.0, 0.6)]),
    (np.array([6.0, 6.0, -np.pi / 4]), [(2.0, 5.0, 0.9), (5.0, 2.0, 0.9)]),
    (np.array([-6.0, 6.0, -3 * np.pi / 4]), [(-5.0, 2.0, 0.9), (-2.0, 5.0, 0.9)]),
]


def run_scenario(actor: MPCActor, goal: np.ndarray, obstacles: list) -> dict:
    x = np.zeros(6)
    pad = np.array([[1e3, 1e3, 0]] * (3 - len(obstacles)))
    obs = np.vstack([obstacles, pad])
    min_clear = np.inf
    converged_t = None
    lyap = float("nan")
    for step in range(N_STEPS):
        _, horizon, solve_ms, lyap = actor.solve(x, goal, obs)  # noqa: F841 (solve_ms unused)
        x = horizon[1]
        for ox, oy, orad in obstacles:
            clear = np.hypot(x[0] - ox, x[1] - oy) - orad - MACHINE_RADIUS
            min_clear = min(min_clear, clear)
        if converged_t is None and np.hypot(x[0] - goal[0], x[1] - goal[1]) < 0.1:
            converged_t = (step + 1) * DT
            break
    final_pos = float(np.hypot(x[0] - goal[0], x[1] - goal[1]))
    return {
        "converged": converged_t is not None,
        "converge_s": converged_t,
        "final_pos_m": round(final_pos, 3),
        "min_obstacle_clearance_m": round(float(min_clear), 3),
        "collision": bool(min_clear < 0.0),
        "lyapunov_final": round(float(lyap), 3),
    }


def main() -> None:
    critic_path = sys.argv[1] if len(sys.argv) > 1 else "loader_critic"
    t0 = time.monotonic()
    actor = MPCActor(AcadosSQPSolver, mpc_n=20, num_obstacles=3)
    build_s = round(time.monotonic() - t0, 1)

    results = [run_scenario(actor, g, o) for g, o in SCENARIOS]
    n = len(results)
    converged = sum(r["converged"] for r in results)
    collided = sum(r["collision"] for r in results)
    times = [r["converge_s"] for r in results if r["converged"]]
    report = {
        "critic_path": critic_path,
        "critic_mtime": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(critic_path))
        ),
        "scenarios": n,
        "converged": converged,
        "convergence_rate": round(converged / n, 3),
        "collisions": collided,
        "collision_free_rate": round((n - collided) / n, 3),
        "converge_time_mean_s": round(float(np.mean(times)), 2) if times else None,
        "converge_time_median_s": round(float(np.median(times)), 2) if times else None,
        "min_clearance_overall_m": round(
            min(r["min_obstacle_clearance_m"] for r in results), 3
        ),
        "solver_build_s": build_s,
        "results": results,
    }
    tag = critic_path.replace("loader_critic", "current").replace("/", "_")
    out = f"RL_outputs/mpc-bench-{tag}-{time.strftime('%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
