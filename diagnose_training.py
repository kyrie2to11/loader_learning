"""Regression checks for vector termination flags and curriculum constraints.

Run: SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python diagnose_training.py
"""

import json

import numpy as np

from loader_navigation_rl.loader_goal_env import LoaderGoalEnv

if __name__ == "__main__":
    env = LoaderGoalEnv(4, 0.2, 25, "cpu")
    weights = {
        "pos": 10,
        "hdg": 0,
        "beta": 0,
        "dot_beta": 0,
        "lin_vel": 0,
        "pos_beta": 0,
        "pos_dot_beta": 0,
    }
    env.set_cost_weights(weights)
    env.reset()
    env.states.zero_()
    env.goals.zero_()
    env.goals[[0, 2, 3], 0] = 10
    env.num_steps.zero_()
    env.num_steps[2] = 125
    _, _, dones, infos = env.step(np.zeros((4, 2), dtype=np.float32))
    actual = [
        {
            "env": i.item(),
            "success": bool(infos[i]["is_success"]),
            "timeout": bool(infos[i]["TimeLimit.truncated"]),
        }
        for i in np.flatnonzero(dones)
    ]
    expected = [
        {"env": 1, "success": True, "timeout": False},
        {"env": 2, "success": False, "timeout": True},
    ]
    assert actual == expected
    # SB3 masks time limits before passing dones to the critic target.
    recorded_terminal = bool(dones[2] and not infos[2]["TimeLimit.truncated"])
    assert not recorded_terminal
    print(
        json.dumps(
            {
                "done_flags": actual,
                "expected": expected,
                "timeout_wrongly_terminal": recorded_terminal,
            }
        )
    )

    # Stage 4: angular-rate constraint active, linear-speed constraint inactive.
    env.set_cost_weights(dict(weights, dot_beta=1))
    desired = np.zeros((2, 7), dtype=np.float32)
    desired[:, 3] = 1
    achieved = desired.copy()
    achieved[0, 5] = np.deg2rad(10)  # Angular velocity should prevent success.
    achieved[1, 6] = 0.5  # Linear velocity alone should NOT prevent success yet.
    reward = np.asarray(env.compute_reward(achieved, desired, [{}, {}]))
    assert reward[0] < 0
    assert reward[1] == 0
    print(
        json.dumps(
            {
                "stage4_rewards": reward.tolist(),
                "angular_rate_violation_declared_success": bool(reward[0] == 0),
            }
        )
    )
    # Training weights are defined in SI units: 0.2 m and 10 degrees
    # should contribute equally before the common reward transform.
    env.set_cost_weights({**weights, "hdg": 1 / np.deg2rad(5)})
    desired = np.zeros((2, 7), dtype=np.float32)
    desired[:, 3] = 1
    achieved = desired.copy()
    achieved[0, 0] = 0.2
    achieved[1, 2] = np.sin(np.deg2rad(10))
    achieved[1, 3] = np.cos(np.deg2rad(10))
    reward = np.asarray(env.compute_reward(achieved, desired, [{}, {}]))
    assert np.allclose(reward[0], reward[1])
    print(json.dumps({"equal_si_weight_rewards": reward.tolist()}))
    env.close()
