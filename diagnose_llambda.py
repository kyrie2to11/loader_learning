"""λl 收敛动力学冒烟测试(小规模、单课程阶段)。

对比两种乘子优化器配置下 λl 的轨迹:
  A: lagrange_lr=1e-2(旧:与主网络共享学习率)
  B: lagrange_lr=3e-4(官方 ALAC 值)

用法:
  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  .venv-train/bin/python diagnose_llambda.py [总步数,默认40000]
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback

from loader_navigation_rl.alac.alac import ALAC
from loader_navigation_rl.loader_goal_env import LoaderGoalEnv
from loader_navigation_rl.utils import CustomCombinedExtractor

STAGE1_W = {
    "pos": 1 / 0.1, "hdg": 0, "beta": 0, "dot_beta": 0,
    "lin_vel": 0, "pos_beta": 0, "pos_dot_beta": 0,
}


class LambdaRecorder(BaseCallback):
    def __init__(self, every=2000):
        super().__init__()
        self.every = every
        self.traj = []

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            self.traj.append((
                self.num_timesteps,
                float(np.exp(self.model.log_llambda.item())),
                float(np.exp(self.model.log_beta.item())),
            ))
        return True


def run(tag: str, lagrange_lr: float, total_steps: int):
    env = LoaderGoalEnv(100, 0.2, 25, "cuda:0")
    env.set_cost_weights(STAGE1_W)
    model = ALAC(
        "MultiInputPolicy", env,
        replay_buffer_class=DictReplayBuffer, verbose=0,
        buffer_size=200_000, learning_starts=10_000, learning_rate=1e-2,
        gradient_steps=3, train_freq=1, gamma=0.99, batch_size=4096,
        policy_kwargs={
            "net_arch": {"pi": [96, 96, 96], "qf": [96, 96, 96]},
            "activation_fn": torch.nn.Softplus,
            "share_features_extractor": True,
            "features_extractor_class": CustomCombinedExtractor,
        },
        target_entropy=-2, tau=0.05, lambda_gp=0.0,
        lagrange_lr=lagrange_lr,
    )
    rec = LambdaRecorder()
    model.learn(total_steps, callback=rec)
    print(f"\n=== {tag} (lagrange_lr={lagrange_lr}) 步数, λl, β ===")
    for row in rec.traj:
        print(f"  {row[0]:>8d}  {row[1]:.4f}  {row[2]:.4f}")
    tail = [r[1] for r in rec.traj[-5:]]
    print(f"  末段 λl 均值: {np.mean(tail):.4f}")
    return rec.traj


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40_000
    a = run("A 旧配置", 1e-2, steps)
    b = run("B 官方配置", 3e-4, steps)
    print("\n结论:A 末段 λl 应顶在 ~1.0;B 应明显低于 A(若仍≈1,见仓库 README 训练说明)")
