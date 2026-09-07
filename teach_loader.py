import os
from datetime import datetime

import numpy as np
import torch
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder

from loader_navigation_rl.alac.alac import ALAC
from loader_navigation_rl.loader_goal_env import LoaderGoalEnv
from loader_navigation_rl.utils import CustomCombinedExtractor

now = datetime.now().strftime("%m_%d__%H_%M")
os.makedirs(f"RL_outputs/{now}", exist_ok=True)

env = LoaderGoalEnv(1000, 0.2, 25, "cuda:0")
env = VecMonitor(env)
env = VecVideoRecorder(
    env,
    "./RL_outputs/video",
    record_video_trigger=lambda x: (x > 1000) and (x % (50 * 151) == 0),
    video_length=5 * 151,
)

model = ALAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=DictReplayBuffer,
    verbose=1,
    buffer_size=2_000_000,
    learning_starts=1_000_000,
    learning_rate=1e-2,
    actor_learning_rate=1e-3,
    critic_learning_rate=5e-3,
    gradient_steps=3,
    train_freq=1,
    gamma=0.99,
    batch_size=150_000,  # 6GB GPU + paper net [48,96,144,96,48] has ~1.5x the
    # activation memory of [96,96,96]; 200k OOMs on the first update.
    policy_kwargs=dict(
        net_arch=dict(
            pi=[48, 96, 144, 96, 48],  # paper Sec. V-A
            qf=[48, 96, 144, 96, 48],
        ),
        activation_fn=torch.nn.Softplus,
        share_features_extractor=True,
        features_extractor_class=CustomCombinedExtractor,
    ),
    tensorboard_log="./RL_outputs/debug",
    target_entropy=-2,
    tau=0.05,
    lambda_gp=0.0,
)

try:
    sd = torch.load("loader_critic").state_dict()
    sd2 = {}
    for k, v in sd.items():
        sd2["qf0." + k] = v
        sd2["qf1." + k] = v
    model.policy.critic.load_state_dict(sd2, strict=True)
    model.critic.load_state_dict(sd2, strict=True)
    model.policy.critic_target.load_state_dict(sd2, strict=True)
    model.critic_target.load_state_dict(sd2, strict=True)
except (FileNotFoundError, RuntimeError, AttributeError):
    print("Couldn't load critic weights")

try:
    model.policy.actor.load_state_dict(torch.load("loader_actor"))
    model.actor.load_state_dict(torch.load("loader_actor"))
except (FileNotFoundError, RuntimeError, AttributeError):
    print("Couldn't load actor weights")

pos_w = 1 / 0.1
hdg_w = 1 / np.deg2rad(5)
beta_w = 1 / np.deg2rad(5)
dot_beta_w = 1 / np.deg2rad(25)
lin_vel_w = 1
pos_beta_w = 5
pos_dot_beta_w = 10

# It makes training easier if we introduce the error terms one by one as the total steps grow, i.e.:
# position error weight: 1     1     1     1     1     1
# hdg error weight:      0     1     1     1     1     1
# beta error weight:     0     0     1     1     1     and so on...
#                        -----------------------------------------
# example total steps:   0    1M    2M    4M    8M    16M
weight_schedule = [
    (
        {
            "pos": pos_w,
            "hdg": 0,
            "beta": 0,
            "dot_beta": 0,
            "lin_vel": 0,
            "pos_beta": 0,
            "pos_dot_beta": 0,
        },
        5_000_000,
    ),
    (
        {
            "pos": pos_w,
            "hdg": hdg_w,
            "beta": 0,
            "dot_beta": 0,
            "lin_vel": 0,
            "pos_beta": 0,
            "pos_dot_beta": 0,
        },
        5_000_000,
    ),
    (
        {
            "pos": pos_w,
            "hdg": hdg_w,
            "beta": beta_w,
            "dot_beta": 0,
            "lin_vel": 0,
            "pos_beta": 0,
            "pos_dot_beta": 0,
        },
        5_000_000,
    ),
    (
        {
            "pos": pos_w,
            "hdg": hdg_w,
            "beta": beta_w,
            "dot_beta": dot_beta_w,
            "lin_vel": 0,
            "pos_beta": 0,
            "pos_dot_beta": 0,
        },
        5_000_000,
    ),
    (
        {
            "pos": pos_w,
            "hdg": hdg_w,
            "beta": beta_w,
            "dot_beta": dot_beta_w,
            "lin_vel": lin_vel_w,
            "pos_beta": 0,
            "pos_dot_beta": 0,
        },
        5_000_000,
    ),
    (
        {
            "pos": pos_w,
            "hdg": hdg_w,
            "beta": beta_w,
            "dot_beta": dot_beta_w,
            "lin_vel": lin_vel_w,
            "pos_beta": pos_beta_w,
            "pos_dot_beta": 0,
        },
        5_000_000,
    ),
    (
        {
            "pos": pos_w,
            "hdg": hdg_w,
            "beta": beta_w,
            "dot_beta": dot_beta_w,
            "lin_vel": lin_vel_w,
            "pos_beta": pos_beta_w,
            "pos_dot_beta": pos_dot_beta_w,
        },
        5_000_000,
    ),
]

total_step = 0
for stage, (w, steps) in enumerate(weight_schedule, start=1):
    env.env.set_cost_weights(w)
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=f"RL_outputs/{now}",
        name_prefix=f"stage_{stage}_model",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )
    model.learn(steps, callback=checkpoint_callback)
    total_step += steps
    llambda = float(np.exp(model.log_llambda.item()))
    beta = float(np.exp(model.log_beta.item()))
    print(
        f"[stage {stage}] {total_step}步: λl={llambda:.4f} β={beta:.4f}"
        + (
            "  ← 论文停训判据 λl≈0.8 已达成"
            if llambda <= 0.8
            else "  (目标 λl≤0.8,若长期徘徊>0.9见 diagnose_llambda.py)"
        )
    )
    if llambda <= 0.8:
        print(f"λl={llambda:.4f}≤0.8,按论文停训判据终止课程,取当前 critic")
        break
    model.save(f"RL_outputs/{now}/stage_{stage}_final.zip")
    model.save_replay_buffer(f"RL_outputs/{now}/stage_{stage}_buffer.pkl")
    model.replay_buffer.reset()  # Clear the outdated buffer with old rewards, and collect new samples in next iter
    torch.save(model.critic.q_networks[0], "loader_critic")
    torch.save(model.actor.state_dict(), "loader_actor")
