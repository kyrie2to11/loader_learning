from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    get_parameters_by_name,
    get_schedule_fn,
    polyak_update,
    update_learning_rate,
)
from torch.nn import functional as F

from loader_navigation_rl.alac.policies import Actor, ALACPolicy, MultiInputPolicy
from loader_navigation_rl.alac.utils import SquaredContinuousCritic
from loader_navigation_rl.utils import compute_gradient_penalty

SelfALAC = TypeVar("SelfALAC", bound="ALAC")


class ALAC(OffPolicyAlgorithm):
    """
    Adaptive Lyapunov-based Actor Critic

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for the Lagrange optimizers and the actor/critic defaults
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param lambda_gp (float, optional): The weight of the gradient penalty term in the critic loss function (default 0.0)
    :param actor_learning_rate: optional actor-specific learning rate or schedule
    :param critic_learning_rate: optional critic-specific learning rate or schedule
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MultiInputPolicy": MultiInputPolicy,
    }

    policy: ALACPolicy
    actor: Actor
    critic: SquaredContinuousCritic
    critic_target: SquaredContinuousCritic

    def __init__(
        self,
        policy: str | type[ALACPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 1,
        target_entropy: str | float = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        lambda_gp: float = 0.0,
        finetune: bool = False,
        actor_learning_rate: float | Schedule | None = None,
        critic_learning_rate: float | Schedule | None = None,
        lagrange_lr: float = 3e-4,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_update_interval = target_update_interval
        self.target_entropy = target_entropy
        self.lambda_gp = lambda_gp
        self.finetune = finetune
        self.actor_lr_schedule = get_schedule_fn(
            learning_rate if actor_learning_rate is None else actor_learning_rate
        )
        self.critic_lr_schedule = get_schedule_fn(
            learning_rate if critic_learning_rate is None else critic_learning_rate
        )

        # Optimizers for the lyapunov loss lagrange variables:
        # 官方 ALAC (Wang et al.) 用独立的小学习率(3e-4)和普通 Adam:
        # - 共享 lr(1e-2)时 Adam 抖动 ~0.01/步,远大于平衡点附近的梯度信号,
        #   λl 会被顶死在上界 clamp=1.0,永远到不了论文的 ~0.8
        # - AdamW 的 weight_decay 恒把 log_llambda 往 0(即 λl=1)拉
        self.lagrange_lr = lagrange_lr
        self.log_beta = th.tensor([np.log(2.0)], device=self.device).requires_grad_(True)
        self.beta_optimizer: th.optim.Adam | None = None
        self.log_llambda = th.tensor([0.0], device=self.device).requires_grad_(True)
        self.llambda_optimizer: th.optim.Adam | None = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.critic_target, ["running_"]
        )

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(
                -np.prod(self.env.action_space.shape).astype(np.float32)
            )  # type: ignore
            print("----------------------------------")
            print(self.target_entropy)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        self.llambda_optimizer = th.optim.Adam(
            [self.log_llambda], lr=self.lagrange_lr
        )
        self.beta_optimizer = th.optim.Adam([self.log_beta], lr=self.lagrange_lr)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        update_learning_rate(
            self.actor.optimizer,
            self.actor_lr_schedule(self._current_progress_remaining),
        )
        update_learning_rate(
            self.critic.optimizer,
            self.critic_lr_schedule(self._current_progress_remaining),
        )

        actor_losses, critic_losses = [], []
        llambda_losses, beta_losses = [], []
        assert self.beta_optimizer is not None
        assert self.llambda_optimizer is not None
        penalty = None
        k = lam = np.nan

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # Select action according to policy
            next_actions_pi, next_log_prob = self.actor.action_log_prob(
                replay_data.next_observations
            )
            # Compute the next L values

            L_values = th.cat(
                self.critic_target(replay_data.observations, replay_data.actions), dim=1
            )
            L_values, _ = th.max(L_values, dim=1, keepdim=True)
            next_L_values = th.cat(
                self.critic_target(replay_data.next_observations, next_actions_pi),
                dim=1,
            )
            next_L_values, _ = th.max(next_L_values, dim=1, keepdim=True)

            # td error
            target_L_values = (
                -replay_data.rewards
                + (1 - replay_data.dones) * self.gamma * next_L_values.detach()
            )
            # Get current L-values estimates for each critic network
            # using action from the replay buffer
            current_L_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            critic_loss = (
                0.5
                * th.stack(
                    [
                        F.mse_loss(current_L, target_L_values)
                        for current_L in current_L_values
                    ]
                ).sum()
            )
            if self.lambda_gp > 0.0:
                penalty = th.as_tensor(
                    compute_gradient_penalty(
                        self.critic,
                        replay_data.observations,
                        replay_data.actions,
                        lambda_gp=self.lambda_gp,
                    )
                )
                critic_loss += penalty
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Ensure positive lagrange multipliers:
            beta = th.exp(self.log_beta)
            llambda = th.exp(self.log_llambda)

            k = 1 - llambda.detach().item()
            lam = min(llambda.detach().item(), self.gamma)
            delta_L = (
                next_L_values
                - L_values.detach()
                + k * (L_values.detach() - lam * next_L_values)
            )

            # Compute actor loss, i.e. solve the inner min problem of the lagrangian:
            actor_loss = th.mean(beta.detach() * log_prob + llambda.detach() * delta_L)
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            actor_losses.append(actor_loss.item())

            beta_loss = -beta * th.mean(self.target_entropy + log_prob.detach())
            self.beta_optimizer.zero_grad()
            beta_loss.backward()
            self.beta_optimizer.step()
            beta_losses.append(beta_loss.item())

            llambda_loss = -llambda * th.mean(delta_L.detach())
            self.llambda_optimizer.zero_grad()
            llambda_loss.backward()
            self.llambda_optimizer.step()
            llambda_losses.append(llambda_loss.item())

            # Early on during the training there will be large violations of the lyapunov constraint due to a bad policy,
            # therefore it is wise to clamp the associated lagrange multiplier from above, so it doesn't dominate the gradients:
            with th.no_grad():
                self.log_beta.clamp_(max=np.log(1))
                self.log_llambda.clamp_(max=np.log(1))

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/beta_loss", np.mean(beta_losses))
        self.logger.record("train/lambda_loss", np.mean(llambda_losses))
        self.logger.record("train/beta", th.exp(self.log_beta).cpu().detach().item())
        self.logger.record(
            "train/lambda", th.exp(self.log_llambda).cpu().detach().item()
        )
        if penalty is not None:
            self.logger.record("train/penalty", penalty.cpu().detach().item())
        self.logger.record("train/k", k)
        self.logger.record("train/lam", lam)

    def learn(
        self: SelfALAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "ALAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfALAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = [
            "policy",
            "actor.optimizer",
            "critic.optimizer",
            "beta_optimizer",
            "llambda_optimizer",
        ]
        saved_pytorch_variables = ["log_beta", "log_llambda"]
        return state_dicts, saved_pytorch_variables

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if (
            self.num_timesteps < learning_starts
            and not (self.use_sde and self.use_sde_at_warmup)
            and not self.finetune
        ):
            # Warmup phase
            unscaled_action = np.array(
                [self.action_space.sample() for _ in range(n_envs)]
            )
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
