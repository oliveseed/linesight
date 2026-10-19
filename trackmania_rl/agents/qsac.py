"""
In this file, we define:
    - The SACPolicy class, a TanhGaussian policy over continuous (steer, throttle, brake) actions.
    - The QuantileCritic class, implementing the QRSAC quantile critic (twin instances zf1/zf2).
    - The Trainer class, which implements the QRSAC training logic in method train_on_batch.
    - The Inferer class, which implements utilities for action selection with and without exploration.

QRSAC is "Quantile Regression Soft Actor Critic", from the Grand Turismo Sophy paper:
    "Outracing champion Gran Turismo drivers with deep reinforcement learning" (Wurman et al., Nature 2022)
Reference PyTorch implementation: https://github.com/shilpa2301/QRSAC (we discard its reference
implementation bugs: 200 gradient steps per batch, per-index n-step loops, critic picking logic).
"""

import math
import typing
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import torch
from torchrl.data import ReplayBuffer

from config_files import config_copy
from trackmania_rl import utilities

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Action vector layout: [steer ∈ [-1, 1], throttle ∈ [0, 1], brake ∈ [0, 1]]
ACTION_DIM = 3
IDX_STEER = 0
IDX_THROTTLE = 1
IDX_BRAKE = 2
DEFAULT_ACTION = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)  # steer 0, full throttle, no brake

LOG_STD_BOUNDS = (-20.0, 2.0)
LOG_2PI = math.log(2.0 * math.pi)


def _make_img_head() -> torch.nn.Module:
    """Conv head for images. Shared structure with the old IQN trunk."""
    img_head_channels = [1, 16, 32, 64, 32]
    activation_function = torch.nn.LeakyReLU
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=img_head_channels[0], out_channels=img_head_channels[1], kernel_size=(4, 4), stride=2),
        activation_function(inplace=True),
        torch.nn.Conv2d(in_channels=img_head_channels[1], out_channels=img_head_channels[2], kernel_size=(4, 4), stride=2),
        activation_function(inplace=True),
        torch.nn.Conv2d(in_channels=img_head_channels[2], out_channels=img_head_channels[3], kernel_size=(3, 3), stride=2),
        activation_function(inplace=True),
        torch.nn.Conv2d(in_channels=img_head_channels[3], out_channels=img_head_channels[4], kernel_size=(3, 3), stride=1),
        activation_function(inplace=True),
        torch.nn.Flatten(),
    )


def _make_float_feature_extractor(float_inputs_dim: int, float_hidden_dim: int) -> torch.nn.Module:
    activation_function = torch.nn.LeakyReLU
    return torch.nn.Sequential(
        torch.nn.Linear(float_inputs_dim, float_hidden_dim),
        activation_function(inplace=True),
        torch.nn.Linear(float_hidden_dim, float_hidden_dim),
        activation_function(inplace=True),
    )


class SACPolicy(torch.nn.Module):
    """TanhGaussian policy over continuous actions.

    forward() returns (mu, log_std). Sampling/squashing is handled *outside* this module
    (in Trainer/Inferer) so that torch.jit.script can compile the module on Windows.
    The action mapping is:
        steer    = tanh(u_0)
        throttle = (tanh(u_1) + 1) / 2
        brake    = (tanh(u_2) + 1) / 2
    """

    def __init__(
        self,
        float_inputs_dim: int,
        float_hidden_dim: int,
        conv_head_output_dim: int,
        dense_hidden_dimension: int,
        action_dim: int,
        float_inputs_mean: npt.NDArray,
        float_inputs_std: npt.NDArray,
    ):
        super().__init__()
        self.img_head = _make_img_head()
        self.float_feature_extractor = _make_float_feature_extractor(float_inputs_dim, float_hidden_dim)
        dense_input_dimension = conv_head_output_dim + float_hidden_dim
        activation_function = torch.nn.LeakyReLU
        self.mu_head = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension, dense_hidden_dimension // 2),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, action_dim),
        )
        self.log_std_head = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension, dense_hidden_dimension // 2),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, action_dim),
        )
        self.initialize_weights()

        self.action_dim = action_dim
        # Plain attributes (not buffers, not parameters) so that weight decay never touches them.
        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

    def initialize_weights(self):
        activation_gain = torch.nn.init.calculate_gain("leaky_relu")
        for module in [self.img_head, self.float_feature_extractor, self.mu_head[:-1]]:
            for m in module:
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                    utilities.init_orthogonal(m, activation_gain)
        utilities.init_orthogonal(self.mu_head[-1])
        # Start near a constant moderate std (e^{-1} ≈ 0.37): zero weights, -1 bias.
        torch.nn.init.zeros_(self.log_std_head[-1].weight)
        torch.nn.init.constant_(self.log_std_head[-1].bias, -1.0)

    def forward(self, img: torch.Tensor, float_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_outputs = self.img_head(img)
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        concat = torch.cat((img_outputs, float_outputs), 1)  # (batch_size, dense_input_dimension)
        mu = self.mu_head(concat)  # (batch_size, action_dim)
        log_std = self.log_std_head(concat)  # (batch_size, action_dim)
        return mu, log_std

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self


class QuantileCritic(torch.nn.Module):
    """QRSAC quantile critic: predicts the distribution (per-quantile) of the mini-race return,
    conditioned on (state, action).

    Follows DSAC/QRSAC QuantileMlp: cos quantile embedding -> Sigmoid gate -> Hadamard mix with
    state-action features -> merge layer -> one value per quantile.
    """

    def __init__(
        self,
        float_inputs_dim: int,
        float_hidden_dim: int,
        conv_head_output_dim: int,
        dense_hidden_dimension: int,
        action_dim: int,
        n_quantiles: int,
        quantile_embedding_dimension: int,
        float_inputs_mean: npt.NDArray,
        float_inputs_std: npt.NDArray,
    ):
        super().__init__()
        self.img_head = _make_img_head()
        self.float_feature_extractor = _make_float_feature_extractor(float_inputs_dim, float_hidden_dim)
        dense_input_dimension = conv_head_output_dim + float_hidden_dim
        activation_function = torch.nn.LeakyReLU
        self.base_fc = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension + action_dim, dense_hidden_dimension // 2),
            activation_function(inplace=True),
        )
        self.tau_fc = torch.nn.Sequential(
            torch.nn.Linear(quantile_embedding_dimension, dense_hidden_dimension // 2),
            torch.nn.LayerNorm(dense_hidden_dimension // 2),
            torch.nn.Sigmoid(),
        )
        self.merge_fc = torch.nn.Sequential(
            torch.nn.Linear(dense_hidden_dimension // 2, dense_hidden_dimension // 2),
            activation_function(inplace=True),
        )
        self.last_fc = torch.nn.Linear(dense_hidden_dimension // 2, 1)
        self.initialize_weights()

        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.quantile_embedding_dimension = quantile_embedding_dimension
        self.cos_bands = torch.arange(1, quantile_embedding_dimension + 1, dtype=torch.float32).to("cuda")  # (embedding_dim, )
        # Plain attributes (not buffers, not parameters) so that weight decay never touches them.
        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

    def initialize_weights(self):
        activation_gain = torch.nn.init.calculate_gain("leaky_relu")
        for module in [self.img_head, self.float_feature_extractor, self.base_fc, self.merge_fc]:
            for m in module:
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                    utilities.init_orthogonal(m, activation_gain)
        utilities.init_orthogonal(self.tau_fc[0], np.sqrt(2) * activation_gain)
        utilities.init_orthogonal(self.last_fc)

    def forward(self, img: torch.Tensor, float_inputs: torch.Tensor, action: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img:          (batch_size, 1, H, W)
            float_inputs: (batch_size, float_input_dim)
            action:       (batch_size, action_dim)
            tau:          (batch_size, n_quantiles) quantile fractions
        Returns:
            z:            (batch_size, n_quantiles) quantile values
        """
        img.shape[0]
        img_outputs = self.img_head(img)
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        state_features = torch.cat((img_outputs, float_outputs, action), 1)  # (batch_size, dense_input_dimension + action_dim)
        h = self.base_fc(state_features)  # (batch_size, dense_hidden//2)

        x = torch.cos(tau.unsqueeze(-1) * self.cos_bands * math.pi)  # (batch_size, n_quantiles, quantile_embedding_dimension)
        x = self.tau_fc(x)  # (batch_size, n_quantiles, dense_hidden//2)

        h = x * h.unsqueeze(1)  # (batch_size, n_quantiles, dense_hidden//2)
        h = self.merge_fc(h)  # (batch_size, n_quantiles, dense_hidden//2)
        z = self.last_fc(h).squeeze(-1)  # (batch_size, n_quantiles)
        return z

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self


class Trainer:
    """QRSAC trainer. One update step (critic, policy, alpha) per batch of transitions.

    Hmm, note on the batch: transitions come from torchrl buffers with the mini-race logic
    already applied by buffer_collate_function. The batch tuple is identical to what the IQN
    trainer consumed:
        (state_img, state_float, action, rewards, next_state_img, next_state_float, gammas_terminal)
    where `action` is now a (batch_size, 3) float tensor.
    """

    __slots__ = (
        "alpha_optimizer",
        "batch_size",
        "critic1",
        "critic2",
        "critic_optimizer",
        "diagnostics",
        "log_alpha",
        "n_quantiles",
        "policy",
        "policy_optimizer",
        "scaler",
        "target_critic1",
        "target_critic2",
        "target_policy",
    )

    def __init__(
        self,
        policy: SACPolicy,
        target_policy: SACPolicy,
        critic1: QuantileCritic,
        critic2: QuantileCritic,
        target_critic1: QuantileCritic,
        target_critic2: QuantileCritic,
        policy_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        alpha_optimizer: torch.optim.Optimizer,
        log_alpha: torch.Tensor,
        scaler: torch.amp.GradScaler,
        batch_size: int,
        n_quantiles: int,
    ):
        self.policy = policy
        self.target_policy = target_policy
        self.critic1 = critic1
        self.critic2 = critic2
        self.target_critic1 = target_critic1
        self.target_critic2 = target_critic2
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.log_alpha = log_alpha
        self.scaler = scaler
        self.batch_size = batch_size
        self.n_quantiles = n_quantiles
        self.diagnostics: defaultdict[str, typing.Any] = defaultdict(float)

    @property
    def alpha(self) -> float:
        with torch.no_grad():
            return self.log_alpha.exp().item()

    def _sample_tau(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """DSAC-style 'iqn' tau sampling: sampled uniformly in [0.1, 1.1] then normalized to sum 1,
        which biases tau away from 0/1 extremes.

        Returns:
            tau: (batch_size, n_quantiles) cumulative quantile fractions
            tau_hat: (batch_size, n_quantiles) midpoints where the critic is evaluated
        """
        presum_tau = torch.rand(batch_size, self.n_quantiles, device="cuda") + 0.1
        presum_tau /= presum_tau.sum(dim=-1, keepdim=True)
        tau = torch.cumsum(presum_tau, dim=1)
        with torch.no_grad():
            tau_hat = torch.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.0
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.0
        return tau, tau_hat

    def _sample_policy(self, net: SACPolicy, img: torch.Tensor, float_inputs: torch.Tensor, deterministic: bool):
        """Draw action(s) from the (tanh-gaussian) policy and compute log_pi with the tanh correction factor."""
        mu, log_std_raw = net(img, float_inputs)
        std = log_std_raw.clamp(*LOG_STD_BOUNDS).exp()
        if deterministic:
            u = mu
        else:
            u = mu + std * torch.randn_like(mu)
        t = torch.tanh(u)
        # Jacobian of the action mapping:
        #   d steer / d u  = (1 - t^2)
        #   d throttle / d u = 0.5 * (1 - t^2)   (same for brake)
        log_jacobian = (torch.log((1.0 - t**2).clamp(min=1e-6)) + torch.tensor([0.0, math.log(0.5), math.log(0.5)], device=u.device)).sum(
            -1
        )
        gaussian_log_prob = -0.5 * ((u - mu) ** 2 / std**2 + 2 * log_std_raw.clamp(*LOG_STD_BOUNDS) + LOG_2PI).sum(-1)
        log_pi = gaussian_log_prob - log_jacobian

        action = torch.stack(
            (t[:, IDX_STEER], 0.5 * (t[:, IDX_THROTTLE] + 1.0), 0.5 * (t[:, IDX_BRAKE] + 1.0)), dim=-1
        )  # (batch_size, action_dim)
        return action, log_pi

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool):
        """
        Implements one iteration of the training loop:
            1) Sample a batch of transitions from the replay buffer
            2) Compute the QRSAC targets (twin quantile critics on target policy)
            3) Update critic (quantile regression), policy (max-min-Q + entropy), and alpha (auto-tune)
            4) Update replay buffer priorities with the absolute TD error

        Returns:
            total_loss: a float
            grad_norm: a float
        """
        self.policy_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.alpha_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            batch, batch_info = buffer.sample(self.batch_size, return_info=True)
            (
                state_img_tensor,
                state_float_tensor,
                action,
                rewards,
                next_state_img_tensor,
                next_state_float_tensor,
                gammas_terminal,
            ) = batch
            if config_copy.prio_alpha > 0:
                IS_weights = torch.from_numpy(batch_info["_weight"]).to("cuda", non_blocking=True)

            # ===============================================
            #   QRSAC TARGET
            # ===============================================
            with torch.no_grad():
                _tau, tau_hat = self._sample_tau(self.batch_size)

                next_action, next_log_pi = self._sample_policy(self.target_policy, next_state_img_tensor, next_state_float_tensor, False)
                z1_next = self.target_critic1(next_state_img_tensor, next_state_float_tensor, next_action, tau_hat)
                z2_next = self.target_critic2(next_state_img_tensor, next_state_float_tensor, next_action, tau_hat)
                z_next_min = torch.minimum(z1_next, z2_next)  # (batch_size, n_quantiles)

                alpha = self.log_alpha.exp().clamp(max=1e6).detach()
                z_target = config_copy.reward_scale * rewards.unsqueeze(-1) + gammas_terminal.unsqueeze(-1) * (
                    z_next_min - alpha * next_log_pi.unsqueeze(-1)
                )  # (batch_size, n_quantiles)

            # ===============================================
            #   CRITIC UPDATE (quantile regression)
            # ===============================================
            z1_pred = self.critic1(state_img_tensor, state_float_tensor, action, tau_hat)  # (batch_size, n_quantiles)
            z2_pred = self.critic2(state_img_tensor, state_float_tensor, action, tau_hat)  # (batch_size, n_quantiles)
            z1_loss = quantile_regression_loss(z1_pred, z_target, tau_hat, config_copy.quantile_huber_kappa)  # (batch_size, )
            z2_loss = quantile_regression_loss(z2_pred, z_target, tau_hat, config_copy.quantile_huber_kappa)  # (batch_size, )
            if config_copy.prio_alpha > 0:
                critic_loss = (IS_weights * (z1_loss + z2_loss)).mean()
            else:
                critic_loss = (z1_loss + z2_loss).mean()

            # ===============================================
            #   POLICY + ALPHA UPDATE
            # ===============================================
            _tau_p, tau_hat_p = self._sample_tau(self.batch_size)
            new_action, log_pi = self._sample_policy(self.policy, state_img_tensor, state_float_tensor, False)
            z1_policy = self.critic1(state_img_tensor, state_float_tensor, new_action, tau_hat_p).mean(dim=-1)
            z2_policy = self.critic2(state_img_tensor, state_float_tensor, new_action, tau_hat_p).mean(dim=-1)
            q_policy = torch.minimum(z1_policy, z2_policy)  # (batch_size, )

            alpha_loss = -(self.log_alpha * (log_pi + config_copy.target_entropy).detach()).mean()
            policy_loss = (alpha.detach() * log_pi - q_policy).mean()

            self.diagnostics["critic_loss"] = critic_loss.item()
            self.diagnostics["policy_loss"] = policy_loss.item()
            self.diagnostics["alpha_loss"] = alpha_loss.item()
            self.diagnostics["alpha"] = self.alpha
            self.diagnostics["mean_q"] = q_policy.mean().item()
            self.diagnostics["mean_log_pi"] = log_pi.mean().item()

            total_loss = critic_loss + policy_loss + alpha_loss

            if do_learn:
                self.scaler.scale(critic_loss).backward()
                self.scaler.scale(policy_loss).backward()
                self.scaler.scale(alpha_loss).backward()

                # Gradient clipping : https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                for optimizer in (self.policy_optimizer, self.critic_optimizer, self.alpha_optimizer):
                    self.scaler.unscale_(optimizer)
                grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config_copy.clip_grad_norm)
                grad_norm_critic = torch.nn.utils.clip_grad_norm_(
                    list(self.critic1.parameters()) + list(self.critic2.parameters()), config_copy.clip_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(self.alpha_optimizer.param_groups[0]["params"], config_copy.clip_grad_norm)
                grad_norm = max(grad_norm_policy.item(), grad_norm_critic.item())

                self.scaler.step(self.policy_optimizer)
                self.scaler.step(self.critic_optimizer)
                self.scaler.step(self.alpha_optimizer)
                self.scaler.update()

                self.diagnostics["grad_norm_policy"] = grad_norm_policy.item()
                self.diagnostics["grad_norm_critic"] = grad_norm_critic.item()

                # ===============================================
                #   UPDATE BUFFER PRIORITIES
                # ===============================================
                if config_copy.prio_alpha > 0:
                    mean_z_pred = (torch.minimum(z1_pred, z2_pred)).mean(axis=1).detach()
                    mean_z_target = z_target.mean(axis=1)
                    priorities = (mean_z_pred - mean_z_target).abs()
                    mask_update_priority = state_float_tensor[:, 0] < config_copy.min_horizon_to_update_priority_actions
                    # Only update the transition priority if the transition was sampled with a sufficiently long-term horizon.
                    buffer.update_priority(
                        batch_info["index"][mask_update_priority],
                        priorities[mask_update_priority].detach().cpu().type(torch.float64),
                    )
            else:
                grad_norm = 0.0

        return total_loss.detach().item(), grad_norm


def quantile_regression_loss(z_pred: torch.Tensor, z_target: torch.Tensor, tau: torch.Tensor, kappa: float) -> torch.Tensor:
    """
    Pinball (quantile) loss with a smooth-huber profile, as in IQN / QRSAC.

    Args:
        z_pred:   (batch_size, n_quantiles) requires grad
        z_target: (batch_size, n_quantiles) detached
        tau:      (batch_size, n_quantiles) quantile fractions at which z_pred was evaluated
        kappa:    huber threshold

    Returns:
        loss: a torch.Tensor of shape (batch_size, ), mean over quantiles of the pinball loss of each sample
    """
    delta = z_target.detach() - z_pred
    abs_delta = delta.abs()
    huber = torch.where(abs_delta <= kappa, 0.5 / kappa * delta**2, abs_delta - 0.5 * kappa)
    weight = torch.where(delta < 0, 1.0 - tau, tau)
    return (weight * huber).mean(dim=-1)


class Inferer:
    __slots__ = (
        "critic",
        "inference_network",
        "is_explo",
    )

    def __init__(self, inference_network: SACPolicy | torch.jit.ScriptModule | torch.nn.Module, critic=None):
        """
        Args:
            inference_network: the (possibly scripted) policy network used for action selection
            critic: optional (possibly scripted) critic1 network used to estimate mean-Q for logging
        """
        self.inference_network = inference_network
        self.critic = critic
        self.is_explo = None

    def infer_actions(
        self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray, deterministic: bool
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Perform inference of a single state through the policy network.

        Args:
            img_inputs_uint8:   a numpy array of shape (1, H, W) and dtype np.uint8
            float_inputs:       a numpy array of shape (float_input_dim, ) and dtype np.float32
            deterministic:      if True, return tanh(mu) (greedy); if False, sample from the policy

        Returns:
            action:      a numpy array of shape (action_dim, ) dtype np.float32
            mu:          a numpy array of shape (action_dim, ) dtype np.float32
            std:         a numpy array of shape (action_dim, ) dtype np.float32
        """
        with torch.no_grad():
            state_img_tensor = (
                torch.from_numpy(img_inputs_uint8)
                .unsqueeze(0)
                .to("cuda", memory_format=torch.channels_last, non_blocking=True, dtype=torch.float32)
                - 128
            ) / 128
            state_float_tensor = torch.from_numpy(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
            mu, log_std_raw = self.inference_network(state_img_tensor, state_float_tensor)
            std = log_std_raw.clamp(*LOG_STD_BOUNDS).exp()
            if deterministic:
                u = mu
            else:
                u = mu + std * torch.randn_like(mu)
            t = torch.tanh(u)
            action = torch.stack(
                (t[:, IDX_STEER], 0.5 * (t[:, IDX_THROTTLE] + 1.0), 0.5 * (t[:, IDX_BRAKE] + 1.0)), dim=-1
            )  # (1, action_dim)
            return (
                action[0].cpu().numpy().astype(np.float32),
                mu[0].cpu().numpy().astype(np.float32),
                std[0].cpu().numpy().astype(np.float32),
            )

    def estimate_q(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray, action: npt.NDArray) -> float:
        """Mean over quantiles of critic1's estimate for the chosen action. 0.0 if no critic provided."""
        if self.critic is None:
            return 0.0
        with torch.no_grad():
            state_img_tensor = (
                torch.from_numpy(img_inputs_uint8)
                .unsqueeze(0)
                .to("cuda", memory_format=torch.channels_last, non_blocking=True, dtype=torch.float32)
                - 128
            ) / 128
            state_float_tensor = torch.from_numpy(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
            action_tensor = torch.from_numpy(np.expand_dims(action, axis=0)).to("cuda", non_blocking=True)
            tau_hat = (torch.arange(0, self.critic.n_quantiles, device="cuda", dtype=torch.float32) + 0.5).unsqueeze(
                0
            ) / self.critic.n_quantiles
            z1 = self.critic(state_img_tensor, state_float_tensor, action_tensor, tau_hat)
            return z1.mean().item()

    def get_exploration_action(
        self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray
    ) -> tuple[npt.NDArray, bool, float, npt.NDArray]:
        """
        Selects an action for the rollout.
        Exploration: during exploitation rollouts (is_explo=True), actions are sampled from the
        TanhGaussian policy (the entropy mechanism replaces epsilon-greedy / Boltzmann exploration).
        During evaluation rollouts (is_explo=False), the deterministic tanh(mu) action is returned.

        Args:
            img_inputs_uint8:   a numpy array of shape (1, H, W) and dtype np.uint8
            float_inputs:       a numpy array of shape (float_input_dim, ) and dtype np.float32

        Returns:
            action:             a numpy array of shape (action_dim, ), the action to send to the vehicle
            is_greedy:          a bool, True when the action equals the deterministic action
            q_value:            a float, critic1's mean-Z estimate for the chosen action (0 if no critic)
            action:             the action again, for per-component logging in the caller
        """
        deterministic = not self.is_explo
        action, _, _ = self.infer_actions(img_inputs_uint8, float_inputs, deterministic)
        q_value = self.estimate_q(img_inputs_uint8, float_inputs, action)
        # is_greedy: under a stochastic policy every sample *is* a policy action, so we report True.
        # This flag only matters if config_copy.discard_non_greedy_actions_in_nsteps (not recommended with QRSAC).
        return action, True, q_value, action


# ------------------------------------------------------------------
# Factories
# ------------------------------------------------------------------


def make_policy(jit: bool, is_inference: bool):
    """Constructs the TanhGaussian policy network.

    Returns:
        (possibly compiled model, uncompiled model). The uncompiled model is the one whose
        state_dict is exchanged between processes and saved on disk.
    """
    uncompiled_model = SACPolicy(
        float_inputs_dim=config_copy.float_input_dim,
        float_hidden_dim=config_copy.float_hidden_dim,
        conv_head_output_dim=config_copy.conv_head_output_dim,
        dense_hidden_dimension=config_copy.dense_hidden_dimension,
        action_dim=ACTION_DIM,
        float_inputs_mean=config_copy.float_inputs_mean,
        float_inputs_std=config_copy.float_inputs_std,
    )
    if jit:
        if config_copy.is_linux:
            compile_mode = None if "rocm" in torch.__version__ else ("max-autotune" if is_inference else "max-autotune-no-cudagraphs")
            model = torch.compile(uncompiled_model, dynamic=False, mode=compile_mode)
        else:
            model = torch.jit.script(uncompiled_model)
    else:
        model = uncompiled_model
    return (
        model.to(device="cuda", memory_format=torch.channels_last).train(),
        uncompiled_model.to(device="cuda", memory_format=torch.channels_last).train(),
    )


def make_critic(jit: bool):
    """Constructs one quantile critic network (compiled wrapper + uncompiled module)."""
    uncompiled_model = QuantileCritic(
        float_inputs_dim=config_copy.float_input_dim,
        float_hidden_dim=config_copy.float_hidden_dim,
        conv_head_output_dim=config_copy.conv_head_output_dim,
        dense_hidden_dimension=config_copy.dense_hidden_dimension,
        action_dim=ACTION_DIM,
        n_quantiles=config_copy.n_quantiles,
        quantile_embedding_dimension=config_copy.quantile_embedding_dimension,
        float_inputs_mean=config_copy.float_inputs_mean,
        float_inputs_std=config_copy.float_inputs_std,
    )
    if jit:
        if config_copy.is_linux:
            compile_mode = None if "rocm" in torch.__version__ else "max-autotune-no-cudagraphs"
            model = torch.compile(uncompiled_model, dynamic=False, mode=compile_mode)
        else:
            model = torch.jit.script(uncompiled_model)
    else:
        model = uncompiled_model
    return model.to(device="cuda", memory_format=torch.channels_last).train(), uncompiled_model.to(
        device="cuda", memory_format=torch.channels_last
    ).train()


def make_untrained_qsac_networks(jit: bool, is_inference: bool):
    """Constructs the six QRSAC networks: policy + twin critics + their target copies.

    Returns:
        (policy, uncompiled_policy, critic1, critic2, target_policy, target_critic1, target_critic2)
    """
    policy, uncompiled_policy = make_policy(jit, is_inference)
    critic1, uncompiled_critic1 = make_critic(jit)
    critic2, uncompiled_critic2 = make_critic(jit)
    target_policy, _ = make_policy(False, False)
    target_critic1, _ = make_critic(False)
    target_critic2, _ = make_critic(False)
    # Copy the online weights into the targets so loading a checkpoint then behaves consistently.
    with torch.no_grad():
        target_policy.load_state_dict(uncompiled_policy.state_dict())
        target_critic1.load_state_dict(uncompiled_critic1.state_dict())
        target_critic2.load_state_dict(uncompiled_critic2.state_dict())
    return (
        policy,
        uncompiled_policy,
        critic1,
        critic2,
        target_policy,
        target_critic1,
        target_critic2,
    )
