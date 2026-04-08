"""
Prior shaping utilities for Flow Matching models.

Instead of fine-tuning DiT weights (as Flow-GRPO does), we keep the DiT frozen
and optimize the noise prior distribution. Since Flow Matching is an ODE, a fixed
noise deterministically maps to a fixed image and reward.

Three update strategies:
- "reward_weighted" (default): On-policy. Uses all samples from the current epoch
  with advantage-based weighting (analogous to GRPO). Every sample contributes:
  high-reward samples pull the prior toward them, low-reward samples push it away.
- "cem": Cross-Entropy Method. Can use historical cached data. Only elite (top-k%)
  samples contribute to the update, discarding the rest.
- "particle": Non-parametric. Reward-weighted resampling from cached noise buffer
  + Gaussian perturbation. Can represent arbitrarily complex, multi-modal distributions.

reward_weighted and cem use GaussianPrior; particle uses ParticlePrior.
"""

import os
import glob
import math
import numpy as np
import torch
from typing import Tuple, Optional, Dict, List


class GaussianPrior:
    """Diagonal Gaussian prior N(mu, diag(sigma^2)) for latent noise sampling.

    Update strategies:
    - reward_weighted: on-policy, all samples contribute via advantage weighting
    - cem: elite selection from (optionally historical) samples

    Regularization modes:
    - "kl": constrain KL(p_shaped || N(0,I)) <= kl_max via binary search
    - "interpolation": mu_new = alpha * mu_target, sigma2_new = (1-alpha) + alpha * sigma2_target
    """

    def __init__(
        self,
        shape: tuple,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        regularization_mode: str = "kl",
        alpha: float = 0.1,
        kl_max: float = 5.0,
    ):
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.regularization_mode = regularization_mode
        self.alpha = alpha
        self.kl_max = kl_max

        # Initialize to standard normal N(0, I)
        self.mu = torch.zeros(shape, dtype=torch.float32)
        self.sigma2 = torch.ones(shape, dtype=torch.float32)

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample z = mu + sigma * eps, eps ~ N(0, I)."""
        eps = torch.randn(batch_size, *self.shape, device=self.device, dtype=self.dtype)
        mu = self.mu.to(self.device, dtype=self.dtype)
        sigma = self.sigma2.sqrt().to(self.device, dtype=self.dtype)
        return mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

    def update_cem(
        self,
        noises: torch.Tensor,
        rewards: np.ndarray,
        elite_ratio: float = 0.1,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """CEM update: select elites, fit weighted Gaussian, apply regularization.

        Args:
            noises: (N, *shape) all collected noise samples.
            rewards: (N,) corresponding rewards.
            elite_ratio: fraction of top samples to use as elites.
            temperature: softmax temperature for reward weighting (0 = uniform over elites).

        Returns:
            Dict of update statistics for logging.
        """
        N = len(rewards)
        K = max(1, int(elite_ratio * N))

        # Select top-K elites
        elite_idx = np.argsort(rewards)[-K:]
        elite_noises = noises[elite_idx].float()  # (K, *shape)
        elite_rewards = rewards[elite_idx]

        # Compute weights
        if temperature > 0:
            # Softmax weighting over elite rewards
            r = torch.tensor(elite_rewards, dtype=torch.float32)
            r = (r - r.mean()) / (r.std() + 1e-8)
            weights = torch.softmax(r / temperature, dim=0)  # (K,)
        else:
            # Uniform over elites
            weights = torch.ones(K, dtype=torch.float32) / K

        # Weighted mean and variance
        weights_expanded = weights.view(K, *([1] * len(self.shape)))
        mu_elite = (weights_expanded * elite_noises).sum(dim=0)
        diff = elite_noises - mu_elite.unsqueeze(0)
        sigma2_elite = (weights_expanded * diff.pow(2)).sum(dim=0)
        # Ensure minimum variance
        sigma2_elite = sigma2_elite.clamp(min=1e-6)

        # Apply regularization
        if self.regularization_mode == "interpolation":
            self._regularize_interpolation(mu_elite, sigma2_elite)
        elif self.regularization_mode == "kl":
            self._regularize_kl(mu_elite, sigma2_elite)
        else:
            raise ValueError(f"Unknown regularization mode: {self.regularization_mode}")

        kl = self.kl_from_standard_normal()
        return {
            "num_elites": K,
            "elite_reward_mean": float(elite_rewards.mean()),
            "elite_reward_min": float(elite_rewards.min()),
            "kl_from_n01": kl,
            "mu_norm": float(self.mu.norm()),
            "sigma_mean": float(self.sigma2.sqrt().mean()),
        }

    def update_reward_weighted(
        self,
        noises: torch.Tensor,
        rewards: np.ndarray,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """On-policy reward-weighted update: all samples contribute via advantages.

        Analogous to GRPO: advantages = (r - mean) / std, then softmax weighting.
        High-reward samples pull the prior toward them, low-reward samples push away.

        Args:
            noises: (N, *shape) noise samples from the CURRENT prior (on-policy).
            rewards: (N,) corresponding rewards.
            temperature: softmax temperature for advantage weighting.

        Returns:
            Dict of update statistics for logging.
        """
        N = len(rewards)
        noises = noises.float()

        # Compute advantages (GRPO-style normalization)
        r = torch.tensor(rewards, dtype=torch.float32)
        advantages = (r - r.mean()) / (r.std() + 1e-8)

        # Softmax weighting over ALL samples
        weights = torch.softmax(advantages / temperature, dim=0)  # (N,)

        # Weighted mean and variance
        weights_expanded = weights.view(N, *([1] * len(self.shape)))
        mu_target = (weights_expanded * noises).sum(dim=0)
        diff = noises - mu_target.unsqueeze(0)
        sigma2_target = (weights_expanded * diff.pow(2)).sum(dim=0)
        sigma2_target = sigma2_target.clamp(min=1e-6)

        # Apply regularization
        if self.regularization_mode == "interpolation":
            self._regularize_interpolation(mu_target, sigma2_target)
        elif self.regularization_mode == "kl":
            self._regularize_kl(mu_target, sigma2_target)
        else:
            raise ValueError(f"Unknown regularization mode: {self.regularization_mode}")

        kl = self.kl_from_standard_normal()
        return {
            "reward_mean": float(r.mean()),
            "reward_std": float(r.std()),
            "reward_max": float(r.max()),
            "reward_min": float(r.min()),
            "kl_from_n01": kl,
            "mu_norm": float(self.mu.norm()),
            "sigma_mean": float(self.sigma2.sqrt().mean()),
        }

    def _regularize_interpolation(self, mu_target: torch.Tensor, sigma2_target: torch.Tensor):
        """Linear interpolation toward target stats, anchored at N(0, I)."""
        self.mu = self.alpha * mu_target
        self.sigma2 = (1 - self.alpha) * torch.ones_like(sigma2_target) + self.alpha * sigma2_target

    def _regularize_kl(self, mu_target: torch.Tensor, sigma2_target: torch.Tensor):
        """Binary search for largest beta s.t. KL(N(beta*mu, (1-beta)+beta*sigma2) || N(0,I)) <= kl_max."""
        lo, hi = 0.0, 1.0
        for _ in range(50):  # 50 iterations gives ~1e-15 precision
            mid = (lo + hi) / 2.0
            mu_cand = mid * mu_target
            sigma2_cand = (1 - mid) + mid * sigma2_target
            kl = self._compute_kl(mu_cand, sigma2_cand)
            if kl <= self.kl_max:
                lo = mid
            else:
                hi = mid

        self.mu = lo * mu_target
        self.sigma2 = (1 - lo) + lo * sigma2_target

    @staticmethod
    def _compute_kl(mu: torch.Tensor, sigma2: torch.Tensor) -> float:
        """KL(N(mu, diag(sigma2)) || N(0, I)) = 0.5 * sum(sigma2 + mu^2 - 1 - log(sigma2))."""
        return 0.5 * (sigma2 + mu.pow(2) - 1 - sigma2.clamp(min=1e-8).log()).sum().item()

    def kl_from_standard_normal(self) -> float:
        """Current KL divergence from N(0, I)."""
        return self._compute_kl(self.mu, self.sigma2)

    def save(self, path: str):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save({
            "mu": self.mu,
            "sigma2": self.sigma2,
            "shape": self.shape,
            "regularization_mode": self.regularization_mode,
            "alpha": self.alpha,
            "kl_max": self.kl_max,
        }, path)

    def load(self, path: str):
        state = torch.load(path, map_location="cpu")
        self.mu = state["mu"]
        self.sigma2 = state["sigma2"]
        self.shape = tuple(state["shape"])


class RewardCache:
    """Disk-based cache for noise -> reward mappings.

    Each epoch's data is saved as a separate .npz file. Since the DiT is frozen
    (ODE is deterministic), all cached data remains permanently valid.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._count = self._count_from_disk()
        # Next file index: append-only, never overwrites on resume
        self._next_idx = len(glob.glob(os.path.join(self.cache_dir, "epoch_*.npz")))

    def _count_from_disk(self) -> int:
        total = 0
        for f in glob.glob(os.path.join(self.cache_dir, "epoch_*.npz")):
            with np.load(f) as data:
                total += len(data["rewards"])
        return total

    def append(self, noises: torch.Tensor, rewards: np.ndarray, epoch: int):
        """Save one epoch's noise-reward pairs to disk (append-only naming)."""
        path = os.path.join(self.cache_dir, f"epoch_{self._next_idx:06d}.npz")
        np.savez_compressed(
            path,
            noises=noises.cpu().to(torch.float16).numpy(),
            rewards=rewards.astype(np.float32),
        )
        self._next_idx += 1
        self._count += len(rewards)

    def load_recent(self, max_epochs: int = -1) -> Tuple[torch.Tensor, np.ndarray]:
        """Load cached epochs. If max_epochs > 0, only load the most recent N."""
        files = sorted(glob.glob(os.path.join(self.cache_dir, "epoch_*.npz")))
        if not files:
            raise RuntimeError(f"No cache files found in {self.cache_dir}")
        if max_epochs > 0:
            files = files[-max_epochs:]

        all_noises = []
        all_rewards = []
        for f in files:
            with np.load(f) as data:
                all_noises.append(torch.from_numpy(data["noises"].astype(np.float32)))
                all_rewards.append(data["rewards"].copy())

        return torch.cat(all_noises, dim=0), np.concatenate(all_rewards)

    @property
    def total_samples(self) -> int:
        return self._count


class ParticlePrior:
    """Non-parametric prior via reward-weighted resampling + perturbation.

    Instead of fitting a Gaussian, maintains a buffer of (noise, reward) pairs.
    Sampling picks a noise from the buffer with probability proportional to
    reward, then adds Gaussian perturbation for exploration. This can represent
    arbitrarily complex, multi-modal distributions.

    p_shaped(z) ∝ Σ_i  w_i · N(z | z_i, σ²I)

    Falls back to N(0, I) when the buffer is empty (first epoch).
    """

    def __init__(
        self,
        shape: tuple,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        perturbation_std: float = 0.1,
        temperature: float = 1.0,
        mix_ratio: float = 0.1,
    ):
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.perturbation_std = perturbation_std
        self.temperature = temperature
        self.mix_ratio = mix_ratio  # fraction of samples drawn from N(0,I) for exploration

        # Buffer (populated via update())
        self.noises = None    # (N, *shape)
        self.rewards = None   # (N,) raw reward values
        self.weights = None   # (N,) sampling probabilities

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample from particle prior, or N(0,I) if buffer is empty."""
        if self.noises is None:
            return torch.randn(batch_size, *self.shape, device=self.device, dtype=self.dtype)

        # Split batch: some from N(0,I) for exploration, rest from buffer
        n_explore = int(batch_size * self.mix_ratio)
        n_exploit = batch_size - n_explore

        samples = []

        if n_exploit > 0:
            # Reward-weighted resampling from buffer
            idx = torch.multinomial(self.weights, n_exploit, replacement=True)
            selected = self.noises[idx].to(self.device, dtype=self.dtype)
            # Add perturbation
            perturbation = torch.randn_like(selected) * self.perturbation_std
            samples.append(selected + perturbation)

        if n_explore > 0:
            samples.append(torch.randn(n_explore, *self.shape, device=self.device, dtype=self.dtype))

        return torch.cat(samples, dim=0)

    def update(self, noises: torch.Tensor, rewards: np.ndarray) -> Dict[str, float]:
        """Update particle buffer with new (noise, reward) pairs.

        Appends to existing buffer and recomputes weights over ALL particles.
        """
        new_noises = noises.float().cpu()
        new_rewards = torch.tensor(rewards, dtype=torch.float32)

        if self.noises is None:
            self.noises = new_noises
            self.rewards = new_rewards
        else:
            self.noises = torch.cat([self.noises, new_noises], dim=0)
            self.rewards = torch.cat([self.rewards, new_rewards], dim=0)

        # Cap buffer (truncate both noises and rewards together)
        max_particles = 50000
        if len(self.noises) > max_particles:
            self.noises = self.noises[-max_particles:]
            self.rewards = self.rewards[-max_particles:]

        # Compute advantage-based weights over ALL buffered rewards
        adv = (self.rewards - self.rewards.mean()) / (self.rewards.std() + 1e-8)
        self.weights = torch.softmax(adv / self.temperature, dim=0)

        return self._compute_stats(self.rewards)

    def update_from_cache(self, cache: 'RewardCache', max_epochs: int = -1) -> Dict[str, float]:
        """Rebuild particle buffer from disk cache."""
        all_noises, all_rewards = cache.load_recent(max_epochs=max_epochs)

        self.noises = all_noises
        r = torch.tensor(all_rewards, dtype=torch.float32)
        self.rewards = r
        # Advantage-based weighting
        advantages = (r - r.mean()) / (r.std() + 1e-8)
        self.weights = torch.softmax(advantages / self.temperature, dim=0)

        return self._compute_stats(r)

    def _compute_stats(self, rewards: torch.Tensor) -> Dict[str, float]:
        return {
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_max": float(rewards.max()),
            "reward_min": float(rewards.min()),
            "buffer_size": len(self.noises),
            "perturbation_std": self.perturbation_std,
            "mix_ratio": self.mix_ratio,
        }

    def save(self, path: str):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save({
            "noises": self.noises,
            "rewards": self.rewards,
            "weights": self.weights,
            "shape": self.shape,
            "perturbation_std": self.perturbation_std,
            "temperature": self.temperature,
            "mix_ratio": self.mix_ratio,
        }, path)

    def load(self, path: str):
        state = torch.load(path, map_location="cpu")
        self.noises = state["noises"]
        self.rewards = state.get("rewards", None)
        self.weights = state["weights"]
        self.shape = tuple(state["shape"])
