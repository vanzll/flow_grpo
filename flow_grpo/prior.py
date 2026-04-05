"""
Prior shaping utilities for Flow Matching models.

Instead of fine-tuning DiT weights (as Flow-GRPO does), we keep the DiT frozen
and optimize the noise prior distribution. Since Flow Matching is an ODE, a fixed
noise deterministically maps to a fixed image and reward. We use the Cross-Entropy
Method (CEM) to iteratively reshape the prior N(0, I) toward high-reward regions,
with regularization to prevent drift from the standard normal.
"""

import os
import glob
import math
import numpy as np
import torch
from typing import Tuple, Optional, Dict, List


class GaussianPrior:
    """Diagonal Gaussian prior N(mu, diag(sigma^2)) for latent noise sampling.

    Supports CEM-based updates with two regularization modes:
    - "kl": constrain KL(p_shaped || N(0,I)) <= kl_max via binary search
    - "interpolation": mu_new = alpha * mu_elite, sigma2_new = (1-alpha) + alpha * sigma2_elite
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

    def _regularize_interpolation(self, mu_elite: torch.Tensor, sigma2_elite: torch.Tensor):
        """Linear interpolation toward elite stats, anchored at N(0, I)."""
        self.mu = self.alpha * mu_elite
        self.sigma2 = (1 - self.alpha) * torch.ones_like(sigma2_elite) + self.alpha * sigma2_elite

    def _regularize_kl(self, mu_elite: torch.Tensor, sigma2_elite: torch.Tensor):
        """Binary search for largest beta s.t. KL(N(beta*mu_e, (1-beta)+beta*sigma2_e) || N(0,I)) <= kl_max."""
        lo, hi = 0.0, 1.0
        for _ in range(50):  # 50 iterations gives ~1e-15 precision
            mid = (lo + hi) / 2.0
            mu_cand = mid * mu_elite
            sigma2_cand = (1 - mid) + mid * sigma2_elite
            kl = self._compute_kl(mu_cand, sigma2_cand)
            if kl <= self.kl_max:
                lo = mid
            else:
                hi = mid

        self.mu = lo * mu_elite
        self.sigma2 = (1 - lo) + lo * sigma2_elite

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
        state = torch.load(path, map_location="cpu", weights_only=True)
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
        # In-memory sample count (avoids re-reading disk every time)
        self._count = self._count_from_disk()

    def _count_from_disk(self) -> int:
        total = 0
        for f in glob.glob(os.path.join(self.cache_dir, "epoch_*.npz")):
            data = np.load(f)
            total += len(data["rewards"])
        return total

    def append(self, noises: torch.Tensor, rewards: np.ndarray, epoch: int):
        """Save one epoch's noise-reward pairs to disk."""
        path = os.path.join(self.cache_dir, f"epoch_{epoch:06d}.npz")
        np.savez_compressed(
            path,
            noises=noises.cpu().to(torch.float16).numpy(),
            rewards=rewards.astype(np.float32),
        )
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
            data = np.load(f)
            all_noises.append(torch.from_numpy(data["noises"].astype(np.float32)))
            all_rewards.append(data["rewards"])

        return torch.cat(all_noises, dim=0), np.concatenate(all_rewards)

    @property
    def total_samples(self) -> int:
        return self._count
