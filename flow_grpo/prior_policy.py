"""
Lightweight prior policy networks for Flow Matching models.

Instead of training the DiT, we train a small policy network π_φ(z|prompt)
that outputs prompt-conditioned noise distributions. The DiT remains frozen.

Two policy types:
- GaussianPolicy: outputs μ(prompt), σ(prompt) via small ConvTranspose decoder
- NormalizingFlowPolicy: conditional affine coupling layers (TODO)

Training: advantage-weighted regression (not policy gradient).
  loss = -Σ w_i · log π_φ(z_i | prompt_i)
  w_i = softmax(advantage_i / temperature)
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class GaussianPolicy(nn.Module):
    """Prompt-conditioned diagonal Gaussian policy for noise sampling.

    Input: prompt embeddings (pooled + mean-pooled sequence)
    Output: μ(prompt), σ(prompt) of shape latent_shape
    Sample: z = μ + σ * ε, ε ~ N(0, I)
    """

    def __init__(
        self,
        prompt_embed_dim: int = 2048,
        seq_embed_dim: int = 4096,
        latent_shape: Tuple[int, int, int] = (16, 64, 64),
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.latent_shape = latent_shape
        C, H, W = latent_shape

        # Input projection: concat(pooled, mean-pooled sequence) → hidden
        input_dim = prompt_embed_dim + seq_embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Spatial decoder: hidden → (C, H, W) for both μ and log_σ
        # Compute number of upsample stages needed: 4 → H (each stage doubles)
        import math as _math
        num_upsample = int(_math.log2(H // 4))  # e.g., H=64 → 4 stages, H=8 → 1 stage
        assert 4 * (2 ** num_upsample) == H, f"H={H} must be 4 * 2^n"

        self.spatial_proj = nn.Linear(hidden_dim, hidden_dim * 4 * 4)
        layers = []
        ch_in = hidden_dim
        for i in range(num_upsample):
            ch_out = max(32, hidden_dim // (2 ** (i + 1)))
            layers.extend([
                nn.ConvTranspose2d(ch_in, ch_out, 4, stride=2, padding=1),
                nn.SiLU(),
            ])
            ch_in = ch_out
        self.decoder = nn.Sequential(*layers)
        decoder_out_ch = ch_in

        # Split into μ and log_σ
        self.mu_head = nn.Conv2d(decoder_out_ch, C, 1)
        self.log_sigma_head = nn.Conv2d(decoder_out_ch, C, 1)

        # Initialize heads to zero → at init, μ≈0 and σ≈1 (close to N(0,I))
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.zeros_(self.log_sigma_head.bias)

    def forward(self, pooled_prompt_embeds: torch.Tensor, prompt_embeds: torch.Tensor):
        """
        Args:
            pooled_prompt_embeds: (B, prompt_embed_dim) from CLIP pooled output
            prompt_embeds: (B, seq_len, seq_embed_dim) from text encoders

        Returns:
            mu: (B, C, H, W)
            log_sigma: (B, C, H, W)
        """
        # Mean-pool the sequence embeddings
        seq_pooled = prompt_embeds.mean(dim=1)  # (B, seq_embed_dim)

        # Concat and project
        x = torch.cat([pooled_prompt_embeds, seq_pooled], dim=-1)  # (B, input_dim)
        x = self.input_proj(x)  # (B, hidden_dim)

        # Spatial decode
        x = self.spatial_proj(x)  # (B, hidden_dim * 4 * 4)
        x = x.view(x.size(0), -1, 4, 4)  # (B, hidden_dim, 4, 4)
        x = self.decoder(x)  # (B, 32, 64, 64)

        mu = self.mu_head(x)  # (B, C, 64, 64)
        log_sigma = self.log_sigma_head(x)  # (B, C, 64, 64)
        return mu, log_sigma

    def sample(self, pooled_prompt_embeds: torch.Tensor, prompt_embeds: torch.Tensor):
        """Sample z ~ N(μ(prompt), σ(prompt)²I)."""
        mu, log_sigma = self.forward(pooled_prompt_embeds, prompt_embeds)
        sigma = log_sigma.exp()
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        return z

    def log_prob(self, z: torch.Tensor, pooled_prompt_embeds: torch.Tensor, prompt_embeds: torch.Tensor):
        """Compute log π_φ(z | prompt) = Σ log N(z_d | μ_d, σ_d²).

        Returns:
            log_prob: (B,) summed over all dimensions
        """
        mu, log_sigma = self.forward(pooled_prompt_embeds, prompt_embeds)
        # log N(z | mu, sigma^2) = -0.5 * ((z-mu)/sigma)^2 - log(sigma) - 0.5*log(2π)
        var = (2 * log_sigma).exp()
        log_p = -0.5 * ((z - mu) ** 2 / var + 2 * log_sigma + math.log(2 * math.pi))
        return log_p.sum(dim=(1, 2, 3))  # (B,)

    def kl_from_standard_normal(self, pooled_prompt_embeds: torch.Tensor, prompt_embeds: torch.Tensor):
        """KL(π_φ(·|prompt) || N(0, I)), averaged over batch."""
        mu, log_sigma = self.forward(pooled_prompt_embeds, prompt_embeds)
        sigma2 = (2 * log_sigma).exp()
        kl = 0.5 * (sigma2 + mu ** 2 - 1 - 2 * log_sigma)
        return kl.sum(dim=(1, 2, 3)).mean()  # scalar

    def entropy(self, pooled_prompt_embeds: torch.Tensor, prompt_embeds: torch.Tensor):
        """Entropy of the policy, averaged over batch."""
        _, log_sigma = self.forward(pooled_prompt_embeds, prompt_embeds)
        # Entropy of diagonal Gaussian: 0.5 * d * (1 + log(2π)) + Σ log(σ)
        d = log_sigma.shape[1] * log_sigma.shape[2] * log_sigma.shape[3]
        ent = 0.5 * d * (1 + math.log(2 * math.pi)) + log_sigma.sum(dim=(1, 2, 3))
        return ent.mean()


def compute_awr_loss(
    policy: GaussianPolicy,
    noises: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    advantages: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Advantage-Weighted Regression loss.

    loss = -Σ w_i · log π_φ(z_i | prompt_i)
    w_i = softmax(advantage_i / temperature)

    Args:
        policy: the prior policy network
        noises: (B, C, H, W) sampled noises
        pooled_prompt_embeds: (B, pooled_dim)
        prompt_embeds: (B, seq_len, embed_dim)
        advantages: (B,) GRPO-style advantages
        temperature: softmax temperature

    Returns:
        loss: scalar
        stats: dict of training statistics
    """
    # Compute weights from advantages
    weights = torch.softmax(advantages / temperature, dim=0)  # (B,)

    # Compute log-prob under current policy
    log_probs = policy.log_prob(noises, pooled_prompt_embeds, prompt_embeds)  # (B,)

    # Weighted NLL
    loss = -(weights * log_probs).sum()

    # Stats for logging
    with torch.no_grad():
        effective_sample_size = 1.0 / (weights ** 2).sum()
        stats = {
            "policy_loss": loss.item(),
            "log_prob_mean": log_probs.mean().item(),
            "log_prob_std": log_probs.std().item(),
            "weight_max": weights.max().item(),
            "weight_min": weights.min().item(),
            "effective_sample_size": effective_sample_size.item(),
        }

    return loss, stats
