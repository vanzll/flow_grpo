"""
Lightweight prior policy networks for Flow Matching models.

Instead of training the DiT, we train a small policy network π_φ(z|prompt)
that outputs prompt-conditioned noise distributions. The DiT remains frozen.

Policy architectures:
- GaussianPolicy: MLP + ConvTranspose decoder (lightweight, ~2M params)
- TransformerPolicy: cross-attention over text tokens + spatial queries (~5-10M params)

Both output diagonal Gaussian parameters (μ, log_σ) of shape latent_shape.

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
        assert H == W, f"Non-square latents not supported: H={H}, W={W}"
        num_upsample = int(math.log2(H // 4))  # e.g., H=64 → 4 stages, H=8 → 1 stage
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

        # Initialize: small weights so output starts near zero (μ≈0, σ≈1)
        # but NOT zero weights (which blocks gradient flow through the conv)
        nn.init.normal_(self.mu_head.weight, std=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.log_sigma_head.weight, std=0.01)
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


class TransformerPolicy(nn.Module):
    """Transformer-based prompt-conditioned policy for noise sampling.

    Uses learnable spatial query tokens that attend to text token sequence
    via cross-attention, then self-attend among themselves. Output is
    upsampled to latent_shape for μ and log_σ.

    Architecture:
        text tokens (B, seq_len, text_dim)
            ↓ project to model_dim
        spatial queries (num_queries learnable, dim=model_dim)
            ↓ cross-attention with text tokens × N layers
            ↓ self-attention among queries × N layers
            ↓ reshape to (spatial_res, spatial_res, model_dim)
            ↓ ConvTranspose upsample to (C, H, W)
            ↓ mu_head + log_sigma_head
    """

    def __init__(
        self,
        prompt_embed_dim: int = 2048,
        seq_embed_dim: int = 4096,
        latent_shape: Tuple[int, int, int] = (16, 64, 64),
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        spatial_res: int = 8,
    ):
        super().__init__()
        C, H, W = latent_shape
        self.latent_shape = latent_shape
        self.spatial_res = spatial_res
        num_queries = spatial_res * spatial_res

        assert H == W, f"Non-square latents not supported: H={H}, W={W}"

        # Text projection: seq_embed_dim → model_dim
        self.text_proj = nn.Linear(seq_embed_dim, model_dim)
        # Pooled prompt projection (injected via adaptive LayerNorm)
        self.pooled_proj = nn.Sequential(
            nn.Linear(prompt_embed_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim * 2),  # scale and shift for AdaLN
        )

        # Learnable spatial queries
        self.spatial_queries = nn.Parameter(torch.randn(1, num_queries, model_dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(model_dim, num_heads, batch_first=True),
                "cross_norm": nn.LayerNorm(model_dim),
                "self_attn": nn.MultiheadAttention(model_dim, num_heads, batch_first=True),
                "self_norm": nn.LayerNorm(model_dim),
                "ffn": nn.Sequential(
                    nn.Linear(model_dim, model_dim * 4),
                    nn.GELU(),
                    nn.Linear(model_dim * 4, model_dim),
                ),
                "ffn_norm": nn.LayerNorm(model_dim),
            }))

        # Upsample from spatial_res to H
        num_upsample = int(math.log2(H // spatial_res))
        assert spatial_res * (2 ** num_upsample) == H

        upsample_layers = []
        ch_in = model_dim
        for i in range(num_upsample):
            ch_out = max(32, model_dim // (2 ** (i + 1)))
            upsample_layers.extend([
                nn.ConvTranspose2d(ch_in, ch_out, 4, stride=2, padding=1),
                nn.SiLU(),
            ])
            ch_in = ch_out
        self.upsample = nn.Sequential(*upsample_layers)
        decoder_out_ch = ch_in

        # Output heads
        self.mu_head = nn.Conv2d(decoder_out_ch, C, 1)
        self.log_sigma_head = nn.Conv2d(decoder_out_ch, C, 1)

        # Small init for heads (not zero — blocks gradient flow)
        nn.init.normal_(self.mu_head.weight, std=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.log_sigma_head.weight, std=0.01)
        nn.init.zeros_(self.log_sigma_head.bias)

    def forward(self, pooled_prompt_embeds: torch.Tensor, prompt_embeds: torch.Tensor):
        """
        Args:
            pooled_prompt_embeds: (B, prompt_embed_dim)
            prompt_embeds: (B, seq_len, seq_embed_dim)
        Returns:
            mu: (B, C, H, W)
            log_sigma: (B, C, H, W)
        """
        B = pooled_prompt_embeds.shape[0]

        # Project text tokens
        text_tokens = self.text_proj(prompt_embeds)  # (B, seq_len, model_dim)

        # AdaLN parameters from pooled prompt
        adaln_params = self.pooled_proj(pooled_prompt_embeds)  # (B, model_dim*2)
        scale, shift = adaln_params.chunk(2, dim=-1)  # each (B, model_dim)

        # Expand spatial queries for batch
        queries = self.spatial_queries.expand(B, -1, -1)  # (B, num_queries, model_dim)

        # Transformer layers
        for layer in self.layers:
            # Cross-attention: queries attend to text tokens
            residual = queries
            queries = layer["cross_norm"](queries)
            queries = residual + layer["cross_attn"](queries, text_tokens, text_tokens)[0]

            # Self-attention among spatial queries
            residual = queries
            queries = layer["self_norm"](queries)
            queries = residual + layer["self_attn"](queries, queries, queries)[0]

            # FFN
            residual = queries
            queries = layer["ffn_norm"](queries)
            queries = residual + layer["ffn"](queries)

        # Apply AdaLN modulation
        queries = queries * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Reshape to spatial grid
        queries = queries.view(B, self.spatial_res, self.spatial_res, -1)
        queries = queries.permute(0, 3, 1, 2)  # (B, model_dim, spatial_res, spatial_res)

        # Upsample to latent resolution
        x = self.upsample(queries)  # (B, decoder_out_ch, H, W)

        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)
        return mu, log_sigma

    def sample(self, pooled_prompt_embeds: torch.Tensor, prompt_embeds: torch.Tensor):
        mu, log_sigma = self.forward(pooled_prompt_embeds, prompt_embeds)
        sigma = log_sigma.exp()
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def log_prob(self, z: torch.Tensor, pooled_prompt_embeds: torch.Tensor, prompt_embeds: torch.Tensor):
        mu, log_sigma = self.forward(pooled_prompt_embeds, prompt_embeds)
        var = (2 * log_sigma).exp()
        log_p = -0.5 * ((z - mu) ** 2 / var + 2 * log_sigma + math.log(2 * math.pi))
        return log_p.sum(dim=(1, 2, 3))

    def kl_from_standard_normal(self, pooled_prompt_embeds: torch.Tensor, prompt_embeds: torch.Tensor):
        mu, log_sigma = self.forward(pooled_prompt_embeds, prompt_embeds)
        sigma2 = (2 * log_sigma).exp()
        kl = 0.5 * (sigma2 + mu ** 2 - 1 - 2 * log_sigma)
        return kl.sum(dim=(1, 2, 3)).mean()

    def entropy(self, pooled_prompt_embeds: torch.Tensor, prompt_embeds: torch.Tensor):
        _, log_sigma = self.forward(pooled_prompt_embeds, prompt_embeds)
        d = log_sigma.shape[1] * log_sigma.shape[2] * log_sigma.shape[3]
        ent = 0.5 * d * (1 + math.log(2 * math.pi)) + log_sigma.sum(dim=(1, 2, 3))
        return ent.mean()


def build_policy(config, pooled_embed_dim, seq_embed_dim, latent_shape):
    """Factory function to create policy based on config."""
    if config.policy.type == "gaussian":
        return GaussianPolicy(
            prompt_embed_dim=pooled_embed_dim,
            seq_embed_dim=seq_embed_dim,
            latent_shape=latent_shape,
            hidden_dim=config.policy.hidden_dim,
        )
    elif config.policy.type == "transformer":
        return TransformerPolicy(
            prompt_embed_dim=pooled_embed_dim,
            seq_embed_dim=seq_embed_dim,
            latent_shape=latent_shape,
            model_dim=config.policy.hidden_dim,
            num_heads=config.policy.get("num_heads", 8),
            num_layers=config.policy.get("num_layers", 4),
            spatial_res=config.policy.get("spatial_res", 8),
        )
    else:
        raise ValueError(f"Unknown policy type: {config.policy.type}")


def compute_awr_loss(
    policy,
    noises: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    advantages: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Advantage-Weighted Regression loss.

    loss = -Σ w_i · log_prob_per_dim_i
    w_i = softmax(advantage_i / temperature)

    Advantages are already per-prompt normalized (GRPO-style), so we use them
    directly as weights via softmax over the full batch. The per-prompt
    normalization ensures advantages are comparable across prompts.

    Log-prob is normalized by dimensionality to keep loss magnitude reasonable.

    Args:
        policy: the prior policy network (can be DDP-wrapped)
        noises: (B, C, H, W) sampled noises
        pooled_prompt_embeds: (B, pooled_dim)
        prompt_embeds: (B, seq_len, embed_dim)
        advantages: (B,) GRPO-style per-prompt normalized advantages
        temperature: softmax temperature

    Returns:
        loss: scalar
        stats: dict of training statistics
    """
    B = noises.shape[0]
    num_dims = noises.shape[1] * noises.shape[2] * noises.shape[3]

    # Compute weights from advantages
    weights = torch.softmax(advantages / temperature, dim=0)  # (B,)

    # Compute log-prob via forward() (DDP-compatible)
    mu, log_sigma = policy(pooled_prompt_embeds, prompt_embeds)
    var = (2 * log_sigma).exp()
    log_probs = -0.5 * ((noises - mu) ** 2 / var + 2 * log_sigma + math.log(2 * math.pi))
    log_probs = log_probs.sum(dim=(1, 2, 3))  # (B,)

    # Weighted NLL (use raw log_probs for proper gradient magnitude)
    loss = -(weights * log_probs).sum() / B

    # Stats for logging
    with torch.no_grad():
        effective_sample_size = 1.0 / (weights ** 2).sum()
        stats = {
            "policy_loss": loss.item(),
            "log_prob_mean": log_probs.mean().item(),
            "log_prob_per_dim": (log_probs / num_dims).mean().item(),
            "weight_max": weights.max().item(),
            "weight_min": weights.min().item(),
            "effective_sample_size": effective_sample_size.item(),
        }

    return loss, stats
