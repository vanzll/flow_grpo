"""
Small Flow Matching DiT for prior noise generation.

Instead of outputting Gaussian parameters (μ, σ), this model learns a velocity
field that transforms N(0,I) into prompt-conditioned "good noise" via multi-step
ODE. Trained with advantage-weighted flow matching MSE (DiffusionNFT-inspired).

Architecture mirrors SD3's DiT but much smaller (~90M vs 2.5B params).
Supports classifier-free guidance (CFG) via prompt dropout.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from diffusers.models.embeddings import PatchEmbed, CombinedTimestepTextProjEmbeddings
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.attention import JointTransformerBlock
from diffusers import FlowMatchEulerDiscreteScheduler


class PriorDiT(nn.Module):
    """Small Flow Matching DiT for prior noise generation.

    Input: noisy latent z_t + timestep + prompt embeddings
    Output: predicted velocity v(z_t, t, prompt)
    Sampling: multi-step ODE from N(0,I) → shaped noise z
    """

    def __init__(
        self,
        sample_size: int = 64,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: int = 16,
        num_layers: int = 6,
        num_attention_heads: int = 12,
        attention_head_dim: int = 64,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 2048,
        pos_embed_max_size: int = 48,
        small_init_output: bool = True,
        output_init_std: float = 1e-4,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        inner_dim = num_attention_heads * attention_head_dim

        # Patch embedding: (B, C, H, W) → (B, num_patches, inner_dim)
        self.pos_embed = PatchEmbed(
            height=sample_size, width=sample_size,
            patch_size=patch_size, in_channels=in_channels,
            embed_dim=inner_dim, pos_embed_max_size=pos_embed_max_size,
        )

        # Timestep + pooled text → conditioning
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=inner_dim,
            pooled_projection_dim=pooled_projection_dim,
        )

        # Text sequence → cross-attention keys/values
        self.context_embedder = nn.Linear(joint_attention_dim, inner_dim)

        # Transformer blocks (MMDiT-style joint attention)
        self.transformer_blocks = nn.ModuleList([
            JointTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                context_pre_only=(i == num_layers - 1),  # last block is context_pre_only
            )
            for i in range(num_layers)
        ])

        # Output: norm + project back to pixel space
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        if small_init_output:
            # Small-initialize output projection so initial velocity ≈ 0 (ODE ≈ identity)
            # This keeps the initial flow close to zero, which can help stability.
            nn.init.normal_(self.proj_out.weight, std=output_init_std)
            nn.init.zeros_(self.proj_out.bias)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, num_patches, patch_size²*C) → (B, C, H, W)"""
        p = self.patch_size
        h = w = self.sample_size // p
        x = x.view(x.shape[0], h, w, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(x.shape[0], self.out_channels, h * p, w * p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity v(z_t, t, prompt).

        Args:
            hidden_states: (B, C, H, W) noisy latent z_t
            encoder_hidden_states: (B, seq_len, joint_attention_dim) text embeddings
            pooled_projections: (B, pooled_projection_dim) pooled text embedding
            timestep: (B,) timestep values

        Returns:
            velocity: (B, C, H, W) predicted velocity
        """
        # Timestep + pooled text conditioning
        temb = self.time_text_embed(timestep, pooled_projections)

        # Patchify + positional embedding
        hidden_states = self.pos_embed(hidden_states)

        # Project text for cross-attention
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Transformer blocks
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

        # Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify
        velocity = self._unpatchify(hidden_states)
        return velocity

    @torch.no_grad()
    def sample(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        num_steps: int = 10,
        cfg_scale: float = 1.0,
        neg_prompt_embeds: Optional[torch.Tensor] = None,
        neg_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ODE sampling: N(0,I) → shaped noise z.

        Returns:
            z: (B, C, H, W) shaped noise
            epsilon: (B, C, H, W) starting noise (for flow matching training)
        """
        B = prompt_embeds.shape[0]
        device = prompt_embeds.device
        dtype = prompt_embeds.dtype

        # Start from N(0, I)
        epsilon = torch.randn(B, self.in_channels, self.sample_size, self.sample_size,
                              device=device, dtype=dtype)
        z = epsilon.clone()

        # Create scheduler for sampling
        scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
        scheduler.set_timesteps(num_steps, device=device)

        for t in scheduler.timesteps:
            timestep = t.expand(B)

            if cfg_scale > 1.0 and neg_prompt_embeds is not None:
                # CFG: run conditional and unconditional
                z_input = torch.cat([z, z], dim=0)
                t_input = torch.cat([timestep, timestep], dim=0)
                pe_input = torch.cat([prompt_embeds, neg_prompt_embeds], dim=0)
                ppe_input = torch.cat([pooled_prompt_embeds, neg_pooled_prompt_embeds], dim=0)

                v_pred = self.forward(z_input, pe_input, ppe_input, t_input)
                v_cond, v_uncond = v_pred.chunk(2, dim=0)
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = self.forward(z, prompt_embeds, pooled_prompt_embeds, timestep)

            z = scheduler.step(v_pred, t, z).prev_sample

        return z, epsilon


def compute_dit_awr_loss(
    model,
    epsilon: torch.Tensor,
    z: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    advantages: torch.Tensor,
    temperature: float = 1.0,
    cfg_drop_rate: float = 0.0,
    adv_clip_max: float = 5.0,
    v_reg_weight: float = 0.01,
    null_prompt_embeds: Optional[torch.Tensor] = None,
    null_pooled_prompt_embeds: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Advantage-weighted flow matching MSE loss (DiffusionNFT-inspired).

    loss = mean(advantage/temperature * ||v_pred - v_target||²)

    Positive advantage → minimize MSE (learn to generate this z from ε)
    Negative advantage → maximize MSE (learn to avoid this z)

    Args:
        model: PriorDiT (can be DDP-wrapped)
        epsilon: (B, C, H, W) starting noise from N(0,I)
        z: (B, C, H, W) generated shaped noise (ODE output)
        prompt_embeds: (B, seq_len, dim)
        pooled_prompt_embeds: (B, pooled_dim)
        advantages: (B,) raw GRPO advantages (can be negative)
        temperature: advantage scaling
        cfg_drop_rate: probability of dropping prompt for CFG training
        adv_clip_max: clamp advantages to [-adv_clip_max, adv_clip_max] for stability
        null_prompt_embeds: (B, seq_len, dim) embeddings of "" for CFG dropout
            (if None, zeros are used — less effective but backward compatible)
        null_pooled_prompt_embeds: (B, pooled_dim) pooled embeddings of ""

    Returns:
        loss: scalar
        stats: dict
    """
    B = z.shape[0]

    # Random timestep in [0, 1] for each sample
    t = torch.rand(B, device=z.device, dtype=z.dtype)

    # Flow matching interpolation: z_t = (1-t)*ε + t*z
    t_expand = t[:, None, None, None]
    z_t = (1 - t_expand) * epsilon + t_expand * z

    # Target velocity: dz_sigma/dsigma = d/dsigma[sigma*eps + (1-sigma)*z] = eps - z
    v_target = epsilon - z

    # CFG training: randomly drop prompt (replace with null prompt embeddings)
    if cfg_drop_rate > 0 and model.training:
        drop_mask = torch.rand(B, device=z.device) < cfg_drop_rate
        if drop_mask.any():
            prompt_embeds = prompt_embeds.clone()
            pooled_prompt_embeds = pooled_prompt_embeds.clone()
            if null_prompt_embeds is not None and null_pooled_prompt_embeds is not None:
                prompt_embeds[drop_mask] = null_prompt_embeds[drop_mask].to(prompt_embeds.dtype)
                pooled_prompt_embeds[drop_mask] = null_pooled_prompt_embeds[drop_mask].to(pooled_prompt_embeds.dtype)
            else:
                prompt_embeds[drop_mask] = 0.0
                pooled_prompt_embeds[drop_mask] = 0.0

    # Convert t to timestep format (1000 * (1-t): t=0→1000 noisy, t=1→0 clean)
    timestep = (1 - t) * 1000

    # Predict velocity through DDP-compatible forward()
    v_pred = model(z_t, prompt_embeds, pooled_prompt_embeds, timestep)

    # Per-sample MSE (averaged over spatial dims)
    mse = ((v_pred - v_target) ** 2).mean(dim=(1, 2, 3))  # (B,)

    # Clamp advantages to prevent extreme negative weights destabilizing training
    advantages = advantages.clamp(min=-adv_clip_max, max=adv_clip_max)

    # Advantage-weighted loss
    weights = advantages / temperature
    awr_loss = (weights * mse).mean()

    # Regularization: penalize velocity magnitude to keep z close to ε (≈ N(0,I))
    # ||v_pred||² penalty: if v_pred ≈ 0, then ODE is identity (z ≈ ε)
    # This prevents z from drifting away from the N(0,I) sphere
    v_reg = (v_pred ** 2).mean()
    loss = awr_loss + v_reg_weight * v_reg

    # Stats
    with torch.no_grad():
        stats = {
            "dit_loss": loss.item(),
            "dit_awr_loss": awr_loss.item(),
            "dit_v_reg": v_reg.item(),
            "dit_mse_mean": mse.mean().item(),
            "advantage_max": advantages.max().item(),
            "advantage_min": advantages.min().item(),
            "v_pred_norm": v_pred.flatten(1).norm(dim=1).mean().item(),
            "v_target_norm": v_target.flatten(1).norm(dim=1).mean().item(),
        }

    return loss, stats
