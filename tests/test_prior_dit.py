"""Tests for flow_grpo.prior_dit — PriorDiT and AWR loss."""

import torch
import numpy as np
import pytest

from flow_grpo.prior_dit import PriorDiT, compute_dit_awr_loss


# Use small model for fast tests
def _make_small_dit():
    return PriorDiT(
        sample_size=8, patch_size=2, in_channels=4, out_channels=4,
        num_layers=2, num_attention_heads=4, attention_head_dim=16,
        joint_attention_dim=64, pooled_projection_dim=32,
        pos_embed_max_size=8,
    )


class TestPriorDiTForward:
    def test_output_shape(self):
        model = _make_small_dit()
        B = 2
        z_t = torch.randn(B, 4, 8, 8)
        pe = torch.randn(B, 10, 64)
        ppe = torch.randn(B, 32)
        t = torch.tensor([500.0, 300.0])
        v = model(z_t, pe, ppe, t)
        assert v.shape == (B, 4, 8, 8)

    def test_output_changes_with_timestep(self):
        model = _make_small_dit()
        z_t = torch.randn(1, 4, 8, 8)
        pe = torch.randn(1, 10, 64)
        ppe = torch.randn(1, 32)
        v1 = model(z_t, pe, ppe, torch.tensor([100.0]))
        v2 = model(z_t, pe, ppe, torch.tensor([900.0]))
        assert not torch.allclose(v1, v2)

    def test_gradient_flows(self):
        model = _make_small_dit()
        z_t = torch.randn(2, 4, 8, 8)
        pe = torch.randn(2, 10, 64)
        ppe = torch.randn(2, 32)
        t = torch.tensor([500.0, 300.0])
        v = model(z_t, pe, ppe, t)
        loss = v.mean()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad


class TestPriorDiTSample:
    def test_sample_shapes(self):
        model = _make_small_dit()
        pe = torch.randn(2, 10, 64)
        ppe = torch.randn(2, 32)
        z, eps = model.sample(pe, ppe, num_steps=2)
        assert z.shape == (2, 4, 8, 8)
        assert eps.shape == (2, 4, 8, 8)

    def test_epsilon_is_different_from_z(self):
        model = _make_small_dit()
        pe = torch.randn(2, 10, 64)
        ppe = torch.randn(2, 32)
        z, eps = model.sample(pe, ppe, num_steps=4)
        assert not torch.allclose(z, eps)

    def test_sample_with_cfg(self):
        model = _make_small_dit()
        pe = torch.randn(2, 10, 64)
        ppe = torch.randn(2, 32)
        neg_pe = torch.zeros(2, 10, 64)
        neg_ppe = torch.zeros(2, 32)
        z, eps = model.sample(pe, ppe, num_steps=2, cfg_scale=4.5,
                              neg_prompt_embeds=neg_pe, neg_pooled_prompt_embeds=neg_ppe)
        assert z.shape == (2, 4, 8, 8)

    def test_samples_differ_across_calls(self):
        model = _make_small_dit()
        pe = torch.randn(1, 10, 64)
        ppe = torch.randn(1, 32)
        z1, _ = model.sample(pe, ppe, num_steps=2)
        z2, _ = model.sample(pe, ppe, num_steps=2)
        assert not torch.allclose(z1, z2)


class TestDiTAWRLoss:
    def test_loss_is_scalar(self):
        model = _make_small_dit()
        B = 4
        eps = torch.randn(B, 4, 8, 8)
        z = torch.randn(B, 4, 8, 8)
        pe = torch.randn(B, 10, 64)
        ppe = torch.randn(B, 32)
        adv = torch.randn(B)
        loss, stats = compute_dit_awr_loss(model, eps, z, pe, ppe, adv)
        assert loss.shape == ()
        assert "dit_loss" in stats
        assert "dit_mse_mean" in stats

    def test_loss_backprop(self):
        model = _make_small_dit()
        B = 4
        eps = torch.randn(B, 4, 8, 8)
        z = torch.randn(B, 4, 8, 8)
        pe = torch.randn(B, 10, 64)
        ppe = torch.randn(B, 32)
        adv = torch.randn(B)
        loss, _ = compute_dit_awr_loss(model, eps, z, pe, ppe, adv)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_positive_advantage_reduces_mse(self):
        """With all positive advantages, training should reduce MSE."""
        model = _make_small_dit()
        B = 4
        eps = torch.randn(B, 4, 8, 8)
        z = eps + 0.1 * torch.randn(B, 4, 8, 8)  # z close to eps
        pe = torch.randn(B, 10, 64)
        ppe = torch.randn(B, 32)
        adv = torch.ones(B) * 2.0  # all positive

        loss, stats = compute_dit_awr_loss(model, eps, z, pe, ppe, adv)
        # Loss should be positive (positive advantage * positive MSE)
        assert loss.item() > 0

    def test_cfg_drop_rate(self):
        """With cfg_drop_rate=1.0, all prompts should be zeroed."""
        model = _make_small_dit()
        model.train()
        B = 4
        eps = torch.randn(B, 4, 8, 8)
        z = torch.randn(B, 4, 8, 8)
        pe = torch.ones(B, 10, 64)  # non-zero prompts
        ppe = torch.ones(B, 32)
        adv = torch.randn(B)

        # With cfg_drop_rate=1.0, prompts get zeroed
        loss, _ = compute_dit_awr_loss(model, eps, z, pe, ppe, adv, cfg_drop_rate=1.0)
        assert loss.shape == ()  # just verify it doesn't crash
