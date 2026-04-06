"""Tests for flow_grpo.prior_policy — GaussianPolicy and AWR loss."""

import torch
import numpy as np
import pytest

from flow_grpo.prior_policy import GaussianPolicy, compute_awr_loss


# ---------------------------------------------------------------------------
# GaussianPolicy — Shape & Init
# ---------------------------------------------------------------------------

class TestGaussianPolicyInit:
    def test_output_shapes(self):
        policy = GaussianPolicy(
            prompt_embed_dim=2048, seq_embed_dim=4096,
            latent_shape=(16, 64, 64), hidden_dim=256,
        )
        B = 4
        pooled = torch.randn(B, 2048)
        seq = torch.randn(B, 154, 4096)
        mu, log_sigma = policy(pooled, seq)
        assert mu.shape == (B, 16, 64, 64)
        assert log_sigma.shape == (B, 16, 64, 64)

    def test_initial_output_near_n01(self):
        """At initialization, μ≈0 and σ≈1 (close to N(0,I))."""
        policy = GaussianPolicy(
            prompt_embed_dim=2048, seq_embed_dim=4096,
            latent_shape=(16, 64, 64), hidden_dim=256,
        )
        pooled = torch.randn(2, 2048)
        seq = torch.randn(2, 154, 4096)
        mu, log_sigma = policy(pooled, seq)
        # Heads initialized to zeros, so outputs should be near zero
        assert mu.abs().mean().item() < 0.1
        assert log_sigma.abs().mean().item() < 0.1

    def test_parameter_count_reasonable(self):
        policy = GaussianPolicy(
            prompt_embed_dim=2048, seq_embed_dim=4096,
            latent_shape=(16, 64, 64), hidden_dim=512,
        )
        num_params = sum(p.numel() for p in policy.parameters())
        # Should be in the low millions, not hundreds of millions
        assert 100_000 < num_params < 20_000_000


# ---------------------------------------------------------------------------
# GaussianPolicy — Sampling
# ---------------------------------------------------------------------------

class TestGaussianPolicySample:
    def test_sample_shape(self):
        policy = GaussianPolicy(
            prompt_embed_dim=128, seq_embed_dim=256,
            latent_shape=(4, 8, 8), hidden_dim=64,
        )
        pooled = torch.randn(8, 128)
        seq = torch.randn(8, 10, 256)
        z = policy.sample(pooled, seq)
        assert z.shape == (8, 4, 8, 8)

    def test_samples_differ_across_calls(self):
        """Sampling is stochastic."""
        policy = GaussianPolicy(
            prompt_embed_dim=128, seq_embed_dim=256,
            latent_shape=(4, 8, 8), hidden_dim=64,
        )
        pooled = torch.randn(1, 128)
        seq = torch.randn(1, 10, 256)
        z1 = policy.sample(pooled, seq)
        z2 = policy.sample(pooled, seq)
        assert not torch.allclose(z1, z2)


# ---------------------------------------------------------------------------
# GaussianPolicy — Log Prob
# ---------------------------------------------------------------------------

class TestGaussianPolicyLogProb:
    def test_log_prob_shape(self):
        policy = GaussianPolicy(
            prompt_embed_dim=128, seq_embed_dim=256,
            latent_shape=(4, 8, 8), hidden_dim=64,
        )
        B = 4
        pooled = torch.randn(B, 128)
        seq = torch.randn(B, 10, 256)
        z = torch.randn(B, 4, 8, 8)
        lp = policy.log_prob(z, pooled, seq)
        assert lp.shape == (B,)

    def test_log_prob_is_negative(self):
        """Log prob of a Gaussian is always negative for high-dim."""
        policy = GaussianPolicy(
            prompt_embed_dim=128, seq_embed_dim=256,
            latent_shape=(4, 8, 8), hidden_dim=64,
        )
        pooled = torch.randn(2, 128)
        seq = torch.randn(2, 10, 256)
        z = torch.randn(2, 4, 8, 8)
        lp = policy.log_prob(z, pooled, seq)
        assert (lp < 0).all()

    def test_log_prob_higher_at_mean(self):
        """log π(μ|prompt) > log π(z_random|prompt) on average."""
        policy = GaussianPolicy(
            prompt_embed_dim=128, seq_embed_dim=256,
            latent_shape=(4, 8, 8), hidden_dim=64,
        )
        pooled = torch.randn(4, 128)
        seq = torch.randn(4, 10, 256)
        mu, _ = policy(pooled, seq)
        lp_at_mean = policy.log_prob(mu, pooled, seq)
        lp_at_random = policy.log_prob(torch.randn_like(mu) * 5, pooled, seq)
        assert (lp_at_mean > lp_at_random).all()

    def test_gradient_flows(self):
        """Verify backward works through log_prob."""
        policy = GaussianPolicy(
            prompt_embed_dim=128, seq_embed_dim=256,
            latent_shape=(4, 8, 8), hidden_dim=64,
        )
        pooled = torch.randn(2, 128)
        seq = torch.randn(2, 10, 256)
        z = torch.randn(2, 4, 8, 8)
        lp = policy.log_prob(z, pooled, seq)
        loss = -lp.mean()
        loss.backward()
        # Check that gradients exist on policy parameters
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
        assert has_grad


# ---------------------------------------------------------------------------
# GaussianPolicy — KL & Entropy
# ---------------------------------------------------------------------------

class TestGaussianPolicyKLEntropy:
    def test_kl_near_zero_at_init(self):
        policy = GaussianPolicy(
            prompt_embed_dim=128, seq_embed_dim=256,
            latent_shape=(4, 8, 8), hidden_dim=64,
        )
        pooled = torch.randn(2, 128)
        seq = torch.randn(2, 10, 256)
        kl = policy.kl_from_standard_normal(pooled, seq)
        # At init (mu≈0, sigma≈1), KL should be small
        assert kl.item() < 100  # Generous bound since init isn't exactly zero

    def test_entropy_is_positive(self):
        policy = GaussianPolicy(
            prompt_embed_dim=128, seq_embed_dim=256,
            latent_shape=(4, 8, 8), hidden_dim=64,
        )
        pooled = torch.randn(2, 128)
        seq = torch.randn(2, 10, 256)
        ent = policy.entropy(pooled, seq)
        assert ent.item() > 0


# ---------------------------------------------------------------------------
# AWR Loss
# ---------------------------------------------------------------------------

class TestAWRLoss:
    def test_loss_is_scalar(self):
        policy = GaussianPolicy(
            prompt_embed_dim=128, seq_embed_dim=256,
            latent_shape=(4, 8, 8), hidden_dim=64,
        )
        B = 8
        noises = torch.randn(B, 4, 8, 8)
        pooled = torch.randn(B, 128)
        seq = torch.randn(B, 10, 256)
        advantages = torch.randn(B)

        loss, stats = compute_awr_loss(policy, noises, pooled, seq, advantages)
        assert loss.shape == ()
        assert "policy_loss" in stats
        assert "effective_sample_size" in stats

    def test_loss_backprop(self):
        policy = GaussianPolicy(
            prompt_embed_dim=128, seq_embed_dim=256,
            latent_shape=(4, 8, 8), hidden_dim=64,
        )
        B = 8
        noises = torch.randn(B, 4, 8, 8)
        pooled = torch.randn(B, 128)
        seq = torch.randn(B, 10, 256)
        advantages = torch.randn(B)

        loss, _ = compute_awr_loss(policy, noises, pooled, seq, advantages)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
        assert has_grad

    def test_high_advantage_gets_more_weight(self):
        """With extreme advantages, loss should be dominated by the high-advantage sample."""
        policy = GaussianPolicy(
            prompt_embed_dim=128, seq_embed_dim=256,
            latent_shape=(4, 8, 8), hidden_dim=64,
        )
        B = 8
        noises = torch.randn(B, 4, 8, 8)
        pooled = torch.randn(B, 128)
        seq = torch.randn(B, 10, 256)

        # One sample has very high advantage
        advantages = torch.zeros(B)
        advantages[0] = 100.0

        loss, stats = compute_awr_loss(policy, noises, pooled, seq, advantages, temperature=1.0)
        # Weight should be concentrated on sample 0
        assert stats["weight_max"] > 0.99
        assert stats["effective_sample_size"] < 2.0

    def test_uniform_advantages_give_equal_weights(self):
        policy = GaussianPolicy(
            prompt_embed_dim=128, seq_embed_dim=256,
            latent_shape=(4, 8, 8), hidden_dim=64,
        )
        B = 8
        noises = torch.randn(B, 4, 8, 8)
        pooled = torch.randn(B, 128)
        seq = torch.randn(B, 10, 256)
        advantages = torch.zeros(B)  # all equal

        _, stats = compute_awr_loss(policy, noises, pooled, seq, advantages)
        assert stats["effective_sample_size"] == pytest.approx(B, abs=0.1)
