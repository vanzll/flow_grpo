"""Tests for flow_grpo.prior — GaussianPrior and RewardCache."""

import os
import tempfile
import shutil
import numpy as np
import torch
import pytest

from flow_grpo.prior import GaussianPrior, RewardCache


# ---------------------------------------------------------------------------
# GaussianPrior — Initialization
# ---------------------------------------------------------------------------

class TestGaussianPriorInit:
    def test_initializes_to_standard_normal(self):
        prior = GaussianPrior(shape=(4, 8, 8))
        assert torch.allclose(prior.mu, torch.zeros(4, 8, 8))
        assert torch.allclose(prior.sigma2, torch.ones(4, 8, 8))

    def test_kl_is_zero_at_init(self):
        prior = GaussianPrior(shape=(4, 8, 8))
        assert prior.kl_from_standard_normal() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# GaussianPrior — Sampling
# ---------------------------------------------------------------------------

class TestGaussianPriorSample:
    def test_sample_shape(self):
        prior = GaussianPrior(shape=(4, 8, 8))
        samples = prior.sample(batch_size=16)
        assert samples.shape == (16, 4, 8, 8)

    def test_sample_dtype_matches(self):
        prior = GaussianPrior(shape=(4, 8, 8), dtype=torch.float32)
        samples = prior.sample(batch_size=2)
        assert samples.dtype == torch.float32

    def test_samples_from_shifted_prior_have_nonzero_mean(self):
        prior = GaussianPrior(shape=(16,))
        prior.mu = torch.ones(16) * 5.0
        samples = prior.sample(batch_size=10000)
        # Mean should be close to 5.0
        assert samples.mean().item() == pytest.approx(5.0, abs=0.2)

    def test_samples_from_scaled_prior_have_correct_std(self):
        prior = GaussianPrior(shape=(16,))
        prior.sigma2 = torch.ones(16) * 4.0  # std = 2
        samples = prior.sample(batch_size=10000)
        assert samples.std().item() == pytest.approx(2.0, abs=0.2)


# ---------------------------------------------------------------------------
# GaussianPrior — KL Divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:
    def test_kl_increases_with_shifted_mean(self):
        prior = GaussianPrior(shape=(4, 8, 8))
        prior.mu = torch.ones(4, 8, 8) * 0.1
        kl = prior.kl_from_standard_normal()
        assert kl > 0

    def test_kl_increases_with_changed_variance(self):
        prior = GaussianPrior(shape=(4, 8, 8))
        prior.sigma2 = torch.ones(4, 8, 8) * 2.0
        kl = prior.kl_from_standard_normal()
        assert kl > 0

    def test_kl_formula_matches_analytical(self):
        """KL(N(mu, sigma2) || N(0,1)) = 0.5 * sum(sigma2 + mu^2 - 1 - log(sigma2))"""
        prior = GaussianPrior(shape=(2,))
        prior.mu = torch.tensor([0.5, -0.3])
        prior.sigma2 = torch.tensor([1.5, 0.8])
        mu, s2 = prior.mu, prior.sigma2
        expected = 0.5 * (s2 + mu**2 - 1 - s2.log()).sum().item()
        assert prior.kl_from_standard_normal() == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# GaussianPrior — Reward-Weighted Update
# ---------------------------------------------------------------------------

class TestRewardWeightedUpdate:
    def test_update_moves_prior_toward_high_reward_samples(self):
        prior = GaussianPrior(shape=(4,), regularization_mode="interpolation", alpha=0.5)
        # Create noises where positive region has high reward
        noises = torch.randn(100, 4)
        rewards = noises.mean(dim=1).numpy()  # reward = mean of noise dims
        prior.update_reward_weighted(noises, rewards, temperature=1.0)
        # mu should shift toward positive values (high-reward region)
        assert prior.mu.mean().item() > 0

    def test_update_returns_stats_dict(self):
        prior = GaussianPrior(shape=(4,))
        noises = torch.randn(50, 4)
        rewards = np.random.randn(50)
        stats = prior.update_reward_weighted(noises, rewards)
        assert "reward_mean" in stats
        assert "kl_from_n01" in stats
        assert "mu_norm" in stats
        assert "sigma_mean" in stats

    def test_kl_regularization_constrains_update(self):
        prior = GaussianPrior(shape=(4,), regularization_mode="kl", kl_max=0.1)
        noises = torch.randn(100, 4)
        # Strong reward signal to try to push prior far
        rewards = noises.mean(dim=1).numpy() * 10
        prior.update_reward_weighted(noises, rewards, temperature=0.1)
        assert prior.kl_from_standard_normal() <= 0.1 + 1e-6

    def test_interpolation_regularization_scales_by_alpha(self):
        noises = torch.randn(100, 4)
        rewards = noises.mean(dim=1).numpy() * 10

        prior_small = GaussianPrior(shape=(4,), regularization_mode="interpolation", alpha=0.01)
        prior_large = GaussianPrior(shape=(4,), regularization_mode="interpolation", alpha=0.5)
        prior_small.update_reward_weighted(noises, rewards, temperature=0.1)
        prior_large.update_reward_weighted(noises, rewards, temperature=0.1)

        # Smaller alpha → smaller mu shift
        assert prior_small.mu.norm().item() < prior_large.mu.norm().item()

    def test_low_temperature_concentrates_on_best_samples(self):
        prior_low_t = GaussianPrior(shape=(4,), regularization_mode="interpolation", alpha=1.0)
        prior_high_t = GaussianPrior(shape=(4,), regularization_mode="interpolation", alpha=1.0)

        torch.manual_seed(42)
        noises = torch.randn(100, 4)
        rewards = noises.mean(dim=1).numpy()

        prior_low_t.update_reward_weighted(noises, rewards, temperature=0.01)
        prior_high_t.update_reward_weighted(noises, rewards, temperature=10.0)

        # Low temperature → mu closer to the best sample
        best_idx = rewards.argmax()
        best_noise = noises[best_idx]
        dist_low = (prior_low_t.mu - best_noise).norm().item()
        dist_high = (prior_high_t.mu - best_noise).norm().item()
        assert dist_low < dist_high


# ---------------------------------------------------------------------------
# GaussianPrior — CEM Update
# ---------------------------------------------------------------------------

class TestCEMUpdate:
    def test_cem_selects_elites_correctly(self):
        prior = GaussianPrior(shape=(4,), regularization_mode="interpolation", alpha=1.0)
        noises = torch.randn(100, 4)
        rewards = noises.mean(dim=1).numpy()
        stats = prior.update_cem(noises, rewards, elite_ratio=0.1, temperature=0.0)
        assert stats["num_elites"] == 10

    def test_cem_with_kl_stays_within_budget(self):
        prior = GaussianPrior(shape=(4,), regularization_mode="kl", kl_max=0.5)
        noises = torch.randn(200, 4)
        rewards = noises.mean(dim=1).numpy() * 10
        prior.update_cem(noises, rewards, elite_ratio=0.1)
        assert prior.kl_from_standard_normal() <= 0.5 + 1e-6

    def test_cem_returns_stats(self):
        prior = GaussianPrior(shape=(4,))
        noises = torch.randn(50, 4)
        rewards = np.random.randn(50)
        stats = prior.update_cem(noises, rewards, elite_ratio=0.2)
        assert "num_elites" in stats
        assert "elite_reward_mean" in stats
        assert stats["num_elites"] == 10


# ---------------------------------------------------------------------------
# GaussianPrior — Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_load_roundtrip(self):
        prior = GaussianPrior(shape=(4, 8, 8), regularization_mode="kl", kl_max=2.0)
        prior.mu = torch.randn(4, 8, 8)
        prior.sigma2 = torch.rand(4, 8, 8) + 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "prior.pt")
            prior.save(path)

            loaded = GaussianPrior(shape=(4, 8, 8))
            loaded.load(path)

            assert torch.allclose(prior.mu, loaded.mu)
            assert torch.allclose(prior.sigma2, loaded.sigma2)

    def test_save_creates_directories(self):
        prior = GaussianPrior(shape=(4,))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "dir", "prior.pt")
            prior.save(path)
            assert os.path.exists(path)

    def test_save_bare_filename_works(self):
        """save() with no directory component should not crash."""
        prior = GaussianPrior(shape=(4,))
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                prior.save("prior.pt")
                assert os.path.exists("prior.pt")
            finally:
                os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# RewardCache — Basic Operations
# ---------------------------------------------------------------------------

class TestRewardCache:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_append_and_load(self):
        cache = RewardCache(self.tmpdir)
        noises = torch.randn(10, 4, 8, 8)
        rewards = np.random.randn(10).astype(np.float32)
        cache.append(noises, rewards, epoch=0)

        loaded_noises, loaded_rewards = cache.load_recent()
        assert loaded_noises.shape == (10, 4, 8, 8)
        assert len(loaded_rewards) == 10
        # Rewards should be exactly preserved (float32 -> float32)
        np.testing.assert_allclose(rewards, loaded_rewards, atol=1e-6)

    def test_total_samples_increments(self):
        cache = RewardCache(self.tmpdir)
        assert cache.total_samples == 0
        cache.append(torch.randn(5, 4), np.zeros(5), epoch=0)
        assert cache.total_samples == 5
        cache.append(torch.randn(3, 4), np.zeros(3), epoch=1)
        assert cache.total_samples == 8

    def test_load_recent_limits_epochs(self):
        cache = RewardCache(self.tmpdir)
        for i in range(10):
            cache.append(torch.randn(5, 4), np.ones(5) * i, epoch=i)

        noises, rewards = cache.load_recent(max_epochs=3)
        # Should only load last 3 epochs = 15 samples
        assert len(rewards) == 15
        # Last 3 epochs have rewards 7, 8, 9
        assert rewards.min() == pytest.approx(7.0)

    def test_load_recent_all_when_negative(self):
        cache = RewardCache(self.tmpdir)
        for i in range(5):
            cache.append(torch.randn(3, 4), np.zeros(3), epoch=i)
        noises, rewards = cache.load_recent(max_epochs=-1)
        assert len(rewards) == 15  # 5 epochs * 3 samples

    def test_empty_cache_raises(self):
        cache = RewardCache(self.tmpdir)
        with pytest.raises(RuntimeError, match="No cache files found"):
            cache.load_recent()


# ---------------------------------------------------------------------------
# RewardCache — Resume / Append-Only
# ---------------------------------------------------------------------------

class TestRewardCacheResume:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_append_only_naming_no_overwrite(self):
        """New RewardCache instance should not overwrite existing files."""
        cache1 = RewardCache(self.tmpdir)
        cache1.append(torch.randn(5, 4), np.ones(5) * 1.0, epoch=0)
        cache1.append(torch.randn(5, 4), np.ones(5) * 2.0, epoch=1)

        # Simulate resume: new instance
        cache2 = RewardCache(self.tmpdir)
        assert cache2.total_samples == 10
        cache2.append(torch.randn(5, 4), np.ones(5) * 3.0, epoch=2)
        assert cache2.total_samples == 15

        # All 3 files should exist
        noises, rewards = cache2.load_recent()
        assert len(rewards) == 15

    def test_resume_does_not_overwrite_files(self):
        """Verify files from first run survive after resume append."""
        cache1 = RewardCache(self.tmpdir)
        original_rewards = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        cache1.append(torch.randn(3, 4), original_rewards, epoch=0)

        # Resume
        cache2 = RewardCache(self.tmpdir)
        cache2.append(torch.randn(2, 4), np.array([40.0, 50.0], dtype=np.float32), epoch=1)

        _, all_rewards = cache2.load_recent()
        # Original rewards should still be there
        np.testing.assert_allclose(all_rewards[:3], original_rewards, atol=1e-6)

    def test_noise_fp16_precision(self):
        """Noises are stored as fp16 — verify acceptable precision for CEM/reward-weighted."""
        cache = RewardCache(self.tmpdir)
        original = torch.randn(10, 4)
        cache.append(original, np.zeros(10), epoch=0)
        loaded, _ = cache.load_recent()
        # fp16 roundtrip: expect ~1e-3 precision for normal-range values
        torch.testing.assert_close(original, loaded, atol=1e-2, rtol=1e-2)
