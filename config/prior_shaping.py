"""
Configs for prior shaping experiments.

Instead of fine-tuning the DiT (as Flow-GRPO does), we keep it frozen and
optimize the noise prior distribution. These configs mirror the Flow-GRPO
PickScore configs for fair comparison.

Three update strategies:
- "reward_weighted" (default): on-policy Gaussian, all samples contribute via advantage weighting.
- "cem": Cross-Entropy Method, Gaussian, elite selection from cached historical data.
- "particle": non-parametric, reward-weighted resampling from noise buffer + perturbation.
  Can represent arbitrarily complex, multi-modal distributions.
"""

import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()


def _add_prior_config(config):
    """Add prior-shaping-specific fields to an existing base config."""
    config.prior = prior = ml_collections.ConfigDict()
    prior.update_method = "reward_weighted"  # "reward_weighted", "cem", or "particle"
    prior.regularization_mode = "kl"         # "kl" or "interpolation" (gaussian methods)
    prior.alpha = 0.1                        # interpolation strength (gaussian methods)
    prior.kl_max = 1e10                       # max KL from N(0,I); 1e10 = effectively no constraint
    prior.elite_ratio = 0.1                  # top fraction (CEM only)
    prior.temperature = 1.0                  # softmax temperature for reward weighting
    prior.cache_dir = "cache/prior_shaping"  # disk cache for noise->reward pairs (always saved)
    prior.use_cache_history = True           # use historical data (CEM and particle)
    prior.max_cache_epochs = 50              # max epochs to load from cache (-1 = all)
    prior.resume_prior_path = ""             # path to a saved prior .pt file
    # Particle-specific
    prior.perturbation_std = 0.1             # perturbation σ after resampling
    prior.mix_ratio = 0.1                    # fraction of samples from N(0,I) for exploration

    # Override training-related fields that are not needed
    config.use_lora = False
    config.train.ema = False
    return config


def pickscore_sd3_prior_1gpu():
    """1-GPU config for prior shaping with PickScore on SD3.5-Medium."""
    config = base.get_config()
    _add_prior_config(config)

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.resolution = 512
    config.mixed_precision = "fp16"

    # Sampling: pure ODE, no SDE noise
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = 4
    config.sample.test_batch_size = 4

    config.num_epochs = 200
    config.eval_freq = 10
    config.save_freq = 20
    config.save_dir = "logs/prior_shaping/pickscore_1gpu"

    config.reward_fn = {"pickscore": 1.0}
    config.prompt_fn = "general_ocr"
    config.per_prompt_stat_tracking = False
    return config


def pickscore_sd3_prior_2gpu():
    """2-GPU config for prior shaping."""
    config = pickscore_sd3_prior_1gpu()
    config.sample.train_batch_size = 8
    config.sample.num_batches_per_epoch = 4
    config.sample.test_batch_size = 8
    config.save_dir = "logs/prior_shaping/pickscore_2gpu"
    return config


def pickscore_sd3_prior_4gpu():
    """4x A40 48GB config for prior shaping.

    Produces 4*8*8 = 256 samples per epoch for CEM update.
    """
    config = pickscore_sd3_prior_1gpu()
    config.sample.train_batch_size = 8
    config.sample.num_batches_per_epoch = 8
    config.sample.test_batch_size = 16
    config.save_dir = "logs/prior_shaping/pickscore_4gpu"
    return config


def pickscore_sd3_particle_4gpu():
    """4x A40 48GB config for particle prior shaping.

    80% from N(0,I), 20% from reward-weighted buffer.
    """
    config = pickscore_sd3_prior_1gpu()
    config.prior.update_method = "particle"
    config.prior.mix_ratio = 0.8              # 80% N(0,I), 20% buffer
    config.prior.perturbation_std = 0.1
    config.prior.temperature = 1.0
    config.sample.train_batch_size = 8
    config.sample.num_batches_per_epoch = 8
    config.sample.test_batch_size = 16
    config.prior.cache_dir = "cache/prior_shaping_particle"
    config.save_dir = "logs/prior_shaping/particle_4gpu"
    return config


def pickscore_sd3_prior_4gpu_smoke():
    """Minimal 4-GPU smoke test: 2 epochs, 2 batches, few steps."""
    config = pickscore_sd3_prior_1gpu()
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 2
    config.sample.num_batches_per_epoch = 1
    config.sample.num_steps = 5
    config.sample.eval_num_steps = 5
    config.sample.test_batch_size = 2
    config.num_epochs = 2
    config.eval_freq = 1
    config.save_freq = 2
    config.save_dir = "logs/prior_shaping/smoke_4gpu"
    config.prior.cache_dir = "cache/prior_shaping_smoke"
    return config
