"""
Configs for prior shaping experiments.

Instead of fine-tuning the DiT (as Flow-GRPO does), we keep it frozen and
optimize the noise prior via CEM. These configs mirror the Flow-GRPO PickScore
configs for fair comparison.
"""

import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def _add_prior_config(config):
    """Add prior-shaping-specific fields to an existing base config."""
    config.prior = prior = ml_collections.ConfigDict()
    prior.regularization_mode = "kl"        # "kl" (default) or "interpolation"
    prior.alpha = 0.1                        # interpolation strength
    prior.kl_max = 5.0                       # max KL(p_shaped || N(0,I))
    prior.elite_ratio = 0.1                  # top fraction for CEM
    prior.temperature = 1.0                  # softmax temperature (0 = hard top-k)
    prior.cache_dir = "cache/prior_shaping"  # disk cache for noise->reward pairs
    prior.use_cache_history = True           # use all historical data for CEM update
    prior.resume_prior_path = ""             # path to a saved prior .pt file

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
