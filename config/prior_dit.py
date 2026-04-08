"""
Configs for Prior DiT training.

Train a small Flow Matching DiT that transforms N(0,I) → prompt-conditioned
"good noise" via multi-step ODE. The big DiT stays frozen.
"""

import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()


def _add_prior_dit_config(config):
    """Add prior-DiT-specific fields."""
    config.prior_dit = dit = ml_collections.ConfigDict()
    dit.num_layers = 8
    dit.num_attention_heads = 16
    dit.attention_head_dim = 64
    dit.patch_size = 2
    dit.num_steps = 10                  # ODE steps for small DiT sampling
    dit.learning_rate = 1e-4
    dit.weight_decay = 1e-4
    dit.cfg_drop_rate = 0.1             # CFG training: probability of dropping prompt
    dit.cfg_scale = 4.5                 # CFG inference guidance scale
    dit.temperature = 1.0               # advantage scaling
    dit.train_every_n_epochs = 1
    dit.cache_dir = "cache/prior_dit"
    dit.resume_path = ""

    config.use_lora = False
    config.train.ema = False
    return config


def pickscore_sd3_dit_1gpu():
    """1-GPU config for Prior DiT with PickScore."""
    config = base.get_config()
    _add_prior_dit_config(config)

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.resolution = 512
    config.mixed_precision = "fp16"

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
    config.save_dir = "logs/prior_dit/pickscore_1gpu"

    config.reward_fn = {"pickscore": 1.0}
    config.prompt_fn = "general_ocr"
    config.per_prompt_stat_tracking = True
    config.sample.global_std = True
    return config


def pickscore_sd3_dit_4gpu():
    """4x A40 48GB config for Prior DiT."""
    config = pickscore_sd3_dit_1gpu()
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = 8
    config.sample.test_batch_size = 8  # keep small to avoid OOM with CFG (effective batch 16)
    config.save_dir = "logs/prior_dit/pickscore_4gpu"
    return config


def pickscore_sd3_dit_8gpu_h20():
    """8x H20 96GB config for Prior DiT."""
    config = pickscore_sd3_dit_1gpu()
    config.sample.train_batch_size = 16
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 8
    config.sample.test_batch_size = 32
    config.prior_dit.cache_dir = "cache/prior_dit_8gpu"
    config.save_dir = "logs/prior_dit/pickscore_8gpu_h20"
    return config


def pickscore_sd3_dit_4gpu_smoke():
    """Minimal smoke test."""
    config = pickscore_sd3_dit_1gpu()
    config.prior_dit.num_layers = 2
    config.prior_dit.num_steps = 2
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 2
    config.sample.num_batches_per_epoch = 1
    config.sample.num_steps = 5
    config.sample.eval_num_steps = 5
    config.sample.test_batch_size = 2
    config.num_epochs = 2
    config.eval_freq = 1
    config.save_freq = 2
    config.save_dir = "logs/prior_dit/smoke_4gpu"
    config.prior_dit.cache_dir = "cache/prior_dit_smoke"
    return config
