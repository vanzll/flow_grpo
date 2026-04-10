"""
Configs for offline Prior DiT training from cached prompt/noise/reward tuples.
"""

import imp
import os

import ml_collections

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()


def _add_prior_dit_config(config):
    config.prior_dit = dit = ml_collections.ConfigDict()
    dit.num_layers = 8
    dit.num_attention_heads = 16
    dit.attention_head_dim = 64
    dit.patch_size = 2
    dit.num_steps = 45
    dit.learning_rate = 1e-4
    dit.weight_decay = 1e-4
    dit.cfg_drop_rate = 0.1
    dit.cfg_scale = 4.5
    dit.v_reg_weight = 0.0
    dit.resume_path = ""
    dit.small_init_output = False
    dit.output_init_std = 1e-4

    config.use_lora = False
    config.train.ema = False
    return config


def _add_offline_config(config):
    config.offline = offline = ml_collections.ConfigDict()
    offline.cache_dir = "cache/prior_dit_8gpu"
    offline.max_cache_files = 0
    offline.num_val_files = 16
    offline.train_batch_size = 256
    offline.val_batch_size = 256
    offline.num_workers = 0
    offline.max_val_batches = 32
    offline.score_source = "advantages"
    offline.weight_transform = "binary_positive"
    offline.weight_temperature = 1.0
    offline.score_clip = 0.0
    offline.normalize_weights = False
    offline.positive_only = False
    offline.min_weight = 0.0
    offline.normalize_by_weight_sum = True
    return config


def pickscore_sd3_dit_offline_1gpu():
    config = base.get_config()
    _add_prior_dit_config(config)
    _add_offline_config(config)

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.resolution = 512
    config.mixed_precision = "fp16"

    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.test_batch_size = 4
    config.train.max_grad_norm = 0.0

    config.num_epochs = 50
    config.eval_freq = 5
    config.save_freq = 10
    config.save_dir = "logs/prior_dit_offline/pickscore_1gpu"

    config.reward_fn = {"pickscore": 1.0}
    return config


def pickscore_sd3_dit_offline_8gpu_h20():
    config = pickscore_sd3_dit_offline_1gpu()
    config.sample.test_batch_size = 32
    config.offline.train_batch_size = 256
    config.offline.val_batch_size = 256
    config.offline.num_val_files = 16
    config.save_dir = "logs/prior_dit_offline/pickscore_8gpu_h20"
    return config


def pickscore_sd3_dit_offline_8gpu_h20_reward():
    config = pickscore_sd3_dit_offline_8gpu_h20()
    config.offline.score_source = "rewards"
    config.offline.weight_transform = "softplus"
    config.offline.weight_temperature = 0.1
    config.save_dir = "logs/prior_dit_offline_reward/pickscore_8gpu_h20"
    return config


def pickscore_sd3_dit_offline_smoke():
    config = pickscore_sd3_dit_offline_1gpu()
    config.prior_dit.num_layers = 2
    config.prior_dit.num_steps = 2
    config.sample.eval_num_steps = 8
    config.sample.test_batch_size = 2
    config.num_epochs = 2
    config.eval_freq = 1
    config.save_freq = 2
    config.offline.max_cache_files = 8
    config.offline.num_val_files = 0
    config.offline.train_batch_size = 2
    config.offline.max_val_batches = 2
    config.save_dir = "logs/prior_dit_offline/smoke"
    return config
