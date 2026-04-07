"""
Configs for prior policy training.

Train a small policy network π_φ(z|prompt) with advantage-weighted regression.
The DiT stays frozen; only the lightweight policy is trained.
"""

import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()


def _add_policy_config(config):
    """Add prior-policy-specific fields."""
    config.policy = policy = ml_collections.ConfigDict()
    policy.type = "gaussian"                 # "gaussian" or "normalizing_flow" (future)
    policy.hidden_dim = 512                  # hidden dim in policy network
    policy.learning_rate = 1e-4              # policy optimizer LR
    policy.weight_decay = 1e-4
    policy.temperature = 1.0                 # softmax temperature for advantage weighting
    policy.train_every_n_epochs = 1          # train policy every N epochs
    policy.kl_weight = 0.01                  # KL(π||N(0,I)) regularization to prevent variance collapse
    policy.cache_dir = "cache/prior_policy"  # disk cache for (prompt, noise, reward, advantage)
    policy.resume_path = ""                  # path to saved policy checkpoint
    # Transformer-specific (only used when type="transformer")
    policy.num_heads = 8                     # number of attention heads
    policy.num_layers = 4                    # number of transformer layers
    policy.spatial_res = 8                   # spatial resolution of query grid (8×8=64 queries)

    # Override training-related fields that are not needed for DiT
    config.use_lora = False
    config.train.ema = False
    return config


def pickscore_sd3_policy_1gpu():
    """1-GPU config for prior policy training with PickScore."""
    config = base.get_config()
    _add_policy_config(config)

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
    config.save_dir = "logs/prior_policy/pickscore_1gpu"

    config.reward_fn = {"pickscore": 1.0}
    config.prompt_fn = "general_ocr"
    config.per_prompt_stat_tracking = True
    config.sample.global_std = True
    return config


def pickscore_sd3_policy_4gpu():
    """4x A40 48GB config for prior policy training."""
    config = pickscore_sd3_policy_1gpu()
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = 8
    config.sample.test_batch_size = 16
    config.save_dir = "logs/prior_policy/pickscore_4gpu"
    return config


def pickscore_sd3_policy_8gpu_h20():
    """8x H20 96GB config for prior policy training."""
    config = pickscore_sd3_policy_1gpu()
    config.sample.train_batch_size = 16
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 8
    config.sample.test_batch_size = 32
    config.policy.cache_dir = "cache/prior_policy_8gpu"
    config.save_dir = "logs/prior_policy/pickscore_8gpu_h20"
    return config


def pickscore_sd3_transformer_8gpu_h20():
    """8x H20 96GB config for Transformer prior policy.

    Transformer policy (~23M params) with cross-attention over text tokens.
    8*16*8 = 1024 samples per epoch, 16 per prompt group.
    H20 96GB: DiT ~9GB + Transformer policy ~0.1GB + batch data ~2GB = ~11GB per card.
    """
    config = pickscore_sd3_policy_1gpu()
    config.policy.type = "transformer"
    config.policy.hidden_dim = 512
    config.policy.num_heads = 8
    config.policy.num_layers = 4
    config.policy.spatial_res = 8
    config.policy.learning_rate = 3e-5       # lower LR for Transformer (vs 1e-4 for MLP)
    config.policy.temperature = 0.5          # sharper advantage weighting for stronger signal
    config.policy.kl_weight = 0.01           # prevent variance collapse

    config.sample.train_batch_size = 16
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 8
    config.sample.test_batch_size = 32

    config.num_epochs = 200
    config.eval_freq = 10
    config.save_freq = 20
    config.policy.cache_dir = "cache/prior_policy_transformer_8gpu"
    config.save_dir = "logs/prior_policy/transformer_8gpu_h20"
    return config


def pickscore_sd3_policy_4gpu_smoke():
    """Minimal smoke test."""
    config = pickscore_sd3_policy_1gpu()
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 2
    config.sample.num_batches_per_epoch = 1
    config.sample.num_steps = 5
    config.sample.eval_num_steps = 5
    config.sample.test_batch_size = 2
    config.num_epochs = 2
    config.eval_freq = 1
    config.save_freq = 2
    config.save_dir = "logs/prior_policy/smoke_4gpu"
    config.policy.cache_dir = "cache/prior_policy_smoke"
    return config
