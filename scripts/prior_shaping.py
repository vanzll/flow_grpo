"""
Prior Shaping for Flow Matching models.

This script keeps the DiT frozen and optimizes the noise prior distribution.
Two update methods: reward-weighted (default, on-policy, GRPO-style) and CEM.
Since Flow Matching is an ODE, a fixed noise deterministically maps to a fixed
image, so noise->reward mappings are permanently valid and cached to disk.

Usage:
    # 1 GPU
    accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
        --num_processes=1 --main_process_port 29501 \
        scripts/prior_shaping.py --config config/prior_shaping.py:pickscore_sd3_prior_1gpu

    # 4 GPU (A40, needs NCCL_P2P_DISABLE=1)
    NCCL_P2P_DISABLE=1 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
        --num_processes=4 --main_process_port 29501 \
        scripts/prior_shaping.py --config config/prior_shaping.py:pickscore_sd3_prior_4gpu
"""

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
import numpy as np
import flow_grpo.rewards
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from flow_grpo.prior import GaussianPrior, ParticlePrior, RewardCache
import torch
import torch.distributed as dist
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/prior_shaping.py", "Prior shaping configuration.")

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset classes (reused from train_sd3.py)
# ---------------------------------------------------------------------------

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


# ---------------------------------------------------------------------------
# Text embedding helper (reused from train_sd3.py)
# ---------------------------------------------------------------------------

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


# ---------------------------------------------------------------------------
# Prior broadcast across distributed processes
# ---------------------------------------------------------------------------

def broadcast_prior(prior, accelerator: Accelerator):
    """Broadcast prior parameters from rank 0 to all processes."""
    if accelerator.num_processes <= 1:
        return
    if isinstance(prior, GaussianPrior):
        mu = prior.mu.to(accelerator.device)
        sigma2 = prior.sigma2.to(accelerator.device)
        dist.broadcast(mu, src=0)
        dist.broadcast(sigma2, src=0)
        prior.mu = mu.cpu()
        prior.sigma2 = sigma2.cpu()
    elif isinstance(prior, ParticlePrior):
        # Broadcast buffer size first (rank 0 sends, others receive)
        if accelerator.is_main_process:
            size = torch.tensor([len(prior.noises) if prior.noises is not None else 0],
                                device=accelerator.device)
        else:
            size = torch.tensor([0], device=accelerator.device)
        dist.broadcast(size, src=0)
        n = size.item()
        if n > 0:
            # Rank 0: send actual data; others: allocate matching receive buffer
            if accelerator.is_main_process:
                noises = prior.noises.to(accelerator.device)
                weights = prior.weights.to(accelerator.device)
            else:
                noises = torch.zeros(n, *prior.shape, device=accelerator.device)
                weights = torch.zeros(n, device=accelerator.device)
            dist.broadcast(noises, src=0)
            dist.broadcast(weights, src=0)
            prior.noises = noises.cpu()
            prior.weights = weights.cpu()


# ---------------------------------------------------------------------------
# Evaluation (mirrors train_sd3.py eval function, using vanilla pipeline)
# ---------------------------------------------------------------------------

def run_eval(pipeline, prior, test_dataloader, text_encoders, tokenizers, config,
             accelerator, epoch, reward_fn, executor, autocast, prefix="eval"):
    """Evaluate shaped prior. prefix controls W&B metric names."""
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)
    for test_batch in tqdm(
        test_dataloader,
        desc="Eval: ",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        batch_size = len(prompts)
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers,
            max_sequence_length=128, device=accelerator.device,
        )

        # Handle last batch potentially being smaller
        neg_embeds = sample_neg_prompt_embeds[:batch_size]
        neg_pooled = sample_neg_pooled_prompt_embeds[:batch_size]

        # Sample from shaped prior and generate via ODE
        noise = prior.sample(batch_size).to(accelerator.device, dtype=pipeline.transformer.dtype)
        with autocast():
            with torch.no_grad():
                images = pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=neg_embeds,
                    negative_pooled_prompt_embeds=neg_pooled,
                    latents=noise,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                ).images

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, _ = rewards_future.result()

        for key, value in rewards.items():
            gathered = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(gathered)

    # Log last batch images + aggregated rewards to W&B
    last_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_prompts_ids = tokenizers[0](
        prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_prompts_ids_gather = accelerator.gather(last_prompts_ids).cpu().numpy()
    last_prompts_gather = pipeline.tokenizer.batch_decode(last_prompts_ids_gather, skip_special_tokens=True)
    last_rewards_gather = {}
    for key, value in rewards.items():
        last_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}

    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_images_gather))
            for idx in range(num_samples):
                image = last_images_gather[idx]
                pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts = [last_prompts_gather[idx] for idx in range(num_samples)]
            sampled_rewards = [
                {k: last_rewards_gather[k][idx] for k in last_rewards_gather}
                for idx in range(num_samples)
            ]
            wandb.log(
                {
                    f"{prefix}_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(
                                f"{k}: {v:.2f}" for k, v in reward.items() if v != -10
                            ),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"{prefix}_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=epoch,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
    )

    if accelerator.is_main_process:
        wandb.init(project="flow_grpo", name=f"{config.prior.update_method}_{config.run_name}")
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    # ---- Load model (frozen, no LoRA) ----
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # Mixed precision dtype
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device, dtype=inference_dtype)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---- Prior ----
    latent_channels = pipeline.transformer.config.in_channels
    latent_h = config.resolution // pipeline.vae_scale_factor
    latent_w = config.resolution // pipeline.vae_scale_factor
    latent_shape = (latent_channels, latent_h, latent_w)

    if config.prior.update_method == "particle":
        prior = ParticlePrior(
            shape=latent_shape,
            device=accelerator.device,
            dtype=inference_dtype,
            perturbation_std=config.prior.perturbation_std,
            temperature=config.prior.temperature,
            mix_ratio=config.prior.mix_ratio,
        )
    else:
        prior = GaussianPrior(
            shape=latent_shape,
            device=accelerator.device,
            dtype=inference_dtype,
            regularization_mode=config.prior.regularization_mode,
            alpha=config.prior.alpha,
            kl_max=config.prior.kl_max,
        )
    if config.prior.resume_prior_path:
        prior.load(config.prior.resume_prior_path)
        logger.info(f"Resumed prior from {config.prior.resume_prior_path}")

    cache = RewardCache(config.prior.cache_dir)

    # ---- Reward function ----
    reward_fn = flow_grpo.rewards.multi_score(accelerator.device, config.reward_fn)
    eval_reward_fn = flow_grpo.rewards.multi_score(accelerator.device, config.reward_fn)
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # ---- Datasets ----
    train_dataset = TextPromptDataset(config.dataset, 'train')
    test_dataset = TextPromptDataset(config.dataset, 'test')

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=TextPromptDataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=TextPromptDataset.collate_fn,
        shuffle=False,
        num_workers=8,
    )

    # Train eval dataloader (first 2048 prompts of train set)
    from torch.utils.data import Subset
    train_eval_size = min(2048, len(train_dataset))
    train_eval_subset = Subset(train_dataset, range(train_eval_size))
    train_eval_dataloader = DataLoader(
        train_eval_subset, batch_size=config.sample.test_batch_size,
        collate_fn=TextPromptDataset.collate_fn, shuffle=False, num_workers=8,
    )

    # Wrap dataloaders with accelerator (for distributed)
    train_dataloader, test_dataloader, train_eval_dataloader = accelerator.prepare(
        train_dataloader, test_dataloader, train_eval_dataloader
    )

    # Negative prompt embeddings for CFG
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)

    # autocast is needed for mixed precision pipeline calls (fp16 latents + fp32 VAE)
    autocast = accelerator.autocast

    # ---- Logging ----
    samples_per_epoch = (
        config.sample.train_batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    )
    logger.info("***** Running prior shaping *****")
    logger.info(f"  Update method = {config.prior.update_method}")
    logger.info(f"  Batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Total samples per epoch = {samples_per_epoch}")
    logger.info(f"  Regularization mode = {config.prior.regularization_mode}")
    logger.info(f"  KL max = {config.prior.kl_max}")
    logger.info(f"  Prior latent shape = {latent_shape}")

    train_iter = iter(train_dataloader)

    # ===========================================================================
    # Main loop
    # ===========================================================================
    for epoch in range(config.num_epochs):
        # ---- Eval (periodic) ----
        if epoch % config.eval_freq == 0:
            run_eval(
                pipeline, prior, test_dataloader, text_encoders, tokenizers,
                config, accelerator, epoch, eval_reward_fn, executor, autocast,
                prefix="eval",
            )
            run_eval(
                pipeline, prior, train_eval_dataloader, text_encoders, tokenizers,
                config, accelerator, epoch, eval_reward_fn, executor, autocast,
                prefix="train_eval",
            )

        # ---- Save prior (periodic) ----
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            prior.save(os.path.join(config.save_dir, "checkpoints", f"prior_epoch_{epoch:06d}.pt"))

        # ---- Sample phase ----
        epoch_noises = []
        epoch_rewards = []
        reward_futures = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers,
                max_sequence_length=128, device=accelerator.device,
            )

            # Sample from shaped prior
            noise = prior.sample(len(prompts)).to(accelerator.device, dtype=inference_dtype)

            # Generate via vanilla SD3 pipeline (ODE, deterministic)
            with autocast():
                with torch.no_grad():
                    images = pipeline(
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[:len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[:len(prompts)],
                        latents=noise,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                    ).images

            # Async reward computation
            reward_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)
            reward_futures.append(reward_future)
            epoch_noises.append(noise.cpu())

        # Wait for all rewards
        for future in tqdm(
            reward_futures,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, _ = future.result()
            epoch_rewards.append(
                torch.as_tensor(rewards["avg"], device=accelerator.device).float()
            )

        # ---- Gather across GPUs ----
        all_noises = accelerator.gather(torch.cat(epoch_noises).to(accelerator.device))
        all_rewards = accelerator.gather(torch.cat(epoch_rewards))

        all_noises_cpu = all_noises.cpu()
        all_rewards_np = all_rewards.cpu().numpy()

        # ---- Cache to disk (always, regardless of update method) ----
        if accelerator.is_main_process:
            cache.append(all_noises_cpu, all_rewards_np, epoch)

        # ---- Prior update (rank 0) ----
        update_stats = {}
        if accelerator.is_main_process:
            if config.prior.update_method == "reward_weighted":
                # On-policy: only use current epoch's samples (analogous to GRPO)
                update_stats = prior.update_reward_weighted(
                    all_noises_cpu,
                    all_rewards_np,
                    temperature=config.prior.temperature,
                )
            elif config.prior.update_method == "cem":
                # CEM: can use historical cached data
                if config.prior.use_cache_history:
                    hist_noises, hist_rewards = cache.load_recent(
                        max_epochs=config.prior.max_cache_epochs
                    )
                else:
                    hist_noises = all_noises_cpu
                    hist_rewards = all_rewards_np
                update_stats = prior.update_cem(
                    hist_noises,
                    hist_rewards,
                    elite_ratio=config.prior.elite_ratio,
                    temperature=config.prior.temperature,
                )
            elif config.prior.update_method == "particle":
                # Particle: rebuild buffer from all cached data
                update_stats = prior.update_from_cache(
                    cache, max_epochs=config.prior.max_cache_epochs
                )
            else:
                raise ValueError(f"Unknown update method: {config.prior.update_method}")

        # Broadcast updated prior to all GPUs
        broadcast_prior(prior, accelerator)

        # ---- Log ----
        if accelerator.is_main_process:
            log_dict = {
                "epoch": epoch,
                "reward_mean": float(all_rewards_np.mean()),
                "reward_std": float(all_rewards_np.std()),
                "reward_max": float(all_rewards_np.max()),
                "reward_min": float(all_rewards_np.min()),
                "cache_total_samples": cache.total_samples,
            }
            if isinstance(prior, GaussianPrior):
                log_dict["prior_kl"] = prior.kl_from_standard_normal()
                log_dict["prior_mu_norm"] = float(prior.mu.norm())
                log_dict["prior_sigma_mean"] = float(prior.sigma2.sqrt().mean())
            log_dict.update({f"update_{k}": v for k, v in update_stats.items()})
            wandb.log(log_dict, step=epoch)

            if isinstance(prior, GaussianPrior):
                logger.info(
                    f"Epoch {epoch}: reward={all_rewards_np.mean():.4f} +/- {all_rewards_np.std():.4f}, "
                    f"KL={prior.kl_from_standard_normal():.4f}, cache={cache.total_samples}"
                )
            else:
                logger.info(
                    f"Epoch {epoch}: reward={all_rewards_np.mean():.4f} +/- {all_rewards_np.std():.4f}, "
                    f"buffer={update_stats.get('buffer_size', 0)}, cache={cache.total_samples}"
                )

    # Final save
    if accelerator.is_main_process:
        prior.save(os.path.join(config.save_dir, "prior_final.pt"))
        logger.info("Training complete. Final prior saved.")


if __name__ == "__main__":
    app.run(main)
