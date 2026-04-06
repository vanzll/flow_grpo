"""
Train a lightweight prior policy network with advantage-weighted regression.

The DiT stays frozen. We train a small network π_φ(z|prompt) that outputs
prompt-conditioned noise distributions. Pipeline mirrors train_sd3.py:
  sample from policy → frozen DiT (ODE) → reward → advantage → train policy

Usage:
    # 1 GPU
    accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
        --num_processes=1 --main_process_port 29501 \
        scripts/train_prior_policy.py --config config/prior_policy.py:pickscore_sd3_policy_1gpu

    # 4 GPU
    NCCL_P2P_DISABLE=1 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
        --num_processes=4 --main_process_port 29501 \
        scripts/train_prior_policy.py --config config/prior_policy.py:pickscore_sd3_policy_4gpu
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
from flow_grpo.prior_policy import GaussianPolicy, compute_awr_loss
from flow_grpo.stat_tracking import PerPromptStatTracker
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/prior_policy.py", "Prior policy configuration.")

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


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(pipeline, policy, test_dataloader, text_encoders, tokenizers, config,
             accelerator, epoch, reward_fn, executor, autocast):
    """Evaluate policy on test set."""
    policy.eval()
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

        neg_embeds = sample_neg_prompt_embeds[:batch_size]
        neg_pooled = sample_neg_pooled_prompt_embeds[:batch_size]

        # Sample from policy
        with torch.no_grad():
            noise = policy.sample(pooled_prompt_embeds, prompt_embeds)
            noise = noise.to(dtype=pipeline.transformer.dtype)

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

    # Log last batch images + rewards to W&B
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
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(
                                f"{k}: {v:.2f}" for k, v in reward.items() if v != -10
                            ),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=epoch,
            )
    policy.train()


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
        wandb.init(project="flow_grpo", name=f"prior_policy_{config.run_name}")
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    # ---- Load frozen DiT ----
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

    # ---- Latent shape ----
    latent_channels = pipeline.transformer.config.in_channels
    latent_h = config.resolution // pipeline.vae_scale_factor
    latent_w = config.resolution // pipeline.vae_scale_factor
    latent_shape = (latent_channels, latent_h, latent_w)

    # ---- Get embedding dims from a dummy forward pass ----
    dummy_pe, dummy_ppe = compute_text_embeddings(
        ["dummy"], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )
    seq_embed_dim = dummy_pe.shape[-1]       # 4096 for SD3.5
    pooled_embed_dim = dummy_ppe.shape[-1]   # 2048 for SD3.5
    del dummy_pe, dummy_ppe

    # ---- Policy network ----
    policy = GaussianPolicy(
        prompt_embed_dim=pooled_embed_dim,
        seq_embed_dim=seq_embed_dim,
        latent_shape=latent_shape,
        hidden_dim=config.policy.hidden_dim,
    ).to(accelerator.device)

    if config.policy.resume_path:
        policy.load_state_dict(torch.load(config.policy.resume_path, map_location=accelerator.device))
        logger.info(f"Resumed policy from {config.policy.resume_path}")

    num_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Policy parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.policy.learning_rate,
        weight_decay=config.policy.weight_decay,
    )

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
        train_dataset, batch_sampler=train_sampler,
        num_workers=1, collate_fn=TextPromptDataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.sample.test_batch_size,
        collate_fn=TextPromptDataset.collate_fn, shuffle=False, num_workers=8,
    )

    # Wrap with accelerator
    policy, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        policy, optimizer, train_dataloader, test_dataloader
    )

    # Negative prompt embeddings for CFG
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)

    # Stat tracker for GRPO advantages
    if config.sample.num_image_per_prompt > 1 and config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        stat_tracker = None

    autocast = accelerator.autocast

    # Cache setup
    cache_dir = config.policy.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # ---- Logging ----
    samples_per_epoch = (
        config.sample.train_batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    )
    logger.info("***** Running prior policy training *****")
    logger.info(f"  Policy type = {config.policy.type}")
    logger.info(f"  Policy params = {num_params:,}")
    logger.info(f"  Batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Total samples per epoch = {samples_per_epoch}")
    logger.info(f"  Num image per prompt = {config.sample.num_image_per_prompt}")
    logger.info(f"  Latent shape = {latent_shape}")

    train_iter = iter(train_dataloader)
    global_step = 0

    # ===========================================================================
    # Main training loop
    # ===========================================================================
    for epoch in range(config.num_epochs):
        # ---- Eval (periodic) ----
        if epoch % config.eval_freq == 0:
            unwrapped_policy = accelerator.unwrap_model(policy)
            run_eval(
                pipeline, unwrapped_policy, test_dataloader, text_encoders, tokenizers,
                config, accelerator, epoch, eval_reward_fn, executor, autocast,
            )

        # ---- Save (periodic) ----
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_path = os.path.join(config.save_dir, "checkpoints", f"policy_epoch_{epoch:06d}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(accelerator.unwrap_model(policy).state_dict(), save_path)

        # ---- Sampling phase ----
        policy.eval()
        epoch_noises = []
        epoch_rewards = []
        epoch_prompt_embeds = []
        epoch_pooled_embeds = []
        epoch_prompts = []
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

            # Sample from policy (no grad during rollout)
            with torch.no_grad():
                noise = accelerator.unwrap_model(policy).sample(
                    pooled_prompt_embeds, prompt_embeds
                ).to(dtype=inference_dtype)

            # Generate via frozen DiT
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

            epoch_noises.append(noise.float().cpu())
            epoch_prompt_embeds.append(prompt_embeds.float().cpu())
            epoch_pooled_embeds.append(pooled_prompt_embeds.float().cpu())
            epoch_prompts.extend(prompts)

        # Wait for rewards
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
        all_noises = accelerator.gather(torch.cat(epoch_noises).to(accelerator.device)).cpu()
        all_rewards = accelerator.gather(torch.cat(epoch_rewards)).cpu().numpy()
        all_prompt_embeds = accelerator.gather(torch.cat(epoch_prompt_embeds).to(accelerator.device)).cpu()
        all_pooled_embeds = accelerator.gather(torch.cat(epoch_pooled_embeds).to(accelerator.device)).cpu()

        # Gather prompts via tokenization roundtrip
        prompt_ids = tokenizers[0](
            epoch_prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt",
        ).input_ids.to(accelerator.device)
        all_prompt_ids = accelerator.gather(prompt_ids).cpu().numpy()
        all_prompts = pipeline.tokenizer.batch_decode(all_prompt_ids, skip_special_tokens=True)

        # ---- Compute advantages (GRPO-style) ----
        if stat_tracker is not None:
            advantages = stat_tracker.update(all_prompts, all_rewards)
            stat_tracker.clear()
        else:
            advantages = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-4)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # ---- Cache to disk ----
        if accelerator.is_main_process:
            cache_path = os.path.join(cache_dir, f"epoch_{epoch:06d}.npz")
            np.savez_compressed(
                cache_path,
                noises=all_noises.to(torch.float16).numpy(),
                rewards=all_rewards.astype(np.float32),
                advantages=advantages.numpy().astype(np.float32),
                prompt_ids=all_prompt_ids.astype(np.int32),
            )

        # ---- Train policy (advantage-weighted regression) ----
        if epoch % config.policy.train_every_n_epochs == 0:
            policy.train()

            # Move data to device
            train_noises = all_noises.to(accelerator.device)
            train_pe = all_prompt_embeds.to(accelerator.device)
            train_ppe = all_pooled_embeds.to(accelerator.device)
            train_advantages = advantages.to(accelerator.device)

            # Forward + loss (use DDP-wrapped policy for gradient sync)
            loss, awr_stats = compute_awr_loss(
                policy,
                train_noises,
                train_ppe,
                train_pe,
                train_advantages,
                temperature=config.policy.temperature,
            )

            # Optional KL regularization
            if config.policy.kl_weight > 0:
                mu_kl, log_sigma_kl = policy(train_ppe, train_pe)
                sigma2_kl = (2 * log_sigma_kl).exp()
                kl = 0.5 * (sigma2_kl + mu_kl ** 2 - 1 - 2 * log_sigma_kl).sum(dim=(1,2,3)).mean()
                loss = loss + config.policy.kl_weight * kl

            # Backward
            optimizer.zero_grad()
            accelerator.backward(loss)
            # Manual grad clip (accelerator.clip_grad_norm_ has bugs with 4D conv params)
            total_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.grad is not None],
                max_norm=1.0, norm_type=2.0, error_if_nonfinite=False,
            )
            grad_norm = total_norm if isinstance(total_norm, torch.Tensor) else torch.tensor(total_norm)
            optimizer.step()
            global_step += 1

            # ---- Log ----
            if accelerator.is_main_process:
                with torch.no_grad():
                    unwrapped = accelerator.unwrap_model(policy)
                    mu, log_sigma = unwrapped(train_ppe[:8], train_pe[:8])
                    policy_kl = unwrapped.kl_from_standard_normal(train_ppe[:8], train_pe[:8])
                    policy_entropy = unwrapped.entropy(train_ppe[:8], train_pe[:8])

                log_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    # Reward
                    "reward_mean": float(all_rewards.mean()),
                    "reward_std": float(all_rewards.std()),
                    "reward_max": float(all_rewards.max()),
                    "reward_min": float(all_rewards.min()),
                    # Advantage
                    "advantage_mean": float(advantages.mean()),
                    "advantage_std": float(advantages.std()),
                    # Policy
                    "policy_loss": awr_stats["policy_loss"],
                    "policy_log_prob_mean": awr_stats["log_prob_mean"],
                    "policy_mu_norm": float(mu.flatten(1).norm(dim=1).mean()),
                    "policy_sigma_mean": float(log_sigma.exp().mean()),
                    "policy_kl_from_n01": float(policy_kl),
                    "policy_entropy": float(policy_entropy),
                    "effective_sample_size": awr_stats["effective_sample_size"],
                    "grad_norm": float(grad_norm),
                }
                wandb.log(log_dict, step=epoch)

                logger.info(
                    f"Epoch {epoch}: reward={all_rewards.mean():.4f} ± {all_rewards.std():.4f}, "
                    f"loss={awr_stats['policy_loss']:.4f}, "
                    f"KL={float(policy_kl):.2f}, σ={float(log_sigma.exp().mean()):.4f}"
                )

    # Final save
    if accelerator.is_main_process:
        save_path = os.path.join(config.save_dir, "policy_final.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(accelerator.unwrap_model(policy).state_dict(), save_path)
        logger.info("Training complete. Final policy saved.")


if __name__ == "__main__":
    app.run(main)
