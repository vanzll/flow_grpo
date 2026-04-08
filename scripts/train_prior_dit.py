"""
Train a small Flow Matching DiT for prior noise generation.

The big DiT stays frozen. We train a small DiT that transforms N(0,I) →
prompt-conditioned "good noise" via multi-step ODE. Training uses
advantage-weighted flow matching MSE (DiffusionNFT-inspired).

Usage:
    # 4 GPU
    NCCL_P2P_DISABLE=1 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
        --num_processes=4 --main_process_port 29501 \
        scripts/train_prior_dit.py --config config/prior_dit.py:pickscore_sd3_dit_4gpu
"""

from collections import defaultdict
import contextlib
import math
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
from flow_grpo.prior_dit import PriorDiT, compute_dit_awr_loss
from flow_grpo.stat_tracking import PerPromptStatTracker
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler, Subset

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/prior_dit.py", "Prior DiT configuration.")

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

def run_eval(pipeline, prior_dit, test_dataloader, text_encoders, tokenizers, config,
             accelerator, epoch, reward_fn, executor, autocast,
             neg_prompt_embed, neg_pooled_prompt_embed, prefix="eval"):
    """Evaluate prior DiT on a dataset."""
    prior_dit.eval()
    sample_neg_pe = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_ppe = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)
    for test_batch in tqdm(
        test_dataloader, desc=f"{prefix}: ",
        disable=not accelerator.is_local_main_process, position=0,
    ):
        prompts, prompt_metadata = test_batch
        batch_size = len(prompts)
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers,
            max_sequence_length=128, device=accelerator.device,
        )

        # Sample from prior DiT (multi-step ODE with CFG)
        with torch.no_grad():
            z, _ = prior_dit.sample(
                prompt_embeds, pooled_prompt_embeds,
                num_steps=config.prior_dit.num_steps,
                cfg_scale=config.prior_dit.cfg_scale,
                neg_prompt_embeds=sample_neg_pe[:batch_size],
                neg_pooled_prompt_embeds=sample_neg_ppe[:batch_size],
            )
            z = z.to(dtype=pipeline.transformer.dtype)

        # Generate via frozen big DiT
        with autocast():
            with torch.no_grad():
                images = pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_pe[:batch_size],
                    negative_pooled_prompt_embeds=sample_neg_ppe[:batch_size],
                    latents=z,
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

    # Log to W&B
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
    prior_dit.train()


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
        wandb.init(project="flow_grpo", name=f"prior_dit_{config.run_name}")
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    # ---- Load frozen big DiT ----
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
        position=1, disable=not accelerator.is_local_main_process,
        leave=False, desc="Timestep", dynamic_ncols=True,
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

    # ---- Build small DiT ----
    dit_config = config.prior_dit
    prior_dit = PriorDiT(
        sample_size=latent_h,
        patch_size=dit_config.patch_size,
        in_channels=latent_channels,
        out_channels=latent_channels,
        num_layers=dit_config.num_layers,
        num_attention_heads=dit_config.num_attention_heads,
        attention_head_dim=dit_config.attention_head_dim,
        joint_attention_dim=4096,   # SD3's text embed dim
        pooled_projection_dim=2048, # SD3's pooled embed dim
    ).to(accelerator.device)

    if dit_config.resume_path:
        prior_dit.load_state_dict(torch.load(dit_config.resume_path, map_location=accelerator.device))
        logger.info(f"Resumed prior DiT from {dit_config.resume_path}")

    num_params = sum(p.numel() for p in prior_dit.parameters())
    logger.info(f"Prior DiT parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        prior_dit.parameters(),
        lr=dit_config.learning_rate,
        weight_decay=dit_config.weight_decay,
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
        rank=accelerator.process_index, seed=42,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=1, collate_fn=TextPromptDataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.sample.test_batch_size,
        collate_fn=TextPromptDataset.collate_fn, shuffle=False, num_workers=8,
    )

    # Train eval (first 2048 prompts)
    train_eval_size = min(2048, len(train_dataset))
    train_eval_subset = Subset(train_dataset, range(train_eval_size))
    train_eval_dataloader = DataLoader(
        train_eval_subset, batch_size=config.sample.test_batch_size,
        collate_fn=TextPromptDataset.collate_fn, shuffle=False, num_workers=8,
    )

    # Wrap with accelerator
    prior_dit, optimizer, train_dataloader, test_dataloader, train_eval_dataloader = accelerator.prepare(
        prior_dit, optimizer, train_dataloader, test_dataloader, train_eval_dataloader
    )

    # Negative prompt embeddings
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)

    # Stat tracker
    if config.sample.num_image_per_prompt > 1 and config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        stat_tracker = None

    autocast = accelerator.autocast

    # Cache
    cache_dir = dit_config.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # Logging
    samples_per_epoch = (
        config.sample.train_batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    )
    logger.info("***** Running prior DiT training *****")
    logger.info(f"  Prior DiT params = {num_params:,}")
    logger.info(f"  ODE steps (small DiT) = {dit_config.num_steps}")
    logger.info(f"  Batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Total samples per epoch = {samples_per_epoch}")
    logger.info(f"  Latent shape = ({latent_channels}, {latent_h}, {latent_w})")

    train_iter = iter(train_dataloader)
    global_step = 0

    # ===========================================================================
    # Main training loop
    # ===========================================================================
    for epoch in range(config.num_epochs):
        # ---- Eval (periodic) ----
        if epoch % config.eval_freq == 0:
            unwrapped_dit = accelerator.unwrap_model(prior_dit)
            run_eval(
                pipeline, unwrapped_dit, test_dataloader, text_encoders, tokenizers,
                config, accelerator, epoch, eval_reward_fn, executor, autocast,
                neg_prompt_embed, neg_pooled_prompt_embed, prefix="eval",
            )
            run_eval(
                pipeline, unwrapped_dit, train_eval_dataloader, text_encoders, tokenizers,
                config, accelerator, epoch, eval_reward_fn, executor, autocast,
                neg_prompt_embed, neg_pooled_prompt_embed, prefix="train_eval",
            )

        # ---- Save (periodic) ----
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_path = os.path.join(config.save_dir, "checkpoints", f"dit_epoch_{epoch:06d}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(accelerator.unwrap_model(prior_dit).state_dict(), save_path)

        # ---- Sampling phase ----
        # z sampled directly from N(0,I), NOT through small DiT
        # ε for flow matching training is sampled independently during training
        epoch_noises = []    # z ~ N(0,I), the big DiT's prior
        epoch_rewards = []
        epoch_prompt_embeds = []
        epoch_pooled_embeds = []
        epoch_prompts = []
        reward_futures = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process, position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers,
                max_sequence_length=128, device=accelerator.device,
            )

            # Sample z directly from N(0,I) — this is the big DiT's prior
            # NOT through small DiT (small DiT is only used at inference time)
            z = torch.randn(
                len(prompts), latent_channels, latent_h, latent_w,
                device=accelerator.device, dtype=inference_dtype,
            )

            # Generate via frozen big DiT
            with autocast():
                with torch.no_grad():
                    images = pipeline(
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[:len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[:len(prompts)],
                        latents=z,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                    ).images

            # Async reward
            reward_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)
            reward_futures.append(reward_future)

            epoch_noises.append(z.float().cpu())
            epoch_prompt_embeds.append(prompt_embeds.float().cpu())
            epoch_pooled_embeds.append(pooled_prompt_embeds.float().cpu())
            epoch_prompts.extend(prompts)

        # Wait for rewards
        for future in tqdm(
            reward_futures, desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process, position=0,
        ):
            rewards, _ = future.result()
            epoch_rewards.append(
                torch.as_tensor(rewards["avg"], device=accelerator.device).float()
            )

        # ---- Gather ONLY rewards for advantage computation (like Flow-GRPO) ----
        local_noises = torch.cat(epoch_noises)      # z ~ N(0,I), this GPU's data only
        local_rewards = torch.cat(epoch_rewards)
        local_prompt_embeds = torch.cat(epoch_prompt_embeds)
        local_pooled_embeds = torch.cat(epoch_pooled_embeds)

        gathered_rewards = accelerator.gather(local_rewards).cpu().numpy()

        prompt_ids = tokenizers[0](
            epoch_prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt",
        ).input_ids.to(accelerator.device)
        all_prompt_ids = accelerator.gather(prompt_ids).cpu().numpy()
        all_prompts = pipeline.tokenizer.batch_decode(all_prompt_ids, skip_special_tokens=True)

        # ---- Compute advantages on gathered rewards (GRPO-style) ----
        if stat_tracker is not None:
            advantages = stat_tracker.update(all_prompts, gathered_rewards)
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards - gathered_rewards.mean()) / (gathered_rewards.std() + 1e-4)

        advantages = torch.as_tensor(advantages, dtype=torch.float32)
        local_advantages = (
            advantages.reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        # ---- Cache to disk (all ranks participate in gather for noises) ----
        all_noises = accelerator.gather(local_noises.to(accelerator.device)).cpu()
        if accelerator.is_main_process:
            cache_path = os.path.join(cache_dir, f"epoch_{epoch:06d}.npz")
            np.savez_compressed(
                cache_path,
                noises=all_noises.to(torch.float16).numpy(),
                rewards=gathered_rewards.astype(np.float32),
                advantages=advantages.numpy().astype(np.float32),
                prompt_ids=all_prompt_ids.astype(np.int32),
            )

        # ---- Train prior DiT on LOCAL shard (proper DDP) ----
        if epoch % dit_config.train_every_n_epochs == 0:
            prior_dit.train()
            optimizer.zero_grad()

            N = len(local_noises)  # per-GPU sample count
            mini_bs = min(16, N)
            num_mini = (N + mini_bs - 1) // mini_bs
            total_loss = 0.0
            last_stats = {}

            for mi in range(num_mini):
                s = mi * mini_bs
                e = min(s + mini_bs, N)
                mb_z = local_noises[s:e].to(accelerator.device)
                # ε sampled INDEPENDENTLY for flow matching (not paired with z)
                mb_eps = torch.randn_like(mb_z)
                mb_pe = local_prompt_embeds[s:e].to(accelerator.device)
                mb_ppe = local_pooled_embeds[s:e].to(accelerator.device)
                mb_adv = local_advantages[s:e]

                mb_size = e - s
                mb_null_pe = neg_prompt_embed.expand(mb_size, -1, -1)
                mb_null_ppe = neg_pooled_prompt_embed.expand(mb_size, -1)

                loss, last_stats = compute_dit_awr_loss(
                    prior_dit, mb_eps, mb_z, mb_pe, mb_ppe, mb_adv,
                    temperature=dit_config.temperature,
                    cfg_drop_rate=dit_config.cfg_drop_rate,
                    v_reg_weight=dit_config.v_reg_weight,
                    adv_clip_max=config.train.adv_clip_max,
                    null_prompt_embeds=mb_null_pe,
                    null_pooled_prompt_embeds=mb_null_ppe,
                )
                loss = loss / num_mini
                accelerator.backward(loss)
                total_loss += loss.item()

            total_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in prior_dit.parameters() if p.grad is not None],
                max_norm=1.0, norm_type=2.0, error_if_nonfinite=False,
            )
            grad_norm = total_norm if isinstance(total_norm, torch.Tensor) else torch.tensor(total_norm)
            optimizer.step()
            global_step += 1
            dit_stats = last_stats
            dit_stats["dit_loss"] = total_loss

            # ---- Log ----
            if accelerator.is_main_process:
                with torch.no_grad():
                    z_norm = local_noises[:8].flatten(1).norm(dim=1).mean()

                log_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "reward_mean": float(gathered_rewards.mean()),
                    "reward_std": float(gathered_rewards.std()),
                    "reward_max": float(gathered_rewards.max()),
                    "reward_min": float(gathered_rewards.min()),
                    "advantage_mean": float(advantages.mean()),
                    "advantage_std": float(advantages.std()),
                    "dit_loss": dit_stats["dit_loss"],
                    "dit_awr_loss": dit_stats["dit_awr_loss"],
                    "dit_v_reg": dit_stats["dit_v_reg"],
                    "dit_mse_mean": dit_stats["dit_mse_mean"],
                    "z_norm": float(z_norm),
                    "v_pred_norm": dit_stats["v_pred_norm"],
                    "v_target_norm": dit_stats["v_target_norm"],
                    "grad_norm": float(grad_norm),
                }
                wandb.log(log_dict, step=epoch)

                logger.info(
                    f"Epoch {epoch}: reward={gathered_rewards.mean():.4f} ± {gathered_rewards.std():.4f}, "
                    f"dit_loss={dit_stats['dit_loss']:.4f}, mse={dit_stats['dit_mse_mean']:.4f}, "
                    f"z_norm={float(z_norm):.1f}"
                )

    # Final save
    if accelerator.is_main_process:
        save_path = os.path.join(config.save_dir, "dit_final.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(accelerator.unwrap_model(prior_dit).state_dict(), save_path)
        logger.info("Training complete. Final DiT saved.")


if __name__ == "__main__":
    app.run(main)
