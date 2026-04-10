"""
Train a small prompt-conditioned Prior DiT offline from cached prompt/noise tuples.

The big DiT stays frozen. Cached samples provide target prior latents z together
with rewards / advantages. We train a small Flow Matching DiT that maps
epsilon ~ N(0, I) to high-value z using a positive-weighted supervised loss,
then periodically evaluate it by sampling z from the small DiT and feeding it
through the frozen big DiT.

Usage:
    NCCL_P2P_DISABLE=1 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
        --num_processes=8 --main_process_port 29511 \
        scripts/train_prior_dit_offline.py --config config/prior_dit_offline.py:pickscore_sd3_dit_offline_8gpu_h20
"""

from collections import defaultdict
from concurrent import futures
from functools import partial
import datetime
import os
import tempfile
import time

from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import StableDiffusion3Pipeline
from ml_collections import config_flags
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset
import tqdm
import wandb

import flow_grpo.rewards
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from flow_grpo.prior_dit import PriorDiT

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    "config/prior_dit_offline.py",
    "Offline Prior DiT configuration.",
)

logger = get_logger(__name__)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r", encoding="utf-8") as f:
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


class OfflineCacheIterableDataset(IterableDataset):
    def __init__(self, file_infos, rank, world_size, shuffle, seed):
        self.file_infos = list(file_infos)
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _assigned_files(self):
        return self.file_infos[self.rank::self.world_size]

    def __len__(self):
        return sum(length for _, length in self._assigned_files())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        assigned = list(self._assigned_files())
        if self.shuffle:
            rng.shuffle(assigned)

        for path, _ in assigned:
            with np.load(path) as data:
                if "prompt_ids" not in data:
                    raise KeyError(f"{path} does not contain prompt_ids.")
                if "advantages" not in data:
                    raise KeyError(f"{path} does not contain advantages.")

                indices = np.arange(len(data["rewards"]))
                if self.shuffle:
                    rng.shuffle(indices)

                for idx in indices:
                    yield {
                        "noises": data["noises"][idx].astype(np.float32),
                        "rewards": np.float32(data["rewards"][idx]),
                        "advantages": np.float32(data["advantages"][idx]),
                        "prompt_ids": data["prompt_ids"][idx].astype(np.int32),
                    }


def offline_cache_collate_fn(examples):
    return {
        "noises": torch.from_numpy(np.stack([x["noises"] for x in examples], axis=0)),
        "rewards": torch.from_numpy(np.asarray([x["rewards"] for x in examples], dtype=np.float32)),
        "advantages": torch.from_numpy(np.asarray([x["advantages"] for x in examples], dtype=np.float32)),
        "prompt_ids": torch.from_numpy(np.stack([x["prompt_ids"] for x in examples], axis=0)),
    }


def compute_text_embeddings(prompts, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompts, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def decode_prompts(prompt_ids, tokenizer):
    return tokenizer.batch_decode(prompt_ids.tolist(), skip_special_tokens=True)


def list_cache_files(cache_dir, max_cache_files=0):
    files = sorted(
        os.path.join(cache_dir, name)
        for name in os.listdir(cache_dir)
        if name.endswith(".npz")
    )
    if max_cache_files > 0:
        files = files[:max_cache_files]
    if not files:
        raise ValueError(f"No cache files found in {cache_dir}.")
    return files


def summarize_cache_files(files):
    file_infos = []
    for path in files:
        with np.load(path) as data:
            if "prompt_ids" not in data or "advantages" not in data:
                raise KeyError(f"{path} must contain prompt_ids and advantages.")
            file_infos.append((path, int(len(data["rewards"]))))
    return file_infos


def split_cache_files(file_infos, num_val_files, world_size):
    total = len(file_infos)
    num_val_files = min(max(0, num_val_files), total)
    train_infos = file_infos[: total - num_val_files]
    val_infos = file_infos[total - num_val_files :]

    def trim_to_world_size(infos):
        usable = (len(infos) // world_size) * world_size
        return infos[:usable], len(infos) - usable

    train_infos, train_dropped = trim_to_world_size(train_infos)
    val_infos, val_dropped = trim_to_world_size(val_infos)
    return train_infos, val_infos, train_dropped, val_dropped


def build_supervised_weights(rewards, advantages, config):
    offline = config.offline

    if offline.score_source == "advantages":
        scores = advantages.float()
    elif offline.score_source == "rewards":
        scores = rewards.float()
    elif offline.score_source == "uniform":
        scores = torch.ones_like(rewards, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported score_source: {offline.score_source}")

    if offline.score_source != "uniform" and offline.score_clip > 0:
        scores = scores.clamp(min=-offline.score_clip, max=offline.score_clip)

    temperature = max(float(offline.weight_temperature), 1e-6)

    if offline.weight_transform == "exp":
        weights = torch.exp(scores / temperature)
    elif offline.weight_transform == "relu":
        weights = torch.relu(scores / temperature)
    elif offline.weight_transform == "softplus":
        weights = F.softplus(scores / temperature)
    elif offline.weight_transform == "sigmoid":
        weights = torch.sigmoid(scores / temperature)
    elif offline.weight_transform == "identity":
        weights = scores
    elif offline.weight_transform == "uniform":
        weights = torch.ones_like(scores)
    else:
        raise ValueError(f"Unsupported weight_transform: {offline.weight_transform}")

    if offline.positive_only:
        weights = weights.clamp_min(0.0)

    weights = weights + float(offline.min_weight)
    if offline.normalize_weights:
        weights = weights / weights.mean().clamp_min(1e-6)
    return weights.float(), scores.float()


def compute_dit_supervised_loss(
    model,
    epsilon,
    z,
    prompt_embeds,
    pooled_prompt_embeds,
    sample_weights,
    cfg_drop_rate=0.0,
    v_reg_weight=0.01,
    null_prompt_embeds=None,
    null_pooled_prompt_embeds=None,
):
    batch_size = z.shape[0]
    t = torch.rand(batch_size, device=z.device, dtype=z.dtype)

    t_expand = t[:, None, None, None]
    z_t = (1 - t_expand) * epsilon + t_expand * z
    v_target = epsilon - z

    if cfg_drop_rate > 0 and model.training:
        drop_mask = torch.rand(batch_size, device=z.device) < cfg_drop_rate
        if drop_mask.any():
            prompt_embeds = prompt_embeds.clone()
            pooled_prompt_embeds = pooled_prompt_embeds.clone()
            if null_prompt_embeds is not None and null_pooled_prompt_embeds is not None:
                prompt_embeds[drop_mask] = null_prompt_embeds[drop_mask].to(prompt_embeds.dtype)
                pooled_prompt_embeds[drop_mask] = null_pooled_prompt_embeds[drop_mask].to(
                    pooled_prompt_embeds.dtype
                )
            else:
                prompt_embeds[drop_mask] = 0.0
                pooled_prompt_embeds[drop_mask] = 0.0

    timestep = (1 - t) * 1000
    v_pred = model(z_t, prompt_embeds, pooled_prompt_embeds, timestep)

    mse = ((v_pred - v_target) ** 2).mean(dim=(1, 2, 3))
    weights = sample_weights.float()
    supervised_loss = (weights * mse).mean()
    v_reg = (v_pred ** 2).mean()
    loss = supervised_loss + v_reg_weight * v_reg

    with torch.no_grad():
        stats = {
            "loss": loss.item(),
            "supervised_loss": supervised_loss.item(),
            "v_reg": v_reg.item(),
            "mse_mean": mse.mean().item(),
            "weight_mean": weights.mean().item(),
            "weight_max": weights.max().item(),
            "weight_min": weights.min().item(),
            "v_pred_norm": v_pred.flatten(1).norm(dim=1).mean().item(),
            "v_target_norm": v_target.flatten(1).norm(dim=1).mean().item(),
        }
    return loss, stats


def reduce_epoch_metrics(accelerator, metric_sums):
    keys = sorted(metric_sums.keys())
    tensor = torch.tensor([metric_sums[key] for key in keys], device=accelerator.device, dtype=torch.float64)
    gathered = accelerator.gather(tensor.unsqueeze(0))
    if not accelerator.is_main_process:
        return None
    reduced = gathered.sum(dim=0).cpu().tolist()
    return {key: value for key, value in zip(keys, reduced)}


def compute_grad_norm(parameters):
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    device = grads[0].device
    total = torch.zeros(1, device=device, dtype=torch.float32)
    for grad in grads:
        total += grad.float().pow(2).sum()
    return total.sqrt().squeeze(0)


def run_eval(
    pipeline,
    prior_dit,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    accelerator,
    epoch,
    reward_fn,
    executor,
    autocast,
    neg_prompt_embed,
    neg_pooled_prompt_embed,
    prefix="eval",
):
    prior_dit.eval()
    sample_neg_pe = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_ppe = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)
    for test_batch in tqdm(
        test_dataloader,
        desc=f"{prefix}: ",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        batch_size = len(prompts)
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=128,
            device=accelerator.device,
        )

        with torch.no_grad():
            z, _ = prior_dit.sample(
                prompt_embeds,
                pooled_prompt_embeds,
                num_steps=config.prior_dit.num_steps,
                cfg_scale=config.prior_dit.cfg_scale,
                neg_prompt_embeds=sample_neg_pe[:batch_size],
                neg_pooled_prompt_embeds=sample_neg_ppe[:batch_size],
            )
            z = z.to(dtype=pipeline.transformer.dtype)

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

    last_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_prompt_ids_gather = accelerator.gather(last_prompt_ids).cpu().numpy()
    last_prompts_gather = pipeline.tokenizer.batch_decode(last_prompt_ids_gather, skip_special_tokens=True)

    last_rewards_gather = {}
    for key, value in rewards.items():
        last_rewards_gather[key] = accelerator.gather(
            torch.as_tensor(value, device=accelerator.device)
        ).cpu().numpy()

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
                {key: last_rewards_gather[key][idx] for key in last_rewards_gather}
                for idx in range(num_samples)
            ]
            wandb.log(
                {
                    f"{prefix}_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{key}: {value:.2f}" for key, value in reward.items() if value != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{
                        f"{prefix}_reward_{key}": np.mean(value[value != -10])
                        for key, value in all_rewards.items()
                    },
                },
                step=epoch,
            )

    prior_dit.train()


def run_cache_validation(
    prior_dit,
    val_dataloader,
    text_encoders,
    tokenizers,
    config,
    accelerator,
    neg_prompt_embed,
    neg_pooled_prompt_embed,
):
    if val_dataloader is None:
        return None

    prior_dit.eval()
    metric_sums = defaultdict(float)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if config.offline.max_val_batches > 0 and batch_idx >= config.offline.max_val_batches:
                break

            prompts = decode_prompts(batch["prompt_ids"], tokenizers[0])
            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts,
                text_encoders,
                tokenizers,
                max_sequence_length=128,
                device=accelerator.device,
            )

            z = batch["noises"].to(accelerator.device, dtype=torch.float32)
            rewards = batch["rewards"].to(accelerator.device, dtype=torch.float32)
            advantages = batch["advantages"].to(accelerator.device, dtype=torch.float32)
            weights, scores = build_supervised_weights(rewards, advantages, config)
            epsilon = torch.randn_like(z)

            null_prompt_embeds = neg_prompt_embed.expand(z.shape[0], -1, -1)
            null_pooled_prompt_embeds = neg_pooled_prompt_embed.expand(z.shape[0], -1)
            loss, stats = compute_dit_supervised_loss(
                prior_dit,
                epsilon,
                z,
                prompt_embeds,
                pooled_prompt_embeds,
                weights,
                cfg_drop_rate=0.0,
                v_reg_weight=config.prior_dit.v_reg_weight,
                null_prompt_embeds=null_prompt_embeds,
                null_pooled_prompt_embeds=null_pooled_prompt_embeds,
            )

            metric_sums["val/num_batches"] += 1.0
            metric_sums["val/num_samples"] += float(z.shape[0])
            metric_sums["val/loss_sum"] += float(loss.item())
            metric_sums["val/reward_sum"] += float(rewards.sum().item())
            metric_sums["val/advantage_sum"] += float(advantages.sum().item())
            metric_sums["val/score_sum"] += float(scores.sum().item())
            metric_sums["val/weight_sum"] += float(weights.sum().item())
            for key, value in stats.items():
                metric_sums[f"val/{key}_sum"] += float(value)

    prior_dit.train()
    reduced = reduce_epoch_metrics(accelerator, metric_sums)
    if reduced is None or reduced["val/num_batches"] == 0:
        return None

    num_batches = reduced["val/num_batches"]
    num_samples = reduced["val/num_samples"]
    return {
        "val/loss": reduced["val/loss_sum"] / num_batches,
        "val/reward_mean": reduced["val/reward_sum"] / num_samples,
        "val/advantage_mean": reduced["val/advantage_sum"] / num_samples,
        "val/score_mean": reduced["val/score_sum"] / num_samples,
        "val/weight_mean": reduced["val/weight_sum"] / num_samples,
        "val/supervised_loss": reduced["val/supervised_loss_sum"] / num_batches,
        "val/v_reg": reduced["val/v_reg_sum"] / num_batches,
        "val/mse_mean": reduced["val/mse_mean_sum"] / num_batches,
        "val/v_pred_norm": reduced["val/v_pred_norm_sum"] / num_batches,
        "val/v_target_norm": reduced["val/v_target_norm_sum"] / num_batches,
    }


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
        wandb.init(project="flow_grpo", name=f"prior_dit_offline_{config.run_name}")

    logger.info(f"\n{config}")
    set_seed(config.seed, device_specific=True)

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

    latent_channels = pipeline.transformer.config.in_channels
    latent_h = config.resolution // pipeline.vae_scale_factor

    dit_config = config.prior_dit
    prior_dit = PriorDiT(
        sample_size=latent_h,
        patch_size=dit_config.patch_size,
        in_channels=latent_channels,
        out_channels=latent_channels,
        num_layers=dit_config.num_layers,
        num_attention_heads=dit_config.num_attention_heads,
        attention_head_dim=dit_config.attention_head_dim,
        joint_attention_dim=4096,
        pooled_projection_dim=2048,
        small_init_output=dit_config.small_init_output,
        output_init_std=dit_config.output_init_std,
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
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        eps=config.train.adam_epsilon,
    )

    reward_fn = flow_grpo.rewards.multi_score(accelerator.device, config.reward_fn)
    eval_reward_fn = flow_grpo.rewards.multi_score(accelerator.device, config.reward_fn)
    executor = futures.ThreadPoolExecutor(max_workers=8)

    train_dataset = TextPromptDataset(config.dataset, "train")
    test_dataset = TextPromptDataset(config.dataset, "test")
    train_eval_size = min(2048, len(train_dataset))
    train_eval_subset = Subset(train_dataset, range(train_eval_size))

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=TextPromptDataset.collate_fn,
        shuffle=False,
        num_workers=8,
    )
    train_eval_dataloader = DataLoader(
        train_eval_subset,
        batch_size=config.sample.test_batch_size,
        collate_fn=TextPromptDataset.collate_fn,
        shuffle=False,
        num_workers=8,
    )

    cache_files = list_cache_files(config.offline.cache_dir, config.offline.max_cache_files)
    cache_infos = summarize_cache_files(cache_files)
    train_infos, val_infos, train_dropped, val_dropped = split_cache_files(
        cache_infos,
        config.offline.num_val_files,
        accelerator.num_processes,
    )
    if not train_infos:
        raise ValueError("No train cache files available after sharding/truncation.")

    train_cache_dataset = OfflineCacheIterableDataset(
        train_infos,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        shuffle=True,
        seed=config.seed,
    )
    train_cache_dataloader = DataLoader(
        train_cache_dataset,
        batch_size=config.offline.train_batch_size,
        collate_fn=offline_cache_collate_fn,
        num_workers=config.offline.num_workers,
        drop_last=True,
    )

    val_cache_dataloader = None
    if val_infos:
        val_cache_dataset = OfflineCacheIterableDataset(
            val_infos,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            shuffle=False,
            seed=config.seed,
        )
        val_cache_dataloader = DataLoader(
            val_cache_dataset,
            batch_size=config.offline.val_batch_size,
            collate_fn=offline_cache_collate_fn,
            num_workers=config.offline.num_workers,
            drop_last=False,
        )

    prior_dit, optimizer = accelerator.prepare(prior_dit, optimizer)
    test_dataloader, train_eval_dataloader = accelerator.prepare(test_dataloader, train_eval_dataloader)

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""],
        text_encoders,
        tokenizers,
        max_sequence_length=128,
        device=accelerator.device,
    )

    autocast = accelerator.autocast
    train_steps_per_rank = len(train_cache_dataloader)

    logger.info("***** Running offline prior DiT training *****")
    logger.info(f"  Prior DiT params = {num_params:,}")
    logger.info(f"  ODE steps (small DiT) = {dit_config.num_steps}")
    logger.info(f"  Cache dir = {config.offline.cache_dir}")
    logger.info(f"  Cache files (total/train/val) = {len(cache_infos)}/{len(train_infos)}/{len(val_infos)}")
    logger.info(f"  Dropped files to align with world size (train/val) = {train_dropped}/{val_dropped}")
    logger.info(f"  Train samples per rank per epoch = {len(train_cache_dataset):,}")
    logger.info(f"  Train steps per rank per epoch = {train_steps_per_rank:,}")
    logger.info(f"  Offline weight source/transform = {config.offline.score_source}/{config.offline.weight_transform}")

    global_step = 0

    for epoch in range(config.num_epochs):
        train_cache_dataset.set_epoch(epoch)

        if epoch > 0 and epoch % config.eval_freq == 0:
            unwrapped_dit = accelerator.unwrap_model(prior_dit)
            run_eval(
                pipeline,
                unwrapped_dit,
                test_dataloader,
                text_encoders,
                tokenizers,
                config,
                accelerator,
                epoch,
                eval_reward_fn,
                executor,
                autocast,
                neg_prompt_embed,
                neg_pooled_prompt_embed,
                prefix="eval",
            )
            run_eval(
                pipeline,
                unwrapped_dit,
                train_eval_dataloader,
                text_encoders,
                tokenizers,
                config,
                accelerator,
                epoch,
                eval_reward_fn,
                executor,
                autocast,
                neg_prompt_embed,
                neg_pooled_prompt_embed,
                prefix="train_eval",
            )

        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_path = os.path.join(config.save_dir, "checkpoints", f"dit_epoch_{epoch:06d}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(accelerator.unwrap_model(prior_dit).state_dict(), save_path)

        metric_sums = defaultdict(float)
        prior_dit.train()

        progress_bar = tqdm(
            train_cache_dataloader,
            desc=f"Epoch {epoch}: offline_train",
            disable=not accelerator.is_local_main_process,
            position=0,
        )
        running = {
            "loss": 0.0,
            "mse": 0.0,
            "reward": 0.0,
            "weight": 0.0,
            "grad_norm": 0.0,
            "count": 0,
        }

        for batch_idx, batch in enumerate(progress_bar, start=1):
            prompts = decode_prompts(batch["prompt_ids"], tokenizers[0])
            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts,
                text_encoders,
                tokenizers,
                max_sequence_length=128,
                device=accelerator.device,
            )

            z = batch["noises"].to(accelerator.device, dtype=torch.float32)
            rewards = batch["rewards"].to(accelerator.device, dtype=torch.float32)
            advantages = batch["advantages"].to(accelerator.device, dtype=torch.float32)
            weights, scores = build_supervised_weights(rewards, advantages, config)
            epsilon = torch.randn_like(z)

            null_prompt_embeds = neg_prompt_embed.expand(z.shape[0], -1, -1)
            null_pooled_prompt_embeds = neg_pooled_prompt_embed.expand(z.shape[0], -1)

            optimizer.zero_grad()
            loss, stats = compute_dit_supervised_loss(
                prior_dit,
                epsilon,
                z,
                prompt_embeds,
                pooled_prompt_embeds,
                weights,
                cfg_drop_rate=config.prior_dit.cfg_drop_rate,
                v_reg_weight=config.prior_dit.v_reg_weight,
                null_prompt_embeds=null_prompt_embeds,
                null_pooled_prompt_embeds=null_pooled_prompt_embeds,
            )
            accelerator.backward(loss)
            grad_norm = compute_grad_norm(prior_dit.parameters())
            optimizer.step()
            global_step += 1

            metric_sums["train/num_batches"] += 1.0
            metric_sums["train/num_samples"] += float(z.shape[0])
            metric_sums["train/loss_sum"] += float(loss.item())
            metric_sums["train/reward_sum"] += float(rewards.sum().item())
            metric_sums["train/advantage_sum"] += float(advantages.sum().item())
            metric_sums["train/score_sum"] += float(scores.sum().item())
            metric_sums["train/weight_sum"] += float(weights.sum().item())
            metric_sums["train/grad_norm_sum"] += float(grad_norm.item())
            for key, value in stats.items():
                metric_sums[f"train/{key}_sum"] += float(value)

            running["count"] += 1
            running["loss"] += float(loss.item())
            running["mse"] += float(stats["mse_mean"])
            running["reward"] += float(rewards.mean().item())
            running["weight"] += float(weights.mean().item())
            running["grad_norm"] += float(grad_norm.item())

            if accelerator.is_local_main_process:
                denom = max(running["count"], 1)
                progress_bar.set_postfix(
                    loss=f"{running['loss'] / denom:.4f}",
                    mse=f"{running['mse'] / denom:.4f}",
                    reward=f"{running['reward'] / denom:.4f}",
                    weight=f"{running['weight'] / denom:.3f}",
                    grad=f"{running['grad_norm'] / denom:.2f}",
                )
                progress_bar.write(
                    "Epoch %d batch %d/%d: loss=%.4f mse=%.4f reward=%.4f weight=%.4f grad=%.2f"
                    % (
                        epoch,
                        batch_idx,
                        train_steps_per_rank,
                        float(loss.item()),
                        float(stats["mse_mean"]),
                        float(rewards.mean().item()),
                        float(weights.mean().item()),
                        float(grad_norm.item()),
                    )
                )

        reduced = reduce_epoch_metrics(accelerator, metric_sums)
        if accelerator.is_main_process and reduced is not None:
            num_batches = reduced["train/num_batches"]
            num_samples = reduced["train/num_samples"]
            train_log = {
                "epoch": epoch,
                "global_step": global_step,
                "train/loss": reduced["train/loss_sum"] / num_batches,
                "train/reward_mean": reduced["train/reward_sum"] / num_samples,
                "train/advantage_mean": reduced["train/advantage_sum"] / num_samples,
                "train/score_mean": reduced["train/score_sum"] / num_samples,
                "train/weight_mean": reduced["train/weight_sum"] / num_samples,
                "train/grad_norm": reduced["train/grad_norm_sum"] / num_batches,
                "train/supervised_loss": reduced["train/supervised_loss_sum"] / num_batches,
                "train/v_reg": reduced["train/v_reg_sum"] / num_batches,
                "train/mse_mean": reduced["train/mse_mean_sum"] / num_batches,
                "train/v_pred_norm": reduced["train/v_pred_norm_sum"] / num_batches,
                "train/v_target_norm": reduced["train/v_target_norm_sum"] / num_batches,
                "train/weight_max": reduced["train/weight_max_sum"] / num_batches,
                "train/weight_min": reduced["train/weight_min_sum"] / num_batches,
            }

            val_log = run_cache_validation(
                prior_dit,
                val_cache_dataloader,
                text_encoders,
                tokenizers,
                config,
                accelerator,
                neg_prompt_embed,
                neg_pooled_prompt_embed,
            )
            if val_log is not None:
                train_log.update(val_log)

            wandb.log(train_log, step=epoch)
            logger.info(
                "Epoch %d summary: loss=%.4f, mse=%.4f, reward=%.4f, weight=%.4f, grad=%.2f, val_loss=%s",
                epoch,
                train_log["train/loss"],
                train_log["train/mse_mean"],
                train_log["train/reward_mean"],
                train_log["train/weight_mean"],
                train_log["train/grad_norm"],
                "n/a" if val_log is None else f"{train_log['val/loss']:.4f}",
            )
        elif not accelerator.is_main_process:
            run_cache_validation(
                prior_dit,
                val_cache_dataloader,
                text_encoders,
                tokenizers,
                config,
                accelerator,
                neg_prompt_embed,
                neg_pooled_prompt_embed,
            )

    if accelerator.is_main_process:
        save_path = os.path.join(config.save_dir, "dit_final.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(accelerator.unwrap_model(prior_dit).state_dict(), save_path)
        logger.info("Offline training complete. Final DiT saved.")


if __name__ == "__main__":
    app.run(main)
