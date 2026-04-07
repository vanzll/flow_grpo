"""实验1: 用极度偏移的 prior 生成图像，观察 reward 崩溃。

对比不同偏移程度下的 PickScore：
- N(0, I)               baseline
- N(μ, I), |μ|=1        轻微偏移
- N(μ, I), |μ|=5        中等偏移
- N(μ, I), |μ|=10       严重偏移
- N(μ, I), |μ|=50       极端偏移

Usage:
    python scripts/demo/eval_shifted_prior.py --num_samples 16
    python scripts/demo/eval_shifted_prior.py --num_samples 32 --device cuda:0
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from flow_grpo.pickscore_scorer import PickScoreScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=16, help="每个偏移程度生成多少张图")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="outputs/shifted_prior_eval")
    parser.add_argument("--prompt", type=str, default="a beautiful photograph of a mountain landscape")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print(f"加载模型 {args.model} ...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipeline.to(args.device)
    pipeline.safety_checker = None

    print("加载 PickScore ...")
    scorer = PickScoreScorer(device=args.device, dtype=torch.float32)

    # Latent shape
    latent_c = pipeline.transformer.config.in_channels
    latent_h = args.resolution // pipeline.vae_scale_factor
    latent_w = args.resolution // pipeline.vae_scale_factor
    shape = (latent_c, latent_h, latent_w)
    dim = latent_c * latent_h * latent_w
    print(f"Latent shape: {shape}, dim: {dim}")

    # 不同偏移程度
    mu_norms = [0, 100, 500, 1000, 5000, 10000, 50000]

    results = []
    for mu_norm in mu_norms:
        print(f"\n{'='*60}")
        print(f"mu_norm = {mu_norm}")
        print(f"{'='*60}")

        # 创建偏移方向（固定随机方向，只改变大小）
        torch.manual_seed(42)
        direction = torch.randn(shape)
        direction = direction / direction.norm() * mu_norm  # 归一化到目标 norm

        per_dim_shift = direction.abs().mean().item()
        print(f"  Per-dim abs mean shift: {per_dim_shift:.4f}")

        # 采样
        noises = torch.randn(args.num_samples, *shape) + direction.unsqueeze(0)
        noises = noises.to(args.device, dtype=torch.float16)

        # 生成图像
        all_images = []
        batch_size = min(4, args.num_samples)
        for i in range(0, args.num_samples, batch_size):
            batch_noise = noises[i:i+batch_size]
            with torch.autocast("cuda"):
                with torch.no_grad():
                    images = pipeline(
                        prompt=[args.prompt] * len(batch_noise),
                        latents=batch_noise,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        output_type="pt",
                        height=args.resolution,
                        width=args.resolution,
                    ).images
            all_images.append(images.cpu())

        all_images = torch.cat(all_images, dim=0)  # (N, C, H, W)

        # PickScore
        pil_images = []
        for img in all_images:
            pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            pil_images.append(pil)

        prompts = [args.prompt] * args.num_samples
        scores = scorer(prompts, pil_images).cpu().numpy()

        mean_score = scores.mean()
        std_score = scores.std()
        results.append((mu_norm, per_dim_shift, mean_score, std_score))

        print(f"  PickScore: {mean_score:.4f} ± {std_score:.4f}")
        print(f"  PickScore (×26): {mean_score*26:.2f} ± {std_score*26:.2f}")

        # 保存图片
        subdir = os.path.join(args.output_dir, f"mu_norm_{mu_norm}")
        os.makedirs(subdir, exist_ok=True)
        for j, pil in enumerate(pil_images):
            pil.save(os.path.join(subdir, f"{j:03d}_score{scores[j]:.4f}.png"))

    # 汇总
    print(f"\n{'='*60}")
    print("汇总结果")
    print(f"{'='*60}")
    print(f"{'mu_norm':>10s} | {'per_dim':>10s} | {'PickScore':>10s} | {'×26':>10s}")
    print("-" * 50)
    for mu_norm, per_dim, mean_s, std_s in results:
        print(f"{mu_norm:>10d} | {per_dim:>10.4f} | {mean_s:>10.4f} | {mean_s*26:>10.2f}")

    # 保存结果
    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(f"prompt: {args.prompt}\n")
        f.write(f"num_samples: {args.num_samples}\n")
        f.write(f"steps: {args.steps}, guidance: {args.guidance_scale}\n\n")
        f.write(f"{'mu_norm':>10s} | {'per_dim':>10s} | {'PickScore':>10s} | {'std':>10s} | {'×26':>10s}\n")
        f.write("-" * 60 + "\n")
        for mu_norm, per_dim, mean_s, std_s in results:
            f.write(f"{mu_norm:>10d} | {per_dim:>10.4f} | {mean_s:>10.4f} | {std_s:>10.4f} | {mean_s*26:>10.2f}\n")

    print(f"\n图片和结果已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
