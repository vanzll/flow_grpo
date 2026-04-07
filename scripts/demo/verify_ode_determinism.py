"""验证 ODE 确定性：同一个 (noise, prompt) 对，两次前向传播的输出和 reward 是否一致。

Usage:
    python scripts/demo/verify_ode_determinism.py --num_samples 20
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from flow_grpo.pickscore_scorer import PickScoreScorer


def generate_batch(pipeline, noises, prompts, steps, guidance_scale, resolution, device):
    """给定 noise 和 prompt，生成图像并返回 tensor。"""
    latents = noises.to(device, dtype=torch.float16)
    with torch.autocast("cuda"):
        with torch.no_grad():
            images = pipeline(
                prompt=prompts,
                latents=latents,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                output_type="pt",
                height=resolution,
                width=resolution,
            ).images
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="outputs/ode_determinism")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print(f"加载模型 {args.model} ...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipeline.to(args.device)
    pipeline.safety_checker = None

    print("加载 PickScore ...")
    scorer = PickScoreScorer(device=args.device, dtype=torch.float32)

    # 固定 prompt 列表
    prompts = [
        "a beautiful sunset over the ocean",
        "a cat sitting on a windowsill",
        "a futuristic city skyline at night",
        "a plate of delicious sushi",
        "a portrait of an old man with a beard",
    ]
    # 重复到 num_samples
    prompts = [prompts[i % len(prompts)] for i in range(args.num_samples)]

    # Latent shape
    latent_c = pipeline.transformer.config.in_channels
    latent_h = args.resolution // pipeline.vae_scale_factor
    latent_w = args.resolution // pipeline.vae_scale_factor

    # 生成固定 noise（用固定 seed）
    torch.manual_seed(12345)
    noises = torch.randn(args.num_samples, latent_c, latent_h, latent_w)

    # === Run 1 ===
    print("\n=== Run 1 ===")
    batch_size = 4
    images_run1 = []
    for i in range(0, args.num_samples, batch_size):
        batch_noises = noises[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        imgs = generate_batch(pipeline, batch_noises, batch_prompts,
                              args.steps, args.guidance_scale, args.resolution, args.device)
        images_run1.append(imgs.cpu())
    images_run1 = torch.cat(images_run1, dim=0)

    # PickScore run 1
    pil_images1 = [Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                   for img in images_run1]
    scores_run1 = scorer(prompts, pil_images1).cpu().numpy()

    # === Run 2（完全相同的 noise 和 prompt）===
    print("\n=== Run 2 ===")
    images_run2 = []
    for i in range(0, args.num_samples, batch_size):
        batch_noises = noises[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        imgs = generate_batch(pipeline, batch_noises, batch_prompts,
                              args.steps, args.guidance_scale, args.resolution, args.device)
        images_run2.append(imgs.cpu())
    images_run2 = torch.cat(images_run2, dim=0)

    # PickScore run 2
    pil_images2 = [Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                   for img in images_run2]
    scores_run2 = scorer(prompts, pil_images2).cpu().numpy()

    # === 对比 ===
    print(f"\n{'='*60}")
    print("ODE 确定性验证")
    print(f"{'='*60}")

    # 像素级对比
    pixel_diff = (images_run1 - images_run2).abs()
    print(f"像素差异: max={pixel_diff.max():.6f}, mean={pixel_diff.mean():.6f}")
    print(f"像素完全一致: {(pixel_diff == 0).all()}")
    print()

    # Reward 对比
    print(f"{'idx':>4s} | {'prompt':>40s} | {'run1':>8s} | {'run2':>8s} | {'diff':>10s}")
    print("-" * 80)
    for i in range(args.num_samples):
        diff = scores_run2[i] - scores_run1[i]
        print(f"{i:>4d} | {prompts[i][:40]:>40s} | {scores_run1[i]:>8.4f} | {scores_run2[i]:>8.4f} | {diff:>+10.6f}")

    print()
    print(f"PickScore Run 1: {scores_run1.mean():.4f} ± {scores_run1.std():.4f}")
    print(f"PickScore Run 2: {scores_run2.mean():.4f} ± {scores_run2.std():.4f}")
    correlation = np.corrcoef(scores_run1, scores_run2)[0, 1]
    print(f"相关系数: {correlation:.6f}")
    print(f"平均差值: {(scores_run2 - scores_run1).mean():.6f}")

    # === Run 3: fp16 roundtrip 验证 ===
    print(f"\n{'='*60}")
    print("fp16 存储 roundtrip 验证")
    print(f"{'='*60}")
    noises_fp16 = noises.to(torch.float16).to(torch.float32)  # fp16 roundtrip
    noise_diff = (noises - noises_fp16).abs()
    print(f"fp16 roundtrip noise 差异: max={noise_diff.max():.6f}, mean={noise_diff.mean():.6f}")

    # 用 fp16 roundtrip 后的 noise 再生成一次
    print("\n=== Run 3 (fp16 roundtrip noise) ===")
    images_run3 = []
    for i in range(0, args.num_samples, batch_size):
        batch_noises = noises_fp16[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        imgs = generate_batch(pipeline, batch_noises, batch_prompts,
                              args.steps, args.guidance_scale, args.resolution, args.device)
        images_run3.append(imgs.cpu())
    images_run3 = torch.cat(images_run3, dim=0)

    pil_images3 = [Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                   for img in images_run3]
    scores_run3 = scorer(prompts, pil_images3).cpu().numpy()

    pixel_diff_fp16 = (images_run1 - images_run3).abs()
    print(f"Run1 vs Run3 像素差异: max={pixel_diff_fp16.max():.6f}, mean={pixel_diff_fp16.mean():.6f}")
    print(f"Run1 vs Run3 PickScore 相关系数: {np.corrcoef(scores_run1, scores_run3)[0, 1]:.6f}")
    print(f"Run3 PickScore: {scores_run3.mean():.4f} ± {scores_run3.std():.4f}")

    # 保存对比图
    for i in range(min(5, args.num_samples)):
        pil_images1[i].save(os.path.join(args.output_dir, f"{i:03d}_run1_score{scores_run1[i]:.4f}.png"))
        pil_images2[i].save(os.path.join(args.output_dir, f"{i:03d}_run2_score{scores_run2[i]:.4f}.png"))
        pil_images3[i].save(os.path.join(args.output_dir, f"{i:03d}_run3_fp16_score{scores_run3[i]:.4f}.png"))

    print(f"\n图片已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
