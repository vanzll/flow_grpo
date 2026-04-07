"""诊断：noise space 里有多少 reward 信号？

固定 prompt，采 N 个 N(0,I) noise，看 PickScore 的方差。
如果方差很小，说明 noise 对 reward 影响极弱，prior shaping 无效。

Usage:
    python scripts/demo/diagnose_noise_signal.py --num_samples 200 --device cuda:1
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
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    prompts_to_test = [
        "a beautiful sunset over the ocean",
        "a cat sitting on a windowsill looking outside",
        "a futuristic city skyline at night with neon lights",
        "a plate of sushi on a wooden table",
        "a portrait of an elderly man with a white beard",
    ]

    print(f"加载模型 {args.model} ...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipeline.to(args.device)
    pipeline.safety_checker = None

    print("加载 PickScore ...")
    scorer = PickScoreScorer(device=args.device, dtype=torch.float32)

    latent_c = pipeline.transformer.config.in_channels
    latent_h = args.resolution // pipeline.vae_scale_factor
    latent_w = args.resolution // pipeline.vae_scale_factor

    print(f"\n采样 {args.num_samples} 个 noise × {len(prompts_to_test)} 个 prompt")
    print(f"Latent shape: ({latent_c}, {latent_h}, {latent_w})\n")

    all_results = []

    for prompt in prompts_to_test:
        print(f"Prompt: \"{prompt}\"")
        rewards = []

        for i in range(0, args.num_samples, args.batch_size):
            bs = min(args.batch_size, args.num_samples - i)
            noises = torch.randn(bs, latent_c, latent_h, latent_w,
                                 device=args.device, dtype=torch.float16)

            with torch.autocast("cuda"):
                with torch.no_grad():
                    images = pipeline(
                        prompt=[prompt] * bs,
                        latents=noises,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        output_type="pt",
                        height=args.resolution,
                        width=args.resolution,
                    ).images

            pil_images = [
                Image.fromarray((img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                for img in images
            ]
            scores = scorer([prompt] * bs, pil_images).cpu().numpy()
            rewards.extend(scores.tolist())

            if (i + bs) % 50 == 0 or i + bs == args.num_samples:
                print(f"  {i+bs}/{args.num_samples} done")

        rewards = np.array(rewards)
        result = {
            "prompt": prompt,
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "min": float(rewards.min()),
            "max": float(rewards.max()),
            "range": float(rewards.max() - rewards.min()),
            "cv": float(rewards.std() / rewards.mean()),  # coefficient of variation
        }
        all_results.append(result)
        print(f"  PickScore: {result['mean']:.4f} ± {result['std']:.4f} "
              f"(range: {result['min']:.4f} ~ {result['max']:.4f})\n")

    # 汇总
    print("=" * 70)
    print("诊断结果汇总")
    print("=" * 70)
    print(f"{'Prompt':<55s} | {'Mean':>6s} | {'Std':>6s} | {'Range':>6s} | {'CV%':>5s}")
    print("-" * 85)
    for r in all_results:
        print(f"{r['prompt'][:55]:<55s} | {r['mean']:>6.4f} | {r['std']:>6.4f} | {r['range']:>6.4f} | {r['cv']*100:>5.1f}")

    avg_std = np.mean([r["std"] for r in all_results])
    avg_range = np.mean([r["range"] for r in all_results])
    avg_cv = np.mean([r["cv"] for r in all_results])

    print("-" * 85)
    print(f"{'平均':<55s} | {'':>6s} | {avg_std:>6.4f} | {avg_range:>6.4f} | {avg_cv*100:>5.1f}")
    print()

    if avg_std < 0.02:
        print("结论：noise 对 reward 影响极弱（std < 0.02）。Prior shaping 空间很小。")
    elif avg_std < 0.05:
        print("结论：noise 有一定信号（0.02 < std < 0.05）。Prior shaping 可能有效但提升有限。")
    else:
        print("结论：noise 信号较强（std > 0.05）。Prior shaping 有显著优化空间。")


if __name__ == "__main__":
    main()
