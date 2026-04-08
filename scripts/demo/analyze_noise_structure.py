"""分析 noise space 的结构：

1. 高 reward noise 是否聚集？分布形状如何？
2. 相近的 noise 是否产生相似的图像？（ODE 连续性）

Usage:
    python scripts/demo/analyze_noise_structure.py --device cuda:0
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from flow_grpo.pickscore_scorer import PickScoreScorer


def gen_and_score(pipeline, scorer, noises, prompts, device):
    BS = 8
    all_scores = []
    all_images = []
    for i in range(0, len(noises), BS):
        bs = min(BS, len(noises) - i)
        n = noises[i:i+bs].to(device, dtype=torch.float16)
        p = prompts[i:i+bs]
        with torch.autocast("cuda"):
            with torch.no_grad():
                imgs = pipeline(prompt=p, latents=n, num_inference_steps=40,
                                guidance_scale=4.5, output_type="pt",
                                height=512, width=512).images
        pils = [Image.fromarray((img.cpu().permute(1,2,0).numpy()*255).astype(np.uint8)) for img in imgs]
        scores = scorer(p, pils).cpu().numpy()
        all_scores.extend(scores.tolist())
        all_images.extend([img.cpu() for img in imgs])
    return np.array(all_scores), all_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_noises", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="outputs/noise_structure")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("加载模型...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16).to(args.device)
    pipeline.safety_checker = None
    scorer = PickScoreScorer(device=args.device, dtype=torch.float32)

    C, H, W = 16, 64, 64
    prompt = "a beautiful photograph of a mountain landscape at sunset"

    # =====================================================================
    # 实验 1：高 reward noise 是否聚集？
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"实验 1：高 reward noise 的聚集性 ({args.num_noises} noises)")
    print(f"{'='*60}\n")

    torch.manual_seed(42)
    noises = torch.randn(args.num_noises, C, H, W)
    scores, _ = gen_and_score(pipeline, scorer, noises, [prompt]*args.num_noises, args.device)

    # 取 top-10% 和 bottom-10%
    K = args.num_noises // 10
    top_idx = np.argsort(scores)[-K:]
    bottom_idx = np.argsort(scores)[:K]
    random_idx = np.random.choice(args.num_noises, K, replace=False)

    top_noises = noises[top_idx].flatten(1)       # (K, 65536)
    bottom_noises = noises[bottom_idx].flatten(1)
    random_noises = noises[random_idx].flatten(1)

    # 计算组内两两距离
    def pairwise_dist(x):
        # x: (K, D)
        dists = []
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                dists.append((x[i] - x[j]).norm().item())
        return np.array(dists)

    dist_top = pairwise_dist(top_noises)
    dist_bottom = pairwise_dist(bottom_noises)
    dist_random = pairwise_dist(random_noises)

    print(f"Top-{K} reward noises 组内距离:    mean={dist_top.mean():.1f}, std={dist_top.std():.1f}")
    print(f"Bottom-{K} reward noises 组内距离: mean={dist_bottom.mean():.1f}, std={dist_bottom.std():.1f}")
    print(f"Random-{K} noises 组内距离:        mean={dist_random.mean():.1f}, std={dist_random.std():.1f}")
    print()

    # N(0,I) 的两个随机点的期望距离 = sqrt(2*d) ≈ 362
    expected_dist = (2 * C * H * W) ** 0.5
    print(f"理论 N(0,I) 两点期望距离: {expected_dist:.1f}")
    print()

    if dist_top.mean() < dist_random.mean() * 0.95:
        print(f"结论：Top noises 更聚集（距离比随机小 {(1-dist_top.mean()/dist_random.mean())*100:.1f}%）")
    else:
        print(f"结论：Top noises 不聚集（距离与随机相当）")

    # 分析 top noises 的分布
    top_noises_full = noises[top_idx]  # (K, C, H, W)
    top_mean = top_noises_full.mean(dim=0)  # (C, H, W)
    top_std = top_noises_full.std(dim=0)
    print(f"\nTop noises 分布:")
    print(f"  均值的 norm: {top_mean.norm():.2f} (N(0,I) 期望: ~{(C*H*W)**0.5 / args.num_noises**0.5:.1f})")
    print(f"  标准差 mean: {top_std.mean():.4f} (N(0,I) 期望: ~1.0)")
    print(f"  标准差 std:  {top_std.std():.4f}")

    # =====================================================================
    # 实验 2：ODE 连续性 — 相近 noise 产生相似图像？
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"实验 2：ODE 连续性（noise 微扰 → 图像变化）")
    print(f"{'='*60}\n")

    base_noise = torch.randn(1, C, H, W)
    perturbation_scales = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    # 生成 base image
    base_score, base_imgs = gen_and_score(pipeline, scorer, base_noise, [prompt], args.device)
    base_img = base_imgs[0]  # (C_img, H_img, W_img)

    print(f"Base noise reward: {base_score[0]:.4f}")
    print()
    print(f"{'Scale':>8s} | {'Reward':>8s} | {'Pixel Diff':>12s} | {'Reward Diff':>12s}")
    print("-" * 50)

    results = []
    for scale in perturbation_scales:
        if scale == 0:
            perturbed = base_noise.clone()
        else:
            direction = torch.randn_like(base_noise)
            direction = direction / direction.norm() * base_noise.norm() * scale
            perturbed = base_noise + direction

        p_score, p_imgs = gen_and_score(pipeline, scorer, perturbed, [prompt], args.device)
        pixel_diff = (base_img - p_imgs[0]).abs().mean().item()
        reward_diff = p_score[0] - base_score[0]

        results.append((scale, p_score[0], pixel_diff, reward_diff))
        print(f"{scale:>8.2f} | {p_score[0]:>8.4f} | {pixel_diff:>12.6f} | {reward_diff:>+12.4f}")

        # 保存图片
        pil = Image.fromarray((p_imgs[0].permute(1,2,0).numpy()*255).astype(np.uint8))
        pil.save(os.path.join(args.output_dir, f"perturb_scale{scale:.2f}_score{p_score[0]:.4f}.png"))

    print()
    # 分析连续性
    scales = [r[0] for r in results if r[0] > 0]
    pixel_diffs = [r[2] for r in results if r[0] > 0]
    if pixel_diffs[0] < 0.01:
        print("结论：小扰动 (0.01) 几乎不改变图像 → ODE 在局部是连续的")
    else:
        print("结论：即使小扰动也改变图像 → ODE 对 noise 敏感")

    # 保存结果
    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(f"Prompt: {prompt}\n\n")
        f.write("实验1: Top noise clustering\n")
        f.write(f"  Top-{K} dist: {dist_top.mean():.1f}\n")
        f.write(f"  Random dist: {dist_random.mean():.1f}\n\n")
        f.write("实验2: ODE continuity\n")
        for scale, score, pdiff, rdiff in results:
            f.write(f"  scale={scale:.2f}: reward={score:.4f}, pixel_diff={pdiff:.6f}\n")


if __name__ == "__main__":
    main()
