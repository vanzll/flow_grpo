"""验证 noise 选择是否能提升 reward。

实验 1：同 prompt，对比 top noise vs random noise vs bottom noise
实验 2：top noise 配 test set 的新 prompt，看能否跨 prompt 泛化

Usage:
    python scripts/demo/verify_noise_selection.py --device cuda:0
"""

import argparse
import os
import random
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from flow_grpo.pickscore_scorer import PickScoreScorer


def generate_and_score(pipeline, scorer, noises, prompts, device, steps=40, cfg=4.5, resolution=512):
    """给定 noise 和 prompt，生成图像并打分。"""
    batch_size = 4
    all_scores = []
    for i in range(0, len(noises), batch_size):
        bs = min(batch_size, len(noises) - i)
        batch_noises = noises[i:i+bs].to(device, dtype=torch.float16)
        batch_prompts = prompts[i:i+bs]
        with torch.autocast("cuda"):
            with torch.no_grad():
                images = pipeline(
                    prompt=batch_prompts,
                    latents=batch_noises,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    output_type="pt",
                    height=resolution,
                    width=resolution,
                ).images
        pil_images = [
            Image.fromarray((img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            for img in images
        ]
        scores = scorer(batch_prompts, pil_images).cpu().numpy()
        all_scores.extend(scores.tolist())
    return np.array(all_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=20, help="实验 1 用多少个 prompt")
    parser.add_argument("--samples_per_prompt", type=int, default=100, help="每个 prompt 采多少 noise")
    parser.add_argument("--num_test", type=int, default=50, help="实验 2 用多少个 test prompt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="outputs/verify_noise_selection")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print("加载模型...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16
    )
    pipeline.to(args.device)
    pipeline.safety_checker = None

    print("加载 PickScore...")
    scorer = PickScoreScorer(device=args.device, dtype=torch.float32)

    latent_c = pipeline.transformer.config.in_channels
    latent_h = 512 // pipeline.vae_scale_factor
    latent_w = latent_h

    # 选 prompt
    train_prompts_file = "dataset/pickscore/train.txt"
    test_prompts_file = "dataset/pickscore/test.txt"
    with open(train_prompts_file) as f:
        all_train_prompts = [l.strip() for l in f if l.strip()]
    with open(test_prompts_file) as f:
        all_test_prompts = [l.strip() for l in f if l.strip()]

    random.seed(42)
    selected_prompts = random.sample(all_train_prompts, args.num_prompts)

    # =====================================================================
    # 实验 1：同 prompt，采 N 个 noise，选 top/bottom/random 对比
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"实验 1：同 prompt + top/bottom/random noise ({args.num_prompts} prompts × {args.samples_per_prompt} noises)")
    print(f"{'='*70}\n")

    exp1_results = []
    for pi, prompt in enumerate(selected_prompts):
        print(f"[{pi+1}/{args.num_prompts}] \"{prompt[:60]}...\"")

        # 采 N 个 noise 并打分
        noises = torch.randn(args.samples_per_prompt, latent_c, latent_h, latent_w)
        prompts_repeated = [prompt] * args.samples_per_prompt
        scores = generate_and_score(pipeline, scorer, noises, prompts_repeated, args.device)

        top_idx = scores.argmax()
        bottom_idx = scores.argmin()

        top_score = scores[top_idx]
        bottom_score = scores[bottom_idx]
        mean_score = scores.mean()
        std_score = scores.std()

        # 验证 top noise 的确定性（重新跑一次）
        top_noise = noises[top_idx:top_idx+1]
        verify_score = generate_and_score(pipeline, scorer, top_noise, [prompt], args.device)[0]

        exp1_results.append({
            "prompt": prompt[:60],
            "top": top_score,
            "bottom": bottom_score,
            "mean": mean_score,
            "std": std_score,
            "verify": verify_score,
            "top_noise": noises[top_idx],
            "top_noise_idx": top_idx,
        })
        print(f"  top={top_score:.4f}, mean={mean_score:.4f}, bottom={bottom_score:.4f}, "
              f"verify={verify_score:.4f}, range={top_score-bottom_score:.4f}")

    # 实验 1 汇总
    print(f"\n{'='*70}")
    print("实验 1 汇总")
    print(f"{'='*70}")
    print(f"{'Prompt':<62s} | {'Top':>6s} | {'Mean':>6s} | {'Bottom':>6s} | {'Range':>6s}")
    print("-" * 95)
    for r in exp1_results:
        print(f"{r['prompt']:<62s} | {r['top']:>6.4f} | {r['mean']:>6.4f} | {r['bottom']:>6.4f} | {r['top']-r['bottom']:>6.4f}")

    avg_top = np.mean([r['top'] for r in exp1_results])
    avg_mean = np.mean([r['mean'] for r in exp1_results])
    avg_bottom = np.mean([r['bottom'] for r in exp1_results])
    print("-" * 95)
    print(f"{'平均':<62s} | {avg_top:>6.4f} | {avg_mean:>6.4f} | {avg_bottom:>6.4f} | {avg_top-avg_bottom:>6.4f}")
    print()
    print(f"Top vs Mean 提升: +{avg_top - avg_mean:.4f} (×26 = +{(avg_top-avg_mean)*26:.2f} PickScore)")
    print(f"Top vs Bottom 差距: {avg_top - avg_bottom:.4f} (×26 = {(avg_top-avg_bottom)*26:.2f} PickScore)")

    # 验证确定性
    verify_diffs = [abs(r['top'] - r['verify']) for r in exp1_results]
    print(f"确定性验证: 平均 |top - verify| = {np.mean(verify_diffs):.6f}")

    # =====================================================================
    # 实验 2：top noise 配 test set 的新 prompt
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"实验 2：top noise 配 test prompt（{args.num_test} 组）")
    print(f"{'='*70}\n")

    # 从实验 1 收集所有 (noise, score) 对，取全局 top
    all_noise_scores = []
    for r in exp1_results:
        all_noise_scores.append((r['top_noise'], r['top']))
    # 按 score 排序取 top
    all_noise_scores.sort(key=lambda x: x[1], reverse=True)

    # 选 test prompts
    test_prompts = random.sample(all_test_prompts, args.num_test)

    # 实验 2a: top noise + test prompt
    # 循环使用 top noise（可能不够 num_test 个）
    top_noises = torch.stack([all_noise_scores[i % len(all_noise_scores)][0] for i in range(args.num_test)])
    print("生成: top noise + test prompt...")
    scores_top = generate_and_score(pipeline, scorer, top_noises, test_prompts, args.device)

    # 实验 2b: random noise + test prompt (baseline)
    random_noises = torch.randn(args.num_test, latent_c, latent_h, latent_w)
    print("生成: random noise + test prompt...")
    scores_random = generate_and_score(pipeline, scorer, random_noises, test_prompts, args.device)

    # 汇总
    print(f"\n{'='*70}")
    print("实验 2 汇总")
    print(f"{'='*70}")
    print(f"Top noise + test prompt:    {scores_top.mean():.4f} ± {scores_top.std():.4f}")
    print(f"Random noise + test prompt: {scores_random.mean():.4f} ± {scores_random.std():.4f}")
    print(f"差值: {scores_top.mean() - scores_random.mean():+.4f} (×26 = {(scores_top.mean()-scores_random.mean())*26:+.2f})")
    print()

    # =====================================================================
    # 最终结论
    # =====================================================================
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")

    exp1_lift = avg_top - avg_mean
    exp2_lift = scores_top.mean() - scores_random.mean()

    print(f"实验 1 (同 prompt): top noise 比 random 高 {exp1_lift:.4f} (×26 = {exp1_lift*26:.2f})")
    print(f"实验 2 (新 prompt): top noise 比 random 高 {exp2_lift:.4f} (×26 = {exp2_lift*26:.2f})")
    print()

    if exp1_lift > 0.01 and exp2_lift > 0.01:
        print("Noise 选择有价值，且能跨 prompt 泛化。Prior shaping 有希望。")
    elif exp1_lift > 0.01 and exp2_lift <= 0.01:
        print("Noise 选择有价值，但不能跨 prompt 泛化。需要 prompt-conditioned policy。")
    else:
        print("Noise 选择价值有限。Prior shaping 这条路很难 work。")

    # 保存结果
    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(f"实验 1: top={avg_top:.4f}, mean={avg_mean:.4f}, bottom={avg_bottom:.4f}, lift={exp1_lift:.4f}\n")
        f.write(f"实验 2: top_noise={scores_top.mean():.4f}, random={scores_random.mean():.4f}, lift={exp2_lift:.4f}\n")


if __name__ == "__main__":
    main()
