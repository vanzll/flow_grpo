"""实验2: 从缓存中收集高 reward 的 noise，前向传播生成图像并评估。

验证：已知高 reward 的 noise 重新生成图像后，PickScore 是否一致？

Usage:
    python scripts/demo/eval_top_noises.py --top_k 50
    python scripts/demo/eval_top_noises.py --top_k 100 --cache_dir cache/prior_shaping
"""

import argparse
import glob
import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from flow_grpo.pickscore_scorer import PickScoreScorer


def load_top_noises(cache_dir, top_k):
    """从缓存中找出 reward 最高的 top_k 个 noise。"""
    files = sorted(glob.glob(os.path.join(cache_dir, "epoch_*.npz")))
    all_noises = []
    all_rewards = []
    all_locations = []

    for f in files:
        epoch_num = int(os.path.basename(f).split("_")[1].split(".")[0])
        with np.load(f) as data:
            noises = data["noises"]  # (N, C, H, W) fp16
            rewards = data["rewards"]  # (N,)
            for i in range(len(rewards)):
                all_noises.append(noises[i])
                all_rewards.append(rewards[i])
                all_locations.append((epoch_num, i))

    all_rewards = np.array(all_rewards)
    top_idx = np.argsort(all_rewards)[-top_k:][::-1]

    results = []
    for idx in top_idx:
        results.append({
            "noise": torch.from_numpy(all_noises[idx].astype(np.float32)),
            "cached_reward": float(all_rewards[idx]),
            "epoch": all_locations[idx][0],
            "index": all_locations[idx][1],
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="cache/prior_shaping")
    parser.add_argument("--top_k", type=int, default=50, help="取 reward 最高的 K 个 noise")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="outputs/top_noises_eval")
    parser.add_argument("--prompt", type=str, default="",
                        help="prompt（留空则用空 prompt，跟训练时一致）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 收集 top-K noise
    print(f"从 {args.cache_dir} 收集 top-{args.top_k} noise ...")
    top_entries = load_top_noises(args.cache_dir, args.top_k)
    print(f"Cached reward 范围: {top_entries[-1]['cached_reward']:.4f} ~ {top_entries[0]['cached_reward']:.4f}")

    # 加载模型
    print(f"加载模型 {args.model} ...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipeline.to(args.device)
    pipeline.safety_checker = None

    print("加载 PickScore ...")
    scorer = PickScoreScorer(device=args.device, dtype=torch.float32)

    # 读取训练用的 prompt 列表（跟训练时一致）
    prompt_file = os.path.join(os.getcwd(), "dataset/pickscore/train.txt")
    if os.path.exists(prompt_file) and not args.prompt:
        with open(prompt_file, "r") as f:
            all_prompts = [line.strip() for line in f.readlines()]
        print(f"使用训练 prompt 列表 ({len(all_prompts)} prompts)")
        use_train_prompts = True
    else:
        use_train_prompts = False

    # 批量生成图像
    batch_size = 4
    all_cached_rewards = []
    all_new_rewards = []

    for i in range(0, len(top_entries), batch_size):
        batch = top_entries[i:i+batch_size]
        noises = torch.stack([e["noise"] for e in batch]).to(args.device, dtype=torch.float16)

        # 选 prompt: 用一个固定 prompt 或留空
        if args.prompt:
            prompts = [args.prompt] * len(batch)
        elif use_train_prompts:
            # 用不同的 prompt 来测试
            prompts = [all_prompts[j % len(all_prompts)] for j in range(i, i+len(batch))]
        else:
            prompts = [""] * len(batch)

        print(f"生成 {i+1}-{i+len(batch)} / {len(top_entries)} ...")
        with torch.autocast("cuda"):
            with torch.no_grad():
                images = pipeline(
                    prompt=prompts,
                    latents=noises,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    output_type="pt",
                    height=args.resolution,
                    width=args.resolution,
                ).images

        # 转 PIL
        pil_images = []
        for img in images.cpu():
            pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            pil_images.append(pil)

        # 打分
        scores = scorer(prompts, pil_images).cpu().numpy()

        for j, (entry, score, pil) in enumerate(zip(batch, scores, pil_images)):
            cached_r = entry["cached_reward"]
            new_r = float(score)
            all_cached_rewards.append(cached_r)
            all_new_rewards.append(new_r)

            # 保存图片
            fname = f"rank{i+j:03d}_cached{cached_r:.4f}_new{new_r:.4f}_e{entry['epoch']}_i{entry['index']}.png"
            pil.save(os.path.join(args.output_dir, fname))

    # 汇总
    all_cached_rewards = np.array(all_cached_rewards)
    all_new_rewards = np.array(all_new_rewards)

    print(f"\n{'='*60}")
    print(f"Top-{args.top_k} Noise 评估结果")
    print(f"{'='*60}")
    print(f"Cached PickScore:  {all_cached_rewards.mean():.4f} ± {all_cached_rewards.std():.4f}")
    print(f"New PickScore:     {all_new_rewards.mean():.4f} ± {all_new_rewards.std():.4f}")
    print(f"Cached (×26):      {all_cached_rewards.mean()*26:.2f}")
    print(f"New (×26):         {all_new_rewards.mean()*26:.2f}")
    print()

    # 对比：N(0,I) baseline
    print("对比 baseline (N(0,I)):")
    print(f"  Baseline PickScore ≈ 0.835 (×26 ≈ 21.72)")
    print(f"  Top-{args.top_k} cached:  {all_cached_rewards.mean():.4f} (×26 = {all_cached_rewards.mean()*26:.2f})")
    print(f"  Top-{args.top_k} re-eval: {all_new_rewards.mean():.4f} (×26 = {all_new_rewards.mean()*26:.2f})")
    print()

    # 一致性检查
    correlation = np.corrcoef(all_cached_rewards, all_new_rewards)[0, 1]
    diff = all_new_rewards - all_cached_rewards
    print(f"Cached vs New 相关系数: {correlation:.4f}")
    print(f"差值: {diff.mean():.4f} ± {diff.std():.4f}")
    print()

    if abs(correlation) < 0.5:
        print("⚠ 相关系数低！可能原因：")
        print("  - 训练时用的 prompt 跟现在不同（PickScore 依赖 prompt-image 匹配）")
        print("  - fp16 存储导致 noise 精度损失")

    # 保存结果
    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(f"Top-{args.top_k} noise evaluation\n")
        f.write(f"Cache dir: {args.cache_dir}\n")
        f.write(f"Steps: {args.steps}, Guidance: {args.guidance_scale}\n\n")
        f.write(f"Cached PickScore: {all_cached_rewards.mean():.4f} ± {all_cached_rewards.std():.4f}\n")
        f.write(f"New PickScore:    {all_new_rewards.mean():.4f} ± {all_new_rewards.std():.4f}\n")
        f.write(f"Correlation:      {correlation:.4f}\n\n")
        f.write(f"{'rank':>5s} | {'cached':>8s} | {'new':>8s} | {'diff':>8s} | {'epoch':>6s} | {'index':>6s}\n")
        f.write("-" * 55 + "\n")
        for i, (cr, nr, entry) in enumerate(zip(all_cached_rewards, all_new_rewards, top_entries)):
            f.write(f"{i:>5d} | {cr:>8.4f} | {nr:>8.4f} | {nr-cr:>+8.4f} | {entry['epoch']:>6d} | {entry['index']:>6d}\n")

    print(f"\n图片和结果已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
