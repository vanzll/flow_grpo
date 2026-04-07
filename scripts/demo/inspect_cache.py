"""查看 prior shaping 缓存数据的统计信息。

Usage:
    python scripts/demo/inspect_cache.py                              # 默认目录
    python scripts/demo/inspect_cache.py --cache_dir cache/my_cache   # 自定义目录
    python scripts/demo/inspect_cache.py --epoch 5                    # 看某个 epoch 的详情
"""

import argparse
import glob
import os
import numpy as np


def inspect_all(cache_dir: str):
    files = sorted(glob.glob(os.path.join(cache_dir, "epoch_*.npz")))
    if not files:
        print(f"目录 {cache_dir} 中没有找到 epoch_*.npz 文件")
        return

    print(f"缓存目录: {cache_dir}")
    print(f"总 epoch 数: {len(files)}")
    print()

    all_rewards = []
    total_size = 0
    for f in files:
        total_size += os.path.getsize(f)
        with np.load(f) as data:
            all_rewards.append(data["rewards"])

    all_rewards = np.concatenate(all_rewards)
    print(f"总样本数: {len(all_rewards)}")
    print(f"总磁盘大小: {total_size / (1024**3):.2f} GB")
    print()

    print("=== Reward 统计 ===")
    print(f"  Mean:   {all_rewards.mean():.4f}")
    print(f"  Std:    {all_rewards.std():.4f}")
    print(f"  Min:    {all_rewards.min():.4f}")
    print(f"  Max:    {all_rewards.max():.4f}")
    print(f"  Median: {np.median(all_rewards):.4f}")
    print()

    # 分位数
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("=== Reward 分位数 ===")
    for p in percentiles:
        print(f"  P{p:2d}: {np.percentile(all_rewards, p):.4f}")
    print()

    # Top-10 和 Bottom-10
    sorted_idx = np.argsort(all_rewards)
    print("=== Top-10 Reward ===")
    for i in sorted_idx[-10:][::-1]:
        epoch = i // 256  # 假设每 epoch 256 个样本
        local_idx = i % 256
        print(f"  reward={all_rewards[i]:.4f}  (epoch={epoch}, index={local_idx})")
    print()

    print("=== Bottom-10 Reward ===")
    for i in sorted_idx[:10]:
        epoch = i // 256
        local_idx = i % 256
        print(f"  reward={all_rewards[i]:.4f}  (epoch={epoch}, index={local_idx})")


def inspect_epoch(cache_dir: str, epoch: int):
    path = os.path.join(cache_dir, f"epoch_{epoch:06d}.npz")
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return

    with np.load(path) as data:
        noises = data["noises"].astype(np.float32)  # fp16 -> fp32 避免 norm 溢出
        rewards = data["rewards"]

    print(f"文件: {path}")
    print(f"大小: {os.path.getsize(path) / (1024**2):.1f} MB")
    print()
    print(f"Noises:  shape={noises.shape}, dtype={noises.dtype}")
    print(f"Rewards: shape={rewards.shape}, dtype={rewards.dtype}")
    print()
    print(f"Reward mean={rewards.mean():.4f}, std={rewards.std():.4f}, "
          f"min={rewards.min():.4f}, max={rewards.max():.4f}")
    print()

    sorted_idx = np.argsort(rewards)
    print("Top-5:")
    for i in sorted_idx[-5:][::-1]:
        print(f"  index={i}, reward={rewards[i]:.4f}, noise_norm={np.linalg.norm(noises[i]):.2f}")
    print()
    print("Bottom-5:")
    for i in sorted_idx[:5]:
        print(f"  index={i}, reward={rewards[i]:.4f}, noise_norm={np.linalg.norm(noises[i]):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看 prior shaping 缓存数据")
    parser.add_argument("--cache_dir", type=str, default="cache/prior_shaping",
                        help="缓存目录路径")
    parser.add_argument("--epoch", type=int, default=None,
                        help="查看某个 epoch 的详情（不指定则显示整体统计）")
    args = parser.parse_args()

    if args.epoch is not None:
        inspect_epoch(args.cache_dir, args.epoch)
    else:
        inspect_all(args.cache_dir)
